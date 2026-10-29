#!/usr/bin/env python3
"""Score a records-style jsonl (id, label, source, group, chat_text) with a
LoRA checkpoint, using the byte-exact score_cands.py recipe.

Every (id, label, p_stop) row written to --out. Longest-first with OOM
fallback, matching eval_qwen.py.
"""
import json, time, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import peft

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/qwen3.5-4b")
ap.add_argument("--ckpt", required=True)
ap.add_argument("--records", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-len", type=int, default=36000)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
id_stop = tok.encode(" STOP", add_special_tokens=False)[-1]
id_ok = tok.encode(" ok", add_special_tokens=False)[-1]
assert id_stop != id_ok
print(f"label tokens: STOP={id_stop} ok={id_ok}", flush=True)

recs = [json.loads(l) for l in open(args.records)]
assert len({r["id"] for r in recs}) == len(recs), "duplicate ids"
recs.sort(key=lambda r: -r.get("n_tokens", 0))
print(f"scoring {len(recs)} records", flush=True)

print("loading model (4-bit NF4)...", flush=True)
cfg = AutoConfig.from_pretrained(args.model)
cfg.text_config.mamba_ssm_dtype = "float16"
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    args.model, config=cfg, quantization_config=quant, device_map="auto")
model.config.use_cache = False
sd = torch.load(args.ckpt)
sd = {k.replace(".ms.0.", ".").replace(".lora_A.", ".lora_A.default.").replace(
      ".lora_B.", ".lora_B.default."): v for k, v in sd.items()}
model = peft.get_peft_model(model, peft.LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj_qkv", "in_proj_z", "out_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"))
missing, unexpected = model.load_state_dict(sd, strict=False)
n_loaded = len(sd) - sum(1 for k in unexpected)
assert n_loaded == len(sd), "checkpoint keys did not match model"
model.eval()

outf = open(args.out, "w")
t0 = time.time()
n_done = 0
n_tokens_total = 0
for r in recs:
    ids_all = tok.encode(r["chat_text"], add_special_tokens=False)[:args.max_len]
    p = None
    ids = ids_all
    for cap in (args.max_len, 36000, 32000, 24000, 16000):
        ids = ids_all[:cap]
        t = torch.tensor([ids]).cuda()
        try:
            with torch.no_grad():
                logits = model(input_ids=t, logits_to_keep=1).logits[0, 0].float()
            p = torch.softmax(logits[[id_stop, id_ok]], dim=0)[0].item()
            break
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"OOM at {len(ids)} tokens, retrying shorter", flush=True)
    assert p is not None
    json.dump({"id": r["id"], "label": r["label"], "source": r.get("source"),
               "group": r.get("group"), "p_stop": p, "n_tokens": len(ids),
               "n_tokens_gemma": r.get("n_tokens", 0)}, outf)
    outf.write("\n")
    outf.flush()
    n_done += 1
    n_tokens_total += len(ids)
    if n_done % 50 == 0:
        el = time.time() - t0
        print(f"{n_done}/{len(recs)} elapsed={el:.0f}s ({n_tokens_total / el:.0f} tok/s)",
              flush=True)
outf.close()
