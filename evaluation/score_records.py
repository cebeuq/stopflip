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
print("SCORE_RECORDS DONE", flush=True)
===train script head===
#!/usr/bin/env python3
"""Fine-tune Qwen3.5-4B as a stop/continue classifier surrogate.

Reconstruction of the (deleted) script that produced /data/qwen4b_ep0.pt,
derived from /work/tools/train_gemma.py on k2 with the
deltas pinned by the ep0 checkpoint itself (all keys `layers.N.ms.0.*`) and
by /data/score_cands.py + tools/attack2.py on k2 (10 target modules):

- QLoRA (4-bit NF4 base, fp16 compute), batch=1 + grad accumulation
- every-N-layer gradient checkpointing with the wrapped layer registered as
  `self.ms = nn.ModuleList([layer])` (this is what produces the `.ms.0.` key
  prefix in the checkpoints and makes use_reentrant=True see the params)
- loss on the single label token (" STOP" / " ok")
- one checkpoint per epoch, saved as qwen4b_ep{N}.pt
"""
import json, os, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
import peft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/qwen3.5-4b")
    ap.add_argument("--records", default="/data/records_gemma.jsonl")
    ap.add_argument("--folds", default="/data/folds.jsonl")
    ap.add_argument("--out", default="/data/wf_v2/out")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=48000)
    ap.add_argument("--group-ckpt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                    help="optional ep0 checkpoint to continue training from")
    ap.add_argument("--probe", action="store_true", help="memory/throughput probe only")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save a step checkpoint every N optimizer steps (0=off)")
    ap.add_argument("--limit-train", type=int, default=0,
                    help="smoke test: use only N first train records")
    ap.add_argument("--save-init", action="store_true", help="save init state dict for key-compat smoke test")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="hard wall-clock budget for training (0=off); on"
                         " expiry saves a partial checkpoint and stops")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    id_stop = tok.encode(" STOP", add_special_tokens=False)[-1]
    id_ok = tok.encode(" ok", add_special_tokens=False)[-1]
    assert id_stop != id_ok
    print(f"label tokens: STOP={id_stop} ok={id_ok}", flush=True)

===trainB log head===
label tokens: STOP=44020 ok=5226
loading model (4-bit NF4)...
Current Triton version 3.2.0 is below the recommended 3.3.0 version. Errors may occur and these issues will not be fixed. Please consider upgrading Triton.
Loading weights:   0%|          | 0/426 [00:00<?, ?it/s]Loading weights:   0%|          | 1/426 [00:00<01:28,  4.78it/s]Loading weights:   6%|▌         | 26/426 [00:00<00:03, 101.43it/s]Loading weights:  11%|█         | 47/426 [00:00<00:02, 140.51it/s]Loading weights:  19%|█▉        | 80/426 [00:00<00:01, 203.42it/s]Loading weights:  27%|██▋       | 114/426 [00:00<00:01, 246.28it/s]Loading weights:  34%|███▍      | 146/426 [00:00<00:01, 266.52it/s]Loading weights:  41%|████      | 175/426 [00:00<00:00, 272.34it/s]Loading weights:  48%|████▊     | 205/426 [00:00<00:00, 277.09it/s]Loading weights:  56%|█████▌    | 239/426 [00:01<00:00, 292.24it/s]Loading weights:  64%|██████▍   | 273/426 [00:01<00:00, 303.68it/s]Loading weights:  72%|███████▏  | 305/426 [00:01<00:00, 305.28it/s]Loading weights:  79%|███████▉  | 336/426 [00:01<00:00, 304.94it/s]Loading weights:  86%|████████▌ | 367/426 [00:01<00:00, 295.35it/s]Loading weights:  94%|█████████▎| 399/426 [00:01<00:00, 300.69it/s]Loading weights: 100%|██████████| 426/426 [00:01<00:00, 258.30it/s]
checkpoint groups of 1: 32
trainable params: 30.5M
train=6133
ep 0 step 5/384 loss=23.0000 elapsed=755s peak=19.2GB
ep 0 step 10/384 loss=11.1250 elapsed=1267s peak=19.2GB
ep 0 step 15/384 loss=5.9688 elapsed=1726s peak=20.4GB
ep 0 step 20/384 loss=0.5234 elapsed=2219s peak=20.4GB
ep 0 step 25/384 loss=2.1250 elapsed=2599s peak=20.4GB
ep 0 step 30/384 loss=1.2266 elapsed=2969s peak=20.4GB
ep 0 step 35/384 loss=0.0488 elapsed=3395s peak=20.4GB
ep 0 step 40/384 loss=0.8242 elapsed=3789s peak=20.4GB
ep 0 step 45/384 loss=0.6914 elapsed=4178s peak=20.4GB
ep 0 step 50/384 loss=1.3125 elapsed=4548s peak=20.4GB
ep 0 step 55/384 loss=0.3867 elapsed=4955s peak=20.4GB
ep 0 step 60/384 loss=1.1406 elapsed=5322s peak=20.4GB
ep 0 step 65/384 loss=0.3867 elapsed=5724s peak=20.4GB
ep 0 step 70/384 loss=0.6914 elapsed=6097s peak=20.4GB
ep 0 step 75/384 loss=1.5000 elapsed=6556s peak=20.4GB
ep 0 step 80/384 loss=0.4746 elapsed=7013s peak=20.4GB
ep 0 step 85/384 loss=0.8242 elapsed=7454s peak=20.4GB
saved step checkpoint qwen4b_step90.pt
ep 0 step 90/384 loss=1.9141 elapsed=7846s peak=20.4GB
ep 0 step 95/384 loss=0.5742 elapsed=8308s peak=20.4GB
ep 0 step 100/384 loss=0.1602 elapsed=8699s peak=20.4GB
ep 0 step 105/384 loss=0.9727 elapsed=9106s peak=20.4GB
ep 0 step 110/384 loss=0.0620 elapsed=9515s peak=20.4GB