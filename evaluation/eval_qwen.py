#!/usr/bin/env python3
"""Evaluate Qwen3.5-4B stop/continue classifier (LoRA ep0) on the val fold."""
import json, time, argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import peft

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/qwen3.5-4b")
ap.add_argument("--ckpt", default="/data/qwen4b_ep0.pt")
ap.add_argument("--records", default="/data/records_gemma.jsonl")
ap.add_argument("--folds", default="/data/folds.jsonl")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--max-len", type=int, default=48000)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="/data/eval_ep0.jsonl")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
id_stop = tok.encode(" STOP", add_special_tokens=False)[-1]
id_ok = tok.encode(" ok", add_special_tokens=False)[-1]
assert id_stop != id_ok

print("loading model (fp16, split across GPUs)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.float16, device_map="auto")
model.config.use_cache = False
sd = torch.load(args.ckpt)
model = peft.get_peft_model(model, peft.LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "in_proj_qkv", "in_proj_z", "out_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM"))
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded ckpt: {len(sd)} keys; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
model.eval()

fold_of = {json.loads(l)["group"]: json.loads(l)["fold"] for l in open(args.folds)}
recs = [json.loads(l) for l in open(args.records)]
val = [r for r in recs if r["label"] in ("STOP", "ok")
       and fold_of.get(r["group"]) == 0]
if args.limit:
    val = val[:args.limit]
print(f"evaluating {len(val)} val records", flush=True)

outf = open(args.out, "w")
t0 = time.time()
n_tokens_total = 0
results = []
for i, r in enumerate(val):
    ids = tok.encode(r["chat_text"], add_special_tokens=False)[:args.max_len]
    t = torch.tensor([ids]).cuda() if next(model.parameters()).is_cuda else torch.tensor([ids]).cuda(0)
    with torch.no_grad():
        logits = model(input_ids=t).logits[0, -1].float()
    p = torch.softmax(logits[[id_stop, id_ok]], dim=0)
    results.append({"id": r["id"], "label": r["label"], "p_stop": p[0].item(),
                    "n_tokens": len(ids), "n_tokens_gemma": r.get("n_tokens", 0)})
    json.dump(results[-1], outf)
    outf.write("\n")
    n_tokens_total += len(ids)
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"{i + 1}/{len(val)} elapsed={el:.0f}s ({n_tokens_total / el:.0f} tok/s)", flush=True)
outf.close()

n = len(results)
correct = sum(1 for r in results if (r["p_stop"] >= 0.5) == (r["label"] == "STOP"))
acc = correct / n
import math
nll = -sum(math.log(max(r["p_stop"], 1e-9) if r["label"] == "STOP" else max(1 - r["p_stop"], 1e-9)) for r in results) / n
tp = sum(1 for r in results if r["label"] == "STOP" and r["p_stop"] >= 0.5)
fp = sum(1 for r in results if r["label"] == "ok" and r["p_stop"] >= 0.5)
tn = sum(1 for r in results if r["label"] == "ok" and r["p_stop"] < 0.5)
fn = sum(1 for r in results if r["label"] == "STOP" and r["p_stop"] < 0.5)
print(f"n={n} acc={acc:.4f} nll={nll:.4f} cm(tp,fp,tn,fn)=({tp},{fp},{tn},{fn})", flush=True)
try:
    from sklearn.metrics import roc_auc_score
    y = [1 if r["label"] == "STOP" else 0 for r in results]
    print(f"AUC={roc_auc_score(y, [r['p_stop'] for r in results]):.4f}", flush=True)
except Exception as e:
    print(f"sklearn AUC failed: {e}", flush=True)
print("EVAL DONE", flush=True)