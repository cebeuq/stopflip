#!/usr/bin/env python3
"""Fine-tune Gemma 4 12B (-it) as a stop/continue classifier surrogate.

- QLoRA (4-bit NF4 base) on one CMP 170HX (64 GB), batch=1 + grad accumulation
- Every-2-layer gradient checkpointing to fit full 181K-token traces
- Loss on the single label token ("STOP" / "ok") following the template's
  generation prompt. Eval: P(STOP) normalized over the two candidate tokens.
"""
import json, os, time, math, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import peft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/gemma-4-12B-it")
    ap.add_argument("--records", default="/data/records_gemma.jsonl")
    ap.add_argument("--folds", default="/data/folds.jsonl")
    ap.add_argument("--out", default="/data/train_gemma_out")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=48000)
    ap.add_argument("--group-ckpt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true", help="memory/throughput probe only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    id_stop = tok.encode(" STOP", add_special_tokens=False)[-1]
    id_ok = tok.encode(" ok", add_special_tokens=False)[-1]
    assert id_stop != id_ok
    print(f"label tokens: STOP={id_stop} ok={id_ok}", flush=True)

    print("loading model (4-bit NF4)...", flush=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False

    # ---- every-N-layer gradient checkpointing --------------------------------
    # Each Group wraps N adjacent decoder layers in ONE checkpoint segment and
    # recomputes per-layer-type position embeddings inside it (Gemma-4 uses
    # different rope params / head dims for sliding vs global layers, so a
    # single shared position_embeddings argument would break later layers).
    _PE = {}

    class Group(nn.Module):
        def __init__(self, ms, rotary, types):
            super().__init__()
            self.ms, self.rotary, self.types = ms, rotary, types

        def forward(self, hidden_states, **kw):
            kw.pop("position_embeddings", None)

            def run_group(hs, _dg=None):
                x = hs
                for m, t in zip(self.ms, self.types):
                    if t not in _PE:
                        pe = self.rotary(x, kw.get("position_ids"), t)
                        _PE[t] = tuple(p.to(torch.bfloat16) for p in pe)
                    x = m(x, position_embeddings=_PE[t], **kw)
                    if isinstance(x, tuple) and len(x) == 1:
                        x = x[0]
                return x

            dummy = hidden_states.new_zeros(1).requires_grad_(True)
            return ckpt(run_group, hidden_states, dummy, use_reentrant=True)

    class LeanNorm(nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.weight = orig.weight

        def forward(self, x):
            return torch.nn.functional.rms_norm(x, x.shape[-1:], self.weight, 1e-6)

    layers = None
    for cand in (model, getattr(model, "model", None),
                 getattr(model, "language_model", None),
                 getattr(getattr(model, "model", None), "language_model", None)):
        if cand is not None and isinstance(getattr(cand, "layers", None), nn.ModuleList):
            layers = cand.layers
            holder = cand
            break
    else:
        raise RuntimeError("could not locate decoder layer list")
    ltypes = list(holder.config.layer_types)
    rotary = holder.rotary_emb
    GN = args.group_ckpt
    for _L in layers:
        for _n in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            setattr(_L, _n, LeanNorm(getattr(_L, _n)))
    holder.layers = nn.ModuleList(
        [Group(layers[i:i + GN], rotary, ltypes[i:i + GN])
         for i in range(0, len(layers), GN) if len(layers[i:i + GN]) == GN])
    print(f"checkpoint groups of {GN}: {len(holder.layers)}", flush=True)

    model = peft.get_peft_model(model, peft.LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M", flush=True)

    # ---- data -------------------------------------------------------------
    fold_of = {json.loads(l)["group"]: json.loads(l)["fold"]
               for l in open(args.folds)}
    recs = [json.loads(l) for l in open(args.records)]
    recs = [r for r in recs if r["label"] in ("STOP", "ok") and r["group"] in fold_of]
    train = [r for r in recs if fold_of[r["group"]] != args.fold]
    val = [r for r in recs if fold_of[r["group"]] == args.fold]
    print(f"train={len(train)} val={len(val)}", flush=True)

    def encode(r):
        ids = tok.encode(r["chat_text"], add_special_tokens=False)
        return ids[:args.max_len], id_stop if r["label"] == "STOP" else id_ok

    def forward_loss(rec):
        _PE.clear()
        ids, lab = encode(rec)
        ids = torch.tensor([ids + [lab]]).cuda()
        # loss on the single label token only: keep just 2 positions of logits
        # (full-vocab logits over 181K positions would need ~190 GB)
        logits = model(input_ids=ids, logits_to_keep=2).logits[0, 0]
        return F.cross_entropy(logits.unsqueeze(0), ids[0, -1:].detach())

    if args.probe:
        longest = sorted(train, key=lambda r: -r.get("n_tokens", 0))[:3]
        for r in longest:
            _PE.clear()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            loss = forward_loss(r)
            loss.backward()
            print(f"probe: tokens={r['n_tokens']} loss={loss.item():.4f} "
                  f"lat={time.time() - t0:.1f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        return

    # ---- train -------------------------------------------------------------
#
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(0.03 * total_steps))

    def lr_lambda(s):
        if s < warmup:
            return s / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    order = list(range(len(train)))
    step = 0
    for epoch in range(args.epochs):
        random.shuffle(order)
        model.train()
        ep_t = time.time()
        i = 0
        while i < len(order):
            opt.zero_grad()
            for _ in range(args.accum):
                if i >= len(order):
                    break
                loss = forward_loss(train[order[i]]) / args.accum
                loss.backward()
                i += 1
            opt.step()
            sched.step()
            step += 1
            if step % 5 == 0:
                print(f"ep {epoch} step {step}/{total_steps} "
                      f"loss={loss.item() * args.accum:.4f} "
                      f"elapsed={time.time() - ep_t:.0f}s "
                      f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        torch.save(peft.get_peft_model_state_dict(model), f"{ckpt_dir}/gemma12b_ep{epoch}.pt")
        print(f"epoch {epoch} done, saved checkpoint", flush=True)

    print("TRAINING COMPLETE", flush=True)

if __name__ == "__main__":
    main()
