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
    cfg = AutoConfig.from_pretrained(args.model)
    cfg.text_config.mamba_ssm_dtype = "float16"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=cfg, quantization_config=quant,
        device_map="auto", dtype=torch.bfloat16)
    model.config.use_cache = False

    _PE = {}

    class Group(nn.Module):
        def __init__(self, ms, rotary, types):
            super().__init__()
            self.ms, self.rotary, self.types = nn.ModuleList(ms), rotary, types

        def forward(self, hidden_states, **kw):
            kw.pop("position_embeddings", None)
            def run_group(hs, _dg=None):
                x = hs
                for m, t in zip(self.ms, self.types):
                    if t not in _PE:
                        try:
                            pe = self.rotary(x, kw.get("position_ids"), t)
                            _PE[t] = tuple(p.to(torch.bfloat16) for p in pe)
                        except (TypeError, KeyError):
                            pe = self.rotary(x, kw.get("position_ids"))
                            _PE[t] = tuple(p.to(torch.bfloat16) for p in pe)
                    x = m(x, position_embeddings=_PE[t], **kw)
                    if isinstance(x, tuple) and len(x) == 1:
                        x = x[0]
                return x

            dummy = hidden_states.new_zeros(1).requires_grad_(True)
            return ckpt(run_group, hidden_states, dummy, use_reentrant=True)

    class LeanNorm(nn.Module):
  # keep structure identical to train_gemma.py; unused for qwen but harmless
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
            if hasattr(_L, _n):
                setattr(_L, _n, LeanNorm(getattr(_L, _n)))
    holder.layers = nn.ModuleList(
        [Group(layers[i:i + GN], rotary, ltypes[i:i + GN])
         for i in range(0, len(layers), GN) if len(layers[i:i + GN]) == GN])
    print(f"checkpoint groups of {GN}: {len(holder.layers)}", flush=True)

    model = peft.get_peft_model(model, peft.LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "in_proj_qkv", "in_proj_z", "out_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train / 1e6:.1f}M", flush=True)

    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu")
        # training model keeps the Group `.ms.0.` wrapping; only normalize
        # the peft adapter index (".lora_A." -> ".lora_A.default.")
        sd = {k.replace(".lora_A.", ".lora_A.default.").replace(
              ".lora_B.", ".lora_B.default."): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - sum(1 for k in unexpected)
        assert n_loaded == len(sd), "init checkpoint keys did not match model"
        print(f"initialized from {args.init_from} ({n_loaded} tensors)", flush=True)

    fold_of = {json.loads(l)["group"]: json.loads(l)["fold"]
               for l in open(args.folds)}
    recs = [json.loads(l) for l in open(args.records)]
    recs = [r for r in recs if r["label"] in ("STOP", "ok") and r["group"] in fold_of]
    train = [r for r in recs if fold_of[r["group"]] != args.fold]
    if args.limit_train:
        train = train[:args.limit_train]
    print(f"train={len(train)}", flush=True)

    def encode(r):
        ids = tok.encode(r["chat_text"], add_special_tokens=False)
        return ids[:args.max_len], id_stop if r["label"] == "STOP" else id_ok

    def forward_loss(rec):
        _PE.clear()
        ids, lab = encode(rec)
        ids = torch.tensor([ids + [lab]]).cuda()
        logits = model(input_ids=ids, logits_to_keep=2).logits[0, 0]
        return F.cross_entropy(logits.unsqueeze(0), ids[0, -1:].detach().cuda())

    if getattr(args, "save_init", False):
        torch.save(peft.get_peft_model_state_dict(model), "/data/wf_v2/smoke_init.pt")
        print("saved smoke_init.pt", flush=True)
    if args.probe:
        longest = sorted(train, key=lambda r: -r.get("n_tokens", 0))[:3]
        tot_t = 0.0
        for r in longest:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            loss = forward_loss(r)
            loss.backward()
            dt = time.time() - t0
            n_tok = len(tok.encode(r["chat_text"], add_special_tokens=False)[:args.max_len])
            tot_t += dt
            print(f"probe: tokens={n_tok} loss={loss.item():.4f} "
                  f"lat={dt:.1f}s tok/s={n_tok/dt:.0f} "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
        print(f"total probe time: {tot_t:.1f}s", flush=True)
        return

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
    gstep = 0
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
            gstep += 1
            if args.save_every and gstep % args.save_every == 0:
                torch.save(peft.get_peft_model_state_dict(model),
                           f"{ckpt_dir}/qwen4b_step{gstep}.pt")
                print(f"saved step checkpoint qwen4b_step{gstep}.pt", flush=True)
            if step % 5 == 0:
                print(f"ep {epoch} step {step}/{total_steps} "
                      f"loss={loss.item() * args.accum:.4f} "
                      f"elapsed={time.time() - ep_t:.0f}s "
                      f"peak={torch.cuda.max_memory_allocated() / 1e9:.1f}GB", flush=True)
        torch.save(peft.get_peft_model_state_dict(model), f"{ckpt_dir}/qwen4b_ep{epoch}.pt")
        print(f"epoch {epoch} done, saved checkpoint", flush=True)

    print("TRAINING COMPLETE", flush=True)
if __name__ == "__main__":
    main()