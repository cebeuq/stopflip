#!/usr/bin/env python3
"""Saliency branch run 2: chunked-GQA sdpa fix + low-p_stop first attack.
Runs INSIDE docker. Args: --budget 200 --max-minutes 50"""
import json, sys, random, argparse, time, os, contextlib

# ---- chunked-GQA sdpa patch: sm75 kernels can't do head_dim=256 with GQA ----
import torch
import torch.nn.functional as F

def make_patched_sdpa():
    _orig = F.scaled_dot_product_attention
    def chunked_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        # grouped-GQA fallback: q (B,Hq,L,D), k/v (B,Hkv,S,D) with Hq % Hk == 0
        if not (v.dim() == 4 and q.dim() == 4 and (q.shape[1] != v.shape[1] or q.shape[-1] > 128)):
            return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        B, Hq, L, D = q.shape
        Hk = v.shape[1]
        assert Hq % Hk == 0
        G = Hq // Hk
        CH = 512
        if L > 30000: CH = 128
        elif L > 8000: CH = 256
        out = torch.empty_like(q)
        for i in range(0, L, CH):
            qs = q[:, :, i:i+CH]                    # (B,Hq,CH,D)
            qs = qs.reshape(B, Hk, G, qs.shape[2], qs.shape[3])
            att = qs @ k.transpose(-1, -2).unsqueeze(2)   # (B,Hk,G,CH,S)
            if is_causal:
                rows = torch.arange(i, i+att.shape[-2], device=q.device)
                cols = torch.arange(k.shape[2], device=q.device)
                att = att.masked_fill_(rows[:, None] < cols[None, :], float("-inf"))
            if scale is not None:
                att = att * scale
            att = torch.softmax(att.float(), dim=-1).to(q.dtype)
            out[:, :, i:i+CH] = (att @ v.unsqueeze(2)).reshape(B, Hq, att.shape[-2], D)
        return out
    return chunked_sdpa

F.scaled_dot_product_attention = make_patched_sdpa()

import contextlib
_NOCTX = contextlib.nullcontext()
sys.path.insert(0, "/data")
from sal_common import parse_chat_text, render_chat_text

DATA = "/data"

def build_fields(messages):
    fields = []
    for mi, msg in enumerate(messages):
        if msg["role"] == "user":
            for bi, b in enumerate(msg["blocks"]):
                if b["type"] == "text" and b["text"]:
                    fields.append({"mi": mi, "bi": bi, "key": "text", "text": b["text"], "use_case": True})
            continue
        if msg["role"] != "assistant":
            continue
        bi = 0
        for b in msg["blocks"]:
            if b["type"] == "text":
                if b["text"]:
                    fields.append({"mi": mi, "bi": bi, "key": "text", "text": b["text"], "use_case": True})
            elif b["type"] == "tool_use":
                for k2, v2 in b["fields"].items():
                    if k2 in b.get("_quoted", ()):
                        fields.append({"mi": mi, "bi": bi, "key": k2, "text": v2, "use_case": k2 != "command"})
            bi += 1
    return fields

from sal_common import perturb_range

def make_candidate(messages, fields, rate, seed):
    new_messages = json.loads(json.dumps(messages, default=list))
    edits = []
    for f in fields:
        text = f["text"]
        if not text:
            continue
        rng_local = random.Random(f'{seed}|{f["mi"]}|{f["bi"]}|{f["key"]}|0|{len(text)}')
        new_text = perturb_range(text, rng_local, rate, 0, len(text), f["use_case"])
        if new_text == text:
            continue
        m = new_messages[f["mi"]]
        if f["key"] == "text":
            m["blocks"][f["bi"]]["text"] = new_text
        else:
            m["blocks"][f["bi"]]["fields"][f["key"]] = new_text
        edits.append({"mi": f["mi"], "bi": f["bi"], "key": f["key"], "old": text, "new": new_text})
    if not edits:
        return None
    return new_messages, edits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--max-minutes", type=float, default=50)
    ap.add_argument("--out", default="/data/sal_out/cands2.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
    import peft

    tok = AutoTokenizer.from_pretrained("/data/qwen3.5-4b")
    id_stop = tok.encode(" STOP", add_special_tokens=False)[-1]
    id_ok = tok.encode(" ok", add_special_tokens=False)[-1]
    print("loading model (4-bit NF4)...", flush=True)
    cfg = AutoConfig.from_pretrained("/data/qwen3.5-4b")
    cfg.text_config.mamba_ssm_dtype = "float16"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        "/data/qwen3.5-4b", config=cfg, quantization_config=quant,
        device_map="auto", max_memory={0: "2.4GiB", 1: "2.4GiB"}, dtype=torch.float16)
    model.config.use_cache = False
    sd = torch.load("/data/qwen4b_ep0.pt")
    sd = {k.replace(".ms.0.", ".").replace(".lora_A.", ".lora_A.default.").replace(
          ".lora_B.", ".lora_B.default."): v for k, v in sd.items()}
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",