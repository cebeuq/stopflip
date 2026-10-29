#!/usr/bin/env python3
"""Live probe harness: message-space perturbations + live Claude probes.

prep:    rebuild messages for selected records, apply T3 unicode perturbation
          in MESSAGE space (several seeds), render chat_text', dump candidates.
probe:   live-probe baseline vs perturbed with k repeats (relabel_v2 protocol).

Run inside docker, mounting the work tree at /work:
  docker run --rm -v /work:/work -w /work \
    <cuda-image> python tools/live_harness.py prep --ids ids.txt --out /work/tmp/cands.jsonl
  docker run --rm -v /work:/work -w /work \
    <cuda-image> python tools/live_harness.py probe --cands /work/tmp/cands_scored.jsonl --out /work/tmp/live_results.jsonl
"""
import json, os, sys, glob, copy, argparse, time, random
import urllib.request, urllib.error

TOOLS_HOME = os.environ.get("WORK", "/work")
sys.path.insert(0, f"{TOOLS_HOME}/tools")
from replay_labeler2 import parse_transcript, get_prompt_e2e

API = "https://api.anthropic.com/v1/messages"
BILLING = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.296.cf4; cc_entrypoint=sdk-cli;"}
TOOLS = [{"name": t, "input_schema": {"type": "object", "properties": {}, "additionalProperties": True}}
         for t in ["Bash", "Read", "Write", "Edit", "Monitor", "ScheduleWakeup", "ToolSearch"]]

CG_GLOB = f"{TOOLS_HOME}/corpus/cybergym-e2e/agent_output_v2/*/20*/trajectory/attempt_1.log"
DS_GLOB = f"{TOOLS_HOME}/corpus/deep-swe-data/jobs_v2/*/*/agent/sessions/projects/*/*.jsonl"
CVB_OUT = f"{TOOLS_HOME}/corpus/cve-bench/logs/cvb_labels"

def get_token():
    with open(f"{TOOLS_HOME}/tmp/oauth_token.txt") as f:
        return f.read().strip()

# ---------------- sanitize / template (build_dataset_v2 verbatim) ----------------

def sanitize_messages(messages):
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        content = m.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        blocks = []
        for c in content:
            if not isinstance(c, dict):
                continue
            t = c.get("type")
            if t == "text":
                blocks.append({"type": "text", "text": c.get("text", "")})
            elif t == "tool_use":
                blocks.append({"type": "tool_use", "id": c.get("id"), "name": c.get("name"), "input": c.get("input")})
            elif t == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": c.get("tool_use_id"), "content": c.get("content")})
        if blocks:
            out.append({"role": m.get("role"), "content": blocks})
    return out

def to_template_msgs(blocks):
    out = []
    for m in blocks:
        role = m.get("role")
        text = None
        tool_calls = []
        for c in m.get("content", []):
            if c.get("type") == "text":
                text = c.get("text", "")
            elif c.get("type") == "tool_use":
                tool_calls.append({"function": {"name": c.get("name"), "arguments": c.get("input")}})
        if text is not None or not tool_calls:
            out.append({"role": role, "content": text if text is not None else ""})
        if tool_calls:
            out.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for c in m.get("content", []):
            if c.get("type") == "tool_result":
                tc = c.get("content")
                ttext = tc if isinstance(tc, str) else json.dumps(tc) if tc is not None else ""
                out.append({"role": "tool", "name": "tool", "tool_call_id": c.get("tool_use_id"), "content": ttext})
    return out

# ---------------- episode discovery (relabel_v2 verbatim) ----------------

def discover_episodes():
    episodes = []
    for lf in sorted(glob.glob(CG_GLOB + ".labels.claude-opus-5-5.jsonl")):
        path = lf.replace(".labels.claude-opus-5-5.jsonl", "")
        repo = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        episodes.append({"type": "cybergym", "repo": repo, "path": path, "labelfile": lf, "key": path})
    for lf in sorted(glob.glob(DS_GLOB + ".labels.claude-opus-5-5.jsonl")):
        episodes.append({"type": "deepswe", "repo": None, "path": lf.replace(".labels.claude-opus-5-5.jsonl", ""), "labelfile": lf, "key": lf})
    for lf in sorted(glob.glob(f"{CVB_OUT}/*.labels.claude-opus-5-5.jsonl")):
        episodes.append({"type": "cvebench", "repo": None, "path": None, "labelfile": lf, "key": lf})
    return episodes

def build_messages(e):
    """relabel_v2.build_messages verbatim."""
    if e["type"] == "cybergym":
        msgs, _ = parse_transcript(e["path"])
        task_prompt = get_prompt_e2e(e["repo"])
        return sanitize_messages([{"role": "user", "content": [{"type": "text", "text": task_prompt}]}] + msgs)
    elif e["type"] == "deepswe":
        return sanitize_messages(parse_transcript(e["path"])[0])
    else:
        from batch_labeler_v2 import cvb_samples
        return sanitize_messages(cvb_samples()[e["key"].split("/")[-1].replace(".labels.claude-opus-5-5.jsonl", "")])

# ---------------- T3 in message space ----------------

CYRILLIC = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
            "x": "х", "m": "м", "i": "і", "s": "ѕ", "j": "ј", "n": "ո", "t": "т"}
ZERO_WIDTH = ["​", "‌", "⁠", "﻿"]
COMBINING = ["́", "̇", "̈"]

def perturb_string(s, rng):
    out = list(s)
    for i, ch in enumerate(out):
        roll = rng.random()
        if roll < 0.012 and ch.lower() in CYRILLIC and ch.isascii():
            r = CYRILLIC[ch.lower()]
            out[i] = r.upper() if ch.isupper() else r
        elif roll < 0.018:
            out[i] = out[i] + rng.choice(ZERO_WIDTH)
        elif roll < 0.021 and ch.isalpha():
            out[i] = out[i] + rng.choice(COMBINING)
        elif roll < 0.026 and ch == " ":
            out[i] = " "
        elif roll < 0.032 and ch.isalpha():
            out[i] = out[i].lower() if out[i].isupper() else out[i].upper()
    return "".join(out)

def perturb_messages(messages, n_tail, seed):
    """Apply T3 to the string fields of the last n_tail messages."""
    msgs = copy.deepcopy(messages)
    rng = random.Random(seed)
    for m in msgs[-n_tail:]:
        for c in m.get("content", []):
            if c.get("type") == "text" and isinstance(c.get("text"), str):
                c["text"] = perturb_string(c["text"], rng)
            elif c.get("type") == "tool_use" and isinstance(c.get("input"), dict):
                inp = c["input"]
                for k, v in list(inp.items()):
                    if isinstance(v, str):
                        inp[k] = perturb_string(v, rng)
    return msgs

# ---------------- live API (relabel_v2 verbatim) ----------------

def call_api(messages, model, cache_bp, max_tokens=1):
    msgs = copy.deepcopy(messages)
    body = {"model": model, "max_tokens": max_tokens, "system": [dict(BILLING)], "messages": msgs, "tools": TOOLS}
    if cache_bp:
        for idx in cache_bp:
            if 0 <= idx < len(msgs):
                msg = msgs[idx]
                content = msg.get("content")
                if content and isinstance(content[-1], dict):
                    content[-1] = dict(content[-1])
                    content[-1]["cache_control"] = {"type": "ephemeral"}
    token = get_token()
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "anthropic-version": "2023-06-01", "content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())

def robust_call(messages, model, cache_bp, max_attempts=24):
    attempt = 0
    while True:
        try:
            r = call_api(messages, model, cache_bp)
            return r.get("stop_reason"), r.get("stop_details"), None
        except urllib.error.HTTPError as ex:
            try:
                err_body = ex.read().decode()[:400]
            except Exception:
                err_body = "?"
            if ex.code in (429, 500, 502, 503, 529, 401):
                time.sleep(min(60, 15 * (attempt + 1)))
                attempt += 1
                if attempt >= max_attempts:
                    return None, None, f"http_{ex.code}. {err_body}"
                continue
            return None, None, f"http_{ex.code}. {err_body}"
        except Exception as ex:
            time.sleep(5)
            attempt += 1
            if attempt >= max_attempts:
                return None, None, str(ex)

# ---------------- prep ----------------

_MSG_CACHE = {}

def do_prep(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{TOOLS_HOME}/tmp/gemma4-12b-it")
    episodes = discover_episodes()
    print(f"{len(episodes)} episodes", flush=True)

    ids = [l.strip() for l in open(args.ids) if l.strip()]
    recs = [json.loads(l) for l in open(f"{TOOLS_HOME}/dataset/v2/records_gemma.jsonl")]
    rec_by_id = {r["id"]: r for r in recs}

    out = open(args.out, "w")
    n_written = 0
    for rid in ids:
        rec = rec_by_id.get(rid)
        if rec is None:
            print(f"SKIP {rid}: not found", flush=True)
            continue
        rec_type, ep_idx, pos = rid.rsplit("-", 2)
        e = episodes[int(ep_idx)]
        if e["type"] != rec_type:
            print(f"SKIP {rid}: type mismatch {e['type']} vs {rec_type}", flush=True)
            continue
        cache_key = e["path"] or e["labelfile"]
        if cache_key not in _MSG_CACHE:
            _MSG_CACHE[cache_key] = build_messages(e)
        messages = _MSG_CACHE[cache_key][:rec["pos"] + 1]
        ct = tok.apply_chat_template(to_template_msgs(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if ct != rec["chat_text"]:
            print(f"RT-DIFF {rid}: round-trip mismatch", flush=True)
            continue
        for seed in range(args.seeds):
            n_tail = 2 if seed < 3 else 3
            mp = perturb_messages(messages, n_tail=n_tail, seed=seed)
            ctp = tok.apply_chat_template(to_template_msgs(mp), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if ctp == ct:
                ctp = tok.apply_chat_template(to_template_msgs(perturb_messages(messages, n_tail=3, seed=seed + 10)), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            rec_out = {"id": rid, "seed": seed, "n_tail": n_tail,
                       "messages": messages, "messages_prime": mp,
                       "chat_text_prime": ctp}
            out.write(json.dumps(rec_out) + "\n")
        n_written += 1
        print(f"prepped {rid}", flush=True)
    print(f"prep done: {n_written} records", flush=True)

def do_probe(args):
    cands = [json.loads(l) for l in open(args.cands)]
    out = open(args.out, "a")
    for cand in cands:
        messages = cand["messages"]
        n = len(messages)
        cache_bp = sorted(set([0, n // 4, n // 2]))
        base_labels, pert_labels = [], []
        for _ in range(args.k):
            sr, sd, err = robust_call(messages, args.model, cache_bp)
            base_labels.append(sr)
            time.sleep(1)
        mp = cand["messages_prime"]
        n2 = len(mp)
        cache_bp2 = sorted(set([0, n2 // 4, n2 // 2]))
        for _ in range(args.k):
            sr, sd, err = robust_call(mp, args.model, cache_bp2)
            pert_labels.append(sr)
            time.sleep(1)
        rec = {"id": cand["id"], "seed": cand.get("seed"), "base": base_labels, "pert": pert_labels}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        print(f"{cand['id']} seed={cand.get('seed')}: base={base_labels} pert={pert_labels}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="prep", choices=["prep", "probe"])
    ap.add_argument("--ids", default=None)
    ap.add_argument("--cands", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args(sys.argv[1:])
    if args.mode == "prep":
        do_prep(args)
    else:
        do_probe(args)
