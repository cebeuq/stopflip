#!/usr/bin/env python3
"""Path B batch replay-labeler.

Replay-labels many transcripts against target models via the raw Messages API.

Features:
- Resume: skips transcripts+models whose label files already exist
- Caching: 4 cache breakpoints (system, task prompt, N/4, N/2) per call
- Subsampling: caps positions per episode (always includes 0 and last)
- Stores: <transcript>.labels.<model>.jsonl (one JSON per position)

Usage:
    python3 batch_labeler.py <episodes_manifest.json> [--models opus,sonnet]
"""
import argparse, copy, glob, json, os, sys, time
import urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_labeler2 import parse_transcript, get_prompt_e2e

API = "https://api.anthropic.com/v1/messages"
BILLING = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.296.cf4; cc_entrypoint=sdk-cli;"}
TOOLS_HOME = "/work"

def find_cg_transcripts():
    """Find all CyberGym transcripts (pilot + V0) -> [(path, repo)]."""
    out = []
    base = f"{TOOLS_HOME}/corpus/cybergym-e2e"
    # pilot dirs (agent_output_*)
    for d in sorted(glob.glob(f"{base}/agent_output_*")):
        for log in sorted(glob.glob(f"{d}/*/20*/trajectory/attempt_1.log")):
            dirname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(log))))
            repo = dirname.split("_")[0]
            out.append({"path": log, "type": "cybergym", "repo": repo, "key": dirname})
    return out

def find_deepswe_transcripts():
    out = []
    for p in sorted(glob.glob(f"{TOOLS_HOME}/dataset/raw/deepswe/jobs/*/*/agent/sessions/projects/*/*.jsonl")):
        out.append({"path": p, "type": "deepswe", "repo": None, "key": p.split("/")[-5][:40]})
    return out

def subsample_positions(positions, cap=40):
    if len(positions) <= cap:
        return positions
    step = (len(positions) - 1) / (cap - 1)
    idxs = sorted(set(int(i * step) for i in range(cap)))
    return [positions[i] for i in idxs]

def sanitize_messages(messages):
    """Strip to role+content only; drop thinking blocks; sanitize content blocks."""
    out = []
    for m in messages:
        content = []
        for c in m.get("content", []):
            if not isinstance(c, dict):
                continue
            if c.get("type") == "thinking":
                continue
            # keep only known-safe fields per block type
            if c.get("type") in ("text",):
                content.append({"type": "text", "text": c.get("text", "")})
            elif c.get("type") == "tool_use":
                content.append({"type": "tool_use", "id": c.get("id"), "name": c.get("name"), "input": c.get("input")})
            elif c.get("type") == "tool_result":
                content.append({"type": "tool_result", "tool_use_id": c.get("tool_use_id"), "content": c.get("content")})
            else:
                content.append(c)
        out.append({"role": m.get("role"), "content": content})
    return out

def build_messages(entry):
    """Parse transcript and return sanitized message list with task prompt prepended if needed."""
    messages, refusals = parse_transcript(entry["path"])
    if entry["type"] == "cybergym":
        task_prompt = get_prompt_e2e(entry["repo"])
        messages = [{"role": "user", "content": [{"type": "text", "text": task_prompt}], "uuid": "task_prompt"}] + messages
    return sanitize_messages(messages), refusals

def call_api(messages, model, max_tokens, token, cache_bp=None):
    if cache_bp:
        msgs = copy.deepcopy(messages)
    else:
        msgs = messages
    if cache_bp:
        for idx in cache_bp:
            if 0 <= idx < len(msgs):
                msg = msgs[idx]
                content = msg["content"]
                if content and isinstance(content[-1], dict):
                    content[-1] = dict(content[-1])
                    content[-1]["cache_control"] = {"type": "ephemeral"}
    else:
        msgs = messages
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [dict(BILLING)],
        "messages": msgs,
    }
    if cache_bp:
        body["system"] = [dict(BILLING)]
        body["system"][0]["cache_control"] = {"type": "ephemeral"}
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["discover", "label"])
    ap.add_argument("--models", default="claude-opus-5-5,claude-sonnet-5-5")
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    entries = find_cg_transcripts() + find_deepswe_transcripts()
    if args.dry_run or args.mode == "discover":
        for e in entries:
            print(f"{e['type']:9s} {e['key'][:40]:42s} {e['path'][:80]}")
        print(f"\nTotal: {len(entries)} transcripts")
        return

    with open(f"{TOOLS_HOME}/tmp/oauth_token.txt") as f:
        token = f.read().strip()

    models = args.models.split(",")
    for e in entries:
        messages, refusals = build_messages(e)
        user_positions = [i for i, m in enumerate(messages) if m["role"] == "user"]
        # subsample
        sampled = subsample_positions(user_positions, args.cap)
        print(f"\n=== {e['key']} ({len(user_positions)} user positions -> {len(sampled)} sampled) ===")
        for model in models:
            out_path = f"{e['path']}.labels.{model}.jsonl"
            if os.path.exists(out_path):
                print(f"  {model}: SKIPPED (exists)")
                continue
            results = []
            # cache breakpoints: task prompt (msg 0), N/4, N/2 of full message list
            n = len(messages)
            cache_bp = sorted(set([0, n // 4, n // 2]))
            for i in sampled:
                try:
                    r = call_api(messages[:i+1], model, 1, token, cache_bp=cache_bp)
                    sr = r.get("stop_reason")
                    sd = r.get("stop_details")
                    label = "STOP" if sr == "refusal" else "ok"
                    results.append({"pos": i, "label": label, "stop_reason": sr, "stop_details": sd})
                    print(f"  [{model}] [{i}] {label}  ({sr})")
                except urllib.error.HTTPError as ex:
                    print(f"  [{model}] [{i}] ERROR {ex.code}: {ex.read().decode()[:100]}")
                except Exception as ex:
                    print(f"  [{model}] [{i}] ERROR {ex}")
                time.sleep(1)
            if results:
                with open(out_path, "w") as f:
                    r_tmp = {"entry": e["key"], "model": model, "type": e["type"], "n_user_positions": len(user_positions), "labels": results}
                with open(out_path, "w") as f:
                    f.write(json.dumps(r_tmp) + "\n")
                stops = sum(1 for r in results if r["label"] == "STOP")
                print(f"  {model}: {stops} stops / {len(results)} positions -> {out_path}")

if __name__ == "__main__":
    main()
