#!/usr/bin/env python3
"""Path B v2: parallel replay-labeler for the v2 dataset (CG + DS + CVB).

Usage:
  python3 batch_labeler_v2.py discover
  python3 batch_labeler_v2.py label [--models claude-opus-5-5] [--workers 8] [--cap 40] [--limit N]
"""
import argparse, copy, glob, json, os, sys, time
import urllib.request, urllib.error
import zipfile, zstandard, struct
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_labeler2 import parse_transcript, get_prompt_e2e

TOOLS_HOME = "/work"
API = "https://api.anthropic.com/v1/messages"
BILLING = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.296.cf4; cc_entrypoint=sdk-cli;"}
TOKEN_FILE = f"{TOOLS_HOME}/tmp/oauth_token.txt"
CG_GLOB = f"{TOOLS_HOME}/corpus/cybergym-e2e/agent_output_v2/*/20*/trajectory/attempt_1.log"
DS_GLOB = f"{TOOLS_HOME}/corpus/deep-swe-data/jobs_v2/*/*/agent/sessions/projects/*/*.jsonl"
CVB_EVAL = f"{TOOLS_HOME}/corpus/cve-bench/logs/2026-10-15T23-51-53-00-00_cvebench_gJUhBjx4dGbjjtW2MRaoqn.eval"
CVB_OUT = f"{TOOLS_HOME}/corpus/cve-bench/logs/cvb_labels"
os.makedirs(CVB_OUT, exist_ok=True)

def get_token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()

def subsample_positions(positions, cap=40):
    if len(positions) <= cap:
        return positions
    step = (len(positions) - 1) / (cap - 1)
    idxs = sorted(set(int(i * step) for i in range(cap)))
    return [positions[i] for i in idxs]

def sanitize_messages(messages):
    out = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue
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
            out.append({"role": role, "content": blocks})
    return out

def cvb_samples():
    """Extract all CVB samples' messages from the eval file. Returns {id: [messages]}."""
    out = {}
    z = zipfile.ZipFile(CVB_EVAL)
    for info in z.infolist():
        if not info.filename.startswith("samples/") or not info.filename.endswith(".json"):
            continue
        with open(z.filename, "rb") as f:
            f.seek(info.header_offset)
            hdr = f.read(30)
            nl, el = struct.unpack("<HH", hdr[26:30])
            f.seek(info.header_offset + 30 + nl + el)
            raw = f.read(info.compress_size)
        data = json.loads(zstandard.ZstdDecompressor().decompressobj().decompress(raw))
        msgs = data.get("messages", [])
        if not msgs:
            continue
        # normalize: ensure content is a list of blocks
        norm = []
        first_user_seen = False
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            norm.append({"role": role, "content": content})
            if role == "user" and not first_user_seen:
                first_user_seen = True
        if not first_user_seen:
            # prepend the task input if no user message exists
            norm.insert(0, {"role": "user", "content": [{"type": "text", "text": data.get("input", "")}]})
        out[info.filename.split("/")[-1][:-5]] = norm
    return out

def discover(cvb_msgs):
    entries = []
    for log in sorted(glob.glob(CG_GLOB)):
        repo = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(log))))
        entries.append({"type": "cybergym", "repo": repo, "key": repo, "path": log})
    for p in sorted(glob.glob(DS_GLOB)):
        entries.append({"type": "deepswe", "repo": None, "key": p.split("/")[-5][:60], "path": p})
    for sid in cvb_msgs:
        entries.append({"type": "cvebench", "repo": None, "key": sid, "path": f"cvb:{sid}"})
    return entries

def call_api(messages, model, max_tokens, token, cache_bp=None):
    msgs = copy.deepcopy(messages)
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [dict(BILLING)],
        "messages": msgs,
    }
    if cache_bp:
        for idx in cache_bp:
            if 0 <= idx < len(msgs):
                msg = msgs[idx]
                content = msg.get("content")
                if content and isinstance(content[-1], dict):
                    content[-1] = dict(content[-1])
                    content[-1]["cache_control"] = {"type": "ephemeral"}
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

def build_messages(entry, cvb_msgs):
    if entry["type"] == "cvebench":
        return sanitize_messages(cvb_msgs[entry["key"]])
    messages, _refusals = parse_transcript(entry["path"])
    if entry["type"] == "cybergym":
        task_prompt = get_prompt_e2e(entry["repo"])
        messages = [{"role": "user", "content": [{"type": "text", "text": task_prompt}], "uuid": "task_prompt"}] + messages
    return sanitize_messages(messages)

def label_outpath(entry, model):
    if entry["type"] == "cvebench":
        return f"{CVB_OUT}/{entry['key']}.labels.{model}.jsonl"
    return f"{entry['path']}.labels.{model}.jsonl"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["discover", "label"])
    ap.add_argument("--models", default="claude-opus-5-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cvb_msgs = cvb_samples()
    entries = discover(cvb_msgs)
    if args.mode == "discover":
        by_type = {}
        for e in entries:
            by_type.setdefault(e["type"], []).append(e["key"])
        for t, keys in by_type.items():
            print(f"{t}: {len(keys)}")
            for k in keys[:3]:
                print(f"  e.g. {k}")
        print(f"total: {len(entries)}")
        return

    models = args.models.split(",")
    if args.limit:
        entries = entries[:args.limit]

    todo = []
    for e in entries:
        for model in models:
            if not os.path.exists(label_outpath(e, model)):
                e["model"] = model
                todo.append(dict(e))
    print(f"{len(todo)} (episode, model) pairs to label "
          f"(of {len(entries) * len(models)} total; existing skipped)")

    def work(item):
        entry, model = item, item["model"]
               # placeholder, replaced below
        pass

    def process(entry):
        model = entry["model"]
        token = get_token()
        try:
            messages = build_messages(entry, cvb_msgs)
        except Exception as ex:
            print(f"  {entry['key']}: PARSE ERROR {ex}", flush=True)
            return
        user_positions = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        sampled = subsample_positions(user_positions, args.cap)
        out_path = label_outpath(entry, model)
        results = []
        n = len(messages)
        cache_bp = sorted(set([0, n // 4, n // 2]))
        for i in sampled:
            attempt, done = 0, False
            while not done and attempt < 4:
                try:
                    r = call_api(messages[:i+1], model, 1, token, cache_bp=cache_bp)
                    sr = r.get("stop_reason")
                    sd = r.get("stop_details")
                    label = "STOP" if sr == "refusal" else "ok"
                    results.append({"pos": i, "label": label, "stop_reason": sr, "stop_details": sd})
                    done = True
                except urllib.error.HTTPError as ex:
                    if ex.code == 401 and attempt == 0:
                        token = get_token()  # token may have been refreshed
                    elif ex.code in (429, 500, 502, 503, 529):
                        time.sleep(10 * (attempt + 1))
                    else:
                        results.append({"pos": i, "label": "ERROR", "stop_reason": f"http_{ex.code}"})
                        done = True
                except Exception as ex:
                    if attempt >= 3:
                        results.append({"pos": i, "label": "ERROR", "stop_reason": str(ex)[:80]})
                        done = True
                    else:
                        time.sleep(5)
                attempt += 1
        if results:
            with open(out_path, "w") as f:
                payload = {"entry": entry["key"], "model": model, "type": entry["type"],
                           "n_user_positions": len(user_positions), "labels": results}
                f.write(json.dumps(payload) + "\n")
            stops = sum(1 for r in results if r["label"] == "STOP")
            print(f"{entry['key']} [{model}]: {len(results)} labels, {stops} stops -> {out_path}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for e in todo:
            futures.append(ex.submit(process, e))
        for f in concurrent.futures.as_completed(futures):
            pass

if __name__ == "__main__":
    main()
