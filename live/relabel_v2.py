#!/usr/bin/env python3
"""Relabel ALL positions with tools defined (fixes 'Tool reference not found' 400s).

- Reads original label files for the exact labeled positions (position parity guaranteed)
- Writes .labels2.claude-opus-5-5.jsonl (originals untouched)
- tools param defines all observed tool names with permissive schemas
"""
import json, os, sys, time, glob, copy
import urllib.request, urllib.error
import concurrent.futures

sys.path.insert(0, "/work/tools")
from replay_labeler2 import parse_transcript, get_prompt_e2e
from batch_labeler_v2 import cvb_samples

TOOLS_HOME = "/work"
CG_GLOB = f"{TOOLS_HOME}/corpus/cybergym-e2e/agent_output_v2/*/20*/trajectory/attempt_1.log"
DS_GLOB = f"{TOOLS_HOME}/corpus/deep-swe-data/jobs_v2/*/*/agent/sessions/projects/*/*.jsonl"
CVB_OUT = f"{TOOLS_HOME}/corpus/cve-bench/logs/cvb_labels"
API = "https://api.anthropic.com/v1/messages"
BILLING = {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.296.cf4; cc_entrypoint=sdk-cli;"}
TOOLS = [{"name": t, "input_schema": {"type": "object", "properties": {}, "additionalProperties": True}}
         for t in ["Bash", "Read", "Write", "Edit", "Monitor", "ScheduleWakeup", "ToolSearch"]]

def get_token():
    with open(f"{TOOLS_HOME}/tmp/oauth_token.txt") as f:
        return f.read().strip()

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

def call_api(messages, model, cache_bp):
    msgs = copy.deepcopy(messages)
    body = {"model": model, "max_tokens": 1, "system": [dict(BILLING)], "messages": msgs, "tools": TOOLS}
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
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        r = json.loads(resp.read())
    return r, time.time() - t0

def robust_call(messages, model, cache_bp):
    attempt = 0
    while True:
        try:
            r, lat = call_api(messages, model, cache_bp)
            return r.get("stop_reason"), r.get("stop_details"), lat, None
        except urllib.error.HTTPError as ex:
            try:
                err_body = ex.read().decode()[:400]
            except Exception:
                err_body = "?"
            if ex.code in (429, 500, 502, 503, 529, 401):
                time.sleep(min(60, 15 * (attempt + 1)))
                attempt += 1
                if attempt >= 24:
                    return None, None, None, f"http_{ex.code}. {err_body}"
                continue
            return None, None, None, f"http_{ex.code}. {err_body}"
        except Exception:
            if attempt >= 8:
                return None, None, None, "retries_exhausted"
            time.sleep(5)
            attempt += 1

print("loading episodes...", flush=True)
cvb_msgs = cvb_samples()

episodes = []
for lf in sorted(glob.glob(CG_GLOB + ".labels.claude-opus-5-5.jsonl")):
    path = lf.replace(".labels.claude-opus--5-5.jsonl", "").replace(".labels.claude-opus-5-5.jsonl", "")
    repo = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
    episodes.append({"type": "cybergym", "repo": repo, "path": path, "labelfile": lf, "key": path})
for lf in sorted(glob.glob(DS_GLOB + ".labels.claude-opus-5-5.jsonl")):
    episodes.append({"type": "deepswe", "repo": None, "path": lf.replace(".labels.claude-opus-5-5.jsonl", ""), "labelfile": lf, "key": lf})
for lf in sorted(glob.glob(f"{CVB_OUT}/*.labels.claude-opus-5-5.jsonl")):
    episodes.append({"type": "cvebench", "repo": None, "path": None, "labelfile": lf, "key": lf})

def outpath(e):
    return e["labelfile"].replace(".labels.claude-opus-5-5.jsonl", ".labels2.claude-opus-5-5.jsonl")

todo = []
for e in episodes:
    if not os.path.exists(outpath(e)):
        todo.append(e)
print(f"{len(todo)} episodes to relabel", flush=True)

def build_messages(e):
    if e["type"] == "cybergym":
        msgs, _ = parse_transcript(e["path"])
        task_prompt = get_prompt_e2e(e["repo"])
        return sanitize_messages([{"role": "user", "content": [{"type": "text", "text": task_prompt}]}] + msgs)
    if e["type"] == "deepswe":
        return sanitize_messages(parse_transcript(e["path"])[0])
    return sanitize_messages(cvb_msgs[e["key"].split("/")[-1].replace(".labels.claude-opus-5-5.jsonl", "")])

def process(e):
    orig = json.loads(open(e["labelfile"]).readline())
    positions = [l["pos"] for l in orig.get("labels", [])]
    if not positions:
        return
    messages = build_messages(e)
    n = len(messages)
    cache_bp = sorted(set([0, n // 4, n // 2]))
    new_labels, agree = [], 0
    for pos in positions:
        sr, sd, lat, err = robust_call(messages[:pos + 1], "claude-opus-5-5", cache_bp)
        label = "STOP" if sr == "refusal" else ("ok" if sr else None)
        if label and any(l["pos"] == pos and l["label"] == label for l in orig.get("labels", [])):
            agree += 1
        new_labels.append({"pos": pos, "label": label, "stop_reason": sr, "stop_details": sd})
    rec = {"entry": orig.get("entry"), "model": "claude-opus-5-5", "type": e["type"],
           "n_user_positions": orig.get("n_user_positions"), "labels": new_labels}
    with open(outpath(e), "w") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{e['type'][:3]} {str(orig.get('entry'))[:40]}: {len(new_labels)} labels, orig-agree {agree}/{len(positions)}", flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(process, e): e for e in todo}
    for f in concurrent.futures.as_completed(futs):
        try:
            f.result()
        except Exception as ex:
            print(f"EPISODE FAILED {futs[f]['labelfile']}: {ex}", flush=True)
print("RELABEL COMPLETE", flush=True)
