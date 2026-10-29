#!/usr/bin/env python3
"""Build dataset/v2: one record per (episode, position).

- Exact reconstruction of what opus saw: same episode discovery, sanitize
  and prefix slicing as relabel_v2.py (labels from the relabel run).
- Serializes each prefix through the training chat template and records
  token counts; raw messages are NOT stored (rebuildable from corpus).
- Output: /work/dataset/v2/records.jsonl

Run inside the training image (needs transformers):
  docker run --rm -v /work:/work -w /work \
    <cuda-image> python tools/build_dataset_v2.py --model Qwen/Qwen3-4B
"""
import json, os, sys, glob, re, argparse

sys.path.insert(0, "/work/tools")
from replay_labeler2 import parse_transcript, get_prompt_e2e
from batch_labeler_v2 import cvb_samples

TOOLS_HOME = "/work"
CG_GLOB = f"{TOOLS_HOME}/corpus/cybergym-e2e/agent_output_v2/*/20*/trajectory/attempt_1.log"
DS_GLOB = f"{TOOLS_HOME}/containers"  # unused
DS_GLOB = f"{TOOLS_HOME}/corpus/deep-swe-data/jobs_v2/*/*/agent/sessions/projects/*/*.jsonl"
CVB_OUT = f"{TOOLS_HOME}/corpus/cve-bench/logs/cvb_labels"
OUT_DIR = f"{TOOLS_HOME}/dataset/v2"

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
    """Sanitized Anthropic blocks -> chat-template friendly message list."""
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

def group_key(e):
    if e["type"] == "cybergym":
        return e["repo"]
    if e["type"] == "deepswe":
        m = re.search(r"jobs_v2/([^/]+)/", e["labelfile"])
        return f"ds-{m.group(1)}" if m else e["key"]
    m = re.search(r"(CVE-\d{4}-\d{4,})", os.path.basename(e["labelfile"]))
    return f"cvb-{m.group(1)}" if m else e["key"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B")
    ap.add_argument("--out", default=f"{OUT_DIR}/records.jsonl")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    episodes = []
    for lf in sorted(glob.glob(CG_GLOB + ".labels.claude-opus-5-5.jsonl")):
        path = lf.replace(".labels.claude-opus-5-5.jsonl", "")
        repo = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))
        episodes.append({"type": "cybergym", "repo": repo, "path": path, "labelfile": lf, "key": path})
    for lf in sorted(glob.glob(DS_GLOB + ".labels.claude-opus-5-5.jsonl")):
        episodes.append({"type": "deepswe", "repo": None, "path": lf.replace(".labels.claude-opus-5-5.jsonl", ""), "labelfile": lf, "key": lf})
    for lf in sorted(glob.glob(f"{CVB_OUT}/*.labels.claude-opus-5-5.jsonl")):
        episodes.append({"type": "cvebench", "repo": None, "path": None, "labelfile": lf, "key": lf})
    print(f"{len(episodes)} episodes", flush=True)

    cvb_msgs = cvb_samples()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print("tokenizer loaded", flush=True)

    n_written = n_skipped = 0
    with open(args.out, "w") as fout:
        for e in episodes:
            lf2 = e["labelfile"].replace(".labels.claude-opus-5-5.jsonl", ".labels2.claude-opus-5-5.jsonl")
            if not os.path.exists(lf2):
                print(f"SKIP missing labels2 for {e['labelfile']}", flush=True)
                n_skipped += 1
                continue
            rec2 = json.loads(open(lf2).readline())
            if not rec2.get("labels"):
                print(f"SKIP no labels for {e['labelfile']}", flush=True)
                n_skipped += 1
                continue
            if e["type"] == "cybergym":
                msgs, _ = parse_transcript(e["path"])
                task_prompt = get_prompt_e2e(e["repo"])
                msgs = sanitize_messages([{"role": "user", "content": [{"type": "text", "text": task_prompt}]}] + msgs)
            elif e["type"] == "deepswe":
                msgs = sanitize_messages(parse_transcript(e["path"])[0])
            else:
                msgs = sanitize_messages(cvb_msgs[e["key"].split("/")[-1].replace(".labels.claude-opus-5-5.jsonl", "")])
            labels = {l["pos"]: l["label"] for l in rec2.get("labels", [])}
            for pos in sorted(labels.keys()):
                prefix = to_template_msgs(msgs[:pos + 1])
                try:
                    ids = tok.apply_chat_template(prefix, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                    n_tokens = len(ids) if isinstance(ids, (list, __import__("typing").get_args(list))) or not hasattr(ids, "keys") else len(ids["input_ids"])
                    chat_text = tok.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                except Exception as ex:
                    n_tokens, chat_text = None, None
                    print(f"TEMPLATE ERROR {e['labelfile']} pos {pos}: {ex}", flush=True)
                rec = {
                    "id": f"{e['type']}-{episodes.index(e)}-{pos}",
                    "source": e["type"], "episode": e["key"], "repo": e.get("repo"),
                    "group": group_key(e), "pos": pos, "label": labels.get(pos),
                    "n_tokens": n_tokens, "chat_text": chat_text,
                }
                fout.write(json.dumps(rec) + "\n")
                n_written += 1
            fout.flush()
            print(f"{e['type'][:3]} {str(e['key'])[:60]}: {len(labels)} positions", flush=True)
    print(f"wrote {n_written} records, skipped {n_skipped} episodes", flush=True)

main()
