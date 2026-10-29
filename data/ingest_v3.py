#!/usr/bin/env python3
"""Convert AFTraj-2K, ATBench, ATBench500, agentic-redteam-benchmark into
our record format (records_gemma.jsonl-compatible, label=null placeholder).

For each source:
  - build a sanitized message list per trajectory (same structure the
    live pipeline's sanitize_messages() produces: role user/assistant,
    blocks text/tool_use/tool_result),
  - emit ONE RECORD PER MESSAGE POSITION pos: chat_text = Gemma-template
    rendering of to_template_msgs(msgs[:pos+1]) with add_generation_prompt=True,
    exactly like tools/build_dataset_v2.py,
  - write a per-trajectory sidecar with the full sanitized message list
    so tools/live_prep_v2.py-style rebuilds can slice prefixes.

Run inside <cuda-image>:
  docker run --rm -v /work:/work -w /work \
    <cuda-image> python corpus/v3/convert_v3.py
"""
import json, os, sys
import pandas as pd

sys.path.insert(0, "/work/tools")
from sal_common import parse_chat_text, render_chat_text

from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained("/work/tmp/gemma4-12b")

OUT = "/work/dataset/v3/candidates"
os.makedirs(OUT, exist_ok=True)


def render_template_msgs(msgs):
    """sanitized msgs -> chat_text via the same code path as build_dataset_v2."""
    from live_harness import to_template_msgs
    tt = to_template_msgs(msgs)
    return TOK.apply_chat_template(tt, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def n_tokens_of(msgs):
    from live_harness import to_template_msgs
    tt = to_template_msgs(msgs)
    ids = TOK.apply_chat_template(tt, tokenize=True, add_generation_prompt=True,
                                  enable_thinking=False)
    return len(ids)


def txt_block(s):
    return {"type": "text", "text": s if isinstance(s, str) else str(s)}


def tu_block(name, args):
    return {"type": "tool_use", "id": "", "name": name, "input": args}


def parse_args_string(a):
    """'arguments' field is usually a JSON string -> dict."""
    if isinstance(a, dict):
        return dict(a)
    if isinstance(a, str):
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {"value": v}
        except Exception:
            return {"value": a}
    return {"value": str(a)}


def check_roundtrip(msgs, tag):
    ct = render_template_msgs(msgs)
    p = parse_chat_text(ct)
    if p is None or render_chat_text(p) != ct:
        return False
    return True


def emit_records(fout, msgs, source, episode, native_base, pos_native_fn):
    n = 0
    for i in range(len(msgs)):
        prefix = msgs[:i + 1]
        chat_text = render_template_msgs(prefix)
        native = dict(native_base)
        if pos_native_fn is not None:
            native.update(pos_native_fn(i))
        rec = {
            "id": f"{source}-{episode}-{i}",
            "source": source,
            "episode": episode,
            "repo": None,
            "group": episode,
            "pos": i,
            "label": None,
            "n_tokens": None,
            "chat_text": chat_text,
            "native": native,
        }
        fout.write(json.dumps(rec) + "\n")
        n += 1
    return n


# ---------------------------------------------------------------- AFTraj

def convert_aftraj():
    n_traj = n_rec = 0
    fails = 0
    with open(f"{OUT}/aftraj.jsonl", "w") as fout, \
         open(f"{OUT}/aftraj_messages.jsonl", "w") as mout:
        for split in ["safe", "unsafe"]:
            df = pd.read_parquet(f"/work/corpus/v3/aftraj/raw/aftraj_{split}.parquet")
            for _, row in df.iterrows():
                turns = row["turns"]
                msgs = []
                for t in turns:
                    role = t["role"]
                    if role == "environment":
                        blocks = [{"type": "tool_result", "tool_use_id": "",
                                   "content": t.get("content", "")}]
                        msgs.append({"role": "user", "content": blocks})
                    elif role == "user":
                        blocks = [txt_block(t.get("content") or t.get("thought") or "")]
                        msgs.append({"role": "user", "content": blocks})
                    else:
                        blocks = []
                        a = t.get("action")
                        if a:
                            try:
                                calls = json.loads(a)
                            except Exception:
                                try:
                                    import ast
                                    calls = ast.literal_eval(a)
                                except Exception:
                                    calls = None
                            if isinstance(calls, list):
                                for c in calls:
                                    if isinstance(c, dict) and "name" in c:
                                        blocks.append(tu_block(c["name"],
                                                   parse_args_string(c.get("arguments"))))
                                    else:
                                        blocks.append(txt_block(str(c)))
                            elif calls is not None:
                                blocks.append(txt_block(a))
                            else:
                                blocks.append(txt_block(a))
                        content = t.get("content")
                        if content:
                            blocks.insert(0, txt_block(content))
                        if not blocks:
                            blocks = [txt_block("")]
                        msgs.append({"role": "assistant", "content": blocks})
                ct = render_template_msgs(msgs)
                p = parse_chat_text(ct)
                if p is None or render_chat_text(p) != ct:
                    fails += 1
                    print(f"AFTRAJ RT-FAIL {row['conv_id']}", flush=True)
                native_base = {
                    "is_unsafe": split == "unsafe",
                    "domain": row["domain"],
                    "unsafe_source": row.get("unsafe_source"),
                    "mistake_step": row.get("mistake_step"),
                    "mistake_agent": row.get("mistake_agent"),
                    "mistake_reason": row.get("mistake_reason"),
                }
                ms = row.get("mistake_step")
                n_traj += 1
                mout.write(json.dumps({
                    "source": "aftraj", "episode": row["conv_id"],
                    "n_msgs": len(msgs), "messages": msgs}) + "\n")

                def pos_native(i, ms=ms, split=split):
                    return {"hint_alarm": bool(
                        split == "unsafe" and ms is not None and i >= ms)}
                n_rec += emit_records(fout, msgs, "aftraj", row["conv_id"],
                                      native_base, pos_native)
    print(f"aftraj: {n_traj} trajectories, {n_rec} records, {fails} roundtrip fails",
          flush=True)
    return n_traj, n_rec


# ---------------------------------------------------------------- ATBench

def convert_atbench():
    n_traj = n_rec = 0
    fails = 0
    for fname, source in [("ATBench_test.json", "atbench"),
                          ("ATBench500_test.json", "atbench500")]:
        data = json.load(open(f"/work/corpus/v3/atbench/raw/{fname}"))
        with open(f"{OUT}/{source}.jsonl", "w") as fout, \
             open(f"{OUT}/{source}_messages.jsonl", "w") as mout:
            for r in data:
                msgs_raw = r.get("contents") or r.get("content")
                conv_id = r.get("conv_id") or r.get("id"))
                if isinstance(msgs_raw[0], list):
                    msgs_raw = msgs_raw[0]
                msgs = []
                for m in msgs_raw:
                    if m["role"] == "user":
                        msgs.append({"role": "user",
                                     "content": [txt_block(m.get("content") or "")]})
                    elif m["role"] == "environment":
                        msgs.append({"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "",
                             "content": m.get("content") or ""}]})
                    else:  # agent
                        blocks = []
                        if m.get("thought"):
                            blocks.append(txt_block(m["thought"]))
                        act = m.get("action")
                        if act:
                            if act.startswith("Complete"):
                                try:
                                    inner = json.loads(act[len("Complete"):])
                                    blocks.append(txt_block(inner.get("response", act)))
                                except Exception:
                                    blocks.append(txt_block(act))
                            else:
                                try:
                                    a = json.loads(act)
                                    if isinstance(a, dict) and "name" in a:
                                        blocks.append(tu_block(a["name"],
                                            parse_args_string(a.get("arguments"))))
                                    else:
                                        blocks.append(txt_block(act))
                                except Exception:
                                    blocks.append(txt_block(act))
                        if not blocks:
                            blocks = [txt_block("")]
                        msgs.append({"role": "assistant", "content": blocks})
                ct = render_template_msgs(msgs)
                p = parse_chat_text(ct)
                if p is None or render_chat_text(p) != ct:
                    fails += 1
                    print(f"{source} RT-FAIL {conv_id}", flush=True)
                native_base = {
                    "label": r.get("label"),
                    "risk_source": r.get("risk_source"),
                    "failure_mode": r.get("failure_mode"),
                    "real_world_harm": r.get("real_world_harm"),
                    "reason": r.get("reason"),
                }
                n_traj += 1
                mout.write(json.dumps({
                    "source": source, "episode": conv_id,
                    "n_msgs": len(msgs), "messages": msgs}) + "\n")
                n_rec += emit_records(fout, msgs, source, str(conv_id),
                                      native_base, None)
    print(f"atbench: {n_traj} trajectories, {n_rec} records, {fails} roundtrip fails",
          flush=True)
    return n_traj, n_rec


# ------------------------------------------------------------- agentic-redteam

def convert_agentic():
    n_traj = n_rec = 0
    fails = 0
    with open(f"{OUT}/agentic-redteam.jsonl", "w") as fout, \
         open(f"{OUT}/agentic-redteam_messages.jsonl", "w") as mout:
        for split in ["gold", "augmented"]:
            df = pd.read_parquet(f"/work/corpus/v3/agentic-redteam/raw/{split}.parquet")
            for _, row in df.iterrows():
                steps = [json.loads(s) for s in row["trajectory"]]
                gt = json.loads(row["ground_truth"])
                drift_step = row.get("drift_step")
                decision = row.get("expected_decision")
                msgs = [{"role": "user", "content": [txt_block(
                    "Goal: " + row["declared_goal"] + "\n\nPlan:\n" +
                    "\n".join(f"{i+1}. {s}" for i, s in enumerate(row["declared_plan"])))]}]
                for st in steps:
                    act = st["action"]
                    blocks = []
                    if st.get("thought"):
                        blocks.append(txt_block(st["thought"]))
                    if act["type"] in ("tool_call", "message"):
                        blocks.append(tu_block(act["name"], act.get("args") or {}))
                    elif act["type"] == "final_answer":
                        blocks.append(txt_block(act.get("raw")
                                                or json.dumps(act.get("args", {}))))
                    else:
                        blocks.append(txt_block(json.dumps(act)))
                    msgs.append({"role": "assistant", "content": blocks})
                    msgs.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "",
                         "content": st.get("observation") or ""}]})
                ct = render_template_msgs(msgs)
                p = parse_chat_text(ct)
                if p is None or render_chat_text(p) != ct:
                    fails += 1
                    print(f"agentic RT-FAIL {row['id']}", flush=True)

                def pos_native(i, drift_step=drift_step, decision=decision):
                    return {"hint_alarm": bool(
                        decision is not None and decision != "CONTINUE"
                        and drift_step is not None
                        and i >= 1 + 2 * drift_step)}
                native_base = {
                    "drift_step": drift_step,
                    "expected_decision_at_drift": decision,
                    "drift_signals_expected": gt.get("drift_signals_expected"),
                    "rationale": gt.get("rationale"),
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "difficulty": row["difficulty"],
                    "tier": row["tier"],
                }
                n_traj += 1
                mout.write(json.dumps({
                    "source": "agentic-redteam", "episode": row["id"],
                    "n_msgs": len(msgs), "messages": msgs}) + "\n")
                n_rec += emit_records(fout, msgs, "agentic-redteam", row["id"],
                                      native_base, pos_native)
    print(f"agentic-redteam: {n_traj} trajectories, {n_rec} records, "
          f"{fails} roundtrip fails", flush=True)
    return n_traj, n_rec


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    if args.only in (None, "aftraj"):
        convert_aftraj()
    if args.only in (None, "atbench"):
        convert_atbench()
    if args.only in (None, "agentic"):
        convert_agentic()
