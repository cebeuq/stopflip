#!/usr/bin/env python3
"""Live prep v2: rebuild baseline messages, apply wf_v2 edit manifests.

Edits in attack_v2.jsonl are anchored on parse_chat_text(chat_text) structure
(sal_common.py): (mi, bi, key, old, new), whole-field replacements.
  1. rebuild the original live message list (live_harness
     discover_episodes/build_messages), truncated to rec["pos"]+1;
  2. verify a byte-exact chat-template round-trip against
     records_gemma.jsonl's chat_text;
  3. apply the edit manifest in PARSE space ((mi, bi) direct indexing,
     old == whole field), verify render == perturbed_chat_text byte-exactly;
  4. map edits into the LIVE message space (block-index alignment between
     parse structure and sanitized live structure; each old substring occurs
     exactly once - whole field);
  5. verify the live-template render of messages_prime reproduces the
     surrogate-scored perturbed_chat_text byte-exactly;
  6. dump {"id", "messages", "messages_prime"} for live_harness probe.
"""
import json, sys, argparse, copy, os

TOOLS_HOME = os.environ.get("WORK", "/work")
sys.path.insert(0, f"{TOOLS_HOME}/tools")
from live_harness import (discover_episodes, build_messages,
                          sanitize_messages, to_template_msgs)
from sal_common import parse_chat_text, render_chat_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", default="/work/tmp/attack_v2.jsonl")
    ap.add_argument("--out", default="/work/tmp/live_cands_v2.jsonl")
    ap.add_argument("--records", default=f"{TOOLS_HOME}/dataset/v2/records_gemma.jsonl")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f"{TOOLS_HOME}/tmp/gemma4-12b-it")

    cands = [json.loads(l) for l in open(args.cands)]
    recs = [json.loads(l) for l in open(args.records)]
    rec_by_id = {r["id"]: r for r in recs}
    print(f"{len(cands)} candidates, {len(recs)} records", flush=True)

    episodes = discover_episodes()
    print(f"{len(episodes)} episodes", flush=True)

    def render(msgs):
        return tok.apply_chat_template(to_template_msgs(msgs), tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    n_ok = 0
    out = open(args.out, "w")
    for cand in cands:
        rid = cand["id"]
        rec = rec_by_id.get(rid)
        if rec is None:
            print(f"SKIP {rid}: not in records", flush=True)
            continue
        rec_type, ep_idx, pos = rid.rsplit("-", 2)
        e = episodes[int(ep_idx)]
        if e["type"] != rec_type:
            print(f"SKIP {rid}: type mismatch", flush=True)
            continue
        # --- live baseline messages + byte-exact round-trip ---
        messages = build_message_list = build_messages(e)[:rec["pos"] + 1]
        ct = render(messages)
        if ct != rec["chat_text"]:
            print(f"RT-DIFF {rid}: baseline round-trip mismatch", flush=True)
            continue
        # --- parse-space application of edits ---
        msgs_p = parse_chat_text(rec["chat_text"])
        if msgs_p is None or render_chat_text(msgs_p) != rec["chat_text"]:
            print(f"PARSE FAIL {rid}", flush=True)
            continue
        for ed in cand["edits"]:
            blk = msgs_p[ed["mi"]]["blocks"][ed["bi"]]
            if ed["key"] == "text":
                assert blk["type"] == "text" and blk["text"] == ed["old"], f"{rid} mi{ed['mi']} bi{ed['bi']} text anchor"
                blk["text"] = ed["new"]
            else:
                assert blk["type"] == "tool_use" and blk["fields"][ed["key"]] == ed["old"], f"{rid} mi{ed['mi']} bi{ed['bi']} {ed['key']} anchor"
                blk["fields"][ed["key"]] = ed["new"]
        if render_chat_text(msgs_p) != cand["perturbed_chat_text"]:
            print(f"PERT-DIFF {rid}: parse-space perturbation mismatch", flush=True)
            continue
        # --- map into live message space ---
        # the chat template's generation prompt parses as one extra trailing
        # assistant message with a single empty text block: drop it (edits
        # never reference it) before aligning with the live message list.
        if (len(msgs_p) == len(messages) + 1
                and msgs_p[-1]["role"] == "assistant"
                and msgs_p[-1]["blocks"] == [{"type": "text", "text": ""}]):
            msgs_p = msgs_p[:-1]
        if len(msgs_p) != len(messages):
            print(f"STRUCT-DIFF {rid}: {len(msgs_p)} parse msgs vs {len(messages)} live msgs", flush=True)
            continue
        mp = copy.deepcopy(messages)
        offsets = {}
        for pmi, (pm, lm) in enumerate(zip(msgs_p, mp)):
            if pm["role"] != lm["role"]:
                print(f"ROLE-DIFF {rid} at {pmi}", flush=True)
            pblocks = pm["blocks"]
            lcontent = lm["content"]
            if len(pblocks) == len(lcontent) + 1 and pblocks[0]["type"] == "text" and pblocks[0].get("text", "") == "":
                poff = 1
            elif len(pblocks) == len(lcontent):
                poff = 0
            else:
                poff = None
            if poff is None:
                print(f"BLOCK-DIFF {rid} msg {pmi}: {len(pblocks)} vs {len(lcontent)}", flush=True)
            offsets[pmi] = poff
        # apply edits in live space: (mi, bi) -> live (mi, bi - poff)
        for ed in cand["edits"]:
            lmi, lbi = ed["mi"], ed["bi"] - offsets.get(ed["mi"], 0)
            lb = mp[lmi]["content"][lbi]
            if ed["key"] == "text":
                assert lb["type"] == "text" and lb["text"] == ed["old"], f"{rid} live text anchor mi{lmi} bi{lbi}"
                lb["text"] = ed["new"]
            else:
                assert lb["type"] == "tool_use" and lb["input"][ed["key"]] == ed["old"], f"{rid} live {ed['key']} anchor mi{lmi} bi{lbi}"
                lb["input"][ed["key"]] = ed["new"]
        if render(mp) != cand["perturbed_chat_text"]:
            print(f"PERT-DIFF-LIVE {rid}: live-space perturbation mismatch", flush=True)
            continue
        out.write(json.dumps({"id": rid, "messages": messages, "messages_prime": mp}) + "\n")
        out.flush()
        print(f"OK {rid} ({len(cand['edits'])} edits)", flush=True)
        n_ok += 1
    print(f"prep done: {n_ok}/{len(cands)}", flush=True)

if __name__ == "__main__":
    main()
