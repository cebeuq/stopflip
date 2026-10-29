#!/usr/bin/env python3
"""Generate escalation candidates (T3 unicode, scaled up) and write cands JSONL. v2: r1 grid + orig rows."""
import json, sys, os, argparse, random, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from escalate_lib import (parse, rebuild, STR, TOOL_START, TOOL_END,
                          perturb_scale, CYRILLIC, MARKERS)

def build_spans(msgs):
    """Return list of (mi, role, key, start, end) field spans over a rebuilt ct."""
    parts = ['<bos>']
    fields = []
    for mi, m in enumerate(msgs):
        if m['role'] == 'user':
            s = len(''.join(parts)) + len('<|turn>user\n')
            parts.append('<|turn>user\n' + m['text'] + '<turn|>\n')
            fields.append((mi, m['role'], 'text', s, s + len(m['text'])))
        else:
            if m['args'] is None:
                s = len(''.join(parts)) + len('<|turn>model\n')
                parts.append('<|turn>model\n' + m['text'] + '<turn|>\n')
                fields.append((mi, m['role'], 'text', s, s + len(m['text'])))
                continue
            # tool_use block
            s = len(''.join(parts)) + len('<|turn>model\n') + len(m['text'])
            argparts = []
            for k, v, p in m['args']:
                if p:
                    argparts.append(f'{k}:{v}')
                else:
                    argparts.append(f'{k}:{STR}{v}{STR}')
            joined = ','.join(argparts)
            parts.append('<|turn>model\n' + m['text'] + f'{TOOL_START}call:{m["tool"]}{{{joined}}}{TOOL_END}<|tool_response>')
            fields.append((mi, m['role'], '__tool__' + m['tool'], s,
                           s + len(m['text']) + len(f'{TOOL_START}call.{m["tool"]}{{{joined}}}{TOOL_END}')))
    return fields

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='/data/records_gemma.jsonl')
    ap.add_argument('--folds', default='/data/folds.jsonl')
    ap.add_argument('--eval-scores', default='/data/eval_ep0.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default='')
    ap.add_argument('--grid', default='r1')
    args = ap.parse_args()

    folds = {json.loads(l)['group']: json.loads(l)['fold'] for l in open(args.folds)}
    evals = {json.loads(l)['id']: json.loads(l) for l in open(args.eval_scores)}
    targets = {}
    for l in open(args.records):
        r = json.loads(l)
        if r.get('label') != 'STOP': continue
        if folds.get(r.get('group')) != 0: continue
        ps = evals.get(r['id'], {}).get('p_stop', None)
        if ps is not None and 0.5 <= ps <= 0.999:
            targets[r['id']] = (r, ps)

    grids = {
        'r1': [
            ('s1w1800',  1.0, 1800, 1, 100),
            ('s2w1800',  2.0, 1800, 1, 101),
            ('s4w1800',  4.0, 1800, 1, 102),
            ('s8w1800',  8.0, 1800, 1, 103),
            ('s4w1800r2', 4.0, 1800, 2, 104),
        ],
        'r2': [
            ('s4wall',  4.0,    0, 1, 200),
            ('s8wall',  8.0,    0, 1, 201),
            ('s16wall', 16.0,   0, 1, 202),
            ('s2wall',  2.0,    0, 1, 203),
            ('s32w1800', 32.0, 1800, 1, 204),
            ('s64wall', 64.0,   0, 1, 205),
            ('s4w1800s2',  4.0, 1800, 2, 206),
            ('s8w3600',  8.0, 3600, 1, 207),
        ],
        'r3': [
            ('s2w1800',  2.0, 1800, 1, 300),
            ('s4w1800',  4.0, 1800, 1, 301),
        ],
    }
    cfgs = grids[args.grid]
    ids = [x.strip() for x in args.ids.split(',') if x.strip()]
    recs = []
    for tid in ids:
        if tid not in targets:
            print('MISSING', tid); continue
        recs.append(tid)
    recs.sort(key=lambda t: targets[t][1])
    out = open(args.out, 'w')
    n = 0
    for tid in recs:
        r, ps = targets[tid]
        ct = r['chat_text']
        try:
            msgs = parse(ct)
        except ValueError as e:
            print('PARSE FAIL', tid, e); continue
        if rebuild(msgs) != ct:
            print('REBUILD FAIL', tid); continue
        fields = build_spans(msgs)
        out.write(json.dumps({'id': tid, 'cfg': 'orig', 'chat_text_prime': ct,
                              'fields': [[mi, role, key, s, e] for (mi, role, key, s, e) in fields]}) + '\n')
        n += 1
        for (name, scale, window, rounds, seed) in cfgs:
            new_msgs = copy.deepcopy(msgs)
            rng = random.Random((seed << 24) ^ (len(ct) & 0xffffff))
            for fi, (mi, role, key, s, e) in enumerate(fields):
                wstart = len(ct) - window if window else 0
                lo = max(s, wstart)
                hi = min(e, len(ct))
                if hi <= lo:
                    continue
                ftext = ct[lo:hi]
                prot = set()
                for i in range(s, lo):
                    prot.add(i - s)
                for i in range(hi, e):
                    prot.add(i - s)
                for mk in MARKERS:
                    st = 0
                    while True:
                        i = ftext.find(mk, st)
                        if i == -1: break
                        prot.update(range(i, i + len(mk)))
                        st = i + 1
                pert = ftext
                for _ in range(rounds):
                    pert = perturb_scale(pert, rng, scale, prot)
                if pert != ftext:
                    new_msgs[mi]['__new__' + str(fi)] = pert
            for fi, (mi, role, key, s, e) in enumerate(fields):
                nv = new_msgs[mi].pop('__new__' + str(fi), None)
                if nv is None:
                    continue
                new_msgs[mi]['text'] = nv
            new_ct = rebuild(new_msgs)
            out.write(json.dumps({'id': tid, 'cfg': name, 'scale': scale, 'window': window,
                                  'rounds': rounds, 'seed': seed, 'chat_text_prime': new_ct,
                  'fields': [[mi, role, key, s, e] for (mi, role, key, s, e) in fields]}) + '\n')
            n += 1
    out.close()
    print('wrote', n, 'candidates')

if __name__ == '__main__':
    main()
===GEN3===
#!/usr/bin/env python3
"""Escalate branch gen v3: sal_common-based structure, per-field edits, fixed spans."""
import json, sys, os, argparse, random, copy
sys.path.insert(0, "/data")
from sal_common import parse_chat_text, render_chat_text
from escalate_lib import perturb_scale, MARKERS

def build_fields(messages):
    """Live-side-aligned editable fields. mirrors sal_run.build_fields (quoted input keys)."""
    fields = []
    for mi, msg in enumerate(messages):
        if msg["role"] == "user":
            for bi, b in enumerate(msg["blocks"]):
                if b["type"] == "text" and b["text"]:
                    fields.append({"mi": mi, "bi": bi, "key": "text", "text": b["text"]})
        elif msg["role"] == "assistant":
            bi = 0
            for b in msg["blocks"]:
                if b["type"] == "text":
                    if b["text"]:
                        fields.append({"mi": mi, "bi": bi, "key": "text", "text": b["text"]})
                elif b["type"] == "tool_use":
                    for k, v in b["fields"].items():
                        if isinstance(v, str) and k in b.get("_quoted", ()):
                            fields.append({"mi": mi, "bi": bi, "key": k, "text": v})
                bi += 1
    return fields

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='/data/records_gemma.jsonl')
    ap.add_argument('--folds', default='/data/folds.jsonl')
    ap.add_argument('--eval-scores', default='/data/eval_ep0.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default='')
    ap.add_argument('--grid', default='r1')
    args = ap.parse_args()

    folds = {json.loads(l)['group']: json.loads(l)['fold'] for l in open(args.folds)}
    evals = {json.loads(l)['id']: json.loads(l) for l in open(args.eval_scores)}
    targets = {}
    for l in open(args.records):
        r = json.loads(l)
        if r.get('label') != 'STOP': continue
        if folds.get(r.get('group')) != 0: continue
        ps = evals.get(r['id'], {}).get('p_stop', None)
        if ps is not None and 0.5 <= ps <= 0.999:
            targets[r['id']] = (r, ps)

    grids = {
        'r1': [
            ('s1w1800',  1.0, 1800, 1, 100),
            ('s2w1800',  2.0, 1800, 1, 101),
            ('s4w1800',  4.0, 1800, 1, 102),
            ('s8w1800',  8.0, 1800, 1, 103),
            ('s4w1800r2', 4.0, 1800, 2, 104),
        ],
        'r2': [
            ('s4wall',  4.0,    0, 1, 200),
            ('s8wall',  8.0,    0, 1, 201),
            ('s16wall', 16.0,   0, 1, 202),
            ('s2wall',  2.0,    0, 1, 203),
            ('s32w1800', 32.0, 1800, 1, 204),
            ('s16wallr2', 16.0, 0, 2, 205),
        ],
        'r4': [
            ('s16w1800r2',  16.0, 1800, 2, 320),
            ('s16w1800r4',  16.0, 1800, 4, 321),
            ('s8w1800r4',   8.0, 1800, 4, 322),
            ('s32w1800r4',  32.0, 1800, 4, 323),
            ('s8wallr2',    8.0, 0, 2, 324),
            ('s16wallr4',  16.0, 0, 4, 325),
        ],
        'r3': [
            ('s8w1800r2',  8.0, 1800, 2, 310),
            ('s8w1800r4',  8.0, 1800, 4, 311),
            ('s16w1800',  16.0, 1800, 1, 312),
            ('s32w1800r2', 32.0, 1800, 2, 313),
            ('s4wallr2',  4.0, 0, 2, 314),
        ],
    }
    cfgs = grids[args.grid]
    ids = [x.strip() for x in args.ids.split(',') if x.strip()]
    recs = []
    for tid in ids:
        if tid not in targets:
            print('MISSING', tid); continue
        recs.append(tid)
    recs.sort(key=lambda t: targets[t][1])
    out = open(args.out, 'w')
    n = 0
    for tid in recs:
        r, ps = targets[tid]
        ct = r['chat_text']
        msgs = parse_chat_text(ct)
        if msgs is None:
            print('PARSE FAIL', tid); continue
        if render_chat_text(msgs) != ct:
            print('REBUILD FAIL', tid); continue
        fields = build_fields(msgs)
        # derive spans by ordered find with tight gap constraint
        spans = []
        pos = 0
        ok = True
        for f in fields:
            s = ct.find(f["text"], pos)
            if s == -1 or s - pos > 200:
                ok = False; break
            spans.append((s, s + len(f["text"])))
            pos = s + len(f["text"])
        if not ok:
            print('SPAN FAIL', tid); continue
        out.write(json.dumps({'id': tid, 'cfg': 'orig', 'chat_text_prime': ct, 'edits': []}) + '\n')
        n += 1
        for (name, scale, window, rounds, seed) in cfgs:
            new_msgs = copy.deepcopy(msgs)
            rng = random.Random((seed << 24) ^ (len(ct) & 0xffffff))
            edits = []
            for fi, f in enumerate(fields):
                s, e = spans[fi]
                ftext = f["text"]
                wstart = len(ct) - window if window else 0
                lo = max(s, wstart)
                hi = min(e, len(ct))
                if hi <= lo:
                    continue
                prot = set()
                for i in range(s, lo):
                    prot.add(i - s)
                for i in range(hi, e):
                    prot.add(i - s)
                for mk in MARKERS:
                    st = 0
                    while True:
                        i = ftext.find(mk, st)
                        if i == -1: break
                        prot.update(range(i, i + len(mk)))
                        st = i + 1
                pert = ftext
                for _ in range(rounds):
                    pert = perturb_scale(pert, rng, scale, prot)
                if pert == ftext:
                    continue
                b = new_msgs[f["mi"]]["blocks"][f["bi"]]
                if new_msgs[f["mi"]]["blocks"][f["bi"]]["type"] == "tool_use":
                    b["fields"][f["key"]] = pert
                else:
                    b["text"] = pert
                edits.append({"mi": f["mi"], "bi": f["bi"], "key": f["key"],
                              "old": ftext, "new": pert})
            new_ct = render_chat_text(new_msgs)
            out.write(json.dumps({'id': tid, 'cfg': name, 'scale': scale, 'window': window,
                                  'rounds': rounds, 'seed': seed, 'chat_text_prime': new_ct,
                                  'edits': edits}) + '\n')
            n += 1
    out.close()
    print('wrote', n, 'candidates')

if __name__ == '__main__':
    main()