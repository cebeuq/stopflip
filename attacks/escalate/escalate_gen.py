#!/usr/bin/env python3
"""Generate escalation candidates (T3 unicode, scaled up) and write cands JSONL."""
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
                           s + len(m['text']) + len(f'{TOOL_START}call:{m["tool"]}{{{joined}}}{TOOL_END}')))
    return fields

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--records', default='/data/records_gemma.jsonl')
    ap.add_argument('--folds', default='/data/folds.jsonl')
    ap.add_argument('--eval-scores', default='/data/eval_ep0.jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', default='')
    ap.add_argument('--grid', default='sweep')
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
        'sweep': [
            ('s1w1800r1',   1.0, 1800, 1, 0),
            ('s2w1800r1',   2.0, 1800, 1, 1),
            ('s4w1800r1',   4.0, 1800, 1, 2),
            ('s8w1800r1',   8.0, 1800, 1, 3),
            ('s16w1800r1', 16.0, 1800, 1, 4),
            ('s4w3600r1',   4.0, 3600, 1, 5),
            ('s8w3600r1',   8.0, 3600, 1, 6),
            ('s16w3600r1', 16.0, 3600, 1, 7),
            ('s4wallr1',    4.0,    0, 1, 8),
            ('s16wallr1',  16.0,    0, 1, 9),
            ('s8w1800r2',   8.0, 1800, 2, 10),
            ('s16w1800r2', 16.0, 1800, 2, 11),
        ],
        'push': [
            ('s24w1800r2', 24.0, 1800, 2, 20),
            ('s32w1800r1', 32.0, 1800, 1, 21),
            ('s48wallr1', 48.0,    0, 1, 22),
            ('s32wallr1', 32.0,    0, 1, 23),
            ('s64wallr1', 64.0,    0, 1, 24),
        ],
    }
    cfgs = grids[args.grid]
    ids = [x.strip() for x in args.ids.split(',') if x.strip()]
    out = open(args.out, 'w')
    n = 0
    for tid in ids:
        if tid not in targets:
            print('MISSING', tid); continue
        r, ps = targets[tid]
        ct = r['chat_text']
        try:
            msgs = parse(ct)
        except ValueError as e:
            print('PARSE FAIL', tid, e); continue
        if rebuild(msgs) != ct:
            print('REBUILD FAIL', tid); continue
        fields = build_spans(msgs)
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
            # rebuild with substitutions
            for fi, (mi, role, key, s, e) in enumerate(fields):
                nv = new_msgs[mi].pop('__new__' + str(fi), None)
                if nv is None:
                    continue
                if role == 'user':
                    new_msgs[mi]['text'] = nv
                else:
                    new_msgs[mi]['text'] = nv
            new_ct = rebuild(new_msgs)
            out.write(json.dumps({'id': tid, 'cfg': name, 'scale': scale, 'window': window,
                                  'rounds': rounds, 'seed': seed, 'chat_text_prime': new_ct}) + '\n')
            n += 1
    out.close()
    print('wrote', n, 'candidates')

if __name__ == '__main__':
    main()