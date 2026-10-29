#!/usr/bin/env python3
"""WF-v3 data prep.

Builds, from /data/records_gemma.jsonl + /data/folds.jsonl + /data/wf_v3/pilot_{records,labels}.jsonl:
  - records_v3.jsonl  (7319 cybergym rows + 384 pilot rows, labels filled)
  - folds_v3.jsonl    (folds.jsonl + 182 pilot groups @ fold 5)
  - fitset.jsonl      (1600 records stratified by (fold, label), folds 1-4)
  - fold0_eval.jsonl  (1276 fold-0 records)
  - pilot_eval.jsonl  (384 pilot records)
"""
import json, random, collections

D = "/data"

# ---- pilot labels (384 canonical id -> STOP/ok from the live API) ----
labels = {}
ndup = 0
for l in open(f"{D}/wf_v3/pilot_labels.jsonl"):
    r = json.loads(l)
    if r.get("label") in ("STOP", "ok"):
        if r["id"] in labels:
            assert labels[r["id"]] == r["label"]
            ndup += 1
        labels[r["id"]] = r["label"]
assert len(labels) == 384, f"expected 384 pilot labels, got {len(labels)} (dups {ndup})"
print(f"pilot labels: {len(labels)} "
      f"({sum(1 for v in labels.values() if v == 'STOP')} STOP / "
      f"{sum(1 for v in labels.values() if v == 'ok')} ok)")

prec = []
for l in open(f"{D}/wf_v3/pilot_records.jsonl"):
    r = json.loads(l)
    assert r["id"] in labels, r["id"]
    r["label"] = labels[r["id"]]
    prec.append(r)
assert len(prec) == 384
assert len({r["id"] for r in prec}) == 384
print(f"pilot records: {len(prec)}")

# ---- cybergym records + folds ----
folds = [json.loads(l) for l in open(f"{D}/folds.jsonl")]
fold_of = {r["group"]: r["fold"] for r in folds}
assert len(fold_of) == len(folds)
recs = [json.loads(l) for l in open(f"{D}/records_gemma.jsonl")]
ids = [r["id"] for r in recs]
assert len(set(ids)) == len(ids), "duplicate ids in records_gemma"

pgroups = {r["group"] for r in prec}
assert not pgroups & set(fold_of), "pilot group collides with cybergym fold group"
pids = {r["id"] for r in prec}
assert not pids & set(ids), "pilot id collides with cybergym id"

# ---- records_v3 + folds_v3 ----
with open(f"{D}/wf_v3/records_v3.jsonl", "w") as f:
    for r in recs + prec:
        json.dump(r, f)
        f.write("\n")
with open(f"{D}/wf_v3/folds_v3.jsonl", "w") as f:
    for r in folds:
        json.dump(r, f)
