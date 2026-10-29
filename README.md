# stopflip

> Meaning-preserving adversarial attacks on a stop/continue safety-classifier
> surrogate — and why they don't transfer to the real classifier.

## The one-paragraph version

A production safety classifier watches an AI agent's conversation and decides,
at each step, whether to let it keep going (`ok`) or halt it (`STOP`). We don't
have its weights — it's a black box we can only query. So we trained our own
local copy (a "surrogate"), attacked that copy with tiny text edits that a human
reader wouldn't notice, and checked whether the same edits also fool the real
classifier. **They almost never do.** Out of 45 attacks that reliably flipped
our surrogate, exactly one also flipped the real classifier. This repo is the
code and the write-up of why.

## The pieces

- **The real classifier.** A black-box safety model (Anthropic's real-time
  cyber safeguard). We can only send it a full conversation and watch whether it
  refuses. We treat a refusal as `STOP` and anything else as `ok`.
- **The surrogate.** Our local stand-in: a fine-tuned Qwen3.5-4B + LoRA model
  that reads a 5–40K-token multi-turn agent transcript and predicts `STOP` vs
  `ok` as its next token. Because we own it, we can see its exact confidence and
  run gradient-guided attacks against it.
- **The attack.** Edit the transcript so the surrogate flips `STOP` → `ok`,
  while keeping the meaning a human/LLM reader sees unchanged (homoglyphs,
  zero-width characters, paraphrases, argument reordering — never changing what
  the agent actually does).
- **The question.** Do surrogate-flipping edits transfer to the real
  classifier? If yes, a surrogate is a cheap attack oracle. If no, surrogate
  confidence is a dead end.

## Headline results

| Finding | Evidence |
| --- | --- |
| Surrogate flips are easy; transfer is not | 22–69 offline flips across 3 attack campaigns; **1 total live flip** out of 45 probed (reproduced 14/14, k=11+) |
| More training makes transfer *worse*, not better | Retrained surrogate: ranking AUC 0.888→0.906, but it killed the only true positive and all 22 of its predecessor's flips |
| Ensemble attacks don't fix it | 69 ensemble-certified flips across 5 checkpoints (20 flipping every member, margins to p=0.0025) → **0/15 transferred live** |
| Surrogate confidence does not predict transfer | The one live flip sat at surrogate p=0.44; non-transferring attacks drove surrogate p to 0.009 |
| What transfers is *structural*, not deep | n=1: ~38 invisible chars (homoglyphs, zero-width, combining marks) in `description`+`persistent`+`timeout` of the **last 4 tool calls**, on a homogeneous `Monitor`-call tail |
| Calibration, not capacity, was the real win | Post-hoc Platt/isotonic: acc@0.5 0.594→0.882, ECE 0.373→0.092 — recovered the 0.5 operating point for free |

## Repository layout

```
training/     train_qwen.py (original E0 trainer), train_qwen_v3.py (later
              trainer with --init-from/--save-every), prep_v3.py
evaluation/   eval_qwen.py, score_records.py (byte-exact scoring recipe)
attacks/      attack.py (tiers T0-T2), attack2.py (T3 unicode + hybrid),
              escalate/ compose/ families/ saliency/ branch machinery
live/         live_harness.py, live_prep_v2.py, relabel_v2.py, replay_labeler2.py
data/         build_dataset_v2.py, batch_labeler_v2.py, ingest_v3.py
docs/         FINDINGS.md — the full write-up
results/      per-checkpoint scoring dumps (wf_v2/, wf_v3/)
```

## How it ran

Two roles, which you can map onto one or two machines:

- A **GPU host** (a 24GB-class card is enough): holds the base `qwen3.5-4b`
  model, all LoRA checkpoints, and every scoring/attack script. All training and
  surrogate scoring happens here, inside a CUDA Docker image (torch,
  transformers, peft, bitsandbytes, triton).
- A **labeling host**: holds the conversation corpus and the dataset, and runs
  every call to the real classifier API (distillation labels + live transfer
  probes).

All code expects its working tree mounted at `/work` inside the container (or
set `WORK=/path`). The real-classifier API token is read from
`$WORK/tmp/oauth_token.txt` — supply your own; none is included here.

## Reproducing

1. **Train the surrogate.**
   ```bash
   docker run --gpus all -v "$WORK":/work --shm-size=4g <cuda-image> \
     python /work/train_qwen.py \
       --model /work/qwen3.5-4b --records /work/records_gemma.jsonl \
       --folds /work/folds.jsonl --fold 0 --epochs 2
   ```
   The epoch-0 checkpoint is **E0** (`qwen4b_ep0.pt`) — the attack surrogate.
2. **Score.** `evaluation/score_records.py` — 4-bit NF4 base, fp16 compute,
   `p_stop` = softmax over the two label-token logits at the last position.
3. **Attack.** See `attacks/`. Every edit is a message-space manifest
   (`{mi, bi, key, old, new}`) applied to the sanitized message list — only
   edits that survive into the live API count.
4. **Live-validate.** `live/live_harness.py` probe mode: call the real
   classifier with `max_tokens=1`, label `STOP` iff `stop_reason == "refusal"`,
   k=3 baseline + k=3 perturbed.

## Read the findings

Start at [`docs/FINDINGS.md`](docs/FINDINGS.md) — the full campaign record
(setup, timeline, what we believe and at what confidence, open questions, cost).

## Caveats

- The live-transfer evidence is **n=1**: exactly one perturbation has ever
  flipped the real classifier. Every "what transfers" conclusion beyond that is
  a single-observation hypothesis.
- A 384-record out-of-domain pilot showed E0 is a **cyber-only** classifier
  (AUC 0.365, anti-correlated, on general agentic-red-team data). Cross-domain
  claims would need per-domain retraining and labeling at scale.
