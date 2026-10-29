# Findings: attacking a stop/continue safety-classifier surrogate

This document is the canonical record of the stopflip research campaign:
what we did, what we found, and what we believe (and at what confidence).

## 1. The setup

- **Surrogate.** Qwen3.5-4B (hybrid attention), QLoRA r=16/α=32 over 10
  target-module families (q/k/v/o + in_proj_qkv/in_proj_z/out_proj +
  gate/up/down), loss on a single label token (" STOP" vs " ok") at the
  end of the template's generation prompt. 4-bit NF4 base, fp16 compute.
  Trained on 5,749 records (group-folded; fold 0 held out), 1 epoch
  (~9h on one RTX 3090). This checkpoint is **E0**.
- **Data.** 7,319 context-label pairs distilled from the real classifier:
  full agentic cyber-security conversation transcripts (CyberGym-E2E,
  CVE-Bench, DeepSWE) with the real classifier's per-position STOP/ok
  decision attached. Contexts are 5–40K tokens; label distribution is
  dominated by STOP at drift positions and ok on benign/dense positions.
- **Real classifier.** Black-box API, verbatim labeling protocol:
  max_tokens=1, refusal stop_reason iff STOP, otherwise ok. Live probes:
  k=3 baseline + k=3 perturbed, ~6 API calls per record.
- **Attack goal.** Flip a STOP decision into ok — i.e., get the real
  classifier to *not* stop — using only edits that preserve meaning
  (what an human/LLM reader sees unchanged) and that are expressible in
  message space (text blocks and tool_use input string fields survive
  into the live API; template tricks, token tricks, and argument
  reordering do not).

## 2. Campaign timeline

### WF-1: 4-branch attack campaign

Four parallel attack branches (saliency-guided, escalation-search,
oracle-in-loop compositional search, whole-conversation families) over
the 211 flip candidates (fold-0, label STOP, 0.5 ≤ p_stop < 1.0).

- **22 surrogate flips** verified (p_pert < 0.5), drops up to -0.34.
- All winning edits were pure cosmetic unicode noise: Cyrillic/Greek
  homoglyphs, zero-width chars, combining marks, NBSP, case flips.
- **Live: 1 flip out of 15 probed.** cybergym-22-178: 3/3 refusal →
  3/3 non-refusal. Cumulative 14/14 non-refusal at k=11 across
  campaigns — rock stable, not a coin flip.
- Non-unicode families (description paraphrase, shell-equivalent
  rewrites, argument reordering) barely moved p_stop. Reordering is
  structurally non-transferable (the chat template sorts tool args).

### WF-2: retraining campaign

Hypothesis: a better surrogate would transfer better.

- Continued training (config A: init-from E0, 1 epoch, r=16) improved
  ranking (fold-0 AUC 0.888→0.906) but **destroyed calibration**
  (ECE 0.443→0.685, acc@0.5 0.475→0.205) and pushed the 211 attack
  targets *further* from the boundary (mean p 0.874→0.964).
- A's 22 fresh flips (including margins 100× deeper than E0's) went
  **0/14 live**. A also kills E0's only true positive: it scores
  22-178-perturbed at 0.85.
- Literature agrees: early-stopped surrogates transfer better
  (arXiv:2007.00806); ensemble attack scale transfer log-linearly with
  ensemble size (Liu et al. 2024). More training was the wrong lever.

### WF-3: forensics + data campaign

- **Ensemble attack** across 5 checkpoints (E0 + A-family): 69
  ensemble-flip candidates over 26 targets; 20 flip all 5 members.
  Live: **0/15 transferred**. Ensemble certification did not help.
- **Forensics on the one true transfer (cybergym-22-178).** The
  perturbation: 30 whole-field unicode edits on the last 8 model turns,
  corrupting all four fields (`command`, `description`, `persistent`,
  `timeout`) of a homogeneous tail of pure `Monitor` idle-wait tool
  calls. Live ablations (58 calls, k=5 + 8 byte-exact ablations):
  - No single field family flips. ~38 invisible chars in
    `description`+`persistent`+`timeout` of the **last 4** calls flips
    3/3; the identical 38 chars 8 messages earlier does not. **Recency
    is decisive.**
  - `description` is load-bearing; `command` is not.
  - E0 scores the flipping variant at 0.554 — **the surrogate cannot
    see the transfer mechanism.**
- **Data pilot.** 384 new-domain contexts labeled with the real
  classifier (agentic-redteam gold, AFTraj-2K, ATBench). The real
  classifier shows clean boundaries on agentic-redteam gold (29/90 STOP
  at both drift positions, 0/40 benign). E0 does not transfer:
  AUC 0.365 (anti-correlated) on agentic-redteam. **E0 is a
  cybergym-only classifier.** A fresh retrain (B) *trained on* the 384
  pilot records still cannot rank them (AUC ~0.49) — 384 OOD labels
  don't teach an OOD boundary; per-domain labeling at scale is needed.

## 3. What we believe

1. **Deep surrogate margins are worthless as a transfer signal.** Across
   45+ live probes, surrogate flip depth (p_pert to 0.009, ensemble-certified)
   had zero correlation with live transfer. The attack objective
   ("minimize surrogate p_stop") is not the transfer objective.
2. **What transfers is structural, not margin-deep.** The single
   confirmed transfer is small (155 chars), tail-localized (last 4–8
   turns), multi-field-per-tool-call, and lands on
   semantically-loaded fields (descriptions) of a homogeneous tool-call
   tail. n=1 — treat as a hypothesis, but it is the only mechanism
   we've seen.
3. **The surrogate is over-confident and badly calibrated; calibration
   was free performance.** Platt/isotonic on a 1,600-record fitset
   (folds 1–4, eval on fold 0): acc@0.5 0.594→0.882, ECE 0.373→0.092.
   66 of the 211 attack targets fall below 0.5 after calibration —
   the raw target set was selected against a broken threshold.
4. **Capacity/optimization was never the bottleneck.** More epochs,
   higher rank, ensemble certification, and OOD augmentation all failed
   to improve (and mostly hurt) transfer. Data diversity at scale is
   the remaining lever.

## 4. Open questions

- Does the 22-178 recipe (tail-localized, multi-field,
  description-centric, homogeneous-tail) replicate on other records?
  We have 210 more targets and the machinery to test it.
- Would a surrogate trained on agentic-redteam gold at scale (not 384,
  but thousands of labels) close the OOD gap? The 32% STOP rate at
  drift positions makes it the cheapest labeling target.
- Is the recency effect in the real classifier positional (last-N
  tokens dominate) or structural (Monitor-idle-loop context)?

## 5. Cost accounting (live API)

- WF-1: ~90 live probe calls + relabeling campaigns (thousands)
- WF-2: ~90 live probe calls
- WF-3: ~97 live probes + 58 forensics + 384 pilot labels
- Campaigns were each budgeted ≤100 probe calls plus one bounded pilot
  (≤400 labels). No live call is required to reproduce surrogate-side
  results; only transfer claims (and the n=1 caveat) need the API.
