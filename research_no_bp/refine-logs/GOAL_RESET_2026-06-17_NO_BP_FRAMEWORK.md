# Goal Reset: Pure No-BP Biomimetic Model Framework

**Date**: 2026-06-17

## Active Goal

Build a new **pure no-BP biomimetic neural model framework** from random/local initialization. The core method must learn through local plasticity, eligibility traces, fixed/random feedback, dendritic/apical error signals, inhibitory microcircuits, phase/oscillatory binding, recurrent/SSM dynamics, and low-precision/no-raw-data state.

The goal is no longer "attach memory to a pretrained LLM" or "win TinyStories with statistical token caches." The next phase must prove that a neural no-BP mechanism can train on controlled QA, compositional reasoning, human-learning-style interference tasks, and token modeling without relying on BP-pretrained backbones or n-gram/Kneser-Ney-style statistics as the final method.

## Hard Anti-Goals

- No BP-pretrained LLM, frozen backbone, or API as the method backbone.
- No n-gram, continuation, Kneser-Ney, or sparse token-count table as the final method.
- No raw-data replay as the main online-learning mechanism.
- No claim of GPT-like competence from TinyStories post-online cache metrics alone.
- API/LLM use is allowed only as an optional evaluator, upper bound, or later interface once the pure no-BP core passes local tests.

## Current Data State

- TinyStories is now under `data/TinyStories/`.
- bAbI QA loader metadata is locally present under `data/babi_qa/`; actual train/test samples still need to be generated or cached before offline training.
- GSM8k-Aug is locally present under `data/GSM8k-Aug/`, but it is not a first-stage training target because parsing and arithmetic language reasoning would dominate the mechanism question.
- CLUTRR is not currently present locally and is not a first-stage dependency.

## Primary Claims For The New Phase

| Claim | Minimum Convincing Evidence |
|---|---|
| C1: Pure no-BP neural mechanisms can learn QA/reasoning beyond local token-cache statistics. | bAbI QA answer classification beats no-memory, reservoir, and pair lookup on held-out templates/compositions, with statistical baselines clearly labeled diagnostic; CLUTRR-style relation classification remains a later optional extension after data download. |
| C2: The useful local update can be analyzed as a biologically plausible approximation to task descent without becoming BP. | Center-difference/BP-neighbor diagnostics show when no-BP updates align with loss-reducing directions; training itself remains local/no-BP. |
| C3: The mechanism explains human-learning phenomena. | Ebbinghaus-style forgetting, WASD->WDAS interference, habituation to repeated input, and simple visual comparison tasks are reproduced by explicit trace/decay/inhibition/gating states. |

## Workstreams

1. **Data and loader reset**
   - Update scripts to support `data/TinyStories/TinyStories-train.txt` and `data/TinyStories/TinyStories-valid.txt`.
   - Add local bAbI loader/export path using `data/babi_qa/`; if only metadata is present, export processed JSONL after the dataset is downloaded/cached.
   - Keep GSM8k and CLUTRR as later stress tests, not first-stage proof.

2. **QA training/evaluation**
   - First target: bAbI `qa1`, then `qa2`, `qa3`, `qa15`, `qa16`.
   - Task form: context/question -> answer id classification.
   - Training: read context and use answer as local target/wrong-winner signal.
   - Evaluation: no test-answer update; report exact-match accuracy, CE, per-task and per-hop metrics.
   - Baselines: no-memory majority, raw lexical retrieval, exact symbolic/statistical lookup, reservoir/e-prop, phase/dendritic/apical no-BP learner.

3. **Gradient-neighbor diagnostic**
   - On tiny networks and toy batches only, compare no-BP update direction with center-difference loss directions and optional BP oracle.
   - Metrics: cosine similarity, sign agreement, one-step loss change.
   - Diagnostic only; not a training algorithm.

4. **Biomimetic architecture evolution**
   - Move from RNN-like trace state toward LSTM/Transformer-like functions without BP:
     - local gates for write/forget/read;
     - dendritic branches as feature subspaces;
     - Hebbian key-value binding as attention-like memory;
     - inhibitory anti-attractor/reset dynamics for loop escape.
   - Use interpretability-style probes to inspect whether branches learn stable features; pretrained SAE/LLM features are not part of the method.

5. **Human-learning benchmark suite**
   - Ebbinghaus forgetting curve: retention after delay/interference.
   - WASD->WDAS remap: proactive interference, forgetting, and relearning speed.
   - Habituation: repeated identical input should reduce plastic update magnitude.
   - Visual comparison: two circles, output larger/smaller, with noise/position controls.

6. **TinyStories integration**
   - Continue using TinyStories only after QA and human-learning tasks validate mechanisms.
   - Statistical token methods remain baselines/diagnostics.
   - Main token metric should emphasize prequential before-update performance, held-out story generalization, generation loop rate, and no-raw-data state.

## First Runs

| Run | Purpose | Decision Gate |
|---|---|---|
| R080 | Data path reset smoke for TinyStories and bAbI | Loaders run locally and write parsed counts/examples. |
| R081 | bAbI QA1 no-BP smoke | Pure no-BP phase/dendritic learner beats no-memory and does not rely on raw retrieval. |
| R082 | bAbI QA2/QA3 multi-hop smoke | If pure no-BP collapses to lookup behavior, redesign state/credit assignment before scaling. |
| R083 | Center-difference diagnostic on compositional cue and bAbI QA1 | Identify whether no-BP update directions are loss-reducing and when they diverge. |
| R084 | WASD->WDAS interference benchmark | Trace/decay/gating mechanism produces measurable interference and relearning dynamics. |

## Success Criteria For This Reset

- A pure no-BP learner trains on bAbI QA locally with no BP/BPTT/pretrained backbone.
- Statistical baselines are reported but never used as the final method claim.
- Center-difference diagnostics explain update behavior without becoming the optimizer.
- At least one human-learning benchmark exposes a mechanism-level effect that maps to trace, decay, inhibition, or gating.
- TinyStories work resumes only after the mechanism has passed QA or human-learning evidence.
