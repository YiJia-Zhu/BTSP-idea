# R090 Deep Bionic Architecture Direction

**Date**: 2026-06-18

## Motivation

R069 (current best token learner) and R089 (role-binding QA) both hit architectural ceilings that are not fixable by tuning within the current design:

1. **Context window is a fixed small integer.** `feature = code[t-2] ⊗ code[t-1]` covers at most order=3 tokens. R039 proved naive order>3 collapses (order=3 CE 4.199, order=4 CE 10.46). The architecture has no mechanism to read evidence from earlier in the sequence.

2. **Network depth is 2 layers (Binding → WTA Readout).** There is no layer-wise abstraction. Every token prediction uses the same representational level. Transformers use 12–96 layers. Even a 2→3 layer transition brings qualitative gains in standard deep learning.

3. **R089 role binding still uses task-specific regex at the input boundary.** `parse_movement()` is hand-coded. A truly general architecture must learn event detection from raw token patterns, not from hand-written parsers.

These are not bugs. They are the natural limit of a single-layer phase/WTA system. The fix requires architecture changes, not hyperparameter tuning.

## Target Architecture: Stacked Predictive Columns

The new target is a **3–4 layer columnar no-BP architecture** where each layer:

- Maintains its own phase code bank and WTA prototype dictionary (columnar specialization)
- Receives bottom-up input from the layer below (forward pass, as now)
- Generates a top-down prediction of the layer below's state
- Propagates only **prediction error** upward (not raw activations)
- Updates locally using the same target-only / apical-error rule already proven in R051–R069

This is directly inspired by cortical microcircuit models (Sacramento et al. 2018, Rao & Ballard 1999) and does not require BP: the top-down prediction acts as the "desired code" for the layer below, exactly matching the `desired_code = target_anchor ⊗ conjugate(other_bound)` logic already in `update_context()`.

### Layer roles (3-layer version)

```
Layer 3 (slowest, most abstract):
  - Input: prediction error from Layer 2
  - Time scale: long trace decay (τ ≈ 32–64 tokens)
  - Role: captures story-level or sentence-level patterns

Layer 2 (medium):
  - Input: prediction error from Layer 1
  - Top-down: predicts Layer 1 state via learned projection
  - Time scale: medium trace decay (τ ≈ 8–16 tokens)
  - Role: captures clause-level or phrase-level patterns

Layer 1 (fastest, most local):
  - Input: raw token phase codes (as now)
  - Top-down: receives prediction from Layer 2; prediction error = actual − prediction
  - Time scale: short window (order=2, as now)
  - Role: bigram/trigram-level binding
```

All layers update locally: no cross-layer gradient chain. The top-down prediction is a fixed random projection (feedback alignment style, as proven in R052) or a locally learned linear map updated by Hebbian rule.

## Target Architecture: Hebbian Key-Value Attention

The second missing component is **arbitrary-length context coverage**. The fix is a Hebbian associative memory that functions like soft attention but uses only local updates:

```
# Write (online, one step per token):
key[t]   = normalize(phase_feature(context_window[t]))  # same encoding as now
value[t] = target_anchor[token[t]]                      # target embedding
M += lr_write * outer(key[t], value[t])                 # rank-1 Hebbian write

# Read (at query time):
query   = normalize(phase_feature(current_context))
scores  = M @ query                                      # all past keys vote
output  = normalize(scores)                              # soft retrieval
```

This is a continuous-time Hopfield network / Kanerva sparse distributed memory. The memory matrix M accumulates all past key-value pairs with exponential decay (old writes fade). It covers the **entire observed sequence** without a fixed window, and requires only matrix-vector multiply at read time.

Key properties:
- No BP. Write rule is pure Hebbian (outer product).
- Covers arbitrary context length. Earlier tokens contribute via their stored key-value associations.
- Composable with existing WTA readout: `M @ query` produces a feature vector that feeds into the existing competitive readout.
- Forgetting is natural: apply per-step decay `M *= (1 - decay)` to implement recency bias.

Reference: Modern Hopfield Networks (Ramsauer et al. 2020), Kanerva 1988 SDM.

## Target Architecture: Learned Event Detector

Replace `parse_movement()` regex in `babi_no_bp_qa_experiment.py` with a **local Hebbian event detector**:

```
# For each token t, compute a soft "is this an action sentence?" score:
event_score[t] = sigmoid(detector_weights @ phase_feature(context_window[t]))

# Update detector_weights locally:
# If this context window is followed by a role-binding event, positive update.
# If not, negative update. No BP: use the WTA readout error as the gate.
delta = target_error * (1 - event_score[t]) * phase_feature(context_window[t])
detector_weights += lr_det * delta
```

This makes the event boundary **learned from token statistics**, not hardcoded. The first version can still use the symbolic parser to provide the initial supervision signal, then gradually switch to the learned detector.

## Experiment Plan

### R090: 2-layer stacked predictive coding token learner

**Script**: `phase_binding_token_experiment.py` (new `--num-layers 2` flag)

**Mechanism**:
- Layer 1: existing phase/trace/apical WTA (R069 best config)
- Layer 2: receives Layer 1 WTA winner's prototype as input; maintains its own phase codes and WTA dict with longer trace decay (τ=32)
- Top-down: Layer 2 projects a predicted Layer 1 feature via fixed random matrix B2 (Lillicrap-style); Layer 1 uses `(actual_feature − B2 @ layer2_activation)` as apical error instead of raw prediction error
- All updates local; no cross-layer BP

**Success criterion**: medium post CE < R069 best (2.253) on ≥2 seeds  
**Failure criterion**: CE ≥ 2.30 on all seeds → stacking adds no benefit at this scale; investigate state dimensions or trace scale mismatch

**Decision gate**: if positive, extend to 3 layers (R091b); if negative, switch to Hebbian KV attention (R091a) instead of depth

---

### R091: Hebbian key-value attention branch (arbitrary-length context)

**Script**: `phase_binding_token_experiment.py` (new `HebbbianKVBranch` class)

**Mechanism**:
- Memory matrix M: shape `(feature_dim, feature_dim)`, initialized zeros, updated online
- Write: `M += lr_write * outer(normalize(key_feature), target_anchor[token])`; after write, `M *= (1 - decay)`; decay=0.002 (half-life ≈ 350 tokens)
- Read: `kv_feature = normalize(M @ query_feature)`; concatenate with existing branch features before WTA readout
- key_feature uses same phase binding as existing branches (order=2); the memory provides a separate channel covering full history

**Success criterion**: medium post CE < R069 (2.253) or held-out story generalization CE gap narrows  
**Failure criterion**: CE degrades or M becomes numerically unstable → reduce lr_write or apply SVD truncation

**Decision gate**: KV branch CE gain compared to R069 determines whether long-range memory or depth is the higher-priority bottleneck

---

### R092: e-prop eligibility trace for longer temporal credit

**Script**: `phase_binding_token_experiment.py` or new `eprop_token_experiment.py`

**Mechanism**:
- Eligibility trace: `e[t] = γ * e[t-1] + phase_feature[t]` (γ=0.95, trace window ≈ 20 tokens)
- At each target event, WTA readout update uses `e[t]` instead of current `phase_feature[t]`
- This propagates credit backward in time through the trace without BPTT
- Combine with R069 apical error: apical gate multiplies the eligibility trace magnitude

**Success criterion**: medium post CE < R069 on sequences with long-range dependencies (held-out long stories)  
**Decision gate**: if CE improves on long stories but not short, confirms temporal credit is the bottleneck

---

### R093: Learned event detector replacing regex in bAbI QA

**Script**: `babi_no_bp_qa_experiment.py` (new `LearnedEventDetectorQALearner` class)

**Mechanism**:
- Phase-encode each sentence in context (order=2 binding over tokens)
- Single-layer local classifier: `event_type_score = W_det @ phase_feature(sentence)`
- W_det updated by WTA rule: pull toward event-type prototype when role-binding update occurs, push away otherwise
- Output: soft event-type probabilities gate the role-binding write (`role_lr * event_score * outer(person_code, location_code)`)
- No regex, no hand-parsed patterns; supervision comes only from QA answer correctness (downstream target signal)

**Success criterion**: QA1 acc ≥ 0.80 and QA2 acc ≥ 0.70 with zero regex parsing  
**Failure criterion**: QA1 acc < 0.50 → event detector is not learning from token patterns; need better sentence encoding or more epochs

**Decision gate**: if learned detector is within 10% of regex-based R089, it becomes the new standard boundary; enables TinyStories-style generation from the same QA learner

---

### R094: Full deep no-BP token learner (R090 + R091 + R092)

Stack the three mechanisms on a single model after R090–R092 individually pass:

- Layer 1: phase/trace/apical WTA (R069 base)
- Layer 2: stacked predictive column (R090)
- KV attention branch: Hebbian full-history memory (R091)
- Eligibility trace: e-prop credit for long sequences (R092)
- All local, no BP, no raw data storage

**Target**: medium post CE < 2.20 (current best 2.253), greedy repeat-2 < 0.25

## What This Does Not Change

- No BP, BPTT, pretrained LLM, frozen backbone, or API in the learning rule.
- No statistical n-gram tables as the final method.
- No raw data replay.
- All updates remain local: each layer uses only its own activations and a target/error signal from the adjacent layer.

## Key Papers

**2025 (priority — replace older Sacramento 2018 route):**

- **Lv et al. (ICML 2025) — Dendritic Localized Learning: Toward Biologically Plausible Algorithm**
  [arXiv 2501.09976](https://arxiv.org/abs/2501.09976) | [PMLR](https://proceedings.mlr.press/v267/lv25c.html) | [code](https://github.com/Lvchangze/Dendritic-Localized-Learning)
  Core idea: each dendritic branch computes a **local loss** using only its own basal (bottom-up) and apical (top-down) activations. Weight updates are fully local — no cross-layer gradient chain, no weight transport problem. Validated on CIFAR-10 and beyond, closing the gap to BP more than any prior local-learning method. This is the direct engineering successor to Sacramento 2018 and should replace the vague "Sacramento-style top-down projection" described earlier in this document. R091 should implement the DLL branch-local loss instead of the fixed random B2 projection.

- **Li et al. (2025) — NoProp: Training Neural Networks without Back-propagation or Forward-propagation**
  [arXiv 2503.24322](https://arxiv.org/abs/2503.24322) | [code](https://github.com/Sid3503/NoProp)
  Core idea: inspired by denoising diffusion — each layer receives a **noisy version of the target embedding** and is trained independently with a local denoising loss. No forward pass across layers needed at train time; each layer is updated in parallel. This is the cleanest no-BP architecture for deep networks because layers are completely decoupled during training. For the token learner: layer l receives `z_l = sqrt(ᾱ_l) * target_embed + sqrt(1-ᾱ_l) * noise` and learns to predict `target_embed` from its local activations. Noise schedule `ᾱ_l` decreases with depth. R092 should implement this as the alternative to DLL for the stacked token learner.

**Foundations (still relevant):**

- Lillicrap et al. (2016) — Random synaptic feedback weights (feedback alignment) — motivates fixed random B in apical error
- Bellec et al. (2020) — e-prop — motivates R093 eligibility trace
- Ramsauer et al. (2020) — Modern Hopfield / Hebbian KV attention — motivates R092 KV branch
- Kanerva (1988) — Sparse Distributed Memory — motivates M decay and capacity analysis
