# GOAL RESET 2026-06-19: One Unified Brain-Inspired Model

## 🚨 STOP — READ BEFORE DOING ANYTHING

**If you are about to:**
- Call any external API (DeepSeek, OpenAI, Claude, etc.) as a teacher or backbone → **STOP. This violates the core constraint.**
- Write any script that uses a pretrained LLM, frozen model, or API endpoint in the learning loop → **STOP.**
- Add a new task-specific module (event detector, credit channel, flip gate, QA head) → **STOP.**
- Continue the bAbI paraphrase/credit loop (R097–R145) → **STOP. It is permanently archived.**
- Read `online_memory_style_api_experiment.py` or any `*api*` script for implementation ideas → **STOP. Those are old archived experiments.**

**The only allowed next action is R096: integrate NoProp + Hebbian KV + eligibility into one unified model, train on TinyStories next-token prediction, no API, no pretrained model.**

Current best result to beat: `phase_trace_noprop_local_inhib_competitive_online_feature_calib_gain` seed0 post CE **2.201** (seeds 1/2 may still be running in background — check before starting new experiments).

---

Build a **single unified no-BP neural model** that pretrains on raw text from scratch, learns through local biologically-plausible plasticity rules, and handles both language generation and reasoning tasks with the same weights — no task-specific modules, no pretrained backbones, no patchwork.

---

## WHAT THIS MEANS

### "Unified"
One model file. One set of weights. One pretraining run.
- NOT: separate scripts for generation vs QA
- NOT: task-specific detectors, credit channels, or flip gates bolted on
- YES: the same pretrained encoder feeds a generation head OR a QA head by swapping only the output layer

### "Similar to GPT"
GPT is pretrained on next-token prediction, then fine-tuned for tasks.
This model does the same, but with no backpropagation:
- Pretraining: next-token prediction on TinyStories/WikiText
- Task adaptation: swap output head, run a few local WTA update steps
- No frozen backbone. No API. Trained from random initialization.

### "Brain-inspired (类脑)"
Every learning rule maps to a known neuroscience mechanism:

| Module | Biological analogue | Algorithm |
|---|---|---|
| Token encoding | Place cells / grid codes | VSA phase codes (Kanerva 1988) |
| Layer 1 local learning | Hebbian LTP/LTD | Target-only phase binding |
| Layer 2 associative memory | Hippocampal CA3 | Hebbian KV matrix (Hopfield 1982 / Ramsauer 2020) |
| Layer 3 deep local error | Apical/basal dendrite compartments | DLL branch-local loss (Lv et al. ICML 2025) |
| Temporal credit | Eligibility traces | e-prop (Bellec et al. NeurIPS 2020) |
| Output competition | Cortical WTA inhibition | Margin perceptron + lateral inhibition |

No global error signal. No weight transport. No forward-locking. Every synapse updates using only locally available information.

### "Efficient"
- State < 10MB after pretraining
- Online learning: updates one token at a time, no replay of raw data
- Inference: single forward pass, no iterative inference steps
- Hardware-friendly: integer quantization ready (proven in R056/R057)

### "No existing pretrained models"
- No Llama, GPT, BERT, or any BP-trained model in the architecture
- No frozen embeddings from BP models
- No API calls as part of the method
- Random initialization → local plasticity → learned representations

---

## THE ARCHITECTURE

Three layers + associative memory. Clean. No special cases.

```
Input: raw token IDs
         │
         ▼
┌────────────────────────────────────────┐
│ LAYER 1 — Phase Binding                │
│ • VSA complex phase codes (fixed rand) │
│ • order=2 context binding              │
│ • Local target-only Hebbian update     │
│ • WTA competitive readout              │
│ Biological: fast cortical columns      │
└────────────────────────────────────────┘
         │ prediction error (apical signal)
         ▼
┌────────────────────────────────────────┐
│ LAYER 2 — Hebbian KV Memory            │
│ • M += lr * outer(key, value)          │
│ • M *= (1 - decay) per step            │
│ • Covers FULL sequence history         │
│ • No fixed context window              │
│ Biological: hippocampal associative    │
│ memory, pattern completion             │
└────────────────────────────────────────┘
         │ prediction error (apical signal)
         ▼
┌────────────────────────────────────────┐
│ LAYER 3 — Deep Local (DLL / NoProp)    │
│ • Branch-local loss per layer:         │
│   loss_l = ||h_l - B_l @ label||^2    │
│ • B_l = fixed random (no weight        │
│   transport, feedback alignment)       │
│ • Slow time-scale, semantic features   │
│ Biological: deep cortical layers,      │
│   top-down predictive coding           │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│ OUTPUT — Next Token Prediction         │
│ • Always predicts over full vocab      │
│ • WTA over token prototypes            │
│ • Generation: greedy/sample decode     │
│ • QA: prompt as context, decode answer │
│   e.g. "Where is Mary? Answer: garden"│
│ • ONE output layer, never swapped      │
└────────────────────────────────────────┘
```

**There is no separate QA head. There is no head-swapping.**
The model always predicts the next token over the full vocabulary.
QA is handled by formatting: "Context + Question + 'Answer:'" → model predicts the answer token.
This is exactly how GPT handles QA. The same local WTA update rule applies to both generation and QA training.

---

## WHAT IS PERMANENTLY DELETED

The following approaches are closed. Do not revisit:

- **bAbI task-specific scripts** as model code: `babi_no_bp_qa_experiment.py`, `babi_delayed_credit_experiment.py`, `babi_attribute_*`, `babi_relation_*`
  → Keep only as evaluation harnesses. Model weights come from unified pretraining.

- **All credit/gating modules** (R100–R133): flip gates, risk prototypes, near-risk channels, counterfactual arbiters, compatibility channels
  → These were patches on a 2-layer model. A deeper model does not need them.

- **All task-specific front-end detectors**: LearnedEventDetector, LearnedQueryDetector, LearnedAttributeStatementDetector, regex parsers
  → Layer 3 learns these representations from text statistics during pretraining.

- **Separate offline token learner**: `phase_binding_token_experiment.py`
  → Replaced by unified online model.

- **bAbI paraphrase/credit loop** (R097–R145): 49 runs of patchwork on a 2-layer architecture
  → Archived. The surface paraphrase problem disappears when depth is sufficient.

---

## CURRENT STATUS AND NEXT STEPS

| Run | Status | Result | Decision |
|---|---|---|---|
| R092 DLL depth | DONE | CE 2.281, TRADEOFF | Did not pass gate 2.253 |
| R093 NoProp depth | IN PROGRESS | seed1: 2.294, seed2: 2.271 | Likely TRADEOFF too |
| R094 Hebbian KV | DONE | CE -0.012, partial-positive | Needs confidence gate |
| R095 e-prop | DONE | acc +0.015, CE +0.077, TRADEOFF | Use as acc branch only |
| R096 Integration | TODO | — | Run after R093 completes |

**If R093 also does not pass CE 2.253:**
Do NOT keep tuning individual components. Proceed directly to R096 integration.
The combined model (depth + KV + eligibility) may achieve what no single component achieves alone. Individual component CE gates are soft constraints, not hard blockers for integration.

**R096 unified integration target:**
- Single model: Layer 1 (phase/WTA) + Layer 2 (Hebbian KV) + Layer 3 (best of DLL/NoProp) + eligibility trace
- Pretrain on TinyStories 50k/10k, vocab=256, seeds=0/1/2
- Success: post CE < 2.20 AND greedy repeat-2 < 0.25

**After R096 passes:**
- Run bAbI QA1/2/3 using ONLY the pretrained unified model + a local WTA output head
- No task-specific modules added
- Success: QA1 acc > 0.80, QA2 acc > 0.60 without any hand-written parser

---

## HARD CONSTRAINTS

1. No BP, BPTT, pretrained LLM, frozen backbone, or external API in the learning rule
2. No statistical n-gram / continuation tables in the final method
3. No raw data replay
4. No task-specific special-case modules in the final architecture
5. All weight updates use only local activations + local target/error signal
6. Final model must support online learning: update on each new token, discard raw input
