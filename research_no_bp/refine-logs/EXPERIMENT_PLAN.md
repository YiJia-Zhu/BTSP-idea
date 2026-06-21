# Experiment Plan

## 🚨 STOP — READ BEFORE DOING ANYTHING

**If you are about to call any external API (DeepSeek, OpenAI, Claude) as teacher/backbone, write any `*api*` script, add a task-specific module, or continue bAbI credit/paraphrase work → STOP. All of that is forbidden.**

**The only allowed next action: R096 unified model integration on TinyStories, no API, no pretrained model.**

Check if R096prep seeds 1/2 are still running before starting anything new.

---

## ⚠️ ACTIVE GOAL RESET: 2026-06-19

### The One Goal: A Unified No-BP Neural Model

**We are building ONE model, not two separate systems.**

The current codebase has two completely separate systems:
- `phase_binding_online_stream_experiment.py` — generation (TinyStories)
- `babi_no_bp_qa_experiment.py` + dozens of task-specific scripts — QA/reasoning

**This is wrong. The end state is a single unified model** that:
1. Takes raw token sequences (no hand-written parsers, no task-specific front-ends)
2. Has 3+ layers with local no-BP updates (DLL or NoProp)
3. Has Hebbian KV associative memory for full-sequence context
4. **Pretrains on text** (next-token prediction, TinyStories/WikiText) from random init
5. **Always outputs next-token probabilities over the full vocabulary — nothing else**
6. QA is handled by prompt formatting: feed "context + question + Answer:" and decode the answer token — exactly like GPT. No separate QA head. No head-swapping. No task-specific modules.

This is the no-BP equivalent of GPT: one architecture, pretrained on text, tasks handled by prompt format.

**Architecture (3 layers, one output):**
```
raw tokens → [L1: phase binding] → [L2: Hebbian KV memory] → [L3: DLL/NoProp] → next-token WTA
                                                                                         ↑
                                                              same output for generation AND QA
```

**bAbI paraphrase/credit loop (R097–R145) is PERMANENTLY ARCHIVED.**
These were patches on a 2-layer architecture. Not part of the unified model.
All task-specific bAbI modules (event detectors, query detectors, credit channels, flip gates) are archived — not ported.

**Next runs in strict order:**
1. **R093** — NoProp decoupled layer training (R092 DLL was TRADEOFF, CE 2.281, did not pass gate 2.253)
2. **R094** — Hebbian KV with confidence gate (R134 partial-positive CE -0.012, needs gate before stacking)
3. **R096** — Integrate depth + KV + eligibility into one unified model file
4. **Unified QA eval** — pretrained unified model + local WTA head on bAbI, zero task-specific modules

Full rationale and architecture diagram: `refine-logs/GOAL_RESET_2026-06-19_DEEP_ARCH_PIVOT.md`

---


## Previous Goal Reset: 2026-06-17

The active research goal is now:

> Build a new **pure no-BP biomimetic neural model framework** from random/local initialization, using local plasticity, eligibility traces, fixed/random feedback, dendritic/apical error, inhibitory microcircuits, phase/oscillatory binding, recurrent/SSM dynamics, and low-precision/no-raw-data state.

This supersedes the older GPT/API-oriented route. Pretrained LLMs, frozen backbones, and APIs are no longer allowed as the method backbone. Statistical token-count methods such as sparse context memory, continuation backoff, n-gram/Kneser-Ney-style caches, and similar tables are diagnostic baselines only.

New first-stage evidence must come from:

- bAbI QA trained with pure no-BP answer selectors; CLUTRR-style relation reasoning is optional later because CLUTRR is not currently downloaded locally;
- center-difference/BP-neighbor diagnostics used only to analyze local update quality;
- human-learning benchmarks such as Ebbinghaus forgetting, WASD->WDAS interference, habituation to repeated input, and simple visual comparison;
- TinyStories token modeling only after mechanism validation, with statistical baselines kept out of the main method claim.

Detailed reset document: `refine-logs/GOAL_RESET_2026-06-17_NO_BP_FRAMEWORK.md`.

**Problem**: 构建替代 BP 的高效仿生物神经学习算法，最终支持无原始数据保存的在线学习，并能面向纯仿生结构、在线学习硬件或 neuromorphic 实现。
**Hard Constraint**: 主方法必须是**纯 no-BP 神经网络类方法**：从随机/局部初始化开始训练，使用局部可塑性、资格迹、固定反馈、树突误差信号、抑制性回路、STDP/BTSP、e-prop、replay 或 reservoir/SSM 动态；不得依赖已经由 BP 训练好的开源 LLM、冻结大模型或 API 主干作为核心性能来源。预训练 LLM/API 只能作为外部上界、后续应用接口或自然语言评测器，不得作为论文主线。
**Method Thesis**: 当前路线需要从“冻结 LLM + 外挂记忆”和“统计 n-gram memory”转向“纯 no-BP 可训练神经结构”。优先级应回到 DEN1810/dendritic-like 树突双室误差、STDP/BTSP 调制版本、e-prop/三因子 eligibility、feedback alignment、抑制性微回路、recurrent/SSM/reservoir 动态。`continuation_backoff` / n-gram-style sparse memory 只能作为调试工具、统计下界或 sanity baseline，不能作为最终方法或核心 claim；TinyStories/API memory 结果只能作为应用侧参考。
**Date**: 2026-06-17

## 当前结果汇总

| 阶段 | 任务 | 最强 no-BP 结果 | 关键对照 | 判断 |
|---|---|---:|---:|---|
| 静态分类 | MNIST 14x14, hidden=128, seeds=3 | `dfa_3factor` acc 0.9042 | BP acc 0.9128 | 支持三因子固定反馈可接近 BP，但任务太简单 |
| 时序 toy | delayed cue next-token, delay=12 | `eprop_3factor` target acc 1.000 | tuned BPTT target acc 1.000; reservoir 0.000 | 支持 eligibility trace + 固定反馈能训练隐藏记忆 |
| 真实文本 token | TinyStories tokenizer medium | `sparse_hebbian_context` CE 4.1126, acc 0.361, 92k tok/s | dendritic CE 4.5777 acc 0.118; torch Llama 80 updates CE 4.8748 acc 0.117 | 当前最强正信号 |
| hybrid backoff | TinyStories tokenizer medium | sparse memory 单独 CE 4.1126 | `hybrid_llama_context` CE 4.5369; `hybrid_dendritic_context` CE 4.5630 | always-on 线性 logits fusion 是负结果 |
| context order ablation | TinyStories tokenizer medium | `sparse_hebbian_context` order=3 CE 4.0796 | order=4 CE 4.1126; order=5 acc 0.370 | order=3 是新的 CE 默认点 |
| semantic-key memory | TinyStories tokenizer medium | combined_context CE 4.0368 | semantic-only CE 5.36；exact-only CE 4.0796 | semantic bucket 只带来小幅增益 |
| compositional cue task | held-out `(a,b)` pairs, target `(a+b) mod K` | `target_only_phase_binding_hebbian` held-out acc 1.000 for K=4/8/12 | pair lookup 0.417/0.146/0.093; scrambled phase control 0.500/0.188/0.213; eprop/reservoir/BPTT fail on K=4/8 | 更强的纯 no-BP 正信号：不再使用 cue->phase 直接教师，只用最终 target 三因子信号学习局部相位码 + 复数绑定 + Hebbian 原型 |
| bAbI QA answer selection | QA1/QA2/QA3, 900 train / 1000 test | `role_binding_state_no_bp` solves QA1/QA2/QA3 at acc 1.000, CE 0.002-0.003, 65,745 bytes, no raw text; earlier QA1 `phase_dendritic_no_bp` acc 0.822 | R082 phase-only boundary: QA2 phase acc 0.179 vs majority 0.187; QA3 phase acc 0.158 vs majority 0.185; hashed lookup at majority | 新的结构化 no-BP state-binding 原型修复了多跳边界：person-location、object-owner、object-location、before-location 都是局部关联矩阵；但输入事件解析仍是任务特定，下一步应学习 event detector 并接回神经读出/生成 |
| learned event + role binding | bAbI QA2/QA3 medium | `role_binding_state_no_bp` with learned local event front-end keeps QA2/QA3 acc 1.000, CE 0.002/0.003; event/person/object/location test acc 1.000; detector state 5,104 bytes | R089 regex-front-end role binding also acc 1.000; phase-only and hashed lookup stay near majority | R090 removes regex event front-end for medium bAbI via fixed random token/position features + no-BP perceptron/prototype updates. 仍有边界：event labels 来自句子局部结构，question parser 仍显式，下一步要做 query parser/弱监督 credit |
| learned event + learned query + role binding | bAbI QA1/QA2/QA3 medium | learned local event/query front-ends plus role-binding state solve QA1/QA2/QA3 at acc 1.000, CE 0.002-0.003, 65,843 bytes; event/query/slot test acc 1.000 | phase-only and hashed lookup remain weak; R090 still used explicit question parser | R091 removes explicit query parser from the main bAbI path. 当前剩余主要边界是 event/query 标签仍来自局部结构监督、语法窄，下一步应做 paraphrase/noise stress 或 delayed QA-level credit |
| attribute/category binding QA | bAbI QA15/QA16, 900 train / 1000 test, seeds=0/1/2 | R137 regex-front-end `attribute_binding_state_no_bp` reaches QA15 acc/CE 1.000/0.0036 and QA16 0.995/0.0417 with ~65.7KB state; R138 learned local statement/query front-end preserves QA15 1.000/0.0036 and QA16 0.995/0.0417 with ~72-73KB state | QA15 phase 0.326, raw retrieval 0.213; QA16 phase 0.460, raw retrieval 0.438; majority 0.213/0.236 | Positive expansion beyond location/object movement: local associative state covers category relation deduction and recency-style attribute induction, and R138 removes the canonical regex parser for QA15/QA16. Boundary moved to paraphrase robustness and delayed final-answer credit |
| attribute paraphrase stress | QA15/QA16 strong paraphrased eval | original-train learned front-end drops to QA15 acc/CE 0.567/2.346 and QA16 0.332/2.255; strong-paraphrase training recovers QA15 1.000/0.0036 and QA16 0.995/0.0417 | normalized diagnostic front-end original-train strong-test reaches QA15 1.000 and QA16 0.995; detector failures localize mainly to statement event acc 0.732/0.552 | R139 shows the attribute front-end is grammar-fragile under unseen paraphrases, but remains locally adaptable when exposed to the new surface distribution. Next should adapt statement-event/value detectors from delayed QA-level answer credit rather than local structure labels |
| attribute delayed QA-credit | QA15/QA16 strong paraphrased credit/eval | statement-only final-answer credit improves QA15 0.567/2.346 -> 1.000/0.0036 and QA16 0.332/2.255 -> 0.970/0.132 over three seeds | strong structural-label upper is QA15 1.000/0.0023 and QA16 0.995/0.042; query credit is harmful and disabled | R140 mostly closes the R139 paraphrase gap without paraphrase-local structure labels. It still uses local surface-cue candidates and leaves a QA16 value-slot gap, so next should distribute answer credit over multiple eligible statements |
| pair statement credit | QA15/QA16 strong paraphrased credit/eval | enabling pair-statement credit gives zero paired delta vs R140 on QA15/QA16 acc and CE across seeds | QA16 does select pair updates in epoch0, but held-out metrics do not move | R141 rules out simple pairwise coordinate replacement as the fix for the QA16 value-slot residual. Next should test broader value-slot consolidation or all-relevant-statement eligibility |
| slot consolidation credit | QA15/QA16 strong paraphrased credit/eval | error-row slot consolidation improves QA16 0.970/0.132 -> 0.995/0.0417 and keeps QA15 1.000/0.0036 | matches strong structural-label upper on QA16 acc 0.995 and slightly beats its CE 0.0420; detector event/entity/value/query/subject all reach 1.000 | R142 supersedes R140 for QA15/QA16 paraphrase adaptation. It is still surface-cue constrained, so next should stress less templated paraphrases or export more bAbI tasks |
| relation-state size/path QA | bAbI QA18/QA19 full split, seeds=0/1/2 | R143 `size_relation_state_no_bp` reaches QA18 acc/CE 0.9307/0.1556 with 64KB state; `path_relation_state_no_bp` reaches QA19 0.9460/0.3010 with 256KB state | majority/raw/hashed/phase are weak: QA18 best non-symbolic diagnostic raw 0.529, phase 0.486; QA19 phase 0.098; symbolic graph upper is 1.000 | Positive expansion beyond movement and attributes: fixed random entity/place codes plus local relation matrices support size transitivity and two-hop path finding without raw replay. Boundary: regex front-end and lossy matrix superposition leave a gap to symbolic upper; next should add learned front-end and branch/inhibitory cleanup |
| learned relation front-end | bAbI QA18/QA19 canonical full split, seeds=0/1/2 | R144 learned local statement/query front-ends exactly preserve R143: QA18 0.9307/0.1556 and QA19 0.9460/0.3010 | Detector held-out type/slot metrics are all 1.000; state overhead is only ~8.5KB over R143; symbolic upper remains 1.000 | Removes the regex parser from the claimable canonical QA18/QA19 path. Remaining gap is now localized to matrix superposition/recurrent readout, not front-end parsing; next should do paraphrase + delayed QA-credit and relation cleanup/inhibition |
| relation paraphrase stress | QA18/QA19 strong surface rewrites | R145 original-train strong-test drops to QA18 0.586/1.743 and QA19 0.060/8.018; strong-train strong-test restores QA18 0.931/0.156 and QA19 0.946/0.301 | Failures localize to relation type/direction: QA18 statement event acc 0.607, QA19 direction acc 0.166, while slots/query mostly stay 1.000 | Confirms R144 front-end is surface-fragile but locally adaptable. Next target is final-answer delayed credit over relation-word candidates, not slot learning |
| bAbI paraphrase stress | QA2/QA3 strong paraphrased eval | original-train learned front-end drops to QA2 acc 0.409 CE 4.457 and QA3 acc 0.739 CE 1.559; strong-paraphrase training recovers QA2/QA3 acc 1.000, CE 0.002/0.003 | normalized diagnostic front-end original-train strong-test acc 0.745 on QA2 and 0.987 on QA3; detector failures localize to event/query-subject slots | R097 shows R091 is grammar-fragile under out-of-distribution paraphrase, but the same local no-BP front-end can adapt when exposed to new surface forms. 下一步应减少局部结构标签，做 delayed QA-level credit / answer-error apical update |
| delayed QA-level credit | QA2/QA3 strong paraphrased eval | seeded answer-credit improves QA2 80 acc/CE 0.400/4.562 -> 0.900/0.843 and QA2 300 0.447/4.357 -> 0.863/1.211 using only final answer credit on strong paraphrase; QA3 tiny/query runs show smaller acc gains but worse CE | cold answer-credit stays at majority; same-surface structural upper remains 1.000 but uses local parser labels | R098 gives the first positive answer-error/apical credit signal for adapting the parser without local labels on the new surface form. It is partial: needs structural seed, QA3 calibration is weak, and full medium requires cached candidate scoring or a learned recurrent/dendritic parser state |
| cached delayed QA-credit scaling | QA2 full / QA3 partial strong paraphrased eval | cached answer-credit scales QA2 to full 900/1000: acc/CE 0.409/4.457 -> 0.834/1.346 in 2m57s, with event acc 0.773->0.891 and query-subject acc 0.357->1.000 | cached QA2 300 is weaker than exact R098 0.807 vs 0.863; QA3 200 is negative, seed 0.785 drops to 0.645 query-only and 0.250 query+event | R099 makes the answer-credit path scalable enough for full QA2 but exposes that QA3 needs gated/rollback credit or a deeper recurrent/dendritic parser; brute answer-credit updates can damage before-location representations |
| gated delayed QA-credit | QA2 full / QA3 partial strong paraphrased eval | error-only + capped credit improves QA2 full to acc/CE 0.975/0.216, above R099 ungated 0.834/1.346; event acc 0.773->0.985 and query-subject acc 0.357->1.000 | QA3 200 still below seed: gated query+event 0.615/3.023 vs seed 0.785/1.100, but avoids R099 collapse 0.250/5.413 | R100 shows apical/error-only gating is the right direction for QA2 and prevents some destructive plasticity, but QA3 needs relation-specific before-location eligibility and slot-protecting credit rather than global query prototype writes |
| channel-protected delayed QA-credit | QA2 full / QA3 partial strong paraphrased eval | same-seed audit shows delayed credit is genuinely positive: QA2 full pre-credit 0.664/2.552 -> query-only 0.799/1.540 -> query+event 0.975/0.216; QA3 200 pre-credit 0.570/3.515 -> query-only 0.605/3.028 -> query+event 0.615/3.023 | event-only QA3 collapses to 0.270/4.852; earlier R097 QA3 seed 0.785 used different random features and is not a fair within-run credit baseline | R101 fixes the comparison by reporting `qa_credit_seeded_pre_credit` and independent query/event channel ablations. Conclusion: answer-error/apical delayed credit works for QA2 and weakly helps QA3, but broad event rewrites are harmful; next needs relation-specific before-location eligibility |
| before-relation delayed QA-credit | QA3 partial strong paraphrased eval | relation-specific before-location eligibility matrix improves same-seed QA3 200 from 0.570/3.515 to 0.660/2.578 with event credit disabled; smoke QA3 80 improves 0.600/3.053 -> 0.700/2.805 | R101 query-only 0.605/3.028; R101 query+event 0.615/3.023; R102 weight=1.0 overconfident at 0.645/3.405; slot-feature variant 0.650/4.069 changes the pre-credit baseline and is not adopted | R102 confirms QA3 should use relation-specific `(object,destination)->previous_location` eligibility instead of broad event rewrites. It still trails the different-seed structural baseline 0.785/1.100, so the next bottleneck is query subject robustness and seed repeats |
| before-relation seed repeat | QA3 partial strong paraphrased eval, seeds 0/1/2 | R102 before-matrix mean acc/CE 0.622/2.756 vs R101 query-only 0.600/2.954; same-seed pre-credit mean 0.537/3.391 | paired deltas are positive for seeds 0/1 but neutral/slightly negative for seed 2: +0.055/-0.450, +0.015/-0.156, -0.005/+0.012 | R103 upgrades R102 from single-seed to partial multi-seed evidence: CE gain is meaningful on average, but accuracy robustness is not solved. Next should stabilize query subject extraction with a learned relation-slot microcircuit |
| query-subject WTA microcircuit | QA3 partial strong paraphrased eval, seeds 0/1/2 | R104 WTA + before matrix mean acc/CE 0.777/1.552, improving every seed over R102 before matrix; WTA pre-credit alone mean 0.677/2.550 | R102 before matrix mean 0.622/2.756; R101 query-only mean 0.600/2.954; paired R104-R102 deltas: +0.270/-2.098, +0.100/-0.612, +0.095/-0.901 | R104 fixes the query subject seed fragility with a local candidate-WTA microcircuit over learned prototypes, while keeping event updates at 0. It is the strongest QA3 partial result so far, but full 900/1000 scaling and seed robustness are still required |
| QA3 full query-subject WTA scaling | QA3 full 900/1000 strong paraphrased eval, seeds 0/1 | R105 WTA pre-credit mean acc/CE 0.815/1.355 and WTA+before-credit mean 0.811/1.219; both beat R097 original structural seed mean 0.576/2.566 | before-credit improves CE on both full seeds but is top-1 neutral/slightly negative: seed0 +0.003/-0.143, seed1 -0.010/-0.129 | R105 confirms R104's query WTA scales to full QA3 and fixes subject robustness without event credit. The before-location credit should become confidence-gated before claiming final top-1 robustness |
| before-credit readout gate | QA3 full 900/1000 strong paraphrased eval, seeds 0/1 | lower before-credit weight 0.25 preserves WTA pre-credit mean accuracy 0.815 while improving CE 1.355->1.251; weight 0.5 gives lower CE 1.219 but mean acc 0.811 | low-margin gate is negative on seed1: margin 1.0/2.0 gives 0.777/1.652 and 0.775/1.658 vs WTA pre 0.778/1.605 | R106 shows before-credit is best treated as a weak calibration signal. Base-margin gating is not enough; next needs learned relation-state confidence |
| before-credit agree-top gate | QA3 full 900/1000 strong paraphrased eval, seeds 0/1 | agree-top weight 0.5 exactly matches WTA pre-credit on both seeds: mean acc/CE 0.815/1.355 with 0 before-relation updates | R106 weight 0.25 keeps mean acc 0.815 and improves CE 1.355->1.251; weight 0.5 improves CE to 1.219 with mean acc 0.811 | R107 is a negative boundary. Hard top-location agreement is accuracy-safe but blocks useful before-credit learning/calibration. Keep R106 weight 0.25 as the current top-1-safe operating point; next gate should use soft relation-state confidence rather than equality of top predictions |
| before-credit confidence gate | QA3 full 900/1000 strong paraphrased eval, seeds 0/1 | confidence gate scale 1.0 with weight 0.5 improves WTA pre-credit mean acc/CE 0.8150/1.3550 -> 0.8155/1.2242, with 167/184 before-relation updates | R106 weight 0.25 gives 0.8150/1.2508; R106 weight 0.5 gives 0.8115/1.2187; R107 agree-top gives no CE gain | R108 was a positive two-seed signal for soft relation confidence. R109 later revises the stronger top-1-safe claim after seed2 and scale sweep; keep R108 as the mechanism introduction, not final operating-point evidence |
| confidence-gate seed/scale sweep | QA3 full 900/1000 strong paraphrased eval, seeds 0/1/2 | confidence scale 0.75 is CE-best: mean acc/CE 0.8090/1.2321; confidence scale 1.25 is accuracy-conservative: 0.8107/1.2702; pre-credit is 0.8113/1.3683 | always weight 0.50 gives 0.8060/1.2387; always weight 0.25 gives 0.8103/1.2743 | R109 revises R108: confidence gate improves the Pareto tradeoff over always-on readout, but no tested scale is strictly top-1-safe across three seeds. Next needs flip-aware arbitration: preserve CE-only gains while blocking harmful answer flips |
| flip-aware confidence diagnostic | QA3 full strong paraphrased eval, seed2, confidence scales 0.75/1.25 | scale0.75 has 2 helpful vs 9 harmful flips, CE delta -0.1099; scale1.25 has 2 helpful vs 4 harmful flips, CE delta -0.0774 | harmful flips are high-confidence, not low-confidence: scale0.75 has 7 high + 2 very_high harmful flips; scale1.25 has 3 high + 1 mid | R110 shows top-1 loss is concentrated in a few answer flips while most same-prediction rows improve CE. Next should add base-margin-aware local inhibitory arbitration rather than a simple confidence threshold |
| margin-aware flip diagnostic | QA3 full strong paraphrased eval, seed2, confidence scales 0.75/1.25 | scale0.75 harmful/helpful mean pre-margin 1.458/0.755 and before-credit margin 3.349/0.896; scale1.25 harmful/helpful mean pre-margin 0.813/0.755 and before-credit margin 2.867/0.396 | helpful and harmful flips overlap in base margin; wrong-to-wrong CE-improved flips can also have high before-credit margin | R111 shows fixed margin thresholds are likely too brittle. Next should train a local flip gate/inhibitory arbiter on train-split final-answer credit rather than hand-selecting thresholds from test diagnostics |
| train-split learned flip gate | QA3 strong paraphrased eval, before-credit-only 300-row seed2, confidence scales 0.75/1.25 | scale1.25 gate improves post-credit acc/CE 0.677/2.093 -> 0.683/2.087 and blocks 2/2 harmful flips while allowing 1/1 helpful flip; scale0.75 improves post 0.673/2.005 -> 0.677/2.001 | full 900/1000 run currently too slow in the repeated state-scoring path; scale0.75 remains too permissive, allowing 3 harmful flips | R112 validates the learned local inhibitory arbiter mechanism on a medium before-credit-only run. Next needs score/state caching and full seed reruns with validation-selected gate hyperparameters |
| cached full flip-gate rerun | QA3 full 900/1000 strong paraphrased eval, seed2, confidence scales 0.75/1.25 | deterministic code caches reduce 300-row runtime 431.6s->66.9s and make full seed2 finish in ~185s; full gate gives scale0.75 post 0.797/1.285 -> gate 0.798/1.285 and scale1.25 unchanged 0.802/1.318 | full gate is limited by train/test flip-risk mismatch: scale1.25 train has 13 allow / 0 block flips while test has 7 allow / 14 block; scale0.75 train 14/3 vs test 6/23 | R113 resolves the runtime blocker but shows the perceptron gate is underdetermined by natural train flips. Next needs conservative one-class/counterfactual negative inhibitory credit |
| one-class conservative flip gate | QA3 full 900/1000 strong paraphrased eval, seed2, confidence scale 1.25 | one-class radius0.5 restores top-1 post 0.802 -> gate 0.804 while keeping CE 1.318, close to post and far better than pre 1.395; it blocks 4/4 harmful flips | radius0.5 also blocks 2/2 helpful flips; radius1.0 and 0.75 are too permissive | R114 gives first full seed2 top-1-safe local arbiter point. It works as safety inhibition, not yet as selective helpful/harmful separation. Next needs a permissive helpful channel or counterfactual negative training |
| class-prototype helpful flip gate | QA3 full 900/1000 strong paraphrased eval, seed2, confidence scale 1.25, class-prototype radii 0.50/0.75 | radius0.50 is top-1 safe like R114, post 0.802/1.318 -> gate 0.804/1.319, and blocks 4/4 harmful flips | it still blocks 2/2 helpful flips and has slightly worse CE than one-class radius0.50; radius0.75 allows 3/4 harmful flips and drops acc to 0.801 | R115 is a negative boundary. Splitting positive train flips into helpful/wrong-improved prototypes does not separate helpful from harmful test flips. Next needs counterfactual negative flip generation and richer local compatibility features |
| counterfactual risk flip gate | QA3 full 900/1000 strong paraphrased eval, seed2, confidence scale 1.25, train-only forced-winner negatives | risk-only class prototypes with identity features store 3,882 counterfactual risk samples and restore top-1 post 0.802 -> gate 0.804 while allowing 1/2 helpful flips | it also allows 1/4 harmful flips and blocks 4/5 wrong-to-wrong CE-improving flips; CE 1.31865 is slightly worse than R114 one-class 1.31827 | R116 partially validates counterfactual inhibitory credit: it opens a helpful channel without test labels, but risk prototypes need richer compatibility features or density/quantile radii before becoming the best gate |
| risk-quantile flip gate | QA3 full seed2 plus 300-row smoke, counterfactual risk class prototypes | q0.90/q0.95 smoke improves post 0.677/2.093 -> gate 0.683/2.087 by allowing 1/1 helpful and blocking 2/2 harmful flips | full q0.90 does not transfer: it blocks 0/21 test flips and equals post 0.802/1.318; wider smoke radii revert to overblocking helpful/CE-improving flips | R117 is a negative full-split boundary. Single quantile radius does not calibrate train/test risk distance; next needs multiple micro-prototypes or explicit before-relation compatibility features |
| risk micro-prototype flip gate | QA3 full seed2 plus 300-row smoke, 4 risk micro-prototypes per answer class | full micro4 q1.00 restores top-1 post 0.802 -> gate 0.804, blocks 3/4 harmful flips, and allows 1/2 helpful flips; smoke micro4 q0.90/q1.00 improves post 0.677/2.093 -> gate 0.683/2.087 | full micro4 q0.90 keeps 2/2 helpful flips but allows 3/4 harmful and only reaches 0.803; q1.00 CE 1.31862 is still worse than R114 one-class 1.31827 | R118 is partial positive but not new best. Multiple local risk centroids help versus single quantile radius, but harmful/helpful separation now needs semantic before-relation compatibility features rather than more radius tuning |
| before-compatibility flip gate | QA3 smoke/full seed2, R118 micro4 q0.90 plus compatibility features or filters | `post_top` filter blocks 3/4 harmful flips on full and smoke remains 0.683/2.087; `post_better` preserves R118 behavior | `post_top` also blocks 2/2 helpful flips and worsens full CE to 1.31892; raw compat features allow a harmful smoke flip and drop smoke acc to 0.680 | R119 is a negative boundary for hand-written compatibility. Compatibility should become a learned separate inhibitory/helpful channel, not raw Euclidean risk dimensions or a hard top-match rule |
| center-difference update diagnostic | compositional cue + bAbI QA1 local batches | local no-BP update aligns with center-difference negative-gradient direction: compositional cosine 0.721, QA1 cosine 0.559; local dCE negative in all medium runs | same-norm random direction average dCE positive: compositional +0.086, QA1 +0.013 | 局部规则有真实 loss-reducing 方向性，但 R082 说明瓶颈在状态表示和多跳 credit/state transition，而不是简单继续调读出 |
| phase-binding token learner | TinyStories tokenizer 50k/10k, vocab=256 | trace+apical(random-feedback)+inhibition WTA online full precision post CE 2.289, acc 0.437, greedy repeat-2 0.383; variable-type 8-bit row CE 2.295 acc 0.438, serialized state 706,841 bytes; 8-bit all-state row CE 2.523 acc 0.462; 8-bit readout-weight-only CE 2.295, inhibition-only CE 2.289, phase_codes-only CE 2.290, phase_prototypes-only CE 2.289, bias-only CE 2.513, counts-only CE 2.513 | phase WTA post CE 2.427 acc 0.422; trace+inhib post CE 2.358 acc 0.429 repeat-2 0.473; fixed-random gate post CE 2.316; 8-bit fixed clip CE 2.713; 16-bit row CE 2.504; online `sparse_context_aux` post CE 1.297 acc 0.571 | 当前最强纯 no-BP CE 是弱 dynamic apical error modulation + inhibition；随机反馈可承载 apical error。variable-type quantization 已显示向量/矩阵 int8 row + count/prior float32 可接近 full precision，同时 deployable state 约 0.707MB；corrected prediction-only anti-score gives a strong smoke repetition boundary but medium remains a CE/repetition tradeoff, and candidate branch-agreement competition worsens loops; next step remains loadable integer checkpoint and prediction consistency verification；统计 auxiliary 仍只作调试/上界 |
| Hebbian KV token branch | TinyStories tokenizer 50k/10k, vocab=256, seeds=0/1/2 | R134 `phase_trace_kv_apical_inhib_competitive_online` mean post CE 2.2668 vs baseline 2.2791; all three seeds improve CE; greedy repeat-2 0.456 vs 0.506. R135 hard `kv_margin` gate recovers mean acc to baseline 0.4580 and reduces retention penalty 3.0866->3.0695 | R134 mean post acc drops 0.4580->0.4557; retention CE worsens 3.0382->3.0866; state grows 2.89MB->3.35MB; naive KV feature concat was negative in smoke. R135 gate removes most CE gain: gated CE 2.2776, seed2 negative vs baseline | Partial positive plus gate boundary: rank-1 Hebbian associative matrix can add useful probability mass without raw text or BP, but hard scalar confidence is too coarse. R096 should not include KV until a learned local apical/inhibitory arbiter preserves CE without acc/retention loss |
| e-prop eligibility readout | TinyStories tokenizer 50k/10k, vocab=256, seeds=0/1/2 | R136 weak finite-window eligibility mix improves mean post acc 0.4580->0.4730 and greedy repeat-2 0.5059->0.4255 | post CE worsens 2.2791->2.3565; retention CE worsens 3.0382->3.0680; full eligibility replacement collapses in smoke | Tradeoff-positive: eligibility trace carries winner-selection and anti-repetition signal but is poorly calibrated for CE. Use it as a candidate/tie-break or anti-loop branch, not as the main probability feature |
| phase token order sweep | TinyStories tokenizer 50k/10k, vocab=256 | order=2 phase CE 3.551 acc 0.284 | order=1 CE 3.603; order=3 tuned CE 4.199; order=4 CE 10.462 | 负结果：naive 多 token 相位乘法退化，下一步应做分支化 dendritic/SSM 或门控组合，而不是简单堆 context order |
| continuation backoff | TinyStories tokenizer medium | continuation_backoff CE 3.3254, acc 0.360 | sparse order=3 CE 4.0796; combined_context CE 4.0368 | 仅作为强统计基线/调试工具；n-gram/Kneser-Ney 风格，不能作为最终路线 |
| online no-raw stream | TinyStories segmented stream | phase_competitive_online post CE 2.427, acc 0.422; continuation_backoff post CE 1.542, acc 0.791 | phase stream pre CE 3.425, online CE 3.229; sparse aux post CE 1.297 | 纯 phase/WTA 在线学习已跑通且有强正增益；统计 memory/continuation 只保留为适应上界和调试基线 |
| memory cap pruning | TinyStories segmented stream | cap=5000: continuation post CE 1.957, 352KB | uncapped CE 1.542, 618KB; cap=2000 CE 2.252, 216KB | 只用于存储-质量调试，不作为核心方法 |
| API-compatible local prototype | synthetic personalization/FAQ | hashed_memory acc 1.000 smoke / 1.000 medium | no-memory acc 0.025 / 0.017; raw retrieval 0.800 | 本地 schema adapter 形态有效 |
| real API schema QA smoke | synthetic personalization/FAQ | API+memory_hint acc 1.000 on 10 questions | API no-memory acc 0.200 | 真实 API 调用链闭合，小样本强正结果 |
| natural FAQ API demo | synthetic customer FAQ | API+memory_hint acc 1.000 on 8 questions | API no-memory acc 0.000 | API 能用 no-BP memory hint 生成自然 FAQ 答案 |
| generated FAQ API scale | generated customer FAQ | API+memory_hint acc 1.000 on 16 questions; local 256-fact acc 1.000 | API no-memory acc 0.000 | 扩展到模板生成多事实空间后仍闭合，但还不是开放域 dialogue |
| dialogue paraphrase FAQ API | generated dialogue-style FAQ | API+memory_hint semantic acc 1.000 on 16 paraphrased questions; local 256-fact acc 1.000 | API no-memory acc 0.000 | 支持从对话式指令更新 memory 后回答改写问句；仍依赖 schema/alias |
| FAQ revision/delete API | generated dialogue-style FAQ revision | local overwrite/delete all 1.000; API overwrite 1.000/delete suppression 1.000 on 4 facts | old-value leak 0.000 | 支持在线覆盖和遗忘，不靠保存旧 raw dialogue；仍存 canonical answer values |
| semantic FAQ router API | generated dialogue-style FAQ | semantic router local 256-fact acc 1.000; API+memory_hint acc 1.000 on 8 questions | API no-memory acc 0.000 | 去掉 alias 必需性，使用 hashed semantic prototypes；代价是 256 facts state 约 825KB |
| compressed semantic FAQ router | generated dialogue-style FAQ | sparse-only cap12 local 256-fact acc 1.000; revision/delete all 1.000 | uncompressed semantic state 824,791 bytes | 256 facts state 降到 292,592 bytes，保留 no-raw semantic routing |
| compressed semantic FAQ router API | generated dialogue-style FAQ | sparse-only cap12 API memory-hint acc 1.000 on 8 questions; local 64-fact acc 1.000 | API no-memory acc 0.000 | 压缩 router 仍能闭合真实 API 链路，state 79,931 bytes on 64 facts |
| value sketch FAQ memory | generated dialogue-style FAQ | sketch store local 256-fact acc 1.000; API memory-hint acc 1.000 on 8 questions | full answer store state 292,614 bytes | 不保存完整 answer text，改存结构化 answer sketch；state 284,248 bytes on 256 facts |
| multi-turn FAQ API session | generated dialogue FAQ support session | semantic sketch memory local acc 1.000 on 14 turns; API acc 1.000 on 10 turns | API no-memory acc 0.200; raw retrieval acc 1.000 | learn/query/revise/delete/query 闭环完成，并产出 human-readable transcript；仍非开放域 GPT-like final eval |
| personalized style API | generated writing preferences | API+style_sketch all-pass acc 1.000 on 8 natural writing prompts; deleted suppression 1.000 | API no-memory all-pass acc 0.000; raw profile acc 1.000 | 从事实召回推进到自然短文生成约束遵守和删后遗忘；仍是规则化偏好，不是开放域人类偏好评测 |
| personalized style judge | blinded API ranker on style outputs | style-sketch best 0.500 on 8 prompts; beats raw-profile 0.750 | no-memory best 0.500; raw-profile best 0.000 | 自然度/可用性评测显示 style-sketch 有价值但并未压倒 no-memory，说明还需更柔和的 hint/render 路线 |
| preference-aware style judge | generated writing preferences | strict style-sketch best 0.875 with learned preference context; soft style-sketch best 0.625 | request-only judge preferred no-memory 1.000 on soft outputs | 个性化评审必须显式给出 learned preference；当前 strict sketch 比 soft sketch 更强，下一步应修偏好语义而不是弱化约束 |

## 最终验收标准

最终结果以“no-BP 在线学习方法在推理样本上接近主流 GPT 输出”为目标，但每轮实验必须先通过本地定量门槛，避免只看个别生成样本。

| 维度 | 验收方式 | 当前门槛 | 最终门槛 |
|---|---|---|---|
| 生成质量 | held-out prompts 的 greedy/sample completion，人眼检查是否自然、少重复、能延续语义 | 明显优于 STDP/BTSP/dendritic/recurrent_3factor | 在目标任务上接近 GPT5.5/API 样本的可读性和任务完成度 |
| CE / PPL | TinyStories 或 `wwwy4/huggingface_datasets` 中合适 next-token/QA 数据集 | 先超过 sparse-only medium CE 4.1126 或至少不退化 | 接近强 frozen/API baseline 的 proxy CE 或偏好分数 |
| Accuracy | next-token top-1、QA/分类准确率、个性化事实召回准确率 | 超过 sparse-only acc 0.361 或在低置信 subset 明显提升 | 在任务准确率上接近主流模型，同时训练更快 |
| 训练速度 | train tokens/s 和 wall-clock 适应时间 | 保持 no-BP memory 的高吞吐优势，避免为了小提升牺牲 10x 速度 | 在线更新显著快于 BP/微调 |
| 在线学习 | prequential / streaming evaluation | 不保存原始文本，只保存统计/压缩 memory state | 实际交互中边用边学，不依赖 replay 原始数据 |
| 存储与隐私 | active contexts、bytes/token、是否可还原原文 | 报告 memory size，并实现 pruning/decay | 可控存储，不保存原始训练样本 |

在把任何结果解释为“可替代 BP 的主方法”之前，必须先在**纯 no-BP 本地结构**上完成 smoke/medium 验证；所有 API / 冻结主干 / 预训练 LLM 只允许出现在对照、接口或后验应用示例中，不允许反向定义主线方法。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: 局部/结构化 no-BP 神经机制能在受控时序和组合任务上替代查表 | 证明算法机制不是普通 n-gram 统计或 pair lookup | delayed cue 中 eprop 成功；组合 cue 中 target-only phase-binding Hebbian 只用最终 target 信号对 held-out pairs 达到 1.000，而 pair lookup/eprop/BPTT/reservoir 失败 | B1, B2 |
| C2: 稀疏/continuation memory 是强统计基线而非最终方法 | 防止把 n-gram/Kneser-Ney 风格记忆误判为可替代 BP 的仿生神经结构 | 用它们作为调试下界，并要求纯 recurrent/SSM/dendritic 主体在关键任务上超过或解释这些基线 | B3, B4 |
| C3: 未来接 GPT/API 应采用置信度门控 adapter，而不是 always-on 线性融合 | 线性融合已被 medium 结果否定 | gated adapter 在低置信 context 才调用 backoff，CE 或 acc 超过 sparse-only，且生成重复不恶化 | B5, B6 |

## Paper Storyline

Main paper must prove:

- “替代 BP”必须先限定为**纯 no-BP 神经网络类小/中规模结构**，不依赖任何 BP 预训练主干。
- 最终候选应是 DEN1810/dendritic-like、STDP/BTSP 调制、e-prop/三因子 eligibility、反馈对齐、抑制性回路或 recurrent/SSM/reservoir 动态这类神经结构。
- 三因子固定反馈、eligibility trace、树突/抑制性局部误差、相位/振荡绑定或 reservoir/SSM 动态必须在受控任务中学习时序 credit assignment 和组合泛化。
- 在真实 token 数据上，稀疏 Hebbian / continuation memory 只能作为调试基线、数据 sanity check 和统计下界；最终主线必须是可训练的纯 no-BP 神经/动态主体，而不是 n-gram/Kneser-Ney-style 表格记忆。
- 与 GPT/API 的连接只能是外部上界或应用展示；论文核心 claim 应围绕纯仿生结构、局部更新和硬件可实现性。

Appendix can support:

- MNIST DFA 接近 BP 的早期结果。
- recurrent_3factor TinyStories 负结果。
- normalized backoff 和 linear hybrid 负结果。
- 更多 context order / memory size / seed sweep。

Experiments intentionally cut:

- 不做 GPT 级规模的从零训练，但必须做纯 no-BP 神经网络类小/中规模 token learner；不能用“预训练 LLM + 外挂 memory”或“n-gram/backoff 统计表”替代这个主线。
- 大规模 API 消耗测试；API 只作为后验对照或评测工具。
- 继续盲目刷 API/FAQ demo；除非它服务于纯 no-BP 主体的评估。
- 继续大规模调 recurrent_3factor 前，应先完成 delay sweep、compositional cue、tree/dendritic error、phase-binding token learner 和可硬件化 memory 的关键判别实验。

## Experiment Blocks

### Block 1: Delayed And Variable Cue Sanity

- Claim tested: C1
- Why this block exists: 防止三因子结果只是在固定 delay 上记位置。
- Dataset / split / task: synthetic cue/filler/target，delay in {4, 8, 12, 16, 20}，再加 variable-delay train/test。
- Compared systems: bigram, reservoir, eprop_3factor, eprop_resampled, tuned BPTT。
- Metrics: target-position accuracy, overall accuracy, loss, train time。
- Setup details: hidden_dim in {12, 24}, seeds=3。
- Success criterion: eprop_3factor 在长 delay 和 variable delay 上显著超过 reservoir/resampled，并接近 tuned BPTT。
- Failure interpretation: 若只在固定 delay 成功，则当前规则更像位置模板而不是稳健时序记忆。
- Table / figure target: Appendix or method sanity figure。
- Priority: MUST-RUN, but lower than B5 because main token result has moved forward。

### Block 2: Compositional Cue Task

- Claim tested: C1
- Why this block exists: 区分联想记忆和简单查表。
- Dataset / split / task: `C_a, F, C_b, F...F, T_(a+b mod K)`，包含 held-out cue pairs。
- Compared systems: pair lookup, hand-coded phase binding, learned cue-to-phase binding, target-only phase binding, scrambled phase control, reservoir, eprop_3factor, eprop_resampled, tuned BPTT。
- Metrics: seen-pair target acc, held-out-pair target acc, loss。
- Setup details: K in {4, 8}, held-out pair split, seeds=3。
- Success criterion: target-only phase binding 在 held-out pairs 上稳定超过 pair lookup 和 scrambled control；若普通 eprop/RNN 失败，则说明需要显式结构化绑定而不是泛型 recurrent credit assignment。
- Failure interpretation: seen 成功 held-out 失败意味着机制更偏局部关联，暂不能宣称组合推理；若只有 hand-coded phase 成功，则说明手工结构过强，需要继续去教师化。
- Table / figure target: Main or appendix depending on outcome。
- Priority: MUST-RUN。

### Block 3: TinyStories Statistical Baselines For Debugging

- Claim tested: C2
- Why this block exists: 固化 n-gram / Kneser-Ney-style 统计基线，作为纯 no-BP 神经主体必须超过或解释的下界；不把它当最终路线。
- Dataset / split / task: TinyStories tokenizer-level next-token，medium 设置。
- Compared systems: STDP, STDP-Bio, BTSP, BTSP-Bio, dendritic_error_1810_lite, recurrent_3factor, phase_binding_token, sparse_hebbian_context, continuation_backoff, low-budget torch_llama upper-bound/control。
- Metrics: CE, ppl, top-1 acc, train tok/s, active contexts, greedy sample repetition。
- Setup details: train_chars=50k, valid_chars=10k, max_vocab=256, context_max_order=4, seeds/config repeats=3 if script supports seeds；phase-binding order sweep 已显示 order=2 最稳。
- Success criterion: 统计基线稳定可复现，并用于暴露数据切片、tokenizer、评测和生成重复问题；pure phase-binding token learner 至少在 CE 或低置信 subset 上超过统计辅助基线，同时明确 top-1 差距。
- Failure interpretation: 如果纯神经 no-BP 主体无法超过这些统计基线，需要重新设计 credit assignment / state dynamics，而不是把统计基线包装成最终方法；若 naive order>2 退化，则改用 dendritic branches、gating 或 SSM state。
- Table / figure target: Baseline / Appendix table。
- Priority: DEBUG-BASELINE。

### Block 4: Context Order, Calibration, And Memory Size Ablation

- Claim tested: C2
- Why this block exists: 找到 sparse memory 的真实收益来源和规模边界。
- Dataset / split / task: TinyStories tokenizer medium。
- Compared systems: max_order in {1, 2, 3, 4, 5, 6}; additive vs normalized; memory pruning; alpha/temperature sweep。
- Metrics: CE, acc, active contexts, bytes per token, repetition rate。
- Setup details: 固定数据和 eval；先单 seed 小 sweep，再复跑最优三组。
- Success criterion: 证明 high-order context 提供主要收益，并得到可控 memory-size/quality tradeoff。
- Failure interpretation: 如果 order>2 没收益，方法更像普通 bigram/trigram，需要重新定位。
- Table / figure target: Main ablation table。
- Priority: MUST-RUN。

### Block 5: Confidence-Gated Online Adapter

- Claim tested: C3
- Why this block exists: 当前 linear hybrid medium 负结果说明不能始终混入 neural logits；需要让 backoff 只处理 memory 低置信场景。
- Dataset / split / task: TinyStories tokenizer medium，先本地 low-budget Llama/dendritic backoff，之后再考虑 API。
- Compared systems: sparse memory only, normalized sparse memory, linear hybrid, gated+dendritic backoff, gated+Llama backoff。
- Metrics: CE, acc, low-confidence subset CE, high-confidence subset CE, gate usage rate, repetition rate, train/eval tok/s。
- Setup details: confidence = highest-order context exists + total row count + max row probability/entropy；threshold sweep。
- Success criterion: gated adapter CE < 4.1126 or acc > 0.361，且 greedy repetition 不差于 sparse-only；低置信 subset 上 backoff 有明确收益。
- Failure interpretation: 若 gated 仍退化，说明当前 neural backoff 太弱或 logits 标度不兼容，先换更强 frozen backbone 再测。
- Table / figure target: Main Table 2 and confidence calibration plot。
- Priority: MUST-RUN FIRST。

### Block 6: Continual No-Raw-Data Online Learning

- Claim tested: C2/C3
- Why this block exists: 直接验证最终目标中“无需保存数据直接在实际中学习”的核心约束。
- Dataset / split / task: TinyStories 按 story/domain/style 分段在线流式输入；训练后丢弃原文，只保留 memory/adapter state。
- Compared systems: phase_competitive_online, sparse memory auxiliary, continuation auxiliary, replay-free vs compressed-stat replay。
- Metrics: prequential CE, adaptation tokens-to-improvement, old-style retention, new-style adaptation, memory growth, deletion/pruning 后性能。
- Setup details: online train/eval interleaving；不得使用原始训练段回放。
- Success criterion: 新风格 CE 在少量 token 内下降，旧风格不过度遗忘，memory size 可通过 pruning 控制；纯 phase/WTA 已满足第一版在线正结果，下一版看低精度和生成质量。
- Failure interpretation: 若只能记住最近局部 n-gram，需引入聚类/semantic key 或 API embedding key。
- Table / figure target: Main online learning figure。
- Priority: MUST-RUN after B5。

### Block 7: Pure No-BP Application Bridge

- Claim tested: C3 and long-term goal
- Why this block exists: 只在纯 no-BP 主线稳定后，再把机制映射到应用接口；不是依赖预训练 LLM 来定义方法，而是用它们做后验接口或评测上界。
- Dataset / split / task: 小规模个性化写作/FAQ/客服风格流，但主方法必须先用纯 no-BP 主体完成；如果接入 API，只能作为对照评估，不得作为核心训练骨干。
- Compared systems: pure no-BP model, raw retrieval baseline, optional API or frozen backbone upper bound。
- Metrics: next-token/proxy CE if logits available, task accuracy, exact recall, hallucination rate, storage bytes, privacy constraint。
- Setup details: 只保存 hashed/compressed context statistics or embedding-keyed memory；若使用 API，仅用于评测，不用于训练主方法。
- Success criterion: 纯 no-BP 主体在个性化/新事实任务上能自洽学习，并满足 no-raw-data storage；若 API 只提供上界，则不影响主结论。
- Failure interpretation: 若纯 no-BP 主体必须借助预训练 LLM 才能有效，则该路线不满足当前项目约束。
- Table / figure target: Demo / application section。
- Priority: NICE-TO-HAVE after pure no-BP core passes。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 固化纯 no-BP 证据 | 复跑 eprop_3factor、dendritic_error；sparse/continuation memory 仅作调试基线 | 至少一个纯 no-BP 神经/动态结构在受控任务稳定强于 reservoir/output-only，并明确区分统计基线 | CPU/GPU minutes | 单次结果偶然 |
| M1 | 机制判别 | delay sweep、trace decay、feedback fixed/resampled、dendritic apical/basal ablation | 证明性能来自 eligibility/feedback/dendritic mechanism，而不是 n-gram 或随机特征 | CPU < 1h | 机制只是查表 |
| M2 | 纯 no-BP token learner | phase/trace/WTA + output inhibition + dendritic/apical local error gating on TinyStories tokenizer | 不依赖 Llama/API 的 CE/acc 明显优于 STDP/BTSP/dendritic 旧 baseline，并持续缩小与统计辅助基线的差距 | CPU/GPU 1-2h | 长文本生成弱 |
| M3 | 可硬件化在线学习 | memory cap、局部更新、整数/低精度状态、no-raw replay | 存储、更新和读出都能解释为局部/硬件友好操作 | CPU/GPU 1-2h | memory 无界增长 |
| M4 | 组合与泛化 | compositional cue、held-out pair、style/fact shift 的纯 no-BP 版本 | 证明不是只记训练 n-gram 或固定位置 | CPU 1-2h | 组合泛化失败 |
| M5 | 外部对照/应用展示 | 可选 API/frozen baseline 只作上界或评测器 | 不改变主结论；若纯 no-BP 不成立则不得用 API 结果补洞 | API 成本受控 | API 结果掩盖方法本体弱点 |

## Compute and Data Budget

- Total estimated GPU-hours before any external-model comparison: < 5 小时，小模型即可。
- API budget rule: API 不再是路线推进条件；只有纯 no-BP 主体已经形成可解释结果后，才可作为评测器或上界对照使用。
- Data preparation needs: TinyStories 分段流式切片；可选构造 style/fact shift、composition split、delayed association split。
- Human evaluation needs: 只有纯 no-BP 模型能生成可读样本后才需要少量人工检查。
- Biggest bottleneck: 纯 no-BP 主体的长程 credit assignment、context 泛化、memory size 控制和硬件可实现性。

## Risks and Mitigations

- Risk: 依赖 BP 预训练 LLM / API 会污染主张，使方法变成“BP 模型外挂记忆”，不适合纯仿生结构或硬件路线。
- Mitigation: 所有主表、主结论、路线图必须以纯 no-BP 从头训练或局部在线更新结构为核心；API/预训练模型只能放在附录、上界或应用展示。

- Risk: sparse / continuation memory 只是 n-gram 或 Kneser-Ney-style 统计计数，不能支撑“类脑学习算法”叙事。
- Mitigation: 明确标记为 DEBUG-BASELINE；只用于调试、下界和 sanity check。最终 claim 必须来自 DEN1810/dendritic-like、STDP/BTSP、e-prop、feedback alignment、recurrent/SSM/reservoir 等纯神经或动态结构。

- Risk: 把“无梯度统计方法”误认为“神经网络类 no-BP 方法”。
- Mitigation: 主方法必须包含可解释的神经状态、突触更新或局部可塑性机制；没有神经状态/突触结构的表格统计方法只能辅助分析。

- Risk: 线性 hybrid 已失败，gated 也可能失败。
- Mitigation: 分 high/low-confidence subset 报告；若 backoff 只在 low-confidence 子集有效，仍有价值。

- Risk: low-budget Llama 或 API baseline 会把研究重心带回 BP 训练模型，而不是纯 no-BP 结构。
- Mitigation: Llama/API 只能作为对照或评测上界；真正 claim 必须在纯 no-BP 主体上提出。

- Risk: no-raw-data memory 仍可能泄漏原文。
- Mitigation: 报告 memory schema、存储 token IDs/hashed keys/统计值，增加 pruning 和 privacy audit。

- Risk: 直接追求 GPT 级别会烧钱且不可证。
- Mitigation: 阶段性目标先定义为在线个性化 adapter，只有在 adapter 明显提升后再扩大。

## Final Checklist

- [x] MNIST DFA 接近 BP 的早期证据已覆盖
- [x] delayed hard toy task 已显示 eprop_3factor 可达 tuned BPTT
- [x] TinyStories sparse Hebbian context 有 medium 正结果
- [x] linear hybrid backoff 已判定为负结果
- [x] confidence-gated adapter
- [x] context order / calibration / memory size ablation
- [x] semantic-key Hebbian memory
- [x] continuation / Kneser-Ney-style backoff memory 已降级为调试基线，不作为最终方法
- [x] continual no-raw-data online learning
- [x] memory cap / pruning tradeoff
- [x] compositional cue task
- [ ] pure no-BP token learner 主线
  - [x] dendritic apical/basal error variant 第一版：branch-wise apical local error gate，seed0/seed1 均改善 CE
  - [x] apical gate ablation: global margin and fixed random feedback match or beat branch-local margin; fixed random gate is weaker
  - [x] low-precision / sparse-state audit 第一轮：8/16-bit projection 保留相对 apical 增益但 CE 校准退化
  - [x] generation rerun for random-feedback apical error: greedy repeat-2 0.473->0.383, controlled repeat-2 0.144->0.106
  - [x] quantization-aware updates / per-state scaling 第一轮：8-bit row scaling 将 apical CE 从 fixed-clip 2.713 改到 2.523，并保留 greedy repetition benefit
  - [x] selective quantization localization: CE 退化主要来自 `output_bias`/log-prior 和 count-like arrays；readout weights、inhibition、phase codes/prototypes 单独 8-bit row 接近 full precision
  - [x] dedicated bias clip sanity: `--low-precision-bias-clip 8.0` 未恢复 all-state row CE，说明不能只靠单独 bias clip
  - [x] cached target-array projection for low-precision wrapper
  - [x] serialized-state manifest/accounting: int8 vectors/matrices + float32 count/prior state recovers apical CE 2.295 with serialized bytes 706,841
  - [x] loadable integer checkpoint and prediction parity check: mixed int8/float32 `.npz` restores into a fresh no-BP learner with 1000-context pred_match 1.000 and max score diff 2.38e-7
  - [x] direct prior ablation: `--phase-bias-weight 0.0` raises variable 8-bit apical CE 2.295->2.523 but improves acc 0.438->0.462 and preserves apical advantage, showing winner selection can be neural while calibration still needs replacement
  - [x] simple global output homeostasis audit: target-up/wrong-down excitability does not recover CE calibration, though it is small and checkpointable
  - [x] feature-conditioned neural calibration first pass: fixed random gate + local target/wrong-winner calibration improves variable 8-bit bias-free CE 2.523->2.478 and acc 0.462->0.474, with checkpoint parity 1.000
  - [x] derived feature-calibration codes: regenerate fixed random codes from seed/architecture, preserving CE/acc/parity while reducing serialized bytes 986,465->724,317
  - [x] checkpoint config/hash validation: derived-state checkpoint stores SHA-256 signature and rejects corrupted/incompatible signatures before loading
  - [x] feature-calibration gate sweep: gate_decay=0.50 improves signed derived checkpoint CE 2.478->2.475 at unchanged 724,317 serialized bytes; dim32 offers smaller 716,093-byte variant with CE 2.480
  - [x] temperature / energy-scale audit: global temperature 0.7 improves bias-free feature-calibrated checkpoint CE 2.475->2.262 at unchanged bytes, beating direct-prior CE 2.295 without statistical token probabilities
  - [x] readout gain model-side calibration: fixed gain 1.428571 reproduces temperature0.7 CE 2.262/acc 0.476 with checkpoint parity 1.000; naive margin dynamic gain negative with CE 2.742
  - [x] local adaptive scalar readout gain: context-gated gain synapse reaches CE 2.276/acc 0.476 with checkpoint parity 1.000, close to fixed gain but below CE 2.262; stronger gains and base_gain=1.0 are negative, and greedy repetition is unchanged because scalar gain cannot change argmax
  - [x] branch-agreement gated readout: token-wise dendritic agreement improves fixed-gain bias-free checkpoint seed0 CE 2.262->2.253 and acc 0.476->0.478 with unchanged serialized bytes 724,317 and checkpoint parity 1.000; seed1 repeats direction weakly at CE 2.372->2.370 and acc 0.460->0.462; generation repetition gain is not seed-robust yet
  - [x] plastic branch-agreement target/wrong updates: small output-row branch-weight plasticity improves seed0 acc to 0.481 with checkpoint parity 1.000 and only 1,536 extra serialized bytes; seed1/seed2 retain small CE/acc direction over fixed gain, but CE-best remains fixed branch agreement and generation repetition does not improve
  - [x] inhibition-pressure-aware plasticity and loop-pressure diagnostics: pressure-gated plastic branch agreement reaches seed0 CE/acc 2.254/0.482 with checkpoint parity 1.000 and seed1/seed2 small positive ranking direction; generation loop metrics are now logged but repetition is still not improved
  - [x] dynamic loop-pressure inhibitory state during generation/observation: token-level loop inhibition is slightly negative, transition-level loop inhibition gives only a tiny CE change (best 2.253652->2.253623) while acc/repetition are unchanged and state grows to 793,193 serialized bytes; smoke checkpoint parity remains 1.000
  - [x] segment-level attractor detector: derived-code segment state can reduce medium greedy repeat-2 0.383->0.305, but only with unacceptable CE/acc cost 2.254/0.482->2.310/0.475 and worse controlled repeat-2, so always-on segment inhibition is not the final mechanism
  - [x] event-gated segment attractor pressure: margin gates protect CE but worsen repetition; inhibition gates can reduce smoke repetition but medium s2/t0.05 is worse than always-on segment inhibition (2.345/0.467, greedy repeat-2 0.333), so scalar gates are not selective enough
  - [x] learned loop-escape competitor: segment-pressure-triggered branch-support escape synapses can reduce medium greedy repeat-2 to 0.241 and preserve controlled repeat-2, but CE/acc cost is too high (2.398/0.473), so the current target/wrong update is too broad
  - [x] candidate-limited loop escape / winner-local anti-loop projection: top-k limiting does not improve CE over broad escape, and winner-suppress only reduces medium greedy repeat-2 to 0.206 by collapsing CE/acc to 2.481/0.385; output-side loop escape is not a stable final route
  - [x] representation-level recurrent/trace branch-state stabilizer: feature-space residual improves smoke CE/acc to 2.101/0.523 with exact checkpoint parity, but medium is neutral/negative versus R069 (2.262/0.482 vs 2.254/0.482), repetition is not improved, and full projection raises serialized state to 1,060,513 bytes
  - [x] low-rank / novelty-gated branch-state projection: rank16 preserves smoke CE/acc gain and shrinks medium serialized state to 737,957 bytes with checkpoint parity 1.000, but medium CE remains worse than R069 and novelty gates do not change greedy repetition
  - [x] low-rank state-space anti-attractor / prediction-only anti-score: update-path bug fixed so teacher-forced updates clear anti pressure; smoke can cut greedy repeat-2 to 0.064 at CE/acc 2.289/0.515, but medium remains a tradeoff (best fixed s0.75 CE/acc 2.409/0.477, greedy repeat-2 0.319, controlled repeat-2 0.121)
  - [x] candidate-local branch competition audit: top-k candidate limiting preserves smoke CE but behaves like the fixed anti-score; adding positive or negative branch-agreement signal worsens greedy loops, so local branch agreement is not the missing gate
  - [ ] next recurrent/generation-state mechanism should separate loop-state escape from branch agreement, likely by learning a local state reset or candidate-specific inhibitory trace rather than rewarding branch consensus
  - [x] **[R092] DLL dendritic localized learning (ICML 2025)**: added `--dll-depth-branch` / `OnlineDLLDeepLocalMemory` to `phase_binding_online_stream_experiment.py`. Each hidden layer uses local loss `||h_l - B_l@label_embed||^2` with fixed random target projection and no cross-layer gradient. Best three-seed candidate is one wide DLL layer + adaptive inhibition: post CE/acc `2.2807 +/- 0.0099` / `0.4720 +/- 0.0059`, state `6.30MB`. This is top-1 competitive but fails the CE success gate `<2.253`; two-layer DLL is worse. Verdict: DONE-TRADEOFF; proceed to R093 NoProp rather than keep widening DLL
  - [x] **[R093] NoProp decoupled layer training (2025)**: added `--noprop-depth-branch` / `OnlineNoPropLocalDenoisingMemory`. Each layer uses noisy target code `z_l = sqrt(a_l)*target_l + sqrt(1-a_l)*noise_l(local_input)`, with a local input map to `z_l` and local denoiser `z_l -> target_l`; deeper layer training uses previous clean target codes rather than a learned forward chain. Best three-seed candidate one wide NoProp layer + adaptive inhibition reaches post CE/acc `2.2782 +/- 0.0115` / `0.4672 +/- 0.0074`, state `10.63MB`. This slightly improves DLL CE but loses acc and costs more state, still failing CE gate `<2.253`. Verdict: DONE-TRADEOFF; deep local target features need calibration/arbitration before R096
  - [x] **[R094] Hebbian key-value attention branch**: implemented pure rank-1 Hebbian KV in `phase_binding_online_stream_experiment.py`. R134 medium seeds show CE gain 2.2791->2.2668 but acc/retention tradeoff; direct associative scoring works better than feature concat. Do not stack into R096 until a local confidence/novelty gate is added
  - [x] **[R135] hard KV confidence gate**: implemented `--kv-gate-mode` over recall norm, base margin, and KV anchor margin. A hard `kv_margin=0.08` gate recovers mean acc to baseline and reduces retention penalty, but collapses CE gain to 2.2776 vs baseline 2.2791 and R134 2.2668; scalar confidence is not enough for R096
  - [x] **[R095/R136] e-prop eligibility trace**: implemented finite-window eligibility readout. Weak mix order4/decay0.90/weight0.25 improves acc 0.4580->0.4730 and greedy repeat-2 0.5059->0.4255, but worsens CE 2.2791->2.3565 and retention 3.0382->3.0680. Use as tie-break/candidate branch, not direct CE feature
  - [x] **[R137] bAbI QA15/QA16 attribute binding**: added `attribute_binding_state_no_bp` to `babi_no_bp_qa_experiment.py`. Three-seed full split solves QA15 at acc/CE `1.000/0.0036` and QA16 at `0.995/0.0417`, beating phase and raw retrieval while storing no raw text. Boundary: parser is task-specific; next should learn the attribute/query front-end and stress paraphrases
  - [x] **[R138] learned QA15/QA16 attribute front-end**: added local learned statement/query detectors with fixed random token/position features, perceptron target/wrong updates, and slot prototypes. Learned mode preserves R137 performance (QA15 `1.000/0.0036`, QA16 `0.995/0.0417`) while detector test event/entity/value/query/subject metrics are all `1.000`; next boundary is paraphrase/delayed-credit adaptation
  - [x] **[R139] QA15/QA16 attribute paraphrase stress**: strong paraphrase eval exposes grammar fragility. Original-train learned front-end drops to QA15 `0.567/2.346` and QA16 `0.332/2.255`, mostly from statement-event detector failures (`0.732/0.552`). Strong-paraphrase training locally recovers QA15 `1.000/0.0036` and QA16 `0.995/0.0417`; normalized aware path is diagnostic only. Next should test delayed QA-level credit for attribute front-end adaptation
  - [x] **[R140] QA15/QA16 attribute delayed QA-credit**: added statement-only final-answer credit over local surface-cue candidates, with no paraphrase-local structure labels. Three-seed strong paraphrase eval improves QA15 `0.567/2.346 -> 1.000/0.0036` and QA16 `0.332/2.255 -> 0.970/0.132`; structural upper remains QA16 `0.995/0.042`. Boundary: candidate generator is cue-constrained and residual QA16 errors are value-slot coverage
  - [x] **[R141] pair statement credit boundary**: added `--enable-pair-statement-credit` to evaluate two-statement replacements. Three-seed paired delta vs R140 is exactly zero for QA15 and QA16 acc/CE, even though QA16 selects some pair updates. Simple pairwise eligibility is not enough; next should consolidate value-slot prototypes or update all answer-relevant statements
  - [x] **[R142] slot consolidation credit**: added `--slot-consolidation-mode`. Error-row slot-only consolidation closes the QA16 residual gap: QA16 `0.970/0.132 -> 0.995/0.0417`, matching structural upper, while QA15 stays `1.000/0.0036`. Detector event/entity/value/query/subject metrics are all `1.000`. Boundary: still relies on local surface cue candidates; next should stress broader paraphrases/tasks
  - [x] **[R143] QA18/QA19 relation-state expansion**: materialized bAbI `en-qa14/17/18/19` through the HF fallback and added `babi_relation_state_experiment.py`. Fixed-random-code relation matrices reach QA18 size reasoning `0.9307/0.1556` and QA19 path finding `0.9460/0.3010`, far above majority/raw/hashed/phase baselines but below symbolic upper `1.000`. Boundary: regex front-end plus matrix superposition gap; next should learn relation detectors and add cleanup/inhibitory arbitration
  - [x] **[R144] learned QA18/QA19 relation front-end**: added local learned size/path statement and query detectors. Learned-front-end relation-state exactly matches R143 on QA18 `0.9307/0.1556` and QA19 `0.9460/0.3010`; detector test metrics are all `1.000`. Remaining gap to symbolic upper is now relation-state superposition/readout, not canonical front-end parsing
  - [x] **[R145] QA18/QA19 relation paraphrase stress**: added strong rewrites for size/path relation words. Original-train strong-test drops to QA18 `0.586/1.743` and QA19 `0.060/8.018`; strong-train strong-test restores R144 levels. Failure localizes to relation type/direction, not slots, setting up delayed final-answer credit over relation-word candidates
  - [ ] **[R096] full deep no-BP integration**: R096-prep now has a positive calibrated deep backbone. NoProp 1x768 + adaptive inhibition + feature calibration + fixed readout gain 1.15 reaches post CE/acc `2.2108 +/- 0.0198` / `0.4749 +/- 0.0067`, beating R092/R093 and R134 CE baselines and passing the old `<2.253` gate on all seeds. It does **not** fully pass R096 yet: mean CE is still above `<2.20`, greedy repeat-2 is `0.496`, and state is `11.75MB`. Next R096 should integrate R095 e-prop and R094 KV only through local arbitration/gating, then solve repetition/state cost before claiming the full deep result
  - [x] recurrent/SSM/reservoir + 三因子 eligibility on TinyStories 已跑；fixed reservoir 有边际信号，eligibility transition 写入未通过 seed 复核
  - [x] target-only phase-binding token learner CE signal on TinyStories
  - [x] branched phase token learner improves CE to 3.282
  - [x] branch readout calibration improves acc to 0.312 but remains below sparse 0.323
  - [x] competitive WTA readout improves acc to 0.322 and CE to 3.195
  - [x] strict online phase/WTA stream improves pre CE 3.425 acc 0.303 to post CE 2.427 acc 0.422
  - [x] generation audit: online phase improves first-token match but greedy repeat-2 remains high; controlled decoding mitigates loops
  - [x] leaky trace/SSM branch improves online phase post CE to 2.389 and acc to 0.427, with modest greedy repetition reduction
  - [x] output fatigue dynamics improves trace+phase post CE to 2.382 and acc to 0.429, with greedy repeat-2 down to 0.606
  - [x] adaptive output inhibition improves trace post CE to 2.358 and seed1 trace CE 2.450->2.410, but generation loops remain
  - [x] context-gated output inhibition improves trace+fatigue acc to 0.435 and repeat-2 to 0.346, but CE worsens to 2.416
  - [x] plastic recurrent/SSM branch implemented; medium plastic SSM+inhib reaches CE 2.377 acc 0.425 but remains below trace+inhib
  - [x] phase-binding context-order sweep; naive order>2 negative
  - [x] composition / held-out cue 泛化
  - [x] target-only phase factorization without cue->phase teacher
  - [ ] hardware-friendly sparse memory / low-precision state audit
- [ ] optional API / pretrained-model upper-bound appendix
  - [x] local synthetic QA proxy
  - [x] real API schema QA smoke
  - [x] natural FAQ API demo
  - [x] generated natural FAQ API scale check
  - [x] dialogue/paraphrase API smoke
  - [x] overwrite/delete revision audit
  - [x] learned semantic routing
  - [x] semantic memory compression
  - [x] compressed API smoke
  - [x] value compression
  - [x] multi-turn human-readable qualitative demo
  - [x] personalized style API generation benchmark
  - [x] personalized style judge benchmark
  - [x] preference-aware style judge benchmark
- [ ] pure no-BP qualitative and quantitative final evaluation

## Latest QA Evaluation Update - 2026-06-18

- R120-R122 extended the bAbI QA3 flip arbiter with a learned before-relation compatibility channel. This remains pure no-BP: train-split final-answer loss provides local third-factor credit, and test labels are never used for updates.
- R120/R121 define a negative boundary: zero-threshold compatibility rescue is over-permissive, and counterfactual negative compatibility updates alone do not prevent harmful flips.
- R122 threshold `0.25` is the current best seed2 top-safe QA3 point: `0.804 / 1.318070`, slightly improving R114 `0.804 / 1.318268` and R118 `0.804 / 1.318617`.
- R123 tested that stricter rescue rule. `risk_compat_class_rescue` at threshold `0.25`, radius `1.0` improves seed2 to `0.805 / 1.317447`, keeps 2/2 helpful flips, and reduces harmful allowed from 2 to 1; across seeds 0/1/2 it is weak-positive but mostly tied with R122 (`0.811000 / 1.269958` -> `0.811333 / 1.269751`).
- R124 diagnosed seed1 risk misses with per-flip nearest-risk diagnostics. All 4 seed1 harmful flips are just outside risk radii (`risk_margin` -0.2709..-0.1046), so they pass through the base `not risk_match` path; global radius buffering blocks harmful flips only by also blocking many CE-improving flips and losing a seed2 helpful flip.
- R125 added a default-off risk-near-miss inhibitory path. Fixed three-seed result is weak-positive for safety: mean acc `0.811333 -> 0.812000`, mean harmful allowed `1.667 -> 1.000`, helpful allowed unchanged `1.667`; mean CE is effectively tied but slightly worse `1.269751 -> 1.269792`. Seed1 harmful allowed drops 4->2; seed0/seed2 accuracy is unchanged. A bug in the first run also clarified that gating must use raw any-prototype `risk_match`, not nearest-prototype diagnostic match.
- R126 replaced R125's global near-margin with a local radius-scaled buffer `0.2 * nearest_risk_scaled_radius`. This supersedes R125 as the current QA3 flip-gate point: mean acc stays `0.812000`, mean harmful allowed stays `1.000`, helpful allowed stays `1.667`, and mean CE improves to `1.269677`, slightly better than R123 `1.269751`.
- R127 added a post-fit fraction sweep and selected `risk_near_radius_fraction` using validation only. Validation selects `0.05`, not R126's `0.20`; the selected test result is `0.811333 / 1.269811`, which does not beat R123 `0.811333 / 1.269751`. Therefore R126 is a test-discovered candidate, not a validation-supported operating point under the current 100-row validation split.
- R128 added a train-internal 20% calibration fold excluded from structural seed, answer-credit, and flip-gate training. This also fails to support R126: all fractions tie on train-calibration metrics, selecting `0.00`; reduced-train test accuracy drops to `0.766`, so the fold is useful diagnostically but not as a final model. The current defensible operating point remains R123-style fraction `0.00`; R126 `0.20` stays a test-discovered mechanism probe.
- R129 added `--flip-gate-risk-near-blocks-rescue`, making near-risk matches a dominant inhibitory veto over compatibility/class rescue. Test-side mean acc improves weakly over R126 (`0.812000 -> 0.812333`), harmful allowed drops (`1.000 -> 0.667`), and helpful allowed stays `1.667`, but CE is slightly worse than R126 (`1.269677 -> 1.269730`). Because it still depends on the non-validated fraction `0.20`, it is a mechanism probe, not a default.
- R130 implemented a train-only learned near-risk inhibitory channel with counterfactual risk-boundary samples and validation-selected thresholds. The selected threshold `-0.1` is negative on test (`0.810667 / 1.270579`), worse than R123/R126/R129, and blocks most helpful flips (`1.667 -> 0.333`). The useful diagnosis is that the channel is swamped by roughly 7.7k inhibit samples versus 6-13 positive flips per seed.
- R131 added balanced resampling for the learned near-risk channel. This fixes R130's overblocking collapse but validation selects threshold `0.1`, giving the same test result as R123 (`0.811333 / 1.269751`, helpful `1.667`, harmful `1.667`). Stronger test-only thresholds are still not claimable and block helpful flips.
- R132 added a train-only auto threshold over balanced synthetic near-risk calibration samples. The calibration set is perfectly separable on all seeds and selects thresholds `0.1/0.0/0.1`, but the selected test result exactly matches R123/R131 (`0.811333 / 1.269751`, helpful `1.667`, harmful `1.667`). This rules out ordinary balanced train calibration as sufficient; the problem is still train/test risk-boundary mismatch.
- R133 split the learned near-risk channel into source-conditioned counterfactual-risk and natural-risk inhibitory readouts. This is negative: mean acc/CE drops to `0.810333 / 1.270101`, helpful allowed drops to `0.667`, and harmful allowed remains `1.667`. Seed1 still allows 4 harmful flips while seed0 overblocks helpful flips.
- Next QA step: keep R123 as the defensible validation-supported baseline and treat R126/R129 as mechanism probes. Stop adding small threshold variants to the current feature set; either generate harder train-only adversarial boundary samples around risk micro-prototypes or move the QA line back to representation learning via relation-state uncertainty, recurrent/dendritic parser state, or query-conditioned inhibition before answer-score arbitration.
