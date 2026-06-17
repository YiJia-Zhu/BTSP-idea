# MANIFEST

日期：2026-06-03

## Documents

- `idea-stage/LITERATURE_REVIEW.md`
- `idea-stage/IDEA_REPORT.md`
- `review-stage/AUTO_REVIEW.md`
- `NARRATIVE_REPORT.md`
- `RESEARCH_PIPELINE_REPORT.md`
- `temporal/TEMPORAL_REPORT.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `llm-token/ITERATION_2026-06-15.md`
- `llm-token/ITERATION_2026-06-15_HYBRID_BACKOFF.md`
- `llm-token/ITERATION_2026-06-15_ONLINE_STREAM.md`
- `llm-token/ITERATION_2026-06-15_STREAM_PRUNING.md`
- `llm-token/ITERATION_2026-06-15_QA_API_PROTOTYPE.md`
- `llm-token/ITERATION_2026-06-15_QA_API_REAL.md`
- `llm-token/ITERATION_2026-06-15_NATURAL_FAQ_API.md`
- `llm-token/ITERATION_2026-06-15_GENERATED_FAQ_API.md`
- `llm-token/ITERATION_2026-06-15_DIALOGUE_FAQ_API.md`
- `llm-token/ITERATION_2026-06-15_FAQ_REVISION_API.md`
- `llm-token/ITERATION_2026-06-15_SEMANTIC_FAQ_ROUTER_API.md`
- `llm-token/ITERATION_2026-06-15_SEMANTIC_ROUTER_COMPRESSION.md`
- `llm-token/ITERATION_2026-06-15_VALUE_SKETCH_API.md`
- `llm-token/ITERATION_2026-06-15_MULTITURN_FAQ_API.md`
- `llm-token/ITERATION_2026-06-15_STYLE_API.md`
- `llm-token/ITERATION_2026-06-16_STYLE_JUDGE.md`
- `llm-token/ITERATION_2026-06-16_STYLE_SOFT_HINT.md`
- `llm-token/ITERATION_2026-06-16_COMPOSITIONAL_CUE.md`
- `llm-token/ITERATION_2026-06-16_TARGET_ONLY_PHASE.md`
- `llm-token/ITERATION_2026-06-16_PHASE_TOKEN_LEARNER.md`
- `llm-token/ITERATION_2026-06-16_PHASE_ONLINE_STREAM.md`
- `llm-token/ITERATION_2026-06-16_PHASE_TRACE_BRANCH.md`
- `llm-token/ITERATION_2026-06-16_OUTPUT_FATIGUE.md`
- `llm-token/ITERATION_2026-06-16_ADAPTIVE_INHIBITION.md`
- `llm-token/ITERATION_2026-06-16_CONTEXT_GATED_INHIBITION.md`
- `llm-token/ITERATION_2026-06-16_PLASTIC_SSM_BRANCH.md`
- `llm-token/ITERATION_2026-06-17_ELIGIBILITY_SSM_BRANCH.md`
- `llm-token/ITERATION_2026-06-17_APICAL_GATING.md`
- `llm-token/ITERATION_2026-06-17_APICAL_ABLATION.md`
- `llm-token/ITERATION_2026-06-17_LOWP_APICAL_AUDIT.md`
- `llm-token/ITERATION_2026-06-17_QUANT_AWARE_APICAL.md`
- `llm-token/ITERATION_2026-06-17_SELECTIVE_QUANT_APICAL.md`
- `llm-token/ITERATION_2026-06-17_SERIALIZED_VARSTATE_APICAL.md`

## Code

- `no_bp_mnist_experiment.py`
- `temporal_sequence_experiment.py`
- `../tinystories_llama_token_experiment.py`
- `../tinystories_online_stream_experiment.py`
- `../online_memory_qa_experiment.py`
- `../online_memory_qa_api_experiment.py`
- `../online_memory_faq_api_experiment.py`
- `../online_memory_faq_multiturn_experiment.py`
- `../online_memory_style_api_experiment.py`
- `../online_memory_style_judge_experiment.py`
- `../compositional_cue_experiment.py`
- `../phase_binding_token_experiment.py`
- `../phase_binding_online_stream_experiment.py`

## Result Directories

- `results/pilot_v1/`
- `results/full_v1/`
- `results/full_v2/`
- `results/full_v3/`
- `temporal/results/delayed_quick_v1/`
- `temporal/results/delayed_hard_v1/`
- `temporal/results/bptt_tuned_v2/`
- `../output/tinystories_recurrent3factor_smoke/`
- `../output/sparse_hebbian_context_smoke/`
- `../output/sparse_hebbian_context_medium/`
- `../output/sparse_hebbian_context_normalized_medium/`
- `../output/hybrid_dendritic_context_medium/`
- `../output/hybrid_llama_context_medium/`
- `../output/online_stream_smoke/`
- `../output/online_stream_medium/`
- `../output/online_stream_cap5000/`
- `../output/online_stream_cap2000/`
- `../output/online_memory_qa_smoke/`
- `../output/online_memory_qa_medium/`
- `../output/online_memory_qa_api_dry/`
- `../output/online_memory_qa_api_run/`
- `../output/online_memory_qa_api_run10/`
- `../output/online_memory_faq_api_dry/`
- `../output/online_memory_faq_api_run/`
- `../output/online_memory_faq_generated_dry/`
- `../output/online_memory_faq_generated_256_dry/`
- `../output/online_memory_faq_generated_api_run/`
- `../output/online_memory_faq_dialogue_dry/`
- `../output/online_memory_faq_dialogue_256_dry/`
- `../output/online_memory_faq_dialogue_api_run/`
- `../output/online_memory_faq_revision_dry/`
- `../output/online_memory_faq_revision_256_dry/`
- `../output/online_memory_faq_revision_api_run/`
- `../output/online_memory_faq_semantic_256_dry/`
- `../output/online_memory_faq_semantic_revision_256_dry/`
- `../output/online_memory_faq_semantic_api_run/`
- `../output/online_memory_faq_semantic_sparseonly_cap12_256_dry/`
- `../output/online_memory_faq_semantic_sparseonly_cap12_revision_256_dry/`
- `../output/online_memory_faq_semantic_sparseonly_cap12_api_run/`
- `../output/online_memory_faq_value_sketch_smoke/`
- `../output/online_memory_faq_value_sketch_256_dry/`
- `../output/online_memory_faq_value_sketch_revision_256_dry/`
- `../output/online_memory_faq_value_sketch_api_run/`
- `../output/online_memory_faq_multiturn_dry/`
- `../output/online_memory_faq_multiturn_api/`
- `../output/online_memory_style_dry/`
- `../output/online_memory_style_api/`
- `../output/online_memory_style_delete_dry/`
- `../output/online_memory_style_delete_api/`
- `../output/online_memory_style_judge_dry/`
- `../output/online_memory_style_judge_api/`
- `../output/online_memory_style_judge_api_fixed/`
- `../output/online_memory_style_soft_dry/`
- `../output/online_memory_style_soft_api/`
- `../output/online_memory_style_soft_judge_api/`
- `../output/online_memory_style_soft_judge_context_api/`
- `../output/online_memory_style_strict_judge_context_api/`
- `../output/compositional_cue_smoke/`
- `../output/compositional_cue_r007/`
- `../output/compositional_cue_phase_smoke/`
- `../output/compositional_cue_phase_r007/`
- `../output/compositional_cue_learned_phase_smoke/`
- `../output/compositional_cue_learned_phase_smoke2/`
- `../output/compositional_cue_learned_phase_r007/`
- `../output/compositional_cue_targetonly_smoke/`
- `../output/compositional_cue_targetonly_r008/`
- `../output/phase_binding_token_smoke/`
- `../output/phase_binding_token_medium/`
- `../output/phase_binding_token_medium_tuned/`
- `../output/phase_binding_token_ema_medium/`
- `../output/phase_binding_token_order1_medium/`
- `../output/phase_binding_token_order2_medium/`
- `../output/phase_binding_token_order3_medium/`
- `../output/phase_binding_token_order3_tuned_medium/`
- `../output/phase_binding_token_order4_medium/`
- `../output/phase_binding_token_order_sweep/`
- `../output/phase_binding_token_branch_smoke/`
- `../output/phase_binding_token_branch_medium/`
- `../output/phase_binding_token_branch_w25_75_medium/`
- `../output/phase_binding_token_branch_w40_60_medium/`
- `../output/phase_binding_token_branch_w75_25_medium/`
- `../output/phase_binding_token_branch_sweep/`
- `../output/phase_binding_token_branch_calib_balanced_medium/`
- `../output/phase_binding_token_branch_calib_acc_medium/`
- `../output/phase_binding_token_branch_calibration/`
- `../output/phase_binding_token_competitive_smoke/`
- `../output/phase_binding_token_competitive_balanced_medium/`
- `../output/phase_binding_token_competitive_acc_medium/`
- `../output/phase_binding_token_competitive_k16_medium/`
- `../output/phase_binding_token_competitive_lr01_k8_medium/`
- `../output/phase_binding_token_competitive_lr05_k8_medium/`
- `../output/phase_binding_token_competitive_acc_random_medium/`
- `../output/phase_binding_token_competitive_sweep/`
- `../output/phase_binding_online_stream_smoke_fixed/`
- `../output/phase_binding_online_stream_medium_fixed/`
- `../output/phase_binding_online_stream_generation_smoke2/`
- `../output/phase_binding_online_stream_generation_medium/`
- `../output/phase_binding_online_stream_trace_smoke/`
- `../output/phase_binding_online_stream_trace_medium/`
- `../output/phase_binding_online_stream_trace_sweep_smoke/`
- `../output/phase_binding_online_stream_trace_sweep_medium/`
- `../output/phase_binding_online_stream_fatigue_smoke/`
- `../output/phase_binding_online_stream_fatigue_medium/`
- `../output/phase_binding_online_stream_inhibition_smoke/`
- `../output/phase_binding_online_stream_inhibition_sweep_smoke/`
- `../output/phase_binding_online_stream_inhibition_medium/`
- `../output/phase_binding_online_stream_inhibition_combo_smoke/`
- `../output/phase_binding_online_stream_inhibition_combo_medium/`
- `../output/phase_binding_online_stream_inhibition_combo_medium_s015_lr005_k1/`
- `../output/phase_binding_online_stream_inhibition_combo_medium_s015_lr005_k1_seed1/`
- `../output/phase_binding_online_stream_inhibition_combo_medium_s025_lr005_k2/`
- `../output/phase_binding_online_stream_gate_inhib_smoke/`
- `../output/phase_binding_online_stream_gate_inhib_sweep_smoke/`
- `../output/phase_binding_online_stream_gate_inhib_tracefix_smoke/`
- `../output/phase_binding_online_stream_gate_inhib_medium_s200_lr020_k4_decay0/`
- `../output/phase_binding_online_stream_plastic_ssm_smoke/`
- `../output/phase_binding_online_stream_plastic_ssm_sweep_smoke/`
- `../output/phase_binding_online_stream_plastic_ssm_medium/`
- `../output/phase_binding_online_stream_trace_plastic_ssm_smoke/`
- `../output/phase_binding_online_stream_elig_ssm_smoke/`
- `../output/phase_binding_online_stream_elig_ssm_sweep_low/`
- `../output/phase_binding_online_stream_elig_ssm_sweep_fixed/`
- `../output/phase_binding_online_stream_elig_ssm_sweep_clip/`
- `../output/phase_binding_online_stream_elig_ssm_medium_fixed/`
- `../output/phase_binding_online_stream_elig_ssm_medium_low/`
- `../output/phase_binding_online_stream_elig_ssm_medium_fixed_seed1/`
- `../output/phase_binding_online_stream_apical_smoke/`
- `../output/phase_binding_online_stream_apical_sweep_s015/`
- `../output/phase_binding_online_stream_apical_sweep_s030/`
- `../output/phase_binding_online_stream_apical_sweep_mneg/`
- `../output/phase_binding_online_stream_apical_medium_s015/`
- `../output/phase_binding_online_stream_apical_medium_s015_seed1/`
- `../output/phase_binding_online_stream_apical_ablate_segment_smoke/`
- `../output/phase_binding_online_stream_apical_ablate_global_smoke/`
- `../output/phase_binding_online_stream_apical_ablate_random_smoke/`
- `../output/phase_binding_online_stream_apical_ablate_fixed_smoke/`
- `../output/phase_binding_online_stream_apical_ablate_segment_medium/`
- `../output/phase_binding_online_stream_apical_ablate_global_medium/`
- `../output/phase_binding_online_stream_apical_ablate_random_medium/`
- `../output/phase_binding_online_stream_apical_ablate_fixed_medium/`
- `../output/phase_binding_online_stream_apical_ablate_global_medium_seed1/`
- `../output/phase_binding_online_stream_apical_ablate_random_medium_seed1/`
- `../output/phase_binding_online_stream_apical_random_generation_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_generation_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp16_medium/`
- `../output/phase_binding_online_stream_lowp_inst_smoke/`
- `../output/phase_binding_online_stream_lowp_row_inst_smoke/`
- `../output/phase_binding_online_stream_apical_random_lowp8_tensor_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp16_row_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_generation_medium/`
- `../output/phase_binding_online_stream_lowp_selective_plastic_smoke/`
- `../output/phase_binding_online_stream_lowp_selective_readout_smoke/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_plastic_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_readout_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_fixed_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_dynamic_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_bias_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_readout_weights_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_inhibition_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_readout_generation_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_clip8_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_tensor_clip8_medium/`
- `../output/phase_binding_online_stream_lowp_bias_clip_smoke/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_biasclip8_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_tensor_biasclip8_medium/`
- `../output/phase_binding_online_stream_lowp_targets_parse_smoke/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_all_except_bias_medium/`
- `../output/phase_binding_online_stream_lowp_phase_codes_smoke/`
- `../output/phase_binding_online_stream_lowp_phase_prototypes_smoke/`
- `../output/phase_binding_online_stream_lowp_phase_banks_smoke/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_codes_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_prototypes_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_counts_medium/`
- `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_banks_medium/`
- `../output/phase_binding_online_stream_lowp_cached_all_smoke/`
- `../output/phase_binding_online_stream_lowp_cached_all_except_bias_smoke/`
- `../output/phase_binding_online_stream_serialized_state_smoke/`
- `../output/phase_binding_online_stream_apical_random_lowp8_varstate_medium/`
- `../output/phase_binding_online_stream_sweep/`

Each result directory contains:

- `results.json`
- `results.csv`
- `summary.csv`
- `history.csv`
- `summary.png`
- `filters_bp.png`
- `filters_dfa_3factor.png`

## Main Result

Use:

- `results/full_v3/results.json`
- `results/full_v3/summary.png`
- `temporal/results/delayed_hard_v1/results.json`
- `temporal/results/delayed_hard_v1/temporal_summary.png`
- `temporal/results/bptt_tuned_v2/results.json`
- `../output/sparse_hebbian_context_medium/metrics.csv`
- `../output/sparse_hebbian_context_normalized_medium/metrics.csv`
- `../output/hybrid_llama_context_medium/metrics.csv`
- `../output/sparse_hebbian_context_medium/greedy_completions.txt`

## Latest Token-Level Result

2026-06-15 TinyStories tokenizer-level loop:

- Positive signal: `sparse_hebbian_context`
- Medium result: CE `4.1126`, PPL `61.11`, accuracy `0.361`, train speed `92k tok/s`
- Baseline in same run: `dendritic_error_1810_lite` CE `4.5777`, accuracy `0.118`
- Low-budget `torch_llama` baseline: CE `4.8748`, accuracy `0.117`
- Boundary: this does not claim GPT-level performance or replacement of full BP pretraining.

## Latest Hybrid Backoff Result

2026-06-15 linear hybrid loop:

- Tested `final_logits = neural_weight * neural_logits + memory_weight * memory_logits`
- `hybrid_llama_context` medium: CE `4.5369`, accuracy `0.357`
- `hybrid_dendritic_context` medium: CE `4.5630`, accuracy `0.342`
- Sparse memory alone remains better: CE `4.1126`, accuracy `0.361`
- Decision: abandon always-on linear fusion; next loop should implement confidence-gated adapter.

## Latest Online Stream Result

2026-06-15 strict prequential no-raw-data stream:

- New script: `../tinystories_online_stream_experiment.py`
- Medium stream result: `continuation_backoff` stream pre CE `3.395`, online CE `3.332`, post CE `1.542`, post accuracy `0.791`
- No raw text replay: only memory statistics are retained after updates.
- Capacity tradeoff: context cap `5000` compresses continuation state from ~`618KB` to ~`352KB`, with post CE `1.957`; cap `2000` gives ~`216KB`, post CE `2.252`.
- Boundary: this supports fast local adaptation, not GPT-like semantic generation.

## Latest QA Prototype Result

2026-06-15 local API-compatible personalization QA:

- New script: `../online_memory_qa_experiment.py`
- Smoke: `hashed_memory` accuracy `1.000`, state `7,102` bytes vs `base_no_memory` `0.025`; raw retrieval `0.800`
- Medium: `hashed_memory` accuracy `1.000`, state `9,276` bytes vs `base_no_memory` `0.017`; raw retrieval `0.800`
- Memory stores hashed features and answer counts only, not raw question/answer text.
- Deletion audit: medium deleted-fact accuracy drops `1.000 -> 0.200`, retained accuracy stays `1.000 -> 1.000`.
- Boundary: this is a synthetic FAQ proxy, not yet a GPT/API language-generation demo.

2026-06-15 real API smoke:

- New script: `../online_memory_qa_api_experiment.py`
- Endpoint shape verified: `https://yzhanghmeng.com/v1/chat/completions`, model `gpt-5.5`
- 10-question schema API run: `api_no_memory` accuracy `0.200` vs `api_memory_hint` accuracy `1.000`.
- Boundary: this is still schema QA, not enough for final GPT/API language-generation claim.

2026-06-15 natural FAQ API demo:

- New script: `../online_memory_faq_api_experiment.py`
- Real API run: `api_no_memory` accuracy `0.000` vs `api_memory_hint` accuracy `1.000` on 8 FAQ questions.
- API+memory generates natural answers from compressed memory hints.
- Boundary: memory stores canonical answer values; not yet strong privacy or large-scale natural dialogue evaluation.

2026-06-15 generated natural FAQ API scale check:

- `../online_memory_faq_api_experiment.py` now supports `--dataset generated`.
- Generated 64-fact dry run: local hashed FAQ memory accuracy `1.000`, state `37,755` bytes.
- Generated 256-fact dry run: local hashed FAQ memory accuracy `1.000`, state `140,372` bytes.
- Generated 64-fact real API run on 16 questions: `api_no_memory` accuracy `0.000` vs `api_memory_hint` accuracy `1.000`.
- Boundary: this is still generated FAQ fact recall; not open-domain dialogue, and memory stores answer values.

2026-06-15 dialogue-style FAQ API prototype:

- `../online_memory_faq_api_experiment.py` now supports `--train-style dialogue` and `--eval-style paraphrase`.
- Dialogue 64-fact dry run: local hashed FAQ memory accuracy `1.000`, state `55,142` bytes.
- Dialogue 256-fact dry run: local hashed FAQ memory accuracy `1.000`, state `209,512` bytes.
- Dialogue 64-fact real API run on 16 paraphrased questions: exact-term CSV has `api_memory_hint` `0.938`; semantic rescore after access-policy fix is `1.000`; `api_no_memory` remains `0.000`.
- Boundary: still schema-assisted and alias-routed; not yet multi-turn revision or learned semantic routing.

2026-06-15 FAQ revision/delete API audit:

- `../online_memory_faq_api_experiment.py` now supports `--run-revision-audit`, `--revision-limit`, and `--revision-api-limit`.
- Added memory operations: `overwrite` clears old hash rows for an intent and writes the new value; `forget` clears rows and suppresses answer hints.
- Local 64-fact revision run: overwrite correctness `1.000`, old-value leak `0.000`, delete suppression `1.000`, retained-other correctness `1.000`, final state `42,903` bytes.
- Local 256-fact revision run: same metrics all pass, final state `185,022` bytes.
- Real API revision run on 4 revised facts: API after overwrite `1.000`, API after delete suppressed `1.000`.
- Boundary: memory still stores canonical answer values; learned semantic routing and human qualitative eval remain open.

2026-06-15 semantic FAQ router API prototype:

- `../online_memory_faq_api_experiment.py` now supports `--router alias|semantic|hybrid` and `--semantic-dim`.
- Semantic router stores hashed sparse semantic prototypes, feature document-frequency counts, and tombstone prototypes; it does not store raw training text.
- Local 64-fact semantic run: accuracy `1.000`, state `212,954` bytes.
- Local 256-fact semantic run: accuracy `1.000`, state `824,791` bytes.
- Local 256-fact semantic revision audit: overwrite correctness `1.000`, old-value leak `0.000`, delete suppression `1.000`, retained-other correctness `1.000`.
- Real API semantic run: `api_no_memory` accuracy `0.000` vs `api_memory_hint` accuracy `1.000` on 8 questions; API revision overwrite/delete both `1.000` on 4 facts.
- Boundary: semantic routing is no longer alias-required, but storage is larger and memory still stores canonical answer values.

2026-06-15 semantic router compression:

- `../online_memory_faq_api_experiment.py` now supports `--semantic-feature-cap`.
- Sparse-only semantic router can be run with `--semantic-dim 0`.
- 256-fact sparse-only cap12 run: local accuracy `1.000`, state `292,592` bytes.
- 256-fact sparse-only cap12 revision run: overwrite correctness `1.000`, old-value leak `0.000`, delete suppression `1.000`, retained-other correctness `1.000`, final state `269,794` bytes.
- 64-fact sparse-only cap12 real API run: `api_no_memory` accuracy `0.000`, `api_memory_hint` accuracy `1.000`, state `79,931` bytes; revision overwrite/delete both `1.000`.
- Boundary: full answer text storage has a sketch alternative; multi-turn human qualitative eval remains open.

2026-06-15 value sketch memory:

- `../online_memory_faq_api_experiment.py` now supports `--answer-store {full,sketch}`.
- Sketch store keeps structured answer payloads instead of full answer strings.
- 256-fact sketch run: local accuracy `1.000`, state `284,248` bytes, `stores_answer_text=false`.
- 256-fact sketch revision run: overwrite correctness `1.000`, old-value leak `0.000`, delete suppression `1.000`.
- 64-fact sketch API run: `api_no_memory` accuracy `0.000`, `api_memory_hint` accuracy `1.000`, state `77,866` bytes, `stores_answer_text=false`.
- Boundary: state reduction is modest because semantic prototypes dominate; multi-turn human evaluation remains open.

2026-06-15 multi-turn FAQ API session:

- New script: `../online_memory_faq_multiturn_experiment.py`
- Session includes learn, query, revise, query, delete, query, and retained-fact query turns.
- Local session: `local_semantic_sketch_memory` accuracy `1.000` on 14 query turns, state `13,556` bytes, `stores_answer_text=false`.
- Real API session on 10 query turns: `api_no_memory` accuracy `0.200`, `api_raw_retrieval` accuracy `1.000`, `api_semantic_sketch_memory` accuracy `1.000`.
- Human-readable transcript saved at `../output/online_memory_faq_multiturn_api/session_transcript.md`.
- Boundary: this is a controlled FAQ support-session demo, not an open-domain GPT-like final evaluation.

2026-06-15 personalized style API benchmark:

- New script: `../online_memory_style_api_experiment.py`
- Task: generate customer-facing drafts while obeying online-learned style/profile constraints.
- Local style sketch memory carries all constraints with accuracy `1.000`, state `6,135` bytes, `stores_preference_text=false`.
- Real API run on 8 active prompts: `api_no_memory` all-pass accuracy `0.000`, `api_raw_profile` `1.000`, `api_style_sketch_memory` `1.000`.
- API deleted-profile suppression: `1.000` on 1 deleted query after forgetting the profile.
- Human-readable transcript saved at `../output/online_memory_style_delete_api/style_transcript.md`.
- Boundary: this is rule-scored personalized writing, not yet a broad human preference or open-domain GPT-like evaluation.

2026-06-16 personalized style judge:

- New script: `../online_memory_style_judge_experiment.py`
- Reuses existing style API outputs from `../output/online_memory_style_delete_api/session_turns.csv`.
- Blinded API judge over 8 prompts: `style_sketch_best_rate=0.500`, `no_memory_best_rate=0.500`, `raw_profile_best_rate=0.000`.
- Pairwise: `style_sketch_beats_raw_profile=0.750`, `style_sketch_beats_no_memory=0.500`.
- Boundary: style sketch controls constraints and often helps, but naturalness is not yet a clean win over no-memory.

2026-06-16 soft style hint and preference-aware judge:

- `../online_memory_style_api_experiment.py` now supports `--hint-style strict|soft`.
- `../online_memory_style_judge_experiment.py` now supports `--judge-context request_only|style_memory|raw_profile`.
- Soft style sketch API run: all-pass accuracy `1.000`, delete suppression `1.000`, state `6,155` bytes, `stores_preference_text=false`.
- Request-only blind judge on soft outputs prefers no-memory on `8/8`; this is a negative result for unconstrained naturalness.
- Preference-aware judge with learned style context: soft style sketch best `0.625` and beats no-memory `1.000`; previous strict style sketch best `0.875` and beats no-memory `1.000`.
- Boundary: personalization quality must be evaluated with learned preferences in the judge context; strict sketch currently beats soft sketch, so the next target is better preference semantics rather than weaker constraints.

2026-06-16 compositional cue and phase-binding Hebbian:

- New script: `../compositional_cue_experiment.py`
- Task: `C_a, F, C_b, ..., Q -> T_((a+b) mod K)` with held-out cue pairs.
- Pure no-BP candidate: locally learned cue-to-phase code, complex phase binding, and target-gated Hebbian prototype readout.
- Result on K=4/8, seeds 0/1/2: `learned_phase_binding_hebbian` and hand-coded `phase_binding_hebbian` held-out accuracy `1.000` for both K values.
- Controls fail on held-out pairs: pair lookup `0.417/0.146`, reservoir `0.000/0.021`, eprop `0.000/0.021`, tuned BPTT `0.000/0.000`.
- Boundary: this is a structured compositional toy, not a language model. It is a better pure no-BP direction than API/LLM adapters or token-statistical backoff.

2026-06-16 target-only phase factorization:

- `../compositional_cue_experiment.py` now includes `target_only_phase_binding_hebbian` and a `--methods` filter.
- This variant removes direct cue->phase labels and uses only the final target class as the teaching signal.
- Result on K=4/8/12, seeds 0/1/2: target-only phase binding held-out accuracy `1.000/1.000/1.000`.
- Controls: pair lookup `0.417/0.146/0.093`; scrambled phase control `0.500/0.188/0.213`.
- Boundary: stronger pure no-BP mechanism evidence, but still a controlled composition task; next step is connecting this binding state to a pure no-BP token learner.

2026-06-16 phase-binding token learner:

- New script: `../phase_binding_token_experiment.py`
- Main method: target-only local phase binding over configurable recent tokenizer IDs, with Hebbian prototype readout and optional fixed dendritic branch sum.
- Medium tuned run, TinyStories 50k/10k chars, vocab 256, eval 5000: `phase_binding_token` CE `3.551`, acc `0.284`.
- EMA readout variant: CE `3.614`, acc `0.298`.
- Ablations/baselines: no-bias phase CE `4.204`, unigram CE `4.652`, `sparse_context_aux` CE `3.833`, acc `0.323`.
- Context-order sweep: order=2 is best phase point; order=1 CE `3.603`, order=3 tuned CE `4.199`, order=4 CE `10.462`.
- Branched phase state: order1/order2 weights `0.5/0.5` gives CE `3.282`, acc `0.295`; weights `0.25/0.75` gives best acc `0.304`, CE `3.350`.
- Readout calibration: balanced CE `3.285`, acc `0.306`; max-acc CE `3.599`, acc `0.312`.
- Competitive WTA readout: neg_k=8 lr=0.02 gives CE `3.230`, acc `0.322`; lr=0.05 gives best CE `3.195`, acc `0.320`.
- Boundary: strongest pure no-BP token learner so far; top-1 is almost tied with sparse context (`0.322` vs `0.323`) and CE is much better. Next work should feed competition back into branch state or run a strict online no-raw stream evaluation.

2026-06-16 phase-binding online stream:

- New script: `../phase_binding_online_stream_experiment.py`
- Main method: strict prequential `phase_competitive_online`, using branch phase state plus local WTA readout; no BP/BPTT/API/pretrained model and no raw-text replay.
- Medium fixed run, TinyStories 50k/10k chars, vocab 256: stream pre CE `3.425`, acc `0.303`; stream online CE `3.229`, acc `0.335`; stream post CE `2.427`, acc `0.422`.
- Retention after online stream: CE `3.272`, acc `0.326`; state `2,367,488` bytes.
- Generation audit on 4 prompts x 48 tokens: phase/WTA first-token match improves `0.500 -> 0.750`, but greedy post `repeat-2=0.771`; controlled decoding reduces repeat-2 to `0.170` and raises distinct-2 to `0.830`, while decoded samples remain semantically weak.
- Auxiliary statistical baseline: `sparse_context_aux` stream post CE `1.297`, acc `0.571`, state `175,568` bytes. It remains a cache/count baseline only, not the final method.
- Boundary: first strong strict-online result for the pure phase/WTA no-BP token learner, plus a clear generation-quality failure mode. Next work is recurrent/SSM branch dynamics, low-precision/sparse state audit, and seed repeats.

2026-06-16 phase trace branch:

- `../phase_binding_online_stream_experiment.py` now supports `--trace-branch`.
- Main new method: `phase_trace_competitive_online`, adding a fixed leaky token-trace / SSM-like branch to phase/WTA.
- Best medium config: `trace_order=16`, `trace_dim=64`, `trace_weight=0.5`, `trace_decay=0.85`.
- Medium result: online CE `3.157`, acc `0.335`; post CE `2.389`, acc `0.427`, state `2,498,560` bytes.
- Baseline phase/WTA in same run: online CE `3.229`, acc `0.335`; post CE `2.427`, acc `0.422`.
- Generation audit: greedy repeat-2 improves `0.771 -> 0.681`, distinct-2 `0.229 -> 0.319`, but text still loops and remains far from GPT/API quality.
- Boundary: trace is a useful pure no-BP state improvement, not a full solution; next step is local inhibitory/fatigue dynamics or plastic recurrent branch.

2026-06-16 output fatigue dynamics:

- `../phase_binding_online_stream_experiment.py` now supports `--output-fatigue`.
- Main new method: `phase_trace_fatigue_competitive_online`, adding local output-neuron fatigue over phase+trace WTA.
- Medium config: `trace_order=16`, `trace_dim=64`, `trace_weight=0.5`, `trace_decay=0.85`, `fatigue_strength=0.75`, `fatigue_decay=0.80`.
- Medium result: online CE `3.141`, acc `0.339`; post CE `2.382`, acc `0.429`; state `2,499,584` bytes.
- Baselines in same run: phase/WTA post CE `2.427`, acc `0.422`; trace/WTA post CE `2.389`, acc `0.427`.
- Generation audit: greedy repeat-2 improves from phase baseline `0.771` to `0.606`, distinct-2 `0.229 -> 0.394`; text still loops and is far from GPT/API quality.
- Boundary: current best pure no-BP online token point, but still only a step toward usable generation.

2026-06-16 adaptive output inhibition:

- `../phase_binding_online_stream_experiment.py` now supports `--adaptive-inhibition`.
- Main new method: `phase_trace_inhib_competitive_online`, adding a locally plastic output inhibition matrix over a recent output trace; optional composition with `OutputFatigueMemory`.
- Best medium config: `trace_order=16`, `trace_dim=64`, `trace_weight=0.5`, `trace_decay=0.85`, `inhibit_strength=0.15`, `inhibit_lr=0.005`, `inhibit_top_k=1`.
- Seed 0: trace+inhibition post CE `2.358`, acc `0.429`; trace+fatigue+inhibition post CE `2.363`, acc `0.432`.
- Seed 1 repeat: trace+inhibition improves trace-only CE `2.450 -> 2.410`, acc `0.420 -> 0.431`; trace+fatigue+inhibition reaches CE `2.414`, acc `0.434`.
- Generation audit is mixed: seed 0 trace+fatigue+inhibition greedy repeat-2 `0.436` vs trace+fatigue `0.606`, but seed 1 fatigue alone is better for repetition. This is a stable CE/acc improvement, not a solved generation method.
- Boundary: new strongest pure no-BP online token CE point; next step is context-gated inhibition or plastic recurrent/SSM state.

2026-06-16 context-gated output inhibition:

- `../phase_binding_online_stream_experiment.py` now supports `--context-gated-inhibition`.
- Main new method: `phase_trace_fatigue_gate_inhib_competitive_online`, using fixed random context gates to localize anti-winner inhibition.
- Implementation note: the first dynamic-gate draft updated `dynamic_gate` without using it in scores; this was fixed via `effective_gate(context)`. After the fix, `gate_decay=0.8` was worse than `0.0`, so medium uses pure context gating.
- Medium config: `gate_dim=64`, `gate_strength=2.0`, `gate_lr=0.02`, `gate_top_k=4`, `gate_decay=0.0`.
- Medium result: trace+fatigue+gate online CE `3.178`, acc `0.331`; post CE `2.416`, acc `0.435`; greedy repeat-2 `0.346`; state `3,613,952` bytes.
- Baseline in same run: trace+fatigue post CE `2.382`, acc `0.429`, greedy repeat-2 `0.606`, state `2,499,584` bytes.
- Boundary: context gate improves top-1 and repetition but worsens CE and costs memory; it is a useful tradeoff, not the current best CE model. Next step is a plastic recurrent/SSM branch.

2026-06-16 plastic recurrent/SSM branch:

- `../phase_binding_online_stream_experiment.py` now supports `--plastic-ssm-branch`.
- Main new methods: `phase_plastic_ssm_competitive_online` and `phase_trace_plastic_ssm_competitive_online`, using target-modulated local Hebbian/Oja transition updates plus WTA readout.
- Medium config: `ssm_order=16`, `ssm_dim=64`, `ssm_decay=0.80`, `ssm_recurrent_scale=0.40`, `ssm_weight=0.5`, `ssm_lr=0.01`, `ssm_target_mix=0.25`.
- Medium result: plastic SSM+inhibition online CE `3.129`, acc `0.333`; post CE `2.377`, acc `0.425`; greedy repeat-2 `0.590`; state `2,843,648` bytes.
- Baseline in same run: trace+inhibition post CE `2.358`, acc `0.429`; trace+fatigue+inhibition post CE `2.363`, acc `0.432`.
- Trace+plastic SSM combo smoke also failed to beat trace+fatigue: post CE `2.041`, acc `0.490` vs trace+fatigue `2.030`, acc `0.492`.
- Boundary: current Oja-style plastic transition is not the right recurrent credit assignment for token modeling. Next recurrent attempt should use eligibility-gated transition writes or dendritic/apical gating.

2026-06-17 eligibility-gated recurrent/SSM branch:

- `../phase_binding_online_stream_experiment.py` now supports `--eligibility-ssm-branch`, `--ssm-eligibility-decay`, `--ssm-eligibility-clip`, and `--method-filter`.
- Main new methods: `phase_elig_ssm_competitive_online` and `phase_trace_elig_ssm_competitive_online`, using decaying eligibility traces to gate local recurrent transition writes.
- Smoke found the best signal in fixed/very-low-write settings: `ssm_weight=0.2`, `ssm_lr=0.0` or `0.003`, suggesting the benefit is a fixed reservoir feature rather than successful transition credit assignment.
- Seed 0 fixed reservoir+inhibition: online CE `3.109`, acc `0.335`; post CE `2.355`, acc `0.431`; greedy repeat-2 `0.527`; state `2,991,104` bytes.
- Seed 0 baseline trace+inhibition: post CE `2.358`, acc `0.429`; trace+fatigue+inhibition: post CE `2.363`, acc `0.432`, greedy repeat-2 `0.436`.
- Seed 1 fixed reservoir+inhibition did not reproduce the CE gain: post CE `2.412`, acc `0.430` vs trace+inhibition `2.410`, acc `0.431`.
- Boundary: eligibility-gated transition writes are neutral/negative in this form. A fixed reservoir branch is optional but not a new best. Next step should be dendritic/apical local prediction-error gating over phase/trace/reservoir features.

2026-06-17 dendritic/apical local error gating:

- `../phase_binding_online_stream_experiment.py` now supports `--apical-gating-branch` with branch-wise target-vs-wrong local prediction-error traces.
- Main new method: `phase_trace_apical_inhib_competitive_online`, using weak apical gates over phase/trace feature segments plus local output inhibition.
- Strong gate default was negative: `apical_strength=0.75`, `max_gate=2.0` hurt post acc on smoke.
- Best config uses weak gates: `apical_decay=0.85`, `apical_strength=0.15`, `apical_min_gate=0.8`, `apical_max_gate=1.25`, `apical_error_clip=1.0`.
- Seed 0: trace+apical+inhibition online CE `3.181`, acc `0.336`; post CE `2.294`, acc `0.435`; greedy repeat-2 `0.383`; state `2,761,740` bytes.
- Seed 0 baseline trace+inhibition: post CE `2.358`, acc `0.429`, repeat-2 `0.473`. Trace+fatigue+inhibition: post CE `2.363`, acc `0.432`, repeat-2 `0.436`.
- Seed 1 repeat: trace+apical+inhibition post CE `2.350`, acc `0.431` vs trace+inhibition post CE `2.410`, acc `0.431`.
- Apical+fatigue+inhibition gives the best top-1: seed0 post acc `0.440`, seed1 post acc `0.436`.
- Boundary: text generation still loops and statistical `sparse_context_aux` remains much stronger as an auxiliary cache baseline, but R051 is the strongest pure no-BP neural/dendritic token learner result so far.

2026-06-17 apical error gate ablation:

- `../phase_binding_online_stream_experiment.py` now supports `--apical-error-mode {segment_margin,global_margin,random_feedback,fixed_random}`.
- Ablation tests whether R051 requires branch-local segment margins or merely a dynamic target-vs-wrong apical error signal.
- Smoke post CE/acc: segment `1.965/0.513`, global `1.955/0.513`, random feedback `1.955/0.510`, fixed random `1.999/0.492`.
- Seed 0 medium: trace+inhib post CE `2.358`; segment `2.294`, global `2.290`, random feedback `2.289`, fixed random `2.316`.
- Seed 1 repeat: trace+inhib post CE `2.410`; global `2.346`, random feedback `2.345`.
- Conclusion: dynamic target-vs-wrong apical error is the key signal; branch-local margin is not required in this setup. Fixed random gate explains only part of the gain. Random feedback matching global margin supports a feedback-alignment-style no-BP apical error pathway.

2026-06-17 low-precision and generation audit for random-feedback apical error:

- `../phase_binding_online_stream_experiment.py` now supports `--low-precision-bits` and `--low-precision-clip` through `LowPrecisionStateWrapper`.
- Full precision random-feedback apical+inhibition keeps the generation benefit: post CE `2.289`, acc `0.437`, greedy repeat-2 `0.383`, controlled repeat-2 `0.106`.
- Baseline trace+inhibition in the same generation run: post CE `2.358`, acc `0.429`, greedy repeat-2 `0.473`, controlled repeat-2 `0.144`.
- 8-bit audit: reported state `690,438` bytes for apical+inhib; post CE `2.713`, acc `0.431`, greedy repeat-2 `0.404`. This compresses state but hurts CE calibration.
- 16-bit audit: reported state `1,380,876` bytes for apical+inhib; post CE `2.504`, acc `0.460`. Accuracy is high, but CE remains worse than full precision.
- Boundary: the wrapper simulates quantization in float arrays, so pickle size is unchanged. True deployment needs serialized low-precision arrays and quantization-aware update/per-state scaling.

2026-06-17 quantization-aware scaling for random-feedback apical error:

- `../phase_binding_online_stream_experiment.py` now supports `--low-precision-scale-mode {fixed,tensor,row}`.
- Per-state scaling substantially improves the 8-bit audit while preserving the dynamic apical advantage. Apical+inhibition post CE: fixed `2.713`, tensor `2.550`, row `2.523`; trace+inhibition 8-bit row is `2.651`.
- 16-bit row gives apical+inhibition CE `2.504`, acc `0.461`, reported state `1,380,876` bytes; this is similar to prior 16-bit fixed and still below full precision CE `2.289`.
- Best 8-bit row generation restores the greedy repetition benefit: trace+inhibition repeat-2 `0.569` vs apical+inhibition `0.388`.
- Boundary: this is still a projection audit, not true serialized integer storage. Next work should target specific matrices/prototypes with row scales and align local learning-rate steps to quantization bins.

2026-06-17 selective quantization audit for random-feedback apical error:

- `../phase_binding_online_stream_experiment.py` now supports `--low-precision-targets` with groups such as `plastic`, `readout`, `phase`, `fixed`, `dynamic`, `bias`, `readout_weights`, and `inhibition`.
- Mixed-precision state byte estimates now count quantized and unquantized arrays separately.
- Main localization: `output_bias`/log-prior quantization reproduces most of the 8-bit CE damage. Apical+inhibition post CE: full `2.289`, all-row `2.523`, bias-only `2.513`.
- WTA/readout weights and inhibition tolerate 8-bit row quantization: apical+inhibition readout-weight-only CE `2.295`, inhibition-only CE `2.289`.
- Fixed random codes and short-term dynamic traces are not the CE bottleneck: fixed-only CE `2.289`, dynamic-only CE `2.289` for apical+inhibition.
- Wider `clip=8.0` for all-state tensor/row did not restore CE (`2.570`/`2.539`), so the next step is variable-type-aware quantization, not a single global clip.
- Dedicated `--low-precision-bias-clip 8.0` also did not restore all-state CE: apical+inhibition row `2.525`, tensor `2.555`. This confirms that bias-only damage is real, but all-state every-step projection has additional cumulative perturbation.
- Phase-side split shows the same pattern: phase code banks and phase prototypes tolerate 8-bit row (`2.290` and `2.289` CE for apical+inhibition), while count-like arrays reproduce the damage (`counts` CE `2.513`, `phase_banks` CE `2.513`).
- `LowPrecisionStateWrapper` now caches target array references after initialization, so projection no longer recursively scans the object graph after every update.

2026-06-17 serialized variable-type state for random-feedback apical error:

- `../phase_binding_online_stream_experiment.py` now reports `serialized_state_bytes`, `serialized_bytes_per_target`, and a per-method `serialized_state_manifest.json` when the low-precision wrapper is active.
- Variable-type 8-bit row keeps vector/matrix state quantized and leaves `output_bias`, `prototype_counts`, and `unigram_counts` as float32.
- Apical+inhibition medium result: post CE `2.295`, acc `0.438`, state bytes `696,585`, serialized bytes `706,841`, pickle bytes `2,765,072`.
- This recovers most of the all-state 8-bit row CE gap (`2.523`) and is close to full precision (`2.289`), while giving an approximately 4x smaller deployable state estimate.
- Boundary: this is still an export/accounting manifest; next step is a loadable integer checkpoint and prediction parity test.

## Planning Updates

| Timestamp | Skill | File | Stage | Description |
|---|---|---|---|---|
| 2026-06-15 13:46 | /experiment-plan | `refine-logs/EXPERIMENT_PLAN_20260615_134637.md` | implementation | current evidence-based plan: sparse Hebbian online memory plus confidence-gated adapter |
| 2026-06-15 13:46 | /experiment-plan | `refine-logs/EXPERIMENT_PLAN.md` | implementation | latest experiment plan copy |
| 2026-06-15 13:46 | /experiment-plan | `refine-logs/EXPERIMENT_TRACKER_20260615_134637.md` | implementation | execution tracker for gated adapter, online no-raw-data test, and API-compatible prototype |
| 2026-06-15 13:46 | /experiment-plan | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | latest experiment tracker copy |
| 2026-06-15 15:24 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_GATED_CONTEXT.md` | implementation | confidence-gated adapter and controlled decoding results |
| 2026-06-15 15:24 | /experiment-bridge | `../output/gated_context_decode_medium/metrics.csv` | implementation | medium token metrics for gated/context decoder |
| 2026-06-15 15:24 | /experiment-bridge | `../output/gated_context_decode_medium/generation_metrics.csv` | implementation | generation repetition/diversity metrics |
| 2026-06-15 15:30 | /experiment-plan | `refine-logs/AUTO_EXPLORATION_STRATEGY_20260615_153046.md` | implementation | persistent autonomous exploration strategy and stop/go rules |
| 2026-06-15 15:30 | /experiment-plan | `refine-logs/AUTO_EXPLORATION_STRATEGY.md` | implementation | latest autonomous exploration strategy |
| 2026-06-15 15:35 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_CONTEXT_ORDER.md` | implementation | context order ablation; order=3 improves CE to 4.0796 |
| 2026-06-15 15:35 | /experiment-bridge | `../output/context_order_ablation/summary.csv` | implementation | context order ablation summary |
| 2026-06-15 15:44 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_SEMANTIC_MEMORY.md` | implementation | semantic-key Hebbian memory and combined-context sweep |
| 2026-06-15 15:44 | /experiment-bridge | `../output/semantic_memory_medium/metrics.csv` | implementation | semantic-key medium metrics |
| 2026-06-15 15:44 | /experiment-bridge | `../output/semantic_memory_sweep/summary.csv` | implementation | semantic-key sweep summary |
| 2026-06-15 17:14 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_CONTINUATION_BACKOFF.md` | implementation | continuation/Kneser-Ney-style no-BP memory; new best CE 3.3254 |
| 2026-06-15 17:14 | /experiment-bridge | `../output/continuation_medium/metrics.csv` | implementation | continuation medium metrics |
| 2026-06-15 17:14 | /experiment-bridge | `../output/continuation_sweep/summary.csv` | implementation | continuation parameter sweep |
| 2026-06-15 18:10 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_ONLINE_STREAM.md` | implementation | strict prequential no-raw-data online stream; continuation post CE 1.542 |
| 2026-06-15 18:10 | /experiment-bridge | `../output/online_stream_medium/summary.csv` | implementation | online stream medium summary |
| 2026-06-15 18:35 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_STREAM_PRUNING.md` | implementation | context-cap pruning tradeoff for online memory |
| 2026-06-15 18:35 | /experiment-bridge | `../output/online_stream_cap5000/summary.csv` | implementation | capacity cap 5000 summary |
| 2026-06-15 18:35 | /experiment-bridge | `../output/online_stream_cap2000/summary.csv` | implementation | capacity cap 2000 summary |
| 2026-06-15 19:10 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_QA_API_PROTOTYPE.md` | implementation | local API-compatible personalization QA prototype |
| 2026-06-15 19:10 | /experiment-bridge | `../output/online_memory_qa_smoke/summary.csv` | implementation | QA smoke summary |
| 2026-06-15 19:10 | /experiment-bridge | `../output/online_memory_qa_medium/summary.csv` | implementation | QA medium summary |
| 2026-06-15 19:45 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_QA_API_REAL.md` | implementation | real API smoke for hashed-memory hint adapter |
| 2026-06-15 19:45 | /experiment-bridge | `../output/online_memory_qa_api_run10/summary.csv` | implementation | real API 10-question schema QA summary |
| 2026-06-15 20:15 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_NATURAL_FAQ_API.md` | implementation | natural FAQ API demo with memory hints |
| 2026-06-15 20:15 | /experiment-bridge | `../output/online_memory_faq_api_run/summary.csv` | implementation | real API 8-question natural FAQ summary |
| 2026-06-15 19:53 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_GENERATED_FAQ_API.md` | implementation | generated natural FAQ API scale check |
| 2026-06-15 19:53 | /experiment-bridge | `../output/online_memory_faq_generated_api_run/summary.csv` | implementation | generated FAQ real API 16-question summary |
| 2026-06-15 19:53 | /experiment-bridge | `../output/online_memory_faq_generated_256_dry/summary.csv` | implementation | generated FAQ 256-fact local scale summary |
| 2026-06-15 19:53 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_DIALOGUE_FAQ_API.md` | implementation | dialogue-style FAQ API prototype with paraphrase eval |
| 2026-06-15 19:53 | /experiment-bridge | `../output/online_memory_faq_dialogue_api_run/summary.csv` | implementation | dialogue FAQ real API 16-question exact-term summary |
| 2026-06-15 19:53 | /experiment-bridge | `../output/online_memory_faq_dialogue_256_dry/summary.csv` | implementation | dialogue FAQ 256-fact local scale summary |
| 2026-06-15 20:05 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_FAQ_REVISION_API.md` | implementation | FAQ overwrite/delete audit for online memory state |
| 2026-06-15 20:05 | /experiment-bridge | `../output/online_memory_faq_revision_api_run/revision_summary.csv` | implementation | real API overwrite/delete 4-fact revision summary |
| 2026-06-15 20:05 | /experiment-bridge | `../output/online_memory_faq_revision_256_dry/revision_summary.csv` | implementation | local 256-fact revision/delete summary |
| 2026-06-15 20:25 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_SEMANTIC_FAQ_ROUTER_API.md` | implementation | learned hashed semantic router for FAQ memory |
| 2026-06-15 20:25 | /experiment-bridge | `../output/online_memory_faq_semantic_api_run/summary.csv` | implementation | semantic router real API 8-question summary |
| 2026-06-15 20:25 | /experiment-bridge | `../output/online_memory_faq_semantic_revision_256_dry/revision_summary.csv` | implementation | semantic router 256-fact overwrite/delete summary |
| 2026-06-15 21:48 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_SEMANTIC_ROUTER_COMPRESSION.md` | implementation | sparse-only semantic router compression |
| 2026-06-15 21:48 | /experiment-bridge | `../output/online_memory_faq_semantic_sparseonly_cap12_256_dry/summary.csv` | implementation | sparse-only cap12 semantic router 256-fact summary |
| 2026-06-15 21:48 | /experiment-bridge | `../output/online_memory_faq_semantic_sparseonly_cap12_revision_256_dry/revision_summary.csv` | implementation | sparse-only cap12 semantic revision/delete summary |
| 2026-06-15 21:55 | /experiment-bridge | `../output/online_memory_faq_semantic_sparseonly_cap12_api_run/summary.csv` | implementation | sparse-only cap12 semantic router real API summary |
| 2026-06-15 22:12 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_VALUE_SKETCH_API.md` | implementation | structured answer sketch memory for FAQ prototype |
| 2026-06-15 22:12 | /experiment-bridge | `../output/online_memory_faq_value_sketch_api_run/summary.csv` | implementation | value sketch real API summary |
| 2026-06-15 22:12 | /experiment-bridge | `../output/online_memory_faq_value_sketch_revision_256_dry/revision_summary.csv` | implementation | value sketch 256-fact revision/delete summary |
| 2026-06-15 22:18 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_MULTITURN_FAQ_API.md` | implementation | multi-turn FAQ support-session demo |
| 2026-06-15 22:18 | /experiment-bridge | `../output/online_memory_faq_multiturn_dry/summary.csv` | implementation | multi-turn FAQ local summary |
| 2026-06-15 22:18 | /experiment-bridge | `../output/online_memory_faq_multiturn_api/summary.csv` | implementation | multi-turn FAQ API summary |
| 2026-06-15 22:45 | /experiment-bridge | `llm-token/ITERATION_2026-06-15_STYLE_API.md` | implementation | personalized style API generation benchmark |
| 2026-06-15 22:45 | /experiment-bridge | `../output/online_memory_style_dry/summary.csv` | implementation | local style sketch memory summary |
| 2026-06-15 22:45 | /experiment-bridge | `../output/online_memory_style_api/summary.csv` | implementation | style API generation summary |
| 2026-06-15 23:02 | /experiment-bridge | `../output/online_memory_style_delete_dry/summary.csv` | implementation | local style delete suppression summary |
| 2026-06-15 23:02 | /experiment-bridge | `../output/online_memory_style_delete_api/summary.csv` | implementation | style API delete suppression summary |
| 2026-06-16 00:15 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_STYLE_JUDGE.md` | implementation | personalized style preference judge |
| 2026-06-16 00:15 | /experiment-bridge | `../output/online_memory_style_judge_api_fixed/judge_summary.csv` | implementation | corrected style judge summary |
| 2026-06-16 01:05 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_STYLE_SOFT_HINT.md` | implementation | soft style hint plus preference-aware judge |
| 2026-06-16 01:05 | /experiment-bridge | `../output/online_memory_style_soft_api/summary.csv` | implementation | soft style hint API generation summary |
| 2026-06-16 01:05 | /experiment-bridge | `../output/online_memory_style_soft_judge_api/judge_summary.csv` | implementation | request-only judge on soft style outputs |
| 2026-06-16 01:05 | /experiment-bridge | `../output/online_memory_style_soft_judge_context_api/judge_summary.csv` | implementation | preference-aware judge on soft style outputs |
| 2026-06-16 01:05 | /experiment-bridge | `../output/online_memory_style_strict_judge_context_api/judge_summary.csv` | implementation | preference-aware judge on strict style outputs |
| 2026-06-16 01:45 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_COMPOSITIONAL_CUE.md` | implementation | pure no-BP compositional cue task with phase-binding Hebbian result |
| 2026-06-16 01:45 | /experiment-bridge | `../output/compositional_cue_phase_r007/summary.csv` | implementation | R007 K=4/8 held-out compositional cue summary |
| 2026-06-16 02:05 | /experiment-bridge | `../output/compositional_cue_learned_phase_r007/summary.csv` | implementation | R007 learned cue-to-phase binding summary |
| 2026-06-16 02:30 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_TARGET_ONLY_PHASE.md` | implementation | target-only local phase factorization without cue phase labels |
| 2026-06-16 02:30 | /experiment-bridge | `../output/compositional_cue_targetonly_r008/summary.csv` | implementation | K=4/8/12 target-only phase held-out summary |
| 2026-06-16 03:05 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_PHASE_TOKEN_LEARNER.md` | implementation | pure phase-binding TinyStories token learner result |
| 2026-06-16 03:05 | /experiment-bridge | `../output/phase_binding_token_medium_tuned/metrics.csv` | implementation | medium tuned phase-binding token learner metrics |
| 2026-06-16 03:20 | /experiment-bridge | `../output/phase_binding_token_ema_medium/metrics.csv` | implementation | EMA readout variant for phase-binding token learner |
| 2026-06-16 03:55 | /experiment-bridge | `../output/phase_binding_token_order_sweep/summary.csv` | implementation | context-order sweep for phase-binding token learner |
| 2026-06-16 04:20 | /experiment-bridge | `../output/phase_binding_token_branch_sweep/summary.csv` | implementation | branched phase token learner sweep |
| 2026-06-16 04:40 | /experiment-bridge | `../output/phase_binding_token_branch_calibration/summary.csv` | implementation | branch readout calibration summary |
| 2026-06-16 05:05 | /experiment-bridge | `../output/phase_binding_token_competitive_sweep/summary.csv` | implementation | competitive WTA branch readout summary |
| 2026-06-16 05:45 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_PHASE_ONLINE_STREAM.md` | implementation | strict online stream for pure phase/WTA token learner |
| 2026-06-16 05:45 | /experiment-bridge | `../output/phase_binding_online_stream_medium_fixed/summary.csv` | implementation | phase/WTA online stream medium summary |
| 2026-06-16 06:20 | /experiment-bridge | `../output/phase_binding_online_stream_generation_medium/generation_summary.csv` | implementation | phase/WTA online generation and repetition audit |
| 2026-06-16 06:55 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_PHASE_TRACE_BRANCH.md` | implementation | leaky trace branch for pure phase/WTA online model |
| 2026-06-16 06:55 | /experiment-bridge | `../output/phase_binding_online_stream_trace_sweep_medium/w050_o16_d085/summary.csv` | implementation | best trace branch medium stream summary |
| 2026-06-16 07:25 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_OUTPUT_FATIGUE.md` | implementation | output-neuron fatigue dynamics for pure phase/trace online model |
| 2026-06-16 07:25 | /experiment-bridge | `../output/phase_binding_online_stream_fatigue_medium/summary.csv` | implementation | phase/trace/fatigue medium stream summary |
| 2026-06-16 08:15 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_ADAPTIVE_INHIBITION.md` | implementation | locally plastic output inhibition for pure phase/trace online model |
| 2026-06-16 08:15 | /experiment-bridge | `../output/phase_binding_online_stream_inhibition_combo_medium_s015_lr005_k1/summary.csv` | implementation | best adaptive inhibition seed0 medium stream summary |
| 2026-06-16 08:15 | /experiment-bridge | `../output/phase_binding_online_stream_inhibition_combo_medium_s015_lr005_k1_seed1/summary.csv` | implementation | adaptive inhibition seed1 repeat check |
| 2026-06-16 09:00 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_CONTEXT_GATED_INHIBITION.md` | implementation | context-gated output inhibition tradeoff for pure phase/trace online model |
| 2026-06-16 09:00 | /experiment-bridge | `../output/phase_binding_online_stream_gate_inhib_medium_s200_lr020_k4_decay0/summary.csv` | implementation | context-gated inhibition medium stream summary |
| 2026-06-16 09:45 | /experiment-bridge | `llm-token/ITERATION_2026-06-16_PLASTIC_SSM_BRANCH.md` | implementation | plastic recurrent/SSM branch boundary result |
| 2026-06-16 09:45 | /experiment-bridge | `../output/phase_binding_online_stream_plastic_ssm_medium/summary.csv` | implementation | plastic recurrent/SSM medium stream summary |
| 2026-06-16 09:45 | /experiment-bridge | `../output/phase_binding_online_stream_trace_plastic_ssm_smoke/summary.csv` | implementation | trace + plastic SSM combo smoke summary |
| 2026-06-17 00:47 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_ELIGIBILITY_SSM_BRANCH.md` | implementation | eligibility-gated recurrent/SSM branch boundary result |
| 2026-06-17 00:47 | /experiment-bridge | `../output/phase_binding_online_stream_elig_ssm_smoke/summary.csv` | implementation | eligibility SSM initial smoke summary |
| 2026-06-17 00:47 | /experiment-bridge | `../output/phase_binding_online_stream_elig_ssm_medium_fixed/summary.csv` | implementation | fixed reservoir medium seed0 summary |
| 2026-06-17 00:47 | /experiment-bridge | `../output/phase_binding_online_stream_elig_ssm_medium_low/summary.csv` | implementation | weak eligibility-write medium seed0 summary |
| 2026-06-17 00:47 | /experiment-bridge | `../output/phase_binding_online_stream_elig_ssm_medium_fixed_seed1/summary.csv` | implementation | fixed reservoir seed1 repeat check |
| 2026-06-17 00:59 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_APICAL_GATING.md` | implementation | dendritic/apical local error gating positive result |
| 2026-06-17 00:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_sweep_s015/summary.csv` | implementation | low-strength apical gate smoke sweep |
| 2026-06-17 00:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_medium_s015/summary.csv` | implementation | apical gating medium seed0 summary |
| 2026-06-17 00:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_medium_s015/generation_summary.csv` | implementation | apical gating seed0 generation/repetition summary |
| 2026-06-17 00:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_medium_s015_seed1/summary.csv` | implementation | apical gating seed1 repeat check |
| 2026-06-17 01:50 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_APICAL_ABLATION.md` | implementation | apical error gate ablation and mechanism refinement |
| 2026-06-17 01:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_ablate_segment_medium/summary.csv` | implementation | branch-local apical error medium summary |
| 2026-06-17 01:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_ablate_global_medium/summary.csv` | implementation | global-margin apical error medium summary |
| 2026-06-17 01:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_ablate_random_medium/summary.csv` | implementation | random-feedback apical error medium summary |
| 2026-06-17 01:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_ablate_fixed_medium/summary.csv` | implementation | fixed-random gate control medium summary |
| 2026-06-17 01:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_ablate_global_medium_seed1/summary.csv` | implementation | global-margin apical error seed1 repeat |
| 2026-06-17 01:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_ablate_random_medium_seed1/summary.csv` | implementation | random-feedback apical error seed1 repeat |
| 2026-06-17 01:59 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_LOWP_APICAL_AUDIT.md` | implementation | low-precision and generation audit for random-feedback apical error |
| 2026-06-17 01:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_generation_medium/summary.csv` | implementation | random-feedback apical generation medium summary |
| 2026-06-17 01:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_generation_medium/generation_summary.csv` | implementation | random-feedback apical repetition metrics |
| 2026-06-17 01:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_generation_medium/summary.csv` | implementation | 8-bit low-precision apical medium summary |
| 2026-06-17 01:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_generation_medium/generation_summary.csv` | implementation | 8-bit low-precision generation metrics |
| 2026-06-17 01:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp16_medium/summary.csv` | implementation | 16-bit low-precision apical medium summary |
| 2026-06-17 02:19 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_QUANT_AWARE_APICAL.md` | implementation | quantization-aware per-tensor/per-row scaling audit for random-feedback apical error |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_lowp_row_inst_smoke/summary.csv` | implementation | low-precision row scaling instantiation smoke summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_tensor_medium/summary.csv` | implementation | 8-bit tensor-scaled low-precision apical medium summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_medium/summary.csv` | implementation | 8-bit row-scaled low-precision apical medium summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp16_row_medium/summary.csv` | implementation | 16-bit row-scaled low-precision apical medium summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_generation_medium/generation_summary.csv` | implementation | 8-bit row-scaled apical generation metrics |
| 2026-06-17 02:19 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_SELECTIVE_QUANT_APICAL.md` | implementation | selective quantization localization for random-feedback apical state |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_bias_medium/summary.csv` | implementation | bias-only 8-bit row quantization summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_readout_weights_medium/summary.csv` | implementation | readout-weight-only 8-bit row quantization summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_inhibition_medium/summary.csv` | implementation | inhibition-only 8-bit row quantization summary |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_readout_generation_medium/generation_summary.csv` | implementation | readout-targeted 8-bit row generation metrics |
| 2026-06-17 02:19 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_clip8_medium/summary.csv` | implementation | all-state 8-bit row wider-clip sanity summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_lowp_bias_clip_smoke/summary.csv` | implementation | dedicated output-bias clip smoke summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_biasclip8_medium/summary.csv` | implementation | all-state 8-bit row with dedicated bias clip summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_tensor_biasclip8_medium/summary.csv` | implementation | all-state 8-bit tensor with dedicated bias clip summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_counts_medium/summary.csv` | implementation | count-like state 8-bit row quantization summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_codes_medium/summary.csv` | implementation | phase code bank 8-bit row quantization summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_prototypes_medium/summary.csv` | implementation | phase prototype 8-bit row quantization summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_row_phase_banks_medium/summary.csv` | implementation | phase bank 8-bit row quantization summary |
| 2026-06-17 02:58 | /experiment-bridge | `../output/phase_binding_online_stream_lowp_cached_all_smoke/summary.csv` | implementation | cached low-precision projection smoke summary |
| 2026-06-17 03:27 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_SERIALIZED_VARSTATE_APICAL.md` | implementation | serialized variable-type state audit for random-feedback apical learner |
| 2026-06-17 03:27 | /experiment-bridge | `../output/phase_binding_online_stream_serialized_state_smoke/summary.csv` | implementation | serialized-state output smoke summary |
| 2026-06-17 03:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_varstate_medium/summary.csv` | implementation | variable-type 8-bit row medium summary |
| 2026-06-17 03:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_varstate_medium/phase_trace_apical_inhib_competitive_online_serialized_state_manifest.json` | implementation | serialized-state manifest for variable-type apical learner |
| 2026-06-17 03:50 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_SERIALIZED_CHECKPOINT_APICAL.md` | implementation | loadable integer checkpoint and prediction parity audit for apical no-BP learner |
| 2026-06-17 03:50 | /experiment-bridge | `../output/phase_binding_online_stream_checkpoint_smoke/summary.csv` | implementation | checkpoint save/load smoke summary |
| 2026-06-17 03:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_checkpoint_medium/summary.csv` | implementation | loadable variable-type checkpoint medium summary |
| 2026-06-17 03:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_checkpoint_medium/phase_trace_apical_inhib_competitive_online_serialized_state.npz` | implementation | mixed int8/float32 loadable checkpoint for apical learner |
| 2026-06-17 03:50 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_checkpoint_medium/phase_trace_inhib_competitive_online_serialized_state.npz` | implementation | mixed int8/float32 loadable checkpoint for trace+inhib learner |
| 2026-06-17 04:00 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_NO_PRIOR_APICAL_AUDIT.md` | implementation | no-direct-token-prior audit for apical no-BP learner |
| 2026-06-17 04:00 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_bias0_medium/summary.csv` | implementation | full-precision bias-free apical audit summary |
| 2026-06-17 04:00 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_varstate_bias0_medium/summary.csv` | implementation | variable 8-bit bias-free apical checkpoint summary |
| 2026-06-17 04:00 | /experiment-bridge | `../output/phase_binding_online_stream_apical_random_lowp8_varstate_bias0_medium/phase_trace_apical_inhib_competitive_online_serialized_state.npz` | implementation | loadable checkpoint for bias-free apical learner |
| 2026-06-17 04:08 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_HOMEOSTASIS_CALIBRATION.md` | implementation | homeostatic output calibration audit for bias-free apical learner |
| 2026-06-17 04:08 | /experiment-bridge | `../output/phase_binding_online_stream_homeostasis_smoke/summary.csv` | implementation | output homeostasis smoke summary |
| 2026-06-17 04:08 | /experiment-bridge | `../output/phase_binding_online_stream_apical_homeo_s025_lr005_medium/summary.csv` | implementation | homeostasis weak setting medium summary |
| 2026-06-17 04:08 | /experiment-bridge | `../output/phase_binding_online_stream_apical_homeo_s050_lr010_medium/summary.csv` | implementation | homeostasis medium setting summary |
| 2026-06-17 04:08 | /experiment-bridge | `../output/phase_binding_online_stream_apical_homeo_s100_lr020_medium/summary.csv` | implementation | homeostasis strong setting summary |
| 2026-06-17 04:08 | /experiment-bridge | `../output/phase_binding_online_stream_apical_homeo_s100_lr020_lowp8_checkpoint_medium/summary.csv` | implementation | variable 8-bit homeostasis checkpoint summary |
| 2026-06-17 04:18 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_FEATURE_CALIBRATION.md` | implementation | feature-conditioned neural calibration for bias-free apical learner |
| 2026-06-17 04:18 | /experiment-bridge | `../output/phase_binding_online_stream_feature_calib_smoke/summary.csv` | implementation | feature-conditioned calibration smoke summary |
| 2026-06-17 04:18 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s050_lr010_d64_medium/summary.csv` | implementation | feature calibration weak setting summary |
| 2026-06-17 04:18 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s100_lr020_d64_medium/summary.csv` | implementation | feature calibration medium setting summary |
| 2026-06-17 04:18 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s150_lr030_d64_medium/summary.csv` | implementation | feature calibration strong setting summary |
| 2026-06-17 04:18 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s150_lr030_lowp8_checkpoint_medium/summary.csv` | implementation | variable 8-bit feature calibration checkpoint summary |
| 2026-06-17 04:18 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s150_lr030_lowp8_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_serialized_state.npz` | implementation | loadable feature-calibration checkpoint |
| 2026-06-17 04:26 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_DERIVED_FEATURE_CODES.md` | implementation | derived feature-calibration fixed-code checkpoint audit |
| 2026-06-17 04:26 | /experiment-bridge | `../output/phase_binding_online_stream_feature_calib_derived_smoke/summary.csv` | implementation | derived feature-code checkpoint smoke summary |
| 2026-06-17 04:26 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s150_lr030_derived_lowp8_checkpoint_medium/summary.csv` | implementation | derived feature-code low-precision checkpoint medium summary |
| 2026-06-17 04:26 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_s150_lr030_derived_lowp8_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_serialized_state.npz` | implementation | loadable feature-calibration checkpoint with derived fixed codes |
| 2026-06-17 04:35 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_CHECKPOINT_SIGNATURE.md` | implementation | config signature and mismatch rejection for derived-state checkpoints |
| 2026-06-17 04:35 | /experiment-bridge | `../output/phase_binding_online_stream_feature_calib_signature_smoke/summary.csv` | implementation | signed derived-checkpoint positive smoke summary |
| 2026-06-17 04:35 | /experiment-bridge | `../output/phase_binding_online_stream_feature_calib_signature_smoke/phase_trace_apical_inhib_competitive_online_feature_calib_serialized_state_bad_signature.npz` | implementation | deliberately corrupted signature checkpoint for rejection smoke |
| 2026-06-17 04:35 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_signature_lowp8_checkpoint_medium/summary.csv` | implementation | signed derived feature-calibration checkpoint medium summary |
| 2026-06-17 04:35 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_signature_lowp8_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_serialized_state.npz` | implementation | signed loadable feature-calibration checkpoint |
| 2026-06-17 04:48 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_FEATURE_GATE_SWEEP.md` | implementation | feature-calibration gate dimension/threshold/decay sweep |
| 2026-06-17 04:48 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_dim32_medium/summary.csv` | implementation | feature calibration dim32 sweep summary |
| 2026-06-17 04:48 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_dim128_medium/summary.csv` | implementation | feature calibration dim128 sweep summary |
| 2026-06-17 04:48 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_thr010_medium/summary.csv` | implementation | feature calibration threshold0.10 sweep summary |
| 2026-06-17 04:48 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_gatedecay050_medium/summary.csv` | implementation | feature calibration gate_decay0.50 sweep summary |
| 2026-06-17 04:48 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_gatedecay050_checkpoint_medium/summary.csv` | implementation | signed checkpoint summary for gate_decay0.50 feature calibration |
| 2026-06-17 04:48 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_gatedecay050_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_serialized_state.npz` | implementation | signed loadable checkpoint for gate_decay0.50 feature calibration |
| 2026-06-17 05:10 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_TEMPERATURE_ENERGY_AUDIT.md` | implementation | temperature/energy-scale audit for bias-free feature-calibrated learner |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp050_medium/summary.csv` | implementation | feature calibration temperature 0.5 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp060_medium/summary.csv` | implementation | feature calibration temperature 0.6 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp070_medium/summary.csv` | implementation | feature calibration temperature 0.7 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp080_medium/summary.csv` | implementation | feature calibration temperature 0.8 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp130_medium/summary.csv` | implementation | feature calibration temperature 1.3 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp160_medium/summary.csv` | implementation | feature calibration temperature 1.6 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp200_medium/summary.csv` | implementation | feature calibration temperature 2.0 summary |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp070_checkpoint_medium/summary.csv` | implementation | signed checkpoint summary for best temperature 0.7 feature calibration |
| 2026-06-17 05:10 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_temp070_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_serialized_state.npz` | implementation | signed loadable checkpoint for temperature 0.7 feature calibration |
