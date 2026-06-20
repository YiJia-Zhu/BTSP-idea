# MANIFEST

日期：2026-06-03

> **统一模型警告（2026-06-19）**：本仓库早期 `Rxxx` 与 `llm-token/ITERATION_*` 文档包含大量探索性、统计 baseline、API adapter、bAbI/QA19 数据集特化诊断。QA19/result-patching 报告 `R180-R252` 及对应输出已物理删除，避免继续误导主线。剩余旧结论不能直接作为最终模型结构依据，必须先通过 `UNIFIED_MODEL_CONTRACT.md`：同一个纯 no-BP 神经模型核心、任务无关训练接口、无预训练主干、无数据集专用模块、优先简洁核心修改。

## Documents

- `UNIFIED_MODEL_CONTRACT.md`
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
- `llm-token/ITERATION_2026-06-17_STATE_SPACE_ANTI_ATTRACTOR.md`
- `llm-token/ITERATION_2026-06-17_PREDICTION_ONLY_FIX_CANDIDATE_COMPETITION.md`
- `refine-logs/GOAL_RESET_2026-06-17_NO_BP_FRAMEWORK.md`
- `refine-logs/R080_DATA_PATH_RESET_2026-06-18.md`
- `refine-logs/R081_PREP_BABI_MATERIALIZATION_2026-06-18.md`
- `refine-logs/R081_BABI_QA1_NO_BP_2026-06-18.md`
- `refine-logs/R082_BABI_MULTIHOP_QA_2026-06-18.md`
- `refine-logs/R083_CENTER_DIFF_UPDATE_DIAGNOSTIC_2026-06-18.md`
- `refine-logs/R089_ROLE_BINDING_BABI_QA_2026-06-18.md`
- `refine-logs/R090_LEARNED_EVENT_BINDING_BABI_QA_2026-06-18.md`
- `refine-logs/R091_LEARNED_QUERY_BINDING_BABI_QA_2026-06-18.md`
- `refine-logs/R097_BABI_PARAPHRASE_STRESS_2026-06-18.md`
- `refine-logs/R098_DELAYED_QA_CREDIT_2026-06-18.md`
- `refine-logs/R099_CACHED_QA_CREDIT_SCALING_2026-06-18.md`
- `refine-logs/R100_GATED_QA_CREDIT_2026-06-18.md`
- `refine-logs/R101_CHANNEL_PROTECTED_QA_CREDIT_2026-06-18.md`
- `refine-logs/R102_BEFORE_RELATION_QA_CREDIT_2026-06-18.md`
- `refine-logs/R103_BEFORE_RELATION_SEED_REPEAT_2026-06-18.md`
- `refine-logs/R104_QUERY_SUBJECT_WTA_2026-06-18.md`
- `refine-logs/R105_QA3_FULL_QUERY_WTA_2026-06-18.md`
- `refine-logs/R106_BEFORE_CREDIT_GATE_2026-06-18.md`
- `refine-logs/R107_BEFORE_CREDIT_AGREE_GATE_2026-06-18.md`
- `refine-logs/R108_BEFORE_CREDIT_CONFIDENCE_GATE_2026-06-18.md`
- `refine-logs/R109_CONFIDENCE_GATE_SEED_SCALE_SWEEP_2026-06-18.md`
- `refine-logs/R110_FLIP_DIAGNOSTIC_2026-06-18.md`
- `refine-logs/R111_MARGIN_DIAGNOSTIC_2026-06-18.md`
- `refine-logs/R112_TRAIN_SPLIT_FLIP_GATE_2026-06-18.md`
- `refine-logs/R113_CACHED_FULL_FLIP_GATE_2026-06-18.md`
- `refine-logs/R114_ONE_CLASS_FLIP_GATE_2026-06-18.md`
- `refine-logs/R115_CLASS_PROTOTYPE_GATE_2026-06-18.md`
- `refine-logs/R116_COUNTERFACTUAL_RISK_GATE_2026-06-18.md`
- `refine-logs/R117_RISK_QUANTILE_GATE_2026-06-18.md`
- `refine-logs/R118_RISK_MICROPROTOTYPE_GATE_2026-06-18.md`
- `refine-logs/R119_BEFORE_COMPAT_FLIP_GATE_2026-06-18.md`
- `../output/babi_delayed_credit_qa3_r118_risk_microprototype_gate/summary.csv`
- `../output/babi_delayed_credit_qa3_r118_risk_microprototype_gate/flip_gate_test_metrics.csv`
- `../output/babi_delayed_credit_qa3_r118_risk_microprototype_gate/smoke_sweep.csv`
- `../output/babi_delayed_credit_qa3_r119_before_compat_gate/summary.csv`
- `../output/babi_delayed_credit_qa3_r119_before_compat_gate/flip_gate_test_metrics.csv`
- `../output/babi_delayed_credit_qa3_r119_before_compat_gate/smoke_sweep.csv`

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
- `../export_babi_qa_jsonl.py`
- `../babi_no_bp_qa_experiment.py`
- `../no_bp_update_alignment_diagnostic.py`
- `../babi_paraphrase_stress_experiment.py`
- `../babi_delayed_credit_experiment.py`

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
- `../output/r080_phase_path_smoke/`
- `../output/babi_no_bp_qa1_smoke/`
- `../output/babi_no_bp_qa1_medium/`
- `../output/babi_no_bp_qa2_medium/`
- `../output/babi_no_bp_qa3_medium/`
- `../output/r083_update_alignment_smoke/`
- `../output/r083_update_alignment_medium/`
- `../output/babi_role_binding_qa1_smoke/`
- `../output/babi_role_binding_qa2_smoke/`
- `../output/babi_role_binding_qa3_smoke/`
- `../output/babi_role_binding_qa1_medium/`
- `../output/babi_role_binding_qa2_medium/`
- `../output/babi_role_binding_qa3_medium/`
- `../output/babi_role_binding_qa2_dim8_smoke/`
- `../output/babi_role_binding_qa2_dim16_smoke/`
- `../output/babi_role_binding_qa3_dim8_smoke/`
- `../output/babi_role_binding_qa3_dim16_smoke/`
- `../output/babi_learned_event_qa2_smoke/`
- `../output/babi_learned_event_qa3_smoke/`
- `../output/babi_learned_event_qa2_medium/`
- `../output/babi_learned_event_qa3_medium/`
- `../output/babi_learned_event_query_qa1_smoke/`
- `../output/babi_learned_event_query_qa2_smoke/`
- `../output/babi_learned_event_query_qa3_smoke/`
- `../output/babi_learned_event_query_qa1_medium/`
- `../output/babi_learned_event_query_qa2_medium/`
- `../output/babi_learned_event_query_qa3_medium/`
- `../output/babi_paraphrase_stress_smoke_fix/`
- `../output/babi_paraphrase_stress_qa2_medium_key/`
- `../output/babi_paraphrase_stress_qa3_medium_key/`
- `../output/babi_delayed_credit_qa2_smoke/`
- `../output/babi_delayed_credit_qa2_query_smoke/`
- `../output/babi_delayed_credit_qa2_seeded_300/`
- `../output/babi_delayed_credit_qa3_query_smoke/`
- `../output/babi_delayed_credit_qa3_event30_smoke/`
- `../output/babi_delayed_credit_qa2_cached_300/`
- `../output/babi_delayed_credit_qa2_cached_full/`
- `../output/babi_delayed_credit_qa3_cached_80/`
- `../output/babi_delayed_credit_qa3_cached_200/`
- `../output/babi_delayed_credit_qa3_cached_query_200/`
- `../output/babi_delayed_credit_qa2_gated_full/`
- `../output/babi_delayed_credit_qa2_queryonly_full/`
- `../output/babi_delayed_credit_qa3_gated_200/`
- `../output/babi_delayed_credit_qa3_gated_query_200/`
- `../output/babi_delayed_credit_qa3_eventonly_200/`
- `../output/babi_delayed_credit_qa3_before_relation_smoke/`
- `../output/babi_delayed_credit_qa3_before_matrix_200/`
- `../output/babi_delayed_credit_qa3_before_relation_200/`
- `../output/babi_delayed_credit_qa3_before_matrix_w05_200/`
- `../output/babi_delayed_credit_qa3_queryonly_seed1_200/`
- `../output/babi_delayed_credit_qa3_queryonly_seed2_200/`
- `../output/babi_delayed_credit_qa3_before_matrix_w05_seed1_200/`
- `../output/babi_delayed_credit_qa3_before_matrix_w05_seed2_200/`
- `../output/babi_delayed_credit_qa3_r103_seed_repeat/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_seed0_200/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_seed1_200/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_seed2_200/`
- `../output/babi_delayed_credit_qa3_r104_query_wta_seed_repeat/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_full_seed0/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_full_seed1/`
- `../output/babi_delayed_credit_qa3_r105_full/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_full_seed1_gate_m1/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w05_full_seed1_gate_m2/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w025_full_seed0/`
- `../output/babi_delayed_credit_qa3_query_wta_before_w025_full_seed1/`
- `../output/babi_delayed_credit_qa3_r106_before_gate/`
- `../output/babi_delayed_credit_qa3_r111_margin_diagnostic/`
- `../output/babi_delayed_credit_qa3_r112_flip_gate_medium/`
- `../output/babi_delayed_credit_qa3_r113_cache_full_gate/`
- `../output/babi_delayed_credit_qa3_r114_oneclass_gate/`
- `../output/babi_delayed_credit_qa3_r115_class_prototype_gate/`
- `../output/babi_delayed_credit_qa3_r116_counterfactual_risk_gate/`
- `../output/babi_delayed_credit_qa3_r117_risk_quantile_gate/`
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
| 2026-06-17 05:20 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_READOUT_GAIN.md` | implementation | model-side readout gain calibration audit for bias-free feature-calibrated learner |
| 2026-06-17 05:20 | /experiment-bridge | `../output/phase_binding_online_stream_readout_gain_smoke/summary.csv` | implementation | readout gain wrapper smoke summary |
| 2026-06-17 05:20 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_fixedgain1429_checkpoint_medium/summary.csv` | implementation | fixed readout gain checkpoint summary |
| 2026-06-17 05:20 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_fixedgain1429_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_serialized_state.npz` | implementation | signed loadable checkpoint with fixed readout gain |
| 2026-06-17 05:20 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_margingain_medium/summary.csv` | implementation | naive margin dynamic readout gain summary |
| 2026-06-17 14:16 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_LOCAL_READOUT_GAIN.md` | implementation | context-local adaptive readout gain audit |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_local_gain_smoke/summary.csv` | implementation | local adaptive readout gain checkpoint smoke summary |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_s015_lr005_medium/summary.csv` | implementation | weak local readout gain medium summary |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_s025_lr010_medium/summary.csv` | implementation | medium local readout gain summary |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_s050_lr020_medium/summary.csv` | implementation | strong local readout gain summary |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_base100_s050_lr020_medium/summary.csv` | implementation | local readout gain learned from base gain 1.0 summary |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_s015_lr005_checkpoint_medium/summary.csv` | implementation | signed checkpoint summary for best local adaptive readout gain |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_s015_lr005_checkpoint_medium/generation_summary.csv` | implementation | generation repetition metrics for best local adaptive readout gain |
| 2026-06-17 14:16 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_localgain_s015_lr005_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_local_gain_serialized_state.npz` | implementation | signed loadable checkpoint for best local adaptive readout gain |
| 2026-06-17 14:30 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_BRANCH_AGREEMENT_READOUT.md` | implementation | branch-agreement readout audit for winner-ordering no-BP learner |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_branch_agreement_smoke/summary.csv` | implementation | branch-agreement readout smoke summary |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_meanmin_s005_medium/summary.csv` | implementation | branch-agreement mean_min strength0.05 summary |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_meanmin_s015_medium/summary.csv` | implementation | branch-agreement mean_min strength0.15 summary |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_posfrac_s010_medium/summary.csv` | implementation | branch-agreement positive_fraction strength0.10 summary |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_medium/summary.csv` | implementation | branch-agreement low_variance strength0.10 summary |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_checkpoint_medium/summary.csv` | implementation | signed checkpoint summary for best CE branch-agreement readout |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_checkpoint_medium/generation_summary.csv` | implementation | generation repetition metrics for best CE branch-agreement readout |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_serialized_state.npz` | implementation | signed loadable checkpoint for best CE branch-agreement readout |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_fixedgain1429_seed1_medium/summary.csv` | implementation | seed1 fixed readout gain baseline summary |
| 2026-06-17 14:30 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_seed1_medium/summary.csv` | implementation | seed1 branch-agreement low_variance summary |
| 2026-06-17 14:58 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_PLASTIC_BRANCH_AGREEMENT.md` | implementation | plastic branch-agreement readout audit |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_meanmin_s015_seed1_medium/summary.csv` | implementation | seed1 branch-agreement mean_min strength0.15 summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_fixedgain1429_seed2_medium/summary.csv` | implementation | seed2 fixed readout gain baseline summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_lowvar_s010_seed2_medium/summary.csv` | implementation | seed2 branch-agreement low_variance summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_meanmin_s015_seed2_medium/summary.csv` | implementation | seed2 branch-agreement mean_min summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_plastic_branch_agreement_smoke/summary.csv` | implementation | plastic branch-agreement checkpoint smoke summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_medium/summary.csv` | implementation | weak plastic branch-agreement medium summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s005_lr005_medium/summary.csv` | implementation | medium plastic branch-agreement summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_plastic_branchagree_only_s005_lr005_medium/summary.csv` | implementation | plastic-only branch-agreement summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_checkpoint_medium/summary.csv` | implementation | signed checkpoint summary for best-accuracy plastic branch-agreement |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_checkpoint_medium/generation_summary.csv` | implementation | generation repetition metrics for best-accuracy plastic branch-agreement |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_serialized_state.npz` | implementation | signed loadable checkpoint for best-accuracy plastic branch-agreement |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_seed1_medium/summary.csv` | implementation | seed1 plastic branch-agreement summary |
| 2026-06-17 14:58 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branchagree_plastic_s002_lr002_seed2_medium/summary.csv` | implementation | seed2 plastic branch-agreement summary |
| 2026-06-17 15:27 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_PRESSURE_GATED_PLASTICITY.md` | implementation | pressure-gated plastic branch-agreement audit and loop-pressure diagnostics |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_pressure_plastic_smoke/summary.csv` | implementation | pressure-gated plastic branch-agreement smoke summary |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_pressure_plastic_smoke/generation_summary.csv` | implementation | smoke generation summary with loop-pressure diagnostics |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t000_medium/summary.csv` | implementation | inhibition-pressure plastic branch-agreement threshold0.00 summary |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_medium/summary.csv` | implementation | inhibition-pressure plastic branch-agreement threshold0.02 summary |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_contextloop_t050_medium/summary.csv` | implementation | context-loop pressure plastic branch-agreement summary |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_checkpoint_medium/summary.csv` | implementation | signed checkpoint summary for inhibition-pressure plastic branch-agreement |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_checkpoint_medium/generation_summary.csv` | implementation | generation loop-pressure metrics for inhibition-pressure plastic branch-agreement |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_checkpoint_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_serialized_state.npz` | implementation | signed loadable checkpoint for inhibition-pressure plastic branch-agreement |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_seed1_medium/summary.csv` | implementation | seed1 inhibition-pressure plastic branch-agreement summary |
| 2026-06-17 15:27 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_pressure_plastic_inhib_t002_seed2_medium/summary.csv` | implementation | seed2 inhibition-pressure plastic branch-agreement summary |
| 2026-06-17 16:06 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_DYNAMIC_LOOP_INHIBITION.md` | implementation | dynamic loop-pressure inhibition audit and boundary result |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_loop_inhibition_smoke/summary.csv` | implementation | token loop inhibition checkpoint smoke summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_transition_loop_inhibition_smoke/summary.csv` | implementation | transition loop inhibition checkpoint smoke summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_transition_loop_inhibition_smoke/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_loop_inhib_serialized_state.npz` | implementation | loadable checkpoint smoke for loop inhibition state |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_loop_inhib_s005_medium/summary.csv` | implementation | token loop inhibition strength0.05 medium summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_loop_inhib_s010_medium/summary.csv` | implementation | token loop inhibition strength0.10 medium summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_loop_inhib_s020_medium/summary.csv` | implementation | token loop inhibition strength0.20 medium summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_transition_loop_s025_medium/summary.csv` | implementation | transition loop inhibition strength0.25 medium summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_transition_loop_s050_medium/summary.csv` | implementation | transition loop inhibition strength0.50 medium summary |
| 2026-06-17 16:06 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_transition_loop_s100_medium/summary.csv` | implementation | transition loop inhibition strength1.00 medium summary |
| 2026-06-17 16:24 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_SEGMENT_ATTRACTOR_INHIBITION.md` | implementation | segment attractor inhibition audit and generation-quality tradeoff |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_baseline_smoke/summary.csv` | implementation | same-data R069 baseline for segment attractor smoke |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_smoke/summary.csv` | implementation | segment attractor checkpoint smoke summary |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_smoke/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_segment_attractor_serialized_state.npz` | implementation | loadable checkpoint smoke for segment attractor dynamic state |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_s010_smoke/summary.csv` | implementation | segment attractor strength0.10 smoke summary |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_s025_smoke/summary.csv` | implementation | segment attractor strength0.25 smoke summary |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_s025_t090_smoke/summary.csv` | implementation | segment attractor strength0.25 threshold0.90 smoke summary |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_segment_attractor_s200_t050_smoke/summary.csv` | implementation | aggressive segment attractor smoke summary |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_segment_attractor_s100_t050_medium/summary.csv` | implementation | medium segment attractor strength1.00 threshold0.50 summary |
| 2026-06-17 16:24 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_segment_attractor_s200_t050_medium/summary.csv` | implementation | medium segment attractor strength2.00 threshold0.50 summary |
| 2026-06-17 17:41 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_EVENT_GATED_SEGMENT_ATTRACTOR.md` | implementation | event-gated segment attractor audit |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_margin_smoke/summary.csv` | implementation | event segment margin-gate smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_inhib_smoke/summary.csv` | implementation | event segment inhibition-gate threshold0.02 smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_or_smoke/summary.csv` | implementation | event segment margin-or-inhibition smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_and_smoke/summary.csv` | implementation | event segment margin-and-inhibition smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_inhib_t005_smoke/summary.csv` | implementation | event segment inhibition-gate threshold0.05 smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_inhib_t010_smoke/summary.csv` | implementation | event segment inhibition-gate threshold0.10 smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_inhib_t020_smoke/summary.csv` | implementation | event segment inhibition-gate threshold0.20 smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_inhib_s100_t005_smoke/summary.csv` | implementation | event segment inhibition-gate strength1.0 threshold0.05 smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_event_segment_inhib_s150_t005_smoke/summary.csv` | implementation | event segment inhibition-gate strength1.5 threshold0.05 smoke summary |
| 2026-06-17 17:41 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_event_segment_inhib_s200_t005_medium/summary.csv` | implementation | medium event-gated segment inhibition summary |
| 2026-06-17 17:59 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_LOOP_ESCAPE_COMPETITOR.md` | implementation | learned loop-escape competitor audit |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_pressure_smoke/summary.csv` | implementation | loop-escape pressure-gated smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_pressure_margin_smoke/summary.csv` | implementation | loop-escape pressure-and-margin smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_or_smoke/summary.csv` | implementation | loop-escape pressure-or-margin smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_strong_smoke/summary.csv` | implementation | strong loop-escape smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_s025_lr002_smoke/summary.csv` | implementation | weak loop-escape smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_s035_lr003_smoke/summary.csv` | implementation | mid loop-escape smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_s040_lr004_smoke/summary.csv` | implementation | mid-high loop-escape smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_s050_lr005_smoke/summary.csv` | implementation | loop-escape strength0.50 smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_s075_lr005_smoke/summary.csv` | implementation | loop-escape strength0.75 smoke summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_loop_escape_s035_lr003_medium/summary.csv` | implementation | medium loop-escape strength0.35 summary |
| 2026-06-17 17:59 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_loop_escape_s050_lr005_medium/summary.csv` | implementation | medium loop-escape strength0.50 summary |
| 2026-06-17 18:13 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_CANDIDATE_LIMITED_LOOP_ESCAPE.md` | implementation | candidate-limited and winner-local loop escape audit |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_topk8_smoke/summary.csv` | implementation | candidate-limited loop escape top-k smoke summary |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_topk8_wrongonly_smoke/summary.csv` | implementation | candidate-limited loop escape wrong-only smoke summary |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_winner_suppress_smoke/summary.csv` | implementation | winner-suppress loop escape smoke summary |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_winner_wrongonly_smoke/summary.csv` | implementation | winner-suppress wrong-only loop escape smoke summary |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_winner_suppress_s025_smoke/summary.csv` | implementation | weak winner-suppress loop escape smoke summary |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_loop_escape_winner_suppress_s035_smoke/summary.csv` | implementation | mid winner-suppress loop escape smoke summary |
| 2026-06-17 18:13 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_loop_escape_winner_suppress_medium/summary.csv` | implementation | medium winner-suppress loop escape summary |
| 2026-06-17 18:32 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_BRANCH_STATE_STABILIZER.md` | implementation | representation-level branch-state stabilizer audit |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_smoke/summary.csv` | implementation | branch-state stabilizer any-gate smoke summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_smoke/generation_summary.csv` | implementation | branch-state stabilizer any-gate smoke generation summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_weak_smoke/summary.csv` | implementation | weak branch-state stabilizer smoke summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_weak_smoke/generation_summary.csv` | implementation | weak branch-state stabilizer smoke generation summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_apical_smoke/summary.csv` | implementation | apical-gated branch-state stabilizer smoke summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_apical_smoke/generation_summary.csv` | implementation | apical-gated branch-state stabilizer smoke generation summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branch_state_apical_medium/summary.csv` | implementation | medium apical-gated branch-state stabilizer summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branch_state_apical_medium/generation_summary.csv` | implementation | medium apical-gated branch-state generation summary |
| 2026-06-17 18:32 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branch_state_apical_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_branch_state_serialized_state.npz` | implementation | loadable low-precision branch-state checkpoint with exact parity |
| 2026-06-17 18:46 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_LOWRANK_BRANCH_STATE.md` | implementation | low-rank branch-state projection and novelty-gate audit |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_lowrank16_smoke/summary.csv` | implementation | rank16 branch-state smoke summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_lowrank16_smoke/generation_summary.csv` | implementation | rank16 branch-state smoke generation summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_lowrank16_novelty_smoke/summary.csv` | implementation | rank16 branch-state novelty threshold0.92 smoke summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_lowrank16_novelty_smoke/generation_summary.csv` | implementation | rank16 branch-state novelty threshold0.92 generation summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_lowrank16_novelty050_smoke/summary.csv` | implementation | rank16 branch-state novelty threshold0.50 smoke summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_lowrank16_novelty050_smoke/generation_summary.csv` | implementation | rank16 branch-state novelty threshold0.50 generation summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branch_state_lowrank16_medium/summary.csv` | implementation | medium rank16 branch-state summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branch_state_lowrank16_medium/generation_summary.csv` | implementation | medium rank16 branch-state generation summary |
| 2026-06-17 18:46 | /experiment-bridge | `../output/phase_binding_online_stream_apical_feature_calib_branch_state_lowrank16_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_branch_state_serialized_state.npz` | implementation | loadable rank16 branch-state checkpoint with exact parity |
| 2026-06-17 19:35 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_STATE_SPACE_ANTI_ATTRACTOR.md` | implementation | state-space anti-attractor and prediction-only anti-score audit |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_score_s100_medium/summary.csv` | implementation | all-observation anti-score medium summary |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_score_s100_medium/generation_summary.csv` | implementation | all-observation anti-score medium generation summary |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s075_medium/summary.csv` | implementation | prediction-only anti-score strength0.75 medium summary |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s075_medium/generation_summary.csv` | implementation | prediction-only anti-score strength0.75 medium generation summary |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_predonly_medium/summary.csv` | implementation | prediction-only anti-score strength1.0 medium summary |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_predonly_medium/generation_summary.csv` | implementation | prediction-only anti-score strength1.0 medium generation summary |
| 2026-06-17 19:35 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_predonly_medium/phase_trace_apical_inhib_competitive_online_feature_calib_gain_branch_agree_plastic_branch_agree_branch_state_serialized_state.npz` | implementation | loadable prediction-only anti-score checkpoint with exact parity |
| 2026-06-17 20:10 | /experiment-bridge | `llm-token/ITERATION_2026-06-17_PREDICTION_ONLY_FIX_CANDIDATE_COMPETITION.md` | implementation | corrected prediction-only anti-score and candidate competition audit |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_fixed_smoke/summary.csv` | implementation | fixed prediction-only anti-score smoke summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_fixed_smoke/generation_summary.csv` | implementation | fixed prediction-only anti-score smoke generation summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s075_fixed_medium/summary.csv` | implementation | fixed prediction-only anti-score strength0.75 medium summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s075_fixed_medium/generation_summary.csv` | implementation | fixed prediction-only anti-score strength0.75 medium generation summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_fixed_medium/summary.csv` | implementation | fixed prediction-only anti-score strength1.0 medium summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_predscore_s100_fixed_medium/generation_summary.csv` | implementation | fixed prediction-only anti-score strength1.0 medium generation summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_candidate_k4_smoke/summary.csv` | implementation | candidate-local anti-score top-k4 smoke summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_anti_candidate_k4_smoke/generation_summary.csv` | implementation | candidate-local anti-score top-k4 smoke generation summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_candidate_agree_w100_smoke/summary.csv` | implementation | positive branch-agreement candidate competition smoke summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_candidate_agree_w100_smoke/generation_summary.csv` | implementation | positive branch-agreement candidate competition generation summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_candidate_agree_wneg100_smoke/summary.csv` | implementation | negative branch-agreement candidate competition smoke summary |
| 2026-06-17 20:10 | /experiment-bridge | `../output/phase_binding_online_stream_branch_state_candidate_agree_wneg100_smoke/generation_summary.csv` | implementation | negative branch-agreement candidate competition generation summary |
| 2026-06-18 19:08 | /experiment-bridge | `refine-logs/R120_R122_COMPAT_CHANNEL_GATE_2026-06-18.md` | report | learned compatibility-channel flip gate report for bAbI QA3 |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r120_r122_compat_channel_gate/summary.csv` | results | R114/R118/R120/R122 QA3 flip-gate summary comparison |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r120_r122_compat_channel_gate/flip_gate_comparison.csv` | results | detailed QA3 compatibility-channel flip breakdown |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r122_compat_channel_cfneg_t025_full_seed2/summary.csv` | results | R122 full QA3 threshold0.25 summary |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r122_compat_channel_cfneg_t050_full_seed2/summary.csv` | results | R122 full QA3 threshold0.50 summary |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r122_compat_channel_cfneg_t100_full_seed2/summary.csv` | results | R122 full QA3 threshold1.00 summary |
| 2026-06-18 19:08 | /experiment-bridge | `refine-logs/R123_STRICT_COMPAT_RESCUE_GATE_2026-06-18.md` | report | strict compatibility rescue gate report for bAbI QA3 |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r123_strict_compat_rescue_gate/flip_gate_comparison.csv` | results | R123 strict compatibility rescue detailed flip comparison |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r123_strict_compat_rescue_gate/seed_repeat.csv` | results | paired R122/R123 QA3 seed repeat |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r123_strict_compat_rescue_gate/seed_summary.csv` | results | R122/R123 three-seed mean and standard deviation |
| 2026-06-18 19:08 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r123_class_rescue_t025_r100_full_seed2/summary.csv` | results | R123 class rescue threshold0.25 radius1.0 seed2 summary |
| 2026-06-18 19:40 | /experiment-bridge | `refine-logs/R124_RISK_MISS_DIAGNOSTIC_2026-06-18.md` | report | risk-miss diagnostic for R123 bAbI QA3 seed1 failure |
| 2026-06-18 19:40 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r124_risk_miss_diagnostic/risk_group_summary.csv` | results | grouped risk/compatibility prototype diagnostics |
| 2026-06-18 19:40 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r124_risk_miss_diagnostic/allowed_harmful_details.csv` | results | allowed harmful flip nearest-risk details |
| 2026-06-18 19:40 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r124_risk_miss_diagnostic/risk_margin_buffer_simulation.csv` | results | simulated risk-margin buffer tradeoff |
| 2026-06-18 20:13 | /experiment-bridge | `refine-logs/R125_RISK_NEAR_MISS_GATE_2026-06-18.md` | report | risk-near-miss inhibitory gate report for bAbI QA3 |
| 2026-06-18 20:13 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r125_near_miss_fixed_gate/seed_summary.csv` | results | R122/R123/R125 three-seed mean comparison |
| 2026-06-18 20:13 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r125_near_miss_fixed_gate/seed_repeat.csv` | results | paired R125 seed repeat and deltas vs R123 |
| 2026-06-18 20:13 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r125_near_miss_fixed_gate/near_miss_diagnostic_groups.csv` | results | raw-vs-nearest risk and near-miss diagnostic groups |
| 2026-06-18 20:23 | /experiment-bridge | `refine-logs/R126_LOCAL_RADIUS_NEAR_RISK_GATE_2026-06-18.md` | report | local-radius near-risk inhibitory gate report for bAbI QA3 |
| 2026-06-18 20:23 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r126_local_near_gate/seed_summary.csv` | results | R122/R123/R125/R126 three-seed mean comparison |
| 2026-06-18 20:23 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r126_local_near_gate/seed_repeat.csv` | results | paired R126 seed repeat and deltas |
| 2026-06-18 20:23 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r126_local_near_gate/near_miss_diagnostic_groups.csv` | results | local-radius near-risk diagnostic groups |
| 2026-06-18 20:39 | /experiment-bridge | `refine-logs/R127_VALIDATION_SELECTED_NEAR_RISK_SWEEP_2026-06-18.md` | report | validation-selected near-risk fraction sweep report for bAbI QA3 |
| 2026-06-18 20:39 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r127_fraction_sweep/fraction_summary.csv` | results | train/validation/test near-risk fraction sweep summary |
| 2026-06-18 20:39 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r127_fraction_sweep/selected_fraction_rows.csv` | results | validation-selected fraction rows |
| 2026-06-18 20:39 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r127_fraction_sweep/selection.json` | results | validation selection rule and selected fraction |
| 2026-06-18 20:54 | /experiment-bridge | `refine-logs/R128_TRAIN_FOLD_NEAR_RISK_CALIBRATION_2026-06-18.md` | report | train-fold near-risk calibration report for bAbI QA3 |
| 2026-06-18 20:54 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r128_train_calib_gate/fraction_summary.csv` | results | train-calibration near-risk fraction sweep summary |
| 2026-06-18 20:54 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r128_train_calib_gate/selection.json` | results | train-calibration selected fraction and mapped full-train test result |
| 2026-06-18 20:54 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r128_train_calib_gate/selected_fraction_rows.csv` | results | train-calibration selected fraction rows |
| 2026-06-18 21:04 | /experiment-bridge | `refine-logs/R129_NEAR_RISK_DOMINANT_INHIBITION_2026-06-18.md` | report | near-risk dominant inhibitory veto report for bAbI QA3 |
| 2026-06-18 21:04 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r129_near_blocks_rescue_gate/comparison_summary.csv` | results | R123/R126/R129 three-seed mean comparison |
| 2026-06-18 21:04 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r129_near_blocks_rescue_gate/paired_deltas.csv` | results | paired R129 deltas against R123 and R126 |
| 2026-06-18 21:04 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r129_near_blocks_rescue_gate/diagnostic_groups.csv` | results | R129 grouped flip diagnostics |
| 2026-06-18 21:38 | /experiment-bridge | `refine-logs/R130_LEARNED_NEAR_RISK_CHANNEL_2026-06-18.md` | report | learned near-risk inhibitory channel report for bAbI QA3 |
| 2026-06-18 21:38 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r130_learned_near_sweep_gate/threshold_summary.csv` | results | learned near-risk threshold sweep summary |
| 2026-06-18 21:38 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r130_learned_near_sweep_gate/selection.json` | results | validation-selected learned near-risk threshold |
| 2026-06-18 21:38 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r130_learned_near_sweep_gate/comparison_summary.csv` | results | R123/R126/R129/R130 comparison |
| 2026-06-18 22:03 | /experiment-bridge | `refine-logs/R131_BALANCED_NEAR_RISK_CHANNEL_2026-06-18.md` | report | balanced learned near-risk channel report for bAbI QA3 |
| 2026-06-18 22:03 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r131_balanced_near_gate/threshold_summary.csv` | results | balanced learned near-risk threshold sweep summary |
| 2026-06-18 22:03 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r131_balanced_near_gate/selection.json` | results | validation-selected balanced learned near-risk threshold |
| 2026-06-18 22:03 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r131_balanced_near_gate/comparison_summary.csv` | results | R123/R126/R129/R130/R131 comparison |
| 2026-06-18 22:35 | /experiment-bridge | `refine-logs/R132_TRAIN_AUTO_NEAR_RISK_THRESHOLD_2026-06-18.md` | report | train-only auto near-risk threshold report for bAbI QA3 |
| 2026-06-18 22:35 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r132_auto_near_gate/auto_calibration_summary.csv` | results | R132 train-only auto-threshold calibration summary |
| 2026-06-18 22:35 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r132_auto_near_gate/threshold_summary.csv` | results | R132 learned near-risk threshold sweep summary |
| 2026-06-18 22:35 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r132_auto_near_gate/selection.json` | results | R132 train-only selected thresholds and caveat |
| 2026-06-18 22:35 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r132_auto_near_gate/comparison_summary.csv` | results | R123/R126/R129/R130/R131/R132 comparison |
| 2026-06-18 22:55 | /experiment-bridge | `refine-logs/R133_SOURCE_CONDITIONED_NEAR_RISK_CHANNEL_2026-06-18.md` | report | source-conditioned learned near-risk channel report for bAbI QA3 |
| 2026-06-18 22:55 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r133_source_near_gate/source_channel_summary.csv` | results | R133 source-channel train calibration summary |
| 2026-06-18 22:55 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r133_source_near_gate/selection.json` | results | R133 source-channel selected thresholds and caveat |
| 2026-06-18 22:55 | /experiment-bridge | `../output/babi_delayed_credit_qa3_r133_source_near_gate/comparison_summary.csv` | results | R123/R126/R129/R130/R131/R132/R133 comparison |
| 2026-06-18 23:35 | /experiment-bridge | `refine-logs/R134_HEBBIAN_KV_TOKEN_BRANCH_2026-06-18.md` | report | Hebbian KV token branch report for TinyStories no-BP learner |
| 2026-06-18 23:35 | /experiment-bridge | `../output/phase_binding_online_stream_r134_hebbian_kv/summary.csv` | results | R134 three-seed aggregate summary |
| 2026-06-18 23:35 | /experiment-bridge | `../output/phase_binding_online_stream_r134_hebbian_kv/per_seed_summary.csv` | results | R134 per-seed medium summary |
| 2026-06-18 23:35 | /experiment-bridge | `../output/phase_binding_online_stream_r134_hebbian_kv/paired_deltas.csv` | results | R134 paired KV-minus-baseline deltas |
| 2026-06-18 23:35 | /experiment-bridge | `../output/phase_binding_online_stream_r134_hebbian_kv/generation_summary_aggregate.csv` | results | R134 generation repetition aggregate |
| 2026-06-18 23:55 | /experiment-bridge | `refine-logs/R135_KV_MARGIN_GATE_2026-06-18.md` | report | hard local KV confidence gate report |
| 2026-06-18 23:55 | /experiment-bridge | `../output/phase_binding_online_stream_r135_kv_margin_gate/summary.csv` | results | R135 R134-vs-gated three-seed aggregate summary |
| 2026-06-18 23:55 | /experiment-bridge | `../output/phase_binding_online_stream_r135_kv_margin_gate/per_seed_summary.csv` | results | R135 per-seed gated KV summary |
| 2026-06-18 23:55 | /experiment-bridge | `../output/phase_binding_online_stream_r135_kv_margin_gate/paired_deltas.csv` | results | R135 paired gate/base and gate/no-gate deltas |
| 2026-06-18 23:55 | /experiment-bridge | `../output/phase_binding_online_stream_r135_kv_margin_gate/generation_summary_aggregate.csv` | results | R135 generation repetition aggregate |
| 2026-06-18 23:59 | /experiment-bridge | `refine-logs/R136_EPROP_TRACE_READOUT_2026-06-18.md` | report | e-prop eligibility trace readout report |
| 2026-06-18 23:59 | /experiment-bridge | `../output/phase_binding_online_stream_r136_eprop_trace/summary.csv` | results | R136 three-seed aggregate summary |
| 2026-06-18 23:59 | /experiment-bridge | `../output/phase_binding_online_stream_r136_eprop_trace/per_seed_summary.csv` | results | R136 per-seed summary |
| 2026-06-18 23:59 | /experiment-bridge | `../output/phase_binding_online_stream_r136_eprop_trace/paired_deltas.csv` | results | R136 paired e-prop minus baseline deltas |
| 2026-06-18 23:59 | /experiment-bridge | `../output/phase_binding_online_stream_r136_eprop_trace/generation_summary_aggregate.csv` | results | R136 generation repetition aggregate |
| 2026-06-18 23:32 | /experiment-bridge | `refine-logs/R137_ATTRIBUTE_BINDING_QA15_QA16_2026-06-18.md` | report | attribute/category binding report for bAbI QA15/QA16 |
| 2026-06-18 23:32 | /experiment-bridge | `../output/babi_attr_binding_qa15_qa16_seed_repeat/aggregate_summary.csv` | results | R137 QA15/QA16 three-seed aggregate summary |
| 2026-06-18 23:32 | /experiment-bridge | `../output/babi_attr_binding_qa15_qa16_seed_repeat/seed_repeat.csv` | results | R137 QA15/QA16 per-seed summary rows |
| 2026-06-18 23:32 | /experiment-bridge | `../output/babi_attr_binding_qa15_qa16_seed_repeat/selection.json` | results | R137 selected method and output directories |
| 2026-06-18 23:43 | /experiment-bridge | `refine-logs/R138_LEARNED_ATTRIBUTE_FRONTEND_QA15_QA16_2026-06-18.md` | report | learned attribute/query front-end report for bAbI QA15/QA16 |
| 2026-06-18 23:43 | /experiment-bridge | `../output/babi_attr_learned_qa15_qa16_seed_repeat/aggregate_summary.csv` | results | R138 learned-front-end QA15/QA16 three-seed aggregate summary |
| 2026-06-18 23:43 | /experiment-bridge | `../output/babi_attr_learned_qa15_qa16_seed_repeat/detector_metric_summary.csv` | results | R138 learned statement/query detector aggregate metrics |
| 2026-06-18 23:43 | /experiment-bridge | `../output/babi_attr_learned_qa15_qa16_seed_repeat/seed_repeat.csv` | results | R138 learned-front-end QA15/QA16 per-seed summary rows |
| 2026-06-18 23:43 | /experiment-bridge | `../output/babi_attr_learned_qa15_qa16_seed_repeat/selection.json` | results | R138 selected modes and output directories |
| 2026-06-18 23:58 | /experiment-bridge | `../babi_attribute_paraphrase_stress_experiment.py` | implementation | QA15/QA16 attribute paraphrase stress harness |
| 2026-06-18 23:58 | /experiment-bridge | `refine-logs/R139_ATTRIBUTE_PARAPHRASE_STRESS_2026-06-18.md` | report | attribute paraphrase stress report for bAbI QA15/QA16 |
| 2026-06-18 23:58 | /experiment-bridge | `../output/babi_attribute_paraphrase_stress_r139/aggregate_summary.csv` | results | R139 QA15/QA16 paraphrase stress three-seed aggregate |
| 2026-06-18 23:58 | /experiment-bridge | `../output/babi_attribute_paraphrase_stress_r139/detector_summary.csv` | results | R139 learned statement/query detector paraphrase aggregate |
| 2026-06-18 23:58 | /experiment-bridge | `../output/babi_attribute_paraphrase_stress_r139/seed_repeat.csv` | results | R139 per-seed QA summary rows |
| 2026-06-18 23:58 | /experiment-bridge | `../output/babi_attribute_paraphrase_stress_r139/selection.json` | results | R139 selected result summary |
| 2026-06-19 00:45 | /experiment-bridge | `../babi_attribute_delayed_credit_experiment.py` | implementation | QA15/QA16 attribute delayed answer-credit experiment |
| 2026-06-19 00:45 | /experiment-bridge | `refine-logs/R140_ATTRIBUTE_DELAYED_CREDIT_2026-06-19.md` | report | attribute delayed QA-credit report for bAbI QA15/QA16 |
| 2026-06-19 00:45 | /experiment-bridge | `../output/babi_attribute_delayed_credit_r140/aggregate_summary.csv` | results | R140 QA15/QA16 delayed-credit three-seed aggregate |
| 2026-06-19 00:45 | /experiment-bridge | `../output/babi_attribute_delayed_credit_r140/detector_summary.csv` | results | R140 detector metrics aggregate |
| 2026-06-19 00:45 | /experiment-bridge | `../output/babi_attribute_delayed_credit_r140/credit_summary.csv` | results | R140 credit update statistics aggregate |
| 2026-06-19 00:45 | /experiment-bridge | `../output/babi_attribute_delayed_credit_r140/selection.json` | results | R140 selected result summary |
| 2026-06-19 01:05 | /experiment-bridge | `refine-logs/R141_PAIR_STATEMENT_CREDIT_2026-06-19.md` | report | pair-statement eligibility credit boundary report |
| 2026-06-19 01:05 | /experiment-bridge | `../output/babi_attribute_pair_credit_r141/pair_aggregate_summary.csv` | results | R141 pair-credit aggregate summary |
| 2026-06-19 01:05 | /experiment-bridge | `../output/babi_attribute_pair_credit_r141/paired_deltas.csv` | results | R141 pair-credit deltas vs R140 |
| 2026-06-19 01:05 | /experiment-bridge | `../output/babi_attribute_pair_credit_r141/pair_credit_summary.csv` | results | R141 pair-credit update statistics |
| 2026-06-19 01:05 | /experiment-bridge | `../output/babi_attribute_pair_credit_r141/selection.json` | results | R141 selected result summary |
| 2026-06-19 01:28 | /experiment-bridge | `refine-logs/R142_SLOT_CONSOLIDATION_2026-06-19.md` | report | slot consolidation credit report for QA15/QA16 |
| 2026-06-19 01:28 | /experiment-bridge | `../output/babi_attribute_slot_consolidation_r142/aggregate_summary.csv` | results | R142 slot-consolidation aggregate summary |
| 2026-06-19 01:28 | /experiment-bridge | `../output/babi_attribute_slot_consolidation_r142/detector_summary.csv` | results | R142 detector metrics aggregate |
| 2026-06-19 01:28 | /experiment-bridge | `../output/babi_attribute_slot_consolidation_r142/credit_summary.csv` | results | R142 consolidation credit statistics |
| 2026-06-19 01:28 | /experiment-bridge | `../output/babi_attribute_slot_consolidation_r142/selection.json` | results | R142 selected result summary |
| 2026-06-19 00:47 | /experiment-bridge | `../babi_relation_state_experiment.py` | implementation | QA18/QA19 relation-state no-BP experiment |
| 2026-06-19 00:47 | /experiment-bridge | `refine-logs/R143_RELATION_STATE_QA18_QA19_2026-06-19.md` | report | relation-state size/path QA expansion report |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_state_r143_final/aggregate_summary.csv` | results | R143 QA18/QA19 three-seed aggregate summary |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_state_r143_final/seed_test_rows.json` | results | R143 per-seed test rows |
| 2026-06-19 00:47 | /experiment-bridge | `refine-logs/R144_LEARNED_RELATION_FRONTEND_QA18_QA19_2026-06-19.md` | report | learned relation front-end report for QA18/QA19 |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_state_r144_learned/aggregate_summary.csv` | results | R144 learned-front-end QA18/QA19 aggregate summary |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_state_r144_learned/detector_summary.csv` | results | R144 learned relation detector aggregate metrics |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_state_r144_learned/seed_test_rows.json` | results | R144 per-seed test rows |
| 2026-06-19 00:47 | /experiment-bridge | `../babi_relation_paraphrase_stress_experiment.py` | implementation | QA18/QA19 relation paraphrase stress harness |
| 2026-06-19 00:47 | /experiment-bridge | `refine-logs/R145_RELATION_PARAPHRASE_STRESS_2026-06-19.md` | report | relation paraphrase stress report for QA18/QA19 |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_paraphrase_stress_r145/aggregate_summary.csv` | results | R145 relation paraphrase stress aggregate summary |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_paraphrase_stress_r145/detector_summary.csv` | results | R145 relation detector stress aggregate metrics |
| 2026-06-19 00:47 | /experiment-bridge | `../output/babi_relation_paraphrase_stress_r145/seed_test_rows.json` | results | R145 per-seed stress rows |
| 2026-06-19 01:30 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R092 DLL local-depth branch for TinyStories no-BP token learner |
| 2026-06-19 01:30 | /experiment-bridge | `refine-logs/R092_DLL_LOCAL_DEPTH_2026-06-19.md` | report | DLL local-depth tradeoff report |
| 2026-06-19 01:30 | /experiment-bridge | `../output/phase_binding_online_stream_r092_dll/summary.csv` | results | R092 DLL three-seed aggregate summary |
| 2026-06-19 01:30 | /experiment-bridge | `../output/phase_binding_online_stream_r092_dll/per_seed_summary.csv` | results | R092 DLL per-seed summary |
| 2026-06-19 01:30 | /experiment-bridge | `../output/phase_binding_online_stream_r092_dll/variant_probe_summary.csv` | results | R092 DLL smoke and medium probe summary |
| 2026-06-19 01:30 | /experiment-bridge | `../output/phase_binding_online_stream_r092_dll/generation_summary_aggregate.csv` | results | R092 DLL generation repetition aggregate |
| 2026-06-19 01:30 | /experiment-bridge | `../output/phase_binding_online_stream_r092_dll/selection.json` | results | R092 selected candidate and gate verdict |
| 2026-06-19 01:43 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R093 NoProp local denoising branch for TinyStories no-BP token learner |
| 2026-06-19 01:43 | /experiment-bridge | `refine-logs/R093_NOPROP_LOCAL_DENOISING_2026-06-19.md` | report | NoProp local-denoising tradeoff report |
| 2026-06-19 01:43 | /experiment-bridge | `../output/phase_binding_online_stream_r093_noprop/summary.csv` | results | R093 NoProp three-seed aggregate summary |
| 2026-06-19 01:43 | /experiment-bridge | `../output/phase_binding_online_stream_r093_noprop/per_seed_summary.csv` | results | R093 NoProp per-seed summary |
| 2026-06-19 01:43 | /experiment-bridge | `../output/phase_binding_online_stream_r093_noprop/variant_probe_summary.csv` | results | R093 NoProp smoke and medium probe summary |
| 2026-06-19 01:43 | /experiment-bridge | `../output/phase_binding_online_stream_r093_noprop/generation_summary_aggregate.csv` | results | R093 NoProp generation repetition aggregate |
| 2026-06-19 01:43 | /experiment-bridge | `../output/phase_binding_online_stream_r093_noprop/selection.json` | results | R093 selected candidate and gate verdict |
| 2026-06-19 01:53 | /experiment-bridge | `refine-logs/R096_PREP_DEEP_CALIBRATION_2026-06-19.md` | report | R096-prep calibrated deep NoProp positive result |
| 2026-06-19 01:53 | /experiment-bridge | `../output/phase_binding_online_stream_r096prep_deep_calib/summary.csv` | results | R096-prep calibrated deep NoProp aggregate summary |
| 2026-06-19 01:53 | /experiment-bridge | `../output/phase_binding_online_stream_r096prep_deep_calib/per_seed_summary.csv` | results | R096-prep per-seed summary |
| 2026-06-19 01:53 | /experiment-bridge | `../output/phase_binding_online_stream_r096prep_deep_calib/comparison_summary.csv` | results | R096-prep comparison against R092/R093/R134 |
| 2026-06-19 01:53 | /experiment-bridge | `../output/phase_binding_online_stream_r096prep_deep_calib/generation_summary_aggregate.csv` | results | R096-prep generation repetition aggregate |
| 2026-06-19 01:53 | /experiment-bridge | `../output/phase_binding_online_stream_r096prep_deep_calib/probe_summary.csv` | results | R096-prep smoke/medium calibration probes |
| 2026-06-19 01:53 | /experiment-bridge | `../output/phase_binding_online_stream_r096prep_deep_calib/selection.json` | results | R096-prep selected candidate and remaining gates |
| 2026-06-19 03:30 | /experiment-bridge | `refine-logs/R155_SYNTHETIC_OBJECT_CARRY_ABLATION_2026-06-19.md` | report | R155 synthetic object-carry difficulty and hop ablation |
| 2026-06-19 03:30 | /experiment-bridge | `../output/synthetic_object_carry_token_r155_m1_d0/summary.csv` | results | R155 m1_d0 baseline vs span summary |
| 2026-06-19 03:30 | /experiment-bridge | `../output/synthetic_object_carry_token_r155_m2_d0/summary.csv` | results | R155 m2_d0 baseline vs span summary |
| 2026-06-19 03:30 | /experiment-bridge | `../output/synthetic_object_carry_token_r155_m2_d2/summary.csv` | results | R155 m2_d2 baseline vs span summary |
| 2026-06-19 03:30 | /experiment-bridge | `../output/synthetic_object_carry_token_r155_m3_d4/summary.csv` | results | R155 m3_d4 baseline vs span summary |
| 2026-06-19 03:30 | /experiment-bridge | `../output/synthetic_object_carry_token_r155_m2_d2_hop1/summary.csv` | results | R155 m2_d2 span hop1 summary |
| 2026-06-19 03:30 | /experiment-bridge | `../output/synthetic_object_carry_token_r155_m2_d2_hop3/summary.csv` | results | R155 m2_d2 span hop3 summary |
| 2026-06-19 03:45 | /experiment-bridge | `refine-logs/R156_LOCAL_SPAN_ARBITRATION_2026-06-19.md` | report | R156 local span arbitration mixed-boundary report |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_gate_smoke/summary.csv` | results | R156 span_gate smoke summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d0_refs/summary.csv` | results | R156 m2_d0 baseline/span small reference summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d2_refs/summary.csv` | results | R156 m2_d2 baseline/span small reference summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d0_hard_t002/summary.csv` | results | R156 m2_d0 hard gate threshold 0.02 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d0_hard_t005/summary.csv` | results | R156 m2_d0 hard gate threshold 0.05 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d0_hard_t020/summary.csv` | results | R156 m2_d0 hard gate threshold 0.20 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d0_hard_t050/summary.csv` | results | R156 m2_d0 hard gate threshold 0.50 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d2_hard_t002/summary.csv` | results | R156 m2_d2 hard gate threshold 0.02 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d2_hard_t005/summary.csv` | results | R156 m2_d2 hard gate threshold 0.05 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d2_hard_t020/summary.csv` | results | R156 m2_d2 hard gate threshold 0.20 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_sweep_m2d2_hard_t050/summary.csv` | results | R156 m2_d2 hard gate threshold 0.50 summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_full_m2d0_hard_t020/summary.csv` | results | R156 full m2_d0 selected hard gate summary |
| 2026-06-19 03:45 | /experiment-bridge | `../output/synthetic_object_carry_token_r156_full_m2d2_hard_t020/summary.csv` | results | R156 full m2_d2 selected hard gate summary |
| 2026-06-19 03:56 | /experiment-bridge | `refine-logs/R157_QUERY_EVENT_ASSEMBLY_2026-06-19.md` | report | R157 query-seeded event assembly partial-positive report |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_event_assembly_smoke/summary.csv` | results | R157 event_assembly smoke summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_m2d0_event_default/summary.csv` | results | R157 m2_d0 medium event_assembly default summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_m2d2_event_default/summary.csv` | results | R157 m2_d2 medium event_assembly default summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_m2d2_event_w3_k3_h2/summary.csv` | results | R157 m2_d2 event sweep w3/k3/h2 summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_m2d2_event_w2_k5_h2/summary.csv` | results | R157 m2_d2 event sweep w2/k5/h2 summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_m2d2_event_w3_k5_h2/summary.csv` | results | R157 m2_d2 event sweep w3/k5/h2 summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_m2d2_event_w2_k3_h3/summary.csv` | results | R157 m2_d2 event sweep w2/k3/h3 summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_full_m2d0_event_default/summary.csv` | results | R157 full m2_d0 event_assembly default summary |
| 2026-06-19 03:56 | /experiment-bridge | `../output/synthetic_object_carry_token_r157_full_m2d2_event_w3_k5_h2/summary.csv` | results | R157 full m2_d2 event_assembly w3/k5/h2 summary |
| 2026-06-19 04:13 | /experiment-bridge | `refine-logs/R158_EVENT_CLEANUP_WTA_2026-06-19.md` | report | R158 event cleanup WTA positive report |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_event_cleanup_smoke/summary.csv` | results | R158 event_cleanup smoke summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_m2d2_cleanup_s1_i0/summary.csv` | results | R158 m2_d2 cleanup sweep scale1 inhibit0 summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_m2d2_cleanup_s2_i0/summary.csv` | results | R158 m2_d2 cleanup sweep scale2 inhibit0 summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_m2d2_cleanup_s2_i025/summary.csv` | results | R158 m2_d2 cleanup sweep scale2 inhibit0.25 summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_m2d2_cleanup_s3_i025/summary.csv` | results | R158 m2_d2 cleanup sweep scale3 inhibit0.25 summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_full_m2d0_cleanup_s3_i025/summary.csv` | results | R158 full m2_d0 event_cleanup summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_full_m2d2_cleanup_s3_i025/summary.csv` | results | R158 full m2_d2 event_cleanup summary |
| 2026-06-19 04:13 | /experiment-bridge | `../output/synthetic_object_carry_token_r158_full_m3d4_cleanup_s3_i025/summary.csv` | results | R158 full m3_d4 event_cleanup summary |
| 2026-06-19 04:40 | /experiment-bridge | `refine-logs/R159_BABI_EVENT_CLEANUP_2026-06-19.md` | report | R159 bAbI event_cleanup port negative-transfer report |
| 2026-06-19 04:40 | /experiment-bridge | `../output/babi_unified_event_cleanup_r159_smoke/summary.csv` | results | R159 event_cleanup QA2 smoke summary |
| 2026-06-19 04:40 | /experiment-bridge | `../output/babi_unified_event_cleanup_r159_smoke_ref_base/summary.csv` | results | R159 same-size microproto smoke reference summary |
| 2026-06-19 04:40 | /experiment-bridge | `../output/babi_unified_event_cleanup_r159_smoke_ref_span/summary.csv` | results | R159 same-size span_sparse smoke reference summary |
| 2026-06-19 04:40 | /experiment-bridge | `../output/babi_unified_event_cleanup_r159_full/summary.csv` | results | R159 full QA2 event_cleanup summary |
| 2026-06-19 04:40 | /experiment-bridge | `../output/babi_unified_event_cleanup_r159_full_ref_base/summary.csv` | results | R159 full QA2 same-setting microproto reference summary |
| 2026-06-19 05:25 | /experiment-bridge | `refine-logs/R160_LOCAL_ROLE_TRANSITION_2026-06-19.md` | report | R160 local role-transition positive report |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_smoke/summary.csv` | results | R160 role-transition QA2 smoke summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_smoke_ref_base_order128/summary.csv` | results | R160 same-order microproto smoke reference summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_smoke_no_gate/summary.csv` | results | R160 smoke no-gate ablation summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_smoke_no_direct/summary.csv` | results | R160 smoke no-direct-role-score ablation summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full/summary.csv` | results | R160 full QA2 role-transition seed0 summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_ref_base/summary.csv` | results | R160 full QA2 microproto seed0 reference summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_seed1/summary.csv` | results | R160 full QA2 role-transition seed1 summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_ref_base_seed1/summary.csv` | results | R160 full QA2 microproto seed1 reference summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_seed2/summary.csv` | results | R160 full QA2 role-transition seed2 summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_ref_base_seed2/summary.csv` | results | R160 full QA2 microproto seed2 reference summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_no_gate/summary.csv` | results | R160 full QA2 no-gate ablation summary |
| 2026-06-19 05:25 | /experiment-bridge | `../output/babi_unified_role_transition_r160_full_no_direct/summary.csv` | results | R160 full QA2 no-direct-role-score ablation summary |
| 2026-06-19 06:05 | /experiment-bridge | `refine-logs/R161_CHANNEL_FINAL_ROLE_TRANSITION_2026-06-19.md` | report | R161 channel-final role-transition positive report |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_smoke_default/summary.csv` | results | R161 QA2 smoke R160 default rerun summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_smoke_channel/summary.csv` | results | R161 QA2 smoke channel-gate summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_smoke_channel_final/summary.csv` | results | R161 QA2 smoke channel-final summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_smoke_channel_final_topk/summary.csv` | results | R161 QA2 smoke channel-final top-k inhibition summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_full_channel/summary.csv` | results | R161 QA2 full seed0 channel-gate summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_full_channel_final/summary.csv` | results | R161 QA2 full seed0 channel-final summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_full_channel_final_seed1/summary.csv` | results | R161 QA2 full seed1 channel-final summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_full_channel_final_seed2/summary.csv` | results | R161 QA2 full seed2 channel-final summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_qa3_channel_final_seed0/summary.csv` | results | R161 QA3 seed0 channel-final pressure summary |
| 2026-06-19 06:05 | /experiment-bridge | `../output/babi_unified_role_transition_r161_qa3_ref_base_seed0/summary.csv` | results | R161 QA3 seed0 microproto reference summary |
| 2026-06-19 06:30 | /experiment-bridge | `../babi_role_gate_alignment_diagnostic.py` | implementation | R162 role-gate center-difference diagnostic harness |
| 2026-06-19 06:30 | /experiment-bridge | `refine-logs/R162_ROLE_GATE_CENTER_DIFF_2026-06-19.md` | report | R162 role-gate center-difference boundary report |
| 2026-06-19 06:30 | /experiment-bridge | `../output/babi_role_gate_alignment_r162_qa2_smoke/role_gate_alignment_summary.csv` | results | R162 QA2 smoke center-difference summary |
| 2026-06-19 06:30 | /experiment-bridge | `../output/babi_role_gate_alignment_r162_qa2_20/role_gate_alignment_summary.csv` | results | R162 QA2 20-row center-difference summary |
| 2026-06-19 07:05 | /experiment-bridge | `refine-logs/R163_QA3_ROLE_TRANSITION_SEED_REPEAT_2026-06-19.md` | report | R163 QA3 role-transition three-seed repeat report |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_seed_repeat/aggregate_summary.csv` | results | R163 QA3 three-seed aggregate summary |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_seed_repeat/per_seed_summary.csv` | results | R163 QA3 per-seed summary |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_seed_repeat/paired_deltas.csv` | results | R163 QA3 paired deltas |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_ref_base_seed1/summary.csv` | results | R163 QA3 microproto seed1 summary |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_ref_base_seed2/summary.csv` | results | R163 QA3 microproto seed2 summary |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_channel_final_seed1/summary.csv` | results | R163 QA3 role-transition seed1 summary |
| 2026-06-19 07:05 | /experiment-bridge | `../output/babi_unified_role_transition_r163_qa3_channel_final_seed2/summary.csv` | results | R163 QA3 role-transition seed2 summary |
| 2026-06-19 07:35 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R164 optional role-event feature cache |
| 2026-06-19 07:35 | /experiment-bridge | `refine-logs/R164_ROLE_EVENT_CACHE_2026-06-19.md` | report | R164 role event cache speed report |
| 2026-06-19 07:35 | /experiment-bridge | `../output/babi_unified_role_transition_r164_event_cache/comparison_summary.csv` | results | R164 cache-vs-no-cache comparison summary |
| 2026-06-19 07:35 | /experiment-bridge | `../output/babi_unified_role_transition_r164_qa3_smoke_nocache/summary.csv` | results | R164 QA3 smoke no-cache summary |
| 2026-06-19 07:35 | /experiment-bridge | `../output/babi_unified_role_transition_r164_qa3_smoke_ecache2048/summary.csv` | results | R164 QA3 smoke event-cache summary |
| 2026-06-19 07:35 | /experiment-bridge | `../output/babi_unified_role_transition_r164_qa3_full_seed0_ecache4096/summary.csv` | results | R164 QA3 full seed0 event-cache summary |
| 2026-06-19 08:05 | /experiment-bridge | `refine-logs/R165_RELATION_TASK_SCAN_2026-06-19.md` | report | R165 relation-task role-transition mixed scan report |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_relation_scan/comparison_summary.csv` | results | R165 QA14/17/18 relation scan comparison summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_qa14_ref_base_seed0/summary.csv` | results | R165 QA14 microproto seed0 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_qa14_role_seed0/summary.csv` | results | R165 QA14 role-transition seed0 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_qa17_ref_base_seed0/summary.csv` | results | R165 QA17 microproto seed0 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_qa17_role_seed0/summary.csv` | results | R165 QA17 role-transition seed0 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_qa18_ref_base_seed0/summary.csv` | results | R165 QA18 microproto seed0 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r165_qa18_role_seed0/summary.csv` | results | R165 QA18 role-transition seed0 summary |
| 2026-06-19 08:35 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R166 optional role-score margin gate |
| 2026-06-19 08:35 | /experiment-bridge | `refine-logs/R166_ROLE_SCORE_GATE_BOUNDARY_2026-06-19.md` | report | R166 role-score gate negative boundary report |
| 2026-06-19 08:35 | /experiment-bridge | `../output/babi_unified_role_transition_r166_role_score_gate_medium/sweep_summary.csv` | results | R166 medium gate sweep detailed summary |
| 2026-06-19 08:35 | /experiment-bridge | `../output/babi_unified_role_transition_r166_role_score_gate_medium/variant_summary.csv` | results | R166 medium gate compact variant summary |
| 2026-06-19 08:35 | /experiment-bridge | `../output/babi_unified_role_transition_r166_qa14_medium_role_score0/summary.csv` | results | R166 QA14 role-score-zero probe summary |
| 2026-06-19 08:35 | /experiment-bridge | `../output/babi_unified_role_transition_r166_qa18_medium_role_score0/summary.csv` | results | R166 QA18 role-score-zero probe summary |
| 2026-06-19 09:10 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R167 branch-separated base/role prototype readout |
| 2026-06-19 09:10 | /experiment-bridge | `refine-logs/R167_BRANCH_SEPARATED_READOUT_2026-06-19.md` | report | R167 branch-separated readout mixed-positive report |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_role_transition_r167_branch_readout_medium/comparison_summary.csv` | results | R167 medium branch-vs-concat comparison summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_role_transition_r167_branch_readout_full/comparison_summary.csv` | results | R167 full seed0 branch-vs-concat comparison summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_role_transition_r167_qa14_full_branch_seed0/summary.csv` | results | R167 QA14 full branch seed0 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_role_transition_r167_qa17_full_branch_seed0/summary.csv` | results | R167 QA17 full branch seed0 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_role_transition_r167_qa18_full_branch_seed0/summary.csv` | results | R167 QA18 full branch seed0 summary |
| 2026-06-19 09:35 | /experiment-bridge | `refine-logs/R168_BRANCH_SCALE_SWEEP_2026-06-19.md` | report | R168 branch role-score scale sweep boundary report |
| 2026-06-19 09:35 | /experiment-bridge | `../output/babi_unified_role_transition_r168_branch_scale_sweep/sweep_summary.csv` | results | R168 medium branch scale sweep summary |
| 2026-06-19 09:35 | /experiment-bridge | `../output/babi_unified_role_transition_r168_branch_scale_full/comparison_summary.csv` | results | R168 full seed0 branch scale comparison summary |
| 2026-06-19 09:35 | /experiment-bridge | `../output/babi_unified_role_transition_r168_qa14_full_branch_r4_seed0/summary.csv` | results | R168 QA14 full branch r4 seed0 summary |
| 2026-06-19 09:35 | /experiment-bridge | `../output/babi_unified_role_transition_r168_qa17_full_branch_r4_seed0/summary.csv` | results | R168 QA17 full branch r4 seed0 summary |
| 2026-06-19 09:35 | /experiment-bridge | `../output/babi_unified_role_transition_r168_qa18_full_branch_r4_seed0/summary.csv` | results | R168 QA18 full branch r4 seed0 summary |
| 2026-06-19 10:05 | /experiment-bridge | `../babi_branch_arbitration_diagnostic.py` | implementation | R169 branch component arbitration diagnostic |
| 2026-06-19 10:05 | /experiment-bridge | `refine-logs/R169_BRANCH_ARBITRATION_DIAGNOSTIC_2026-06-19.md` | report | R169 branch arbitration diagnostic report |
| 2026-06-19 10:05 | /experiment-bridge | `../output/babi_branch_arbitration_r169_summary/component_summary.csv` | results | R169 branch component aggregate summary |
| 2026-06-19 10:05 | /experiment-bridge | `../output/babi_branch_arbitration_r169_summary/flip_summary.csv` | results | R169 branch flip aggregate summary |
| 2026-06-19 10:05 | /experiment-bridge | `../output/babi_branch_arbitration_r169_qa14_full/branch_component_summary.csv` | results | R169 QA14 full branch component summary |
| 2026-06-19 10:05 | /experiment-bridge | `../output/babi_branch_arbitration_r169_qa17_full/branch_component_summary.csv` | results | R169 QA17 full branch component summary |
| 2026-06-19 10:05 | /experiment-bridge | `../output/babi_branch_arbitration_r169_qa18_full/branch_component_summary.csv` | results | R169 QA18 full branch component summary |
| 2026-06-19 06:00 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R170 default-off local branch arbiters: prototype/WTA and adaptive base-margin inhibition |
| 2026-06-19 06:00 | /experiment-bridge | `refine-logs/R170_BRANCH_ARBITER_BOUNDARY_2026-06-19.md` | report | R170 branch arbiter boundary report |
| 2026-06-19 06:00 | /experiment-bridge | `../output/babi_unified_role_transition_r170_qa14_full_base_adapt_m02/summary.csv` | results | R170 QA14 full adaptive base-margin arbiter summary |
| 2026-06-19 06:00 | /experiment-bridge | `../output/babi_unified_role_transition_r170_qa17_full_base_adapt_m02/summary.csv` | results | R170 QA17 full adaptive base-margin arbiter summary |
| 2026-06-19 06:00 | /experiment-bridge | `../output/babi_unified_role_transition_r170_qa18_full_base_adapt_m02/summary.csv` | results | R170 QA18 full adaptive base-margin arbiter summary |
| 2026-06-19 06:50 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R171 parallel joint rescue prototype bank for branch-separated role-transition readout |
| 2026-06-19 06:50 | /experiment-bridge | `refine-logs/R171_JOINT_RESCUE_BRANCH_2026-06-19.md` | report | R171 joint rescue branch mixed report |
| 2026-06-19 06:50 | /experiment-bridge | `../output/babi_unified_role_transition_r171_qa14_full_joint2/summary.csv` | results | R171 QA14 full joint rescue scale2 summary |
| 2026-06-19 06:50 | /experiment-bridge | `../output/babi_unified_role_transition_r171_qa17_full_joint2/summary.csv` | results | R171 QA17 full joint rescue scale2 summary |
| 2026-06-19 06:50 | /experiment-bridge | `../output/babi_unified_role_transition_r171_qa18_full_joint2/summary.csv` | results | R171 QA18 full joint rescue scale2 summary |
| 2026-06-19 07:25 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R172 local top-k/WTA cleanup switches for joint rescue score delta |
| 2026-06-19 07:25 | /experiment-bridge | `refine-logs/R172_JOINT_CANDIDATE_CLEANUP_2026-06-19.md` | report | R172 joint candidate cleanup boundary report |
| 2026-06-19 07:25 | /experiment-bridge | `../output/babi_unified_role_transition_r172_qa14_medium_joint2_top4/summary.csv` | results | R172 QA14 medium joint rescue top4 cleanup summary |
| 2026-06-19 07:25 | /experiment-bridge | `../output/babi_unified_role_transition_r172_qa17_medium_joint2_top1/summary.csv` | results | R172 QA17 medium joint rescue top1 cleanup summary |
| 2026-06-19 07:25 | /experiment-bridge | `../output/babi_unified_role_transition_r172_qa18_medium_joint2_top1/summary.csv` | results | R172 QA18 medium joint rescue top1 cleanup summary |
| 2026-06-19 08:05 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R173 target/wrong-modulated joint suppression prototype bank |
| 2026-06-19 08:05 | /experiment-bridge | `refine-logs/R173_JOINT_SUPPRESSION_TRACE_2026-06-19.md` | report | R173 joint suppression trace mixed report |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r173_qa14_full_joint2_suppress1_m02/summary.csv` | results | R173 QA14 full joint suppression scale1 margin0.2 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r173_qa17_full_joint2_suppress1_m02/summary.csv` | results | R173 QA17 full joint suppression scale1 margin0.2 summary |
| 2026-06-19 08:05 | /experiment-bridge | `../output/babi_unified_role_transition_r173_qa18_full_joint2_suppress1_m02/summary.csv` | results | R173 QA18 full joint suppression scale1 margin0.2 summary |
| 2026-06-19 06:46 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R174 evidence-conditioned joint suppression modes and stats |
| 2026-06-19 06:46 | /experiment-bridge | `refine-logs/R174_EVIDENCE_CONDITIONED_SUPPRESSION_2026-06-19.md` | report | R174 evidence-conditioned suppression positive boundary report |
| 2026-06-19 06:46 | /experiment-bridge | `../output/babi_unified_role_transition_r174_qa17_medium_allwrong_repro/summary.csv` | results | R174 QA17 medium all-wrong default reproduction summary |
| 2026-06-19 06:46 | /experiment-bridge | `../output/babi_unified_role_transition_r174_qa14_full_protect_direct_dt005/summary.csv` | results | R174 QA14 full protect-direct suppression summary |
| 2026-06-19 06:46 | /experiment-bridge | `../output/babi_unified_role_transition_r174_qa17_full_protect_direct_dt005/summary.csv` | results | R174 QA17 full protect-direct suppression summary |
| 2026-06-19 06:46 | /experiment-bridge | `../output/babi_unified_role_transition_r174_qa18_full_protect_direct_dt005/summary.csv` | results | R174 QA18 full protect-direct suppression summary |
| 2026-06-19 06:52 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R175 default-off joint-aware branch arbiter variants |
| 2026-06-19 06:52 | /experiment-bridge | `refine-logs/R175_JOINT_AWARE_BRANCH_ARBITER_2026-06-19.md` | report | R175 joint-aware branch arbiter negative boundary report |
| 2026-06-19 06:52 | /experiment-bridge | `../output/babi_unified_role_transition_r175_qa14_medium_joint_arbiter/summary.csv` | results | R175 QA14 medium joint-aware arbiter summary |
| 2026-06-19 06:52 | /experiment-bridge | `../output/babi_unified_role_transition_r175_qa17_medium_joint_arbiter/summary.csv` | results | R175 QA17 medium joint-aware arbiter summary |
| 2026-06-19 06:52 | /experiment-bridge | `../output/babi_unified_role_transition_r175_qa18_medium_joint_arbiter/summary.csv` | results | R175 QA18 medium joint-aware arbiter summary |
| 2026-06-19 07:00 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R176 base-margin rescue branch controller |
| 2026-06-19 07:00 | /experiment-bridge | `refine-logs/R176_BASE_PROTECTION_RESCUE_CONTROLLER_2026-06-19.md` | report | R176 base-protection rescue controller negative boundary report |
| 2026-06-19 07:00 | /experiment-bridge | `../output/babi_unified_role_transition_r176_qa14_medium_base_margin_rescue_rr05/summary.csv` | results | R176 QA14 medium base-margin rescue rr0.5 summary |
| 2026-06-19 07:00 | /experiment-bridge | `../output/babi_unified_role_transition_r176_qa17_medium_base_margin_rescue_rr05/summary.csv` | results | R176 QA17 medium base-margin rescue rr0.5 summary |
| 2026-06-19 07:00 | /experiment-bridge | `../output/babi_unified_role_transition_r176_qa18_medium_base_margin_rescue_rr05/summary.csv` | results | R176 QA18 medium base-margin rescue rr0.5 summary |
| 2026-06-19 07:10 | /experiment-bridge | `../babi_branch_arbitration_diagnostic.py` | implementation | R177 joint-aware branch agreement diagnostic |
| 2026-06-19 07:10 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R177 agreement-sensitive base protection branch mode |
| 2026-06-19 07:10 | /experiment-bridge | `refine-logs/R177_AGREEMENT_BASE_PROTECTION_2026-06-19.md` | report | R177 agreement-sensitive base protection boundary report |
| 2026-06-19 07:10 | /experiment-bridge | `../output/babi_branch_arbitration_r177_qa14_joint_agreement/branch_pair_agreement_summary.csv` | results | R177 QA14 single-task pair agreement diagnostic |
| 2026-06-19 07:10 | /experiment-bridge | `../output/babi_branch_arbitration_r177_qa17_joint_agreement/branch_pair_agreement_summary.csv` | results | R177 QA17 single-task pair agreement diagnostic |
| 2026-06-19 07:10 | /experiment-bridge | `../output/babi_branch_arbitration_r177_qa18_joint_agreement/branch_pair_agreement_summary.csv` | results | R177 QA18 single-task pair agreement diagnostic |
| 2026-06-19 07:10 | /experiment-bridge | `../output/babi_unified_role_transition_r177_qa18_full_agree_base_protect/summary.csv` | results | R177 QA18 full agreement base-protect sanity summary |
| 2026-06-19 07:15 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R178 conflict-only binary prototype branch arbiter |
| 2026-06-19 07:15 | /experiment-bridge | `refine-logs/R178_CONFLICT_PROTO_ARBITER_2026-06-19.md` | report | R178 conflict-only prototype arbiter negative boundary report |
| 2026-06-19 07:15 | /experiment-bridge | `../output/babi_unified_role_transition_r178_qa14_medium_conflict_proto/summary.csv` | results | R178 QA14 medium conflict-proto summary |
| 2026-06-19 07:15 | /experiment-bridge | `../output/babi_unified_role_transition_r178_qa17_medium_conflict_proto/summary.csv` | results | R178 QA17 medium conflict-proto summary |
| 2026-06-19 07:15 | /experiment-bridge | `../output/babi_unified_role_transition_r178_qa18_medium_conflict_proto/summary.csv` | results | R178 QA18 medium conflict-proto summary |
| 2026-06-19 07:19 | /analyze-results | `refine-logs/R179_CONFLICT_FEATURE_SEPARABILITY_2026-06-19.md` | report | R179 conflict feature separability diagnostic report |
| 2026-06-19 07:19 | /analyze-results | `../output/babi_branch_arbitration_r179_conflict_feature_scan/conflict_feature_summary.csv` | results | R179 conflict feature summary |
| 2026-06-19 07:19 | /analyze-results | `../output/babi_branch_arbitration_r179_conflict_feature_scan/threshold_scan.csv` | results | R179 conflict threshold rule scan |
| 2026-06-19 07:19 | /analyze-results | `../output/babi_branch_arbitration_r179_conflict_feature_scan/best_aggregate_rules.csv` | results | R179 best aggregate margin rules |
| 2026-06-19 07:27 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R180 full-answer metrics for multi-token bAbI QA answers |
| 2026-06-19 07:27 | /experiment-bridge | `../output/babi_unified_token_qa_r180_full_answer_smoke/summary.csv` | results | R180 QA19 full-answer smoke summary |
| 2026-06-19 07:27 | /experiment-bridge | `../output/babi_unified_token_qa_r180_full_answer_smoke/predictions_sample.csv` | results | R180 QA19 full-answer smoke sample predictions |
| 2026-06-19 07:36 | /experiment-bridge | `../output/babi_unified_qa19_r181_microproto_medium_fullseq/summary.csv` | results | R181 QA19 microproto medium full-answer summary |
| 2026-06-19 07:36 | /experiment-bridge | `../output/babi_unified_qa19_r181_role_r174_medium_fullseq/summary.csv` | results | R181 QA19 R174-style role-transition medium full-answer summary |
| 2026-06-19 07:36 | /analyze-results | `../output/babi_unified_qa19_r181_medium_comparison/comparison_summary.csv` | results | R181 QA19 medium variant comparison summary |
| 2026-06-19 07:36 | /analyze-results | `../output/babi_unified_qa19_r181_medium_comparison/baseline_summary.csv` | results | R181 QA19 majority and random baseline summary |
| 2026-06-19 07:51 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R182 default-off answer-slot local prototype readout |
| 2026-06-19 07:51 | /experiment-bridge | `../output/babi_unified_qa19_r182_aslot_smoke/summary.csv` | results | R182 QA19 answer-slot smoke summary |
| 2026-06-19 07:51 | /experiment-bridge | `../output/babi_unified_qa19_r182_microproto_aslot_medium_s2/summary.csv` | results | R182 QA19 microproto answer-slot medium summary |
| 2026-06-19 07:51 | /experiment-bridge | `../output/babi_unified_qa19_r182_role_r174_aslot_medium_s2/summary.csv` | results | R182 QA19 role answer-slot medium summary |
| 2026-06-19 07:51 | /analyze-results | `../output/babi_unified_qa19_r182_answer_slot_comparison/comparison_summary.csv` | results | R182 QA19 answer-slot comparison summary |
| 2026-06-19 07:51 | /analyze-results | `../output/babi_unified_qa19_r182_answer_slot_comparison/slot_stats_summary.csv` | results | R182 QA19 answer-slot stats summary |
| 2026-06-19 08:02 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R183 role-hop answer-slot feature mode |
| 2026-06-19 08:02 | /experiment-bridge | `../output/babi_unified_qa19_r183_role_hop_aslot_smoke/summary.csv` | results | R183 QA19 role-hop answer-slot smoke summary |
| 2026-06-19 08:02 | /experiment-bridge | `../output/babi_unified_qa19_r183_role_hop_aslot_medium_s2/summary.csv` | results | R183 QA19 role-hop answer-slot medium summary |
| 2026-06-19 08:02 | /analyze-results | `../output/babi_unified_qa19_r183_role_hop_comparison/comparison_summary.csv` | results | R183 QA19 role-hop comparison summary |
| 2026-06-19 08:02 | /analyze-results | `../output/babi_unified_qa19_r183_role_hop_comparison/slot_feature_summary.csv` | results | R183 QA19 role-hop slot feature summary |
| 2026-06-19 08:12 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R184 parser-free edge-path answer-slot feature mode |
| 2026-06-19 08:12 | /experiment-bridge | `../output/babi_unified_qa19_r184_edge_path_aslot_smoke/summary.csv` | results | R184 QA19 edge-path answer-slot smoke summary |
| 2026-06-19 08:12 | /experiment-bridge | `../output/babi_unified_qa19_r184_edge_path_aslot_medium_s2/summary.csv` | results | R184 QA19 edge-path answer-slot medium summary |
| 2026-06-19 08:12 | /analyze-results | `../output/babi_unified_qa19_r184_edge_path_comparison/comparison_summary.csv` | results | R184 QA19 edge-path comparison summary |
| 2026-06-19 08:12 | /analyze-results | `../output/babi_unified_qa19_r184_edge_path_comparison/slot_feature_summary.csv` | results | R184 QA19 edge-path slot feature summary |
| 2026-06-19 08:18 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R185 edge-path WTA cleanup feature mode and local cleanup stats |
| 2026-06-19 08:18 | /experiment-bridge | `../output/babi_unified_qa19_r185_edge_path_wta_aslot_smoke/summary.csv` | results | R185 QA19 edge-path WTA smoke summary |
| 2026-06-19 08:18 | /experiment-bridge | `../output/babi_unified_qa19_r185_edge_path_wta_aslot_medium_s2/summary.csv` | results | R185 QA19 learned edge-path WTA medium summary |
| 2026-06-19 08:18 | /experiment-bridge | `../output/babi_unified_qa19_r185_edge_path_wta_supportonly_medium_s2/summary.csv` | results | R185 QA19 support-only edge-path WTA medium summary |
| 2026-06-19 08:18 | /experiment-bridge | `../output/babi_unified_qa19_r185_edge_path_wta_support_top2_medium_s2/summary.csv` | results | R185 QA19 support-top2 edge-path WTA medium summary |
| 2026-06-19 08:18 | /analyze-results | `../output/babi_unified_qa19_r185_edge_path_wta_comparison/comparison_summary.csv` | results | R185 QA19 edge-path WTA comparison summary |
| 2026-06-19 08:18 | /analyze-results | `../output/babi_unified_qa19_r185_edge_path_wta_comparison/slot_feature_summary.csv` | results | R185 QA19 edge-path WTA cleanup stats summary |
| 2026-06-19 08:34 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R186 edge-path soft multi-candidate answer-slot feature mode |
| 2026-06-19 08:34 | /experiment-bridge | `../output/babi_unified_qa19_r186_edge_path_soft_aslot_smoke/summary.csv` | results | R186 QA19 edge-path soft smoke summary |
| 2026-06-19 08:34 | /experiment-bridge | `../output/babi_unified_qa19_r186_edge_path_soft_aslot_medium_s2/summary.csv` | results | R186 QA19 soft t0.20 consistency0.50 medium summary |
| 2026-06-19 08:34 | /experiment-bridge | `../output/babi_unified_qa19_r186_edge_path_soft_support_aslot_medium_s2/summary.csv` | results | R186 QA19 soft t0.20 consistency0.00 medium summary |
| 2026-06-19 08:34 | /experiment-bridge | `../output/babi_unified_qa19_r186_edge_path_soft_temp05_aslot_medium_s2/summary.csv` | results | R186 QA19 soft t0.50 consistency0.50 medium summary |
| 2026-06-19 08:34 | /experiment-bridge | `../output/babi_unified_qa19_r186_edge_path_soft_support_temp1_aslot_medium_s2/summary.csv` | results | R186 QA19 soft t1.00 consistency0.00 medium summary |
| 2026-06-19 08:34 | /analyze-results | `../output/babi_unified_qa19_r186_edge_path_soft_comparison/comparison_summary.csv` | results | R186 QA19 soft edge-path comparison summary |
| 2026-06-19 08:34 | /analyze-results | `../output/babi_unified_qa19_r186_edge_path_soft_comparison/slot_feature_summary.csv` | results | R186 QA19 soft edge-path slot/cleanup stats summary |
| 2026-06-19 08:48 | /experiment-bridge | `../output/babi_unified_qa19_r187_r184_edge_path_aslot_medium_s1/summary.csv` | results | R187 QA19 R184 edge-path seed1 paired baseline summary |
| 2026-06-19 08:48 | /experiment-bridge | `../output/babi_unified_qa19_r187_r184_edge_path_aslot_medium_s2/summary.csv` | results | R187 QA19 R184 edge-path seed2 paired baseline summary |
| 2026-06-19 08:48 | /experiment-bridge | `../output/babi_unified_qa19_r187_r186_soft_t020_c000_aslot_medium_s1/summary.csv` | results | R187 QA19 R186 soft seed1 summary |
| 2026-06-19 08:48 | /experiment-bridge | `../output/babi_unified_qa19_r187_r186_soft_t020_c000_aslot_medium_s2/summary.csv` | results | R187 QA19 R186 soft seed2 summary |
| 2026-06-19 08:48 | /analyze-results | `../output/babi_unified_qa19_r187_soft_seed_repeat_comparison/paired_summary.csv` | results | R187 QA19 paired seed summary |
| 2026-06-19 08:48 | /analyze-results | `../output/babi_unified_qa19_r187_soft_seed_repeat_comparison/aggregate_summary.csv` | results | R187 QA19 three-seed aggregate summary |
| 2026-06-19 08:48 | /analyze-results | `../output/babi_unified_qa19_r187_soft_seed_repeat_comparison/slot_feature_summary.csv` | results | R187 QA19 seed-repeat slot/cleanup stats summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_qa19_r188_soft_t020_c050_aslot_medium_s1/summary.csv` | results | R188 QA19 soft t0.20 consistency0.50 seed1 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_qa19_r188_soft_t020_c050_aslot_medium_s2/summary.csv` | results | R188 QA19 soft t0.20 consistency0.50 seed2 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_qa19_r188_soft_t050_c050_aslot_medium_s1/summary.csv` | results | R188 QA19 soft t0.50 consistency0.50 seed1 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_qa19_r188_soft_t050_c050_aslot_medium_s2/summary.csv` | results | R188 QA19 soft t0.50 consistency0.50 seed2 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_qa19_r188_soft_t100_c000_aslot_medium_s1/summary.csv` | results | R188 QA19 soft t1.00 consistency0.00 seed1 summary |
| 2026-06-19 09:10 | /experiment-bridge | `../output/babi_unified_qa19_r188_soft_t100_c000_aslot_medium_s2/summary.csv` | results | R188 QA19 soft t1.00 consistency0.00 seed2 summary |
| 2026-06-19 09:10 | /analyze-results | `../output/babi_unified_qa19_r188_validation_selected_soft_comparison/candidate_summary.csv` | results | R188 QA19 soft candidate sweep summary |
| 2026-06-19 09:10 | /analyze-results | `../output/babi_unified_qa19_r188_validation_selected_soft_comparison/baseline_summary.csv` | results | R188 QA19 paired R184 baseline summary |
| 2026-06-19 09:10 | /analyze-results | `../output/babi_unified_qa19_r188_validation_selected_soft_comparison/selection_summary.csv` | results | R188 QA19 validation-selected per-seed summary |
| 2026-06-19 09:10 | /analyze-results | `../output/babi_unified_qa19_r188_validation_selected_soft_comparison/aggregate_summary.csv` | results | R188 QA19 validation-selected aggregate summary |
| 2026-06-19 09:26 | /experiment-bridge | `../output/babi_unified_qa19_r189_r184_edge_path_full_s0/summary.csv` | results | R189 QA19 full-limit R184 edge-path seed0 summary |
| 2026-06-19 09:26 | /experiment-bridge | `../output/babi_unified_qa19_r189_r186_soft_t020_c000_full_s0/summary.csv` | results | R189 QA19 full-limit R186 soft edge seed0 summary |
| 2026-06-19 09:26 | /analyze-results | `../output/babi_unified_qa19_r189_full_limit_comparison/comparison_summary.csv` | results | R189 QA19 full-limit comparison summary |
| 2026-06-19 09:26 | /analyze-results | `../output/babi_unified_qa19_r189_full_limit_comparison/slot_feature_summary.csv` | results | R189 QA19 full-limit slot/feature summary |
| 2026-06-19 09:45 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R190 edge-path soft direct score channel |
| 2026-06-19 09:45 | /experiment-bridge | `../output/babi_unified_qa19_r190_edge_path_soft_direct_smoke/summary.csv` | results | R190 QA19 direct edge-path smoke summary |
| 2026-06-19 09:45 | /experiment-bridge | `../output/babi_unified_qa19_r190_edge_path_soft_direct_medium_s0/summary.csv` | results | R190 QA19 direct edge-path scale1.0 medium summary |
| 2026-06-19 09:45 | /experiment-bridge | `../output/babi_unified_qa19_r190_edge_path_soft_direct_medium_s0_scale05/summary.csv` | results | R190 QA19 direct edge-path scale0.5 medium summary |
| 2026-06-19 09:45 | /experiment-bridge | `../output/babi_unified_qa19_r190_edge_path_soft_direct_medium_s0_scale20/summary.csv` | results | R190 QA19 direct edge-path scale2.0 medium summary |
| 2026-06-19 09:45 | /analyze-results | `../output/babi_unified_qa19_r190_edge_path_soft_direct_comparison/comparison_summary.csv` | results | R190 QA19 direct edge-path comparison summary |
| 2026-06-19 10:04 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R191 full prediction export and R192 direct margin gate |
| 2026-06-19 10:04 | /experiment-bridge | `../babi_prediction_flip_diagnostic.py` | implementation | R191 prediction flip diagnostic utility |
| 2026-06-19 10:04 | /experiment-bridge | `../output/babi_unified_qa19_r191_r186_soft_fullpred_s0/summary.csv` | results | R191 R186 soft full-prediction rerun summary |
| 2026-06-19 10:04 | /experiment-bridge | `../output/babi_unified_qa19_r191_r190_direct_s10_fullpred_s0/summary.csv` | results | R191 R190 direct full-prediction rerun summary |
| 2026-06-19 10:04 | /analyze-results | `../output/babi_unified_qa19_r191_direct_flip_diagnostic/flip_summary.csv` | results | R191 R190-vs-R186 helpful/harmful flip summary |
| 2026-06-19 10:04 | /analyze-results | `../output/babi_unified_qa19_r191_direct_flip_diagnostic/flip_by_target.csv` | results | R191 target-level flip summary |
| 2026-06-19 10:04 | /experiment-bridge | `../output/babi_unified_qa19_r192_direct_gate_medium_s0/summary.csv` | results | R192 direct gate medium summary |
| 2026-06-19 10:04 | /analyze-results | `../output/babi_unified_qa19_r192_direct_gate_comparison/comparison_summary.csv` | results | R192 direct gate comparison summary |
| 2026-06-19 10:04 | /analyze-results | `../output/babi_unified_qa19_r192_gate_vs_r186_flip_diagnostic/flip_summary.csv` | results | R192 gate versus R186 flip summary |
| 2026-06-19 10:04 | /analyze-results | `../output/babi_unified_qa19_r192_gate_vs_r190_flip_diagnostic/flip_summary.csv` | results | R192 gate versus R190 flip summary |
| 2026-06-19 10:19 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R193 answer-slot coupling local prototype readout |
| 2026-06-19 10:19 | /experiment-bridge | `../output/babi_unified_qa19_r193_slot_coupling_smoke/summary.csv` | results | R193 slot coupling smoke summary |
| 2026-06-19 10:19 | /experiment-bridge | `../output/babi_unified_qa19_r193_slot_coupling_scale05_medium_s0/summary.csv` | results | R193 slot coupling scale0.5 medium summary |
| 2026-06-19 10:19 | /experiment-bridge | `../output/babi_unified_qa19_r193_slot_coupling_medium_s0/summary.csv` | results | R193 slot coupling scale1.0 medium summary |
| 2026-06-19 10:19 | /experiment-bridge | `../output/babi_unified_qa19_r193_direct_coupling_medium_s0/summary.csv` | results | R193 direct+coupling medium summary |
| 2026-06-19 10:19 | /analyze-results | `../output/babi_unified_qa19_r193_slot_coupling_comparison/comparison_summary.csv` | results | R193 slot coupling comparison summary |
| 2026-06-19 10:19 | /analyze-results | `../output/babi_unified_qa19_r193_coupling_vs_r186_flip_diagnostic/flip_summary.csv` | results | R193 coupling scale1.0 versus R186 flip summary |
| 2026-06-19 10:19 | /analyze-results | `../output/babi_unified_qa19_r193_coupling_s05_vs_r186_flip_diagnostic/flip_summary.csv` | results | R193 coupling scale0.5 versus R186 flip summary |
| 2026-06-19 10:19 | /analyze-results | `../output/babi_unified_qa19_r193_direct_coupling_vs_r186_flip_diagnostic/flip_summary.csv` | results | R193 direct+coupling versus R186 flip summary |
| 2026-06-19 10:45 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R194 component-margin prediction logging |
| 2026-06-19 10:45 | /experiment-bridge | `../babi_component_margin_diagnostic.py` | implementation | R194 component-margin comparison utility |
| 2026-06-19 10:45 | /experiment-bridge | `../output/babi_unified_qa19_r194_r186_soft_components_s0_exact/summary.csv` | results | R194 exact R186 soft component rerun summary |
| 2026-06-19 10:45 | /experiment-bridge | `../output/babi_unified_qa19_r194_r193_coupling_components_s0_exact/summary.csv` | results | R194 exact R193 coupling component rerun summary |
| 2026-06-19 10:45 | /analyze-results | `../output/babi_unified_qa19_r194_component_margin_diagnostic/component_summary.csv` | results | R194 slot-level component-margin summary |
| 2026-06-19 10:45 | /analyze-results | `../output/babi_unified_qa19_r194_sequence_flip_diagnostic/flip_summary.csv` | results | R194 R193-vs-R186 full-answer sequence flip summary |
| 2026-06-19 11:05 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R195 answer-slot wrong-winner cleanup mechanism |
| 2026-06-19 11:05 | /experiment-bridge | `../babi_component_margin_diagnostic.py` | implementation | R195 cleanup component diagnostic fields |
| 2026-06-19 11:05 | /experiment-bridge | `../output/babi_unified_qa19_r195_wrong_cleanup_s010_medium_s0/summary.csv` | results | R195 cleanup scale0.10 medium summary |
| 2026-06-19 11:05 | /experiment-bridge | `../output/babi_unified_qa19_r195_wrong_cleanup_s025_medium_s0/summary.csv` | results | R195 cleanup scale0.25 medium summary |
| 2026-06-19 11:05 | /experiment-bridge | `../output/babi_unified_qa19_r195_wrong_cleanup_s050_medium_s0/summary.csv` | results | R195 cleanup scale0.50 medium summary |
| 2026-06-19 11:05 | /analyze-results | `../output/babi_unified_qa19_r195_wrong_cleanup_comparison/comparison_summary.csv` | results | R195 cleanup comparison summary |
| 2026-06-19 11:05 | /analyze-results | `../output/babi_unified_qa19_r195_s010_component_diagnostic/component_summary.csv` | results | R195 cleanup scale0.10 component diagnostic summary |
| 2026-06-19 11:25 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R196 evidence-protected wrong-cleanup gate |
| 2026-06-19 11:25 | /experiment-bridge | `../output/babi_unified_qa19_r196_protected_cleanup_t050_medium_s0/summary.csv` | results | R196 protected cleanup threshold0.5 medium summary |
| 2026-06-19 11:25 | /experiment-bridge | `../output/babi_unified_qa19_r196_protected_cleanup_t100_medium_s0/summary.csv` | results | R196 protected cleanup threshold1.0 medium summary |
| 2026-06-19 11:25 | /experiment-bridge | `../output/babi_unified_qa19_r196_protected_cleanup_t200_medium_s0/summary.csv` | results | R196 protected cleanup threshold2.0 medium summary |
| 2026-06-19 11:25 | /analyze-results | `../output/babi_unified_qa19_r196_protected_cleanup_comparison/comparison_summary.csv` | results | R196 protected cleanup comparison summary |
| 2026-06-19 11:25 | /analyze-results | `../output/babi_unified_qa19_r196_t200_component_diagnostic/component_summary.csv` | results | R196 threshold2.0 component diagnostic summary |
| 2026-06-19 11:46 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R197 ordered-pair conflict-local rescue mechanism |
| 2026-06-19 11:46 | /experiment-bridge | `../babi_component_margin_diagnostic.py` | implementation | R197 conflict-rescue component diagnostic fields |
| 2026-06-19 11:46 | /experiment-bridge | `../output/babi_unified_qa19_r197_conflict_rescue_smoke/summary.csv` | results | R197 conflict rescue smoke summary |
| 2026-06-19 11:46 | /experiment-bridge | `../output/babi_unified_qa19_r197_conflict_rescue_s010_medium_s0/summary.csv` | results | R197 conflict rescue scale0.10 medium summary |
| 2026-06-19 11:46 | /experiment-bridge | `../output/babi_unified_qa19_r197_conflict_rescue_s025_medium_s0/summary.csv` | results | R197 conflict rescue scale0.25 medium summary |
| 2026-06-19 11:46 | /experiment-bridge | `../output/babi_unified_qa19_r197_conflict_rescue_s050_medium_s0/summary.csv` | results | R197 conflict rescue scale0.50 medium summary |
| 2026-06-19 11:46 | /experiment-bridge | `../output/babi_unified_qa19_r197_conflict_rescue_s025_components_s0/summary.csv` | results | R197 conflict rescue scale0.25 component rerun summary |
| 2026-06-19 11:46 | /analyze-results | `../output/babi_unified_qa19_r197_conflict_rescue_comparison/comparison_summary.csv` | results | R197 conflict rescue comparison summary |
| 2026-06-19 11:46 | /analyze-results | `../output/babi_unified_qa19_r197_s010_vs_r193_flip_diagnostic/flip_summary.csv` | results | R197 scale0.10 versus R193 full-answer flip summary |
| 2026-06-19 11:46 | /analyze-results | `../output/babi_unified_qa19_r197_s025_vs_r193_flip_diagnostic/flip_summary.csv` | results | R197 scale0.25 versus R193 full-answer flip summary |
| 2026-06-19 11:46 | /analyze-results | `../output/babi_unified_qa19_r197_s050_vs_r193_flip_diagnostic/flip_summary.csv` | results | R197 scale0.50 versus R193 full-answer flip summary |
| 2026-06-19 11:46 | /analyze-results | `../output/babi_unified_qa19_r197_s025_component_diagnostic/component_summary.csv` | results | R197 scale0.25 slot-level component-margin summary |
| 2026-06-19 12:41 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R198 conflict-rescue prefix-consistency gate |
| 2026-06-19 12:41 | /experiment-bridge | `../output/babi_unified_qa19_r198_prefix_gate_smoke/summary.csv` | results | R198 prefix gate smoke summary |
| 2026-06-19 12:41 | /experiment-bridge | `../output/babi_unified_qa19_r198_prefix_gate_obspred_s025_medium_s0/summary.csv` | results | R198 observed_pred prefix gate medium summary |
| 2026-06-19 12:41 | /experiment-bridge | `../output/babi_unified_qa19_r198_prefix_gate_obspred_m005_s025_medium_s0/summary.csv` | results | R198 observed_pred_margin0.05 prefix gate medium summary |
| 2026-06-19 12:41 | /experiment-bridge | `../output/babi_unified_qa19_r198_prefix_gate_margin005_s025_medium_s0/summary.csv` | results | R198 margin0.05 prefix gate medium summary |
| 2026-06-19 12:41 | /experiment-bridge | `../output/babi_unified_qa19_r198_prefix_gate_margin020_s025_medium_s0/summary.csv` | results | R198 margin0.20 prefix gate medium summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_prefix_gate_comparison/comparison_summary.csv` | results | R198 prefix gate comparison summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_prefix_gate_comparison/gate_stats_summary.csv` | results | R198 prefix gate stats summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_obspred_vs_r193_flip_diagnostic/flip_summary.csv` | results | R198 observed_pred versus R193 flip summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_obspred_m005_vs_r193_flip_diagnostic/flip_summary.csv` | results | R198 observed_pred_margin0.05 versus R193 flip summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_margin005_vs_r193_flip_diagnostic/flip_summary.csv` | results | R198 margin0.05 versus R193 flip summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_margin020_vs_r193_flip_diagnostic/flip_summary.csv` | results | R198 margin0.20 versus R193 flip summary |
| 2026-06-19 12:41 | /analyze-results | `../output/babi_unified_qa19_r198_margin020_vs_r197_s025_flip_diagnostic/flip_summary.csv` | results | R198 margin0.20 versus R197 s0.25 flip summary |
| 2026-06-19 12:57 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R199 predicted-prefix eligibility training path |
| 2026-06-19 12:57 | /experiment-bridge | `../output/babi_unified_qa19_r199_predprefix_conflict_smoke/summary.csv` | results | R199 predicted-prefix conflict smoke summary |
| 2026-06-19 12:57 | /experiment-bridge | `../output/babi_unified_qa19_r199_predprefix_conflict_s025_medium_s0/summary.csv` | results | R199 predicted-prefix conflict scale0.25 medium summary |
| 2026-06-19 12:57 | /experiment-bridge | `../output/babi_unified_qa19_r199_predprefix_coupling_conflict_s025_medium_s0/summary.csv` | results | R199 predicted-prefix coupling+conflict scale0.25 medium summary |
| 2026-06-19 12:57 | /experiment-bridge | `../output/babi_unified_qa19_r199_predprefix_coupling_conflict_s010_medium_s0/summary.csv` | results | R199 predicted-prefix coupling+conflict scale0.10 medium summary |
| 2026-06-19 12:57 | /analyze-results | `../output/babi_unified_qa19_r199_predprefix_comparison/comparison_summary.csv` | results | R199 predicted-prefix comparison summary |
| 2026-06-19 12:57 | /analyze-results | `../output/babi_unified_qa19_r199_predprefix_comparison/predprefix_stats_summary.csv` | results | R199 predicted-prefix stats summary |
| 2026-06-19 12:57 | /analyze-results | `../output/babi_unified_qa19_r199_pp_conflict_s025_vs_r193_flip_diagnostic/flip_summary.csv` | results | R199 predicted-prefix conflict scale0.25 versus R193 flip summary |
| 2026-06-19 12:57 | /analyze-results | `../output/babi_unified_qa19_r199_pp_coupling_conflict_s025_vs_r193_flip_diagnostic/flip_summary.csv` | results | R199 predicted-prefix coupling+conflict scale0.25 versus R193 flip summary |
| 2026-06-19 12:57 | /analyze-results | `../output/babi_unified_qa19_r199_pp_coupling_conflict_s010_vs_r193_flip_diagnostic/flip_summary.csv` | results | R199 predicted-prefix coupling+conflict scale0.10 versus R193 flip summary |
| 2026-06-19 12:57 | /analyze-results | `../output/babi_unified_qa19_r199_pp_coupling_conflict_s010_vs_r197_s010_flip_diagnostic/flip_summary.csv` | results | R199 predicted-prefix coupling+conflict scale0.10 versus R197 s0.10 flip summary |
| 2026-06-19 13:22 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R200 predicted-prefix target top-k gate and local learning-rate scale |
| 2026-06-19 13:22 | /experiment-bridge | `../output/babi_unified_qa19_r200_predprefix_topk4_lr025_smoke/summary.csv` | results | R200 top4 lr0.25 smoke summary |
| 2026-06-19 13:22 | /experiment-bridge | `../output/babi_unified_qa19_r200_predprefix_top1_s010_medium_s0/summary.csv` | results | R200 predicted-prefix top1 medium summary |
| 2026-06-19 13:22 | /experiment-bridge | `../output/babi_unified_qa19_r200_predprefix_top2_s010_medium_s0/summary.csv` | results | R200 predicted-prefix top2 medium summary |
| 2026-06-19 13:22 | /experiment-bridge | `../output/babi_unified_qa19_r200_predprefix_topk4_s010_medium_s0/summary.csv` | results | R200 predicted-prefix top4 medium summary |
| 2026-06-19 13:22 | /experiment-bridge | `../output/babi_unified_qa19_r200_predprefix_lr025_s010_medium_s0/summary.csv` | results | R200 predicted-prefix lr0.25 medium summary |
| 2026-06-19 13:22 | /experiment-bridge | `../output/babi_unified_qa19_r200_predprefix_topk4_lr025_s010_medium_s0/summary.csv` | results | R200 predicted-prefix top4 lr0.25 medium summary |
| 2026-06-19 13:22 | /analyze-results | `../output/babi_unified_qa19_r200_top1_vs_r193_flip_diagnostic/flip_summary.csv` | results | R200 top1 versus R193 full-answer flip summary |
| 2026-06-19 13:22 | /analyze-results | `../output/babi_unified_qa19_r200_top2_vs_r193_flip_diagnostic/flip_summary.csv` | results | R200 top2 versus R193 full-answer flip summary |
| 2026-06-19 13:22 | /analyze-results | `../output/babi_unified_qa19_r200_top1_vs_r199_flip_diagnostic/flip_summary.csv` | results | R200 top1 versus R199 full-answer flip summary |
| 2026-06-19 13:22 | /analyze-results | `../output/babi_unified_qa19_r200_top2_vs_r199_flip_diagnostic/flip_summary.csv` | results | R200 top2 versus R199 full-answer flip summary |
| 2026-06-19 13:31 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R201 decoupled predicted-prefix coupling wrong-credit switch |
| 2026-06-19 13:31 | /experiment-bridge | `../output/babi_unified_qa19_r201_targetonly_cc_smoke/summary.csv` | results | R201 target-only coupling+conflict smoke summary |
| 2026-06-19 13:31 | /experiment-bridge | `../output/babi_unified_qa19_r201_pp_coupling_s010_medium_s0/summary.csv` | results | R201 predicted-prefix coupling medium summary |
| 2026-06-19 13:31 | /experiment-bridge | `../output/babi_unified_qa19_r201_pp_coupling_targetonly_s010_medium_s0/summary.csv` | results | R201 predicted-prefix coupling target-only medium summary |
| 2026-06-19 13:31 | /experiment-bridge | `../output/babi_unified_qa19_r201_pp_coupling_conflict_targetonly_s010_medium_s0/summary.csv` | results | R201 predicted-prefix coupling+conflict target-only medium summary |
| 2026-06-19 13:31 | /analyze-results | `../output/babi_unified_qa19_r201_coupling_targetonly_vs_r193_flip_diagnostic/flip_summary.csv` | results | R201 coupling target-only versus R193 full-answer flip summary |
| 2026-06-19 13:31 | /analyze-results | `../output/babi_unified_qa19_r201_coupling_targetonly_vs_r199_flip_diagnostic/flip_summary.csv` | results | R201 coupling target-only versus R199 full-answer flip summary |
| 2026-06-19 13:31 | /analyze-results | `../output/babi_unified_qa19_r201_coupling_vs_r193_flip_diagnostic/flip_summary.csv` | results | R201 coupling versus R193 full-answer flip summary |
| 2026-06-19 13:31 | /analyze-results | `../output/babi_unified_qa19_r201_coupling_vs_r199_flip_diagnostic/flip_summary.csv` | results | R201 coupling versus R199 full-answer flip summary |
| 2026-06-19 13:40 | /experiment-bridge | `../output/babi_unified_qa19_r193_slot_coupling_medium_s1/summary.csv` | results | R202 R193 seed1 medium summary |
| 2026-06-19 13:40 | /experiment-bridge | `../output/babi_unified_qa19_r193_slot_coupling_medium_s2/summary.csv` | results | R202 R193 seed2 medium summary |
| 2026-06-19 13:40 | /experiment-bridge | `../output/babi_unified_qa19_r199_predprefix_coupling_conflict_s010_medium_s1/summary.csv` | results | R202 R199 seed1 medium summary |
| 2026-06-19 13:40 | /experiment-bridge | `../output/babi_unified_qa19_r199_predprefix_coupling_conflict_s010_medium_s2/summary.csv` | results | R202 R199 seed2 medium summary |
| 2026-06-19 13:40 | /experiment-bridge | `../output/babi_unified_qa19_r201_pp_coupling_targetonly_s010_medium_s1/summary.csv` | results | R202 R201 target-only seed1 medium summary |
| 2026-06-19 13:40 | /experiment-bridge | `../output/babi_unified_qa19_r201_pp_coupling_targetonly_s010_medium_s2/summary.csv` | results | R202 R201 target-only seed2 medium summary |
| 2026-06-19 13:48 | /experiment-bridge | `../babi_slot1_error_source_diagnostic.py` | implementation | R203 QA19 slot1 error-source diagnostic script |
| 2026-06-19 13:48 | /experiment-bridge | `../output/babi_unified_qa19_r203_r193_components_s0/prediction_components.csv` | results | R203 R193 component-margin rerun rows |
| 2026-06-19 13:48 | /experiment-bridge | `../output/babi_unified_qa19_r203_r199_components_s0/prediction_components.csv` | results | R203 R199 component-margin rerun rows |
| 2026-06-19 13:48 | /experiment-bridge | `../output/babi_unified_qa19_r203_r201_targetonly_components_s0/prediction_components.csv` | results | R203 R201 target-only component-margin rerun rows |
| 2026-06-19 13:48 | /analyze-results | `../output/babi_unified_qa19_r203_r193_slot1_error_source_s0/slot1_error_summary.csv` | results | R203 R193 slot1 error-source summary |
| 2026-06-19 13:48 | /analyze-results | `../output/babi_unified_qa19_r203_r199_slot1_error_source_s0/slot1_error_summary.csv` | results | R203 R199 slot1 error-source summary |
| 2026-06-19 13:48 | /analyze-results | `../output/babi_unified_qa19_r203_r201_targetonly_slot1_error_source_s0/slot1_error_summary.csv` | results | R203 R201 target-only slot1 error-source summary |
| 2026-06-19 13:53 | /experiment-bridge | `../babi_slot0_path_source_diagnostic.py` | implementation | R204 QA19 slot0 path-source diagnostic script |
| 2026-06-19 13:53 | /analyze-results | `../output/babi_unified_qa19_r204_r193_slot0_path_source_s0/slot0_path_summary.csv` | results | R204 R193 slot0 path-source summary |
| 2026-06-19 13:53 | /analyze-results | `../output/babi_unified_qa19_r204_r199_slot0_path_source_s0/slot0_path_summary.csv` | results | R204 R199 slot0 path-source summary |
| 2026-06-19 13:53 | /analyze-results | `../output/babi_unified_qa19_r204_r201_targetonly_slot0_path_source_s0/slot0_path_summary.csv` | results | R204 R201 target-only slot0 path-source summary |
| 2026-06-19 14:06 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R205 slot0 conflict-rescue feature fallback for min-slot0 |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_rescue_smoke/summary.csv` | results | R205 slot0 conflict rescue smoke summary |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_s005_medium_s0/summary.csv` | results | R205 slot0 conflict scale0.05 medium summary |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_s010_medium_s0/summary.csv` | results | R205 slot0 conflict scale0.10 medium summary |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_s025_medium_s0/summary.csv` | results | R205 slot0 conflict scale0.25 seed0 medium summary |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_s025_medium_s1/summary.csv` | results | R205 slot0 conflict scale0.25 seed1 medium summary |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_s025_medium_s2/summary.csv` | results | R205 slot0 conflict scale0.25 seed2 medium summary |
| 2026-06-19 14:06 | /analyze-results | `../output/babi_unified_qa19_r205_s025_vs_r193_flip_diagnostic/flip_summary.csv` | results | R205 scale0.25 versus R193 flip summary |
| 2026-06-19 14:06 | /experiment-bridge | `../output/babi_unified_qa19_r205_slot0_conflict_s025_components_s0/prediction_components.csv` | results | R205 scale0.25 component-margin rows |
| 2026-06-19 14:06 | /analyze-results | `../output/babi_unified_qa19_r205_s025_slot0_path_source_s0/slot0_path_summary.csv` | results | R205 scale0.25 slot0 path-source summary |
| 2026-06-19 14:06 | /analyze-results | `../output/babi_unified_qa19_r205_s025_slot1_error_source_s0/slot1_error_summary.csv` | results | R205 scale0.25 slot1 error-source summary |
| 2026-06-19 14:36 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R206 conflict-rescue positive-support safety gate |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support010_smoke/summary.csv` | results | R206 support0.10 smoke summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support050_smoke/summary.csv` | results | R206 support0.50 smoke summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support100_smoke/summary.csv` | results | R206 support1.00 smoke summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support150_smoke/summary.csv` | results | R206 support1.50 smoke summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support200_smoke/summary.csv` | results | R206 support2.00 smoke summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support100_medium_s0/summary.csv` | results | R206 support1.00 medium seed0 summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support150_medium_s0/summary.csv` | results | R206 support1.50 medium seed0 summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support200_medium_s0/summary.csv` | results | R206 support2.00 medium seed0 summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support200_medium_s1/summary.csv` | results | R206 support2.00 medium seed1 summary |
| 2026-06-19 14:36 | /experiment-bridge | `../output/babi_unified_qa19_r206_support200_medium_s2/summary.csv` | results | R206 support2.00 medium seed2 summary |
| 2026-06-19 14:36 | /analyze-results | `../output/babi_unified_qa19_r206_sup2_vs_r193_flip_diagnostic/flip_summary.csv` | results | R206 support2.00 versus R193 flip summary |
| 2026-06-19 14:36 | /analyze-results | `../output/babi_unified_qa19_r206_sup2_vs_r205_flip_diagnostic/flip_summary.csv` | results | R206 support2.00 versus R205 flip summary |
| 2026-06-19 14:36 | /analyze-results | `../output/babi_unified_qa19_r206_sup15_vs_r193_flip_diagnostic/flip_summary.csv` | results | R206 support1.50 versus R193 flip summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain105_smoke/summary.csv` | results | R207 gain1.05 smoke summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain110_smoke/summary.csv` | results | R207 gain1.10 smoke summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain115_smoke/summary.csv` | results | R207 gain1.15 smoke summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain120_smoke/summary.csv` | results | R207 gain1.20 smoke summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain125_smoke/summary.csv` | results | R207 gain1.25 smoke summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain120_medium_s0/summary.csv` | results | R207 gain1.20 medium seed0 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain120_medium_s1/summary.csv` | results | R207 gain1.20 medium seed1 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain120_medium_s2/summary.csv` | results | R207 gain1.20 medium seed2 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain125_medium_s0/summary.csv` | results | R207 gain1.25 medium seed0 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain125_medium_s1/summary.csv` | results | R207 gain1.25 medium seed1 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain125_medium_s2/summary.csv` | results | R207 gain1.25 medium seed2 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain130_medium_s0/summary.csv` | results | R207 gain1.30 medium seed0 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain130_medium_s1/summary.csv` | results | R207 gain1.30 medium seed1 summary |
| 2026-06-19 15:18 | /experiment-bridge | `../output/phase_binding_online_stream_r207_gain130_medium_s2/summary.csv` | results | R207 gain1.30 medium seed2 summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_local_gain_s010_lr002_smoke/summary.csv` | results | R208 local gain strength0.10 lr0.002 smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_local_gain_s020_lr002_smoke/summary.csv` | results | R208 local gain strength0.20 lr0.002 smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_local_gain_s010_lr005_smoke/summary.csv` | results | R208 local gain strength0.10 lr0.005 smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_local_gain_s010_lr002_marg_smoke/summary.csv` | results | R208 local gain margin-protected smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_margin_gain_c05_smoke/summary.csv` | results | R208 margin gain center0.5 smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_margin_gain_c10_smoke/summary.csv` | results | R208 margin gain center1.0 smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_margin_gain_c15_smoke/summary.csv` | results | R208 margin gain center1.5 smoke summary |
| 2026-06-19 15:43 | /experiment-bridge | `../output/phase_binding_online_stream_r208_margin_gain_g130_c10_smoke/summary.csv` | results | R208 margin gain1.30 center1.0 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R209 CE-mode local readout gain update switch |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg120_s010_lr0002_clip10_smoke/summary.csv` | results | R209 CE local gain bg1.20 s0.10 lr0.002 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg120_s020_lr0002_clip10_smoke/summary.csv` | results | R209 CE local gain bg1.20 s0.20 lr0.002 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg120_s010_lr0005_clip05_smoke/summary.csv` | results | R209 CE local gain bg1.20 s0.10 lr0.005 clip0.5 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg115_s020_lr0002_clip10_smoke/summary.csv` | results | R209 CE local gain bg1.15 s0.20 lr0.002 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg120_s035_lr0010_clip10_smoke/summary.csv` | results | R209 CE local gain bg1.20 s0.35 lr0.010 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg120_s050_lr0005_clip10_smoke/summary.csv` | results | R209 CE local gain bg1.20 s0.50 lr0.005 smoke summary |
| 2026-06-19 14:56 | /experiment-bridge | `../output/phase_binding_online_stream_r209_ce_gain_bg120_s035_lr0010_clip10_medium_s0/summary.csv` | results | R209 CE local gain best-smoke medium seed0 summary |
| 2026-06-19 15:07 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R210 dynamic local gain scope and retention reset evaluation switch |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_fixed_gain120_retreset_smoke/summary.csv` | results | R210 fixed gain1.20 reset-retention smoke summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_ce_gain_persist_bg120_s035_lr0010_retreset_smoke/summary.csv` | results | R210 persistent CE local gain bg1.20 reset-retention smoke summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_ce_gain_dynamic_bg120_s035_lr0010_retreset_smoke/summary.csv` | results | R210 dynamic CE local gain bg1.20 reset-retention smoke summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_ce_gain_dynamic_bg115_s050_lr0010_retreset_smoke/summary.csv` | results | R210 dynamic CE local gain bg1.15 s0.50 lr0.010 smoke summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_ce_gain_dynamic_bg115_s075_lr0010_retreset_smoke/summary.csv` | results | R210 dynamic CE local gain bg1.15 s0.75 lr0.010 smoke summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_ce_gain_dynamic_bg115_s050_lr0020_retreset_smoke/summary.csv` | results | R210 dynamic CE local gain bg1.15 s0.50 lr0.020 smoke summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_fixed_gain120_retreset_medium_s0/summary.csv` | results | R210 fixed gain1.20 reset-retention medium seed0 summary |
| 2026-06-19 15:07 | /experiment-bridge | `../output/phase_binding_online_stream_r210_ce_gain_dynamic_bg120_s035_lr0010_retreset_medium_s0/summary.csv` | results | R210 dynamic CE local gain bg1.20 reset-retention medium seed0 summary |
| 2026-06-19 15:19 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R211 transient feature calibration wrapper and feature calibration dynamic scope |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s010_lr0005_d32_retreset_smoke/summary.csv` | results | R211 transient feature calibration strength0.10 lr0.005 smoke summary |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s025_lr0005_d32_retreset_smoke/summary.csv` | results | R211 transient feature calibration strength0.25 lr0.005 smoke summary |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s025_lr0010_d32_retreset_smoke/summary.csv` | results | R211 transient feature calibration strength0.25 lr0.010 smoke summary |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s050_lr0005_d32_retreset_smoke/summary.csv` | results | R211 transient feature calibration strength0.50 lr0.005 smoke summary |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s025_lr0010_d32_retreset_medium_s0/summary.csv` | results | R211 transient feature calibration medium seed0 summary |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s025_lr0010_d32_retreset_medium_s1/summary.csv` | results | R211 transient feature calibration medium seed1 summary |
| 2026-06-19 15:19 | /experiment-bridge | `../output/phase_binding_online_stream_r211_transfeat_s025_lr0010_d32_retreset_medium_s2/summary.csv` | results | R211 transient feature calibration medium seed2 summary |
| 2026-06-19 15:25 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R212 top-k and near-miss controls for transient feature calibration |
| 2026-06-19 15:25 | /experiment-bridge | `../output/phase_binding_online_stream_r212_transfeat_s025_lr0010_score16_update8_retreset_smoke/summary.csv` | results | R212 transient feature calibration score16 update8 smoke summary |
| 2026-06-19 15:25 | /experiment-bridge | `../output/phase_binding_online_stream_r212_transfeat_s025_lr0010_score8_update4_retreset_smoke/summary.csv` | results | R212 transient feature calibration score8 update4 smoke summary |
| 2026-06-19 15:25 | /experiment-bridge | `../output/phase_binding_online_stream_r212_transfeat_s025_lr0010_score32_update16_retreset_smoke/summary.csv` | results | R212 transient feature calibration score32 update16 smoke summary |
| 2026-06-19 15:25 | /experiment-bridge | `../output/phase_binding_online_stream_r212_transfeat_s025_lr0010_score16_update8_margin05_retreset_smoke/summary.csv` | results | R212 transient feature calibration score16 update8 margin0.5 smoke summary |
| 2026-06-19 15:31 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R213 soft rank and margin weighting controls for transient feature calibration |
| 2026-06-19 15:31 | /experiment-bridge | `../output/phase_binding_online_stream_r213_transfeat_s025_lr0010_ranktau8_retreset_smoke/summary.csv` | results | R213 transient feature calibration rank tau8 smoke summary |
| 2026-06-19 15:31 | /experiment-bridge | `../output/phase_binding_online_stream_r213_transfeat_s025_lr0010_ranktau16_retreset_smoke/summary.csv` | results | R213 transient feature calibration rank tau16 smoke summary |
| 2026-06-19 15:31 | /experiment-bridge | `../output/phase_binding_online_stream_r213_transfeat_s025_lr0010_margintau1_retreset_smoke/summary.csv` | results | R213 transient feature calibration margin tau1 smoke summary |
| 2026-06-19 15:31 | /experiment-bridge | `../output/phase_binding_online_stream_r213_transfeat_s025_lr0010_ranktau16_margintau2_retreset_smoke/summary.csv` | results | R213 transient feature calibration rank tau16 margin tau2 smoke summary |
| 2026-06-19 15:42 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R214 transient wrong-winner-only inhibition wrapper |
| 2026-06-19 15:42 | /experiment-bridge | `../output/phase_binding_online_stream_r214_transfeat_wininhib_s010_lr0005_retreset_smoke/summary.csv` | results | R214 transient winner inhibition strength0.10 lr0.005 smoke summary |
| 2026-06-19 15:42 | /experiment-bridge | `../output/phase_binding_online_stream_r214_transfeat_wininhib_s025_lr0005_retreset_smoke/summary.csv` | results | R214 transient winner inhibition strength0.25 lr0.005 smoke summary |
| 2026-06-19 15:42 | /experiment-bridge | `../output/phase_binding_online_stream_r214_transfeat_wininhib_s025_lr0010_retreset_smoke/summary.csv` | results | R214 transient winner inhibition strength0.25 lr0.010 smoke summary |
| 2026-06-19 15:42 | /experiment-bridge | `../output/phase_binding_online_stream_r214_transfeat_wininhib_s050_lr0005_retreset_smoke/summary.csv` | results | R214 transient winner inhibition strength0.50 lr0.005 smoke summary |
| 2026-06-19 15:42 | /experiment-bridge | `../output/phase_binding_online_stream_r214_transfeat_wininhib_s025_lr0010_retreset_medium_s0/summary.csv` | results | R214 transient winner inhibition strength0.25 lr0.010 medium seed0 summary |
| 2026-06-19 15:42 | /experiment-bridge | `../output/phase_binding_online_stream_r214_transfeat_wininhib_s025_lr0005_retreset_medium_s0/summary.csv` | results | R214 transient winner inhibition strength0.25 lr0.005 medium seed0 summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay099_s025_lr0005_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.99 lr0.005 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay095_s025_lr0005_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.95 lr0.005 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay090_s025_lr0005_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.90 lr0.005 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay0999_s025_lr0005_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.999 lr0.005 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay09995_s025_lr0005_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.9995 lr0.005 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay0999_s025_lr0010_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.999 lr0.010 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay09995_s025_lr0010_retreset_smoke/summary.csv` | results | R215 winner inhibition decay0.9995 lr0.010 smoke summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay0999_s025_lr0010_retreset_medium_s0/summary.csv` | results | R215 winner inhibition decay0.999 lr0.010 medium seed0 summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay0999_s025_lr0010_retreset_medium_s1/summary.csv` | results | R215 winner inhibition decay0.999 lr0.010 medium seed1 summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay0999_s025_lr0010_retreset_medium_s2/summary.csv` | results | R215 winner inhibition decay0.999 lr0.010 medium seed2 summary |
| 2026-06-19 15:58 | /experiment-bridge | `../output/phase_binding_online_stream_r215_transfeat_wininhib_decay09995_s025_lr0010_retreset_medium_s0/summary.csv` | results | R215 winner inhibition decay0.9995 lr0.010 medium seed0 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R216 role branch arbiter repeated-evidence min-count safety gate |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa18_smoke_conflict_proto_mc2/summary.csv` | results | R216 QA18 smoke conflict_proto min-count2 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa14_medium_conflict_proto_mc2/summary.csv` | results | R216 QA14 medium conflict_proto min-count2 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa17_medium_conflict_proto_mc2/summary.csv` | results | R216 QA17 medium conflict_proto min-count2 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa18_medium_conflict_proto_mc2/summary.csv` | results | R216 QA18 medium conflict_proto min-count2 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa14_medium_conflict_proto_mc4/summary.csv` | results | R216 QA14 medium conflict_proto min-count4 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa17_medium_conflict_proto_mc4/summary.csv` | results | R216 QA17 medium conflict_proto min-count4 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa18_medium_conflict_proto_mc4/summary.csv` | results | R216 QA18 medium conflict_proto min-count4 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa14_medium_conflict_proto_mc8/summary.csv` | results | R216 QA14 medium conflict_proto min-count8 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa17_medium_conflict_proto_mc8/summary.csv` | results | R216 QA17 medium conflict_proto min-count8 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa18_medium_conflict_proto_mc8/summary.csv` | results | R216 QA18 medium conflict_proto min-count8 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa14_medium_conflict_proto_mc16/summary.csv` | results | R216 QA14 medium conflict_proto min-count16 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa17_medium_conflict_proto_mc16/summary.csv` | results | R216 QA17 medium conflict_proto min-count16 summary |
| 2026-06-19 16:14 | /experiment-bridge | `../output/babi_unified_role_transition_r216_qa18_medium_conflict_proto_mc16/summary.csv` | results | R216 QA18 medium conflict_proto min-count16 summary |
| 2026-06-19 16:21 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R217 rich local conflict features for branch arbiter |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa18_smoke_conflict_proto_rich/summary.csv` | results | R217 QA18 smoke conflict_proto rich summary |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa14_medium_conflict_proto_rich/summary.csv` | results | R217 QA14 medium conflict_proto rich summary |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa17_medium_conflict_proto_rich/summary.csv` | results | R217 QA17 medium conflict_proto rich summary |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa18_medium_conflict_proto_rich/summary.csv` | results | R217 QA18 medium conflict_proto rich summary |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa14_medium_conflict_proto_rich_mc8/summary.csv` | results | R217 QA14 medium conflict_proto rich min-count8 summary |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa17_medium_conflict_proto_rich_mc8/summary.csv` | results | R217 QA17 medium conflict_proto rich min-count8 summary |
| 2026-06-19 16:21 | /experiment-bridge | `../output/babi_unified_role_transition_r217_qa18_medium_conflict_proto_rich_mc8/summary.csv` | results | R217 QA18 medium conflict_proto rich min-count8 summary |
| 2026-06-19 17:10 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R218 target-rank/top-k candidate diagnostics for token stream evaluation |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_topk_diag_base_smoke/summary.csv` | results | R218 fixed gain1.20 smoke top-k diagnostic summary |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_topk_diag_r211_smoke/summary.csv` | results | R218 R211 transient feature smoke top-k diagnostic summary |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_topk_diag_branch_state_smoke/summary.csv` | results | R218 branch-state rank32 smoke summary |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_base_medium_s0/summary.csv` | results | R218 fixed gain1.20 medium seed0 top-k diagnostic summary |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_branch_state_medium_s0/summary.csv` | results | R218 branch-state rank32 medium seed0 summary |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_branch_state_medium_s1/summary.csv` | results | R218 branch-state rank32 medium seed1 summary |
| 2026-06-19 17:10 | /experiment-bridge | `../output/phase_binding_online_stream_r218_branch_state_medium_s2/summary.csv` | results | R218 branch-state rank32 medium seed2 summary |
| 2026-06-19 17:28 | /experiment-bridge | `../phase_binding_online_stream_experiment.py` | implementation | R219 branch-state target-rank update gate |
| 2026-06-19 17:28 | /experiment-bridge | `../output/phase_binding_online_stream_r219_branch_state_top8_smoke/summary.csv` | results | R219 branch-state target-top8 smoke summary |
| 2026-06-19 17:45 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R220 bAbI target-rank/top-k answer diagnostics |
| 2026-06-19 17:45 | /experiment-bridge | `../output/babi_unified_r220_topk_diag_qa1_smoke/summary.csv` | results | R220 bAbI QA1 smoke top-k diagnostic summary |
| 2026-06-19 17:45 | /experiment-bridge | `../output/babi_unified_r220_topk_diag_qa1_smoke/predictions_sample.csv` | results | R220 bAbI QA1 prediction sample with target-rank fields |
| 2026-06-19 18:05 | /experiment-bridge | `../output/babi_unified_qa19_r221_r193_exact_topk_diag_s0/summary.csv` | results | R221 QA19 exact R193 summary with top-k rank diagnostics |
| 2026-06-19 18:05 | /experiment-bridge | `../output/babi_unified_qa19_r221_r193_exact_topk_diag_s0/predictions_sample.csv` | results | R221 QA19 exact R193 prediction sample with target-rank fields |
| 2026-06-19 18:35 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R222 default-off low-rank answer candidate arbiter |
| 2026-06-19 18:35 | /experiment-bridge | `../output/babi_unified_qa19_r222_baseline_smoke/summary.csv` | results | R222 QA19 R193-compatible baseline smoke summary |
| 2026-06-19 18:35 | /experiment-bridge | `../output/babi_unified_qa19_r222_candidate_arbiter_smoke/summary.csv` | results | R222 QA19 candidate arbiter x0.10 smoke summary |
| 2026-06-19 18:35 | /experiment-bridge | `../output/babi_unified_qa19_r222_candidate_arbiter_x005_smoke/summary.csv` | results | R222 QA19 candidate arbiter x0.05 smoke summary |
| 2026-06-19 18:35 | /experiment-bridge | `../output/babi_unified_qa19_r222_candidate_arbiter_x0025_smoke/summary.csv` | results | R222 QA19 candidate arbiter x0.025 smoke summary |
| 2026-06-19 18:55 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R223 support gate for low-rank answer candidate arbiter |
| 2026-06-19 18:55 | /experiment-bridge | `../output/babi_unified_qa19_r223_candidate_arbiter_sup10_smoke/summary.csv` | results | R223 QA19 candidate arbiter support1.0 smoke summary |
| 2026-06-19 18:55 | /experiment-bridge | `../output/babi_unified_qa19_r223_candidate_arbiter_sup20_smoke/summary.csv` | results | R223 QA19 candidate arbiter support2.0 smoke summary |
| 2026-06-19 19:20 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R224 default-off one-step greedy answer lookahead |
| 2026-06-19 19:20 | /experiment-bridge | `../output/babi_unified_qa19_r224_lookahead_k4_w05_smoke/summary.csv` | results | R224 QA19 lookahead k4 weight0.5 smoke summary |
| 2026-06-19 19:20 | /experiment-bridge | `../output/babi_unified_qa19_r224_lookahead_k4_smoke/summary.csv` | results | R224 QA19 lookahead k4 weight1.0 smoke summary |
| 2026-06-19 19:20 | /experiment-bridge | `../output/babi_unified_qa19_r224_lookahead_k4_w2_smoke/summary.csv` | results | R224 QA19 lookahead k4 weight2.0 smoke summary |
| 2026-06-19 19:20 | /experiment-bridge | `../output/babi_unified_qa19_r224_lookahead_k4_w1_medium_s0/summary.csv` | results | R224 QA19 lookahead k4 weight1.0 medium seed0 summary |
| 2026-06-19 19:35 | /experiment-bridge | `../babi_component_oracle_diagnostic.py` | implementation | R225 component-oracle diagnostic script for bAbI prediction components |
| 2026-06-19 19:35 | /experiment-bridge | `../output/babi_unified_qa19_r225_component_oracle_r193_s0/component_summary.csv` | results | R225 QA19 per-component summary on R193 component rows |
| 2026-06-19 19:35 | /experiment-bridge | `../output/babi_unified_qa19_r225_component_oracle_r193_s0/component_oracle_summary.csv` | results | R225 QA19 component oracle summary on R193 component rows |
| 2026-06-19 17:59 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R226 default-off candidate-path direct scoring mode for QA19 edge-path direct evidence |
| 2026-06-19 17:59 | /experiment-bridge | `../output/babi_unified_qa19_r226_soft_feature_s05_smoke/summary.csv` | results | R226 old soft-feature direct smoke A/B summary |
| 2026-06-19 17:59 | /experiment-bridge | `../output/babi_unified_qa19_r226_candidate_path_direct_s05_smoke/summary.csv` | results | R226 candidate-score direct smoke A/B summary |
| 2026-06-19 17:59 | /experiment-bridge | `../output/babi_unified_qa19_r226_soft_feature_s05_medium_s0/summary.csv` | results | R226 old soft-feature direct medium seed0 summary |
| 2026-06-19 17:59 | /experiment-bridge | `../output/babi_unified_qa19_r226_candidate_path_direct_s05_medium_s0/summary.csv` | results | R226 candidate-score direct medium seed0 summary |
| 2026-06-19 18:08 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R227 default-off structured source/path/other direct scoring mode |
| 2026-06-19 18:08 | /experiment-bridge | `../output/babi_unified_qa19_r227_structured_direct_s05_smoke/summary.csv` | results | R227 structured direct side0.50 path0.35 other0.15 smoke summary |
| 2026-06-19 18:08 | /experiment-bridge | `../output/babi_unified_qa19_r227_structured_side075_path025_smoke/summary.csv` | results | R227 structured direct side0.75 path0.25 smoke summary |
| 2026-06-19 18:08 | /experiment-bridge | `../output/babi_unified_qa19_r227_structured_side05_path05_smoke/summary.csv` | results | R227 structured direct side0.50 path0.50 smoke summary |
| 2026-06-19 18:17 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R228 default-off reward-punish edge-path cleanup credit mode |
| 2026-06-19 18:17 | /experiment-bridge | `../output/babi_unified_qa19_r228_selected_learn05_smoke/summary.csv` | results | R228 selected-target learned cleanup smoke summary |
| 2026-06-19 18:17 | /experiment-bridge | `../output/babi_unified_qa19_r228_reward_learn05_smoke/summary.csv` | results | R228 reward-punish learned scale0.5 smoke summary |
| 2026-06-19 18:17 | /experiment-bridge | `../output/babi_unified_qa19_r228_reward_learn10_smoke/summary.csv` | results | R228 reward-punish learned scale1.0 smoke summary |
| 2026-06-19 18:23 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R229 edge-path candidate fields in bAbI component rows |
| 2026-06-19 18:23 | /experiment-bridge | `../babi_path_candidate_diagnostic.py` | implementation | R229 path-candidate component-row diagnostic summarizer |
| 2026-06-19 18:23 | /experiment-bridge | `../output/babi_unified_qa19_r229_pathdiag_smoke/prediction_components.csv` | results | R229 QA19 diagnostic smoke component rows with edge-path fields |
| 2026-06-19 18:23 | /experiment-bridge | `../output/babi_unified_qa19_r229_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R229 QA19 path-candidate diagnostic summary |
| 2026-06-19 18:30 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R230 default-off soft multi-candidate edge-path cleanup eligibility |
| 2026-06-19 18:30 | /experiment-bridge | `../output/babi_unified_qa19_r230_softelig_learn05_smoke/summary.csv` | results | R230 soft path eligibility learned scale0.5 smoke summary |
| 2026-06-19 18:30 | /experiment-bridge | `../output/babi_unified_qa19_r230_softelig_learn10_smoke/summary.csv` | results | R230 soft path eligibility learned scale1.0 smoke summary |
| 2026-06-19 18:41 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R231 default-off margin-gated soft edge-path cleanup eligibility |
| 2026-06-19 18:41 | /experiment-bridge | `../output/babi_unified_qa19_r231_margin_gated_softelig_learn05_smoke/summary.csv` | results | R231 margin-gated soft eligibility mg0.1 ms0.1 ma0.25 smoke summary |
| 2026-06-19 18:41 | /experiment-bridge | `../output/babi_unified_qa19_r231_margin_gated_softelig_mg02_ms0_ma025_smoke/summary.csv` | results | R231 margin-gated soft eligibility mg0.2 ms0 ma0.25 smoke summary |
| 2026-06-19 18:41 | /experiment-bridge | `../output/babi_unified_qa19_r231_margin_gated_softelig_mg01_ms0_ma0_smoke/summary.csv` | results | R231 margin-gated soft eligibility clean ambiguity-only smoke summary |
| 2026-06-19 18:48 | /experiment-bridge | `../output/babi_unified_qa19_r232_r222_pathdiag_smoke/prediction_components.csv` | results | R232 R222 matched smoke component rows |
| 2026-06-19 18:48 | /experiment-bridge | `../output/babi_unified_qa19_r232_r231_mg01_ms0_ma0_pathdiag_smoke/prediction_components.csv` | results | R232 R231 clean matched smoke component rows |
| 2026-06-19 18:48 | /experiment-bridge | `../output/babi_unified_qa19_r232_r222_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R232 R222 path-candidate diagnostic summary |
| 2026-06-19 18:48 | /experiment-bridge | `../output/babi_unified_qa19_r232_r231_mg01_ms0_ma0_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R232 R231 clean path-candidate diagnostic summary |
| 2026-06-19 18:57 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R233 default-off learned-margin escape edge-path cleanup credit mode |
| 2026-06-19 18:57 | /experiment-bridge | `../output/babi_unified_qa19_r233_learned_margin_escape_dom1_esc05_smoke/summary.csv` | results | R233 learned-margin escape dom1 esc0.5 smoke summary |
| 2026-06-19 18:57 | /experiment-bridge | `../output/babi_unified_qa19_r233_learned_margin_escape_dom05_esc10_smoke/summary.csv` | results | R233 learned-margin escape dom0.5 esc1 smoke summary |
| 2026-06-19 18:57 | /experiment-bridge | `../output/babi_unified_qa19_r233_learned_margin_escape_dom05_esc10_pathdiag_smoke/prediction_components.csv` | results | R233 learned-margin escape strong variant component rows |
| 2026-06-19 18:57 | /experiment-bridge | `../output/babi_unified_qa19_r233_learned_margin_escape_dom05_esc10_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R233 learned-margin escape strong variant path-candidate diagnostic summary |
| 2026-06-19 19:07 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R234 default-off transient path inhibition during edge-path candidate ranking |
| 2026-06-19 19:07 | /experiment-bridge | `../output/babi_unified_qa19_r234_transient_inhibit_dom05_tis05_smoke/summary.csv` | results | R234 transient path inhibition tis0.5 lr0.5 smoke summary |
| 2026-06-19 19:07 | /experiment-bridge | `../output/babi_unified_qa19_r234_transient_inhibit_dom05_tis02_lr025_smoke/summary.csv` | results | R234 transient path inhibition tis0.2 lr0.25 smoke summary |
| 2026-06-19 19:07 | /experiment-bridge | `../output/babi_unified_qa19_r234_transient_inhibit_dom05_tis01_lr01_smoke/summary.csv` | results | R234 transient path inhibition tis0.1 lr0.1 smoke summary |
| 2026-06-19 19:07 | /experiment-bridge | `../output/babi_unified_qa19_r234_transient_inhibit_dom05_tis02_lr025_pathdiag_smoke/prediction_components.csv` | results | R234 transient path inhibition middle variant component rows |
| 2026-06-19 19:07 | /experiment-bridge | `../output/babi_unified_qa19_r234_transient_inhibit_dom05_tis02_lr025_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R234 transient path inhibition middle variant path-candidate diagnostic summary |
| 2026-06-19 19:16 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R235 transient inhibition key option with compact path-hash trace |
| 2026-06-19 19:16 | /experiment-bridge | `../output/babi_unified_qa19_r235_pathhash_inhibit_dom05_tis02_lr025_smoke/summary.csv` | results | R235 path-hash transient inhibition tis0.2 lr0.25 smoke summary |
| 2026-06-19 19:16 | /experiment-bridge | `../output/babi_unified_qa19_r235_pathhash_inhibit_dom05_tis05_lr05_smoke/summary.csv` | results | R235 path-hash transient inhibition tis0.5 lr0.5 smoke summary |
| 2026-06-19 19:16 | /experiment-bridge | `../output/babi_unified_qa19_r235_pathhash_inhibit_dom05_tis05_lr05_pathdiag_smoke/prediction_components.csv` | results | R235 path-hash strong variant component rows |
| 2026-06-19 19:16 | /experiment-bridge | `../output/babi_unified_qa19_r235_pathhash_inhibit_dom05_tis05_lr05_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R235 path-hash strong variant path-candidate diagnostic summary |
| 2026-06-19 19:28 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R236 anchor-path transient inhibition with source/mid/destination candidate structure |
| 2026-06-19 19:28 | /experiment-bridge | `../output/babi_unified_qa19_r236_anchorpath_inhibit_dom05_tis02_lr025_smoke/summary.csv` | results | R236 anchor-path transient inhibition tis0.2 lr0.25 smoke summary |
| 2026-06-19 19:28 | /experiment-bridge | `../output/babi_unified_qa19_r236_anchorpath_inhibit_dom05_tis05_lr05_smoke/summary.csv` | results | R236 anchor-path transient inhibition tis0.5 lr0.5 smoke summary |
| 2026-06-19 19:28 | /experiment-bridge | `../output/babi_unified_qa19_r236_anchorpath_inhibit_dom05_tis05_lr05_pathdiag_smoke/prediction_components.csv` | results | R236 anchor-path strong variant component rows |
| 2026-06-19 19:28 | /experiment-bridge | `../output/babi_unified_qa19_r236_anchorpath_inhibit_dom05_tis05_lr05_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R236 anchor-path strong variant path-candidate diagnostic summary |
| 2026-06-19 19:37 | /experiment-bridge | `../babi_unified_token_qa_experiment.py` | implementation | R237 anchor-keyed transient runner-up boost trace |
| 2026-06-19 19:37 | /experiment-bridge | `../output/babi_unified_qa19_r237_anchor_inhibit_runnerboost_b03_lr05_m01_smoke/summary.csv` | results | R237 anchor-path runner boost scale0.3 smoke summary |
| 2026-06-19 19:37 | /experiment-bridge | `../output/babi_unified_qa19_r237_anchor_inhibit_runnerboost_b06_lr05_m01_smoke/summary.csv` | results | R237 anchor-path runner boost scale0.6 smoke summary |
| 2026-06-19 19:37 | /experiment-bridge | `../output/babi_unified_qa19_r237_anchor_inhibit_runnerboost_b06_lr05_m01_pathdiag_smoke/prediction_components.csv` | results | R237 anchor-path runner boost scale0.6 component rows |
| 2026-06-19 19:37 | /experiment-bridge | `../output/babi_unified_qa19_r237_anchor_inhibit_runnerboost_b06_lr05_m01_pathdiag_summary_smoke/path_candidate_summary.csv` | results | R237 anchor-path runner boost scale0.6 path-candidate diagnostic summary |
| 2026-06-19 19:51 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R238 default-off consistency/runner-learned gates for transient runner boost |
| 2026-06-19 19:51 | /run-experiment | `../babi_path_candidate_diagnostic.py` | implementation | R238 selected/runner consistency fields in path-candidate diagnostic summaries |
| 2026-06-19 19:51 | /run-experiment | `../output/babi_unified_qa19_r238_r237_repro_boost06_diag_s0/summary.csv` | results | R238 R237 boost0.6 repro summary with component rows |
| 2026-06-19 19:51 | /run-experiment | `../output/babi_unified_qa19_r238_consistency_boost_cm0_diag_s0/summary.csv` | results | R238 consistency-gated runner boost cm0 smoke summary |
| 2026-06-19 19:51 | /run-experiment | `../output/babi_unified_qa19_r238_consistency_boost_cm005_diag_s0/summary.csv` | results | R238 consistency-gated runner boost cm0.05 smoke summary |
| 2026-06-19 19:51 | /run-experiment | `../output/babi_unified_qa19_r238_r237_repro_boost06_path_diag_s0/path_candidate_summary.csv` | results | R238 R237 repro path-candidate diagnostic summary |
| 2026-06-19 19:51 | /run-experiment | `../output/babi_unified_qa19_r238_consistency_boost_cm0_path_diag_s0/path_candidate_summary.csv` | results | R238 cm0 path-candidate diagnostic summary |
| 2026-06-19 19:51 | /run-experiment | `../output/babi_unified_qa19_r238_consistency_boost_cm005_path_diag_s0/path_candidate_summary.csv` | results | R238 cm0.05 path-candidate diagnostic summary |
| 2026-06-19 20:04 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R239 default-off runner counterfactual target-margin gate for transient boost |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_r237_repro_boost06_diag_s0/summary.csv` | results | R239 R237 boost0.6 repro summary with runner counterfactual diagnostics |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_cfboost_gain0_diag_s0/summary.csv` | results | R239 counterfactual boost gain0 smoke summary |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_cfboost_gainm005_diag_s0/summary.csv` | results | R239 counterfactual boost gain-0.05 smoke summary |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_cfboost_gain0_b15_lr10_diag_s0/summary.csv` | results | R239 counterfactual boost gain0 high-amplitude smoke summary |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_r237_repro_boost06_path_diag_s0/path_candidate_summary.csv` | results | R239 R237 repro path-candidate diagnostic summary |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_cfboost_gain0_path_diag_s0/path_candidate_summary.csv` | results | R239 counterfactual gain0 path-candidate diagnostic summary |
| 2026-06-19 20:04 | /run-experiment | `../output/babi_unified_qa19_r239_cfboost_gain0_b15_lr10_path_diag_s0/path_candidate_summary.csv` | results | R239 high-amplitude counterfactual path-candidate diagnostic summary |
| 2026-06-19 20:22 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R240 default-off selected-vs-runner edge-path pair arbiter |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_s03_diag_s0/summary.csv` | results | R240 answer-error pair arbiter scale0.3 smoke summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_s06_diag_s0/summary.csv` | results | R240 answer-error pair arbiter scale0.6 smoke summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_cfpos_s06_diag_s0/summary.csv` | results | R240 counterfactual-positive pair arbiter scale0.6 smoke summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_cfpos_s03_diag_s0/summary.csv` | results | R240 counterfactual-positive pair arbiter scale0.3 smoke summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_cfpos_s06_m02_diag_s0/summary.csv` | results | R240 counterfactual-positive pair arbiter scale0.6 margin0.2 safety check summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_s03_path_diag_s0/path_candidate_summary.csv` | results | R240 answer-error scale0.3 path-candidate diagnostic summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_s06_path_diag_s0/path_candidate_summary.csv` | results | R240 answer-error scale0.6 path-candidate diagnostic summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_cfpos_s06_path_diag_s0/path_candidate_summary.csv` | results | R240 counterfactual-positive scale0.6 path-candidate diagnostic summary |
| 2026-06-19 20:22 | /run-experiment | `../output/babi_unified_qa19_r240_runner_arbiter_cfpos_s03_path_diag_s0/path_candidate_summary.csv` | results | R240 counterfactual-positive scale0.3 path-candidate diagnostic summary |
| 2026-06-19 20:33 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R241 count-gated and separate-negative selected-vs-runner edge-path pair arbiter |
| 2026-06-19 20:33 | /run-experiment | `../output/babi_unified_qa19_r241_runner_arbiter_cfpos_s06_mc2_diag_s0/summary.csv` | results | R241 cf-positive pair arbiter min-count2 smoke summary |
| 2026-06-19 20:33 | /run-experiment | `../output/babi_unified_qa19_r241_runner_arbiter_cfpos_s06_negsep_diag_s0/summary.csv` | results | R241 cf-positive pair arbiter separate-negative smoke summary |
| 2026-06-19 20:33 | /run-experiment | `../output/babi_unified_qa19_r241_runner_arbiter_cfpos_s06_mc2_negsep_diag_s0/summary.csv` | results | R241 cf-positive pair arbiter min-count2 separate-negative smoke summary |
| 2026-06-19 20:33 | /run-experiment | `../output/babi_unified_qa19_r241_runner_arbiter_cfpos_s06_mc2_path_diag_s0/path_candidate_summary.csv` | results | R241 min-count2 path-candidate diagnostic summary |
| 2026-06-19 20:33 | /run-experiment | `../output/babi_unified_qa19_r241_runner_arbiter_cfpos_s06_negsep_path_diag_s0/path_candidate_summary.csv` | results | R241 separate-negative path-candidate diagnostic summary |
| 2026-06-19 20:33 | /run-experiment | `../output/babi_unified_qa19_r241_runner_arbiter_cfpos_s06_mc2_negsep_path_diag_s0/path_candidate_summary.csv` | results | R241 min-count2 separate-negative path-candidate diagnostic summary |
| 2026-06-19 20:43 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R242 rich-gap selected-vs-runner pair arbiter feature mode |
| 2026-06-19 20:43 | /run-experiment | `../output/babi_unified_qa19_r242_runner_arbiter_richgap_cfpos_s06_mc2_diag_s0/summary.csv` | results | R242 rich-gap min-count2 smoke summary |
| 2026-06-19 20:43 | /run-experiment | `../output/babi_unified_qa19_r242_runner_arbiter_richgap_cfpos_s06_negsep_diag_s0/summary.csv` | results | R242 rich-gap separate-negative smoke summary |
| 2026-06-19 20:43 | /run-experiment | `../output/babi_unified_qa19_r242_runner_arbiter_richgap_cfpos_s06_mc2_negsep_diag_s0/summary.csv` | results | R242 rich-gap min-count2 separate-negative smoke summary |
| 2026-06-19 20:43 | /run-experiment | `../output/babi_unified_qa19_r242_runner_arbiter_richgap_cfpos_s06_mc2_path_diag_s0/path_candidate_summary.csv` | results | R242 rich-gap min-count2 path-candidate diagnostic summary |
| 2026-06-19 20:43 | /run-experiment | `../output/babi_unified_qa19_r242_runner_arbiter_richgap_cfpos_s06_negsep_path_diag_s0/path_candidate_summary.csv` | results | R242 rich-gap separate-negative path-candidate diagnostic summary |
| 2026-06-19 20:43 | /run-experiment | `../output/babi_unified_qa19_r242_runner_arbiter_richgap_cfpos_s06_mc2_negsep_path_diag_s0/path_candidate_summary.csv` | results | R242 rich-gap min-count2 separate-negative path-candidate diagnostic summary |
| 2026-06-19 20:50 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R243 fixed local dendritic closure score for edge-path ranking |
| 2026-06-19 20:50 | /run-experiment | `../output/babi_unified_qa19_r243_closure025_diag_s0/summary.csv` | results | R243 closure scale0.25 smoke summary |
| 2026-06-19 20:50 | /run-experiment | `../output/babi_unified_qa19_r243_closure05_diag_s0/summary.csv` | results | R243 closure scale0.50 smoke summary |
| 2026-06-19 20:50 | /run-experiment | `../output/babi_unified_qa19_r243_closure025_path_diag_s0/path_candidate_summary.csv` | results | R243 closure scale0.25 path-candidate diagnostic summary |
| 2026-06-19 20:50 | /run-experiment | `../output/babi_unified_qa19_r243_closure05_path_diag_s0/path_candidate_summary.csv` | results | R243 closure scale0.50 path-candidate diagnostic summary |
| 2026-06-19 21:01 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R244 learned local positive/negative closure prototype banks |
| 2026-06-19 21:01 | /run-experiment | `../output/babi_unified_qa19_r244_closure_proto_s03_mc1_diag_s0/summary.csv` | results | R244 closure prototype scale0.3 min-count1 smoke summary |
| 2026-06-19 21:01 | /run-experiment | `../output/babi_unified_qa19_r244_closure_proto_s06_mc1_diag_s0/summary.csv` | results | R244 closure prototype scale0.6 min-count1 smoke summary |
| 2026-06-19 21:01 | /run-experiment | `../output/babi_unified_qa19_r244_closure_proto_s06_mc2_diag_s0/summary.csv` | results | R244 closure prototype scale0.6 min-count2 smoke summary |
| 2026-06-19 21:01 | /run-experiment | `../output/babi_unified_qa19_r244_closure_proto_s03_mc1_path_diag_s0/path_candidate_summary.csv` | results | R244 closure prototype scale0.3 min-count1 path-candidate diagnostic summary |
| 2026-06-19 21:01 | /run-experiment | `../output/babi_unified_qa19_r244_closure_proto_s06_mc1_path_diag_s0/path_candidate_summary.csv` | results | R244 closure prototype scale0.6 min-count1 path-candidate diagnostic summary |
| 2026-06-19 21:01 | /run-experiment | `../output/babi_unified_qa19_r244_closure_proto_s06_mc2_path_diag_s0/path_candidate_summary.csv` | results | R244 closure prototype scale0.6 min-count2 path-candidate diagnostic summary |
| 2026-06-19 21:10 | /run-experiment | `../synthetic_two_hop_closure_experiment.py` | implementation | R245 controlled synthetic two-hop closure credit diagnostic |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_support_s0/summary.csv` | results | R245 support-only seed0 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_support_s1/summary.csv` | results | R245 support-only seed1 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_support_s2/summary.csv` | results | R245 support-only seed2 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_fixed_s0/summary.csv` | results | R245 fixed closure seed0 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_fixed_s1/summary.csv` | results | R245 fixed closure seed1 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_fixed_s2/summary.csv` | results | R245 fixed closure seed2 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_proto_s0/summary.csv` | results | R245 closure prototype16 seed0 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_proto_s1/summary.csv` | results | R245 closure prototype16 seed1 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_proto_s2/summary.csv` | results | R245 closure prototype16 seed2 summary |
| 2026-06-19 21:10 | /run-experiment | `../output/twohop_r245_proto64_s0/summary.csv` | results | R245 closure prototype64 seed0 capacity probe summary |
| 2026-06-19 21:18 | /run-experiment | `../synthetic_two_hop_closure_experiment.py` | implementation | R246 compact closure-affinity controller method |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_slots2_s05_s0/summary.csv` | results | R246 compact affinity slots2 scale0.5 seed0 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_slots2_s05_s1/summary.csv` | results | R246 compact affinity slots2 scale0.5 seed1 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_slots2_s05_s2/summary.csv` | results | R246 compact affinity slots2 scale0.5 seed2 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_s05_s0/summary.csv` | results | R246 compact affinity scale0.5 seed0 16-slot summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_s10_s0/summary.csv` | results | R246 compact affinity scale1.0 seed0 16-slot summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_s20_s0/summary.csv` | results | R246 compact affinity scale2.0 seed0 16-slot summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_s40_s0/summary.csv` | results | R246 compact affinity scale4.0 seed0 16-slot summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_slots4_s05_s0/summary.csv` | results | R246 compact affinity slots4 scale0.5 seed0 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_affinity_slots8_s05_s0/summary.csv` | results | R246 compact affinity slots8 scale0.5 seed0 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_stress_fixed_c075_s0/summary.csv` | results | R246 closure-confounded fixed closure seed0 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_stress_fixed_c075_s1/summary.csv` | results | R246 closure-confounded fixed closure seed1 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_stress_fixed_c075_s2/summary.csv` | results | R246 closure-confounded fixed closure seed2 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_stress_affinity_slots2_s10_c075_s0/summary.csv` | results | R246 closure-confounded compact affinity slots2 scale1.0 seed0 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_stress_affinity_slots2_s10_c075_s1/summary.csv` | results | R246 closure-confounded compact affinity slots2 scale1.0 seed1 summary |
| 2026-06-19 21:18 | /run-experiment | `../output/twohop_r246_stress_affinity_slots2_s10_c075_s2/summary.csv` | results | R246 closure-confounded compact affinity slots2 scale1.0 seed2 summary |
| 2026-06-19 21:45 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R247 default-off compact edge-path affinity controller for QA19 |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_r237_repro_v256_s0/summary.csv` | results | R247 current-control R237-style repro summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_affinity_s025_mc1_v256_s0/summary.csv` | results | R247 affinity scale0.25 min-count1 summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_affinity_s05_mc1_v256_s0/summary.csv` | results | R247 affinity scale0.5 min-count1 summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_affinity_s05_mc2_v256_s0/summary.csv` | results | R247 affinity scale0.5 min-count2 summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_affinity_s10_mc1_v256_s0/summary.csv` | results | R247 affinity scale1.0 min-count1 summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_affinity_s10_mc2_v256_s0/summary.csv` | results | R247 affinity scale1.0 min-count2 summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_r237_repro_v256_pathdiag_s0/path_candidate_summary.csv` | results | R247 current-control path diagnostic summary |
| 2026-06-19 21:45 | /run-experiment | `../output/babi_unified_qa19_r247_affinity_s025_mc1_v256_pathdiag_s0/path_candidate_summary.csv` | results | R247 affinity scale0.25 min-count1 path diagnostic summary |
| 2026-06-19 22:02 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R248 default-off margin/conflict gates for compact edge-path affinity controller |
| 2026-06-19 22:02 | /run-experiment | `../output/babi_unified_qa19_r248_affinity_gate_mg010_dom0_cp0_v256_s0/summary.csv` | results | R248 margin-only affinity gate mg0.10 summary |
| 2026-06-19 22:02 | /run-experiment | `../output/babi_unified_qa19_r248_affinity_gate_mg005_dom05_cp002_v256_s0/summary.csv` | results | R248 affinity gate mg0.05 learned-conflict summary |
| 2026-06-19 22:02 | /run-experiment | `../output/babi_unified_qa19_r248_affinity_gate_mg010_dom05_cp002_v256_s0/summary.csv` | results | R248 affinity gate mg0.10 learned-conflict summary |
| 2026-06-19 22:02 | /run-experiment | `../output/babi_unified_qa19_r248_affinity_gate_mg015_dom05_cp002_v256_s0/summary.csv` | results | R248 affinity gate mg0.15 learned-conflict summary |
| 2026-06-19 22:02 | /run-experiment | `../output/babi_unified_qa19_r248_affinity_gate_mg015_dom05_cp002_v256_diag_s0/prediction_components.csv` | results | R248 best safety gate component diagnostic rows |
| 2026-06-19 22:02 | /run-experiment | `../output/babi_unified_qa19_r248_affinity_gate_mg015_dom05_cp002_v256_pathdiag_s0/path_candidate_summary.csv` | results | R248 best safety gate path-candidate diagnostic summary |
| 2026-06-19 22:27 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R249 default-off slot-homeostatic edge-path usage trace |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s005_lr010_min1_v256_s0/summary.csv` | results | R249 homeostasis scale0.05 lr0.10 seed0 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s010_lr010_min1_v256_s0/summary.csv` | results | R249 homeostasis scale0.10 lr0.10 seed0 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s020_lr010_min1_v256_s0/summary.csv` | results | R249 homeostasis scale0.20 lr0.10 seed0 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s010_lr020_min1_v256_s0/summary.csv` | results | R249 homeostasis scale0.10 lr0.20 seed0 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s050_lr050_min1_v256_s0/summary.csv` | results | R249 high-strength homeostasis seed0 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_r248_mg015_seed1_v256/summary.csv` | results | R249 paired seed-repeat R248 seed1 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_r248_mg015_seed2_v256/summary.csv` | results | R249 paired seed-repeat R248 seed2 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s050_lr050_min1_seed1_v256/summary.csv` | results | R249 high-strength homeostasis seed1 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s050_lr050_min1_seed2_v256/summary.csv` | results | R249 high-strength homeostasis seed2 summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s005_lr010_min1_v256_pathdiag_s0/path_candidate_summary.csv` | results | R249 mild homeostasis path-candidate diagnostic summary |
| 2026-06-19 22:27 | /run-experiment | `../output/babi_unified_qa19_r249_homeo_s050_lr050_min1_v256_pathdiag_s0/path_candidate_summary.csv` | results | R249 high-strength homeostasis path-candidate diagnostic summary |
| 2026-06-19 22:43 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R250 default-off conditional edge-path homeostasis gate |
| 2026-06-19 22:43 | /run-experiment | `../output/babi_unified_qa19_r250_cond_homeo_dom05_sm005_s050_lr050_min1_v256_s0/summary.csv` | results | R250 conditional homeostasis dom0.5 sm0.05 seed0 summary |
| 2026-06-19 22:43 | /run-experiment | `../output/babi_unified_qa19_r250_cond_homeo_dom01_sm010_s050_lr050_min1_v256_s0/summary.csv` | results | R250 conditional homeostasis dom0.1 sm0.10 seed0 summary |
| 2026-06-19 22:43 | /run-experiment | `../output/babi_unified_qa19_r250_cond_homeo_dom01_sm025_s050_lr050_min1_v256_s0/summary.csv` | results | R250 conditional homeostasis dom0.1 sm0.25 seed0 summary |
| 2026-06-19 22:43 | /run-experiment | `../output/babi_unified_qa19_r250_cond_homeo_dom05_sm005_s050_lr050_min1_v256_diag_s0/prediction_components.csv` | results | R250 conditional homeostasis component diagnostic rows |
| 2026-06-19 22:43 | /run-experiment | `../output/babi_unified_qa19_r250_cond_homeo_dom05_sm005_s050_lr050_min1_v256_pathdiag_s0/path_candidate_summary.csv` | results | R250 conditional homeostasis path-candidate diagnostic summary |
| 2026-06-19 22:55 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R251 default-off soft edge-path homeostasis modulation |
| 2026-06-19 22:55 | /run-experiment | `../output/babi_unified_qa19_r251_soft_homeo_s2_floor05_dom05_sm005_s050_lr050_min1_v256_s0/summary.csv` | results | R251 soft homeostasis scale2 floor0.5 seed0 summary |
| 2026-06-19 22:55 | /run-experiment | `../output/babi_unified_qa19_r251_soft_homeo_s2_floor075_dom05_sm005_s050_lr050_min1_v256_s0/summary.csv` | results | R251 soft homeostasis scale2 floor0.75 seed0 summary |
| 2026-06-19 22:55 | /run-experiment | `../output/babi_unified_qa19_r251_soft_homeo_s2_floor075_dom05_sm005_s050_lr050_min1_seed1_v256/summary.csv` | results | R251 soft homeostasis scale2 floor0.75 seed1 summary |
| 2026-06-19 22:55 | /run-experiment | `../output/babi_unified_qa19_r251_soft_homeo_s2_floor075_dom05_sm005_s050_lr050_min1_seed2_v256/summary.csv` | results | R251 soft homeostasis scale2 floor0.75 seed2 summary |
| 2026-06-19 22:55 | /run-experiment | `../output/babi_unified_qa19_r251_soft_homeo_s2_floor075_dom05_sm005_s050_lr050_min1_v256_diag_s0/prediction_components.csv` | results | R251 soft homeostasis component diagnostic rows |
| 2026-06-19 22:55 | /run-experiment | `../output/babi_unified_qa19_r251_soft_homeo_s2_floor075_dom05_sm005_s050_lr050_min1_v256_pathdiag_s0/path_candidate_summary.csv` | results | R251 soft homeostasis path-candidate diagnostic summary |
| 2026-06-19 23:07 | /run-experiment | `../babi_unified_token_qa_experiment.py` | implementation | R252 default-off trace saturation for edge-path homeostasis |
| 2026-06-19 23:07 | /run-experiment | `../output/babi_unified_qa19_r252_trace_sat_thr005_gain125_s050_lr050_min1_v256_s0/summary.csv` | results | R252 trace saturation threshold0.05 gain1.25 seed0 summary |
| 2026-06-19 23:07 | /run-experiment | `../output/babi_unified_qa19_r252_trace_sat_thr010_gain15_s050_lr050_min1_v256_s0/summary.csv` | results | R252 trace saturation threshold0.10 gain1.5 seed0 summary |
| 2026-06-19 23:07 | /run-experiment | `../output/babi_unified_qa19_r252_trace_sat_thr015_gain20_s050_lr050_min1_v256_s0/summary.csv` | results | R252 trace saturation threshold0.15 gain2.0 seed0 summary |
| 2026-06-19 23:07 | /run-experiment | `../output/babi_unified_qa19_r252_trace_sat_thr020_gain20_s050_lr050_min1_v256_s0/summary.csv` | results | R252 trace saturation threshold0.20 gain2.0 seed0 summary |
| 2026-06-19 23:07 | /run-experiment | `../output/babi_unified_qa19_r252_trace_sat_thr020_gain20_s050_lr050_min1_v256_diag_s0/prediction_components.csv` | results | R252 trace saturation component diagnostic rows |
| 2026-06-19 23:07 | /run-experiment | `../output/babi_unified_qa19_r252_trace_sat_thr020_gain20_s050_lr050_min1_v256_pathdiag_s0/path_candidate_summary.csv` | results | R252 trace saturation path-candidate diagnostic summary |
| 2026-06-20 00:45 | /run-experiment | `../phase_binding_online_stream_experiment.py` | implementation | U001 unified NoProp core with internal inhibition, calibration, gain, and optional eligibility pressure |
| 2026-06-20 00:45 | /run-experiment | `refine-logs/U001_UNIFIED_NOPROP_CORE_2026-06-20.md` | results | U001 unified NoProp core report |
| 2026-06-20 00:45 | /run-experiment | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | Tracker entry for U001 |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_smoke_seed0_v2/summary.csv` | results | U001 smoke seed0 v2 summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_smoke_seed0_v2/generation_summary.csv` | results | U001 smoke seed0 v2 generation summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_smoke_v2_elig005_seed0/summary.csv` | results | U001 eligibility weight0.05 smoke boundary summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_medium_v2_seed0/summary.csv` | results | U001 medium v2 seed0 summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_medium_v2_seed1/summary.csv` | results | U001 medium v2 seed1 summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_medium_v2_seed2/summary.csv` | results | U001 medium v2 seed2 summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_medium_v2_seed0/generation_summary.csv` | results | U001 medium v2 seed0 generation summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_medium_v2_seed1/generation_summary.csv` | results | U001 medium v2 seed1 generation summary |
| 2026-06-20 00:45 | /run-experiment | `../output/u001_unified_noprop_medium_v2_seed2/generation_summary.csv` | results | U001 medium v2 seed2 generation summary |
| 2026-06-20 13:04 | /run-experiment | `../u002_attention_no_bp_experiment.py` | implementation | U002 Transformer-inspired no-BP attention prototype with position codes and center-difference diagnostic |
| 2026-06-20 13:04 | /run-experiment | `refine-logs/U002_ATTENTION_NO_BP_CORE_2026-06-20.md` | results | U002 attention no-BP core report |
| 2026-06-20 13:04 | /run-experiment | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | Tracker entry for U002 |
| 2026-06-20 13:04 | /run-experiment | `../output/u002_attention_tinystories_smoke_v2/summary.csv` | results | U002 TinyStories base attention smoke summary |
| 2026-06-20 13:04 | /run-experiment | `../output/u002_attention_temporal_smoke_v2/summary.csv` | results | U002 temporal smoke summary |
| 2026-06-20 13:04 | /run-experiment | `../output/u002_attention_tiny_attn0_b2/summary.csv` | results | U002 TinyStories no-attention ablation summary |
| 2026-06-20 13:04 | /run-experiment | `../output/u002_attention_tiny_attn05_b4/summary.csv` | results | U002 TinyStories deeper blocks ablation summary |
| 2026-06-20 13:04 | /run-experiment | `../output/u002_attention_tiny_ctx64_b2/summary.csv` | results | U002 TinyStories longer context ablation summary |
| 2026-06-20 13:04 | /run-experiment | `../output/u002_attention_tiny_d128_b2/summary.csv` | results | U002 TinyStories wider model ablation summary |
| 2026-06-20 14:46 | /run-experiment | `../u003_error_microcircuit_no_bp_experiment.py` | implementation | U003 error-neuron microcircuit no-BP prototype with direct/layered/hybrid error feedback and hidden center-difference diagnostics |
| 2026-06-20 14:46 | /run-experiment | `refine-logs/U003_ERROR_MICROCIRCUIT_CORE_2026-06-20.md` | results | U003 error-neuron microcircuit report |
| 2026-06-20 14:46 | /run-experiment | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | Tracker entry for U003 |
| 2026-06-20 14:46 | /run-experiment | `../output/u003_error_mc_temporal_smoke/summary.csv` | results | U003 temporal smoke summary |
| 2026-06-20 14:46 | /run-experiment | `../output/u003_error_mc_tinystories_b4_seed0/summary.csv` | results | U003 TinyStories 4-layer random-feedback summary |
| 2026-06-20 14:46 | /run-experiment | `../output/u003_error_mc_tinystories_b4_transpose_seed0/summary.csv` | results | U003 TinyStories 4-layer transpose-feedback upper-bound summary |
| 2026-06-20 14:46 | /run-experiment | `../output/u003_error_mc_tinystories_b8_seed0/summary.csv` | results | U003 TinyStories 8-layer random-feedback depth summary |
| 2026-06-20 14:46 | /run-experiment | `../output/u003_error_mc_tinystories_b8_transpose_seed0/summary.csv` | results | U003 TinyStories 8-layer transpose-feedback depth upper-bound summary |
| 2026-06-20 16:00 | /run-experiment | `../u004_paper_error_microcircuit_llm_experiment.py` | implementation | U004 paper-faithful error-neuron microcircuit LLM adapter with `WPP/WIP/BPI/BII`, residual stack, optional attention context, and TinyStories/GSM8k next-token mixing |
| 2026-06-20 16:00 | /run-experiment | `refine-logs/U004_PAPER_ERROR_MICROCIRCUIT_LLM_2026-06-20.md` | results | U004 paper-faithful error-microcircuit report |
| 2026-06-20 16:00 | /run-experiment | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | Tracker entry for U004 |
| 2026-06-20 16:00 | /run-experiment | `../output/u004_paper_mc_tiny_fa_b4_sparse_v2_seed0/summary.csv` | results | U004 TinyStories FA skip sparse-v2 summary |
| 2026-06-20 16:00 | /run-experiment | `../output/u004_paper_mc_tiny_fa_layered_b4_seed0/summary.csv` | results | U004 TinyStories FA layered summary |
| 2026-06-20 16:00 | /run-experiment | `../output/u004_paper_mc_tiny_fa_b4_attn05_v2_seed0/summary.csv` | results | U004 TinyStories FA skip attention0.5 sparse-v2 summary |
| 2026-06-20 16:00 | /run-experiment | `../output/u004_paper_mc_tiny_bp_b4_sparse_v2_seed0/summary.csv` | results | U004 TinyStories BP layered diagnostic upper-bound summary |
| 2026-06-20 16:00 | /run-experiment | `../output/u004_paper_mc_mix_fa_b4_sparse_v2_seed0/summary.csv` | results | U004 TinyStories+GSM8k-Aug mix FA skip summary |
| 2026-06-20 16:37 | /run-experiment | `../u005_reference_errormc_llm_adapter.py` | implementation | U005 direct adapter around cloned `Error-Neuron-Microcircuits` reference `init_MC/errormc_model` |
| 2026-06-20 16:37 | /run-experiment | `refine-logs/U005_REFERENCE_ERRORMC_ADAPTER_2026-06-20.md` | results | U005 direct reference adapter report |
| 2026-06-20 16:37 | /run-experiment | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | Tracker entry for U005 |
| 2026-06-20 16:37 | /run-experiment | `../output/u005_reference_errormc_tiny_fa_smoke/summary.csv` | results | U005 TinyStories reference FA layered logit-scale8 summary |
| 2026-06-20 16:37 | /run-experiment | `../output/u005_reference_errormc_tiny_bp_smoke/summary.csv` | results | U005 TinyStories reference BP layered logit-scale8 summary |
| 2026-06-20 16:37 | /run-experiment | `../output/u005_reference_errormc_tiny_fa_logit1/summary.csv` | results | U005 TinyStories reference FA layered logit-scale1 summary |
| 2026-06-20 16:37 | /run-experiment | `../output/u005_reference_errormc_tiny_bp_logit1/summary.csv` | results | U005 TinyStories reference BP layered logit-scale1 summary |
| 2026-06-20 18:05 | /run-experiment | `../u006_reference_stream_errormc_llm_adapter.py` | implementation | U006 direct reference streaming errormc LLM adapter with token/document stream, position-coded input, multi-step presentation, and TinyStories/GSM8k mixing |
| 2026-06-20 18:05 | /run-experiment | `refine-logs/U006_REFERENCE_STREAM_ERRORMC_LLM_2026-06-20.md` | results | U006 reference streaming error-microcircuit report |
| 2026-06-20 18:05 | /run-experiment | `refine-logs/EXPERIMENT_TRACKER.md` | implementation | Tracker entry for U006 |
| 2026-06-20 18:05 | /run-experiment | `../output/u006_reference_stream_errormc_tiny_fa_steps5/summary.csv` | results | U006 TinyStories FA layered 5-step no-bias summary |
| 2026-06-20 18:05 | /run-experiment | `../output/u006_reference_stream_errormc_tiny_fa_steps5_unigram/summary.csv` | results | U006 TinyStories FA layered 5-step unigram-readout summary |
| 2026-06-20 18:05 | /run-experiment | `../output/u006_reference_stream_errormc_tiny_fa_steps20_unigram/summary.csv` | results | U006 TinyStories FA layered 20-step unigram-readout summary |
| 2026-06-20 18:05 | /run-experiment | `../output/u006_reference_stream_errormc_tiny_bp_steps20_unigram/summary.csv` | results | U006 TinyStories BP diagnostic 20-step unigram-readout summary |
| 2026-06-20 18:05 | /run-experiment | `../output/u006_reference_stream_errormc_mix_smoke/summary.csv` | results | U006 TinyStories+GSM8k-Aug mix stream smoke summary |
| 2026-06-20 18:43 | /run-experiment | `../output/u006_reference_stream_errormc_mix_epoch1_fa_b4_steps20/summary.csv` | results | U006 TinyStories+GSM8k-Aug mixed one-epoch FA summary |
| 2026-06-20 18:43 | /run-experiment | `../output/u006_reference_stream_errormc_mix_epoch1_fa_b4_steps20/greedy_samples.txt` | results | U006 mixed one-epoch greedy samples showing punctuation-loop collapse |
| 2026-06-20 19:18 | /run-experiment | `../u007_reference_dpc_llm_adapter.py` | implementation | U007 direct reference dPC/Rao-Ballard predictive-coding LLM adapter |
| 2026-06-20 19:18 | /run-experiment | `refine-logs/U007_REFERENCE_DPC_RAO_BALLARD_2026-06-20.md` | results | U007 reference dPC/Rao-Ballard report |
| 2026-06-20 19:18 | /run-experiment | `../output/u007_reference_dpc_bp_sps_smoke_stable/summary.csv` | results | U007 stable dPC/BP/SPS small-scale smoke summary |
| 2026-06-20 19:18 | /run-experiment | `../output/u007_reference_dpc_bp_sps_smoke_logit1/summary.csv` | results | U007 dPC/BP/SPS logit-scale boundary summary |
| 2026-06-20 19:18 | /run-experiment | `../output/u007_reference_dpc_bp_sps_smoke_scale01/summary.csv` | results | U007 dPC/BP/SPS init-scale boundary summary |
| 2026-06-20 19:18 | /run-experiment | `../output/u007_reference_dpc_bp_sps_smoke_centered/summary.csv` | results | U007 dPC/BP/SPS centered-target boundary summary |
| 2026-06-20 19:42 | /run-experiment | `../Error-Neuron-Microcircuits/numpy_model/src/init_MC.py` | implementation | U008 minimal dPC source BPP initialization shape fix |
| 2026-06-20 19:42 | /run-experiment | `../Error-Neuron-Microcircuits/numpy_model/src/microcircuit.py` | implementation | U008 minimal dPC source `BPP_init` pass-through fix |
| 2026-06-20 19:42 | /run-experiment | `refine-logs/U008_DPC_SOURCE_SHAPE_AUDIT_2026-06-20.md` | results | U008 dPC source shape audit report |
| 2026-06-20 19:42 | /run-experiment | `../output/u008_dpc_fa_shape_fix_smoke/summary.csv` | results | U008 dPC/FA tiny shape-fix smoke summary |
| 2026-06-20 19:42 | /run-experiment | `../output/u008_dpc_fa_b4_shape_fix_stable/summary.csv` | results | U008 dPC/FA d64 b4 stable-scale summary |
| 2026-06-20 19:42 | /run-experiment | `../output/u008_dpc_fa_b4_shape_fix_scale01/summary.csv` | results | U008 dPC/FA d64 b4 larger-scale summary |
| 2026-06-20 19:42 | /run-experiment | `../output/u008_dpc_fa_b4_shape_fix_sps/summary.csv` | results | U008 dPC/FA d64 b4 SPS summary |
| 2026-06-20 19:42 | /run-experiment | `../output/u008_dpc_fa_mix_smoke_shape_fix/summary.csv` | results | U008 dPC/FA TinyStories+GSM8k mix smoke summary |
