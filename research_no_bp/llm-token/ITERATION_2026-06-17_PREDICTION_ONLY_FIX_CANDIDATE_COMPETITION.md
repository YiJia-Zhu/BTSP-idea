# 2026-06-17 Prediction-Only Fix And Candidate Competition

R078's prediction-only anti-score result was contaminated by a code path bug: `observe()` cleared anti pressure on teacher-forced observations, but `update()` still applied anti-attractor adjustment after each online target update. This round fixes that path and tests whether local candidate competition can improve generation without turning into global token suppression.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

Fix:

- `BranchStateStabilizerMemory.update()` now calls `observe_state_vector(..., apply_anti=not self.anti_prediction_only)`.

New CLI:

- `--branch-state-anti-candidate-top-k`
- `--branch-state-anti-candidate-center`
- `--branch-state-anti-candidate-agreement-weight`

Candidate competition:

- If `anti_candidate_top_k > 0`, anti-score is restricted to the base top-k candidates.
- `anti_candidate_center` zero-centers the candidate adjustment inside the local candidate set.
- `anti_candidate_agreement_weight` injects branch-agreement signal into the same local candidate pool. Positive values reward branch consensus; negative values suppress over-consensus candidates.
- No BP, pretrained LLM/API backbone, raw replay, or statistical token-probability table is used.

## Results

Fixed prediction-only anti-score:

| setting | post CE | post acc | greedy repeat-2 | controlled repeat-2 | parity |
|---|---:|---:|---:|---:|---:|
| fixed s1.0 smoke | 2.289405958 | 0.515384615 | 0.063829787 | 0.056737589 | n/a |
| fixed s0.75 medium | 2.409156262 | 0.477144325 | 0.319148936 | 0.120567376 | 1.000 |
| fixed s1.0 medium | 2.409156262 | 0.477144325 | 0.375886525 | 0.127659574 | 1.000 |

Candidate-local variants on smoke:

| setting | post CE | post acc | greedy repeat-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|
| candidate top-k16, centered | 2.289405958 | 0.515384615 | 0.063829787 | 0.056737589 |
| candidate top-k4, centered | 2.289405958 | 0.515384615 | 0.063829787 | 0.049645390 |
| top-k4 + agreement +1.0 | 2.289405958 | 0.515384615 | 0.510638298 | 0.120567376 |
| top-k4 + agreement -1.0 | 2.289405958 | 0.515384615 | 0.638297872 | 0.063829787 |

## Interpretation

- The update-path fix matters. It restores the intended prediction-only semantics on teacher-forced online updates.
- Smoke generation can be made much less repetitive, but the fixed mechanism still does not transfer cleanly to medium.
- Candidate top-k limiting alone is not enough; on smoke it behaves like the fixed anti-score.
- Branch agreement is the wrong local signal for loop escape. Rewarding agreement deepens attractors, and suppressing agreement also damages greedy generation.
- Next step should use a candidate-specific inhibitory trace or local state reset signal that is learned from loop recurrence, not a branch-consensus reward.

## Artifacts

- `output/phase_binding_online_stream_branch_state_anti_predscore_s100_fixed_smoke/`
- `output/phase_binding_online_stream_branch_state_anti_predscore_s075_fixed_medium/`
- `output/phase_binding_online_stream_branch_state_anti_predscore_s100_fixed_medium/`
- `output/phase_binding_online_stream_branch_state_anti_candidate_k4_smoke/`
- `output/phase_binding_online_stream_branch_state_candidate_agree_w100_smoke/`
- `output/phase_binding_online_stream_branch_state_candidate_agree_wneg100_smoke/`
