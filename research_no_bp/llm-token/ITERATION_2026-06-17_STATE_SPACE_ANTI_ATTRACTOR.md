# 2026-06-17 State-Space Anti-Attractor

Correction: R079 later found that `BranchStateStabilizerMemory.update()` did not respect `--branch-state-anti-prediction-only`; it still applied anti-attractor pressure on teacher-forced online updates. The prediction-only medium results below are therefore superseded by `ITERATION_2026-06-17_PREDICTION_ONLY_FIX_CANDIDATE_COMPETITION.md`. The all-observation anti-score boundary remains valid.

R076 showed that low-rank branch-state residuals preserve state compression but do not fix free-running loops. This round adds feature-space anti-attractor pressure and splits it into two modes: all-observation anti-score and prediction-only anti-score.

## Method Change

Code: `phase_binding_online_stream_experiment.py`

New CLI:

- `--branch-state-anti-strength`
- `--branch-state-anti-threshold`
- `--branch-state-anti-orthogonal`
- `--branch-state-anti-score-strength`
- `--branch-state-anti-prediction-only`

Mechanism:

- Keep a short bank of recent branch-state slots.
- If the current branch state is too similar to recent states, subtract an anti-attractor vector in feature space.
- Optionally add an orthogonalized input component.
- `anti_score_strength` projects that anti vector back through the feature readout to perturb scores.
- `anti-prediction-only` clears anti pressure on teacher-forced observations and only activates from generated predictions.
- No BP, pretrained LLM, API backbone, raw replay, or token-probability table is used.

## Results

Base reference:

- R069 pressure-gated plastic branch-agreement base

All-observation anti-score:

| setting | post CE | post acc | greedy repeat-2 | controlled repeat-2 |
|---|---:|---:|---:|---:|
| R069 base | 2.218069850 | 0.501923077 | 0.439716312 | 0.070921986 |
| rank16 branch-state | 2.129223634 | 0.513461538 | 0.567375887 | 0.141843972 |
| anti-score s1.0 smoke | 2.268239182 | 0.505769231 | 0.397163121 | 0.078014184 |
| anti-score s1.0 medium | 2.606256620 | 0.455572676 | 0.312056738 | 0.134751773 |
| anti-score s2.0 smoke | 2.472287320 | 0.494230769 | 0.248226950 | 0.078014184 |

Prediction-only anti-score:

| setting | post CE | post acc | greedy repeat-2 | controlled repeat-2 | parity |
|---|---:|---:|---:|---:|---:|
| smoke s0.75 | 2.284854901 | 0.517307692 | 0.319148936 | 0.049645390 | n/a |
| medium s0.75 | 2.390659787 | 0.479712378 | 0.382978723 | 0.148936170 | 1.000 |
| medium s1.0 | 2.383539342 | 0.479198767 | 0.340425532 | 0.113475177 | 1.000 |

## Interpretation

- All-observation anti-score is too broad: it can lower greedy repetition, but CE/acc degrade quickly.
- Prediction-only anti-score is the best boundary so far, but medium still does not beat R069 on CE and does not improve controlled repetition.
- The next useful move is a more local branch/candidate competition rule, not a stronger scalar anti-score.

## Artifacts

- `output/phase_binding_online_stream_branch_state_anti_score_s100_medium/`
- `output/phase_binding_online_stream_branch_state_anti_predscore_s075_medium/`
- `output/phase_binding_online_stream_branch_state_anti_predscore_s100_predonly_medium/`
