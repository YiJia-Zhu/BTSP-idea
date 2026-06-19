# R174 Evidence-Conditioned Joint Suppression

**Date**: 2026-06-19

## Goal

R173 added a target/wrong-modulated inhibitory prototype bank for joint rescue
candidates. It improved QA17 and QA18 CE, but hurt QA14 because the same rule
could suppress candidates supported by strong direct role evidence.

R174 tests whether suppression should depend on local evidence type: avoid
writing an inhibitory trace when the wrong candidate is already supported by a
direct role score. This keeps the mechanism no-BP, local, default-off, and based
only on derived numeric traces.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-joint-suppress-mode {all_wrong,protect_direct,joint_only}`
- `--role-joint-suppress-direct-threshold`
- `--role-joint-suppress-joint-threshold`

`all_wrong` preserves the R173 rule. `protect_direct` skips a suppression write
when the wrong candidate has direct role evidence above the threshold. `joint_only`
is available for later probes: it suppresses only candidates with joint-rescue
evidence above threshold and direct-role evidence below threshold.

The scorer now records the pre-suppression joint rescue delta in
`last_joint_rescue_delta`, and suppression stats report candidate count, writes,
direct-evidence skips, and joint-evidence skips.

## Runs

Smoke:

- `output/babi_unified_role_transition_r174_qa18_smoke_protect_direct`

Medium, exact R173 config except `protect_direct`, `train-limit=300`,
`eval-limit=300`:

- `output/babi_unified_role_transition_r174_qa14_medium_protect_direct_dt005_exact`
- `output/babi_unified_role_transition_r174_qa17_medium_protect_direct_dt005_exact`
- `output/babi_unified_role_transition_r174_qa18_medium_protect_direct_dt005_exact`

Default-path reproduction check:

- `output/babi_unified_role_transition_r174_qa17_medium_allwrong_repro`

Full seed0, exact R173 config except `protect_direct`:

- `output/babi_unified_role_transition_r174_qa14_full_protect_direct_dt005`
- `output/babi_unified_role_transition_r174_qa17_full_protect_direct_dt005`
- `output/babi_unified_role_transition_r174_qa18_full_protect_direct_dt005`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE | Suppress writes | Direct skips |
|---|---|---:|---:|---:|---:|
| QA14 | R171 joint2 | 0.353 | 1.6785 | - | - |
| QA14 | R173 all-wrong suppression | 0.350 | 1.6801 | 225 | - |
| QA14 | R174 protect-direct dt0.05 | 0.370 | 1.6726 | 47 | 171 |
| QA17 | R171 joint2 | 0.540 | 0.7012 | - | - |
| QA17 | R173 all-wrong suppression | 0.553 | 0.7011 | 135 | - |
| QA17 | R174 protect-direct dt0.05 | 0.553 | 0.7011 | 134 | 1 |
| QA18 | R171 joint2 | 0.880 | 0.4824 | - | - |
| QA18 | R173 all-wrong suppression | 0.883 | 0.4774 | 51 | - |
| QA18 | R174 protect-direct dt0.05 | 0.883 | 0.4774 | 50 | 1 |

The R174 code reproduces the R173 QA17 medium result exactly under
`all_wrong`: `0.553/0.7011`, with 135 candidates and 135 suppression writes.

## Full Results

Test split:

| Task | Variant | Accuracy | CE | Suppress writes | Direct skips |
|---|---|---:|---:|---:|---:|
| QA14 | R167 branch r8 | 0.398 | 1.6529 | - | - |
| QA14 | R171 joint2 | 0.383 | 1.6487 | - | - |
| QA14 | R173 all-wrong suppression | 0.377 | 1.6507 | 741 | - |
| QA14 | R174 protect-direct dt0.05 | 0.394 | 1.6482 | 117 | 610 |
| QA17 | R167 branch r8 | 0.512 | 0.7076 | - | - |
| QA17 | R171 joint2 | 0.513 | 0.7007 | - | - |
| QA17 | R173 all-wrong suppression | 0.529 | 0.6986 | 457 | - |
| QA17 | R174 protect-direct dt0.05 | 0.529 | 0.6986 | 456 | 1 |
| QA18 | R168 branch r4 | 0.911 | 0.4759 | - | - |
| QA18 | R171 joint2 | 0.903 | 0.4599 | - | - |
| QA18 | R173 all-wrong suppression | 0.903 | 0.4549 | 144 | - |
| QA18 | R174 protect-direct dt0.05 | 0.903 | 0.4549 | 143 | 1 |

## Interpretation

The direct-evidence condition is selective. QA14 has many direct-supported
candidates that R173 would suppress: full QA14 skips 610 out of 727 suppression
candidates and recovers accuracy from `0.377` to `0.394`. QA17 and QA18 almost
never trigger the protection condition, so their R173 gains are preserved.

This supports the branch-diagnostic interpretation from R169: QA14 needs direct
role rescue, QA17/QA18 mainly benefit from suppressing unsupported joint
candidates. The current mechanism is not a full branch arbiter, but it is a
local inhibitory/write-gating rule that reduces one task-specific interference
mode without adding raw replay, labels beyond the online answer token, or BP.

## Boundary

R174 still does not exceed every previous branch-specific best: QA14 full
accuracy remains slightly below R167 branch r8 (`0.394` vs `0.398`), and QA18
accuracy remains below R168 branch r4 (`0.903` vs `0.911`) despite better CE.
The mechanism is best viewed as a safer suppression rule, not a complete
winner-selection solution.

Next step: combine evidence-conditioned suppression with a local branch arbiter
that can choose base/direct/joint paths per prompt, rather than relying on one
fixed readout mixture.
