# R172 Joint Candidate Cleanup

**Date**: 2026-06-19

## Goal

R171 showed that the joint rescue branch can improve CE but does not reliably
improve top-1 accuracy. R172 tests whether a local WTA/candidate cleanup on the
joint branch can turn that probability-mass gain into better winner selection.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-joint-rescue-top-k`
- `--role-joint-rescue-inhibit`

These operate only on the joint rescue score delta. The base and role branch
readouts are unchanged. Default `top-k=0` preserves R171 exactly. With `top-k>0`,
the joint branch keeps only its strongest local candidate outputs; optional
inhibition can suppress non-winners. The mechanism is local, no-BP, and uses no
validation labels or raw replay.

## Runs

Smoke:

- `output/babi_unified_role_transition_r172_qa17_smoke_joint2_top2`

Medium seed0, `train-limit=300 eval-limit=300`:

- `output/babi_unified_role_transition_r172_qa14_medium_joint2_top2`
- `output/babi_unified_role_transition_r172_qa17_medium_joint2_top2`
- `output/babi_unified_role_transition_r172_qa18_medium_joint2_top2`
- `output/babi_unified_role_transition_r172_qa14_medium_joint2_top4`
- `output/babi_unified_role_transition_r172_qa17_medium_joint2_top1`
- `output/babi_unified_role_transition_r172_qa18_medium_joint2_top1`

No full run was launched because the medium boundary did not beat R171/R167 on
the target top-1 metric.

## Results

Medium test split:

| Task | Variant | Accuracy | CE |
|---|---|---:|---:|
| QA14 | R167 branch r8 | 0.340 | 1.6808 |
| QA14 | R171 joint2 | 0.353 | 1.6785 |
| QA14 | joint2 top2 | 0.310 | 1.9260 |
| QA14 | joint2 top4 | 0.337 | 1.8849 |
| QA17 | R167 branch r8 | 0.527 | 0.7001 |
| QA17 | R171 joint2 | 0.540 | 0.7012 |
| QA17 | joint2 top2 | 0.540 | 0.7012 |
| QA17 | joint2 top1 | 0.503 | 1.0998 |
| QA18 | R167 branch r8 | 0.890 | 0.4969 |
| QA18 | R171 joint2 | 0.880 | 0.4824 |
| QA18 | joint2 top2 | 0.880 | 0.4824 |
| QA18 | joint2 top1 | 0.863 | 0.3565 |

## Findings

1. Blind top-k cleanup is not the missing winner-selection mechanism. QA14
   top2/top4 both hurt CE and do not improve accuracy. QA17 top2 is identical to
   R171 because the answer space is effectively binary; top1 is harmful.

2. QA18 top1 is a useful diagnostic: CE improves sharply (`0.4824 -> 0.3565`),
   but accuracy drops (`0.880 -> 0.863`). The joint branch can become very
   confident, but without a target/wrong cleanup signal it can confidently
   choose the wrong candidate.

3. R172 confirms that candidate cleanup must be credit-modulated. A label-free
   WTA over joint scores is too crude.

## Interpretation

The current branch stack can form useful evidence, but its cleanup circuit must
learn which candidates to suppress from answer-token error. This points away
from static WTA and toward a local inhibitory memory that is updated when the
winner is wrong, similar to a downstream interneuron/candidate suppression
trace.

## Next Step

Implement a target/wrong-modulated candidate suppression trace for the joint
path:

- when the joint top candidate is wrong, locally inhibit that candidate under
  the current branch/role feature;
- when the target is under-ranked, disinhibit or reinforce its candidate trace;
- keep the mechanism default-off and compare against R171 joint2 and R167
  branch r8.

R172 is **DONE-BOUNDARY**. It keeps the top-k cleanup switches as diagnostic
tools, but it is not a new main method.
