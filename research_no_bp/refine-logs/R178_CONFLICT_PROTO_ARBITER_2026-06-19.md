# R178 Conflict-Only Prototype Arbiter

**Date**: 2026-06-19

## Goal

R177 showed that base-vs-rescue disagreement is informative, but hand-written
margin thresholds do not reliably select the useful base fallbacks. R178 tests a
local no-BP binary prototype memory trained only on disagreement events.

## Implementation

Extended `babi_unified_token_qa_experiment.py` with:

- `--role-branch-arbiter conflict_proto`

The mode reuses the branch arbiter prototype bank, but unlike R175's six-way
WTA arbiter it only competes between:

- `base_only`
- `--role-branch-arbiter-default`, set to `base_plus_direct_joint`

It only updates on prompts where `base_only` and the default rescue path predict
different tokens. The target branch is whichever of the two has the better
answer-token target margin under the current prompt. This is local, no-BP,
answer-token-modulated, and stores derived features rather than raw text.

## Runs

Smoke:

- `output/babi_unified_role_transition_r178_qa18_smoke_conflict_proto`

Medium seed0, exact R174 protect-direct config plus `conflict_proto`:

- `output/babi_unified_role_transition_r178_qa14_medium_conflict_proto`
- `output/babi_unified_role_transition_r178_qa17_medium_conflict_proto`
- `output/babi_unified_role_transition_r178_qa18_medium_conflict_proto`

## Medium Results

Test split:

| Task | Variant | Accuracy | CE | Base choices | Target base updates |
|---|---|---:|---:|---:|---:|
| QA14 | R174 protect-direct | 0.370 | 1.6726 | - | - |
| QA14 | R178 conflict-proto | 0.310 | 1.7004 | 191 | 51 |
| QA17 | R174 protect-direct | 0.553 | 0.7011 | - | - |
| QA17 | R178 conflict-proto | 0.527 | 0.7049 | 65 | 23 |
| QA18 | R174 protect-direct | 0.883 | 0.4774 | - | - |
| QA18 | R178 conflict-proto | 0.880 | 0.4776 | 13 | 6 |

## Interpretation

The mechanism works mechanically: it updates only on conflict cases and keeps
the competition binary. However, it still over-selects `base_only` on QA14 and
QA17, where R177 diagnostics showed rescue should usually win conflicts. QA18
does not benefit either; the few selected base fallbacks are not enough and are
slightly harmful in the medium split.

The likely failure is feature ambiguity. The current arbiter feature encodes
branch margins, pairwise agreements, and predicted token codes, but it does not
carry enough task-agnostic evidence to distinguish "QA18-like base should win"
from "QA14/QA17-like rescue should win" conflict cases. A stronger default bias
toward rescue may avoid harm, but would mostly collapse back to R174.

## Boundary

R178 rejects a naive disagreement-only binary prototype classifier as the next
branch controller. The useful signal is still there, but it needs either richer
conflict features or a more conservative local rule that only writes/uses base
fallbacks when there is repeated evidence for the same conflict pattern.

Next step: analyze conflict feature separability directly, or add a high-precision
base fallback memory that requires repeated base-win evidence before overriding
the rescue path.
