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

## Code

- `no_bp_mnist_experiment.py`
- `temporal_sequence_experiment.py`

## Result Directories

- `results/pilot_v1/`
- `results/full_v1/`
- `results/full_v2/`
- `results/full_v3/`
- `temporal/results/delayed_quick_v1/`
- `temporal/results/delayed_hard_v1/`
- `temporal/results/bptt_tuned_v2/`

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
