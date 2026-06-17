# Iteration 2026-06-15: Generated Natural FAQ API Scale Check

## Purpose

Extend the natural FAQ prototype beyond eight handcrafted examples. The target is an API-compatible online memory adapter that learns new customer facts from a stream, stores no raw training examples, and supplies compact memory hints to a frozen/API model.

## Code

- `online_memory_faq_api_experiment.py`
- Added `--dataset generated` with deterministic generated FAQ facts.
- Generated facts cover return windows, shipping times, support emails, warranty lengths, pickup locations, discounts, cancellation deadlines, and access policies.
- Subjects are made unique with stable labels, so the local memory must select the intended fact rather than a repeated template.

## Commands

```bash
python -m py_compile online_memory_faq_api_experiment.py

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_generated_dry \
  --dataset generated --fact-limit 64 --api-limit 16

python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_generated_256_dry \
  --dataset generated --fact-limit 256 --api-limit 16

API_KEY=... python online_memory_faq_api_experiment.py \
  --out-dir output/online_memory_faq_generated_api_run \
  --dataset generated --fact-limit 64 --api-limit 16 --run-api --api-timeout 90
```

## Results

| Run | Method | Accuracy | State bytes | Raw examples stored | Answer values stored |
|---|---:|---:|---:|---:|---:|
| generated 64 dry | local_hashed_faq_memory | 1.000 | 37,755 | false | true |
| generated 256 dry | local_hashed_faq_memory | 1.000 | 140,372 | false | true |
| generated 64 API, 16 questions | api_no_memory | 0.000 | 0 | false | false |
| generated 64 API, 16 questions | api_memory_hint | 1.000 | 37,755 | false | true |

Sample API behavior:

- No memory: answers `I don't know.` for synthetic facts.
- With memory hint: produces natural answers such as return windows, shipping times, support emails, warranties, and discounts from the hinted canonical value.

## Interpretation

This is a positive M5 result for the API-compatible prototype. It shows the frozen/API model can act as a language generator while the no-BP memory supplies newly learned facts without retaining raw training statements or user questions.

The result is still not a final GPT-like online learner:

- The memory stores canonical answer values, so it is not a strong privacy compression result.
- The task is generated FAQ, not open-domain dialogue.
- The memory performs exact-ish symbolic fact selection with hashed sparse features; it does not yet learn broad semantic generalization.

## Next Steps

1. Add a dialogue-style stream where facts are introduced across multiple conversational turns.
2. Add paraphrase robustness beyond the two templated questions per fact.
3. Test deletion/overwrite for the natural FAQ memory, not only the schema QA memory.
4. Explore storing compressed/generated summaries or value sketches instead of canonical answer strings.
