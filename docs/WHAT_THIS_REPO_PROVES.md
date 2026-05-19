# What This Repo Proves

This repository proves the QTDM arbiter mechanism, not the full private system.

It demonstrates evidence-bounded prediction:

- A query retrieves similar labeled precedent cases.
- Retrieved labels become a weighted empirical outcome distribution.
- Optional semantic tilt can move the answer only by selecting a quantile inside that distribution.
- The response exposes the evidence case IDs used to support the result.

It demonstrates refusal gates:

- Unsupported queries can refuse.
- Weak, sparse, or diffuse neighborhoods can refuse.
- Missing labels can return `semantic_support_only` instead of an invented prediction.

It demonstrates inspectable outputs:

- Prediction and interval
- Distribution summary
- Confidence/support diagnostics
- Evidence IDs
- Refusal reason when no answer is justified

It does not prove broad generality yet. The included fixture dataset is intentionally small and exists to make the public mechanism reproducible.

Larger validation requires fixed public datasets, preregistered train/test splits, calibration curves, interval-width reporting, and published tolerance rules.

