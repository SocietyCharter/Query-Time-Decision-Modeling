# QTDM BUILD DOCTRINE — SEMANTIC PRECEDENT ENGINE

> **READ THIS FIRST.** Every file in `qtdm_arbiter/` must comply with this doctrine.
> If code conflicts with this document, the code is wrong.

---

Mission statement:
QTDM builds a prediction range from what happened in similar past cases, then uses semantic evidence to move the prediction to the best-supported point inside that range.

QTDM does not let the LLM guess the answer.
QTDM does not use confidence as the prediction.
QTDM does not become Ridge after retrieval.

QTDM is:
similar cases → observed outcome distribution → semantic tilt → grounded prediction → uncertainty/explanation

---

## 1. Core operating principle

QTDM should always answer:

> "What happened before in cases like this, and where inside that observed range does this case most likely belong?"

The case data builds the range.

The LLM loop only helps decide whether the current query belongs:
- lower than typical
- near typical
- higher than typical

It must not directly output the target value.

---

## 2. Correct pipeline

1. Query comes in
2. Retrieve similar historical cases
3. Extract observed target values from those cases
4. Build weighted empirical outcome distribution
5. Ask LLM/DCS for semantic tilt only
6. Convert semantic tilt into a quantile position
7. Select prediction from the case-built distribution
8. Compute confidence after prediction
9. Explain support, missing context, and counterexamples

---

## 3. Distribution from cases

For retrieved cases:

```
case_i = historical case
y_i = observed outcome
s_i = semantic similarity to query
w_i = semantic weight
```

Build the outcome distribution:

```
P_N(y) = sum_{i in N} w_i K_h(y - y_i)
```

Weighted empirical CDF:

```
F_N(t) = sum_{i in N} w_i 1(y_i <= t)
```

Weighted quantile:

```
Q_N(p) = inf {t : F_N(t) >= p}
```

Baseline prediction:

```
y_hat_0 = Q_N(0.50)
```

Outcome band:

```
I_80 = [Q_N(0.10), Q_N(0.90)]
```

This is the range QTDM is allowed to predict inside.

---

## 4. Semantic tilt from LLM/DCS

The LLM/DCS stage does not predict:

```
pl_eqt = 249 K
```

It predicts a bounded semantic tilt:

```
z_sem ∈ [-2, 2]
```

Meaning:
- z_sem < 0 → current case belongs lower in the case distribution
- z_sem = 0 → current case is typical for the retrieved cases
- z_sem > 0 → current case belongs higher in the case distribution

Example output:

```json
{
  "semantic_tilt": -0.8,
  "data_says": "small rocky planet around cool M-type star with moderate insolation",
  "missing_context": ["atmospheric composition", "measured spectrum quality"],
  "supporting_reason": "similar temperate M-dwarf rocky cases cluster below the neighborhood median",
  "counter_reason": "some close-in M-dwarf planets remain much hotter"
}
```

---

## 5. Convert tilt into prediction

Convert semantic tilt into a quantile:

```
p_sem = Phi(z_sem)
```

Then predict from the observed case distribution:

```
y_hat_QTDM = Q_N(p_sem)
```

So:
- z_sem = -1.0 → lower-side prediction
- z_sem = 0.0 → median prediction
- z_sem = +1.0 → upper-side prediction

The LLM moves the estimate along the distribution, not outside it.

---

## 6. Counterexample and baseline checks

Build comparison distributions:

```
P_q(y) = query-neighborhood distribution
P_b(y) = baseline/background distribution
P_c(y) = counterexample distribution
```

Distribution overlap:

```
O(P, Q) = integral min(P(y), Q(y)) dy
```

Counter pressure:

```
O_c = O(P_q, P_c)
```

Baseline separation:

```
M_b = 1 - O(P_q, P_b)
```

If counter overlap is high, reduce confidence or escalate.

---

## 7. Confidence comes after prediction

Confidence must not create the point estimate.
Confidence evaluates whether the point estimate is trustworthy.

```
K_eff = 1 / sum(w_i^2)

E_K = min(K_eff / K_target, 1)

L = labeled_cases / retrieved_cases

eta = 1 - clamp((Q_N(0.90) - Q_N(0.10)) / (Q_b(0.95) - Q_b(0.05) + eps), 0, 1)

Conf = E_K * L * A * eta * (1 - O_c) * M_b * kappa
```

Where:
- E_K = effective sample score
- L = label coverage
- A = source quality
- eta = distribution narrowness
- (1 - O_c) = counterexample penalty
- M_b = baseline separation
- kappa = context coverage

---

## 8. Refusal / escalation rules

QTDM should refuse or escalate when:
- too few labeled cases
- effective K is too low
- distribution is too wide
- counterexample overlap is too high
- semantic tilt is unsupported
- missing context is too important
- query distribution looks like baseline

Decision rule:

```
Decision =
  refuse,     if Conf < theta_min
  escalate,   if O_c > theta_counter
  y_hat_QTDM, otherwise
```

---

## 9. Required output shape

```json
{
  "prediction": 251.0,
  "prediction_method": "semantic_tilt_quantile",
  "semantic_tilt": -0.8,
  "semantic_quantile": 0.2119,
  "prediction_low": 230.0,
  "prediction_high": 310.0,
  "confidence": 0.72,
  "data_says": "...",
  "missing_context": ["..."],
  "supporting_cases": ["..."],
  "counter_cases": ["..."],
  "distribution_summary": {
    "q10": 230.0,
    "q50": 260.0,
    "q90": 310.0,
    "mean": 267.4,
    "k_eff": 12.1
  },
  "comparison": {
    "baseline_overlap": 0.31,
    "counter_overlap": 0.18,
    "baseline_separation": 0.69
  }
}
```

---

## 10. Hard guardrails

- The LLM may not output the final target value.
- The LLM may only output: semantic_tilt, supporting context, missing context, counterexample reasoning, confidence rationale.
- The final prediction must come from: `y_hat = Q_N(Phi(z_sem))`
- If there is no reliable case-built distribution, QTDM must refuse or return `semantic_support_only`.

---

## 11. One-line build rule

Every QTDM improvement must either improve the case-built outcome distribution, improve the semantic tilt placement inside that distribution, or improve the explanation of why that placement is supported.

---

## 12. What not to build

- Do not rebuild QTDM as a confidence scorer.
- Do not make Ridge the center.
- Do not let the LLM directly guess target values.
- Do not punish the system with aggressive gates before the prediction layer is evaluated.
- Do not confuse uncertainty math with point-estimate math.

---

## 13. Final repo banner

**QTDM is a semantic precedent engine.**

It builds an outcome distribution from similar past cases, uses semantic evidence to choose where the current case belongs inside that distribution, and returns a grounded prediction with uncertainty, support, missing context, and counterexamples.

---

*Adopted 2026-05-01. Supersedes all prior QTDM design notes on point estimation methodology.*

