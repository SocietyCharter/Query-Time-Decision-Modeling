# QTDM Arbiter: Technical Specification for Retrieval Service Integration

## Overview

### Purpose
QTDM Arbiter is a retrieval-conditioned decision service that transforms historical case retrieval into actionable decision support.

### Core Principle
- Embeddings select the neighborhood
- Classical estimation produces the decision signal
- Language models assist in extraction and explanation, but do not drive scoring

## System Responsibilities

### Retrieval Service Responsibilities
- Case ingestion
- Entity resolution
- Evidence storage
- Vector retrieval
- Metadata filtering
- Provenance tracking
- Historical case access

### QTDM Arbiter Responsibilities
- Request intake
- Neighborhood validation
- Feature matrix assembly
- Runtime model fitting
- Confidence scoring
- Fallback/refusal logic
- Evidence-linked response generation

## Mathematical Framework

### 1. Neighborhood Selection
```python
def select_neighborhood(query_case, retrieval_client):
    """
    Select top-K comparable cases from the retrieval service
    
    Args:
        query_case: Current case to match
        retrieval_client: Retrieval service client
    
    Returns:
        Filtered neighborhood of comparable cases
    """
    embedding = encode(query_case)
    
    # Retrieve top-K similar cases
    neighborhood = retrieval_client.retrieve_similar_cases(
        embedding=embedding,
        filters={
            'entity_type': query_case.entity_type,
            'vertical': query_case.vertical,
            'status': 'active',
            # Additional optional filters
        },
        k=25  # Configurable neighborhood size
    )
    
    return validate_neighborhood(neighborhood)
```

### 2. Case Weighting Strategies
```python
def compute_case_weights(neighborhood, weighting_strategy='softmax'):
    """
    Compute weights for retrieved cases
    
    Strategies:
    - 'linear': w_i = max(similarity_score, 0)
    - 'inverse_distance': w_i = 1 / (distance + epsilon)
    - 'softmax': exp(lambda * similarity) / sum(exp(lambda * similarities))
    """
    if weighting_strategy == 'softmax':
        weights = softmax([case.similarity_score for case in neighborhood])
    elif weighting_strategy == 'inverse_distance':
        weights = [1 / (1 - case.similarity_score + 1e-5) for case in neighborhood]
    else:  # linear
        weights = [max(case.similarity_score, 0) for case in neighborhood]
    
    return weights
```

### 3. Model Selection and Fitting
```python
def fit_local_model(neighborhood, target_type):
    """
    Fit a local model based on neighborhood and target type
    
    Supported targets:
    - Regression: Weighted Ridge Regression
    - Binary Classification: Weighted Logistic Regression
    - Weak Support: Weighted k-NN
    """
    if target_type == 'regression':
        model = WeightedRidgeRegression()
    elif target_type == 'binary_classification':
        model = WeightedLogisticRegression()
    else:
        model = WeightedKNearestNeighbors()
    
    model.fit(
        X=[case.features for case in neighborhood],
        y=[case.target for case in neighborhood],
        weights=compute_case_weights(neighborhood)
    )
    
    return model
```

### 4. Confidence Scoring
```python
def compute_confidence(neighborhood, model):
    """
    Compute local support and confidence
    
    Factors:
    - Usable neighbor count
    - Similarity coherence
    - Feature completeness
    - Target variance
    - Model stability
    """
    diagnostics = {
        'neighbors_total': len(neighborhood),
        'neighbors_usable': len([n for n in neighborhood if n.is_usable]),
        'feature_completeness': compute_feature_completeness(neighborhood),
        'target_variance': compute_target_variance(neighborhood),
        'model_stability': model.compute_stability()
    }
    
    confidence_score = aggregate_diagnostics(diagnostics)
    return confidence_score, diagnostics
```

## Response Specification

### Canonical Request
```json
{
    "request_id": "unique_identifier",
    "target_type": "regression|binary_classification|ranking",
    "target_name": "specific_prediction_target",
    "entity_type": "domain_specific_entity",
    "query_summary": "brief_case_description",
    "filters": {},
    "features": {},
    "policy": {}
}
```

### Canonical Response
```json
{
    "request_id": "matching_request_id",
    "status": "ok|fallback|refused",
    "prediction": null,
    "prediction_type": null,
    "confidence": 0.0,
    "support_score": 0.0,
    "model_used": null,
    "neighbors_requested": 0,
    "neighbors_used": 0,
    "evidence_case_ids": [],
    "fallback_used": false,
    "refusal_reason": null,
    "diagnostics": {},
    "explanation": {}
}
```

## Implementation Guidelines

### Feature Policy
- Ingest wide, model narrow
- Use 10-25 canonical features per domain
- 1-3 targets per domain

### Explanation Policy
**Allowed:**
- Top contributing features
- Closest precedent cases
- Support rationale
- Refusal rationale

**Not Allowed:**
- Unsupported narrative
- Invented causal claims
- Explanations not tied to retrieved cases

## Deployment Considerations

### Minimal Module Layout
```
qtdm_arbiter/
├── api/
├── core/
├── models/
├── integration/
├── audit/
└── tests/
```

## Use Cases

### When to Use QTDM Arbiter
- Prediction needed
- Ranking decision required
- Precedent matters
- Decision support context

### When to Avoid
- Fact retrieval only
- No clear target
- No meaningful historical cases
- Purely generative tasks

## Failure Mode Mitigation
- Monitor for:
  - False similarity
  - Schema drift
  - Sparse labels
  - Noisy normalization
  - Segment contamination
  - Overconfident predictions

## One-Line Description
QTDM Arbiter: A retrieval-conditioned decision service for a retrieval-backed workflow that turns comparable historical cases into a temporary local model, returning prediction, confidence, and evidence.
