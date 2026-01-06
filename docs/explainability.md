# Explainability Documentation

## Overview

This document explains how the Clinical Case Similarity System computes similarity scores and why specific cases are flagged as matches. Explainability is critical for clinical trust and regulatory compliance.

---

## Similarity Computation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY INPUT                                │
│  (symptoms, ECG, labs, demographics, imaging)                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TEXT PREPROCESSING                            │
│  - Lowercase normalization                                       │
│  - Medical synonym expansion                                     │
│  - Abbreviation standardization                                 │
│  - Unit normalization                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING GENERATION                            │
│  Model: sentence-transformers/all-MiniLM-L6-v2                  │
│  Output: 384-dimensional dense vector per domain                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FAISS SIMILARITY SEARCH                        │
│  Algorithm: Inner Product (cosine similarity)                   │
│  Index type: IndexFlatIP (exact search)                         │
│  Per-domain independent search                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COMPOSITE SCORE CALCULATION                     │
│                                                                  │
│  composite = Σ (weight[domain] × similarity[domain])            │
│                                                                  │
│  Default weights:                                               │
│    symptoms:     0.30                                           │
│    ecg:          0.25                                           │
│    labs:         0.25                                           │
│    demographics: 0.10                                           │
│    imaging:      0.10                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   THRESHOLD APPLICATION                          │
│                                                                  │
│  if composite >= 0.65:                                          │
│      flag_for_review = True                                     │
│                                                                  │
│  Alert levels:                                                  │
│    0.65-0.75: Routine (low priority)                           │
│    0.75-0.85: Significant (medium priority)                    │
│    0.85-1.00: Critical (high priority)                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Understanding Similarity Scores

### What the Score Means

| Score Range | Interpretation |
|-------------|----------------|
| 0.90 - 1.00 | Near-identical clinical presentation |
| 0.80 - 0.90 | Very similar case, likely related condition |
| 0.70 - 0.80 | Significant overlap in clinical features |
| 0.60 - 0.70 | Moderate similarity, worth reviewing |
| 0.50 - 0.60 | Some shared features, lower confidence |
| < 0.50 | Limited similarity, likely different conditions |

### What the Score Does NOT Mean

- ❌ Probability of same diagnosis
- ❌ Clinical certainty
- ❌ Recommendation strength
- ❌ Statistical significance

---

## Domain-Level Explanations

For each matched case, the system provides domain-level breakdown:

### Example Output

```json
{
  "query_id": "Q_20260105_001",
  "matched_case": {
    "case_id": "PMID_12345678",
    "source": "PubMed",
    "title": "Rare presentation of Brugada syndrome..."
  },
  "scores": {
    "composite": 0.78,
    "breakdown": {
      "symptoms": {
        "score": 0.85,
        "weight": 0.30,
        "contribution": 0.255,
        "matched_features": [
          "syncope",
          "palpitations",
          "family history of sudden death"
        ]
      },
      "ecg": {
        "score": 0.82,
        "weight": 0.25,
        "contribution": 0.205,
        "matched_features": [
          "coved ST elevation V1-V3",
          "Type 1 Brugada pattern"
        ]
      },
      "labs": {
        "score": 0.45,
        "weight": 0.25,
        "contribution": 0.113,
        "matched_features": [
          "normal electrolytes",
          "normal cardiac enzymes"
        ]
      },
      "demographics": {
        "score": 0.90,
        "weight": 0.10,
        "contribution": 0.090,
        "matched_features": [
          "male",
          "age 35-45",
          "Asian ethnicity"
        ]
      },
      "imaging": {
        "score": 0.72,
        "weight": 0.10,
        "contribution": 0.072,
        "matched_features": [
          "normal echocardiogram",
          "normal cardiac MRI"
        ]
      }
    }
  },
  "explanation": {
    "primary_match_reason": "Strong symptom and ECG pattern overlap",
    "confidence_factors": [
      "Both cases present with syncope and palpitations",
      "Both show Type 1 Brugada ECG pattern",
      "Similar demographic profile"
    ],
    "uncertainty_factors": [
      "Lab values are non-specific",
      "Different geographic origin"
    ]
  }
}
```

---

## Why Was This Case Flagged?

For each flagged case, the system answers:

### 1. What domains contributed most?

The contribution is calculated as:
```
contribution = weight × similarity_score
```

Domains are ranked by contribution to show which clinical aspects drove the match.

### 2. What specific features matched?

The system identifies specific clinical features that align:
- Symptom terms that appear in both cases
- ECG findings with similar descriptions
- Lab abnormalities in similar ranges
- Demographic factors that align

### 3. What is the confidence level?

Confidence is adjusted based on:
- Number of domains with strong matches
- Data completeness of both cases
- Presence of rare disease indicators
- Consistency across domains

---

## Embedding Model Explanation

### Model: sentence-transformers/all-MiniLM-L6-v2

- **Architecture**: DistilBERT-based transformer
- **Training**: Contrastive learning on sentence pairs
- **Output**: 384-dimensional dense vectors
- **Similarity metric**: Cosine similarity

### How Text Becomes Vectors

1. **Tokenization**: Text split into subword tokens
2. **Encoding**: Tokens processed through transformer layers
3. **Pooling**: Token embeddings averaged to single vector
4. **Normalization**: Vector normalized to unit length

### Why This Model?

- Fast inference (important for query latency)
- Good semantic understanding of medical text
- Reasonable balance of quality and speed
- Well-documented and widely validated

---

## Threshold Justification

### Composite Threshold: 0.65

This threshold was chosen based on:

1. **Precision/Recall tradeoff**: Lower threshold catches more true positives but increases false positives
2. **Clinical safety**: Conservative threshold to avoid missing important matches
3. **Empirical testing**: Validated against known similar case pairs
4. **Expert review**: Calibrated with clinician feedback

### Domain Thresholds

| Domain | Threshold | Rationale |
|--------|-----------|-----------|
| Symptoms | 0.50 | Lower to capture varied descriptions |
| ECG | 0.55 | Higher due to objective findings |
| Labs | 0.50 | Accounts for normal variation |
| Demographics | 0.40 | Lower as exact match less important |
| Imaging | 0.45 | Accounts for reporting variation |

---

## Limitations of Similarity Scores

### Known Limitations

1. **Semantic ambiguity**: Same words can mean different things
2. **Terminology variation**: Different terms for same concept
3. **Missing context**: Embeddings don't capture full clinical context
4. **Data quality**: Source case quality affects matching
5. **Distribution shift**: Model trained on general text, not just medical

### Mitigation Strategies

1. Medical synonym expansion during preprocessing
2. Multiple domain matching (not just single domain)
3. Human review requirement for all flagged cases
4. Confidence adjustments for uncertainty
5. Regular threshold recalibration

---

## Audit Trail

Every similarity match is logged with:

```json
{
  "timestamp": "2026-01-05T14:30:00Z",
  "query_case_id": "Q_20260105_001",
  "matched_case_id": "PMID_12345678",
  "source": "PubMed",
  "domain_scores": {
    "symptoms": 0.85,
    "ecg": 0.82,
    "labs": 0.45,
    "demographics": 0.90,
    "imaging": 0.72
  },
  "composite_score": 0.78,
  "flagged": true,
  "alert_level": "significant",
  "weights_used": "balanced",
  "thresholds_version": "1.0"
}
```

This enables:
- Post-hoc analysis of match quality
- Threshold calibration studies
- Regulatory compliance documentation
- Performance monitoring over time

---

## Questions for Clinician Review

When reviewing a flagged match, consider:

1. Do the highlighted features truly align clinically?
2. Are there important differences not captured by the system?
3. Is the matched case from a reputable source?
4. Does the temporal context (date of case) matter?
5. Are there confounding factors the system missed?

---

*For technical details on the similarity algorithms, see the source code in `app.py`.*
