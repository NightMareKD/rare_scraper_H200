# Validation Strategy

## Overview

This document outlines the validation approach for the Clinical Case Similarity System, ensuring that similarity matches are clinically meaningful and the system operates safely.

---

## Validation Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                     VALIDATION LAYERS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Technical Validation                                  │
│  ├── Unit tests for all components                              │
│  ├── Integration tests for pipeline                             │
│  └── Performance benchmarks                                     │
│                                                                  │
│  Layer 2: Data Quality Validation                               │
│  ├── Source verification                                        │
│  ├── Schema compliance checking                                 │
│  └── Deduplication accuracy                                     │
│                                                                  │
│  Layer 3: Similarity Validation                                 │
│  ├── Known-pair testing                                         │
│  ├── Precision/recall measurement                               │
│  └── Threshold calibration                                      │
│                                                                  │
│  Layer 4: Clinical Validation                                   │
│  ├── Expert clinician review                                    │
│  ├── False positive analysis                                    │
│  └── Use case testing                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Technical Validation

### Unit Tests

| Component | Tests | Coverage Target |
|-----------|-------|-----------------|
| ConfigLoader | Config loading, caching, validation | 95% |
| CheckpointManager | Read/write, atomicity, recovery | 95% |
| HardwareDetector | GPU detection, mode selection | 90% |
| AuditLogger | Logging format, file handling | 95% |
| GPUTracker | Usage tracking, quota calculation | 95% |

### Integration Tests

```python
# Example integration test structure
def test_full_pipeline():
    # Given: Raw case data
    raw_cases = load_test_cases("test/fixtures/raw_cases/")
    
    # When: Full pipeline runs
    orchestrator = PipelineOrchestrator()
    orchestrator.run_daily_pipeline()
    
    # Then: All stages complete successfully
    assert len(get_normalized_cases()) == len(raw_cases)
    assert len(get_embedded_cases()) == len(raw_cases)
    assert faiss_index_exists()
```

### Performance Benchmarks

| Operation | Target | Acceptable |
|-----------|--------|------------|
| Embedding generation | 100 cases/min | 50 cases/min |
| FAISS query | <100ms | <500ms |
| Pipeline startup | <30s | <60s |
| Checkpoint save | <1s | <5s |

---

## Layer 2: Data Quality Validation

### Source Verification

For each data source:

1. **URL accessibility**: Verify source URLs are reachable
2. **Content format**: Validate response structure
3. **Data freshness**: Check publication dates
4. **License compliance**: Verify open access status

### Schema Compliance

```python
def validate_case(case: dict) -> ValidationResult:
    errors = []
    
    # Required fields
    if not case.get('symptoms'):
        errors.append("Missing required field: symptoms")
    
    # Type checking
    if case.get('age') and not isinstance(case['age'], (int, float)):
        errors.append("Invalid type for age")
    
    # Range validation
    if case.get('age') and (case['age'] < 0 or case['age'] > 120):
        errors.append("Age out of valid range")
    
    # Completeness score
    completeness = count_populated_fields(case) / total_fields
    if completeness < 0.4:
        errors.append(f"Low completeness: {completeness:.2%}")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)
```

### Deduplication Accuracy

**Method**: DOI + Title hash comparison

**Metrics tracked**:
- True duplicate rate
- False duplicate rate (incorrectly rejected)
- Unique case yield

**Target**: <1% false duplicate rate

---

## Layer 3: Similarity Validation

### Known-Pair Testing

Create a gold standard dataset of known similar cases:

```yaml
# test/fixtures/known_pairs.yaml
pairs:
  - case_a: "PMID_12345678"
    case_b: "PMID_23456789"
    relationship: "same_condition"
    expected_similarity: "high"  # >0.75
    
  - case_a: "PMID_34567890"
    case_b: "PMID_45678901"
    relationship: "similar_presentation"
    expected_similarity: "medium"  # 0.60-0.75
    
  - case_a: "PMID_56789012"
    case_b: "PMID_67890123"
    relationship: "unrelated"
    expected_similarity: "low"  # <0.50
```

### Precision/Recall Measurement

**Definitions**:
- **True Positive**: Similar cases correctly flagged
- **False Positive**: Unrelated cases incorrectly flagged
- **False Negative**: Similar cases missed
- **True Negative**: Unrelated cases correctly not flagged

**Targets**:
| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Precision | >0.80 | >0.70 |
| Recall | >0.85 | >0.75 |
| F1 Score | >0.82 | >0.72 |

### Threshold Calibration

**Process**:

1. Generate similarity scores for all known pairs
2. Plot ROC curve
3. Identify optimal threshold for desired precision/recall balance
4. Validate on held-out test set
5. Adjust thresholds in `config/thresholds.yaml`

```python
def calibrate_thresholds(known_pairs, similarity_scores):
    # Calculate metrics at different thresholds
    thresholds = np.arange(0.40, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        predictions = similarity_scores >= threshold
        precision = calculate_precision(known_pairs, predictions)
        recall = calculate_recall(known_pairs, predictions)
        f1 = 2 * (precision * recall) / (precision + recall)
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Select threshold with best F1 while maintaining minimum precision
    valid = [r for r in results if r['precision'] >= 0.70]
    optimal = max(valid, key=lambda x: x['f1'])
    
    return optimal['threshold']
```

---

## Layer 4: Clinical Validation

### Expert Review Process

1. **Sample selection**: Random sample of flagged matches (n=100/month)
2. **Blinded review**: Clinicians review without knowing system scores
3. **Rating scale**: 
   - 1 = Not similar at all
   - 2 = Marginally similar
   - 3 = Moderately similar
   - 4 = Very similar
   - 5 = Highly similar
4. **Consensus**: Multiple reviewers, disagreements resolved by discussion

### False Positive Analysis

For each identified false positive:

1. Document the case pair
2. Analyze which domains contributed to incorrect match
3. Identify root cause:
   - Terminology ambiguity?
   - Data quality issue?
   - Threshold too low?
   - Model limitation?
4. Implement corrective action if pattern identified

### Use Case Testing

Test system with realistic clinical scenarios:

**Scenario 1: Rare Disease Identification**
```
Input: 35-year-old male with syncope, family history 
       of sudden death, Type 1 Brugada ECG pattern
Expected: Match similar Brugada syndrome cases
Validation: Confirm matched cases are relevant
```

**Scenario 2: Differential Diagnosis Support**
```
Input: Chest pain, elevated troponin, normal coronaries
Expected: Match cases with similar presentation 
         (myocarditis, Takotsubo, etc.)
Validation: Confirm differential diagnoses are represented
```

**Scenario 3: Rare Presentation of Common Disease**
```
Input: Young patient with atypical MI presentation
Expected: Match similar atypical cases
Validation: Confirm educational value of matches
```

---

## Validation Schedule

| Activity | Frequency | Responsible |
|----------|-----------|-------------|
| Unit tests | Every commit | Automated CI |
| Integration tests | Daily | Automated CI |
| Data quality checks | Daily | Automated |
| Known-pair testing | Weekly | Automated |
| Threshold review | Monthly | Data team |
| Clinical review | Monthly | Clinical advisors |
| Full validation report | Quarterly | Combined team |

---

## Validation Metrics Dashboard

Track and display:

```
┌─────────────────────────────────────────────────────────────────┐
│                   VALIDATION DASHBOARD                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Data Quality                                                    │
│  ├── Cases ingested (7 days): 8,432                             │
│  ├── Validation pass rate: 94.2%                                │
│  ├── Duplicate rate: 2.1%                                       │
│  └── Schema compliance: 97.8%                                   │
│                                                                  │
│  Similarity Performance                                          │
│  ├── Precision (known pairs): 0.84                              │
│  ├── Recall (known pairs): 0.88                                 │
│  ├── F1 Score: 0.86                                             │
│  └── Threshold: 0.65                                            │
│                                                                  │
│  Clinical Review                                                 │
│  ├── Cases reviewed (30 days): 100                              │
│  ├── Clinician agreement rate: 78%                              │
│  ├── Mean quality rating: 3.6/5                                 │
│  └── False positive rate: 12%                                   │
│                                                                  │
│  System Health                                                   │
│  ├── Pipeline success rate: 99.2%                               │
│  ├── Query latency (p95): 180ms                                 │
│  └── Uptime: 99.8%                                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Corrective Action Process

When validation identifies issues:

1. **Document**: Log issue in tracking system
2. **Classify**: Severity (critical/high/medium/low)
3. **Investigate**: Root cause analysis
4. **Remediate**: Implement fix
5. **Verify**: Confirm fix addresses issue
6. **Monitor**: Track for recurrence

---

## Continuous Improvement

### Feedback Loop

```
Clinician Feedback → Analysis → Threshold Adjustment → Validation → Deployment
         ↑                                                              │
         └──────────────────────────────────────────────────────────────┘
```

### Version Control

All validation artifacts are versioned:
- Threshold configurations
- Known-pair datasets
- Validation results
- Clinical review records

---

*Last validation review: January 5, 2026*
