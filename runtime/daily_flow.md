# Daily Execution Flow

## Overview

This document describes the complete daily execution cycle of the Clinical Case Similarity System, designed to complete full processing within the 6-day build window using efficient GPU quota management.

---

## Daily Schedule (UTC)

```
┌────────────────────────────────────────────────────────────────┐
│                    DAILY TIMELINE (UTC)                         │
├────────────────────────────────────────────────────────────────┤
│ 00:00 ──── Quota Reset                                          │
│ 02:00 ──── Primary Processing Window Opens                      │
│   │                                                             │
│   ├── Stage 1: CPU Ingestion (Scraping)                        │
│   ├── Stage 2: CPU Normalization                               │
│   ├── Stage 3: GPU Distillation (if quota available)          │
│   ├── Stage 4: GPU Embedding (if quota available)             │
│   └── Stage 5: FAISS Indexing                                  │
│                                                                 │
│ 06:00 ──── Primary Processing Window Closes                     │
│ 06:00 ──── Serving Mode (Query handling)                        │
│ 14:00 ──── Secondary Processing Window (if needed)              │
│ 16:00 ──── Secondary Window Closes                              │
│ 22:00 ──── Hard Stop (No GPU after this time)                   │
│ 23:59 ──── Prepare for next day                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Stage Details

### Stage 1: Scraping (CPU)

**Duration:** 30-60 minutes  
**Mode:** CPU_ONLY safe  
**Resumable:** Yes

```
┌─────────────────────────────────────────────────────────────┐
│                     SCRAPING STAGE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load sources.yaml configuration                         │
│  2. Load scraped_urls.txt checkpoint                        │
│  3. For each enabled source:                                │
│     a. Check rate limits                                    │
│     b. Fetch new content                                    │
│     c. Save to data/raw/{source}/                          │
│     d. Update scraped_urls.txt                             │
│  4. Log completion to ingestion.log                         │
│                                                              │
│  Output: data/raw/{source}/{case_id}.json                   │
│  Checkpoint: checkpoints/scraped_urls.txt                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Daily targets:**
- PubMed: ~1,000 new cases
- ESC: ~50 new cases
- AHA: ~30 new cases
- Journals: ~200 new cases

### Stage 2: Normalization (CPU)

**Duration:** 20-40 minutes  
**Mode:** CPU_ONLY safe  
**Resumable:** Yes

```
┌─────────────────────────────────────────────────────────────┐
│                   NORMALIZATION STAGE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Load clinical_schema.yaml                               │
│  2. Load normalized_ids.txt checkpoint                      │
│  3. For each raw case not yet normalized:                   │
│     a. Parse raw content                                    │
│     b. Extract fields per schema                            │
│     c. Apply normalization rules                            │
│     d. Validate completeness                                │
│     e. Save to structured/ or rejected/                     │
│     f. Update normalized_ids.txt                            │
│  4. Deduplicate against existing cases                      │
│                                                              │
│  Output: data/structured/normalized_cases.jsonl             │
│  Rejected: data/structured/rejected_cases.jsonl             │
│  Checkpoint: checkpoints/normalized_ids.txt                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Stage 3: Distillation (GPU Required)

**Duration:** 20-30 GPU minutes  
**Mode:** GPU_AVAILABLE only  
**Resumable:** Yes

```
┌─────────────────────────────────────────────────────────────┐
│                   DISTILLATION STAGE                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Check GPU quota availability                            │
│  2. Load llm_reviewed_ids.txt checkpoint                    │
│  3. Start GPU session timer                                 │
│  4. For each normalized case not yet reviewed:              │
│     a. Load case into LLM context                           │
│     b. Classify: rare vs non-rare                           │
│     c. Extract key clinical features                        │
│     d. Save to distilled/rare_cases.jsonl or               │
│        distilled/non_rare_cases.jsonl                       │
│     e. Update llm_reviewed_ids.txt                          │
│     f. Check quota remaining                                │
│  5. End GPU session, log usage                              │
│                                                              │
│  Output: data/distilled/rare_cases.jsonl                    │
│  Output: data/distilled/non_rare_cases.jsonl                │
│  Checkpoint: checkpoints/llm_reviewed_ids.txt               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Batch processing:**
- Batch size: 16 cases
- Cases per GPU minute: ~30-50
- Daily capacity: ~1,800 cases (at 60 min quota)

### Stage 4: Embedding (GPU Required)

**Duration:** 15-25 GPU minutes  
**Mode:** GPU_AVAILABLE only  
**Resumable:** Yes

```
┌─────────────────────────────────────────────────────────────┐
│                    EMBEDDING STAGE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Check GPU quota availability                            │
│  2. Load embedded_ids.txt checkpoint                        │
│  3. Start GPU session timer                                 │
│  4. For each distilled case not yet embedded:               │
│     a. Generate embeddings per domain:                      │
│        - symptoms → symptoms.npy                            │
│        - ecg → ecg.npy                                      │
│        - labs → labs.npy                                    │
│        - demographics → demographics.npy                    │
│        - imaging → imaging.npy                              │
│     b. Append to numpy arrays                               │
│     c. Update embedded_ids.txt                              │
│  5. End GPU session, log usage                              │
│                                                              │
│  Output: data/embeddings/{domain}.npy                       │
│  Checkpoint: checkpoints/embedded_ids.txt                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Batch processing:**
- Batch size: 64 cases
- Cases per GPU minute: ~100-200
- Daily capacity: ~6,000 cases (at 30 min)

### Stage 5: Indexing (CPU or GPU)

**Duration:** 5-15 minutes  
**Mode:** CPU_ONLY safe (slower) or GPU (faster)  
**Resumable:** Rebuild-based

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING STAGE                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Check if new embeddings exist                           │
│  2. For each domain:                                        │
│     a. Load existing index (if any)                         │
│     b. Load new embeddings                                  │
│     c. Add to FAISS index (append mode)                     │
│     d. Save updated index                                   │
│  3. Update faiss_version.json                               │
│  4. Verify index integrity                                  │
│                                                              │
│  Output: data/faiss/{domain}.index                          │
│  Checkpoint: checkpoints/faiss_version.json                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Stage 6: Serving

**Duration:** Continuous  
**Mode:** Any  

```
┌─────────────────────────────────────────────────────────────┐
│                     SERVING STAGE                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  System is ready to accept similarity queries               │
│                                                              │
│  1. Load FAISS indices into memory                          │
│  2. Accept queries via Gradio interface                     │
│  3. For each query:                                         │
│     a. Generate query embeddings (may use GPU)              │
│     b. Search each domain index                             │
│     c. Compute weighted composite score                     │
│     d. Apply thresholds                                     │
│     e. Return ranked results                                │
│     f. Log to audit.log                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6-Day Build Schedule

| Day | Scraping Target | Normalization | Distillation | Embedding | Cumulative Cases |
|-----|-----------------|---------------|--------------|-----------|------------------|
| 1 | 10,000 | 10,000 | 1,800 | 1,800 | 1,800 |
| 2 | 10,000 | 10,000 | 3,600 | 3,600 | 5,400 |
| 3 | 10,000 | 10,000 | 5,400 | 5,400 | 10,800 |
| 4 | 15,000 | 15,000 | 7,200 | 7,200 | 18,000 |
| 5 | 20,000 | 20,000 | 9,000 | 9,000 | 27,000 |
| 6 | 20,000 | 20,000 | 33,000* | 33,000* | 60,000 |

*Day 6 uses extended GPU time or catch-up processing

---

## Pipeline State Machine

```
         ┌──────────────────────────────────────────┐
         │                                          │
         ▼                                          │
    ┌─────────┐     ┌──────────────┐     ┌────────────┐
    │ SCRAPING │────▶│ NORMALIZATION│────▶│ DISTILL.   │
    └─────────┘     └──────────────┘     └────────────┘
         ▲                                     │
         │                                     │
         │              ┌─────────┐            │
         │              │ SERVING │◀───────────┤
         │              └─────────┘            │
         │                   ▲                 ▼
         │                   │          ┌────────────┐
         │                   │          │ EMBEDDING  │
         │                   │          └────────────┘
         │                   │                 │
         │                   │                 ▼
         │                   │          ┌────────────┐
         │                   └──────────│ INDEXING   │
         │                              └────────────┘
         │                                     │
         └─────────────────────────────────────┘
                    (Next day cycle)
```

---

## Failure Recovery

If any stage fails:

1. **Checkpoint is preserved** - Progress is not lost
2. **Automatic retry on restart** - System resumes from last checkpoint
3. **Skip to next stage if possible** - Non-blocking failures are logged

See `failover.md` for detailed recovery procedures.

---

## Monitoring

Key metrics tracked per day:

- Cases scraped
- Cases normalized
- Cases distilled
- Cases embedded
- GPU minutes used
- Errors encountered
- Queue depth

All metrics logged to `logs/ingestion.log`.
