# Failover & Recovery Procedures

## Overview

This document describes how the Clinical Case Similarity System handles failures, crashes, and restarts. The system is designed to be fully resumable from any point of failure.

---

## Checkpoint-Based Recovery

The system uses checkpoint files to track progress:

| Checkpoint File | Purpose | Format |
|-----------------|---------|--------|
| `scraped_urls.txt` | URLs already scraped | One URL per line |
| `normalized_ids.txt` | Cases normalized | One case ID per line |
| `llm_reviewed_ids.txt` | Cases reviewed by LLM | One case ID per line |
| `embedded_ids.txt` | Cases with embeddings | One case ID per line |
| `faiss_version.json` | Index version info | JSON object |

---

## Recovery Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     SYSTEM RESTART                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Load All Checkpoints                         │
│                                                              │
│  scraped_urls = load("checkpoints/scraped_urls.txt")        │
│  normalized_ids = load("checkpoints/normalized_ids.txt")    │
│  reviewed_ids = load("checkpoints/llm_reviewed_ids.txt")    │
│  embedded_ids = load("checkpoints/embedded_ids.txt")        │
│  faiss_version = load("checkpoints/faiss_version.json")     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              2. Determine Current Stage                      │
│                                                              │
│  if len(scraped_urls) == 0:                                 │
│      stage = SCRAPING                                       │
│  elif len(normalized_ids) < len(scraped_urls):              │
│      stage = NORMALIZATION                                  │
│  elif len(reviewed_ids) < len(normalized_ids):              │
│      stage = DISTILLATION                                   │
│  elif len(embedded_ids) < len(reviewed_ids):                │
│      stage = EMBEDDING                                      │
│  elif faiss_version.case_count < len(embedded_ids):         │
│      stage = INDEXING                                       │
│  else:                                                       │
│      stage = SERVING                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              3. Resume from Stage                            │
│                                                              │
│  orchestrator.run_stage(stage, execution_mode)              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Failure Scenarios

### Scenario 1: Crash During Scraping

**Symptoms:**
- Process terminated unexpectedly
- `scraped_urls.txt` may be incomplete

**Recovery:**
1. System restarts
2. Loads `scraped_urls.txt`
3. Resumes scraping from where it left off
4. Already-scraped URLs are skipped

**Data safety:**
- Raw files are written atomically
- Partial downloads are discarded

### Scenario 2: Crash During Normalization

**Symptoms:**
- `normalized_ids.txt` incomplete
- Some raw cases not normalized

**Recovery:**
1. System loads both checkpoints
2. Identifies unnormalized cases: `scraped - normalized`
3. Resumes normalization

**Data safety:**
- Normalized cases written to JSONL with atomic appends
- Incomplete writes are auto-recovered

### Scenario 3: Crash During GPU Processing

**Symptoms:**
- GPU session ended unexpectedly
- `llm_reviewed_ids.txt` or `embedded_ids.txt` incomplete

**Recovery:**
1. GPU tracker shows incomplete session
2. System resumes from last checkpoint
3. GPU quota is NOT refunded (usage already occurred)

**Prevention:**
- Frequent checkpointing (every 100 items)
- Session duration limits
- Memory monitoring

### Scenario 4: FAISS Index Corruption

**Symptoms:**
- Index file corrupted or missing
- `faiss_version.json` inconsistent

**Recovery:**
1. System detects version mismatch
2. Triggers full index rebuild from embeddings
3. All embedding files are re-indexed

**Prevention:**
- Atomic file writes for indices
- Version tracking
- Periodic integrity checks

### Scenario 5: Out of Memory (OOM)

**Symptoms:**
- Process killed by OOM killer
- No graceful shutdown

**Recovery:**
1. System restarts with reduced batch size
2. `gpu_policy.yaml` settings apply
3. Processing continues with smaller batches

**Prevention:**
- Memory monitoring
- Automatic batch size reduction
- CPU offloading

### Scenario 6: Hugging Face Space Restart

**Symptoms:**
- Space container restarted
- All in-memory state lost

**Recovery:**
1. Persistent storage (`/data`) preserved
2. Checkpoints loaded from persistent storage
3. Pipeline resumes automatically

**Critical:** All data directories must be on persistent storage!

---

## Checkpoint File Formats

### scraped_urls.txt
```
https://pubmed.ncbi.nlm.nih.gov/12345678/
https://pubmed.ncbi.nlm.nih.gov/12345679/
https://www.escardio.org/cases/case-001
...
```

### normalized_ids.txt
```
PMID_12345678
PMID_12345679
ESC_case-001
...
```

### llm_reviewed_ids.txt
```
PMID_12345678
PMID_12345679
ESC_case-001
...
```

### embedded_ids.txt
```
PMID_12345678
PMID_12345679
ESC_case-001
...
```

### faiss_version.json
```json
{
  "version": 15,
  "last_updated": "2026-01-05T04:30:00",
  "case_count": 45000,
  "previous_count": 43200,
  "indices": {
    "symptoms": {"size": 45000, "dimension": 384},
    "ecg": {"size": 45000, "dimension": 384},
    "labs": {"size": 45000, "dimension": 384},
    "demographics": {"size": 45000, "dimension": 384},
    "imaging": {"size": 45000, "dimension": 384}
  }
}
```

---

## Manual Recovery Procedures

### Reset Specific Stage

To restart a specific stage from scratch:

```bash
# Reset distillation
rm checkpoints/llm_reviewed_ids.txt
rm data/distilled/rare_cases.jsonl
rm data/distilled/non_rare_cases.jsonl

# Restart system
python app.py
```

### Force Full Rebuild

To rebuild everything from raw data:

```bash
# Keep only raw data
rm -rf data/structured/*
rm -rf data/distilled/*
rm -rf data/embeddings/*
rm -rf data/faiss/*
rm checkpoints/*.txt
rm checkpoints/faiss_version.json

# Restart system
python app.py
```

### Repair Corrupted Checkpoint

If a checkpoint file is corrupted:

```bash
# 1. Check file
head -n 10 checkpoints/normalized_ids.txt
tail -n 10 checkpoints/normalized_ids.txt

# 2. If corrupted, rebuild from data
ls data/structured/ | sed 's/.json//' > checkpoints/normalized_ids.txt
```

---

## Monitoring Recovery

After recovery, check:

1. **Logs:**
   ```bash
   tail -f logs/ingestion.log
   ```

2. **Checkpoint consistency:**
   ```bash
   wc -l checkpoints/*.txt
   ```

3. **Data integrity:**
   ```bash
   ls -la data/faiss/*.index
   python -c "import faiss; idx = faiss.read_index('data/faiss/symptoms.index'); print(idx.ntotal)"
   ```

---

## Emergency Contacts

For issues that cannot be auto-recovered:

1. Check `logs/audit.log` for last successful operations
2. Review `logs/ingestion.log` for error details
3. Consult system documentation
4. Open GitHub issue with logs attached

---

## Prevention Best Practices

1. **Always use persistent storage** for `/data` directory
2. **Monitor GPU quota** to avoid mid-operation cutoff
3. **Keep checkpoint frequency high** (every 100 items)
4. **Test recovery** periodically by simulating failures
5. **Back up checkpoints** before major operations
6. **Version control configs** to track changes
