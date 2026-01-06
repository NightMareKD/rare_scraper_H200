# 🏥 Clinical Case Similarity Detection System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚠️ CRITICAL DISCLAIMER

> **THIS IS NOT A DIAGNOSTIC TOOL**
>
> This system is designed for **clinical decision SUPPORT only**. It is intended to assist qualified healthcare professionals by identifying similar historical cases for educational and research purposes.
>
> - ❌ NOT for direct patient diagnosis
> - ❌ NOT for treatment recommendations
> - ❌ NOT a substitute for clinical judgment
> - ✅ For research and educational purposes
> - ✅ Requires clinician review of all results
> - ✅ Must be used within appropriate clinical governance

---

## 📋 Overview

This system aggregates, normalizes, and indexes clinical case reports from publicly available medical literature to enable similarity-based case retrieval. It is designed to help clinicians and researchers identify rare disease patterns by finding cases with similar clinical presentations.

### Key Features

- 🔍 **Multi-dimensional similarity search** across symptoms, ECG, labs, demographics, and imaging
- 📊 **Weighted composite scoring** with configurable importance factors
- 🛡️ **Clinical safety thresholds** to minimize false positives
- 📝 **Full audit trail** for regulatory compliance
- ⚡ **Optimized for Hugging Face Spaces** with GPU quota management
- 🔄 **Resumable pipeline** with checkpoint-based processing

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INGESTION (CPU)                              │
│  PubMed → NCBI → ESC → AHA → Journals                          │
│                      ↓                                          │
│              data/raw/ (persistent)                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                 NORMALIZATION (CPU)                             │
│  Text → Clinical Schema → Structured JSON                       │
│                      ↓                                          │
│              data/structured/                                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                 DISTILLATION (GPU)                              │
│  LLM Review → Rare/Non-rare Classification                      │
│                      ↓                                          │
│              data/distilled/                                    │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                  EMBEDDING (GPU)                                │
│  Clinical Text → Dense Vectors (per domain)                     │
│                      ↓                                          │
│              data/embeddings/                                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   INDEXING (CPU/GPU)                            │
│  Vectors → FAISS Indices (per domain)                           │
│                      ↓                                          │
│              data/faiss/                                        │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SERVING                                      │
│  Query → Multi-index Search → Weighted Composite → Results      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- 50GB+ disk space for data

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/clinical-similarity.git
cd clinical-similarity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the system
python app.py
```

### Hugging Face Spaces Deployment

1. Create a new Space on Hugging Face
2. Upload all files maintaining directory structure
3. Configure persistent storage for `/data` directory
4. Set environment variables as needed
5. The system will auto-detect GPU availability

---

## ⚙️ Configuration

### `config/sources.yaml`
Defines allowed data sources and legal boundaries for scraping.

### `config/clinical_schema.yaml`
Defines the clinical normalization schema (symptoms, ECG, labs, etc.).

### `config/weights.yaml`
Configures similarity importance weights per clinical domain.

### `config/thresholds.yaml`
Sets clinical safety thresholds for flagging cases.

### `config/gpu_policy.yaml`
Controls GPU usage quotas and batch sizes.

---

## 📊 Similarity Scoring

The system computes similarity across five clinical domains:

| Domain | Default Weight | Description |
|--------|----------------|-------------|
| Symptoms | 30% | Clinical presentation |
| ECG | 25% | Electrocardiogram findings |
| Labs | 25% | Laboratory values |
| Demographics | 10% | Age, sex, history |
| Imaging | 10% | Imaging findings |

**Composite Score** = Σ (domain_weight × domain_similarity)

Cases are flagged when composite score exceeds configured threshold.

---

## 📁 Directory Structure

```
/app
├── app.py                 # Main orchestrator
├── requirements.txt       # Pinned dependencies
├── config/               # Configuration files
├── runtime/              # Runtime documentation
├── data/                 # Persistent data storage
│   ├── raw/             # Unmodified source content
│   ├── structured/      # Normalized cases
│   ├── distilled/       # LLM-reviewed cases
│   ├── embeddings/      # Vector embeddings
│   └── faiss/           # FAISS indices
├── checkpoints/          # Processing checkpoints
├── logs/                 # System logs
└── docs/                 # Documentation
```

---

## 🔒 Security & Compliance

- All source data is from **publicly available** medical literature
- **Audit logs** track all similarity matches
- **Clinical disclaimers** are prominently displayed
- System designed with **explainability** in mind
- Compatible with medical device software guidelines

---

## 📝 Logging & Audit

| Log File | Purpose |
|----------|---------|
| `logs/ingestion.log` | Data collection events |
| `logs/normalization.log` | Schema mapping events |
| `logs/gpu_usage.log` | GPU quota tracking |
| `logs/similarity.log` | Query results |
| `logs/audit.log` | Clinical audit trail |

---

## 🧪 Validation Strategy

1. **Threshold Calibration**: Empirically tuned on known case pairs
2. **False Positive Control**: Conservative thresholds to minimize noise
3. **Clinical Review Loop**: All flagged cases require human review
4. **Continuous Monitoring**: Performance metrics tracked over time

See [docs/validation_strategy.md](docs/validation_strategy.md) for details.

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👥 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

---

## 📧 Contact

For questions about clinical use or research collaboration, please open an issue.

---

**Remember: This tool supports clinical decision-making. It does not replace it.**
