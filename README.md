# DoS-Aware Lightweight Transformer-Based Intrusion Detection System

This repository contains the implementation of a **research-grade Transformer-based Intrusion Detection System (IDS)** designed for **Denial-of-Service (DoS) attack detection** using the **CIC-IDS 2017 dataset**.

The system follows a **task-aware minimalist design philosophy**, prioritizing:

- Computational efficiency
- Real-time deployability
- High detection accuracy for DoS attacks

The implementation is suitable for **final-year projects, research publications, and experimental reproducibility**.

---

## 📌 Project Highlights

- Lightweight **Transformer Encoder-only architecture**
- No CNNs, no LSTMs, no hybrid complexity
- DoS-aware temporal modeling using **burst window sequencing**
- Handles **severe class imbalance**
- GPU-accelerated training and inference
- Research-reproducible experiment pipeline

---

## 🗂 Directory Structure

```
Implementation/
│
├── data/
│ ├── raw/
│ │ └── Wednesday-workingHours.csv
│ └── processed/
│ ├── train/
│ ├── val/
│ └── test/
│
├── configs/
│ └── config.yaml
│
├── src/
│ ├── data/
│ ├── models/
│ ├── training/
│ ├── utils/
│ └── main.py
│
├── logs/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

- **CIC-IDS 2017**
- File used: `Wednesday-workingHours.csv`
- Reason: Contains multiple DoS attack types:
  - DoS Hulk
  - DoS GoldenEye
  - DoS Slowloris
  - DoS SlowHTTPTest

---

## ⚙️ Environment Setup (Python 3.12.11)

### 1️⃣ Create Virtual Environment

```bash
python3.12 -m venv FYP_env
source FYP_env/bin/activate      # Linux/Mac
FYP_env\Scripts\activate         # Windows
```
