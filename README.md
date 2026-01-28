# Transformer-Based Intrusion Detection System for DoS Attacks

This repository contains the implementation of a research-grade Transformer-based Intrusion Detection System (IDS) designed for Denial-of-Service (DoS) attack detection using the CIC-IDS 2017 dataset.

The system follows a task-aware minimalist design philosophy, prioritizing:

*   Computational efficiency
*   Real-time deployability
*   High detection accuracy for DoS attacks

The implementation is suitable for final-year projects, research publications, and reproducible experimentation.

## 📌 Project Highlights

*   Lightweight Transformer encoder-only architecture
    *   ❌ No CNNs, ❌ No LSTMs, ❌ No hybrid fusion models
*   DoS-aware temporal modeling via burst window sequencing
*   Explicit handling of severe class imbalance
*   GPU-accelerated training and inference
*   Deterministic, research-reproducible pipeline

## 🗂 Directory Structure

```
Implementation/
│
├── data/
│   ├── raw/
│   │   └── Wednesday-workingHours.csv   (not tracked in Git)
│   └── processed/                        (generated locally)
│       ├── train/
│       ├── val/
│       └── test/
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── utils/
│   └── main.py
│
├── logs/
├── results/
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

CIC-IDS 2017

File used: `Wednesday-workingHours.csv`

Attack classes considered:

*   BENIGN
*   DoS Hulk
*   DoS GoldenEye
*   DoS Slowloris
*   DoS SlowHTTPTest

⚠️ Rare non-DoS attacks (e.g., Heartbleed) are excluded to maintain task consistency, which is standard practice in IDS research.

## 📥 Dataset Setup (IMPORTANT)

Due to GitHub file size restrictions, the dataset is not included in this repository.

### 🔹 Step-by-Step Instructions

1.  Download the CIC-IDS 2017 dataset from the official source:
    👉 [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)

2.  Extract the following file:
    `Wednesday-workingHours.csv`

3.  Inside the project root, create the directory structure:

    ```
    data/
    ├── raw/
    │   └── Wednesday-workingHours.csv
    └── processed/
        ├── train/
        ├── val/
        └── test/
    ```

4.  Place `Wednesday-workingHours.csv` inside:
    `data/raw/`

5.  Run the pipeline:
    `python src/main.py`

    ✔ Train / validation / test splits will be generated automatically and stored locally in `data/processed/`.

## ⚙️ Environment Setup (Python 3.12.11)

### 1️⃣ Create Virtual Environment

```bash
python3.12 -m venv FYP_env
source FYP_env/bin/activate      # Linux / macOS
FYP_env\\Scripts\\activate         # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the Experiment

```bash
python src/main.py
```

The pipeline will:

*   Perform preprocessing & feature pruning
*   Apply DoS-aware temporal windowing
*   Train a lightweight Transformer encoder
*   Evaluate on validation and test sets
*   Log accuracy, precision, recall, F1-score
*   Save confusion matrix and paper-ready results

## 📈 Outputs

After execution, the following will be generated locally:

*   `logs/` → experiment logs
*   `results/`
    *   `metrics.json`
    *   `results.tex` (LaTeX table for paper)
    *   `confusion_matrix.png`

## 🔁 Reproducibility

*   Fixed random seed
*   Deterministic data splits
*   Fully configurable via `configs/config.yaml`

Once the dataset is placed correctly, all results are reproducible.

## 📜 License & Usage

This repository is intended for:

*   Academic research
*   Final-year projects
*   Experimental validation

If you use this work in a publication, please cite appropriately.