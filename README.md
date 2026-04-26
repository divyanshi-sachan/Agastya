# Agastya

**Agastya** is a research codebase for **automated contract understanding**: turning long-form agreements into structured, machine-readable signals (e.g., whether specific clause types are present and what the text says). The project is developed in **phases**: **Phase 1** focuses on **transparent, reproducible classical ML** baselines, while **Phase 2** introduces **Deep Learning (Transformers)** for state-of-the-art performance.

## Contents

- [Problem](#problem)
- [Approach (Phase 1)](#approach-phase-1)
- [Approach (Phase 2)](#approach-phase-2)
- [Approach (Phase 3)](#approach-phase-3)
- [Results (high level)](#results-high-level)
- [Folder structure](#folder-structure)
- [Usage (Phase 2)](#usage-phase-2)
- [Usage (Phase 3)](#usage-phase-3)
- [Reproducible setup](#reproducible-setup)
- [Development practices](#development-practices)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [Roadmap](#roadmap)

---

## Problem

Legal agreements are long, specialized, and unevenly digitized. Stakeholders without legal training face **high cognitive load** and **information asymmetry** when reviewing terms (payment, termination, liability, assignment, etc.). At machine-learning scale, contract understanding also raises engineering questions: **severe class imbalance** across clause categories, **multiple rows per document**, and **heavy-tailed** text lengths—all of which break naive “shuffle and accuracy” workflows.

This repository studies **clause-level prediction** on CUAD: given labeled contract structure, how far can **interpretable bag-of-words models** go when splits and metrics are chosen to match the real constraints of the data?

---

## Approach (Phase 1)

| Layer | Choice | Rationale |
|--------|--------|-----------|
| **Data** | CUAD v1 (`master_clauses.csv`, full-contract text, JSON) | Public, well-documented benchmark for contract QA / understanding |
| **Unit of analysis** | Long-format rows (per contract × clause category) with document-level grouping | Avoids leakage: the same contract must not appear in both train and validation |
| **Features** | `TfidfVectorizer` (word / character n-grams as configured in the notebook) | Strong baseline for legal surface form; sparse, inspectable weights |
| **Augmentation** | Optional `log1p` clause length, scaled alongside TF-IDF | EDA-motivated: length is informative without duplicating token counts |
| **Models** | **Linear SVM** (`LinearSVC`, `class_weight="balanced"`) as the primary model; **Multinomial Naive Bayes** as a fast baseline | Phase 1 mandate: scikit-learn–centric, no deep learning |
| **Evaluation** | **Macro-F1** (primary under imbalance), plus precision, recall, accuracy, and confusion matrices where appropriate | Aligns with rubric-style reporting and Part 02 EDA conclusions |

Methodology and design decisions are documented in **`progress.md`**, **`agent.md`**, and the Phase 1 notebooks under `notebooks/Phase_1/`.

---

## Approach (Phase 2)

| Layer | Choice | Rationale |
|--------|--------|-----------|
| **Data** | Processed CUAD v1 (`train.csv`, `val.csv`, `test.csv`) | Grouped splits to ensure document-level isolation and consistent evaluation. |
| **Model** | **Legal-BERT** (backbone: `nlpaueb/legal-bert-base-uncased`) | Domain-specific pre-training significantly improves legal text comprehension. |
| **Architecture** | BERT with Length-Aware Linear Head | Combines transformer representations with clause length features (normalized log-length). |
| **Preprocessing** | Sliding window segmentation (max 128-512 tokens) | Handles long contracts by breaking them into manageable, overlapping spans. |
| **Training** | PyTorch + HuggingFace Transformers | Standardized pipeline for fine-tuning, using `AdamW` and weighted loss for imbalance. |
| **Evaluator** | Macro-F1 (primary), Confusion Matrix | Rigorous per-class testing to avoid accuracy bias on rare clause types. |

Phase 2 implementation details and experiments are found in `src/phase2/` and `notebooks/Phase_2/`.

---

## Approach (Phase 3)

| Layer | Choice | Rationale |
|--------|--------|-----------|
| **Hybrid stack** | Legal-BERT + Bayesian Network + Interface Layer | Combines strong clause semantics with explicit probabilistic reasoning over contract risk. |
| **OCR ingestion** | Tiered extractor (`pdfplumber` first, EasyOCR fallback) | Keeps digital PDFs fast/accurate while still supporting scanned contracts and images. |
| **Interface layer** | Evidence encoder + confidence mapper + cross-clause feature extractor | Converts neural outputs into hard and soft evidence for Bayesian inference. |
| **Probabilistic model** | 8-node Bayesian Network with EM refinement | Encodes legal priors, then adapts CPTs from training evidence. |
| **Evaluation** | Ablation: ML-only vs DL-only vs Hybrid | Measures whether neuro-symbolic fusion provides gains over standalone models. |
| **Delivery surface** | Streamlit app (`app/streamlit_app.py`) | Provides interactive upload-to-risk-analysis workflow and explainable traces. |

Phase 3 implementation details and experiments are found in `src/phase3/`, `notebooks/Phase_3/`, and `reports/phase3/`.

---

## Results (high level)

Work to date is **notebook-driven** and intended to be **auditable** (tables, plots, and narrative in each part).

### Phase 1: Classical Baselines
- **Dataset quality & EDA** ([`Part_02_Dataset_Quality_EDA.ipynb`](notebooks/Phase_1/Part_02_Dataset_Quality_EDA.ipynb)): **510** contracts, **41** categories; identified severe class imbalance and extreme text lengths.
- **Feature engineering & baselines** ([`Part_03_Feature_Engineering.ipynb`](notebooks/Phase_1/Part_03_Feature_Engineering.ipynb)): **LinearSVC** reached competitive baselines (~0.5-0.6 Macro-F1 on selected categories) using TF-IDF and character n-grams.

### Phase 2: Deep Learning (Legal-BERT)
The transition to deep learning yielded a significant performance jump by leveraging context-aware embeddings.

| Metric | Result | Notes |
|--------|--------|-------|
| **Macro-F1** | **0.746** | Strong performance across heterogeneous legal clauses. |
| **Accuracy** | **0.824** | Good overall classification on the held-out test set. |
| **Validation Loss** | **18.68** | Stable convergence over multiple training epochs. |

- **Fine-Tuning** ([`07_Training_Bert.ipynb`](notebooks/Phase_2/07_Training_Bert.ipynb)): Achieved convergence with `Legal-BERT` using 128-token windows and document-level splits.
- **Evaluation** ([`08_evaluation.ipynb`](notebooks/Phase_2/08_evaluation.ipynb)): Detailed per-class analysis (refer to `phase2_results/results.json` and the `confusion_matrix.png`).
- **Interpretability** ([`09_interpretability.ipynb`](notebooks/Phase_2/09_interpretability.ipynb)): Initial exploration of attention weights and feature salience for legal entities.

### Phase 3: Hybrid Reasoning (Neuro-Symbolic)
Phase 3 introduces an end-to-end hybrid pipeline integrating OCR, Legal-BERT outputs, and Bayesian risk inference.

- **Bayesian Network + CPT setup** ([`10_bn_structure_and_cpts.ipynb`](notebooks/Phase_3/10_bn_structure_and_cpts.ipynb)): Defines the graph structure and seed/refined CPT behavior.
- **Hybrid pipeline demo** ([`11_hybrid_pipeline_demo.ipynb`](notebooks/Phase_3/11_hybrid_pipeline_demo.ipynb)): Demonstrates evidence encoding, posterior risk inference, and traceable outputs.
- **Ablation study** ([`12_ablation_study.ipynb`](notebooks/Phase_3/12_ablation_study.ipynb)): Compares ML-only, DL-only, and Hybrid configurations.
- **Interpretability report** ([`13_interpretability_report.ipynb`](notebooks/Phase_3/13_interpretability_report.ipynb)): Summarizes posterior behavior and cross-phase progression.
- **Artifacts**: Metrics and diagnostics are tracked in `reports/phase3/` (CSV/JSON) and models/checkpoints in `results/phase3/`.

---

## Folder structure

**In one line:** datasets under `data/`, reproducible Phase 1 analysis under `notebooks/Phase_1/`, future pipeline code under `src/`, static references under `Doc /`, and project metadata at the repo root.

### At a glance

| Location | What it is |
|----------|------------|
| **`data/CUAD_v1/`** | CUAD v1: `master_clauses.csv`, `CUAD_v1.json`, `full_contract_txt/`, `full_contract_pdf/`, `label_group_xlsx/`, and related assets. |
| **`notebooks/Phase_1/`** | Runnable notebooks: literature → EDA → feature engineering / baselines → theoretical rigor. |
| **`notebooks/Phase_2/`** | OCR → segmentation → preprocessing → baselines → transformers → multitask → evaluation → ablations → interpretability. |
| **`src/`** | Phase 1 reserved layout (`ocr`, …, `reporting`) at top level; **Phase 2 code** lives under **`src/phase2/`** (OCR, segmentation, data loaders, models, training, evaluation, interpretability, utils). |
| **`configs/`** | YAML defaults for data, model, and training (`*_config.yaml`). |
| **`scripts/`** | CLI entry points: `train.py`, `evaluate.py`, `run_ocr.py`. |
| **`data/raw/`**, **`data/interim/`**, **`data/processed/`** | Phase 2 ingestion layout (CUAD or scanned PDFs → OCR/segments → `train.csv` / `val.csv` / `test.csv`). Existing CUAD v1 copy remains at **`data/CUAD_v1/`** for Phase 1. |
| **`results/phase2/`** | Generated plots, checkpoints, and logs (keep large binaries out of git unless intentional). |
| **`reports/phase2/`** | Write-ups: methodology, experiments, results, ablations, conclusions. |
| **`Doc /`** | PDFs (papers, rubric, etc.); not executed by the pipeline. |
| **Root (`*.md`, `requirements.txt`)** | `README.md` (this file), `requirements.txt`, `progress.md` (handoff notes), `agent.md` (Phase 1 rules), `project.md` (phase roadmap). |

### Tree

```
Agastya/
├── configs/                # Phase 2 YAML (model / training / data)
├── data/CUAD_v1/
├── data/                   # CUAD_v1 (Phase 1) + raw/interim/processed (Phase 2)
├── notebooks/
│   ├── Phase_1/            # Part_01 … Part_05 notebooks
│   └── Phase_2/            # 01 … 09 Phase 2 notebooks
├── reports/phase2/
├── results/phase2/
├── scripts/                # train, evaluate, run_ocr
├── src/
│   ├── __init__.py
│   ├── ocr/                # reserved / future shared layout
│   ├── phase2/             # Phase 2 package
│   ├── segmentation/
│   ├── models/
│   ├── reasoning/
│   └── reporting/
├── Doc /                   # Reference PDFs
├── README.md
└── requirements.txt
```

**Phase 1** logic currently lives in **`notebooks/Phase_1/`**. **`src/phase2/`** is the scaffold for modular Phase 2 code (OCR through interpretability); Phase 1 notebooks remain the source of truth for classical baselines.

### Phase 2 layout

Runnable narrative and experiments: **`notebooks/Phase_2/`** (`01`–`09`). Library code: **`src/phase2/`**. Paths in configs default to **`data/processed/*.csv`** and **`results/phase2/`**; adjust **`configs/data_config.yaml`** if you symlink CUAD from **`data/CUAD_v1/`** into **`data/raw/CUAD/`**.

```
Agastya/
├── configs/
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── data_config.yaml
├── data/
│   ├── CUAD_v1/              # Phase 1 copy (unchanged)
│   ├── raw/
│   │   ├── CUAD/
│   │   └── scanned_contracts/
│   ├── interim/
│   │   ├── ocr_outputs/
│   │   └── segmented_clauses/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── Phase_1/
│   └── Phase_2/              # 01_ocr_pipeline … 09_interpretability
├── reports/
│   └── phase2/
├── results/
│   └── phase2/
│       ├── plots/
│       ├── models/
│       └── logs/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_ocr.py
└── src/
    └── phase2/
        ├── ocr/
        ├── segmentation/
        ├── data/
        ├── models/
        ├── training/
        ├── evaluation/
        ├── interpretability/
        └── utils/
```

---

## Usage (Phase 2)

### Running the BERT Pipeline

1. **Preprocessing:** Segment CUAD data into training/val/test splits.
   ```bash
   python scripts/run_ocr.py  # If starting from raw PDFs
   # Or use the preprocessing notebook 05_data_preprocessing.ipynb
   ```

2. **Training:** Fine-tune Legal-BERT on the processed segments.
   ```bash
   python scripts/train.py --config configs/training_config.yaml
   ```

3. **Evaluation:** Generate metrics and plots for the test set.
   ```bash
   python scripts/evaluate.py --model_path phase2_results/legal_bert_phase2.pt
   ```

### Notebook Exploration
The **`notebooks/Phase_2/`** directory contains the full developmental narrative, from **01_architectural_logic** to **08_evaluation**.

---

## Usage (Phase 3)

### 1) Train / refresh Phase 3 components

```bash
python -m src.phase3.risk_bert_train_cli
python -m src.phase3.bayesian.train_cli
python -m src.phase3.tier3_train_cli
```

### 2) Run hybrid evaluation and ablations

```bash
python -m src.phase3.hybrid_eval_cli
python -m src.phase3.ablation_cli
```

### 3) Launch the demo app

```bash
streamlit run app/streamlit_app.py
```

### 4) Run Phase 3 tests

```bash
pytest tests/phase3 -v
```

---

## Reproducible setup

**Prerequisites**

- **Python 3.10+** recommended (matches current `scikit-learn` / `pandas` stacks).
- **Git** for version-aligned clones.
- **Poppler** (required for PDF rasterization in Phase 3 OCR fallback):
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `sudo apt install poppler-utils`

**Steps**

```bash
git clone <your-repository-url>.git
cd Agastya

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Data:** This repository includes CUAD v1 under `data/CUAD_v1/` so notebooks run without a separate download step. If you mirror the project without large files, obtain CUAD from the [Atticus Project](https://www.atticusprojectai.org/cuad) and place it at the same paths.

**Determinism:** Notebooks use explicit **`random_state`** where randomness enters (e.g., splits, linear SVM). Re-run cells top-to-bottom after pulling changes.

---

## Development practices

- **Commits:** Prefer **small, coherent commits** with **clear messages** (e.g., [Conventional Commits](https://www.conventionalcommits.org/): `feat(notebooks): add grouped split for Part 03`, `docs: expand README results section`). This makes progress visible to reviewers and matches “steady development” expectations.
- **Phase boundaries:** **Phase 1** — no transformers/BERT for modeling; see **`project.md`** / **`agent.md`**. Later phases may add deep learning and hybrid reasoning on top of this baseline.
- **Housekeeping:** Virtual environments (`.venv/`) and Jupyter checkpoints are **gitignored**; keep generated artifacts out of version control unless they are deliberately released.

---

## Team

- Divyanshi Sachan  
- Subham Mahapatra  

---

## Acknowledgments

Analysis uses the **Contract Understanding Atticus Dataset (CUAD)**. Cite the dataset and license terms from the official CUAD release when publishing or redistributing.

---

## Roadmap

- **Phase 1 (Complete):** Classical ML baselines (SVM/NB) on CUAD.
- **Phase 2 (Complete):** Deep Learning transition with Domain-Specific Transformers (Legal-BERT).
- **Phase 3 (In Progress):** Hybrid reasoning pipeline with OCR, interface-layer evidence fusion, Bayesian risk inference, and Streamlit-based analysis UI.
