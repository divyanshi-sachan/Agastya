# Agastya

**Agastya** is a research codebase for **automated contract understanding**: turning long-form agreements into structured, machine-readable signals (e.g., whether specific clause types are present and what the text says). The project is developed in **phases**; **Phase 1** focuses on **transparent, reproducible classical ML** on the public [CUAD](https://www.atticusprojectai.org/cuad) corpus—no transformer or deep-learning models in this phase.

## Contents

- [Problem](#problem)
- [Approach (Phase 1)](#approach-phase-1)
- [Results (high level)](#results-high-level)
- [Folder structure](#folder-structure)
- [Usage](#usage)
- [Reproducible setup](#reproducible-setup)
- [Development practices](#development-practices)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [Roadmap (beyond Phase 1)](#roadmap-beyond-phase-1)

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

## Results (high level)

Work to date is **notebook-driven** and intended to be **auditable** (tables, plots, and narrative in each part).

- **Dataset quality & EDA** ([`Part_02_Dataset_Quality_EDA.ipynb`](notebooks/Phase_1/Part_02_Dataset_Quality_EDA.ipynb)): **510** contracts, **41** categories, **20,910** long-format rows; characterization of **Yes/No vs free-form** fields, **class imbalance**, **clause length** tails, and **filename / text join** edge cases (see **`progress.md`** for the decision checklist).
- **Feature engineering & baselines** ([`Part_03_Feature_Engineering.ipynb`](notebooks/Phase_1/Part_03_Feature_Engineering.ipynb)): builds sparse **X** and label **y**, **grouped** train/validation split, and an **ablation** of **TF-IDF vs TF-IDF + log-length** on example categories. On the committed split, **LinearSVC** benefits from adding length on average over the ablation pair; **MultinomialNB** is markedly weaker on **macro-F1** in the same setup—**open the notebook** for the full metric table, coefficient plots, and sparsity analysis.
- **Literature & theory** ([`Part_01_Literature_Review.ipynb`](notebooks/Phase_1/Part_01_Literature_Review.ipynb), [`Part_04_Theoretical_Rigor.ipynb`](notebooks/Phase_1/Part_04_Theoretical_Rigor.ipynb)): positions the work and ties modeling choices to textbook ML arguments.

> **Note:** Validation sets are small for rare categories (~102 contracts at a 20% grouped holdout). Treat single-split **macro-F1** as **illustrative**; the notebooks discuss stability (e.g., multiple seeds or grouped CV) for report-grade estimates.

---

## Folder structure

**In one line:** datasets under `data/`, reproducible Phase 1 analysis under `notebooks/Phase_1/`, future pipeline code under `src/`, static references under `Doc /`, and project metadata at the repo root.

### At a glance

| Location | What it is |
|----------|------------|
| **`data/CUAD_v1/`** | CUAD v1: `master_clauses.csv`, `CUAD_v1.json`, `full_contract_txt/`, `full_contract_pdf/`, `label_group_xlsx/`, and related assets. |
| **`notebooks/Phase_1/`** | Runnable notebooks: literature → EDA → feature engineering / baselines → theoretical rigor. |
| **`src/`** | Python package layout (`ocr`, `segmentation`, `models`, `reasoning`, `reporting`) for later phases; Phase 1 experiments live in notebooks, not here. |
| **`Doc /`** | PDFs (papers, rubric, etc.); not executed by the pipeline. |
| **Root (`*.md`, `requirements.txt`)** | `README.md` (this file), `requirements.txt`, `progress.md` (handoff notes), `agent.md` (Phase 1 rules), `project.md` (phase roadmap). |

### Tree

```
Agastya/
├── data/CUAD_v1/
├── notebooks/
│   └── Phase_1/            # Part_01 … Part_04 notebooks
├── src/
│   ├── ocr/
│   ├── segmentation/
│   ├── models/
│   ├── reasoning/
│   └── reporting/
├── Doc /                   # Reference PDFs
├── README.md
└── requirements.txt
```

**Phase 1** logic currently lives in **`notebooks/Phase_1/`**. **`src/`** is reserved for modular pipeline code (OCR, segmentation, hybrid reasoning) so unfinished modules stay separate from the graded, notebook-based analysis.

---

## Usage

### Run the analysis notebooks

From the repository root (after environment setup below):

```bash
jupyter lab
```

Recommended order:

1. `notebooks/Phase_1/Part_01_Literature_Review.ipynb`
2. `notebooks/Phase_1/Part_02_Dataset_Quality_EDA.ipynb`
3. `notebooks/Phase_1/Part_03_Feature_Engineering.ipynb`
4. `notebooks/Phase_1/Part_04_Theoretical_Rigor.ipynb`

Paths in notebooks assume the **working directory is the repo root** so `data/CUAD_v1/...` resolves correctly.

### Python package (future / optional)

When pipeline modules are implemented under `src/`, install in editable mode:

```bash
pip install -e .
```

*(Add a minimal `pyproject.toml` or `setup.cfg` when you are ready to expose `agastya` as an installable package.)*

---

## Reproducible setup

**Prerequisites**

- **Python 3.10+** recommended (matches current `scikit-learn` / `pandas` stacks).
- **Git** for version-aligned clones.

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

## Roadmap (beyond Phase 1)

Longer-term goals (not necessarily implemented in this tree) include OCR-backed ingestion, clause segmentation, transformer-based encoders, probabilistic reasoning over clause dependencies, and an end-user reporting layer. **Phase 1** deliberately grounds the project in **reproducible classical ML** and rigorous EDA before adding those components.
