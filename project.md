#  Project: Agastya – AI Contract Analysis System

##  Overview
Agastya is an AI system for analyzing legal contracts. It extracts text from documents, segments clauses, classifies them, and detects risks.

---

##  Phase-Based Development

This project is divided into 3 phases:

### Phase 1 (CURRENT)
- Machine Learning ONLY
- No deep learning or transformers allowed

### Phase 2
- Deep Learning (BERT, transformers)

### Phase 3
- Hybrid system (ML + DL + reasoning)

---

##  Current Goal (Phase 1)

Task: Clause Classification

Input:
Clause text

Output:
Clause category:
- Payment
- Termination
- Confidentiality
- Liability
- Governing Law

---

## 📊 Dataset

Primary dataset:
- CUAD (Contract Understanding Atticus Dataset)

Data format:
Clause text → Label

---

## ⚙️ Pipeline (Phase 1)

Contract → OCR  → Clause Segmentation → TF-IDF → ML Model → Prediction

---

##  Machine Learning Approach

### Feature Engineering
- TF-IDF (main)
- N-grams (important)
- Keyword features (optional)
- Clause length (optional)

---

### Models

Baseline:
- Naive Bayes

Main model:
- SVM

---

##  Evaluation Metrics

- F1-score (primary)
- Precision
- Recall
- Confusion Matrix

---

##  Restrictions

- Do NOT use BERT
- Do NOT use transformers
- Do NOT use deep learning

---

##  Expected Output

- Trained ML model
- Predictions
- Evaluation results
- Basic EDA insights