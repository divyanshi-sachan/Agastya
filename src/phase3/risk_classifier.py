"""Tier 3: Direct contract-level risk classifier trained on Legal-BERT [CLS] embeddings.

Trains a lightweight 3-class classifier (Low / Medium / High) by:
1. Extracting mean-pooled [CLS] embeddings from all clause segments per contract.
2. Fitting a calibrated LogisticRegression on the val set for Platt scaling.
3. Persisting the classifier for use in the hybrid ensemble.

Ensemble variants
-----------------
- ``ensemble_predict``               — 2-way: BN + frozen-BERT LR probe
- ``ensemble_predict_with_finetuned``— 3-way: BN + frozen-BERT LR probe + fine-tuned BERT
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from src.phase3.risk_bert import RiskBertTrainer as _RiskBertTrainer

from src.phase2.models.bert_classifier import BertWithLengthClassifier
from src.phase2.segmentation.clause_splitter import split_clauses
from src.phase3.hybrid_eval import (
    _CONFIDENTIALITY_LABELS,
    _DISPUTE_LABELS,
    _LIABILITY_LABELS,
    _PAYMENT_LABELS,
    _TERMINATION_LABELS,
    build_contract_dataset,
)

_RISK_CLASSES = ["Low", "Medium", "High"]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _load_bert(checkpoint_path: str, label_map_path: str, model_name: str = "nlpaueb/legal-bert-base-uncased"):
    with open(label_map_path) as f:
        label2id = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertWithLengthClassifier(
        model_name=model_name,
        num_classes=len(label2id),
        use_length_feature=True,
        download_pretrained_backbone=True,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return tokenizer, model


def _embed_contract(text: str, tokenizer, model, max_length: int = 256) -> np.ndarray:
    """Return mean-pooled [CLS] vector over all clause segments (shape: 768,)."""
    clauses = split_clauses(text)
    if not clauses:
        return np.zeros(768, dtype=np.float32)

    embeddings = []
    for clause in clauses:
        if not clause.strip():
            continue
        encoded = tokenizer(clause, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            out = model.bert(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            cls_vec = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        embeddings.append(cls_vec)

    return np.mean(embeddings, axis=0).astype(np.float32) if embeddings else np.zeros(768, dtype=np.float32)


def extract_embeddings_for_split(
    csv_path: str,
    checkpoint_path: str,
    label_map_path: str,
    output_path: str,
) -> tuple[np.ndarray, list[str]]:
    """Extract and cache [CLS] embeddings for all contracts in a CSV split."""
    out = Path(output_path)
    if out.exists():
        data = np.load(str(out), allow_pickle=True)
        print(f"  Loaded cached embeddings from {out}")
        return data["X"], data["y"].tolist()

    df = pd.read_csv(csv_path)
    contract_df = build_contract_dataset(df)
    tokenizer, model = _load_bert(checkpoint_path, label_map_path)

    X, y = [], []
    for i, row in contract_df.iterrows():
        print(f"  Embedding contract {i+1}/{len(contract_df)}: {row['filename'][:40]}", end="\r")
        emb = _embed_contract(row["text"], tokenizer, model)
        X.append(emb)
        y.append(row["true_risk"])

    print()
    X_arr = np.vstack(X)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), X=X_arr, y=np.array(y))
    print(f"  Saved embeddings → {out}")
    return X_arr, y


# ---------------------------------------------------------------------------
# Classifier training (Tier 3, Step 10)
# ---------------------------------------------------------------------------

def train_risk_classifier(
    train_csv: str = "data/processed/train.csv",
    val_csv: str = "data/processed/val.csv",
    checkpoint_path: str = "results/phase2/models/legal_bert_phase2.pt",
    label_map_path: str = "results/phase2/label2id.json",
    output_model_path: str = "results/phase3/risk_classifier.pkl",
    train_cache: str = "results/phase3/train_embeddings.npz",
    val_cache: str = "results/phase3/val_embeddings.npz",
) -> dict:
    """Train a 3-class MLP risk classifier on Legal-BERT [CLS] embeddings."""
    print("=== Extracting train embeddings ===")
    X_train, y_train = extract_embeddings_for_split(
        train_csv, checkpoint_path, label_map_path, train_cache
    )
    print("=== Extracting val embeddings ===")
    X_val, y_val = extract_embeddings_for_split(
        val_csv, checkpoint_path, label_map_path, val_cache
    )

    print(f"\nTrain: {X_train.shape}, classes: {dict(pd.Series(y_train).value_counts())}")
    print(f"Val:   {X_val.shape},   classes: {dict(pd.Series(y_val).value_counts())}")

    # Logistic Regression works very well on high-dim BERT embeddings
    # and supports native probability calibration (no CalibratedClassifierCV needed).
    base_clf = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=42,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf = CalibratedClassifierCV(base_clf, cv=5, method="sigmoid")
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, labels=_RISK_CLASSES, average="macro", zero_division=0)
    print(f"\nVal Macro-F1 (direct classifier): {val_f1:.4f}")
    print(classification_report(y_val, val_preds, labels=_RISK_CLASSES, zero_division=0))

    out = Path(output_model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(clf, f)
    print(f"Classifier saved → {out}")

    return {"val_macro_f1": val_f1, "model_path": str(out)}


def load_risk_classifier(path: str = "results/phase3/risk_classifier.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Ensemble fusion (Tier 3, Step 11)
# ---------------------------------------------------------------------------

def ensemble_predict(
    contract_text: str,
    hybrid_pipeline,
    risk_classifier,
    tokenizer,
    bert_model,
    bn_weight: float = 0.5,
    clf_weight: float = 0.5,
) -> dict:
    """Blend BN posterior with direct classifier probabilities.

    bn_weight + clf_weight should sum to 1.0.
    """
    # BN path
    bn_result = hybrid_pipeline.predict(contract_text)
    bn_probs = np.array([
        bn_result["risk_probabilities"].get(c, 0.0) for c in _RISK_CLASSES
    ])

    # Direct classifier path
    emb = _embed_contract(contract_text, tokenizer, bert_model)
    clf_probs = risk_classifier.predict_proba(emb.reshape(1, -1))[0]
    # Align class order with _RISK_CLASSES
    clf_classes = list(risk_classifier.classes_)
    clf_probs_aligned = np.array([
        clf_probs[clf_classes.index(c)] if c in clf_classes else 0.0
        for c in _RISK_CLASSES
    ])

    # Weighted blend
    blended = bn_weight * bn_probs + clf_weight * clf_probs_aligned
    blended /= blended.sum()  # renormalize
    predicted_risk = _RISK_CLASSES[int(np.argmax(blended))]

    return {
        **bn_result,
        "risk_level": predicted_risk,
        "risk_probabilities": {c: float(blended[i]) for i, c in enumerate(_RISK_CLASSES)},
        "bn_probs": {c: float(bn_probs[i]) for i, c in enumerate(_RISK_CLASSES)},
        "clf_probs": {c: float(clf_probs_aligned[i]) for i, c in enumerate(_RISK_CLASSES)},
        "blend_weights": {"bn": bn_weight, "clf": clf_weight},
    }


# ---------------------------------------------------------------------------
# 3-Way Ensemble (BN + frozen-BERT LR probe + fine-tuned BERT)
# ---------------------------------------------------------------------------

def ensemble_predict_with_finetuned(
    contract_text: str,
    hybrid_pipeline,
    risk_classifier,
    tokenizer,
    bert_model,
    finetuned_tokenizer,
    finetuned_model,
    bn_weight: float = 0.35,
    clf_weight: float = 0.30,
    finetuned_weight: float = 0.35,
    max_length: int = 512,
) -> dict:
    """Blend BN posterior, frozen-BERT LR probe, and fine-tuned BERT probabilities.

    All three weights must sum to 1.0 (they are re-normalised internally so
    minor floating-point drift is tolerated).

    Args:
        contract_text:      Full concatenated contract text.
        hybrid_pipeline:    ``AgastyaHybridPipeline`` instance (BN path).
        risk_classifier:    Calibrated sklearn classifier (frozen-BERT LR probe path).
        tokenizer:          Tokeniser for the frozen BERT embedding model.
        bert_model:         Frozen ``BertWithLengthClassifier`` (for CLS embedding extraction).
        finetuned_tokenizer:Tokeniser for the fine-tuned ``RiskBertHead``.
        finetuned_model:    Fine-tuned ``RiskBertHead`` (INT8 weights, eval mode).
        bn_weight:          Weight for BN posterior (default 0.35).
        clf_weight:         Weight for frozen-BERT LR probe (default 0.30).
        finetuned_weight:   Weight for fine-tuned BERT (default 0.35).
        max_length:         Tokeniser truncation length for fine-tuned model (default 512).

    Returns:
        Extended result dict with keys from ``ensemble_predict`` plus:
        - ``finetuned_probs``: raw probabilities from the fine-tuned model
        - ``blend_weights``:   all three weight values
    """
    assert abs(bn_weight + clf_weight + finetuned_weight - 1.0) < 1e-4, (
        f"Weights must sum to 1.0; got {bn_weight + clf_weight + finetuned_weight:.4f}"
    )

    # -- Path 1: BN posterior --
    bn_result = hybrid_pipeline.predict(contract_text)
    bn_probs = np.array(
        [bn_result["risk_probabilities"].get(c, 0.0) for c in _RISK_CLASSES],
        dtype=np.float64,
    )

    # -- Path 2: frozen-BERT + LR probe --
    emb = _embed_contract(contract_text, tokenizer, bert_model)
    clf_probs = risk_classifier.predict_proba(emb.reshape(1, -1))[0]
    clf_classes = list(risk_classifier.classes_)
    clf_probs_aligned = np.array(
        [clf_probs[clf_classes.index(c)] if c in clf_classes else 0.0 for c in _RISK_CLASSES],
        dtype=np.float64,
    )

    # -- Path 3: fine-tuned BERT --
    ft_prob_dict = _RiskBertTrainer.predict_proba(
        contract_text, finetuned_tokenizer, finetuned_model, max_length=max_length
    )
    ft_probs = np.array([ft_prob_dict.get(c, 0.0) for c in _RISK_CLASSES], dtype=np.float64)

    # -- Weighted blend + renormalise --
    blended = bn_weight * bn_probs + clf_weight * clf_probs_aligned + finetuned_weight * ft_probs
    total = blended.sum()
    if total > 0:
        blended /= total
    predicted_risk = _RISK_CLASSES[int(np.argmax(blended))]

    return {
        **bn_result,
        "risk_level": predicted_risk,
        "risk_probabilities": {c: float(blended[i]) for i, c in enumerate(_RISK_CLASSES)},
        "bn_probs":         {c: float(bn_probs[i])          for i, c in enumerate(_RISK_CLASSES)},
        "clf_probs":        {c: float(clf_probs_aligned[i]) for i, c in enumerate(_RISK_CLASSES)},
        "finetuned_probs":  {c: float(ft_probs[i])         for i, c in enumerate(_RISK_CLASSES)},
        "blend_weights":    {"bn": bn_weight, "clf": clf_weight, "finetuned": finetuned_weight},
    }
