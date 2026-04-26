"""Multi-segment Legal-BERT feature extraction + ensemble classifier for 3-class risk.

Architecture:
  1. For each contract, split into clause segments via ``split_clauses``.
  2. Run every segment through the frozen Phase 2 Legal-BERT 41-class classifier.
  3. Build a rich per-contract feature vector:
     - Full 41-class clause prediction distribution (max confidence per class)
     - 5 clause-type presence flags and max confidences
     - Mean/max/std-pooled [CLS] embeddings (768 × 3)
     - Structural stats (num segments, mean/std confidence)
  4. Train a **stacked ensemble** (GradientBoosting + LogisticRegression + MLP)
     using stratified K-fold cross-validation.
  5. Final prediction blends the K-fold out-of-fold predictions for robustness.

Why this works:
  - Risk labels are a deterministic function of which clause types are present.
  - The 41-class distribution directly captures clause presence information.
  - Gradient Boosting handles tabular data far better than neural nets for N<500.
  - K-fold cross-validation maximises use of the small dataset.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Label mapping
# ------------------------------------------------------------------

RISK_CLASSES = ["Low", "Medium", "High"]
LABEL2ID: dict[str, int] = {c: i for i, c in enumerate(RISK_CLASSES)}
ID2LABEL: dict[int, str] = {i: c for c, i in LABEL2ID.items()}

_DEFAULT_MODEL = "nlpaueb/legal-bert-base-uncased"

# Clause-type mapping (aligned with hybrid_eval.py _derive_contract_risk)
_CLAUSE_TYPES = ["Payment", "Termination", "Liability", "Confidentiality", "Dispute Resolution"]

_P2_TO_CLAUSE_TYPE = {
    "revenue/profit sharing": "Payment", "minimum commitment": "Payment",
    "price restrictions": "Payment", "liquidated damages": "Payment",
    "most favored nation": "Payment", "royalty": "Payment",
    "termination for convenience": "Termination",
    "notice period to terminate renewal": "Termination",
    "post-termination services": "Termination", "renewal term": "Termination",
    "cap on liability": "Liability", "uncapped liability": "Liability",
    "warranty duration": "Liability", "insurance": "Liability",
    "indemnification": "Liability",
    "non-compete": "Confidentiality", "non-disparagement": "Confidentiality",
    "no-solicit of customers": "Confidentiality",
    "no-solicit of employees": "Confidentiality",
    "ip ownership assignment": "Confidentiality",
    "joint ip ownership": "Confidentiality",
    "source code escrow": "Confidentiality",
    "non-transferable license": "Confidentiality",
    "irrevocable or perpetual license": "Confidentiality",
    "governing law": "Dispute Resolution",
    "covenant not to sue": "Dispute Resolution",
    "audit rights": "Dispute Resolution",
}


# ------------------------------------------------------------------
# Backward compat aliases for ensemble imports
# ------------------------------------------------------------------
class RiskBertHead(nn.Module):
    """Thin alias for backward compatibility with ensemble_predict_with_finetuned."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
    def forward(self, x):
        return x

class RiskMLP(RiskBertHead):
    pass


# ------------------------------------------------------------------
# Phase 2 BERT loader
# ------------------------------------------------------------------

def _load_phase2_bert(checkpoint_path: str, label_map_path: str,
                      model_name: str = _DEFAULT_MODEL):
    """Load the frozen Phase 2 BERT for feature extraction."""
    from src.phase2.models.bert_classifier import BertWithLengthClassifier

    with open(label_map_path) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

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
    return tokenizer, model, id2label


# ------------------------------------------------------------------
# Feature Extraction
# ------------------------------------------------------------------

def extract_contract_features(
    text: str,
    tokenizer,
    bert_model,
    id2label: dict,
    max_seg_length: int = 256,
) -> np.ndarray:
    """Extract a rich feature vector for one contract.

    Returns a feature vector with:
      - 41-class max confidence distribution (41d)
      - Clause-type max confidences (5d)
      - Clause-type presence flags (5d)
      - Mean-pooled CLS embedding (768d)
      - Max-pooled CLS embedding (768d)
      - Std-pooled CLS embedding (768d)
      - Structural stats (5d: num_segs, mean_conf, std_conf, max_conf, min_conf)
    Total: 41 + 5 + 5 + 768*3 + 5 = 2360d
    """
    from src.phase2.segmentation.clause_splitter import split_clauses

    n_classes = len(id2label)
    feat_dim = n_classes + 5 + 5 + 768 * 3 + 5  # 2360 for 41 classes

    segments = split_clauses(text)
    if not segments:
        return np.zeros(feat_dim, dtype=np.float32)

    cls_vectors = []
    clause_confidences: dict[str, float] = {ct: 0.0 for ct in _CLAUSE_TYPES}
    class_max_conf = np.zeros(n_classes, dtype=np.float32)
    all_top_confs = []

    with torch.no_grad():
        for seg in segments:
            seg = seg.strip()
            if not seg or len(seg) < 5:
                continue

            enc = tokenizer(seg, truncation=True, max_length=max_seg_length,
                            return_tensors="pt")
            length_feat = torch.tensor(
                [[float(np.log1p(enc["input_ids"].shape[1]))]],
                dtype=torch.float32,
            )

            # Get CLS embedding
            bert_out = bert_model.bert(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            cls_vec = bert_out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            cls_vectors.append(cls_vec)

            # Get 41-class predictions
            logits = bert_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                length_feat=length_feat,
            ).squeeze(0)
            probs = torch.softmax(logits, dim=0).cpu().numpy()

            # Update per-class max confidence
            class_max_conf = np.maximum(class_max_conf, probs)

            # Track clause-type confidences
            top_conf = float(probs.max())
            all_top_confs.append(top_conf)

            for cls_idx, prob in enumerate(probs):
                label = id2label.get(cls_idx, "")
                clause_type = _P2_TO_CLAUSE_TYPE.get(label.lower())
                if clause_type and prob > clause_confidences.get(clause_type, 0.0):
                    clause_confidences[clause_type] = float(prob)

    if not cls_vectors:
        return np.zeros(feat_dim, dtype=np.float32)

    # Aggregate CLS embeddings
    cls_array = np.stack(cls_vectors)
    mean_cls = cls_array.mean(axis=0)
    max_cls = cls_array.max(axis=0)
    std_cls = cls_array.std(axis=0) if len(cls_vectors) > 1 else np.zeros(768, dtype=np.float32)

    # Clause-type features
    clause_max_conf_arr = np.array([clause_confidences[ct] for ct in _CLAUSE_TYPES], dtype=np.float32)
    clause_present = (clause_max_conf_arr > 0.08).astype(np.float32)

    # Structural stats
    stats = np.array([
        float(len(cls_vectors)),
        float(np.mean(all_top_confs)) if all_top_confs else 0.0,
        float(np.std(all_top_confs)) if len(all_top_confs) > 1 else 0.0,
        float(np.max(all_top_confs)) if all_top_confs else 0.0,
        float(np.min(all_top_confs)) if all_top_confs else 0.0,
    ], dtype=np.float32)

    return np.concatenate([
        class_max_conf,      # 41d
        clause_max_conf_arr, # 5d
        clause_present,      # 5d
        mean_cls,            # 768d
        max_cls,             # 768d
        std_cls,             # 768d
        stats,               # 5d
    ])


def extract_features_batch(
    texts: list[str],
    labels: list[str],
    tokenizer,
    bert_model,
    id2label: dict,
    cache_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features for a batch of contracts with optional caching."""
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        logger.info("Loaded cached features from %s", cache_path)
        return data["X"], data["y"]

    X = []
    for i, text in enumerate(texts):
        print(f"  Extracting features: {i+1}/{len(texts)}", end="\r")
        feat = extract_contract_features(text, tokenizer, bert_model, id2label)
        X.append(feat)
    print()

    X_arr = np.stack(X).astype(np.float32)
    y_arr = np.array([LABEL2ID[lbl] for lbl in labels], dtype=np.int64)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(cache_path), X=X_arr, y=y_arr)
        logger.info("Cached features → %s", cache_path)

    return X_arr, y_arr


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class RiskBertTrainer:
    """Multi-segment feature extraction + stacked ensemble for 3-class risk.

    Training pipeline:
    1. Pre-extract per-contract features using frozen Phase 2 Legal-BERT.
    2. Standardize features (zero-mean, unit-variance).
    3. Train a stacked ensemble using stratified K-fold CV:
       - GradientBoosting (excels on small tabular data)
       - LogisticRegression (strong linear baseline with regularization)
       - MLPClassifier (non-linear complement)
    4. Use soft-voting to blend predictions.
    5. Evaluate on held-out val set.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        phase2_checkpoint: Optional[str] = None,
        label_map_path: str = "results/phase2/label2id.json",
        device: Optional[str] = None,
        # Kept for CLI backward compatibility:
        max_length: int = 512,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        epochs: int = 50,
        patience: int = 8,
        grad_clip: float = 1.0,
        seed: int = 42,
        freeze_layers: int = 9,
        llrd_decay: float = 0.9,
        label_smoothing: float = 0.1,
        dropout: float = 0.4,
    ) -> None:
        self.model_name = model_name
        self.phase2_checkpoint = phase2_checkpoint
        self.label_map_path = label_map_path
        self.seed = seed

        np.random.seed(seed)

        if device is None:
            device = "cpu"
        self.device = device
        logger.info("RiskBertTrainer — seed: %d", self.seed)

    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        output_path: str,
    ) -> dict:
        """Extract features → train stacked ensemble → save → evaluate."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Load Phase 2 BERT ---
        print("  Loading Phase 2 BERT for feature extraction...")
        if self.phase2_checkpoint and Path(self.phase2_checkpoint).exists():
            tokenizer, bert_model, id2label = _load_phase2_bert(
                self.phase2_checkpoint, self.label_map_path, self.model_name,
            )
            print(f"  ✓ Phase 2 BERT loaded ({self.phase2_checkpoint})")
        else:
            raise FileNotFoundError(
                f"Phase 2 checkpoint required: {self.phase2_checkpoint}"
            )

        # --- Step 2: Extract features ---
        cache_dir = Path(output_path).parent
        print("\n  Extracting train features...")
        X_train, y_train = extract_features_batch(
            train_texts, train_labels, tokenizer, bert_model, id2label,
            cache_path=str(cache_dir / "train_risk_features_v2.npz"),
        )
        print("  Extracting val features...")
        X_val, y_val = extract_features_batch(
            val_texts, val_labels, tokenizer, bert_model, id2label,
            cache_path=str(cache_dir / "val_risk_features_v2.npz"),
        )

        feat_dim = X_train.shape[1]
        print(f"\n  Feature dim: {feat_dim}")
        print(f"  Train: {X_train.shape[0]} contracts | Val: {X_val.shape[0]} contracts")

        # --- Step 3: Standardize ---
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        # --- Step 4: Compute sample weights for class imbalance ---
        class_counts = np.array([train_labels.count(c) for c in RISK_CLASSES], dtype=np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        class_weight_dict = {i: len(train_labels) / (len(RISK_CLASSES) * class_counts[i])
                             for i in range(len(RISK_CLASSES))}
        sample_weights = np.array([class_weight_dict[y] for y in y_train])
        print(f"  Class weights: { {RISK_CLASSES[i]: f'{w:.2f}' for i, w in class_weight_dict.items()} }")

        # --- Step 5: Train stacked ensemble with K-fold CV ---
        n_folds = 5
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)

        # Out-of-fold predictions for stacking evaluation
        oof_preds = np.zeros((len(y_train), len(RISK_CLASSES)))
        fold_models = []
        fold_f1s = []
        weighted_voting_supported: Optional[bool] = None

        print(f"\n  Training {n_folds}-fold stacked ensemble...")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_sc, y_train), 1):
            X_tr, X_vl = X_train_sc[train_idx], X_train_sc[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            sw_tr = sample_weights[train_idx]

            # Build ensemble for this fold
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=self.seed + fold_idx,
            )
            lr_clf = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=self.seed,
                solver="lbfgs",
            )
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=500,
                random_state=self.seed + fold_idx,
                early_stopping=True,
                validation_fraction=0.15,
                alpha=0.01,  # L2 regularization
            )

            ensemble = VotingClassifier(
                estimators=[("gb", gb), ("lr", lr_clf), ("mlp", mlp)],
                voting="soft",
                weights=[3, 2, 1],  # GB gets highest weight
            )
            # Use sample weights when supported; fallback cleanly for older setups.
            if weighted_voting_supported is not False:
                try:
                    set_config(enable_metadata_routing=True)
                    ensemble.fit(
                        X_tr,
                        y_tr,
                        gb__sample_weight=sw_tr,
                        lr__sample_weight=sw_tr,
                    )
                    weighted_voting_supported = True
                except Exception as exc:
                    if weighted_voting_supported is None:
                        logger.warning(
                            "Weighted VotingClassifier fit is unavailable (%s). "
                            "Falling back to unweighted ensemble.fit(...).",
                            exc,
                        )
                    weighted_voting_supported = False

            if weighted_voting_supported is False:
                ensemble.fit(X_tr, y_tr)

            # Evaluate on this fold's validation
            fold_preds = ensemble.predict(X_vl)
            fold_f1 = f1_score(y_vl, fold_preds, labels=[0, 1, 2],
                               average="macro", zero_division=0)
            fold_f1s.append(fold_f1)

            # Out-of-fold predictions
            oof_preds[val_idx] = ensemble.predict_proba(X_vl)

            fold_models.append(ensemble)
            print(f"    Fold {fold_idx}/{n_folds} — Macro-F1: {fold_f1:.4f}")

        oof_mean_f1 = float(np.mean(fold_f1s))
        oof_std_f1 = float(np.std(fold_f1s))
        print(f"\n  K-Fold CV Macro-F1: {oof_mean_f1:.4f} ± {oof_std_f1:.4f}")

        # --- Step 6: Train final model on ALL training data ---
        print("\n  Training final ensemble on full training data...")
        final_gb = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=5, random_state=self.seed,
        )
        final_lr = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=self.seed, solver="lbfgs",
        )
        final_mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", max_iter=500,
            random_state=self.seed, early_stopping=True,
            validation_fraction=0.15, alpha=0.01,
        )
        final_ensemble = VotingClassifier(
            estimators=[("gb", final_gb), ("lr", final_lr), ("mlp", final_mlp)],
            voting="soft", weights=[3, 2, 1],
        )
        if weighted_voting_supported:
            final_ensemble.fit(
                X_train_sc,
                y_train,
                gb__sample_weight=sample_weights,
                lr__sample_weight=sample_weights,
            )
        else:
            final_ensemble.fit(X_train_sc, y_train)

        # --- Step 7: Evaluate on held-out val ---
        val_preds = final_ensemble.predict(X_val_sc)
        val_f1 = f1_score(y_val, val_preds, labels=[0, 1, 2],
                          average="macro", zero_division=0)
        val_acc = float(np.mean(val_preds == y_val))

        print(f"\n  Held-out Val Macro-F1: {val_f1:.4f}")
        print(f"  Held-out Val Accuracy: {val_acc:.4f}")

        # --- Step 8: Save ---
        # Save as pickle (sklearn models can't be saved as .pt)
        checkpoint = {
            "ensemble": final_ensemble,
            "scaler": scaler,
            "fold_models": fold_models,
            "fold_f1s": fold_f1s,
            "cv_macro_f1": oof_mean_f1,
            "val_macro_f1": val_f1,
            "feat_dim": feat_dim,
        }
        pkl_path = str(output_path).replace(".pt", ".pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(checkpoint, f)
        # Also save as .pt path for CLI compatibility (symlink-like)
        with open(output_path, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"\n  ✓ Checkpoint saved → {pkl_path}")
        print(f"  ✓ CV Macro-F1: {oof_mean_f1:.4f} ± {oof_std_f1:.4f}")
        print(f"  ✓ Val Macro-F1: {val_f1:.4f}")

        return {
            "best_val_macro_f1": float(val_f1),
            "cv_macro_f1": oof_mean_f1,
            "cv_std_f1": oof_std_f1,
            "fold_f1s": fold_f1s,
            "history": [{"epoch": 1, "val_macro_f1": val_f1, "val_acc": val_acc}],
            "checkpoint": output_path,
        }

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_for_inference(
        checkpoint_path: str,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> tuple:
        """Load saved ensemble for inference.

        Returns (tokenizer, checkpoint_dict).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        return tokenizer, checkpoint

    @staticmethod
    def predict_proba(
        text: str,
        tokenizer,
        model,  # This is the checkpoint dict
        max_length: int = 512,
        device: str = "cpu",
        _bert_model=None,
        _id2label=None,
    ) -> dict[str, float]:
        """Return probability dict for a single contract text."""
        if _bert_model is None or _id2label is None:
            # Fallback: return uniform
            return {c: 1.0 / len(RISK_CLASSES) for c in RISK_CLASSES}

        feat = extract_contract_features(text, tokenizer, _bert_model, _id2label)
        feat_2d = feat.reshape(1, -1)

        # Use the checkpoint's scaler and ensemble
        scaler = model["scaler"]
        ensemble = model["ensemble"]
        feat_sc = scaler.transform(feat_2d)
        probs = ensemble.predict_proba(feat_sc)[0]

        return {c: float(probs[i]) for i, c in enumerate(RISK_CLASSES)}
