"""CLI: Fine-tune Legal-BERT multi-segment feature extractor + MLP on 3-class risk.

Usage
-----
python -m src.phase3.risk_bert_train_cli \\
    --train-csv  data/processed/train.csv \\
    --val-csv    data/processed/val.csv \\
    --bert-checkpoint results/phase2/models/legal_bert_phase2.pt \\
    --label-map  results/phase2/label2id.json \\
    --out        results/phase3/models/risk_bert_finetuned.pt \\
    --epochs     50 \\
    --lr         1e-3 \\
    --batch      16

The script:
1. Loads train/val CSVs and derives 3-class risk labels (same heuristic as hybrid_eval.py).
2. Extracts per-contract features using ALL clause segments (not truncated to 512 tokens).
3. Trains a lightweight MLP classifier on the features.
4. Saves the MLP checkpoint + runs post-training evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from src.phase3.hybrid_eval import build_contract_dataset
from src.phase3.risk_bert import RISK_CLASSES, RiskBertTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_split(csv_path: str) -> tuple[list[str], list[str]]:
    """Load a CSV split and return (texts, labels) for the 3-class risk task."""
    df = pd.read_csv(csv_path)
    contract_df = build_contract_dataset(df)
    texts  = contract_df["text"].tolist()
    labels = contract_df["true_risk"].tolist()
    logger.info(
        "Loaded %d contracts from %s | class dist: %s",
        len(texts),
        csv_path,
        {c: labels.count(c) for c in RISK_CLASSES},
    )
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Legal-BERT multi-segment feature extractor + MLP for 3-class contract risk."
    )

    # Data
    parser.add_argument("--train-csv",  default="data/processed/train.csv",
                        help="Path to training CSV (must have filename, text, label columns).")
    parser.add_argument("--val-csv",    default="data/processed/val.csv",
                        help="Path to validation CSV.")

    # Model
    parser.add_argument("--model-name", default="nlpaueb/legal-bert-base-uncased",
                        help="HuggingFace model ID for backbone + tokeniser.")
    parser.add_argument("--bert-checkpoint", default=None,
                        help="Phase 2 checkpoint (.pt) — REQUIRED for feature extraction.")
    parser.add_argument("--label-map", default="results/phase2/label2id.json",
                        help="Path to Phase 2 label2id.json.")

    # Output
    parser.add_argument("--out", default="results/phase3/models/risk_bert_finetuned.pt",
                        help="Where to save the fine-tuned MLP checkpoint.")
    parser.add_argument("--history-out", default="reports/phase3/risk_bert_history.json",
                        help="Where to save training history JSON.")

    # Hyperparameters
    parser.add_argument("--epochs",       type=int,   default=50,   help="Max training epochs.")
    parser.add_argument("--lr",           type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--batch",        type=int,   default=16,   help="Training batch size.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--patience",     type=int,   default=8,    help="Early-stop patience (epochs).")
    parser.add_argument("--seed",         type=int,   default=42,   help="Random seed.")
    parser.add_argument("--device",       default=None,
                        help="Force device: cpu | mps | cuda. Auto-detected if omitted.")

    # Regularisation
    parser.add_argument("--dropout",         type=float, default=0.4, help="Dropout rate in MLP (default 0.4).")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon (default 0.1).")

    # Kept for backward CLI compatibility (unused by new arch)
    parser.add_argument("--max-len",      type=int,   default=512,  help="(unused) Kept for compat.")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,  help="(unused) Kept for compat.")
    parser.add_argument("--grad-clip",    type=float, default=1.0,  help="Gradient clip max_norm.")
    parser.add_argument("--freeze-layers",   type=int,   default=9,   help="(unused) Kept for compat.")
    parser.add_argument("--llrd-decay",      type=float, default=0.9, help="(unused) Kept for compat.")

    # Extras
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip post-training val evaluation.")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" Agastya — Legal-BERT Multi-Segment Risk Classifier")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────
    print("\n[1/3] Loading data...")
    train_texts, train_labels = _load_split(args.train_csv)
    val_texts,   val_labels   = _load_split(args.val_csv)

    # ── Train ──────────────────────────────────────────────────────
    print("\n[2/3] Training multi-segment feature extractor + MLP...")
    trainer = RiskBertTrainer(
        model_name=args.model_name,
        phase2_checkpoint=args.bert_checkpoint,
        label_map_path=args.label_map,
        device=args.device,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        grad_clip=args.grad_clip,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
    )
    result = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_path=args.out,
    )

    # ── Save history ───────────────────────────────────────────────
    history_out = Path(args.history_out)
    history_out.parent.mkdir(parents=True, exist_ok=True)
    history_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Training history saved → {history_out}")

    # ── Post-training evaluation ───────────────────────────────────
    if not args.skip_eval:
        print("\n[3/3] Post-training val evaluation...")
        from src.phase3.risk_bert import (
            RiskBertTrainer as _T,
            _load_phase2_bert,
            extract_contract_features,
        )

        tok, mlp_model = _T.load_for_inference(args.out, model_name=args.model_name)
        _, bert_model, id2label = _load_phase2_bert(
            args.bert_checkpoint, args.label_map, args.model_name,
        )

        preds = []
        for i, text in enumerate(val_texts):
            print(f"  Evaluating: {i+1}/{len(val_texts)}", end="\r")
            proba = _T.predict_proba(
                text, tok, mlp_model,
                _bert_model=bert_model, _id2label=id2label,
            )
            pred = max(proba, key=proba.get)
            preds.append(pred)
        print()

        macro_f1 = f1_score(val_labels, preds, labels=RISK_CLASSES,
                            average="macro", zero_division=0)
        print(f"\n  Val Macro-F1 (multi-segment MLP): {macro_f1:.4f}")
        print(classification_report(val_labels, preds, labels=RISK_CLASSES, zero_division=0))
    else:
        print("\n[3/3] Skipping val evaluation (--skip-eval).")

    print("\n" + "=" * 60)
    print(f"  Best val Macro-F1: {result['best_val_macro_f1']:.4f}")
    print(f"  Checkpoint:        {args.out}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
