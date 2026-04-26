"""CLI: Train Tier 3 risk classifier and run ensemble evaluation.

Supports two evaluation modes:
  - 2-way ensemble: BN posterior + frozen-BERT LR probe  (default)
  - 3-way ensemble: BN + frozen-BERT LR probe + fine-tuned BERT
    (activated when --risk-bert-checkpoint is provided)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.phase3.bayesian.bootstrap import ensure_seed_model
from src.phase3.hybrid_pipeline import AgastyaHybridPipeline
from src.phase3.risk_classifier import (
    _RISK_CLASSES,
    _load_bert,
    ensemble_predict,
    ensemble_predict_with_finetuned,
    load_risk_classifier,
    train_risk_classifier,
)
from src.phase3.risk_bert import RiskBertTrainer as _RiskBertTrainer
from src.phase3.hybrid_eval import build_contract_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier 3: Train risk classifier + ensemble eval.")
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--val-csv", default="data/processed/val.csv")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--bn-model", default="results/phase3/bayesian_network.pkl")
    parser.add_argument("--bert-checkpoint", default="results/phase2/models/legal_bert_phase2.pt")
    parser.add_argument("--label-map", default="results/phase2/label2id.json")
    parser.add_argument("--clf-out", default="results/phase3/risk_classifier.pkl")
    parser.add_argument("--eval-out", default="reports/phase3/hybrid_eval.json")
    parser.add_argument("--bn-weight", type=float, default=0.45)
    parser.add_argument("--clf-weight", type=float, default=0.55)
    parser.add_argument("--skip-train", action="store_true", help="Skip training; load existing clf")
    # 3-way ensemble (fine-tuned BERT)
    parser.add_argument(
        "--risk-bert-checkpoint",
        default=None,
        help="Path to fine-tuned RiskBertHead checkpoint (.pt). "
             "When provided, runs 3-way ensemble (BN + LR probe + fine-tuned BERT).",
    )
    parser.add_argument("--risk-bert-bn-weight",  type=float, default=0.35,
                        help="BN weight for 3-way ensemble (default 0.35).")
    parser.add_argument("--risk-bert-clf-weight", type=float, default=0.30,
                        help="LR probe weight for 3-way ensemble (default 0.30).")
    parser.add_argument("--risk-bert-ft-weight",  type=float, default=0.35,
                        help="Fine-tuned BERT weight for 3-way ensemble (default 0.35).")
    parser.add_argument("--model-name", default="nlpaueb/legal-bert-base-uncased",
                        help="HuggingFace model ID (must match fine-tuned checkpoint).")
    args = parser.parse_args()

    # --- Train or load classifier ---
    if args.skip_train and Path(args.clf_out).exists():
        print("Loading existing risk classifier...")
        clf = load_risk_classifier(args.clf_out)
    else:
        print("Training Tier 3 risk classifier...")
        result = train_risk_classifier(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            checkpoint_path=args.bert_checkpoint,
            label_map_path=args.label_map,
            output_model_path=args.clf_out,
        )
        clf = load_risk_classifier(args.clf_out)
        print(f"Classifier trained. Val Macro-F1: {result['val_macro_f1']:.4f}")

    # --- Load BN pipeline + BERT for ensemble ---
    print("\nLoading BN hybrid pipeline...")
    model_path = ensure_seed_model(args.bn_model)
    pipeline = AgastyaHybridPipeline(
        bn_model_path=model_path,
        bert_checkpoint_path=args.bert_checkpoint,
        label_map_path=args.label_map,
    )
    print("Loading BERT for embedding extraction...")
    tokenizer, bert_model = _load_bert(args.bert_checkpoint, args.label_map)

    # --- Run ensemble evaluation on test set ---
    print("\nRunning ensemble evaluation on test set...")
    test_df = pd.read_csv(args.test_csv)
    contract_df = build_contract_dataset(test_df)

    y_true, y_pred = [], []
    for i, row in contract_df.iterrows():
        print(f"  Contract {i+1}/{len(contract_df)}", end="\r")
        result = ensemble_predict(
            contract_text=row["text"],
            hybrid_pipeline=pipeline,
            risk_classifier=clf,
            tokenizer=tokenizer,
            bert_model=bert_model,
            bn_weight=args.bn_weight,
            clf_weight=args.clf_weight,
        )
        y_true.append(row["true_risk"])
        y_pred.append(result["risk_level"])

    print()
    labels = _RISK_CLASSES
    payload = {
        "task": "contract_risk_level_3way_ensemble",
        "n_contracts": int(len(contract_df)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "blend_weights": {"bn": args.bn_weight, "clf": args.clf_weight},
        "ground_truth": "Derived from expanded Phase 2 test labels (26-label type-scoring).",
    }

    out = Path(args.eval_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== ENSEMBLE EVALUATION RESULTS ===")
    print(json.dumps(payload, indent=2))
    print(f"\nSaved → {out}")

    # --- 3-way ensemble (optional, only when fine-tuned checkpoint is provided) ---
    if args.risk_bert_checkpoint and Path(args.risk_bert_checkpoint).exists():
        print("\n=== 3-WAY ENSEMBLE EVALUATION (BN + LR probe + fine-tuned BERT) ===")
        print(f"  Loading fine-tuned checkpoint: {args.risk_bert_checkpoint}")
        ft_tok, ft_model = _RiskBertTrainer.load_for_inference(
            args.risk_bert_checkpoint, model_name=args.model_name
        )

        y_true_3way, y_pred_3way = [], []
        for i, row in contract_df.iterrows():
            print(f"  Contract {i+1}/{len(contract_df)}", end="\r")
            result_3way = ensemble_predict_with_finetuned(
                contract_text=row["text"],
                hybrid_pipeline=pipeline,
                risk_classifier=clf,
                tokenizer=tokenizer,
                bert_model=bert_model,
                finetuned_tokenizer=ft_tok,
                finetuned_model=ft_model,
                bn_weight=args.risk_bert_bn_weight,
                clf_weight=args.risk_bert_clf_weight,
                finetuned_weight=args.risk_bert_ft_weight,
            )
            y_true_3way.append(row["true_risk"])
            y_pred_3way.append(result_3way["risk_level"])

        print()
        labels = _RISK_CLASSES
        payload_3way = {
            "task": "contract_risk_level_3way_ensemble_with_finetuned",
            "n_contracts": int(len(contract_df)),
            "macro_f1": float(f1_score(y_true_3way, y_pred_3way, labels=labels,
                                       average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true_3way, y_pred_3way)),
            "precision": float(precision_score(y_true_3way, y_pred_3way, labels=labels,
                                               average="macro", zero_division=0)),
            "recall": float(recall_score(y_true_3way, y_pred_3way, labels=labels,
                                         average="macro", zero_division=0)),
            "blend_weights": {
                "bn": args.risk_bert_bn_weight,
                "clf": args.risk_bert_clf_weight,
                "finetuned": args.risk_bert_ft_weight,
            },
            "finetuned_checkpoint": args.risk_bert_checkpoint,
        }

        out_3way = Path(args.eval_out).with_stem(
            Path(args.eval_out).stem + "_3way_finetuned"
        )
        out_3way.parent.mkdir(parents=True, exist_ok=True)
        out_3way.write_text(json.dumps(payload_3way, indent=2), encoding="utf-8")

        print("\n=== 3-WAY ENSEMBLE RESULTS ===")
        print(json.dumps(payload_3way, indent=2))
        print(f"\nSaved → {out_3way}")

        delta_f1 = payload_3way["macro_f1"] - payload["macro_f1"]
        print(f"\n  ΔMacro-F1 (3-way vs 2-way): {delta_f1:+.4f}")
    elif args.risk_bert_checkpoint:
        print(
            f"\n[WARNING] --risk-bert-checkpoint path not found: {args.risk_bert_checkpoint}\n"
            "  Run `python -m src.phase3.risk_bert_train_cli` first to produce the checkpoint."
        )


if __name__ == "__main__":
    main()
