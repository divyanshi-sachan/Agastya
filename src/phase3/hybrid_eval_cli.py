"""CLI wrapper to generate Hybrid eval JSON artifact."""

from __future__ import annotations

import argparse

from src.phase3.hybrid_eval import generate_hybrid_eval_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 3 Hybrid evaluation artifact.")
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--out", default="reports/phase3/hybrid_eval.json")
    parser.add_argument("--bn-model", default="results/phase3/bayesian_network.pkl")
    parser.add_argument("--bert-checkpoint", default="results/phase2/models/legal_bert_phase2.pt")
    parser.add_argument("--label-map", default="results/phase2/label2id.json")
    parser.add_argument("--adapter-path", default="results/phase2/models/legal_bert_lora_adapter")
    args = parser.parse_args()

    payload = generate_hybrid_eval_artifact(
        test_csv_path=args.test_csv,
        output_json_path=args.out,
        bn_model_path=args.bn_model,
        bert_checkpoint_path=args.bert_checkpoint,
        label_map_path=args.label_map,
        adapter_path=args.adapter_path,
    )
    print(f"Saved: {args.out}")
    print(payload)


if __name__ == "__main__":
    main()

