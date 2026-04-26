"""CLI entrypoint for training the Phase 3 BN with EM."""

from __future__ import annotations

import argparse

from src.phase3.bayesian.em_trainer import train_bn_from_phase2_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 Bayesian network (EM).")
    parser.add_argument(
        "--train-csv",
        default="data/processed/train.csv",
        help="Path to processed Phase 2 train.csv",
    )
    parser.add_argument(
        "--out-model",
        default="results/phase3/bayesian_network.pkl",
        help="Output pickle path for trained BN",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=30,
        help="EM iterations",
    )
    args = parser.parse_args()

    model, bn_train_df = train_bn_from_phase2_processed(
        train_csv_path=args.train_csv,
        output_model_path=args.out_model,
        n_iter=args.n_iter,
    )
    print(f"Trained rows: {len(bn_train_df)}")
    print(f"Output model: {args.out_model}")
    print(f"Model valid: {model.check_model()}")


if __name__ == "__main__":
    main()

