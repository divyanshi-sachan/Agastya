"""CLI to generate strict Phase 3 ablation CSV."""

from __future__ import annotations

import argparse

from src.phase3.ablation import write_ablation_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Phase 3 ablation table from artifacts.")
    parser.add_argument(
        "--out",
        default="reports/phase3/ablation_results.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()
    table = write_ablation_table(output_path=args.out)
    print(f"Saved: {args.out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

