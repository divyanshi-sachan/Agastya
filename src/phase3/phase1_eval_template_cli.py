"""Create a Phase 1 evaluation JSON template for strict ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_template() -> dict:
    return {
        "task": "phase1_clause_classification",
        "macro_f1": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "notes": "Fill these values from your validated Phase 1 evaluation run.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Phase 1 results JSON template.")
    parser.add_argument(
        "--out",
        default="results/phase1/results.json",
        help="Output path for Phase 1 results template JSON.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Exists already, not overwriting: {out}")
        return
    out.write_text(json.dumps(build_template(), indent=2), encoding="utf-8")
    print(f"Created template: {out}")


if __name__ == "__main__":
    main()

