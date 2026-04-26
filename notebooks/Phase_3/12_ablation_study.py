from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Ensure project root is importable when notebook runs from notebooks/Phase_3.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "Phase_3":
    PROJECT_ROOT = PROJECT_ROOT.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3.ablation import build_ablation_table, write_ablation_table

print(f"Project root: {PROJECT_ROOT}")

# Build ablation table using explicit PROJECT_ROOT-anchored paths
ablation_df = build_ablation_table(
    phase2_results_path=str(PROJECT_ROOT / "results" / "phase2" / "results.json"),
    hybrid_results_path=str(PROJECT_ROOT / "reports" / "phase3" / "hybrid_eval.json"),
)

# Save updated CSV
ablation_csv = PROJECT_ROOT / "reports" / "phase3" / "ablation_results.csv"
ablation_df.to_csv(ablation_csv, index=False)

ablation_df

# === Ablation Bar Chart ===
metric_cols = ["Macro-F1", "Accuracy", "Precision", "Recall"]
plot_ready = ablation_df.copy()
for col in metric_cols:
    plot_ready[col] = pd.to_numeric(plot_ready[col], errors="coerce")

melted = plot_ready.melt(
    id_vars=["Configuration", "Status"],
    value_vars=metric_cols,
    var_name="Metric",
    value_name="Score",
).dropna(subset=["Score"])

plt.figure(figsize=(10, 4))
if len(melted) > 0:
    palette = {"DL_Only": "#F4A261", "Hybrid": "#2A9D8F"}
    sns.barplot(data=melted, x="Metric", y="Score", hue="Configuration", palette=palette)
    plt.ylim(0, 1)
    plt.title("Ablation Study — DL vs Hybrid", fontsize=14, fontweight="bold")
    plt.ylabel("Score")
    plt.legend(title="Configuration")
else:
    plt.text(0.5, 0.5, "No computed ablation metrics available yet", ha="center", va="center")
    plt.title("Ablation Comparison")
    plt.axis("off")
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "reports" / "phase3" / "figures" / "ablation_bar_chart.png", dpi=150, bbox_inches="tight")
plt.show()

# === Status Summary ===
status_view = ablation_df[["Configuration", "Macro-F1", "Accuracy", "Status", "Notes"]]
status_view

# === Load and print Phase 2 + Hybrid result files for reference ===
p2 = PROJECT_ROOT / "results" / "phase2" / "results.json"
p3 = PROJECT_ROOT / "reports" / "phase3" / "hybrid_eval.json"

print("=== Phase 2 Results ===")
if p2.exists():
    print(json.dumps(json.loads(p2.read_text()), indent=2))

print("\n=== Phase 3 Hybrid Results ===")
if p3.exists():
    print(json.dumps(json.loads(p3.read_text()), indent=2))

print(f"\nAblation table saved to: {ablation_csv}")
