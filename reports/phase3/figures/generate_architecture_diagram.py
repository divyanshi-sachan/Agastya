"""Generate Phase 3 architecture diagram with tensor-shape labels."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_zone(ax, xy, width, height, color, title):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.5,
        edgecolor="black",
        facecolor=color,
        alpha=0.25,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height - 0.04,
        title,
        ha="center",
        va="top",
        fontsize=10,
        weight="bold",
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_zone(ax, (0.03, 0.60), 0.22, 0.34, "#b39ddb", "DL Zone (Legal-BERT)")
    add_zone(ax, (0.29, 0.60), 0.30, 0.34, "#fff59d", "Interface Layer")
    add_zone(ax, (0.63, 0.60), 0.20, 0.34, "#ffcc80", "Bayesian Network")
    add_zone(ax, (0.85, 0.60), 0.12, 0.34, "#a5d6a7", "Output")

    boxes = {
        "input": (0.05, 0.78, "Clauses\\n(text list)"),
        "bert": (0.15, 0.78, "Legal-BERT\\nembeddings (N,768)\\nlogits (N,41)"),
        "encoder": (0.34, 0.78, "Evidence Encoding\\n(5,) hard evidence"),
        "virtual": (0.47, 0.78, "Virtual Evidence\\nnode -> [P0,P1]"),
        "features": (0.57, 0.78, "Cross-Clause\\nscalar conflict"),
        "bn": (0.69, 0.78, "Belief Propagation\\nBN inference"),
        "risk": (0.90, 0.78, "Risk Report\\ndict / JSON"),
    }

    for x, y, label in boxes.values():
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"),
        )

    arrows = [
        ("input", "bert", "(N,text)"),
        ("bert", "encoder", "(N,41) -> (5,)"),
        ("bert", "virtual", "(N,41) -> dict"),
        ("bert", "features", "(N,768) -> scalar"),
        ("encoder", "bn", "hard evidence"),
        ("virtual", "bn", "virtual evidence"),
        ("features", "bn", "conflict signal"),
        ("bn", "risk", "P(L/M/H)"),
    ]
    for src, dst, label in arrows:
        x1, y1, _ = boxes[src]
        x2, y2, _ = boxes[dst]
        ax.annotate(
            "",
            xy=(x2 - 0.03, y2),
            xytext=(x1 + 0.03, y1),
            arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
        )
        ax.text((x1 + x2) / 2, y1 + 0.03, label, fontsize=8, ha="center")

    ax.set_title("Agastya Phase 3 Neuro-Symbolic Architecture", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig("reports/phase3/figures/architecture_diagram.png", dpi=300)
    fig.savefig("reports/phase3/figures/architecture_diagram.svg")


if __name__ == "__main__":
    main()

