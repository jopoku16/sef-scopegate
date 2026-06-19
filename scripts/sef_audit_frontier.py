"""Create independent ScopeGate triage-frontier outputs for Signed Evidence Flow."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def external_frontiers() -> List[Dict[str, float | str]]:
    runs = pd.read_csv(RES_DIR / "sef_external_replication_runs.csv")
    mapping = {
        0.00: "conflict_triage_error_00",
        0.10: "conflict_triage_error_10",
        0.20: "conflict_triage_error_20",
        0.30: "conflict_triage_error_30",
    }
    rows: List[Dict[str, float | str]] = []
    for dataset, block in runs.groupby("dataset", sort=False):
        for rate, column in mapping.items():
            rows.append(
                {
                    "source": dataset,
                    "review_rate": rate,
                    "accepted_error": float(block[column].mean()),
                }
            )
    return rows


def write_summary(rows: List[Dict[str, float | str]]) -> None:
    path = RES_DIR / "sef_audit_frontier_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "review_rate", "accepted_error"])
        writer.writeheader()
        writer.writerows(rows)


def make_frontier_figure(rows: List[Dict[str, float | str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    supported = {"Electricity", "Magic telescope"}
    reversals = {"Adult income", "Mammography", "PC1 defects", "KC1 defects"}
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    for dataset, block in frame.groupby("source", sort=False):
        if dataset in supported:
            color, linewidth, alpha = "#2a9d8f", 3.0, 1.0
        elif dataset in reversals:
            color, linewidth, alpha = "#e76f51", 2.0, 0.9
        else:
            color, linewidth, alpha = "#6c757d", 1.3, 0.65
        ax.plot(
            block["review_rate"], block["accepted_error"], marker="o",
            color=color, linewidth=linewidth, alpha=alpha, label=dataset,
        )
    ax.set_xlabel("Fraction of high-confidence cases sent to review")
    ax.set_ylabel("Error among accepted high-confidence cases")
    ax.set_title("Independent ScopeGate triage frontiers")
    ax.legend(frameon=False, ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure10_evidence_reliability_frontier.png", dpi=220)


def main() -> None:
    rows = external_frontiers()
    write_summary(rows)
    make_frontier_figure(rows)


if __name__ == "__main__":
    main()
