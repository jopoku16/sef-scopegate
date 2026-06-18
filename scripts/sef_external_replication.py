"""Independent external replication suite for Signed Evidence Flow.

This script reuses the ScopeGate question on public OpenML data sets that were
not used in the main development narrative. The aim is not to cherry-pick a
single friendly data set. It asks, across several domains, whether SEF conflict
adds error-ranking information after confidence and attribution entropy are
already known, and whether the positive-direction review rule is supported.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sef_scope_diagnostic import analyze_dataset, paired_sign_flip_p


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("Adult income", "adult", 2, "Socioeconomic"),
    ("Spambase", "spambase", 1, "Communication"),
    ("Phoneme", "phoneme", 1, "Speech"),
    ("Mammography", "mammography", 1, "Healthcare"),
    ("Credit approval", "credit-approval", 1, "Credit"),
    ("QSAR biodeg.", "qsar-biodeg", 1, "Chemistry"),
    ("PC1 defects", "pc1", 1, "Software"),
    ("KC1 defects", "kc1", 1, "Software"),
    ("Electricity", "electricity", 1, "Market"),
    ("Magic telescope", "MagicTelescope", 1, "Physics"),
]

DOMAIN_COLORS = {
    "Healthcare": "#4c78a8",
    "Credit": "#e45756",
    "Socioeconomic": "#72b7b2",
    "Communication": "#f58518",
    "Speech": "#54a24b",
    "Chemistry": "#b279a2",
    "Software": "#9d755d",
    "Market": "#ff9da6",
    "Physics": "#bab0ac",
}


def summarize_external(runs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        runs.groupby(["dataset", "domain", "n", "p_raw"], as_index=False)
        .agg(
            high_confidence_errors=("high_confidence_errors", "mean"),
            base_conditional_auc=("base_conditional_auc", "mean"),
            augmented_conditional_auc=("augmented_conditional_auc", "mean"),
            delta_conditional_auc=("delta_conditional_auc", "mean"),
            conflict_error_gap=("conflict_error_gap", "mean"),
            diagnostic_eligible_rate=("diagnostic_eligible", "mean"),
            diagnostic_pass_rate=("diagnostic_pass", "mean"),
            median_monotone_p=("monotone_p_value", "median"),
            conflict_error_auc=("conflict_error_auc", "mean"),
            low_confidence_error_auc=("low_confidence_error_auc", "mean"),
            rows_per_second=("rows_per_second", "mean"),
        )
    )
    p_values = (
        runs.groupby("dataset")["delta_conditional_auc"]
        .apply(paired_sign_flip_p)
        .rename("delta_sign_flip_p")
        .reset_index()
    )
    summary = summary.merge(p_values, on="dataset", how="left")
    order = {label: i for i, (label, _, _, _) in enumerate(DATASETS)}
    return summary.sort_values("dataset", key=lambda s: s.map(order)).reset_index(drop=True)


def write_table(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & $n$ & Base AUC & +SEF AUC & $\\Delta$AUC & Gap & Eligible/pass \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        pass_text = "--" if pd.isna(row["diagnostic_pass_rate"]) else f"{row['diagnostic_pass_rate']:.2f}"
        lines.append(
            f"{row['dataset']} & {int(row['n'])} & {row['base_conditional_auc']:.3f} & "
            f"{row['augmented_conditional_auc']:.3f} & {row['delta_conditional_auc']:+.3f} & "
            f"{row['conflict_error_gap']:+.3f} & "
            f"{row['diagnostic_eligible_rate']:.2f}/{pass_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_external_replication_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    colors = [DOMAIN_COLORS.get(d, "#777777") for d in summary["domain"]]
    x = np.arange(len(summary))

    axes[0].bar(x, summary["delta_conditional_auc"], color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("AUC gain from SEF conflict")
    axes[0].set_title("Increment beyond confidence and entropy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["dataset"], rotation=35, ha="right")

    axes[1].bar(x, summary["conflict_error_gap"], color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("High-conflict minus low-conflict error")
    axes[1].set_title("Direction of the conflict-risk relation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["dataset"], rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure19_external_replication.png", dpi=220)


def main() -> None:
    seeds = range(10)
    runs = pd.concat(
        [analyze_dataset(label, name, version, domain, seeds) for label, name, version, domain in DATASETS],
        ignore_index=True,
    )
    summary = summarize_external(runs)
    runs.to_csv(RES_DIR / "sef_external_replication_runs.csv", index=False)
    summary.to_csv(RES_DIR / "sef_external_replication_summary.csv", index=False)
    write_table(summary)
    make_figure(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
