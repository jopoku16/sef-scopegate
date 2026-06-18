"""Empirical validation for the SEF conformal reliability screen."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sef import evidence_scores


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def benchmark_specs() -> List[Tuple[str, np.ndarray, np.ndarray]]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()
    iris_mask = iris.target != 0
    wine_mask = wine.target != 0
    digits_mask = np.isin(digits.target, [3, 5])
    return [
        ("Iris", iris.data[iris_mask], (iris.target[iris_mask] == 2).astype(int)),
        ("Wine", wine.data[wine_mask], (wine.target[wine_mask] == 2).astype(int)),
        ("Breast cancer", cancer.data, cancer.target.astype(int)),
        ("Digits", digits.data[digits_mask], (digits.target[digits_mask] == 5).astype(int)),
    ]


def signed_evidence(model: LogisticRegression, x_scaled: np.ndarray, baseline_scaled: np.ndarray) -> np.ndarray:
    return (x_scaled - baseline_scaled) * model.coef_[0]


def split_run(label: str, x: np.ndarray, y: np.ndarray, seed: int, alpha: float = 0.10) -> Dict[str, float | str]:
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    x_train, x_cal, y_train, y_cal = train_test_split(
        x_temp, y_temp, test_size=0.35, random_state=1000 + seed, stratify=y_temp
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_cal_s = scaler.transform(x_cal)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=2500, class_weight="balanced")
    model.fit(x_train_s, y_train)
    baseline = x_train_s.mean(axis=0)

    ev_cal = signed_evidence(model, x_cal_s, baseline)
    ev_test = signed_evidence(model, x_test_s, baseline)
    res_cal = evidence_scores(ev_cal).reliable_evidence
    res_test = evidence_scores(ev_test).reliable_evidence
    unreliability_cal = 1.0 - res_cal
    unreliability_test = 1.0 - res_test

    n_cal = len(unreliability_cal)
    q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    threshold = float(np.quantile(unreliability_cal, q_level, method="higher"))
    accepted = unreliability_test <= threshold

    pred = model.predict(x_test_s)
    error = (pred != y_test).astype(float)
    return {
        "dataset": label,
        "seed": float(seed),
        "alpha": alpha,
        "nominal_accepted_fraction": 1.0 - alpha,
        "accepted_fraction": float(accepted.mean()),
        "flagged_fraction": float((~accepted).mean()),
        "all_error": float(error.mean()),
        "accepted_error": float(error[accepted].mean()) if accepted.any() else float("nan"),
        "flagged_error": float(error[~accepted].mean()) if (~accepted).any() else float("nan"),
        "threshold": threshold,
    }


def main() -> None:
    rows: List[Dict[str, float | str]] = []
    for label, x, y in benchmark_specs():
        for seed in range(50):
            rows.append(split_run(label, x, y, seed))

    runs = pd.DataFrame(rows)
    runs.to_csv(RES_DIR / "sef_conformal_screen_runs.csv", index=False)
    summary = (
        runs.groupby("dataset", as_index=False)
        .agg(
            accepted_fraction=("accepted_fraction", "mean"),
            accepted_fraction_sd=("accepted_fraction", "std"),
            all_error=("all_error", "mean"),
            accepted_error=("accepted_error", "mean"),
            flagged_error=("flagged_error", "mean"),
            flagged_error_sd=("flagged_error", "std"),
        )
    )
    summary.to_csv(RES_DIR / "sef_conformal_screen_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Dataset & Accepted fraction & All error & Accepted error & Flagged error \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"{row['dataset']} & {row['accepted_fraction']:.3f} & "
            f"{row['all_error']:.3f} & {row['accepted_error']:.3f} & "
            f"{row['flagged_error']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_conformal_screen_table.tex").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x_pos = np.arange(len(summary))
    width = 0.28
    ax.bar(x_pos - width, summary["all_error"], width, label="All cases", color="#9aa0a6")
    ax.bar(x_pos, summary["accepted_error"], width, label="Accepted", color="#54a24b")
    ax.bar(x_pos + width, summary["flagged_error"], width, label="Flagged", color="#e45756")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(summary["dataset"])
    ax.set_ylabel("Error rate")
    ax.set_title("Conformal SEF screen: flagged cases are riskier")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure15_conformal_screen_validation.png", dpi=220)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
