"""Compare SEF conflict with attribution entropy and Gini-style spread.

This addresses a natural reviewer question:

    Is SEF conflict just attribution entropy under another name?

The script uses the same public healthcare data sets as the real-data benchmark.
For each train-test split, it restricts attention to predictions that already
look confident, then compares how well different attribution summaries identify
riskier cases.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sef import evidence_scores


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


DATASETS = [
    ("Diabetes", "diabetes", 1),
    ("Heart disease", "heart-statlog", 1),
    ("Blood transfusion", "blood-transfusion-service-center", 1),
]


def load_openml_binary(name: str, version: int) -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_openml(name=name, version=version, as_frame=True)
    frame = data.frame.copy()
    target = data.target.name if getattr(data, "target", None) is not None else frame.columns[-1]
    if target not in frame.columns:
        target = frame.columns[-1]
    y_raw = frame.pop(target)
    y_cat = pd.Categorical(y_raw)
    if len(y_cat.categories) != 2:
        raise ValueError(f"{name} is not binary")
    return frame.reset_index(drop=True), y_cat.codes.astype(int)


def make_pipeline(x: pd.DataFrame) -> Pipeline:
    categorical = [col for col in x.columns if str(x[col].dtype) in {"category", "object"}]
    numeric = [col for col in x.columns if col not in categorical]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="lbfgs")
    return Pipeline([("pre", pre), ("model", model)])


def evidence_from_pipeline(pipe: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame) -> np.ndarray:
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    z_train = pre.transform(x_train)
    z_test = pre.transform(x_test)
    baseline = z_train.mean(axis=0)
    beta = model.coef_[0]
    return (z_test - baseline) * beta


def attribution_spread(evidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_ev = np.abs(evidence)
    denom = abs_ev.sum(axis=1, keepdims=True) + 1e-12
    p = abs_ev / denom
    k = evidence.shape[1]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(max(k, 2))
    gini_spread = 1.0 - (p**2).sum(axis=1)
    return entropy, gini_spread


def top_quartile_error(error: np.ndarray, score: np.ndarray) -> float:
    cut = np.quantile(score, 0.75)
    return float(error[score >= cut].mean())


def score_auc(error: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(error)) < 2:
        return float("nan")
    return float(roc_auc_score(error.astype(int), score))


def analyze_dataset(label: str, openml_name: str, version: int, seeds: range) -> pd.DataFrame:
    x, y = load_openml_binary(openml_name, version)
    rows = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.35, random_state=seed, stratify=y
        )
        pipe = make_pipeline(x_train)
        pipe.fit(x_train, y_train)

        prob = pipe.predict_proba(x_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        error = (pred != y_test).astype(float)
        confidence = np.maximum(prob, 1.0 - prob)
        high_conf = confidence >= np.quantile(confidence, 0.50)

        evidence = evidence_from_pipeline(pipe, x_train, x_test)
        sef = evidence_scores(evidence)
        entropy, gini_spread = attribution_spread(evidence)

        scores = {
            "SEF conflict": sef.conflict,
            "Attribution entropy": entropy,
            "Attribution Gini spread": gini_spread,
            "Low confidence": 1.0 - confidence,
        }
        for method, score in scores.items():
            err_hc = error[high_conf]
            score_hc = score[high_conf]
            rows.append(
                {
                    "dataset": label,
                    "openml_name": openml_name,
                    "openml_version": version,
                    "seed": seed,
                    "method": method,
                    "top_quartile_error": top_quartile_error(err_hc, score_hc),
                    "error_auc": score_auc(err_hc, score_hc),
                    "mean_score": float(np.mean(score_hc)),
                }
            )

        rows.append(
            {
                "dataset": label,
                "openml_name": openml_name,
                "openml_version": version,
                "seed": seed,
                "method": "SEF-entropy rank correlation",
                "top_quartile_error": float(pd.Series(sef.conflict[high_conf]).corr(pd.Series(entropy[high_conf]), method="spearman")),
                "error_auc": float("nan"),
                "mean_score": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    usable = runs[runs["method"] != "SEF-entropy rank correlation"].copy()
    summary = (
        usable.groupby(["dataset", "method"], as_index=False)
        .agg(
            top_quartile_error=("top_quartile_error", "mean"),
            top_quartile_error_sd=("top_quartile_error", "std"),
            error_auc=("error_auc", "mean"),
            error_auc_sd=("error_auc", "std"),
        )
    )

    corr = (
        runs[runs["method"] == "SEF-entropy rank correlation"]
        .groupby("dataset", as_index=False)
        .agg(sef_entropy_spearman=("top_quartile_error", "mean"))
    )
    summary = summary.merge(corr, on="dataset", how="left")
    dataset_order = {label: i for i, (label, _, _) in enumerate(DATASETS)}
    method_order = {
        "SEF conflict": 0,
        "Attribution entropy": 1,
        "Attribution Gini spread": 2,
        "Low confidence": 3,
    }
    summary["_dataset_order"] = summary["dataset"].map(dataset_order)
    summary["_method_order"] = summary["method"].map(method_order)
    return (
        summary.sort_values(["_dataset_order", "_method_order"])
        .drop(columns=["_dataset_order", "_method_order"])
        .reset_index(drop=True)
    )


def write_latex_table(summary: pd.DataFrame) -> None:
    order = ["SEF conflict", "Attribution entropy", "Attribution Gini spread", "Low confidence"]
    lines = [
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Dataset & Ranking score & Top-quartile error & Error AUC \\\\",
        "\\midrule",
    ]
    for dataset in summary["dataset"].drop_duplicates():
        block = summary[summary["dataset"] == dataset].set_index("method")
        for method in order:
            row = block.loc[method]
            lines.append(
                f"{dataset} & {method} & {row['top_quartile_error']:.3f} & {row['error_auc']:.3f} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_entropy_comparison_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame) -> None:
    order = ["SEF conflict", "Attribution entropy", "Attribution Gini spread", "Low confidence"]
    colors = {
        "SEF conflict": "#4c78a8",
        "Attribution entropy": "#f58518",
        "Attribution Gini spread": "#b279a2",
        "Low confidence": "#72b7b2",
    }
    datasets = summary["dataset"].drop_duplicates().tolist()
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i, method in enumerate(order):
        vals = []
        for dataset in datasets:
            vals.append(
                float(
                    summary[(summary["dataset"] == dataset) & (summary["method"] == method)][
                        "top_quartile_error"
                    ].iloc[0]
                )
            )
        offset = (i - 1.5) * width
        ax.bar(x + offset, vals, width=width, label=method, color=colors[method])
        for xi, val in zip(x + offset, vals):
            ax.text(xi, val + 0.005, f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Error among top-risk quartile")
    ax.set_title("SEF conflict is not just attribution entropy")
    ax.legend(frameon=False, ncols=2)
    ax.set_ylim(0, max(summary["top_quartile_error"]) * 1.28)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure14_entropy_comparison.png", dpi=220)


def main() -> None:
    runs = pd.concat(
        [analyze_dataset(label, name, version, range(50)) for label, name, version in DATASETS],
        ignore_index=True,
    )
    summary = summarize(runs)
    runs.to_csv(RES_DIR / "sef_entropy_comparison_runs.csv", index=False)
    summary.to_csv(RES_DIR / "sef_entropy_comparison_summary.csv", index=False)
    write_latex_table(summary)
    make_figure(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
