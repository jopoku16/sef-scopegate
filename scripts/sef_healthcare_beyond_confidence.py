"""Healthcare benchmark showing SEF value beyond confidence.

This script uses public OpenML healthcare data sets. The main question is not
whether SEF beats confidence as a general triage rule. The question is sharper:
among predictions that already look confident, does SEF conflict still identify
a riskier subset?
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

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
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


def analyze_dataset(label: str, openml_name: str, version: int, seeds: range) -> tuple[pd.DataFrame, pd.DataFrame]:
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

        evidence = evidence_from_pipeline(pipe, x_train, x_test)
        sef = evidence_scores(evidence)

        high_conf = confidence >= np.quantile(confidence, 0.50)
        conflict_high_conf = sef.conflict[high_conf]
        low_cut = np.quantile(conflict_high_conf, 0.25)
        high_cut = np.quantile(conflict_high_conf, 0.75)
        high_conf_low_conflict = high_conf & (sef.conflict <= low_cut)
        high_conf_high_conflict = high_conf & (sef.conflict >= high_cut)

        low_err = error[high_conf_low_conflict].mean()
        high_err = error[high_conf_high_conflict].mean()
        rows.append(
            {
                "dataset": label,
                "openml_name": openml_name,
                "openml_version": version,
                "seed": seed,
                "n": len(y),
                "p_raw": x.shape[1],
                "test_error": error.mean(),
                "auc": roc_auc_score(y_test, prob),
                "high_confidence_error": error[high_conf].mean(),
                "high_confidence_low_conflict_error": low_err,
                "high_confidence_high_conflict_error": high_err,
                "gap": high_err - low_err,
                "mean_conflict": sef.conflict.mean(),
            }
        )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["dataset", "openml_name", "openml_version", "n", "p_raw"], as_index=False)
        .agg(
            test_error=("test_error", "mean"),
            auc=("auc", "mean"),
            high_confidence_error=("high_confidence_error", "mean"),
            high_confidence_low_conflict_error=("high_confidence_low_conflict_error", "mean"),
            high_confidence_high_conflict_error=("high_confidence_high_conflict_error", "mean"),
            gap=("gap", "mean"),
            split_success_rate=("gap", lambda x: float((x > 0).mean())),
        )
    )
    summary["ratio"] = (
        summary["high_confidence_high_conflict_error"]
        / summary["high_confidence_low_conflict_error"].replace(0, np.nan)
    )
    return runs, summary


def write_latex_table(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Dataset & $n$ & Error & High-conf. low-conflict & High-conf. high-conflict & Split success \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"{row['dataset']} & {int(row['n'])} & {row['test_error']:.3f} & "
            f"{row['high_confidence_low_conflict_error']:.3f} & "
            f"{row['high_confidence_high_conflict_error']:.3f} & "
            f"{row['split_success_rate']:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_healthcare_beyond_confidence_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(summary))
    width = 0.34
    low = summary["high_confidence_low_conflict_error"].to_numpy()
    high = summary["high_confidence_high_conflict_error"].to_numpy()
    ax.bar(x - width / 2, low, width, label="High confidence, low SEF conflict", color="#54a24b")
    ax.bar(x + width / 2, high, width, label="High confidence, high SEF conflict", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"].tolist())
    ax.set_ylabel("Error rate")
    ax.set_title("Healthcare benchmarks: SEF conflict separates risk beyond confidence")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(high.max(), low.max()) * 1.25)
    for i, val in enumerate(low):
        ax.text(i - width / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for i, val in enumerate(high):
        ax.text(i + width / 2, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure13_healthcare_beyond_confidence.png", dpi=220)


def main() -> None:
    all_runs = []
    all_summary = []
    seeds = range(50)
    for label, name, version in DATASETS:
        runs, summary = analyze_dataset(label, name, version, seeds)
        all_runs.append(runs)
        all_summary.append(summary)

    runs_df = pd.concat(all_runs, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    runs_df.to_csv(RES_DIR / "sef_healthcare_beyond_confidence_runs.csv", index=False)
    summary_df.to_csv(RES_DIR / "sef_healthcare_beyond_confidence_summary.csv", index=False)
    write_latex_table(summary_df)
    make_figure(summary_df)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
