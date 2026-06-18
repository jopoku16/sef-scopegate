"""Finance and credit-risk benchmark for Signed Evidence Flow.

The benchmark asks a narrow question: among predictions that already look
confident, does signed evidence conflict still identify a riskier subset?
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
    ("German credit", "credit-g", 1),
    ("Bank marketing", "bank-marketing", 1),
    ("Credit-card default", "default-of-credit-card-clients", 1),
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
    categorical = [col for col in x.columns if str(x[col].dtype) in {"category", "object", "bool"}]
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
    return (z_test - baseline) * model.coef_[0]


def attribution_entropy(evidence: np.ndarray) -> np.ndarray:
    abs_ev = np.abs(evidence)
    p = abs_ev / (abs_ev.sum(axis=1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(axis=1) / np.log(max(evidence.shape[1], 2))


def auc_or_nan(error: np.ndarray, score: np.ndarray) -> float:
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

        evidence = evidence_from_pipeline(pipe, x_train, x_test)
        sef = evidence_scores(evidence)
        entropy = attribution_entropy(evidence)

        high_conf = confidence >= np.quantile(confidence, 0.50)
        conflict_hc = sef.conflict[high_conf]
        low_cut = np.quantile(conflict_hc, 0.25)
        high_cut = np.quantile(conflict_hc, 0.75)
        low_group = high_conf & (sef.conflict <= low_cut)
        high_group = high_conf & (sef.conflict >= high_cut)

        err_hc = error[high_conf]
        rows.append(
            {
                "dataset": label,
                "openml_name": openml_name,
                "seed": seed,
                "n": len(y),
                "p_raw": x.shape[1],
                "test_error": float(error.mean()),
                "auc": auc_or_nan(y_test, prob),
                "high_confidence_error": float(err_hc.mean()),
                "high_confidence_low_conflict_error": float(error[low_group].mean()),
                "high_confidence_high_conflict_error": float(error[high_group].mean()),
                "gap": float(error[high_group].mean() - error[low_group].mean()),
                "sef_error_auc_high_conf": auc_or_nan(err_hc, sef.conflict[high_conf]),
                "low_confidence_error_auc_high_conf": auc_or_nan(err_hc, 1.0 - confidence[high_conf]),
                "entropy_error_auc_high_conf": auc_or_nan(err_hc, entropy[high_conf]),
            }
        )
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["dataset", "openml_name", "n", "p_raw"], as_index=False)
        .agg(
            test_error=("test_error", "mean"),
            auc=("auc", "mean"),
            high_confidence_error=("high_confidence_error", "mean"),
            high_confidence_low_conflict_error=("high_confidence_low_conflict_error", "mean"),
            high_confidence_high_conflict_error=("high_confidence_high_conflict_error", "mean"),
            gap=("gap", "mean"),
            split_success_rate=("gap", lambda x: float((x > 0).mean())),
            sef_error_auc_high_conf=("sef_error_auc_high_conf", "mean"),
            low_confidence_error_auc_high_conf=("low_confidence_error_auc_high_conf", "mean"),
            entropy_error_auc_high_conf=("entropy_error_auc_high_conf", "mean"),
        )
    )


def write_latex(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & $n$ & Error & High-conf. low-conflict & High-conf. high-conflict & Success & SEF AUC \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"{row['dataset']} & {int(row['n'])} & {row['test_error']:.3f} & "
            f"{row['high_confidence_low_conflict_error']:.3f} & "
            f"{row['high_confidence_high_conflict_error']:.3f} & "
            f"{row['split_success_rate']:.2f} & "
            f"{row['sef_error_auc_high_conf']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_finance_credit_benchmark_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(summary))
    width = 0.34
    low = summary["high_confidence_low_conflict_error"].to_numpy()
    high = summary["high_confidence_high_conflict_error"].to_numpy()
    ax.bar(x - width / 2, low, width, label="High confidence, low SEF conflict", color="#4c78a8")
    ax.bar(x + width / 2, high, width, label="High confidence, high SEF conflict", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"].tolist())
    ax.set_ylabel("Error rate")
    ax.set_title("Finance and credit-risk benchmarks beyond confidence")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(high.max(), low.max()) * 1.25)
    for i, val in enumerate(low):
        ax.text(i - width / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    for i, val in enumerate(high):
        ax.text(i + width / 2, val + 0.004, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure16_finance_credit_benchmark.png", dpi=220)


def main() -> None:
    all_runs = []
    seeds = range(25)
    for label, name, version in DATASETS:
        all_runs.append(analyze_dataset(label, name, version, seeds))
    runs = pd.concat(all_runs, ignore_index=True)
    summary = summarize(runs)
    runs.to_csv(RES_DIR / "sef_finance_credit_benchmark_runs.csv", index=False)
    summary.to_csv(RES_DIR / "sef_finance_credit_benchmark_summary.csv", index=False)
    write_latex(summary)
    make_figure(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
