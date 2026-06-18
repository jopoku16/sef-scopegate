"""Reviewer-facing scope diagnostics for Signed Evidence Flow.

This script asks two questions on held-out, high-confidence predictions:

1. Does SEF conflict improve error ranking after confidence and attribution
   entropy are already known?
2. Does error increase with conflict strongly enough to justify conflict-based
   review in the target population?

The first question is answered with cross-fitted error-risk models. The second
uses a one-sided permutation test whose statistic is the covariance between
conflict rank and the observed error indicator. No test labels are used to fit
the prediction model or to construct SEF scores.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sef import evidence_scores


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DATASETS = [
    ("Diabetes", "diabetes", 1, "Healthcare"),
    ("Heart disease", "heart-statlog", 1, "Healthcare"),
    ("Blood transfusion", "blood-transfusion-service-center", 1, "Healthcare"),
    ("German credit", "credit-g", 1, "Finance"),
    ("Bank marketing", "bank-marketing", 1, "Finance"),
    ("Credit-card default", "default-of-credit-card-clients", 1, "Finance"),
]


def load_binary(name: str, version: int) -> tuple[pd.DataFrame, np.ndarray]:
    data = fetch_openml(name=name, version=version, as_frame=True)
    frame = data.frame.copy()
    target = data.target.name if getattr(data, "target", None) is not None else frame.columns[-1]
    if target not in frame.columns:
        target = frame.columns[-1]
    y = pd.Categorical(frame.pop(target)).codes.astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name} is not binary")
    return frame.reset_index(drop=True), y


def make_pipeline(x: pd.DataFrame) -> Pipeline:
    categorical = [c for c in x.columns if str(x[c].dtype) in {"category", "object", "bool"}]
    numeric = [c for c in x.columns if c not in categorical]
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


def evidence_from_pipeline(
    pipe: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[np.ndarray, float, float]:
    """Return exact linear logit contributions and component timings.

    For a linear logit and mean transformed-feature reference, these are the
    interventional linear-SHAP values on the logit scale. This identity does not
    claim conditional SHAP validity when transformed features are dependent.
    """
    pre = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    start = perf_counter()
    z_train = pre.transform(x_train)
    z_test = pre.transform(x_test)
    baseline = z_train.mean(axis=0)
    evidence = (z_test - baseline) * model.coef_[0]
    attribution_seconds = perf_counter() - start
    start = perf_counter()
    _ = evidence_scores(evidence)
    sef_seconds = perf_counter() - start
    return evidence, attribution_seconds, sef_seconds


def attribution_entropy(evidence: np.ndarray) -> np.ndarray:
    absolute = np.abs(evidence)
    weights = absolute / (absolute.sum(axis=1, keepdims=True) + 1e-12)
    return -(weights * np.log(weights + 1e-12)).sum(axis=1) / np.log(max(evidence.shape[1], 2))


def auc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def cross_fitted_auc(error: np.ndarray, features: np.ndarray, seed: int) -> float:
    error = error.astype(int)
    minority = int(np.bincount(error, minlength=2).min())
    if minority < 3:
        return float("nan")
    folds = min(5, minority)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("risk", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )
    probability = cross_val_predict(model, features, error, cv=cv, method="predict_proba")[:, 1]
    return auc_or_nan(error, probability)


def monotone_permutation_test(
    error: np.ndarray, conflict: np.ndarray, seed: int, permutations: int = 499
) -> tuple[float, float, float]:
    rank = pd.Series(conflict).rank(method="average").to_numpy(dtype=float)
    rank = (rank - rank.mean()) / (rank.std() + 1e-12)
    centered_error = error - error.mean()
    statistic = float(np.mean(rank * centered_error))
    rng = np.random.default_rng(seed + 7919)
    null = np.empty(permutations, dtype=float)
    for b in range(permutations):
        null[b] = np.mean(rank * rng.permutation(centered_error))
    p_value = float((1 + np.sum(null >= statistic)) / (permutations + 1))
    low = conflict <= np.quantile(conflict, 0.25)
    high = conflict >= np.quantile(conflict, 0.75)
    gap = float(error[high].mean() - error[low].mean())
    return statistic, p_value, gap


def analyze_dataset(
    label: str, openml_name: str, version: int, domain: str, seeds: range
) -> pd.DataFrame:
    x, y = load_binary(openml_name, version)
    categorical_share = float(
        np.mean([str(x[c].dtype) in {"category", "object", "bool"} for c in x.columns])
    )
    rows = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.35, random_state=seed, stratify=y
        )
        pipe = make_pipeline(x_train)
        pipe.fit(x_train, y_train)
        probability = pipe.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        error = (prediction != y_test).astype(float)
        confidence = np.maximum(probability, 1.0 - probability)

        evidence, attribution_seconds, sef_seconds = evidence_from_pipeline(pipe, x_train, x_test)
        sef = evidence_scores(evidence)
        entropy = attribution_entropy(evidence)
        high_confidence = confidence >= np.quantile(confidence, 0.50)

        e = error[high_confidence]
        c = sef.conflict[high_confidence]
        h = entropy[high_confidence]
        low_conf = 1.0 - confidence[high_confidence]
        mass = sef.mass[high_confidence]
        base = np.column_stack([low_conf, h])
        augmented = np.column_stack([low_conf, h, c])
        base_auc = cross_fitted_auc(e, base, seed)
        augmented_auc = cross_fitted_auc(e, augmented, seed)
        statistic, p_value, gap = monotone_permutation_test(e, c, seed)
        low_group_n = int(np.sum(c <= np.quantile(c, 0.25)))
        high_group_n = int(np.sum(c >= np.quantile(c, 0.75)))
        diagnostic_eligible = bool(
            (e.sum() >= 20)
            and ((len(e) - e.sum()) >= 20)
            and (low_group_n >= 25)
            and (high_group_n >= 25)
        )

        rows.append(
            {
                "dataset": label,
                "domain": domain,
                "seed": seed,
                "n": len(y),
                "p_raw": x.shape[1],
                "categorical_share": categorical_share,
                "minority_share": float(min(y.mean(), 1.0 - y.mean())),
                "test_error": float(error.mean()),
                "brier": float(brier_score_loss(y_test, probability)),
                "high_confidence_n": int(high_confidence.sum()),
                "high_confidence_errors": int(e.sum()),
                "base_conditional_auc": base_auc,
                "augmented_conditional_auc": augmented_auc,
                "delta_conditional_auc": augmented_auc - base_auc,
                "conflict_error_auc": auc_or_nan(e, c),
                "entropy_error_auc": auc_or_nan(e, h),
                "low_confidence_error_auc": auc_or_nan(e, low_conf),
                "mass_error_auc": auc_or_nan(e, mass),
                "monotone_statistic": statistic,
                "monotone_p_value": p_value,
                "conflict_error_gap": gap,
                "diagnostic_eligible": float(diagnostic_eligible),
                "diagnostic_pass": (
                    float((gap > 0) and (p_value <= 0.05)) if diagnostic_eligible else float("nan")
                ),
                "attribution_seconds": attribution_seconds,
                "sef_seconds": sef_seconds,
                "rows_per_second": len(x_test) / max(attribution_seconds + sef_seconds, 1e-12),
            }
        )
    return pd.DataFrame(rows)


def paired_sign_flip_p(values: pd.Series, seed: int = 20260618, draws: int = 9999) -> float:
    values = values.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        null[i] = np.mean(values * rng.choice([-1.0, 1.0], size=len(values)))
    return float((1 + np.sum(null >= observed)) / (draws + 1))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
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
            low_confidence_error_auc=("low_confidence_error_auc", "mean"),
            conflict_error_auc=("conflict_error_auc", "mean"),
            attribution_seconds=("attribution_seconds", "mean"),
            sef_seconds=("sef_seconds", "mean"),
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
        "Dataset & Base AUC & +SEF AUC & $\\Delta$AUC & Gap & Eligible/pass & Time/1k rows \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        ms_per_1000 = 1_000_000.0 / row["rows_per_second"]
        pass_text = "--" if pd.isna(row["diagnostic_pass_rate"]) else f"{row['diagnostic_pass_rate']:.2f}"
        lines.append(
            f"{row['dataset']} & {row['base_conditional_auc']:.3f} & "
            f"{row['augmented_conditional_auc']:.3f} & {row['delta_conditional_auc']:+.3f} & "
            f"{row['conflict_error_gap']:+.3f} & "
            f"{row['diagnostic_eligible_rate']:.2f}/{pass_text} & "
            f"{ms_per_1000:.1f} ms \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_scope_diagnostic_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    colors = ["#4c78a8" if d == "Healthcare" else "#e45756" for d in summary["domain"]]
    axes[0].bar(summary["dataset"], summary["delta_conditional_auc"], color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Cross-fitted AUC gain from SEF conflict")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_title("Increment beyond confidence and entropy")

    axes[1].bar(summary["dataset"], summary["diagnostic_pass_rate"], color=colors)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Fraction of splits passing monotone-risk test")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_title("Finite-sample deployment diagnostic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure18_scope_diagnostic.png", dpi=220)


def main() -> None:
    seeds = range(15)
    runs = pd.concat(
        [analyze_dataset(label, name, version, domain, seeds) for label, name, version, domain in DATASETS],
        ignore_index=True,
    )
    summary = summarize(runs)
    runs.to_csv(RES_DIR / "sef_scope_diagnostic_runs.csv", index=False)
    summary.to_csv(RES_DIR / "sef_scope_diagnostic_summary.csv", index=False)
    write_table(summary)
    make_figure(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
