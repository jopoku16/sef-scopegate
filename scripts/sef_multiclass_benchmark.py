"""Multi-class validation for Signed Evidence Flow.

The experiment audits the predicted class against its strongest rival. Pairwise
signed evidence is computed from a multinomial logistic model, matching the
multi-class construction in the manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def standard_datasets() -> list[tuple[str, np.ndarray, np.ndarray, range]]:
    iris = load_iris()
    wine = load_wine()
    digits = load_digits()
    return [
        ("Iris (3 classes)", iris.data, iris.target, range(30)),
        ("Wine (3 classes)", wine.data, wine.target, range(30)),
        ("Digits (10 classes)", digits.data, digits.target, range(30)),
    ]


def covertype_sample(seed: int = 20260617, n_sample: int = 120_000) -> tuple[np.ndarray, np.ndarray]:
    data = fetch_covtype(data_home=DATA_DIR, download_if_missing=True)
    x = data.data.astype(np.float32)
    y = data.target.astype(int) - 1
    rng = np.random.default_rng(seed)
    selected = []
    per_class = n_sample // len(np.unique(y))
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        take = min(per_class, len(idx))
        selected.append(rng.choice(idx, size=take, replace=False))
    selected_idx = np.concatenate(selected)
    rng.shuffle(selected_idx)
    return x[selected_idx], y[selected_idx]


def fit_model(x_train: np.ndarray, y_train: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    z_train = scaler.fit_transform(x_train)
    model = LogisticRegression(
        max_iter=2500,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
    )
    model.fit(z_train, y_train)
    return scaler, model


def pairwise_evidence(
    model: LogisticRegression,
    z_train: np.ndarray,
    z_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = model.decision_function(z_test)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    predicted = np.argmax(scores, axis=1)
    runner_scores = scores.copy()
    runner_scores[np.arange(len(scores)), predicted] = -np.inf
    runner = np.argmax(runner_scores, axis=1)

    beta = model.coef_
    intercept = model.intercept_
    baseline = z_train.mean(axis=0)
    evidence = np.empty((len(z_test), z_test.shape[1] + 1), dtype=float)
    gaps = np.empty(len(z_test), dtype=float)

    for i, (winner, rival) in enumerate(zip(predicted, runner)):
        beta_gap = beta[winner] - beta[rival]
        baseline_gap = (intercept[winner] - intercept[rival]) + beta_gap @ baseline
        evidence[i, 0] = baseline_gap
        evidence[i, 1:] = beta_gap * (z_test[i] - baseline)
        gaps[i] = scores[i, winner] - scores[i, rival]
    return evidence, predicted, runner, gaps


def conflict_score(evidence: np.ndarray) -> np.ndarray:
    support = np.maximum(evidence, 0.0).sum(axis=1)
    opposition = np.maximum(-evidence, 0.0).sum(axis=1)
    return 2.0 * np.minimum(support, opposition) / (support + opposition + 1e-12)


def accepted_error(error: np.ndarray, risk: np.ndarray, review_rate: float = 0.30) -> float:
    review_n = int(np.floor(review_rate * len(error)))
    order = np.argsort(-risk)
    accepted = np.ones(len(error), dtype=bool)
    if review_n > 0:
        accepted[order[:review_n]] = False
    return float(error[accepted].mean())


def run_split(name: str, x: np.ndarray, y: np.ndarray, seed: int) -> dict[str, float | str]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    scaler, model = fit_model(x_train, y_train)
    z_train = scaler.transform(x_train)
    z_test = scaler.transform(x_test)
    evidence, predicted_index, _, gaps = pairwise_evidence(model, z_train, z_test)
    predicted = model.classes_[predicted_index]
    error = (predicted != y_test).astype(float)
    conflict = conflict_score(evidence)
    probability = model.predict_proba(z_test)
    confidence = probability.max(axis=1)

    high_conf = confidence >= np.quantile(confidence, 0.50)
    conflict_hc = conflict[high_conf]
    low_cut = np.quantile(conflict_hc, 0.25)
    high_cut = np.quantile(conflict_hc, 0.75)
    low_group = high_conf & (conflict <= low_cut)
    high_group = high_conf & (conflict >= high_cut)

    conflict_risk = conflict
    confidence_risk = 1.0 - confidence
    hybrid = 0.5 * standardize(conflict_risk) + 0.5 * standardize(confidence_risk)
    return {
        "dataset": name,
        "seed": seed,
        "n": len(y),
        "classes": len(np.unique(y)),
        "test_error": float(error.mean()),
        "mean_top_two_gap": float(gaps.mean()),
        "high_confidence_error": float(error[high_conf].mean()),
        "high_confidence_low_conflict_error": float(error[low_group].mean()),
        "high_confidence_high_conflict_error": float(error[high_group].mean()),
        "gap": float(error[high_group].mean() - error[low_group].mean()),
        "sef_review_error_30": accepted_error(error, conflict_risk),
        "confidence_review_error_30": accepted_error(error, confidence_risk),
        "hybrid_review_error_30": accepted_error(error, hybrid),
    }


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["dataset", "n", "classes"], as_index=False)
        .agg(
            test_error=("test_error", "mean"),
            high_confidence_error=("high_confidence_error", "mean"),
            high_confidence_low_conflict_error=("high_confidence_low_conflict_error", "mean"),
            high_confidence_high_conflict_error=("high_confidence_high_conflict_error", "mean"),
            conflict_gap=("gap", "mean"),
            split_success_rate=("gap", lambda x: float((x > 0).mean())),
            sef_review_error_30=("sef_review_error_30", "mean"),
            confidence_review_error_30=("confidence_review_error_30", "mean"),
            hybrid_review_error_30=("hybrid_review_error_30", "mean"),
        )
    )


def write_table(summary: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & Classes & Error & HC low conflict & HC high conflict & Success & Hybrid review \\\\",
        "\\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"{row['dataset']} & {int(row['classes'])} & {row['test_error']:.3f} & "
            f"{row['high_confidence_low_conflict_error']:.3f} & "
            f"{row['high_confidence_high_conflict_error']:.3f} & "
            f"{row['split_success_rate']:.2f} & {row['hybrid_review_error_30']:.3f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_multiclass_benchmark_table.tex").write_text("\n".join(lines), encoding="utf-8")


def make_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(summary))
    width = 0.34
    low = summary["high_confidence_low_conflict_error"].to_numpy()
    high = summary["high_confidence_high_conflict_error"].to_numpy()
    ax.bar(x - width / 2, low, width, label="High confidence, low conflict", color="#4c78a8")
    ax.bar(x + width / 2, high, width, label="High confidence, high conflict", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"].tolist())
    ax.set_ylabel("Error rate")
    ax.set_title("Multi-class SEF: evidence conflict within confident predictions")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(0.03, high.max(), low.max()) * 1.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure17_multiclass_benchmark.png", dpi=220)


def main() -> None:
    rows = []
    for name, x, y, seeds in standard_datasets():
        for seed in seeds:
            rows.append(run_split(name, x, y, seed))
    x_cov, y_cov = covertype_sample()
    for seed in range(5):
        rows.append(run_split("Covertype (7 classes)", x_cov, y_cov, seed))
    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    runs.to_csv(RES_DIR / "sef_multiclass_benchmark_runs.csv", index=False)
    summary.to_csv(RES_DIR / "sef_multiclass_benchmark_summary.csv", index=False)
    write_table(summary)
    make_figure(summary)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
