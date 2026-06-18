"""Confidence-masked conflict stress test for Signed Evidence Flow.

This experiment creates a setting where a fitted model can be confident while
the evidence behind the prediction is internally conflicted. The data-generating
process is deliberately transparent: when large supporting and opposing signals
arrive together, the outcome becomes less reliable even if the fitted linear
score remains far from zero.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sef import audit_by_budget, evidence_scores


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
FIG_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_conflict_masked_data(n: int = 60000, seed: int = 20260617):
    rng = np.random.default_rng(seed)

    support_driver = rng.lognormal(mean=0.35, sigma=0.75, size=n)
    opposition_driver = rng.lognormal(mean=0.35, sigma=0.75, size=n)
    weak_signal = rng.normal(size=n)
    nuisance = rng.normal(size=(n, 5))

    x = np.column_stack([support_driver, opposition_driver, weak_signal, nuisance])

    support = 1.45 * support_driver
    opposition = 1.30 * opposition_driver
    weak = 0.35 * weak_signal
    net = support - opposition + weak
    mass = np.abs(support) + np.abs(opposition) + np.abs(weak) + 1e-12
    conflict = 2.0 * np.minimum(np.abs(support), np.abs(opposition)) / mass

    # High conflict reduces the reliability of the label without removing the
    # apparent model margin. A linear model can therefore look confident in some
    # cases where the evidence is genuinely mixed.
    damping = 1.0 + 2.75 * conflict * np.sqrt(mass)
    p = sigmoid(2.6 * net / damping)
    y = rng.binomial(1, p)
    return x, y, conflict


def sefrisk_from_linear_model(model: LogisticRegression, scaler: StandardScaler, x_train: np.ndarray, x_test: np.ndarray):
    x_test_s = scaler.transform(x_test)
    baseline = scaler.transform(x_train).mean(axis=0)
    coefs = model.coef_[0]
    evidence = (x_test_s - baseline) * coefs
    scores = evidence_scores(evidence)
    return evidence, scores


def main() -> None:
    x, y, true_conflict = make_conflict_masked_data()
    x_train, x_test, y_train, y_test, _, conflict_test = train_test_split(
        x, y, true_conflict, test_size=0.35, random_state=41, stratify=y
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(x_train_s, y_train)

    prob = model.predict_proba(x_test_s)[:, 1]
    pred = (prob >= 0.5).astype(int)
    error = (pred != y_test).astype(float)
    confidence = np.maximum(prob, 1.0 - prob)
    _, scores = sefrisk_from_linear_model(model, scaler, x_train, x_test)

    high_conf = confidence >= np.quantile(confidence, 0.60)
    conflict_inside_confident = scores.conflict[high_conf]
    low_conflict_cut = np.quantile(conflict_inside_confident, 0.25)
    high_conflict_cut = np.quantile(conflict_inside_confident, 0.75)
    high_conf_low_conflict = high_conf & (scores.conflict <= low_conflict_cut)
    high_conf_high_conflict = high_conf & (scores.conflict >= high_conflict_cut)

    review_rates = np.linspace(0, 0.5, 11)
    frontier = []
    for rate in review_rates:
        # Compare triage rules only among predictions that already look
        # confident. This tests whether SEF adds information after confidence
        # has already selected seemingly safe cases.
        sef = audit_by_budget(error[high_conf], scores.risk[high_conf], float(rate))
        conf = audit_by_budget(error[high_conf], 1.0 - confidence[high_conf], float(rate))
        rank_conf = pd.Series(1.0 - confidence[high_conf]).rank(pct=True).to_numpy()
        rank_sef = pd.Series(scores.risk[high_conf]).rank(pct=True).to_numpy()
        hybrid = audit_by_budget(error[high_conf], 0.5 * rank_conf + 0.5 * rank_sef, float(rate))
        frontier.append(
            {
                "review_rate": float(rate),
                "accepted_error_sef": sef["accepted_error"],
                "accepted_error_confidence": conf["accepted_error"],
                "accepted_error_hybrid": hybrid["accepted_error"],
            }
        )
    frontier_df = pd.DataFrame(frontier)

    review_30 = int(np.argmin(np.abs(frontier_df["review_rate"].to_numpy() - 0.30)))

    summary = {
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, prob)),
        "log_loss": float(log_loss(y_test, prob)),
        "overall_error": float(error.mean()),
        "high_confidence_error": float(error[high_conf].mean()),
        "high_confidence_low_conflict_error": float(error[high_conf_low_conflict].mean()),
        "high_confidence_high_conflict_error": float(error[high_conf_high_conflict].mean()),
        "high_confidence_low_conflict_n": int(high_conf_low_conflict.sum()),
        "high_confidence_high_conflict_n": int(high_conf_high_conflict.sum()),
        "sef_30_review_error": float(frontier_df.loc[review_30, "accepted_error_sef"]),
        "confidence_30_review_error": float(frontier_df.loc[review_30, "accepted_error_confidence"]),
        "hybrid_30_review_error": float(frontier_df.loc[review_30, "accepted_error_hybrid"]),
        "true_conflict_corr_with_sef_conflict": float(np.corrcoef(conflict_test, scores.conflict)[0, 1]),
    }

    pd.DataFrame([summary]).to_csv(RES_DIR / "sef_confidence_masking_summary.csv", index=False)
    frontier_df.to_csv(RES_DIR / "sef_confidence_masking_frontier.csv", index=False)

    table = pd.DataFrame(
        [
            {
                "Group": "All test cases",
                "Cases": len(y_test),
                "Error": error.mean(),
            },
            {
                "Group": "High confidence",
                "Cases": int(high_conf.sum()),
                "Error": error[high_conf].mean(),
            },
            {
                "Group": "High confidence, low SEF conflict",
                "Cases": int(high_conf_low_conflict.sum()),
                "Error": error[high_conf_low_conflict].mean(),
            },
            {
                "Group": "High confidence, high SEF conflict",
                "Cases": int(high_conf_high_conflict.sum()),
                "Error": error[high_conf_high_conflict].mean(),
            },
        ]
    )
    table_lines = [
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Group & Cases & Error \\\\",
        "\\midrule",
    ]
    for row in table.to_dict("records"):
        table_lines.append(f"{row['Group']} & {int(row['Cases'])} & {row['Error']:.3f} \\\\")
    table_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (RES_DIR / "sef_confidence_masking_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    bars = [
        summary["overall_error"],
        summary["high_confidence_error"],
        summary["high_confidence_low_conflict_error"],
        summary["high_confidence_high_conflict_error"],
    ]
    labels = ["All", "High\nconfidence", "High conf.\nlow conflict", "High conf.\nhigh conflict"]
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#e45756"]
    axes[0].bar(labels, bars, color=colors)
    axes[0].set_ylabel("Error rate")
    axes[0].set_title("Conflict separates risk inside confident predictions")
    axes[0].set_ylim(0, max(bars) * 1.25)
    for i, val in enumerate(bars):
        axes[0].text(i, val + 0.006, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].plot(frontier_df["review_rate"], frontier_df["accepted_error_sef"], marker="o", label="SEF risk")
    axes[1].plot(frontier_df["review_rate"], frontier_df["accepted_error_confidence"], marker="s", label="Low confidence")
    axes[1].plot(frontier_df["review_rate"], frontier_df["accepted_error_hybrid"], marker="^", label="Confidence + SEF")
    axes[1].set_xlabel("Fraction sent to review")
    axes[1].set_ylabel("Accepted-case error")
    axes[1].set_title("Review triage under masked conflict")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure12_confidence_masking_stress_test.png", dpi=220)
    print(pd.DataFrame([summary]).T)


if __name__ == "__main__":
    main()
