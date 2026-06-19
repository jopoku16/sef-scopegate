"""Stress-test the SEF identification and conditional-value results.

The experiment uses discrete confidence strata so the base error-risk model
can represent every confidence-specific error rate without imposing a smooth
functional form. In the identified scenario, evidence mass is fixed within
each stratum and conflict is therefore determined by confidence. In the hidden
mass scenario, evidence mass varies within a stratum and error risk also
depends on conflict.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
SEED = 20260618


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_scenario(
    rng: np.random.Generator,
    scenario: str,
    n: int = 6000,
    levels: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stratum = rng.integers(0, levels, size=n)
    abs_net = 0.35 + 2.65 * stratum / (levels - 1)

    if scenario == "identified":
        mass = abs_net + 0.35 + 0.20 * abs_net
    elif scenario == "hidden_mass":
        mass = abs_net + rng.gamma(shape=1.8, scale=0.65, size=n)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    conflict = 1.0 - abs_net / mass
    base_logit = -2.0 + 0.75 * (3.0 - abs_net)
    if scenario == "hidden_mass":
        base_logit = base_logit + 2.40 * conflict
    error = rng.binomial(1, sigmoid(base_logit))
    return stratum, conflict, error


def one_hot(stratum: np.ndarray, levels: int = 12) -> np.ndarray:
    return np.eye(levels, dtype=float)[stratum]


def cross_fitted_metrics(
    stratum: np.ndarray,
    conflict: np.ndarray,
    error: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    x_base = one_hot(stratum)
    x_aug = np.column_stack([x_base, conflict])
    p_base = np.zeros(error.size)
    p_aug = np.zeros(error.size)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for train, test in cv.split(x_base, error):
        base = LogisticRegression(C=1e6, max_iter=2000)
        aug = LogisticRegression(C=1e6, max_iter=2000)
        base.fit(x_base[train], error[train])
        aug.fit(x_aug[train], error[train])
        p_base[test] = base.predict_proba(x_base[test])[:, 1]
        p_aug[test] = aug.predict_proba(x_aug[test])[:, 1]

    brier_gain = brier_score_loss(error, p_base) - brier_score_loss(error, p_aug)
    auc_gain = roc_auc_score(error, p_aug) - roc_auc_score(error, p_base)
    return float(brier_gain), float(auc_gain)


def conditional_variance(stratum: np.ndarray, conflict: np.ndarray) -> float:
    frame = pd.DataFrame({"stratum": stratum, "conflict": conflict})
    by_stratum = frame.groupby("stratum")["conflict"].agg(["var", "size"])
    weights = by_stratum["size"] / by_stratum["size"].sum()
    return float((weights * by_stratum["var"].fillna(0.0)).sum())


def run_experiment(repetitions: int = 50) -> pd.DataFrame:
    rows = []
    for scenario_index, scenario in enumerate(["identified", "hidden_mass"]):
        for rep in range(repetitions):
            seed = SEED + 1000 * scenario_index + rep
            rng = np.random.default_rng(seed)
            stratum, conflict, error = simulate_scenario(rng, scenario)
            brier_gain, auc_gain = cross_fitted_metrics(
                stratum, conflict, error, seed
            )
            rows.append(
                {
                    "scenario": scenario,
                    "replicate": rep,
                    "conditional_conflict_variance": conditional_variance(
                        stratum, conflict
                    ),
                    "brier_gain": brier_gain,
                    "auc_gain": auc_gain,
                    "error_rate": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)


def save_outputs(results: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RES_DIR / "sef_identification_stress_test.csv", index=False)

    summary = (
        results.groupby("scenario")
        .agg(
            conditional_variance_mean=("conditional_conflict_variance", "mean"),
            conditional_variance_sd=("conditional_conflict_variance", "std"),
            brier_gain_mean=("brier_gain", "mean"),
            brier_gain_sd=("brier_gain", "std"),
            auc_gain_mean=("auc_gain", "mean"),
            auc_gain_sd=("auc_gain", "std"),
            positive_brier_gain_rate=("brier_gain", lambda x: np.mean(x > 0)),
        )
        .reset_index()
    )
    summary.to_csv(
        RES_DIR / "sef_identification_stress_test_summary.csv", index=False
    )

    labels = ["Identified\nmass", "Hidden\nmass"]
    colors = ["#7A8B99", "#0E7C7B"]
    order = ["identified", "hidden_mass"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    metrics = [
        ("conditional_conflict_variance", r"$E\{\mathrm{Var}(C\mid Q)\}$"),
        ("brier_gain", "Cross-fitted Brier gain"),
        ("auc_gain", "Cross-fitted error-AUC gain"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        values = [
            results.loc[results["scenario"] == scenario, metric].to_numpy()
            for scenario in order
        ]
        box = ax.boxplot(values, patch_artist=True, widths=0.58, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.axhline(0.0, color="#444444", linewidth=0.9)
        ax.set_xticks([1, 2], labels)
        ax.set_title(title)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Conflict helps only when evidence mass varies beyond confidence",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        FIG_DIR / "figure21_identification_stress_test.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    results = run_experiment()
    save_outputs(results)
    summary = pd.read_csv(
        RES_DIR / "sef_identification_stress_test_summary.csv"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
