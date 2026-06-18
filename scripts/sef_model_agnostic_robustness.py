"""Model-agnostic robustness checks for Signed Evidence Flow.

This script tests SEF on non-linear black-box classifiers. It uses perturbation
contrasts rather than model coefficients:

    E_j(x) = f(x) - E_ref[f(x with feature j replaced by reference values)].

The experiment is repeated across seeds and compares SEF review triage with
ordinary confidence-based triage.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


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


def model_specs(seed: int):
    return [
        (
            "Random forest",
            RandomForestClassifier(
                n_estimators=90,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "Gradient boosting",
            make_pipeline(
                StandardScaler(),
                HistGradientBoostingClassifier(
                    max_iter=110,
                    learning_rate=0.055,
                    l2_regularization=0.01,
                    random_state=seed,
                ),
            ),
        ),
    ]


def score_model(model, x: np.ndarray) -> np.ndarray:
    return logit(model.predict_proba(x)[:, 1])


def model_agnostic_evidence(
    model,
    x_test: np.ndarray,
    x_ref: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    n, p = x_test.shape
    f0 = score_model(model, x_test)
    evidence = np.zeros((n, p), dtype=float)
    ref_value = np.median(x_ref, axis=0)
    for j in range(p):
        x_rep = x_test.copy()
        x_rep[:, j] = ref_value[j]
        evidence[:, j] = f0 - score_model(model, x_rep)
    return evidence


def reference_stability(
    model,
    x_test: np.ndarray,
    x_ref: np.ndarray,
    evidence: np.ndarray,
    rng: np.random.Generator,
    b: int = 4,
    m_ref: int = 6,
) -> np.ndarray:
    sign0 = np.sign(evidence)
    agree = np.zeros_like(evidence, dtype=float)
    for _ in range(b):
        ev_b = model_agnostic_evidence(model, x_test, x_ref, rng)
        agree += (np.sign(ev_b) == sign0).astype(float)
    stab_j = agree / b
    weights = np.abs(evidence)
    return (weights * stab_j).sum(axis=1) / (weights.sum(axis=1) + 1e-12)


def sef_scores(evidence: np.ndarray) -> Dict[str, np.ndarray]:
    s_pos = np.maximum(evidence, 0.0).sum(axis=1)
    s_neg = np.maximum(-evidence, 0.0).sum(axis=1)
    mass = s_pos + s_neg
    conflict = 2.0 * np.minimum(s_pos, s_neg) / (mass + 1e-12)
    direction = (s_pos - s_neg) / (mass + 1e-12)
    return {"support": s_pos, "opposition": s_neg, "conflict": conflict, "direction": direction}


def quartile_error(err: np.ndarray, score: np.ndarray, high: bool) -> float:
    q = np.quantile(score, 0.75 if high else 0.25)
    mask = score >= q if high else score <= q
    return float(err[mask].mean())


def accepted_error(err: np.ndarray, risk: np.ndarray, review_rate: float) -> float:
    n = len(err)
    review_n = int(np.floor(review_rate * n))
    order = np.argsort(-risk)
    review = np.zeros(n, dtype=bool)
    if review_n > 0:
        review[order[:review_n]] = True
    return float(err[~review].mean())


def run_one(dataset: str, x: np.ndarray, y: np.ndarray, model_name: str, model, seed: int) -> Dict[str, float | str]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    model.fit(x_train, y_train)
    rng = np.random.default_rng(seed + 1009)
    evidence = model_agnostic_evidence(model, x_test, x_train, rng)
    scores = sef_scores(evidence)
    # For this black-box robustness benchmark we focus on the fast
    # model-agnostic conflict screen. Stability is tested separately in the
    # smaller perturbation experiments.
    stability = np.ones(evidence.shape[0], dtype=float)
    res = (1.0 - scores["conflict"]) * stability
    prob = model.predict_proba(x_test)[:, 1]
    confidence = np.maximum(prob, 1.0 - prob)
    pred = (prob >= 0.5).astype(int)
    err = (pred != y_test).astype(float)

    row: Dict[str, float | str] = {
        "dataset": dataset,
        "model": model_name,
        "seed": float(seed),
        "test_error": float(err.mean()),
        "low_conflict_error": quartile_error(err, scores["conflict"], high=False),
        "high_conflict_error": quartile_error(err, scores["conflict"], high=True),
        "high_res_error": quartile_error(err, res, high=True),
        "low_res_error": quartile_error(err, res, high=False),
        "mean_conflict": float(scores["conflict"].mean()),
        "mean_stability": float(stability.mean()),
        "mean_res": float(res.mean()),
    }
    for rr in [0.10, 0.20, 0.30]:
        row[f"sef_triage_error_{int(rr*100)}"] = accepted_error(err, 1.0 - res, rr)
        row[f"confidence_triage_error_{int(rr*100)}"] = accepted_error(err, 1.0 - confidence, rr)
    return row


def aggregate(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    groups = sorted({(str(r["dataset"]), str(r["model"])) for r in rows})
    out = []
    numeric_keys = [k for k in rows[0].keys() if k not in {"dataset", "model", "seed"}]
    for dataset, model in groups:
        subset = [r for r in rows if r["dataset"] == dataset and r["model"] == model]
        row: Dict[str, float | str] = {"dataset": dataset, "model": model, "runs": float(len(subset))}
        for key in numeric_keys:
            vals = np.array([float(r[key]) for r in subset], dtype=float)
            row[f"{key}_mean"] = float(vals.mean())
            row[f"{key}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def overall_summary(rows: List[Dict[str, float | str]]) -> Dict[str, float]:
    keys = [
        "test_error",
        "low_conflict_error",
        "high_conflict_error",
        "high_res_error",
        "low_res_error",
        "sef_triage_error_30",
        "confidence_triage_error_30",
    ]
    return {key: float(np.mean([float(r[key]) for r in rows])) for key in keys}


def save_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_latex_table(rows: List[Dict[str, float | str]]) -> None:
    path = RES_DIR / "sef_model_agnostic_robustness_table.tex"
    with path.open("w") as f:
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Model & Error & Low conflict & High conflict & High RES \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['dataset']} & {row['model']} & "
                f"{row['test_error_mean']:.3f} & "
                f"{row['low_conflict_error_mean']:.3f} & "
                f"{row['high_conflict_error_mean']:.3f} & "
                f"{row['high_res_error_mean']:.3f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def _font(size: int = 18) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_summary_figure(summary: Dict[str, float], path: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((70, 35), "Model-agnostic SEF robustness across datasets and models", fill=(20, 20, 20), font=_font(28))
    labels = ["overall", "low conflict", "high conflict", "high RES", "low RES"]
    values = [
        summary["test_error"],
        summary["low_conflict_error"],
        summary["high_conflict_error"],
        summary["high_res_error"],
        summary["low_res_error"],
    ]
    colors = [
        (90, 90, 90, 230),
        (42, 113, 142, 230),
        (188, 86, 61, 230),
        (31, 114, 82, 230),
        (184, 113, 38, 230),
    ]
    x0, y0, plot_w, plot_h = 150, 130, 1180, 650
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    ymax = max(values) * 1.25
    bar_w = plot_w / len(values) * 0.52
    for k, (label, val, color) in enumerate(zip(labels, values, colors)):
        cx = x0 + (k + 0.5) * plot_w / len(values)
        h = val / ymax * plot_h
        draw.rectangle([cx - bar_w / 2, y0 + plot_h - h, cx + bar_w / 2, y0 + plot_h], fill=color)
        draw.text((cx - 32, y0 + plot_h - h - 32), f"{val:.3f}", fill=(20, 20, 20), font=_font(16))
        draw.text((cx - 58, y0 + plot_h + 25), label, fill=(20, 20, 20), font=_font(16))
    draw.text((60, 420), "error rate", fill=(20, 20, 20), font=_font(18))
    img.save(path)


def make_triage_comparison(summary: Dict[str, float], path: Path) -> None:
    width, height = 1500, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((70, 35), "Review triage at 30% review: SEF versus confidence", fill=(20, 20, 20), font=_font(28))
    labels = ["no review", "SEF risk", "confidence risk"]
    values = [summary["test_error"], summary["sef_triage_error_30"], summary["confidence_triage_error_30"]]
    colors = [(90, 90, 90, 230), (31, 114, 82, 230), (188, 86, 61, 230)]
    x0, y0, plot_w, plot_h = 210, 130, 1000, 610
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    ymax = max(values) * 1.25
    bar_w = 170
    for k, (label, val, color) in enumerate(zip(labels, values, colors)):
        cx = x0 + (k + 0.5) * plot_w / len(values)
        h = val / ymax * plot_h
        draw.rectangle([cx - bar_w / 2, y0 + plot_h - h, cx + bar_w / 2, y0 + plot_h], fill=color)
        draw.text((cx - 34, y0 + plot_h - h - 34), f"{val:.3f}", fill=(20, 20, 20), font=_font(18))
        draw.text((cx - 70, y0 + plot_h + 25), label, fill=(20, 20, 20), font=_font(18))
    draw.text((60, 410), "accepted error", fill=(20, 20, 20), font=_font(18))
    img.save(path)


def main() -> None:
    seeds = list(range(20))
    rows: List[Dict[str, float | str]] = []
    for dataset, x, y in benchmark_specs():
        for seed in seeds:
            for model_name, model in model_specs(seed):
                rows.append(run_one(dataset, x, y, model_name, model, seed))
    agg = aggregate(rows)
    summary = overall_summary(rows)
    save_csv(RES_DIR / "sef_model_agnostic_robustness_runs.csv", rows)
    save_csv(RES_DIR / "sef_model_agnostic_robustness_summary.csv", agg)
    with (RES_DIR / "sef_model_agnostic_overall_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])
    save_latex_table(agg)
    make_summary_figure(summary, FIG_DIR / "figure9_model_agnostic_robustness.png")
    make_triage_comparison(summary, FIG_DIR / "figure10_model_agnostic_triage_comparison.png")


if __name__ == "__main__":
    main()
