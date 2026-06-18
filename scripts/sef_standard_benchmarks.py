"""Repeated standard benchmark experiments for Signed Evidence Flow.

This script keeps the paper's first benchmark simple, but no longer relies on a
single train-test split. Each task is repeated over many random splits, and the
table reports mean errors with uncertainty.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def benchmark_specs() -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()
    digits = load_digits()
    iris_mask = iris.target != 0
    wine_mask = wine.target != 0
    digits_mask = np.isin(digits.target, [3, 5])
    return [
        ("Iris: versicolor vs virginica", iris.data[iris_mask], (iris.target[iris_mask] == 2).astype(int), "classic flower morphology"),
        ("Wine: class 1 vs class 2", wine.data[wine_mask], (wine.target[wine_mask] == 2).astype(int), "chemical wine measurements"),
        ("Breast cancer: benign vs malignant", cancer.data, cancer.target.astype(int), "diagnostic tumor measurements"),
        ("Digits: 3 vs 5", digits.data[digits_mask], (digits.target[digits_mask] == 5).astype(int), "handwritten digit images"),
    ]


def fit_scaled_logistic(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    model = LogisticRegression(max_iter=2500, solver="lbfgs", class_weight="balanced")
    model.fit(x_train_s, y_train)
    return scaler, model


def signed_evidence(model: LogisticRegression, x_scaled: np.ndarray, baseline_scaled: np.ndarray) -> np.ndarray:
    beta = model.coef_[0]
    return (x_scaled - baseline_scaled) * beta


def sef_scores(evidence: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    s_pos = np.maximum(evidence, 0.0).sum(axis=1)
    s_neg = np.maximum(-evidence, 0.0).sum(axis=1)
    mass = s_pos + s_neg
    conflict = 2.0 * np.minimum(s_pos, s_neg) / (mass + eps)
    direction = (s_pos - s_neg) / (mass + eps)
    return {
        "support": s_pos,
        "opposition": s_neg,
        "mass": mass,
        "conflict": conflict,
        "direction": direction,
        "flip_margin": np.abs(s_pos - s_neg) / 2.0,
    }


def bootstrap_stability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    evidence: np.ndarray,
    b: int = 40,
    seed: int = 20260617,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    sign0 = np.sign(evidence)
    agree = np.zeros_like(evidence, dtype=float)
    used = 0

    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_train[idx])) < 2:
            continue
        scaler_b, model_b = fit_scaled_logistic(x_train[idx], y_train[idx])
        x_train_b_s = scaler_b.transform(x_train[idx])
        x_test_b_s = scaler_b.transform(x_test)
        base_b = x_train_b_s.mean(axis=0)
        ev_b = signed_evidence(model_b, x_test_b_s, base_b)
        agree += (np.sign(ev_b) == sign0).astype(float)
        used += 1

    if used == 0:
        return np.ones(evidence.shape[0], dtype=float)
    stab_j = agree / used
    weights = np.abs(evidence)
    return (weights * stab_j).sum(axis=1) / (weights.sum(axis=1) + 1e-12)


def group_error(error: np.ndarray, score: np.ndarray, q: float, high: bool) -> float:
    cut = np.quantile(score, q)
    mask = score >= cut if high else score <= cut
    return float(error[mask].mean())


def accepted_error(error: np.ndarray, risk: np.ndarray, review_rate: float) -> float:
    n = len(error)
    review_n = int(np.floor(review_rate * n))
    order = np.argsort(-risk)
    review = np.zeros(n, dtype=bool)
    if review_n > 0:
        review[order[:review_n]] = True
    accepted = ~review
    return float(error[accepted].mean()) if accepted.any() else float("nan")


def learned_reject_risk(
    x_train_s: np.ndarray,
    y_train: np.ndarray,
    model: LogisticRegression,
    x_test_s: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train a small reject model to predict mistakes from validation cases."""
    x_fit, x_rej, y_fit, y_rej = train_test_split(
        x_train_s, y_train, test_size=0.35, random_state=10_000 + seed, stratify=y_train
    )
    base = LogisticRegression(max_iter=2500, solver="lbfgs", class_weight="balanced")
    base.fit(x_fit, y_fit)
    val_prob = base.predict_proba(x_rej)[:, 1]
    val_conf = np.maximum(val_prob, 1.0 - val_prob)
    val_margin = np.abs(val_prob - 0.5)
    val_pred = (val_prob >= 0.5).astype(int)
    val_error = (val_pred != y_rej).astype(int)

    if len(np.unique(val_error)) < 2:
        prob = model.predict_proba(x_test_s)[:, 1]
        return 1.0 - np.maximum(prob, 1.0 - prob)

    reject_x = np.column_stack([val_conf, val_margin])
    reject = GradientBoostingClassifier(random_state=20_000 + seed, max_depth=2, n_estimators=40)
    reject.fit(reject_x, val_error)

    test_prob = model.predict_proba(x_test_s)[:, 1]
    test_conf = np.maximum(test_prob, 1.0 - test_prob)
    test_margin = np.abs(test_prob - 0.5)
    return reject.predict_proba(np.column_stack([test_conf, test_margin]))[:, 1]


def run_split(name: str, x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[Dict[str, float | str], List[Dict[str, float | str]]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    scaler, model = fit_scaled_logistic(x_train, y_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    baseline = x_train_s.mean(axis=0)
    evidence = signed_evidence(model, x_test_s, baseline)
    scores = sef_scores(evidence)
    stability = bootstrap_stability(x_train, y_train, x_test, evidence, seed=30_000 + seed)
    res = (1.0 - scores["conflict"]) * stability
    prob = model.predict_proba(x_test_s)[:, 1]
    confidence = np.maximum(prob, 1.0 - prob)
    pred = model.predict(x_test_s)
    error = (pred != y_test).astype(float)

    sef_risk = 1.0 - res
    conf_risk = 1.0 - confidence
    hybrid_risk = 0.5 * standardize(sef_risk) + 0.5 * standardize(conf_risk)
    reject_risk = learned_reject_risk(x_train_s, y_train, model, x_test_s, seed)

    row = {
        "dataset": name,
        "seed": float(seed),
        "n": float(x.shape[0]),
        "p": float(x.shape[1]),
        "test_error": float(error.mean()),
        "low_conflict_error": group_error(error, scores["conflict"], 0.25, high=False),
        "high_conflict_error": group_error(error, scores["conflict"], 0.75, high=True),
        "low_res_error": group_error(error, res, 0.25, high=False),
        "high_res_error": group_error(error, res, 0.75, high=True),
        "sef_review_error_30": accepted_error(error, sef_risk, 0.30),
        "confidence_review_error_30": accepted_error(error, conf_risk, 0.30),
        "hybrid_review_error_30": accepted_error(error, hybrid_risk, 0.30),
        "learned_reject_review_error_30": accepted_error(error, reject_risk, 0.30),
        "mean_conflict": float(scores["conflict"].mean()),
        "mean_stability": float(stability.mean()),
        "mean_res": float(res.mean()),
    }
    curve = triage_curve(name, error, sef_risk, conf_risk, hybrid_risk, reject_risk, seed)
    return row, curve


def standardize(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def triage_curve(
    name: str,
    error: np.ndarray,
    sef_risk: np.ndarray,
    confidence_risk: np.ndarray,
    hybrid_risk: np.ndarray,
    reject_risk: np.ndarray,
    seed: int,
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    review_grid = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40])
    risks = [
        ("SEF", sef_risk),
        ("confidence", confidence_risk),
        ("confidence+SEF", hybrid_risk),
        ("learned reject", reject_risk),
    ]
    for review_rate in review_grid:
        for method, risk in risks:
            rows.append(
                {
                    "dataset": name,
                    "seed": float(seed),
                    "method": method,
                    "review_rate": float(review_rate),
                    "accepted_fraction": 1.0 - float(review_rate),
                    "accepted_error": accepted_error(error, risk, float(review_rate)),
                }
            )
    return rows


def summarize(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    datasets = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(str(row["dataset"]))
    metrics = [
        "test_error",
        "low_conflict_error",
        "high_conflict_error",
        "high_res_error",
        "low_res_error",
        "sef_review_error_30",
        "confidence_review_error_30",
        "hybrid_review_error_30",
        "learned_reject_review_error_30",
    ]
    out = []
    for dataset in datasets:
        block = [r for r in rows if r["dataset"] == dataset]
        item: Dict[str, float | str] = {
            "dataset": dataset,
            "n": float(block[0]["n"]),
            "p": float(block[0]["p"]),
            "splits": float(len(block)),
        }
        for metric in metrics:
            vals = np.array([float(r[metric]) for r in block], dtype=float)
            item[metric] = float(np.nanmean(vals))
            item[f"{metric}_se"] = float(np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals))))
            item[f"{metric}_ci"] = 1.96 * float(item[f"{metric}_se"])
        item["conflict_success_rate"] = float(np.mean([float(r["high_conflict_error"]) > float(r["low_conflict_error"]) for r in block]))
        out.append(item)
    return out


def save_results(rows: List[Dict[str, float | str]], summary: List[Dict[str, float | str]], curves: List[Dict[str, float | str]]) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(RES_DIR / "sef_standard_benchmarks_runs.csv", rows)
    write_csv(RES_DIR / "sef_standard_benchmarks_summary.csv", summary)
    write_csv(RES_DIR / "sef_triage_curves.csv", curves)

    with (RES_DIR / "sef_standard_benchmarks_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Error & Low conflict & High conflict & High RES & Low RES & Success \\\\\n")
        f.write("\\midrule\n")
        for row in summary:
            f.write(
                f"{row['dataset']} & "
                f"{mean_ci(row, 'test_error')} & "
                f"{mean_ci(row, 'low_conflict_error')} & "
                f"{mean_ci(row, 'high_conflict_error')} & "
                f"{mean_ci(row, 'high_res_error')} & "
                f"{mean_ci(row, 'low_res_error')} & "
                f"{row['conflict_success_rate']:.2f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    with (RES_DIR / "sef_selective_baseline_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Dataset & SEF & Confidence & Confidence+SEF & Learned reject \\\\\n")
        f.write("\\midrule\n")
        for row in summary:
            f.write(
                f"{row['dataset']} & "
                f"{mean_ci(row, 'sef_review_error_30')} & "
                f"{mean_ci(row, 'confidence_review_error_30')} & "
                f"{mean_ci(row, 'hybrid_review_error_30')} & "
                f"{mean_ci(row, 'learned_reject_review_error_30')} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def write_csv(path: Path, rows: List[Dict[str, float | str]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(row: Dict[str, float | str], key: str) -> str:
    return f"{float(row[key]):.3f} $\\pm$ {float(row[key + '_ci']):.3f}"


def _font(size: int = 18) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_bar_figure(rows: List[Dict[str, float | str]], path: Path, title: str, left_key: str, right_key: str, left_label: str, right_label: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((80, 35), title, fill=(20, 20, 20), font=_font(28))
    x0, y0 = 230, 130
    plot_w, plot_h = 1120, 650
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    ymax = max(max(float(row[left_key]) for row in rows), max(float(row[right_key]) for row in rows), 0.05) * 1.35
    for i in range(1, 5):
        y = y0 + i * plot_h / 5
        draw.line([x0, y, x0 + plot_w, y], fill=(225, 225, 225), width=1)
    group_w = plot_w / len(rows)
    bar_w = group_w * 0.25
    for k, row in enumerate(rows):
        cx = x0 + group_w * k + group_w / 2
        vals = [(float(row[left_key]), (42, 113, 142, 235)), (float(row[right_key]), (188, 86, 61, 235))]
        for j, (val, color) in enumerate(vals):
            h = val / ymax * plot_h
            bx = cx + (j - 0.5) * bar_w * 1.3
            draw.rectangle([bx - bar_w / 2, y0 + plot_h - h, bx + bar_w / 2, y0 + plot_h], fill=color)
            draw.text((bx - 32, y0 + plot_h - h - 28), f"{val:.3f}", fill=(20, 20, 20), font=_font(14))
        label = str(row["dataset"]).split(":")[0]
        draw.text((cx - 70, y0 + plot_h + 25), label, fill=(20, 20, 20), font=_font(17))
    draw.rectangle([980, 70, 1010, 95], fill=(42, 113, 142, 235))
    draw.text((1020, 68), left_label, fill=(20, 20, 20), font=_font(17))
    draw.rectangle([980, 105, 1010, 130], fill=(188, 86, 61, 235))
    draw.text((1020, 103), right_label, fill=(20, 20, 20), font=_font(17))
    draw.text((70, 410), "mean error", fill=(20, 20, 20), font=_font(18))
    img.save(path)


def make_triage_figure(curves: List[Dict[str, float | str]], path: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((80, 35), "Review triage across repeated standard benchmarks", fill=(20, 20, 20), font=_font(28))
    x0, y0 = 180, 130
    plot_w, plot_h = 1120, 650
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    for i in range(1, 5):
        y = y0 + i * plot_h / 5
        draw.line([x0, y, x0 + plot_w, y], fill=(225, 225, 225), width=1)
        x = x0 + i * plot_w / 5
        draw.line([x, y0, x, y0 + plot_h], fill=(225, 225, 225), width=1)

    methods = ["SEF", "confidence", "confidence+SEF", "learned reject"]
    colors = {
        "SEF": (31, 114, 82, 255),
        "confidence": (65, 105, 190, 255),
        "confidence+SEF": (190, 90, 50, 255),
        "learned reject": (120, 75, 150, 255),
    }
    review_rates = sorted({float(row["review_rate"]) for row in curves})
    ymax = max(float(row["accepted_error"]) for row in curves if not np.isnan(float(row["accepted_error"])))
    ymax = max(0.04, ymax * 1.25)
    for m, method in enumerate(methods):
        pts = []
        for rr in review_rates:
            vals = [
                float(row["accepted_error"])
                for row in curves
                if row["method"] == method and abs(float(row["review_rate"]) - rr) < 1e-12
            ]
            avg = float(np.nanmean(vals))
            px = x0 + rr / max(review_rates) * plot_w if max(review_rates) > 0 else x0
            py = y0 + plot_h - avg / ymax * plot_h
            pts.append((px, py))
        draw.line(pts, fill=colors[method], width=4)
        for px, py in pts:
            draw.ellipse([px - 6, py - 6, px + 6, py + 6], fill=colors[method])
        ly = 72 + m * 34
        draw.line([970, ly + 12, 1015, ly + 12], fill=colors[method], width=5)
        draw.text((1025, ly), method, fill=(20, 20, 20), font=_font(16))

    draw.text((530, 825), "fraction sent to review", fill=(20, 20, 20), font=_font(18))
    draw.text((45, 430), "accepted error", fill=(20, 20, 20), font=_font(18))
    img.save(path)


def main() -> None:
    rows: List[Dict[str, float | str]] = []
    curves: List[Dict[str, float | str]] = []
    for name, x, y, _ in benchmark_specs():
        for seed in range(50):
            row, curve = run_split(name, x, y, seed)
            rows.append(row)
            curves.extend(curve)
    summary = summarize(rows)
    save_results(rows, summary, curves)
    make_bar_figure(
        summary,
        FIG_DIR / "figure5_standard_conflict_benchmarks.png",
        "Repeated standard benchmarks: high-conflict cases are riskier",
        "low_conflict_error",
        "high_conflict_error",
        "low conflict",
        "high conflict",
    )
    make_bar_figure(
        summary,
        FIG_DIR / "figure6_standard_res_benchmarks.png",
        "Repeated standard benchmarks: reliable evidence separates risk",
        "high_res_error",
        "low_res_error",
        "high RES",
        "low RES",
    )
    make_triage_figure(curves, FIG_DIR / "figure7_triage_curve.png")
    for row in summary:
        print(row["dataset"], row["test_error"], row["conflict_success_rate"])


if __name__ == "__main__":
    main()
