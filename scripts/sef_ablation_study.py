"""Ablation study for Signed Evidence Flow.

Compares risk screens built from conflict, stability, RES, confidence, and a
combined confidence+SEF risk score.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


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


def fit_model(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    model = LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")
    model.fit(x_train_s, y_train)
    return scaler, model


def signed_evidence(model: LogisticRegression, x_scaled: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (x_scaled - baseline) * model.coef_[0]


def sef_components(evidence: np.ndarray) -> Dict[str, np.ndarray]:
    s_pos = np.maximum(evidence, 0.0).sum(axis=1)
    s_neg = np.maximum(-evidence, 0.0).sum(axis=1)
    mass = s_pos + s_neg
    conflict = 2 * np.minimum(s_pos, s_neg) / (mass + 1e-12)
    return {"conflict": conflict}


def bootstrap_stability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    evidence: np.ndarray,
    b: int = 60,
    seed: int = 20260617,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    sign0 = np.sign(evidence)
    agree = np.zeros_like(evidence, dtype=float)
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_train[idx])) < 2:
            continue
        scaler_b, model_b = fit_model(x_train[idx], y_train[idx])
        train_s = scaler_b.transform(x_train[idx])
        test_s = scaler_b.transform(x_test)
        ev_b = signed_evidence(model_b, test_s, train_s.mean(axis=0))
        agree += (np.sign(ev_b) == sign0).astype(float)
    stab_j = agree / b
    weights = np.abs(evidence)
    return (weights * stab_j).sum(axis=1) / (weights.sum(axis=1) + 1e-12)


def accepted_error(err: np.ndarray, risk: np.ndarray, review_rate: float = 0.30) -> float:
    n = len(err)
    review_n = int(np.floor(review_rate * n))
    order = np.argsort(-risk)
    review = np.zeros(n, dtype=bool)
    if review_n > 0:
        review[order[:review_n]] = True
    return float(err[~review].mean())


def run_dataset(name: str, x: np.ndarray, y: np.ndarray, seed: int = 42) -> List[Dict[str, float | str]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=seed, stratify=y
    )
    scaler, model = fit_model(x_train, y_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    evidence = signed_evidence(model, x_test_s, x_train_s.mean(axis=0))
    conflict = sef_components(evidence)["conflict"]
    stability = bootstrap_stability(x_train, y_train, x_test, evidence, seed=seed + 17)
    res = (1.0 - conflict) * stability
    prob = model.predict_proba(x_test_s)[:, 1]
    confidence = np.maximum(prob, 1.0 - prob)
    pred = model.predict(x_test_s)
    err = (pred != y_test).astype(float)

    risks = {
        "no review": np.zeros_like(err),
        "conflict only": conflict,
        "stability only": 1.0 - stability,
        "RES": 1.0 - res,
        "confidence": 1.0 - confidence,
        "confidence + RES": 0.5 * (1.0 - confidence) + 0.5 * (1.0 - res),
    }
    rows = []
    for method, risk in risks.items():
        rows.append(
            {
                "dataset": name,
                "seed": seed,
                "method": method,
                "review_rate": 0.30,
                "accepted_error": float(err.mean()) if method == "no review" else accepted_error(err, risk),
            }
        )
    return rows


def aggregate(rows: List[Dict[str, float | str]]) -> List[Dict[str, float | str]]:
    methods = ["no review", "conflict only", "stability only", "RES", "confidence", "confidence + RES"]
    out = []
    for method in methods:
        vals = np.array([float(row["accepted_error"]) for row in rows if row["method"] == method])
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out.append({
            "method": method,
            "accepted_error": float(vals.mean()),
            "accepted_error_se": se,
            "accepted_error_ci": 1.96 * se,
            "runs": len(vals),
        })
    return out


def write_outputs(rows: List[Dict[str, float | str]], agg: List[Dict[str, float | str]]) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with (RES_DIR / "sef_ablation_runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (RES_DIR / "sef_ablation_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)
    with (RES_DIR / "sef_ablation_table.tex").open("w") as f:
        f.write("\\begin{tabular}{lr}\n")
        f.write("\\toprule\n")
        f.write("Risk score & Accepted error \\\\\n")
        f.write("\\midrule\n")
        for row in agg:
            f.write(
                f"{row['method']} & "
                f"{row['accepted_error']:.3f} $\\pm$ {row['accepted_error_ci']:.3f} \\\\\n"
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


def make_figure(agg: List[Dict[str, float | str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((70, 35), "SEF ablation: accepted error after reviewing 30% of cases", fill=(20, 20, 20), font=_font(28))
    x0, y0, plot_w, plot_h = 150, 130, 1200, 640
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    vals = [float(row["accepted_error"]) for row in agg]
    ymax = max(vals) * 1.25
    colors = [
        (90, 90, 90, 230),
        (188, 86, 61, 230),
        (184, 113, 38, 230),
        (31, 114, 82, 230),
        (42, 113, 142, 230),
        (92, 84, 150, 230),
    ]
    bar_w = plot_w / len(agg) * 0.56
    for k, (row, color) in enumerate(zip(agg, colors)):
        val = float(row["accepted_error"])
        cx = x0 + (k + 0.5) * plot_w / len(agg)
        h = val / ymax * plot_h
        draw.rectangle([cx - bar_w / 2, y0 + plot_h - h, cx + bar_w / 2, y0 + plot_h], fill=color)
        draw.text((cx - 30, y0 + plot_h - h - 30), f"{val:.3f}", fill=(20, 20, 20), font=_font(15))
        label = str(row["method"])
        if len(label) > 13:
            parts = label.split(" ")
            label = " ".join(parts[:1]) + "\n" + " ".join(parts[1:])
        draw.text((cx - 58, y0 + plot_h + 18), label, fill=(20, 20, 20), font=_font(15))
    draw.text((45, 415), "accepted error", fill=(20, 20, 20), font=_font(18))
    img.save(FIG_DIR / "figure11_ablation_study.png")


def main() -> None:
    rows: List[Dict[str, float | str]] = []
    for seed in range(50):
        for name, x, y in benchmark_specs():
            rows.extend(run_dataset(name, x, y, seed=seed))
    agg = aggregate(rows)
    write_outputs(rows, agg)
    make_figure(agg)


if __name__ == "__main__":
    main()
