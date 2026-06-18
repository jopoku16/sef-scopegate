"""Large real-data benchmark for Signed Evidence Flow.

Uses the Covertype data set through scikit-learn. The full data set has
581,012 rows and 54 features. For speed and reproducibility, the script draws a
large stratified working sample and evaluates SEF reliability triage.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def load_sample(seed: int = 20260617, n_sample: int = 120_000) -> Tuple[np.ndarray, np.ndarray, int]:
    data = fetch_covtype(data_home=DATA_DIR, download_if_missing=True)
    x = data.data.astype(np.float32)
    # Binary task: forest cover type 2 versus all other cover types.
    y = (data.target == 2).astype(int)
    rng = np.random.default_rng(seed)
    idx_pos = np.flatnonzero(y == 1)
    idx_neg = np.flatnonzero(y == 0)
    n_pos = min(len(idx_pos), n_sample // 2)
    n_neg = min(len(idx_neg), n_sample - n_pos)
    sample_idx = np.concatenate([
        rng.choice(idx_pos, size=n_pos, replace=False),
        rng.choice(idx_neg, size=n_neg, replace=False),
    ])
    rng.shuffle(sample_idx)
    return x[sample_idx], y[sample_idx], x.shape[0]


def fit_model(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, SGDClassifier]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=1500,
        tol=1e-4,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train_s, y_train)
    return scaler, model


def signed_evidence(model: SGDClassifier, x_scaled: np.ndarray, baseline_scaled: np.ndarray) -> np.ndarray:
    beta = model.coef_[0]
    return (x_scaled - baseline_scaled) * beta


def sef_scores(evidence: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    s_pos = np.maximum(evidence, 0.0).sum(axis=1)
    s_neg = np.maximum(-evidence, 0.0).sum(axis=1)
    mass = s_pos + s_neg
    conflict = 2 * np.minimum(s_pos, s_neg) / (mass + eps)
    return {
        "support": s_pos,
        "opposition": s_neg,
        "conflict": conflict,
        "direction": (s_pos - s_neg) / (mass + eps),
        "mass": mass,
    }


def perturbation_stability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    evidence: np.ndarray,
    b: int = 25,
    seed: int = 9281,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = x_train.shape[0]
    sign0 = np.sign(evidence)
    agree = np.zeros_like(evidence, dtype=float)
    sub_n = min(n, 35_000)

    for k in range(b):
        idx = rng.choice(n, size=sub_n, replace=True)
        if len(np.unique(y_train[idx])) < 2:
            continue
        scaler_b, model_b = fit_model(x_train[idx], y_train[idx])
        x_train_b_s = scaler_b.transform(x_train[idx])
        x_test_b_s = scaler_b.transform(x_test)
        base_b = x_train_b_s.mean(axis=0)
        ev_b = signed_evidence(model_b, x_test_b_s, base_b)
        agree += (np.sign(ev_b) == sign0).astype(float)

    stab_j = agree / b
    weights = np.abs(evidence)
    return (weights * stab_j).sum(axis=1) / (weights.sum(axis=1) + 1e-12)


def summarize(err: np.ndarray, conflict: np.ndarray, res: np.ndarray) -> Dict[str, float]:
    hi_conf = conflict >= np.quantile(conflict, 0.75)
    lo_conf = conflict <= np.quantile(conflict, 0.25)
    hi_res = res >= np.quantile(res, 0.75)
    lo_res = res <= np.quantile(res, 0.25)
    return {
        "test_error": float(err.mean()),
        "low_conflict_error": float(err[lo_conf].mean()),
        "high_conflict_error": float(err[hi_conf].mean()),
        "high_res_error": float(err[hi_res].mean()),
        "low_res_error": float(err[lo_res].mean()),
        "mean_conflict": float(conflict.mean()),
        "mean_res": float(res.mean()),
    }


def bootstrap_ci(values: np.ndarray, seed: int = 20260617, b: int = 400) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return float("nan"), float("nan")
    boots = np.empty(b, dtype=float)
    for k in range(b):
        idx = rng.integers(0, len(values), size=len(values))
        boots[k] = values[idx].mean()
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def summarize_with_ci(err: np.ndarray, conflict: np.ndarray, res: np.ndarray) -> List[Dict[str, float | str]]:
    groups = {
        "test_error": np.ones_like(err, dtype=bool),
        "low_conflict_error": conflict <= np.quantile(conflict, 0.25),
        "high_conflict_error": conflict >= np.quantile(conflict, 0.75),
        "high_res_error": res >= np.quantile(res, 0.75),
        "low_res_error": res <= np.quantile(res, 0.25),
    }
    rows: List[Dict[str, float | str]] = []
    for k, (metric, mask) in enumerate(groups.items()):
        vals = err[mask]
        lo, hi = bootstrap_ci(vals, seed=20260617 + k)
        rows.append({
            "metric": metric,
            "value": float(vals.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "n": int(mask.sum()),
        })
    return rows


def triage_curve(err: np.ndarray, risk: np.ndarray) -> list[dict[str, float]]:
    rows = []
    n = len(err)
    for review_rate in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
        review_n = int(np.floor(review_rate * n))
        order = np.argsort(-risk)
        review = np.zeros(n, dtype=bool)
        if review_n > 0:
            review[order[:review_n]] = True
        accepted = ~review
        lo, hi = bootstrap_ci(err[accepted], seed=20260617 + int(review_rate * 1000))
        rows.append({
            "review_rate": float(review_rate),
            "accepted_fraction": float(accepted.mean()),
            "accepted_error": float(err[accepted].mean()),
            "accepted_error_ci_low": lo,
            "accepted_error_ci_high": hi,
        })
    return rows


def save_outputs(summary: Dict[str, float], summary_ci: List[Dict[str, float | str]], curve: list[dict[str, float]], n_full: int, n_sample: int) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with (RES_DIR / "sef_covtype_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["full_rows", n_full])
        writer.writerow(["working_sample_rows", n_sample])
        for key, value in summary.items():
            writer.writerow([key, value])

    with (RES_DIR / "sef_covtype_summary_ci.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_ci[0].keys()))
        writer.writeheader()
        writer.writerows(summary_ci)

    with (RES_DIR / "sef_covtype_triage_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    lookup = {row["metric"]: row for row in summary_ci}
    with (RES_DIR / "sef_covtype_table.tex").open("w") as f:
        f.write("\\begin{tabular}{lrr}\n")
        f.write("\\toprule\n")
        f.write("Quantity & Estimate & 95\\% bootstrap CI \\\\\n")
        f.write("\\midrule\n")
        labels = [
            ("test_error", "Overall error"),
            ("low_conflict_error", "Low conflict error"),
            ("high_conflict_error", "High conflict error"),
            ("high_res_error", "High RES error"),
            ("low_res_error", "Low RES error"),
        ]
        for metric, label in labels:
            row = lookup[metric]
            f.write(
                f"{label} & {row['value']:.3f} & "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] \\\\\n"
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


def make_figure(curve: list[dict[str, float]], path: Path) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((80, 35), "Covertype large-data triage with SEF risk", fill=(20, 20, 20), font=_font(28))
    x0, y0 = 180, 130
    plot_w, plot_h = 1120, 650
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    for i in range(1, 5):
        y = y0 + i * plot_h / 5
        x = x0 + i * plot_w / 5
        draw.line([x0, y, x0 + plot_w, y], fill=(225, 225, 225), width=1)
        draw.line([x, y0, x, y0 + plot_h], fill=(225, 225, 225), width=1)
    ymax = max(row["accepted_error"] for row in curve) * 1.25
    ymax = max(ymax, 0.05)
    pts = []
    for row in curve:
        rr = row["review_rate"]
        er = row["accepted_error"]
        px = x0 + rr / 0.5 * plot_w
        py = y0 + plot_h - er / ymax * plot_h
        pts.append((px, py, rr, er))
    draw.line([(p[0], p[1]) for p in pts], fill=(31, 114, 82, 255), width=5)
    for px, py, _, er in pts:
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(31, 114, 82, 255))
        draw.text((px - 24, py - 34), f"{er:.3f}", fill=(20, 20, 20), font=_font(13))
    draw.text((520, 825), "fraction sent to review", fill=(20, 20, 20), font=_font(18))
    draw.text((45, 430), "accepted error", fill=(20, 20, 20), font=_font(18))
    img.save(path)


def main() -> None:
    x, y, n_full = load_sample()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.35, random_state=42, stratify=y
    )
    scaler, model = fit_model(x_train, y_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    base = x_train_s.mean(axis=0)
    evidence = signed_evidence(model, x_test_s, base)
    scores = sef_scores(evidence)
    stability = perturbation_stability(x_train, y_train, x_test, evidence)
    res = (1.0 - scores["conflict"]) * stability
    pred = model.predict(x_test_s)
    err = (pred != y_test).astype(float)
    prob = model.predict_proba(x_test_s)
    _ = log_loss(y_test, prob)
    summary = summarize(err, scores["conflict"], res)
    summary_ci = summarize_with_ci(err, scores["conflict"], res)
    curve = triage_curve(err, risk=1.0 - res)
    save_outputs(summary, summary_ci, curve, n_full=n_full, n_sample=x.shape[0])
    make_figure(curve, FIG_DIR / "figure8_covtype_large_triage.png")


if __name__ == "__main__":
    main()
