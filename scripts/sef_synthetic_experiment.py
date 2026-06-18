"""Synthetic experiments for Signed Evidence Flow.

The script creates a data set where one feature group supports the positive
class and another feature group opposes it. A logistic model can be confident
even when both groups are large and fighting. SEF exposes that conflict.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def make_data(n: int = 2500, seed: int = 20260617) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_support = rng.normal(0.0, 1.0, size=n)
    x_oppose = rng.normal(0.0, 1.0, size=n)
    x_noise = rng.normal(0.0, 1.0, size=(n, 4))

    # The two leading variables point in opposite directions. Some cases are
    # confident because both signals are strong but one wins slightly.
    logit = 2.2 * x_support - 2.0 * x_oppose + 0.25 * x_noise[:, 0]
    prob = sigmoid(logit)
    y = rng.binomial(1, prob)
    x = np.column_stack([x_support, x_oppose, x_noise])
    beta = np.array([2.2, -2.0, 0.25, 0.0, 0.0, 0.0])
    return x, y, beta


def signed_evidence(x: np.ndarray, beta: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Exact signed evidence for a linear logit model."""
    return (x - baseline) * beta


def sef_scores(evidence: np.ndarray, eps: float = 1e-12) -> Dict[str, np.ndarray]:
    pos = np.maximum(evidence, 0.0)
    neg = np.maximum(-evidence, 0.0)
    s_pos = pos.sum(axis=1)
    s_neg = neg.sum(axis=1)
    mass = s_pos + s_neg
    direction = (s_pos - s_neg) / (mass + eps)
    conflict = 2.0 * np.minimum(s_pos, s_neg) / (mass + eps)
    flip_margin = np.abs(s_pos - s_neg) / 2.0
    return {
        "support": s_pos,
        "opposition": s_neg,
        "mass": mass,
        "direction": direction,
        "conflict": conflict,
        "flip_margin": flip_margin,
    }


def estimate_stability(
    x: np.ndarray,
    beta: np.ndarray,
    baseline: np.ndarray,
    evidence: np.ndarray,
    b: int = 150,
    noise_scale: float = 0.20,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, p = x.shape
    sign0 = np.sign(evidence)
    agree = np.zeros((n, p), dtype=float)

    for _ in range(b):
        beta_b = beta + rng.normal(0.0, noise_scale, size=p)
        base_b = baseline + rng.normal(0.0, 0.05, size=p)
        ev_b = signed_evidence(x, beta_b, base_b)
        agree += (np.sign(ev_b) == sign0).astype(float)

    stab_j = agree / b
    weights = np.abs(evidence)
    overall = (weights * stab_j).sum(axis=1) / (weights.sum(axis=1) + 1e-12)
    return stab_j, overall


def calibration_summary(conflict: np.ndarray, stability: np.ndarray, y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    res = (1.0 - conflict) * stability
    err = (pred != y).astype(float)

    hi_conf = conflict >= np.quantile(conflict, 0.75)
    lo_conf = conflict <= np.quantile(conflict, 0.25)
    hi_res = res >= np.quantile(res, 0.75)
    lo_res = res <= np.quantile(res, 0.25)

    return {
        "mean_error": float(err.mean()),
        "high_conflict_error": float(err[hi_conf].mean()),
        "low_conflict_error": float(err[lo_conf].mean()),
        "high_res_error": float(err[hi_res].mean()),
        "low_res_error": float(err[lo_res].mean()),
        "mean_conflict": float(conflict.mean()),
        "mean_stability": float(stability.mean()),
        "mean_res": float(res.mean()),
    }


def save_summary(summary: Dict[str, float]) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    with (RES_DIR / "sef_synthetic_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def make_figures(
    x: np.ndarray,
    prob: np.ndarray,
    y: np.ndarray,
    evidence: np.ndarray,
    scores: Dict[str, np.ndarray],
    stability: np.ndarray,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = (1.0 - scores["conflict"]) * stability

    confidence = np.maximum(prob, 1 - prob)
    _scatter_plot(
        scores["support"],
        scores["opposition"],
        scores["conflict"],
        FIG_DIR / "figure1_evidence_flow_map.png",
        "SEF evidence map: support, opposition, and conflict",
        "supporting evidence S+(x)",
        "opposing evidence S-(x)",
        color_label="conflict C(x)",
        diagonal=True,
    )
    _scatter_plot(
        confidence,
        scores["conflict"],
        res,
        FIG_DIR / "figure2_confidence_vs_conflict.png",
        "High model confidence can still hide evidence conflict",
        "model confidence",
        "SEF conflict C(x)",
        color_label="reliable evidence score",
    )

    bins = np.linspace(0, 1, 11)
    idx = np.digitize(scores["conflict"], bins) - 1
    centers, errors = [], []
    pred = (prob >= 0.5).astype(int)
    for k in range(len(bins) - 1):
        mask = idx == k
        if mask.sum() >= 20:
            centers.append((bins[k] + bins[k + 1]) / 2)
            errors.append(float(np.mean(pred[mask] != y[mask])))
    _line_plot(
        np.array(centers),
        np.array(errors),
        FIG_DIR / "figure3_conflict_error_curve.png",
        "Prediction error rises when evidence becomes conflicted",
        "SEF conflict bin",
        "classification error",
    )

    example_idx = np.argsort(scores["conflict"])[-1]
    labels = ["support feature", "opposing feature", "noise 1", "noise 2", "noise 3", "noise 4"]
    _barh_plot(
        labels,
        evidence[example_idx],
        FIG_DIR / "figure4_single_case_signed_evidence.png",
        "One conflicted case: evidence pushes both ways",
        "signed evidence E_j(x)",
    )


def _font(size: int = 18) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * max(0.0, min(1.0, t)))


def _cmap(t: float) -> Tuple[int, int, int]:
    # Blue-green-yellow palette, readable on white.
    if t < 0.5:
        u = t / 0.5
        return (_lerp(36, 54, u), _lerp(95, 158, u), _lerp(140, 105, u))
    u = (t - 0.5) / 0.5
    return (_lerp(54, 230, u), _lerp(158, 190, u), _lerp(105, 70, u))


def _draw_axes(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str, xlabel: str, ylabel: str) -> None:
    left, top, right, bottom = box
    draw.rectangle([left, top, right, bottom], outline=(60, 60, 60), width=2)
    f_title = _font(26)
    f_label = _font(18)
    draw.text((left, 24), title, fill=(20, 20, 20), font=f_title)
    draw.text(((left + right) // 2 - 110, bottom + 36), xlabel, fill=(30, 30, 30), font=f_label)
    draw.text((left, top - 34), ylabel, fill=(30, 30, 30), font=f_label)
    for i in range(1, 5):
        x = left + i * (right - left) // 5
        y = top + i * (bottom - top) // 5
        draw.line([x, top, x, bottom], fill=(225, 225, 225), width=1)
        draw.line([left, y, right, y], fill=(225, 225, 225), width=1)


def _scale(v: np.ndarray, lo: float, hi: float, a: int, b: int) -> np.ndarray:
    if hi <= lo:
        return np.full_like(v, (a + b) / 2, dtype=float)
    return a + (v - lo) * (b - a) / (hi - lo)


def _scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    color_label: str,
    diagonal: bool = False,
) -> None:
    width, height = 1400, 1000
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    box = (130, 105, 1125, 835)
    _draw_axes(draw, box, title, xlabel, ylabel)
    left, top, right, bottom = box
    xlo, xhi = float(np.min(x)), float(np.max(x))
    ylo, yhi = float(np.min(y)), float(np.max(y))
    padx = 0.05 * (xhi - xlo + 1e-9)
    pady = 0.05 * (yhi - ylo + 1e-9)
    xlo, xhi = xlo - padx, xhi + padx
    ylo, yhi = ylo - pady, yhi + pady
    xs = _scale(x, xlo, xhi, left, right)
    ys = _scale(y, ylo, yhi, bottom, top)
    clo, chi = float(np.min(c)), float(np.max(c))
    if diagonal:
        limlo = min(xlo, ylo)
        limhi = max(xhi, yhi)
        x1 = float(_scale(np.array([limlo]), xlo, xhi, left, right)[0])
        x2 = float(_scale(np.array([limhi]), xlo, xhi, left, right)[0])
        y1 = float(_scale(np.array([limlo]), ylo, yhi, bottom, top)[0])
        y2 = float(_scale(np.array([limhi]), ylo, yhi, bottom, top)[0])
        draw.line([x1, y1, x2, y2], fill=(0, 0, 0, 120), width=3)
    order = np.linspace(0, len(xs) - 1, min(len(xs), 1300), dtype=int)
    for i in order:
        t = (float(c[i]) - clo) / (chi - clo + 1e-12)
        col = _cmap(t) + (165,)
        draw.ellipse([xs[i] - 5, ys[i] - 5, xs[i] + 5, ys[i] + 5], fill=col)
    _draw_colorbar(draw, (1190, 165, 1235, 775), color_label)
    img.save(path)


def _draw_colorbar(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], label: str) -> None:
    left, top, right, bottom = box
    steps = bottom - top
    for i in range(steps):
        t = 1 - i / max(1, steps - 1)
        draw.line([left, top + i, right, top + i], fill=_cmap(t) + (255,), width=1)
    draw.rectangle([left, top, right, bottom], outline=(50, 50, 50), width=1)
    draw.text((left - 25, bottom + 18), "low", fill=(30, 30, 30), font=_font(15))
    draw.text((left - 25, top - 28), "high", fill=(30, 30, 30), font=_font(15))
    draw.text((left - 25, top - 60), label, fill=(30, 30, 30), font=_font(16))


def _line_plot(x: np.ndarray, y: np.ndarray, path: Path, title: str, xlabel: str, ylabel: str) -> None:
    width, height = 1400, 1000
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    box = (130, 105, 1245, 835)
    _draw_axes(draw, box, title, xlabel, ylabel)
    left, top, right, bottom = box
    xs = _scale(x, 0, 1, left, right)
    ymax = max(0.01, float(np.max(y)) * 1.2)
    ys = _scale(y, 0, ymax, bottom, top)
    pts = list(zip(xs.tolist(), ys.tolist()))
    if len(pts) > 1:
        draw.line(pts, fill=(12, 87, 130, 255), width=5)
    for px, py in pts:
        draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(12, 87, 130, 255))
    img.save(path)


def _barh_plot(labels: list[str], values: np.ndarray, path: Path, title: str, xlabel: str) -> None:
    width, height = 1400, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    f_title = _font(28)
    f_label = _font(18)
    draw.text((70, 35), title, fill=(20, 20, 20), font=f_title)
    left, mid, right = 380, 720, 1240
    top = 135
    row_h = 95
    vmax = max(0.1, float(np.max(np.abs(values))))
    draw.line([mid, top - 35, mid, top + row_h * len(labels)], fill=(0, 0, 0, 180), width=3)
    for k, (lab, val) in enumerate(zip(labels, values)):
        y = top + k * row_h
        draw.text((70, y + 8), lab, fill=(30, 30, 30), font=f_label)
        length = int(abs(float(val)) / vmax * (right - mid - 20))
        if val >= 0:
            draw.rectangle([mid, y, mid + length, y + 38], fill=(44, 122, 63, 230))
        else:
            draw.rectangle([mid - length, y, mid, y + 38], fill=(178, 59, 59, 230))
        draw.text((mid + length + 12 if val >= 0 else mid - length - 85, y + 4), f"{val:.2f}", fill=(20, 20, 20), font=f_label)
    draw.text((550, 820), xlabel, fill=(30, 30, 30), font=f_label)
    img.save(path)


def main() -> None:
    x, y, beta = make_data()
    baseline = x.mean(axis=0)
    evidence = signed_evidence(x, beta, baseline)
    scores = sef_scores(evidence)
    _, stability = estimate_stability(x, beta, baseline, evidence)
    logits = (x - baseline) @ beta
    prob = sigmoid(logits)
    pred = (prob >= 0.5).astype(int)
    summary = calibration_summary(scores["conflict"], stability, y, pred)
    save_summary(summary)
    make_figures(x, prob, y, evidence, scores, stability)


if __name__ == "__main__":
    main()
