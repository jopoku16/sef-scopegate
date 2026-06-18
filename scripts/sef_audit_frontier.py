"""Create Evidence Reliability Frontier outputs for Signed Evidence Flow."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def average_standard_frontier() -> List[Dict[str, float | str]]:
    rows = read_csv(RES_DIR / "sef_triage_curves.csv")
    out = []
    rates = sorted({float(row["review_rate"]) for row in rows})
    for rate in rates:
        vals = [
            float(row["accepted_error"])
            for row in rows
            if row["method"] == "SEF" and abs(float(row["review_rate"]) - rate) < 1e-12
        ]
        out.append({"source": "standard", "review_rate": rate, "accepted_error": float(np.mean(vals))})
    return out


def covtype_frontier() -> List[Dict[str, float | str]]:
    rows = read_csv(RES_DIR / "sef_covtype_triage_curve.csv")
    return [
        {
            "source": "covertype",
            "review_rate": float(row["review_rate"]),
            "accepted_error": float(row["accepted_error"]),
        }
        for row in rows
    ]


def model_agnostic_frontier() -> List[Dict[str, float | str]]:
    rows = read_csv(RES_DIR / "sef_model_agnostic_robustness_runs.csv")
    mapping = {
        0.00: "test_error",
        0.10: "sef_triage_error_10",
        0.20: "sef_triage_error_20",
        0.30: "sef_triage_error_30",
    }
    out = []
    for rate, key in mapping.items():
        vals = [float(row[key]) for row in rows]
        out.append({"source": "black-box", "review_rate": rate, "accepted_error": float(np.mean(vals))})
    return out


def write_summary(rows: List[Dict[str, float | str]]) -> None:
    path = RES_DIR / "sef_audit_frontier_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "review_rate", "accepted_error"])
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int = 18) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_frontier_figure(rows: List[Dict[str, float | str]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 1500, 950
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    draw.text((70, 35), "Evidence Reliability Frontier for SEF-Audit", fill=(20, 20, 20), font=_font(30))

    x0, y0, plot_w, plot_h = 180, 130, 1120, 650
    draw.rectangle([x0, y0, x0 + plot_w, y0 + plot_h], outline=(55, 55, 55), width=2)
    for i in range(1, 5):
        y = y0 + i * plot_h / 5
        x = x0 + i * plot_w / 5
        draw.line([x0, y, x0 + plot_w, y], fill=(225, 225, 225), width=1)
        draw.line([x, y0, x, y0 + plot_h], fill=(225, 225, 225), width=1)

    colors = {
        "standard": (31, 114, 82, 255),
        "covertype": (188, 86, 61, 255),
        "black-box": (42, 113, 142, 255),
    }
    labels = {
        "standard": "standard datasets",
        "covertype": "large Covertype",
        "black-box": "black-box robustness",
    }
    ymax = max(float(row["accepted_error"]) for row in rows) * 1.18
    ymax = max(ymax, 0.05)
    xmax = max(float(row["review_rate"]) for row in rows)

    for source in ["standard", "covertype", "black-box"]:
        source_rows = sorted([row for row in rows if row["source"] == source], key=lambda r: float(r["review_rate"]))
        pts = []
        for row in source_rows:
            rr = float(row["review_rate"])
            er = float(row["accepted_error"])
            px = x0 + rr / xmax * plot_w
            py = y0 + plot_h - er / ymax * plot_h
            pts.append((px, py, rr, er))
        if len(pts) > 1:
            draw.line([(p[0], p[1]) for p in pts], fill=colors[source], width=5)
        for px, py, _, er in pts:
            draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=colors[source])
        ly = 92 + 34 * list(colors.keys()).index(source)
        draw.rectangle([980, ly, 1010, ly + 20], fill=colors[source])
        draw.text((1020, ly - 3), labels[source], fill=(20, 20, 20), font=_font(17))

    draw.text((525, 825), "fraction sent to review", fill=(20, 20, 20), font=_font(19))
    draw.text((43, 420), "accepted error", fill=(20, 20, 20), font=_font(19))
    img.save(FIG_DIR / "figure10_evidence_reliability_frontier.png")


def main() -> None:
    rows = average_standard_frontier() + covtype_frontier() + model_agnostic_frontier()
    write_summary(rows)
    make_frontier_figure(rows)


if __name__ == "__main__":
    main()
