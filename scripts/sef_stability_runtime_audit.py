"""Measure the cost of bootstrap stability relative to one fit-and-score pass."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split

from sef_standard_benchmarks import (
    benchmark_specs,
    bootstrap_stability,
    fit_scaled_logistic,
    signed_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"


def one_pass(x_train, y_train, x_test):
    scaler, model = fit_scaled_logistic(x_train, y_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    evidence = signed_evidence(model, x_test_s, x_train_s.mean(axis=0))
    return evidence


def main() -> None:
    rows = []
    for dataset_index, (name, x, y, _) in enumerate(benchmark_specs()):
        x_train, x_test, y_train, _ = train_test_split(
            x, y, test_size=0.35, random_state=20260618, stratify=y
        )

        base_times = []
        stability_times = []
        for repeat in range(3):
            start = perf_counter()
            evidence = one_pass(x_train, y_train, x_test)
            base_times.append(perf_counter() - start)

            start = perf_counter()
            bootstrap_stability(
                x_train,
                y_train,
                x_test,
                evidence,
                b=40,
                seed=20260618 + 100 * dataset_index + repeat,
            )
            stability_times.append(perf_counter() - start)

        base_seconds = float(np.median(base_times))
        stability_seconds = float(np.median(stability_times))
        rows.append(
            {
                "dataset": name,
                "n": len(x),
                "p": x.shape[1],
                "perturbations": 40,
                "base_fit_score_seconds": base_seconds,
                "stability_seconds": stability_seconds,
                "stability_to_base_ratio": stability_seconds / base_seconds,
            }
        )

    RES_DIR.mkdir(parents=True, exist_ok=True)
    output = RES_DIR / "sef_stability_runtime.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['dataset']}: base={row['base_fit_score_seconds']:.4f}s, "
            f"B=40 stability={row['stability_seconds']:.4f}s, "
            f"ratio={row['stability_to_base_ratio']:.1f}x"
        )


if __name__ == "__main__":
    main()
