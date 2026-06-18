"""Run the core reproducibility suite for the SEF manuscript."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "sef_synthetic_experiment.py",
    "sef_standard_benchmarks.py",
    "sef_covtype_benchmark.py",
    "sef_model_agnostic_robustness.py",
    "sef_ablation_study.py",
    "sef_confidence_masking_stress_test.py",
    "sef_healthcare_beyond_confidence.py",
    "sef_entropy_comparison.py",
    "sef_conformal_screen_validation.py",
    "sef_finance_credit_benchmark.py",
    "sef_multiclass_benchmark.py",
    "sef_scope_diagnostic.py",
    "sef_external_replication.py",
    "sef_stability_runtime_audit.py",
]


def main() -> None:
    for script in SCRIPTS:
        path = ROOT / "scripts" / script
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
