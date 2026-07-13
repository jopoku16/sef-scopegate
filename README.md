# Signed Evidence Flow

This repository contains code, public-data experiment scripts, generated CSV summaries, and figures for Signed Evidence Flow (SEF).

SEF turns fitted predictions and signed attribution maps into evidence diagnostics:

- supporting evidence,
- opposing evidence,
- evidence conflict,
- perturbation stability,
- reliable evidence scores,
- direction checks for triage use.

No private or proprietary data are included.

## Setup

Create a Python environment and install dependencies:

```text
python -m pip install -r requirements.txt
```

## Run the Code

Run the core experiment suite:

```text
python scripts/run_core_experiments.py
```

Verify mathematical identities used by the implementation:

```text
python scripts/verify_mathematical_invariants.py
```

The scripts write generated CSV files to `results/` and generated images to `figures/`.

## Data Sources

The experiments use public benchmark data from scikit-learn, UCI, and OpenML.

| Study | Source | Identifier/version |
|---|---|---|
| Standard benchmarks | scikit-learn | Iris, Wine, Breast Cancer, Digits |
| Large benchmark | UCI via scikit-learn | Covertype |
| Healthcare-style benchmarks | OpenML | diabetes v1, heart-statlog v1, blood-transfusion-service-center v1 |
| Finance benchmarks | OpenML | credit-g v1, bank-marketing v1, default-of-credit-card-clients v1 |
| Multi-class validation | scikit-learn/UCI | Iris, Wine, Digits, Covertype |
| External replication | OpenML | adult v2, spambase v1, phoneme v1, mammography v1, credit-approval v1, qsar-biodeg v1, pc1 v1, kc1 v1, electricity v1, MagicTelescope v1 |

OpenML scripts record the dataset name and version in their output files.

## Main Scripts

```text
scripts/sef.py
```

Core reusable SEF utilities.

```text
scripts/run_core_experiments.py
```

Runs the main reproducibility suite.

```text
scripts/verify_mathematical_invariants.py
```

Checks algebraic and numerical invariants against the shared implementation.

Additional experiment scripts:

```text
scripts/sef_synthetic_experiment.py
scripts/sef_standard_benchmarks.py
scripts/sef_covtype_benchmark.py
scripts/sef_multiclass_benchmark.py
scripts/sef_model_agnostic_robustness.py
scripts/sef_ablation_study.py
scripts/sef_confidence_masking_stress_test.py
scripts/sef_healthcare_beyond_confidence.py
scripts/sef_entropy_comparison.py
scripts/sef_finance_credit_benchmark.py
scripts/sef_scope_diagnostic.py
scripts/sef_external_replication.py
scripts/sef_conformal_screen_validation.py
scripts/sef_stability_runtime_audit.py
scripts/sef_audit_frontier.py
scripts/sef_identification_stress_test.py
```

## Reproducibility Notes

The scripts use fixed seed ranges:

- standard, selective-baseline, ablation, conformal-screen, healthcare-style, and entropy-comparison runs use seeds `0` through `49`;
- the identification stress test uses deterministic repetitions starting from seed `20260618`;
- model-agnostic robustness uses seeds `0` through `19`;
- finance benchmarks use seeds `0` through `24`;
- ScopeGate uses seeds `0` through `14`, plus seed `20260618` for a paired sign-flip check;
- external replication uses seeds `0` through `9`;
- the shared SEF implementation uses `eps = 1e-12` for numerical stability.

Covertype sampling uses seed `20260617`.

## Outputs

Generated CSV summaries are stored in `results/`.

Generated figures are stored in `figures/`.

This repository intentionally contains only code, public-data outputs, and reproducibility materials.
