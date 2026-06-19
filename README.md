# Signed Evidence Flow

Working manuscript for Paper 5:

**Signed Evidence Flow: Conflict-Aware and Stability-Calibrated Data Analysis**

The goal is to create a new model-agnostic data-analysis algorithm that turns predictions into signed evidence systems.

## Reproduce the paper

Create a Python environment and install the dependencies:

```text
python -m pip install -r requirements.txt
```

Run the full core experiment suite:

```text
python scripts/run_core_experiments.py
```

Verify the main mathematical identities against the shared implementation:

```text
python scripts/verify_mathematical_invariants.py
```

Compile the manuscript:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The scripts use fixed, visible seed ranges. Repeated standard benchmarks,
ablation experiments, conformal validation, and healthcare studies use seeds
starting at zero. Covertype uses seed `20260617` for sampling and bootstrap
intervals. Every generated table is saved in `results/`, and every generated
figure is saved in `figures/`.

More exactly:

- standard, selective-baseline, ablation, conformal-screen, healthcare, and
  entropy-comparison experiments use seeds `0` through `49`;
- the identification stress test uses 50 deterministic repetitions starting
  from seed `20260618`;
- model-agnostic black-box robustness uses seeds `0` through `19`;
- finance stress tests use seeds `0` through `24`;
- ScopeGate uses seeds `0` through `14`, plus seed `20260618` for the paired
  sign-flip test;
- independent external replication uses seeds `0` through `9`;
- the shared SEF implementation uses `eps = 1e-12` for numerical stability.

## Data sources

| Study | Source | Identifier/version |
|---|---|---|
| Standard benchmarks | scikit-learn | Iris, Wine, Breast Cancer, Digits |
| Large benchmark | UCI via scikit-learn | Covertype |
| Healthcare | OpenML | diabetes v1, heart-statlog v1, blood-transfusion-service-center v1 |
| Finance stress test | OpenML | credit-g v1, bank-marketing v1, default-of-credit-card-clients v1 |
| Multi-class validation | scikit-learn/UCI | Iris, Wine, Digits, Covertype |
| External replication | OpenML | adult v2, spambase v1, phoneme v1, mammography v1, credit-approval v1, qsar-biodeg v1, pc1 v1, kc1 v1, electricity v1, MagicTelescope v1 |

The OpenML scripts record the dataset name and version in their output files.
No private or proprietary data are used.

## Core idea

Most models give a prediction. Some explanation tools show feature contributions.

Signed Evidence Flow goes further:

- What evidence supports the prediction?
- What evidence opposes it?
- How much conflict is inside the evidence?
- Does the evidence survive perturbation?
- Can we calibrate a reliability flag?

## Main mathematical objects

- Signed evidence terms: `E_j(x)`
- Supporting evidence: `S^+(x)`
- Opposing evidence: `S^-(x)`
- Evidence conflict: `C(x)`
- Evidence direction: `D(x)`
- Flip margin: `m_flip(x)`
- Evidence stability: `Stab(x)`
- Reliable evidence score: `RES(x)`

## Included analyses

The latest manuscript version includes:

- a mathematical scope section;
- a practical guidance section;
- a confidence-masked conflict stress test;
- a healthcare benchmark beyond confidence;
- an attribution entropy/Gini comparison;
- an if-and-only-if identification theorem showing exactly when confidence
  determines conflict for a complete, decision-aligned decomposition;
- an exact conditional-risk decomposition stating when conflict adds
  predictive value beyond confidence and other audit variables;
- a negative-control stress test where conflict is fully identified by
  confidence and correctly gives no cross-fitted gain;
- an executable invariant audit for the main algebraic, robustness, and
  conditional-risk identities;
- a held-out permutation gate for positive evidence-risk direction;
- cross-fitted incremental-value tests beyond confidence and attribution entropy;
- a runtime audit and an explicit analysis of the finance reversals;
- a corrected black-box robustness summary that matches the generated table
  and figure;
- a stronger ScopeGate framing that treats finance reversals as a central
  deployment warning rather than a side note;
- an independent ten-data-set external replication suite over 139,325 public
  observations;
- an external SEF-versus-entropy comparison and direction-aware triage
  frontiers on the same independent suite;
- exact seed, epsilon, and data-source details for reproducibility;
- a method-comparison table.

## Current experiment

The first synthetic experiment is in:

```text
scripts/sef_synthetic_experiment.py
```

It creates a data set where one feature group supports the positive class and another feature group opposes it. The experiment shows that high model confidence can still hide high evidence conflict.

Current generated outputs:

```text
figures/figure1_evidence_flow_map.png
figures/figure2_confidence_vs_conflict.png
figures/figure3_conflict_error_curve.png
figures/figure4_single_case_signed_evidence.png
results/sef_synthetic_summary.csv
figures/figure21_identification_stress_test.png
results/sef_identification_stress_test.csv
results/sef_identification_stress_test_summary.csv
```

Headline result from the first run:

```text
overall error: 0.162
low-conflict error: 0.062
high-conflict error: 0.358
high-RES error: 0.043
low-RES error: 0.360
```

## Standard benchmark experiment

The standard benchmark experiment is in:

```text
scripts/sef_standard_benchmarks.py
```

It uses standard scikit-learn datasets:

- Iris: versicolor vs virginica
- Wine: class 1 vs class 2
- Breast Cancer Wisconsin: benign vs malignant
- Digits: 3 vs 5

Generated outputs:

```text
figures/figure5_standard_conflict_benchmarks.png
figures/figure6_standard_res_benchmarks.png
figures/figure7_triage_curve.png
results/sef_standard_benchmarks_summary.csv
results/sef_standard_benchmarks_table.tex
results/sef_triage_curves.csv
```

Headline result:

```text
high-conflict predictions have higher error in all four standard tasks
highest-RES quartile has zero errors in all four standard tasks in this run
reviewing the riskiest 30% by SEF risk lowers accepted-case error from 0.034 to 0.008 on average
```

## Large real-data benchmark

The large benchmark script is in:

```text
scripts/sef_covtype_benchmark.py
```

It uses the Covertype data set through scikit-learn. The full source data has 581,012 rows and 54 features. The current run uses a stratified working sample of 120,000 rows for speed.

Generated outputs:

```text
figures/figure8_covtype_large_triage.png
results/sef_covtype_summary.csv
results/sef_covtype_triage_curve.csv
```

Headline result:

```text
Covertype overall error: 0.249
low-conflict error: 0.096
high-conflict error: 0.409
high-RES error: 0.094
low-RES error: 0.409
reviewing the riskiest 50% by SEF risk lowers accepted-case error from 0.249 to 0.149
```

## Model-agnostic robustness benchmark

The model-agnostic robustness script is in:

```text
scripts/sef_model_agnostic_robustness.py
```

It runs 160 black-box prediction problems:

```text
4 standard datasets x 2 nonlinear models x 20 random splits
```

The two model classes are random forests and histogram gradient boosting. SEF evidence is computed by feature replacement, not by model coefficients.

Generated outputs:

```text
figures/figure9_model_agnostic_robustness.png
figures/figure10_model_agnostic_triage_comparison.png
results/sef_model_agnostic_robustness_runs.csv
results/sef_model_agnostic_robustness_summary.csv
results/sef_model_agnostic_overall_summary.csv
results/sef_model_agnostic_robustness_table.tex
```

Headline result:

```text
overall error: 0.036
low-conflict error: 0.004
high-conflict error: 0.098
high-conflict predictions are about 24x as error-prone as low-conflict predictions, with the ratio inflated by near-zero low-conflict errors
```

## SEF-Audit and Evidence Reliability Frontier

Core reusable SEF utilities are in:

```text
scripts/sef.py
```

The frontier script is in:

```text
scripts/sef_audit_frontier.py
```

Generated outputs:

```text
figures/figure10_evidence_reliability_frontier.png
results/sef_audit_frontier_summary.csv
```

SEF-Audit ranks cases by:

```text
SEF risk = 1 - reliable evidence score
```

and sends the highest-risk cases to review.

## Ablation study

The ablation script is in:

```text
scripts/sef_ablation_study.py
```

It compares:

```text
no review
conflict only
stability only
RES
confidence
confidence + RES
```

Generated outputs:

```text
figures/figure11_ablation_study.png
results/sef_ablation_runs.csv
results/sef_ablation_summary.csv
results/sef_ablation_table.tex
```

Headline result at a 30% review budget:

```text
no-review error: 0.034
conflict-only error: 0.008
RES error: 0.008
confidence + RES error: 0.000
```

## Confidence-masked conflict stress test

The confidence-masked stress-test script is in:

```text
scripts/sef_confidence_masking_stress_test.py
```

It creates a transparent setting where predictions can look confident even when
strong supporting and opposing evidence are both present. This tests the core
SEF claim that confidence and evidence conflict are related but not identical.

Generated outputs:

```text
figures/figure12_confidence_masking_stress_test.png
results/sef_confidence_masking_summary.csv
results/sef_confidence_masking_frontier.csv
results/sef_confidence_masking_table.tex
```

Headline result:

```text
overall error: 0.277
high-confidence error: 0.111
high-confidence, low-conflict error: 0.071
high-confidence, high-conflict error: 0.174
at 30% review inside high-confidence cases:
  SEF-only accepted-case error: 0.088
  confidence-only accepted-case error: 0.074
  confidence + SEF accepted-case error: 0.073
```

## Healthcare benchmark beyond confidence

The healthcare benchmark script is in:

```text
scripts/sef_healthcare_beyond_confidence.py
```

It uses three public OpenML data sets:

```text
Diabetes
Heart disease
Blood transfusion
```

The benchmark asks whether SEF conflict separates risk even after model
confidence has already selected predictions that look safe.

Generated outputs:

```text
figures/figure13_healthcare_beyond_confidence.png
results/sef_healthcare_beyond_confidence_runs.csv
results/sef_healthcare_beyond_confidence_summary.csv
results/sef_healthcare_beyond_confidence_table.tex
```

Headline mean errors inside high-confidence predictions:

```text
Diabetes: low-conflict 0.050 vs high-conflict 0.177
Heart disease: low-conflict 0.010 vs high-conflict 0.143
Blood transfusion: low-conflict 0.162 vs high-conflict 0.244
```

## Attribution entropy comparison

The attribution entropy comparison script is in:

```text
scripts/sef_entropy_comparison.py
```

It compares SEF conflict with:

```text
attribution entropy
attribution Gini-style spread
low confidence
```

The comparison is run inside high-confidence predictions on the same healthcare
benchmark suite.

Generated outputs:

```text
figures/figure14_entropy_comparison.png
results/sef_entropy_comparison_runs.csv
results/sef_entropy_comparison_summary.csv
results/sef_entropy_comparison_table.tex
```

Headline result:

```text
On diabetes and heart disease, SEF conflict ranks risky confident cases better
than attribution entropy and Gini spread. On blood transfusion, confidence is
still the strongest ranking score.
```

## ScopeGate diagnostic

The reviewer-facing scope and conditional-value audit is in:

```text
scripts/sef_scope_diagnostic.py
```

It compares a cross-fitted error model using low confidence and exact linear
attribution entropy with an augmented model that also uses SEF conflict. It also
runs a one-sided permutation diagnostic before high-conflict review is allowed.

Generated outputs:

```text
figures/figure18_scope_diagnostic.png
results/sef_scope_diagnostic_runs.csv
results/sef_scope_diagnostic_summary.csv
results/sef_scope_diagnostic_table.tex
```

The result is deliberately scoped. Conflict adds conditional error-ranking AUC
on Diabetes, German Credit, Bank Marketing, and Credit Card Default. On the two
large finance data sets the direction is negative, so conflict is informative
but high-conflict review is not valid. This distinction is now central to the
paper rather than hidden in the limitations.

## Independent external replication

The independent replication suite is in:

```text
scripts/sef_external_replication.py
```

It runs the ScopeGate-style conditional-value test on ten additional public
OpenML data sets that were not used in the earlier development examples.

Generated outputs:

```text
figures/figure19_external_replication.png
figures/figure20_external_entropy_comparison.png
figures/figure10_evidence_reliability_frontier.png
results/sef_external_replication_runs.csv
results/sef_external_replication_summary.csv
results/sef_external_replication_table.tex
results/sef_external_entropy_table.tex
results/sef_audit_frontier_summary.csv
```

Headline result:

```text
SEF conflict adds conditional AUC on 7 of 10 external data sets.
Electricity is effectively neutral after rounding, although its positive-direction
diagnostic passes in every split.
The largest gains are Magic Telescope (+0.144), Mammography (+0.065),
Adult Income (+0.056), and PC1 defects (+0.035). The positive-direction
review diagnostic passes clearly on Electricity and Magic Telescope.
This supports the scoped claim: conflict often carries information, but
ScopeGate decides whether high-conflict review is safe in the target domain.
The direct entropy comparison is intentionally mixed: SEF ranks errors better
than entropy on several tasks, entropy is better on others, and low confidence
remains the strongest raw score on eight of ten tasks. The independent triage
frontiers show why the direction gate is required before deployment.
```

## Stability runtime audit

The bootstrap cost audit is in:

```text
scripts/sef_stability_runtime_audit.py
```

It writes `results/sef_stability_runtime.csv`. On the four standard tasks,
40 bootstrap refits took 0.188--0.257 seconds, or 30.1--39.2 times one
fit-and-score pass on the test machine. These figures are machine-dependent;
the script is included so users can measure the cost in their own setting.

## Authors

Jeffery Opoku, The University of Texas Rio Grande Valley  
David Banahene, Florida International University
