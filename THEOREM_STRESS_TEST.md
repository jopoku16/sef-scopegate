# Theorem Stress Test

This note records the current mathematical stress test for the manuscript.

## What Was Tightened

- The paper now states the mathematical scope before the main definitions.
- The algebraic results are described as pointwise results for a fixed model, reference distribution, and audited point.
- The stability concentration theorems now state the needed independence condition for perturbation indicators.
- The conformal result is tied to the standard exchangeability assumption.
- The paper now says clearly that SEF is not a causal method unless extra causal assumptions are added.
- The perturbation evidence correction was fixed so that the residual is added evenly across features when a complete decomposition is required.
- The paper now includes a direct comparison with attribution entropy and Gini-style attribution spread, so the novelty claim no longer rests only on verbal positioning.
- The earlier one-split zero-error language was softened and marked as illustrative rather than treated as core evidence.

## Proof Check

- Conflict-direction identity: correct. It follows from `a + b = |a - b| + 2 min(a, b)`.
- Hidden evidence mass corollary: correct when `epsilon = 0`, `M(x) > 0`, and `N(x) != 0`.
- Flip margin theorem: correct under the paper's definition of moving signed evidence from the winning side to the losing side.
- Feature-level stability concentration: correct under independent perturbation indicators.
- Uniform stability concentration: correct by union bound over features.
- Nested acceptance sets: correct up to quantile ties.
- Conformal screening guarantee: correct under exchangeability of calibration and test scores.

## Remaining Reviewer Risks

- SEF depends on the chosen attribution or evidence construction.
- Feature replacement can be weak when features are highly correlated.
- Confidence is still a strong triage baseline in clean benchmark tasks.
- The current experiments are mainly tabular.
- On blood-transfusion data, low confidence is a stronger ranking score than SEF conflict. The manuscript now states this honestly.

The manuscript now states these risks directly, which should make the paper stronger rather than weaker.
