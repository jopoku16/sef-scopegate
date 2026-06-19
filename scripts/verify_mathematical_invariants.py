"""Numerically verify the main SEF identities against the released code."""

from __future__ import annotations

import numpy as np

from sef import evidence_scores


SEED = 20260619
ABS_TOL = 1e-10
REL_TOL = 1e-10


def assert_close(name: str, left: np.ndarray, right: np.ndarray) -> None:
    left = np.asarray(left)
    right = np.asarray(right)
    error = float(np.max(np.abs(left - right)))
    if not np.allclose(left, right, atol=ABS_TOL, rtol=REL_TOL):
        raise AssertionError(f"{name} failed: maximum error {error:.3e}")
    print(f"PASS {name}: maximum absolute error {error:.3e}")


def verify_pointwise_identities(rng: np.random.Generator) -> None:
    evidence = rng.normal(size=(10000, 16))
    scores = evidence_scores(evidence, eps=0.0)
    net = evidence.sum(axis=1)

    assert_close(
        "conflict-direction identity",
        scores.conflict,
        1.0 - np.abs(scores.direction),
    )
    assert_close(
        "hidden evidence mass identity",
        scores.mass,
        np.abs(net) / (1.0 - scores.conflict),
    )

    positive = np.maximum(evidence, 0.0).sum(axis=1)
    negative = np.maximum(-evidence, 0.0).sum(axis=1)
    cone_distance = np.minimum(positive, negative)
    assert_close(
        "distance-to-one-sided-cones identity",
        scores.conflict,
        2.0 * cone_distance / scores.mass,
    )


def verify_perturbation_radius(rng: np.random.Generator) -> None:
    evidence = rng.normal(size=(5000, 12))
    net = evidence.sum(axis=1)
    keep = np.abs(net) > 1e-5
    evidence = evidence[keep]
    net = net[keep]

    delta = rng.normal(size=evidence.shape)
    l1 = np.abs(delta).sum(axis=1)
    scale = 0.95 * np.abs(net) / l1
    perturbed = evidence + delta * scale[:, None]
    if not np.all(np.sign(perturbed.sum(axis=1)) == np.sign(net)):
        raise AssertionError("perturbation-radius guarantee failed")
    print("PASS perturbation-radius guarantee")


def verify_logistic_identification(rng: np.random.Generator) -> None:
    abs_net = rng.choice(np.linspace(0.2, 3.0, 15), size=10000)
    confidence = 1.0 / (1.0 + np.exp(-abs_net))
    mass = abs_net + 0.4 + 0.1 * abs_net
    conflict = 1.0 - abs_net / mass

    for value in np.unique(confidence):
        within = conflict[confidence == value]
        if np.ptp(within) > ABS_TOL:
            raise AssertionError("logistic identification corollary failed")
    print("PASS logistic identification corollary")


def verify_conditional_risk_identity(rng: np.random.Generator) -> None:
    z = rng.integers(0, 5, size=20000)
    c = rng.integers(0, 4, size=20000)
    probability = 0.05 + 0.07 * z + 0.06 * c
    loss = rng.binomial(1, np.clip(probability, 0.01, 0.95)).astype(float)

    mu0 = np.zeros_like(loss)
    mu1 = np.zeros_like(loss)
    for z_value in np.unique(z):
        z_mask = z == z_value
        mu0[z_mask] = loss[z_mask].mean()
        for c_value in np.unique(c):
            cell = z_mask & (c == c_value)
            mu1[cell] = loss[cell].mean()

    risk_gain = np.mean((loss - mu0) ** 2) - np.mean((loss - mu1) ** 2)
    projection_gain = np.mean((mu1 - mu0) ** 2)
    assert_close(
        "conditional-risk projection identity",
        np.array([risk_gain]),
        np.array([projection_gain]),
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    verify_pointwise_identities(rng)
    verify_perturbation_radius(rng)
    verify_logistic_identification(rng)
    verify_conditional_risk_identity(rng)
    print("All mathematical invariant checks passed.")


if __name__ == "__main__":
    main()
