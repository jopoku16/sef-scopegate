"""Core utilities for Signed Evidence Flow (SEF).

This small module is intentionally lightweight. It keeps the paper's algorithm
executable without requiring a full package installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass
class SEFScores:
    support: np.ndarray
    opposition: np.ndarray
    mass: np.ndarray
    direction: np.ndarray
    conflict: np.ndarray
    stability: np.ndarray
    reliable_evidence: np.ndarray
    risk: np.ndarray


def evidence_scores(evidence: np.ndarray, stability: np.ndarray | None = None, eps: float = 1e-12) -> SEFScores:
    """Compute SEF support, opposition, conflict, and reliability scores."""
    evidence = np.asarray(evidence, dtype=float)
    s_pos = np.maximum(evidence, 0.0).sum(axis=1)
    s_neg = np.maximum(-evidence, 0.0).sum(axis=1)
    mass = s_pos + s_neg
    direction = (s_pos - s_neg) / (mass + eps)
    conflict = 2.0 * np.minimum(s_pos, s_neg) / (mass + eps)
    if stability is None:
        stability = np.ones(evidence.shape[0], dtype=float)
    reliability = (1.0 - conflict) * stability
    return SEFScores(
        support=s_pos,
        opposition=s_neg,
        mass=mass,
        direction=direction,
        conflict=conflict,
        stability=np.asarray(stability, dtype=float),
        reliable_evidence=reliability,
        risk=1.0 - reliability,
    )


def feature_replacement_evidence(
    score_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Model-agnostic SEF evidence from median reference replacement.

    Parameters
    ----------
    score_fn:
        Function returning a real-valued model score for each row of x.
    x:
        Data matrix to explain.
    reference:
        Reference data matrix used to define baseline feature values.
    """
    x = np.asarray(x)
    ref = np.median(np.asarray(reference), axis=0)
    base_score = score_fn(x)
    evidence = np.zeros_like(x, dtype=float)
    for j in range(x.shape[1]):
        x_rep = x.copy()
        x_rep[:, j] = ref[j]
        evidence[:, j] = base_score - score_fn(x_rep)
    return evidence


def audit_by_budget(error: np.ndarray, risk: np.ndarray, review_rate: float) -> Dict[str, float]:
    """Send the riskiest cases to review and compute accepted-case error."""
    error = np.asarray(error, dtype=float)
    risk = np.asarray(risk, dtype=float)
    n = len(error)
    review_n = int(np.floor(review_rate * n))
    order = np.argsort(-risk)
    review = np.zeros(n, dtype=bool)
    if review_n > 0:
        review[order[:review_n]] = True
    accepted = ~review
    return {
        "review_rate": float(review_rate),
        "accepted_fraction": float(accepted.mean()),
        "accepted_error": float(error[accepted].mean()) if accepted.any() else float("nan"),
    }


def reliability_frontier(error: np.ndarray, risk: np.ndarray, grid: np.ndarray | None = None) -> list[Dict[str, float]]:
    """Compute the Evidence Reliability Frontier for a grid of review rates."""
    if grid is None:
        grid = np.linspace(0.0, 0.5, 11)
    return [audit_by_budget(error, risk, float(review_rate)) for review_rate in grid]

