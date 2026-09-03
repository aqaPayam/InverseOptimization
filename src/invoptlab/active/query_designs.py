"""Small, explicitly query-sensitive MIN scenarios and separate test designs.

Pools are fixed independently of theta*. Test designs are returned separately,
never inserted into the public environment or passed to an algorithm.
"""

from dataclasses import dataclass

import numpy as np

from .config import (ActiveScenarioConfig, DecisionSpaceConfig, ParameterNoiseConfig,
                     QuerySpaceConfig)
from .query_spaces import normalize_rows


@dataclass
class QueryDesign:
    scenario: ActiveScenarioConfig
    test_queries: np.ndarray
    candidate_groups: np.ndarray
    description: str


def build_query_sensitive_scenarios(*, seed: int = 0, horizon: int = 12,
                                   parameter_sigma: float = 0.02) -> list[QueryDesign]:
    """2D threshold, 2D rare-information, 4D missing-coordinate coverage.

    Evaluation is uniform over fresh query ratios (and over pairs in 4D), not
    over the deliberately imbalanced training candidate pool. No oracle places
    candidates near a hidden true threshold. All scenarios use cardinality one.
    """
    def pair(q):
        return normalize_rows(np.column_stack([q, np.ones(len(q))]))

    def midpoints(low, high, n):
        return low + (np.arange(n) + 0.5) / n * (high - low)

    def scenario(name, theta, queries):
        return ActiveScenarioConfig(
            name=name, dimension=len(theta), horizon=horizon, seed=seed, true_theta=theta,
            decision_space=DecisionSpaceConfig(kind="fixed_cardinality", cardinality=1),
            query_space=QuerySpaceConfig(kind="explicit", candidates=queries),
            parameter_noise=ParameterNoiseConfig(kind="isotropic", sigma=parameter_sigma),
        )

    threshold = pair(np.linspace(0.25, 2.25, 128))
    designs = [QueryDesign(
        scenario("query-threshold-2d", [1., 1.1], threshold),
        pair(midpoints(0.25, 2.25, 96)), np.zeros(128, dtype=int),
        "Localize the noisy ratio threshold; fresh ratios cover the entire range.",
    )]
    rng = np.random.default_rng(17)  # Same pools for every experimental seed.
    ratios = np.r_[np.linspace(0.15, 0.35, 380), np.linspace(0.5, 1.75, 20)]
    groups = np.r_[np.zeros(380, dtype=int), np.ones(20, dtype=int)]
    order = rng.permutation(len(ratios))
    designs.append(QueryDesign(
        scenario("query-rare-2d", [1., 1.1], pair(ratios[order])),
        pair(midpoints(0.5, 1.75, 96)), groups[order],
        "95% redundant low ratios; evaluate new ratios in the informative region.",
    ))
    blocks, labels, test = [], [], []
    for j, count in ((1, 288), (2, 16), (3, 16)):
        block = np.zeros((count, 4))
        block[:, 0], block[:, j] = -np.linspace(0.25, 2.25, count), -1.
        blocks.append(normalize_rows(block))
        labels.extend([j] * count)
        heldout = np.zeros((32, 4))
        heldout[:, 0], heldout[:, j] = -midpoints(0.25, 2.25, 32), -1.
        test.append(normalize_rows(heldout))
    queries = np.vstack(blocks)
    order = rng.permutation(len(queries))
    designs.append(QueryDesign(
        scenario("query-coverage-4d", [1., 0.7, 1.3, 1.7], queries[order]),
        np.vstack(test), np.asarray(labels)[order],
        "90/5/5% pair coverage; all three ratios share coordinate 1; balanced new tests.",
    ))
    return designs
