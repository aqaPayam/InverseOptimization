"""The user-named algorithms and eight predeclared comparison scenarios.

Pedro is the ORIGINAL uniform-query INCENTER method, not a uniform-query
ablation of the loss sampler. The latter remains only in historical notebook 14.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from ..exceptions import CapabilityError, SolverError, ValidationError
from .algorithms import UniformRandomIncenterAlgorithm
from .config import (ActiveScenarioConfig, DecisionSpaceConfig, ParameterNoiseConfig,
                     QuerySpaceConfig)
from .evaluation import ActiveEvaluationConfig, evaluate_active_run
from .langevin import NestedLangevinActiveAlgorithm, NestedLangevinConfig
from .query_spaces import normalize_rows, random_unit
from .runner import ActiveBenchmarkRunner
from .samd import OnlineSAMDConfig, UniformOnlineSAMDAlgorithm
from .stopping import RegretStoppingConfig
from .types import ActiveAction


class PedroAlgorithm(UniformRandomIncenterAlgorithm):
    """Uniform next S; theta_hat is the hard-consistency cone incenter."""

    name = "Pedro algorithm"


class GeniousPedroAlgorithm(UniformRandomIncenterAlgorithm):
    """Pedro's incenter with minimum normalized decision-margin queries.

    The first query is uniform because D_0 contains no constraints. After each
    valid incenter update, every available candidate is scored by the distance
    from the incenter to its nearest predicted decision boundary. No latent
    parameter, test query, probability distribution, or parameter sample is
    used. An invalid incenter triggers the agreed uniform-query fallback while
    the estimate remains explicitly invalid in benchmark evaluation.

    This exact implementation requires a finite, enumerable decision space,
    which covers all eight predeclared Pedro/Score comparison scenarios.
    """

    name = "Genious Pedro"

    def reset(self, context, rng) -> None:
        super().reset(context, rng)
        try:
            decisions = np.asarray(self.decision_problem.enumerate_decisions(), dtype=float)
        except CapabilityError as exc:
            raise ValidationError(
                "Genious Pedro requires a finite enumerable public decision space"
            ) from exc
        if (decisions.ndim != 2 or decisions.shape[0] < 2
                or decisions.shape[1] != context.dimension
                or not np.all(np.isfinite(decisions))):
            raise ValidationError(
                "Genious Pedro requires at least two finite enumerable decisions"
            )
        self._margin_decisions = decisions

    def _candidate_margin(self, query: np.ndarray) -> tuple[float, np.ndarray, np.ndarray | None]:
        """Return distance to the closest informative decision boundary."""

        query = np.asarray(query, dtype=float)
        theta = self._estimate
        predicted = self.decision_problem.minimize(query * theta, self.rng)
        differences = (self._margin_decisions - predicted) * query
        norms = np.linalg.norm(differences, axis=1)
        informative = norms > 1e-12
        if not np.any(informative):
            return float("inf"), predicted, None
        candidate_indices = np.flatnonzero(informative)
        margins = differences[informative] @ theta / norms[informative]
        minimum = float(np.min(margins))
        if minimum < -max(1e-8, 10 * self.tolerance):
            raise SolverError(
                "the public MIN solution has a negative normalized competitor margin"
            )
        margins = np.maximum(margins, 0.0)
        minimum = float(np.min(margins))
        nearest_positions = np.flatnonzero(
            np.isclose(margins, minimum, rtol=1e-10, atol=1e-12)
        )
        nearest_position = int(nearest_positions[0])
        nearest = self._margin_decisions[candidate_indices[nearest_position]].copy()
        return minimum, predicted.copy(), nearest

    def _fallback_action(self, reason: str) -> ActiveAction:
        index = self._select_query_index()
        return ActiveAction(
            query=self.context.query_candidates[index].copy(),
            theta_hat=self._estimate.copy(),
            diagnostics={
                "query_rule": "uniform-random-fallback",
                "fallback_reason": reason,
                "candidate_index": index,
                "selected_margin": None,
                "candidate_margins": None,
                "predicted_decision": None,
                "nearest_alternative": None,
                "margin_tie_candidate_indices": [],
                "constraint_count": int(self.constraints_.shape[0]),
                "incenter_radius": self.incenter_radius_,
                "estimate_status_before_query": self.estimate_status_,
                "all_constraints_exact": all(
                    bool(item["exact"]) for item in self.constraint_sources_
                ),
            },
        )

    def propose(self, history) -> ActiveAction:
        if not history:
            return self._fallback_action("D_0 is empty; the first query is uniform")
        if self.estimate_status_ != "valid":
            return self._fallback_action(
                f"the current incenter estimate is {self.estimate_status_}"
            )

        if self.allow_repeated_queries:
            available = np.arange(self.context.query_candidates.shape[0], dtype=int)
        else:
            if not self._available_queries:
                raise ValidationError("the non-repeating query set has been exhausted")
            available = np.asarray(self._available_queries, dtype=int)

        scores = np.full(self.context.query_candidates.shape[0], np.inf)
        predictions: dict[int, np.ndarray] = {}
        alternatives: dict[int, np.ndarray | None] = {}
        for index in available:
            score, predicted, alternative = self._candidate_margin(
                self.context.query_candidates[index]
            )
            scores[index] = score
            predictions[int(index)] = predicted
            alternatives[int(index)] = alternative

        finite_available = available[np.isfinite(scores[available])]
        if finite_available.size == 0:
            return self._fallback_action(
                "no candidate has a distinct informative feature alternative"
            )
        best = float(np.min(scores[finite_available]))
        tied = finite_available[
            np.isclose(scores[finite_available], best, rtol=1e-10, atol=1e-12)
        ]
        index = int(self.rng.choice(tied))
        if not self.allow_repeated_queries:
            self._available_queries.remove(index)
        serializable_scores = [
            float(value) if np.isfinite(value) else None for value in scores
        ]
        alternative = alternatives[index]
        return ActiveAction(
            query=self.context.query_candidates[index].copy(),
            theta_hat=self._estimate.copy(),
            diagnostics={
                "query_rule": "minimum-normalized-decision-margin",
                "fallback_reason": None,
                "candidate_index": index,
                "selected_margin": best,
                "candidate_margins": serializable_scores,
                "predicted_decision": predictions[index].copy(),
                "nearest_alternative": (
                    None if alternative is None else alternative.copy()
                ),
                "margin_tie_candidate_indices": tied.tolist(),
                "constraint_count": int(self.constraints_.shape[0]),
                "incenter_radius": self.incenter_radius_,
                "estimate_status_before_query": self.estimate_status_,
                "all_constraints_exact": all(
                    bool(item["exact"]) for item in self.constraint_sources_
                ),
            },
        )


class ScoreBaseAlgorithm(NestedLangevinActiveAlgorithm):
    """Disagreement next S; theta_hat is the mean of parameter samples."""

    def __init__(self, config: NestedLangevinConfig | None = None):
        settings = config or NestedLangevinConfig(record_chain_trace=False)
        if settings.point_estimate != "mean" or settings.query_policy != "disagreement":
            raise ValidationError("Score base model requires mean estimation and disagreement queries")
        super().__init__(settings)
        self.name = "Score base model"


@dataclass
class ComparisonDesign:
    family: str
    title: str
    scenario: ActiveScenarioConfig
    test_queries: dict[str, np.ndarray]
    candidate_groups: np.ndarray
    group_labels: list[str]
    description: str


def build_pedro_score_scenarios(*, seed: int = 0, horizon: int = 20) -> list[ComparisonDesign]:
    """Eight 4D/6D noisy MIN cases. Pools and tests never depend on hidden theta.

    120 training candidates and 120 fresh queries per test distribution.
    Five paired seeds vary theta and noise, but not the predeclared geometry.
    Cases 4/7/8 share theta, pool and tests to isolate the noise change.
    """
    titles = ["Connecting two groups", "Several boundaries to locate",
              "Similar queries versus varied queries", "Ordinary balanced choice",
              "Ordinary subset selection", "Small budget-constrained selection",
              "Stronger parameter noise", "Query-dependent parameter noise"]
    families = ["bridge-6d", "boundaries-4d", "redundancy-6d", "balanced-choice-4d",
                "balanced-subset-6d", "knapsack-6d", "strong-noise-4d", "query-noise-4d"]
    descriptions = [
        "90% comparisons within two groups; 10% connect the groups. Balanced test weights all five pairs equally.",
        "Three equally available reference-coordinate comparisons; broad ratios require boundary refinement.",
        "80% queries cluster around two fixed profiles; 20% are diverse. Balanced test weights the three groups equally.",
        "Choose one item under dense uniformly distributed unit queries; no rare-information construction.",
        "Choose exactly three of six items; dense unit queries involve every coordinate.",
        "Choose a binary bundle with weights [1,2,2,3,3,4] and budget 6; dense unit queries.",
        "Same geometry and hidden theta as balanced choice; parameter sigma increases from .02 to .08.",
        "Same geometry and hidden theta as balanced choice; sigma(s)=.02+.08*abs(s[0]).",
    ]
    designs = []
    for index, (family, title) in enumerate(zip(families, titles)):
        base = 3 if index in (6, 7) else index
        d = 4 if base in (1, 3) else 6
        pool_rng = np.random.default_rng(np.random.SeedSequence([271828, base]))
        test_rng = np.random.default_rng(np.random.SeedSequence([314159, base]))
        theta_rng = np.random.default_rng(np.random.SeedSequence([161803, base, seed]))
        theta = theta_rng.uniform(.8, 1.2, size=d)
        if base not in (0, 1):
            theta *= np.array([-1, 1, -1, 1, -1, 1])[:d]
        decision = DecisionSpaceConfig(kind="fixed_cardinality", cardinality=1)
        tests = {}
        if base in (0, 1):
            pairs = ([(0, 1), (1, 2), (3, 4), (4, 5), (2, 3)] if base == 0
                     else [(0, 1), (0, 2), (0, 3)])
            counts = [27, 27, 27, 27, 12] if base == 0 else [40, 40, 40]

            def pair_pool(counts, rng, grid=False):
                blocks, labels = [], []
                for group, ((left, right), n) in enumerate(zip(pairs, counts)):
                    block = np.zeros((n, d))
                    q = np.linspace(.25, 2.25, n) if grid else rng.uniform(.25, 2.25, n)
                    block[:, left], block[:, right] = -q, -1.
                    blocks.append(normalize_rows(block))
                    labels.extend([group] * n)
                order = rng.permutation(sum(counts))
                return np.vstack(blocks)[order], np.asarray(labels)[order]

            candidates, groups = pair_pool(counts, pool_rng, grid=True)
            tests["ordinary"] = pair_pool(counts, test_rng)[0]
            if base == 0:
                tests["balanced"] = pair_pool([24] * 5, test_rng)[0]
            labels = [f"coordinate {a+1} vs {b+1}" for a, b in pairs]
        elif base == 2:
            decision = DecisionSpaceConfig(kind="fixed_cardinality", cardinality=2)
            centers = normalize_rows(np.array([[-1, -.6, -.2, .2, .6, 1],
                                                [.3, -.8, .6, -.5, .9, -.2]]))

            def clustered_pool(counts, rng):
                blocks = [normalize_rows(centers[g] + .08*rng.normal(size=(counts[g], d)))
                          for g in range(2)] + [random_unit(counts[2], d, rng)]
                groups = np.repeat(np.arange(3), counts)
                order = rng.permutation(sum(counts))
                return np.vstack(blocks)[order], groups[order]

            candidates, groups = clustered_pool([48, 48, 24], pool_rng)
            tests["ordinary"] = clustered_pool([48, 48, 24], test_rng)[0]
            tests["balanced"] = clustered_pool([40, 40, 40], test_rng)[0]
            labels = ["profile 1", "profile 2", "diverse"]
        else:
            candidates = random_unit(120, d, pool_rng)
            groups, labels = np.zeros(120, dtype=int), ["dense unit queries"]
            tests["ordinary"] = random_unit(120, d, test_rng)
            if base == 4:
                decision = DecisionSpaceConfig(kind="fixed_cardinality", cardinality=3)
            elif base == 5:
                decision = DecisionSpaceConfig(kind="structured", C_ub=[[1, 2, 2, 3, 3, 4]],
                                               r_ub=[6], max_enumeration=64)
        noise = (ParameterNoiseConfig(kind="query_dependent", query_profile="absolute_first",
                                      minimum_scale=.02, maximum_scale=.10) if index == 7
                 else ParameterNoiseConfig(kind="isotropic", sigma=.08 if index == 6 else .02))
        scenario = ActiveScenarioConfig(name=f"comparison-{family}-s{seed}", dimension=d,
            horizon=horizon, seed=seed, true_theta=theta.tolist(), decision_space=decision,
            query_space=QuerySpaceConfig(kind="explicit", candidates=candidates),
            parameter_noise=noise)
        designs.append(ComparisonDesign(family, title, scenario, tests, groups, labels,
                                        descriptions[index]))
    return designs


def comparison_fingerprint(design: ComparisonDesign, algorithm: str,
                           score_config: NestedLangevinConfig,
                           samd_config: OnlineSAMDConfig | None = None) -> str:
    """Cache identity includes settings, hidden tests and relevant source bytes."""
    payload = {"protocol": 1, "scenario": design.scenario.to_dict(), "algorithm": algorithm,
               "score_config": asdict(score_config),
               "samd_config": asdict(samd_config or OnlineSAMDConfig()),
               "tests": {k: v.tolist() for k, v in design.test_queries.items()}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _run_comparison_design(design: ComparisonDesign, directory: str | Path, *,
                           score_config: NestedLangevinConfig | None,
                           samd_config: OnlineSAMDConfig | None,
                           use_cache: bool, progress,
                           include_genious_pedro: bool,
                           include_online_samd: bool) -> list[dict]:
    cfg = score_config or NestedLangevinConfig(record_chain_trace=False)
    samd_cfg = samd_config or OnlineSAMDConfig()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    methods = [("Pedro algorithm", PedroAlgorithm)]
    if include_genious_pedro:
        methods.append(("Genious Pedro", GeniousPedroAlgorithm))
    methods.append(("Score base model", lambda: ScoreBaseAlgorithm(cfg)))
    if include_online_samd:
        methods.append(("Uniform Online SAMD", lambda: UniformOnlineSAMDAlgorithm(samd_cfg)))
    slugs = {
        "Pedro algorithm": "pedro",
        "Genious Pedro": "genious-pedro",
        "Score base model": "score-base",
        "Uniform Online SAMD": "uniform-online-samd",
    }
    estimators = {
        "Pedro algorithm": "incenter",
        "Genious Pedro": "incenter",
        "Score base model": "mean of parameter samples",
        "Uniform Online SAMD": "online signed exponentiated ASL",
    }
    for label, constructor in methods:
        fingerprint = comparison_fingerprint(design, label, cfg, samd_cfg)
        slug = slugs[label]
        destination = directory / f"{design.family}-seed{design.scenario.seed}-{slug}.json"
        if use_cache and destination.exists():
            cached = json.loads(destination.read_text(encoding="utf-8"))
            if (cached.get("metadata", {}).get("comparison_fingerprint") == fingerprint
                    and len(cached.get("records", [])) == design.scenario.horizon):
                progress(f"CACHED {design.family} seed={design.scenario.seed} {label}")
                results.append(cached)
                continue
        progress(f"START {design.family} seed={design.scenario.seed} {label}")
        algorithm = constructor()
        run = ActiveBenchmarkRunner(stopping_config=RegretStoppingConfig(enabled=False)).run(
            design.scenario, algorithm, algorithm_seed=design.scenario.seed)
        evaluations = {name: evaluate_active_run(run, ActiveEvaluationConfig(
            evaluate_trajectory=True, learning_regret_threshold=.01,
            learning_angular_threshold_degrees=5.), test_queries=queries).to_dict()
            for name, queries in design.test_queries.items()}
        run.evaluation = evaluations["ordinary"]
        decisions = algorithm.context.decision_problem
        flips = sum(not np.array_equal(
            decisions.minimize(run.true_theta*r.query, np.random.default_rng(0)), r.observed_decision)
            for r in run.records)
        indices = [r.action_diagnostics["candidate_index"] for r in run.records]
        run.metadata.update(comparison_fingerprint=fingerprint, comparison_family=design.family,
            runtime_versions={"python": platform.python_version(), "numpy": np.__version__,
                              "scipy": scipy.__version__},
            comparison_title=design.title, comparison_description=design.description,
            comparison_estimator=estimators[label],
            evaluations_by_distribution=evaluations,
            candidate_group_labels=design.group_labels,
            selected_candidate_groups=design.candidate_groups[indices].tolist(),
            parameter_noise_decision_flips=flips,
            first_invalid_step=next((r.step for r in run.records
                if r.update_diagnostics.get("estimate_status") != "valid"), None))
        if len(run.records) != design.scenario.horizon:
            raise RuntimeError("fixed-horizon comparison ended before T")
        if label in {"Pedro algorithm", "Genious Pedro"} and any(
                not r.update_diagnostics["constraints_exact"] for r in run.records):
            raise RuntimeError("this small comparison requires exact Pedro constraints")
        # Atomic checkpoint: incomplete writes cannot be reused as finished runs.
        temporary = destination.with_suffix(".json.tmp")
        run.save_json(temporary)
        temporary.replace(destination)
        results.append(run.to_dict())
        progress(f"DONE {design.family} seed={design.scenario.seed} {label}: "
                 f"status={run.evaluation['final_status']}, "
                 f"angle={run.evaluation['final_angular_error_degrees']}, "
                 f"regret={run.evaluation['final_normalized_regret']}, "
                 f"seconds={run.runtime_seconds:.1f}")
    return results


def run_pedro_score_design(design: ComparisonDesign, directory: str | Path, *,
                           score_config: NestedLangevinConfig | None = None,
                           use_cache: bool = True, progress=print) -> list[dict]:
    """Run exactly Pedro and Score Base; checkpoint every complete run."""

    return _run_comparison_design(
        design,
        directory,
        score_config=score_config,
        samd_config=None,
        use_cache=use_cache,
        progress=progress,
        include_genious_pedro=False,
        include_online_samd=False,
    )


def run_three_algorithm_design(design: ComparisonDesign, directory: str | Path, *,
                               score_config: NestedLangevinConfig | None = None,
                               use_cache: bool = True, progress=print) -> list[dict]:
    """Run Pedro, Genious Pedro and Score Base under the identical protocol.

    Algorithmic invalid estimates remain in every trajectory with ``None``
    metrics. Unexpected exceptions fail loudly; they are not disguised as cone
    failures. Each complete run is checkpointed atomically.
    """

    return _run_comparison_design(
        design,
        directory,
        score_config=score_config,
        samd_config=None,
        use_cache=use_cache,
        progress=progress,
        include_genious_pedro=True,
        include_online_samd=False,
    )


def run_four_algorithm_design(design: ComparisonDesign, directory: str | Path, *,
                              score_config: NestedLangevinConfig | None = None,
                              samd_config: OnlineSAMDConfig | None = None,
                              use_cache: bool = True, progress=print) -> list[dict]:
    """Run Pedro, Genious Pedro, Score Base, and Uniform Online SAMD.

    Uniform Online SAMD receives the same one-query-per-time-step protocol and
    uses exactly one update from the newly received public observation.  This
    function checkpoints complete runs but does not alter the existing two- or
    three-algorithm comparison entry points.
    """

    return _run_comparison_design(
        design,
        directory,
        score_config=score_config,
        samd_config=samd_config,
        use_cache=use_cache,
        progress=progress,
        include_genious_pedro=True,
        include_online_samd=True,
    )
