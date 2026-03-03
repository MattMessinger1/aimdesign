"""
"""

decision_quality_simulator.py
===============================

This module provides a simple, extensible framework for evaluating the
"value of information" (VOI) that a materials science workflow like
Open Benchmark Framework (OBF) can deliver compared to ad ‑ ‑hoc evidence
collection.  It is not a fully featured product on its own; rather it
implements a skeleton that you can build upon when constructing a
campaign‑ level simulator that mimics the decision quality gains you
might achieve by adopting OBF in the lab.

The simulator works by generating hypothetical material candidates and
propagating them through a series of manufacturing/measurement stages.
At each stage it models the quality of evidence gathered under two
scenarios:

* ``ad_hoc`` — evidence is incomplete, noisy and difficult to
  normalise; metadata may be missing.
* ``obf`` — evidence adheres to OBF protocols (e.g. MDS and OTV
  metadata standards) so that bias and missingness are reduced.

The simulator records metrics such as time ‑ decision,
false ‑ positive/false ‑ negative rates, and value of information.  These
outputs are returned as a pandas ``DataFrame`` for downstream
analysis and visualisation.

Because every research programme is different, most functions in this
module are designed to be overridden or extended.  The default
implementations provide simple, deterministic calculations and random
noise to illustrate how the pieces fit together.  See the README in
``lab_simulators/README.md`` for a high ‑ level description of the
overall architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Any
import numpy as np
import pandas as pd


@dataclass
class Candidate:
    """Represents a hypothetical material candidate.

    Attributes
    ----------
    props : Dict[str, float]
        A dictionary of intrinsic properties (e.g. dielectric constant,
        loss tangent, moisture uptake).  The keys and ranges are
        arbitrary; customise via ``generate_candidates``.
    latent_risk : float
        A latent risk score (0–1) used to model "unknown unknowns" that
        could cause failure despite favourable measured properties.
    """

    props: Dict[str, float]
    latent_risk: float


@dataclass
class StageResult:
    """Stores the outcome of a candidate passing through a stage.

    Attributes
    ----------
    evidence : Dict[str, float]
        Observed metrics at this stage.  For example, a rough
        measurement of loss tangent, moisture uptake or adhesion.
    success : bool
        Whether the candidate passed the stage (e.g. did not crack or
        delaminate).  A simple threshold on measured properties is used
        by default.
    """

    evidence: Dict[str, float]
    success: bool


class DecisionQualitySimulator:
    """Simulate decision quality between ad ‑ ‑hoc and OBF evidence collection.

    This class orchestrates the generation of candidates, the
    propagation through multiple stages, the evaluation of evidence
    quality and the calculation of decision metrics.

    Parameters
    ----------
    stages : List[Callable[[Candidate], StageResult]]
        A sequence of functions, one per stage, that define how to
        propagate a candidate and return stage results.
    evidence_model : Callable[[StageResult, bool], Dict[str, float]]
        A function that takes the ``StageResult`` and a boolean flag
        ``obf`` (``True`` for OBF, ``False`` for ad ‑ ‑hoc) and returns
        evidence as measured by the chosen workflow.  The default
        implementation adds Gaussian noise and missingness for
        ``ad_hoc`` and leaves values unchanged for ``obf``.
    decision_rule : Callable[[List[Dict[str, float]]], bool]
        A function that takes a list of evidence dictionaries (one per
        stage) and outputs a boolean decision (e.g. whether to pursue
        further experiments).  The default rule rejects candidates if
        any measured property is below a threshold.
    random_state : int, optional
        Seed for reproducible random number generation.
    """

    def __init__(
        self,
        stages: List[Callable[[Candidate], StageResult]],
        evidence_model: Callable[[StageResult, bool], Dict[str, float]] = None,
        decision_rule: Callable[[List[Dict[str, float]]], bool] = None,
        random_state: int = 42,
    ) -> None:
        self.stages = stages
        self.evidence_model = evidence_model or self.default_evidence_model
        self.decision_rule = decision_rule or self.default_decision_rule
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def generate_candidates(self, n: int = 100) -> List[Candidate]:
        """Generate a list of synthetic material candidates.

        Each candidate is assigned random intrinsic properties drawn
        uniformly from [0, 1] and a latent risk score.  Override this
        method to use domain ‑ specific property distributions.

        Parameters
        ----------
        n : int
            Number of candidates to generate.

        Returns
        -------
        List[Candidate]
            The generated candidates.
        """
        candidates = []
        for _ in range(n):
            props = {
                'dielectric_constant': float(self.rng.uniform(2.5, 4.5)),
                'loss_tangent': float(self.rng.uniform(0.001, 0.02)),
                'moisture_uptake': float(self.rng.uniform(0.0, 0.05)),
            }
            latent_risk = float(self.rng.uniform(0.0, 1.0))
            candidates.append(Candidate(props, latent_risk))
        return candidates

    # ------------------------------------------------------------------
    # Evidence collection models
    # ------------------------------------------------------------------
    def default_evidence_model(self, result: StageResult, obf: bool) -> Dict[str, float]:
        """Default model for transforming stage results into evidence.

        For ad ‑ ‑hoc evidence (``obf=False``) the model applies Gaussian
        noise (5 % relative) and randomly removes one measurement to
        mimic missing metadata.  For OBF evidence (``obf=True``) the
        measurements are returned unchanged.

        Parameters
        ----------
        result : StageResult
            The raw outcome of a stage.
        obf : bool
            If ``True``, return high ‑ ‑quality evidence (no noise or
            missingness); otherwise return noisy, sometimes incomplete
            evidence.

        Returns
        -------
        Dict[str, float]
            The observed evidence dictionary.
        """
        evidence = result.evidence.copy()
        if not obf:
            # Apply 5 % relative noise
            for key, val in evidence.items():
                noise = self.rng.normal(scale=0.05 * val)
                evidence[key] = max(val + noise, 0.0)
            # Randomly drop one measurement
            if evidence and self.rng.random() < 0.3:
                drop_key = self.rng.choice(list(evidence.keys()))
                evidence.pop(drop_key)
        return evidence

    # ------------------------------------------------------------------
    # Decision rules
    # ------------------------------------------------------------------
    def default_decision_rule(self, evidences: List[Dict[str, float]]) -> bool:
        """Default decision rule based on thresholding measurements.

        The candidate is accepted (i.e. the programme continues
        experimenting) if all measured properties across all stages
        exceed predetermined thresholds.  Missing measurements count
        against the candidate (i.e. lead to rejection).

        Parameters
        ----------
        evidences : List[Dict[str, float]]
            Evidence dictionaries produced by the evidence model for
            each stage.

        Returns
        -------
        bool
            ``True`` if the candidate is accepted, ``False`` otherwise.
        """
        thresholds = {
            'dielectric_constant': 3.0,
            'loss_tangent': 0.015,
            'moisture_uptake': 0.02,
        }
        for evidence in evidences:
            for key, threshold in thresholds.items():
                val = evidence.get(key)
                if val is None or (
                    key == 'loss_tangent' and val > threshold
                ) or (
                    key != 'loss_tangent' and val < threshold
                ):
                    return False
        return True

    # ------------------------------------------------------------------
    # Stage simulation
    # ------------------------------------------------------------------
    def run_stage(self, candidate: Candidate, stage_fn: Callable[[Candidate], StageResult]) -> StageResult:
        """Run a single stage function on a candidate.

        The default implementation simply calls the provided stage
        function.  Override this method if you need to add additional
        behaviour around the stage (e.g. side ‑ effects, logging).

        Parameters
        ----------
        candidate : Candidate
            The candidate being processed.
        stage_fn : Callable[[Candidate], StageResult]
            A function that takes a candidate and returns a StageResult.

        Returns
        -------
        StageResult
            The result of processing the candidate.
        """
        return stage_fn(candidate)

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    def run(self, n_candidates: int = 100) -> pd.DataFrame:
        """Run the decision quality simulation.

        For each generated candidate, the simulator executes all
        configured stages, collects evidence for both ad ‑ ‑hoc and OBF
        workflows, and applies the decision rule.  Summary metrics
        including acceptance status and stage successes are returned in
        a pandas DataFrame.  Additional metrics (e.g. VOI) can be
        computed downstream.

        Parameters
        ----------
        n_candidates : int
            Number of candidates to simulate.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per candidate and columns for
            intrinsic properties, stage successes and acceptance flags
            for both ad ‑ ‑hoc and OBF evidence scenarios.
        """
        candidates = self.generate_candidates(n_candidates)
        records: List[Dict[str, Any]] = []

        for cand in candidates:
            # Simulate each stage once to obtain ground truth results
            stage_results = [self.run_stage(cand, s) for s in self.stages]
            # Collect evidence under both workflows
            evidences_ad_hoc = [self.evidence_model(res, obf=False) for res in stage_results]
            evidences_obf = [self.evidence_model(res, obf=True) for res in stage_results]

            # Apply decision rule
            accepted_ad_hoc = self.decision_rule(evidences_ad_hoc)
            accepted_obf = self.decision_rule(evidences_obf)

            # Construct record
            record = {
                **cand.props,
                'latent_risk': cand.latent_risk,
                'ad_hoc_accepted': accepted_ad_hoc,
                'obf_accepted': accepted_obf,
            }
            # Add stage success flags (ground truth)
            for i, res in enumerate(stage_results):
                record[f'stage_{i+1}_success'] = res.success
            records.append(record)

        df = pd.DataFrame(records)
        return df


def simple_stage_factory(thresholds: Dict[str, Tuple[float, bool]]) -> Callable[[Candidate], StageResult]:
    """Create a simple stage function with specified thresholds.

    This helper factory returns a function that when called on a
    candidate, compares the candidate's intrinsic properties against
    thresholds and returns a ``StageResult``.  The ``thresholds``
    parameter maps property names to a tuple of (threshold, direction)
    where ``direction=True`` means the property must be greater than
    the threshold to pass, and ``False`` means it must be less than
    the threshold.

    Parameters
    ----------
    thresholds : Dict[str, Tuple[float, bool]]
        Mapping from property name to a threshold and direction.

    Returns
    -------
    Callable[[Candidate], StageResult]
        A function that implements the stage logic.
    """
    def stage(cand: Candidate) -> StageResult:
        evidence: Dict[str, float] = {}
        success = True
        for prop, (thresh, must_be_greater) in thresholds.items():
            value = cand.props.get(prop, np.nan)
            evidence[prop] = value
            if must_be_greater:
                success = success and (value >= thresh)
            else:
                success = success and (value <= thresh)
        # Incorporate latent risk: a high latent risk can cause failure
        if cand.latent_risk > 0.8:
            success = False
        return StageResult(evidence, success)

    return stage


if __name__ == '__main__':
    # Example usage: run a basic simulation with two stages.
    stage1 = simple_stage_factory({
        'dielectric_constant': (3.0, True),
        'loss_tangent': (0.015, False),
    })
    stage2 = simple_stage_factory({
        'moisture_uptake': (0.02, False),
    })
    sim = DecisionQualitySimulator(stages=[stage1, stage2])
    df = sim.run(n_candidates=10)
    print(df.head())
