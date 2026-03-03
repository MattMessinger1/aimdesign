# Lab OBF Material Simulators

This directory contains skeleton Python modules for two simulators
designed to evaluate new materials under the **Open Bench Format (OBF)**
framework.  These simulators provide a starting point for building
decision‑quality and physics‑based analyses of candidate materials.  They
are meant to live under the `lab.aimdesign.ai` environment, but can be
run locally for development and testing.

## Simulators

### 1. Decision Quality Simulator (`decision_quality_simulator.py`)

This simulator models the *campaign‑level* process of evaluating a new
material.  It focuses on the **value of information** provided by OBF
evidence versus ad‑hoc evidence.  The high‑level loop is:

1.  **Generate candidate materials** with intrinsic properties and
    metadata completeness.  The current implementation uses simple
    random distributions; real implementations should draw from known
    distributions or sampling plans.
2.  **Propagate candidates through process stages**.  Each stage (e.g.,
    synthesis, casting/B‑stage, storage/handling, lamination/cure,
    via formation, plating/adhesion, pattern/yield, reliability) can
    introduce variability and failure modes.  The simulator applies
    simple transformations to candidate properties at each stage.
3.  **Evaluate evidence quality**.  Two evidence modes are modeled:
    *ad‑hoc* (with missing metadata and bias sources) and **OBF**
    (with lower missingness and reference runs for normalization).
4.  **Make decisions** based on simulated outcomes, tracking metrics
    such as time‑to‑decision, false positives/negatives and value of
    information per metadata field.

The simulator returns both raw run data and summary statistics so you
can assess the benefit of OBF adoption and identify which metadata
fields contribute most to decision quality.

### 2. OTV Physics Simulator (`otv_physics_simulator.py`)

This simulator models the *screening‑level* physics of candidate
materials for the OBF Tier 0–2 gates.  It contains placeholder
implementations for the following modules:

* **M1 – Dielectric constant/loss (`m1_dielectric_loss`)**: models
  electromagnetic response of the material (e.g., effective permittivity,
  loss tangent) given frequency, conductor roughness and copper state.
* **Moisture uptake (`m2_moisture_uptake`)**: models diffusion and
  solubility, optionally with temperature/humidity dependence.
* **M4 – Adhesion retention (`m4_adhesion_retention`)**: models bond
  strength and degradation under environmental cycling.
* **M2 – Via formation/yield (`m2_via_chain_yield`)**: models the
  compatibility of the material with via formation, desmear and plating,
  producing yield and resistance distributions.
* **M3 – CAF/SIR (`m3_caf_sir`)**: models electrochemical hazard (ion
  content, moisture, bias) and predicts surface insulation resistance
  failures.

Each module returns simulated outputs, which are passed through a gate
logic replicating OBF Tier 0–2 outcome definitions.  The simulator
aggregates module outputs into a data structure matching the OBF
dataset template, so you can validate synthetic runs with your existing
analysis scripts.

## Next Steps

These modules provide **interfaces and documentation** rather than
complete models.  To make the simulators realistic you will need to:

* Replace placeholder probability distributions with distributions
  informed by your experimental data.
* Implement physics‑based models or surrogate models calibrated to
  experiments (e.g., using `scipy`, `pint` and material libraries).
* Extend the gating logic to match the latest OBF specification.
* Build a simple CLI or web API for running parameter sweeps and
  visualizing results.

Contributions are welcome!  Use these skeletons as a foundation for
your lab’s simulators and adapt them to the specific needs of your
materials and processes.
