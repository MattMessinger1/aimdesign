"""
otv_physics_simulator.py
========================

This module implements a coarse physics‑based simulator for the
"Open Test Vehicle" (OTV) modules defined in the Open Benchmark
Framework (OBF).  The goal of this simulator is to allow rapid
screening of candidate dielectric films or laminates prior to
fabrication.  For each module (Dk/Df & loss, moisture uptake,
adhesion retention, via yield and CAF/SIR) the simulator computes
approximate metrics based on input material properties and operating
conditions.  These metrics are then fed into gate logic (Gate 0–2) as
defined in OBF.

The models used here are deliberately simplistic; they should be
replaced with more accurate constitutive laws, empirical fits or
multiphysics solvers as your research progresses.  Nonetheless, they
provide a practical starting point for building a realistic
simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd


@dataclass
class Material:
    """Represents a candidate material or laminate under evaluation.

    Attributes
    ----------
    dielectric_constant : float
        Relative permittivity (Dk) of the material.
    loss_tangent : float
        Loss tangent (Df) representing dielectric loss.
    moisture_uptake : float
        Equilibrium moisture uptake fraction at standard conditions.
    adhesion_energy : float
        Base adhesion energy (J/m^2) before aging.
    via_yield : float
        Baseline via yield (fraction of good vias) under ideal process.
    ion_content : float
        Ionic contaminant content (ppm) relevant for CAF/SIR behaviour.
    name : str, optional
        Name or identifier of the material.
    """

    dielectric_constant: float
    loss_tangent: float
    moisture_uptake: float
    adhesion_energy: float
    via_yield: float
    ion_content: float
    name: str = field(default="candidate")


@dataclass
class OperatingConditions:
    """Defines the operating and test conditions for the OTV simulation.

    Attributes
    ----------
    frequency : float
        Measurement frequency (Hz) for dielectric loss calculation.
    temperature : float
        Ambient temperature (°C).
    humidity : float
        Relative humidity (0–1).
    stress_cycles : int
        Number of thermal/humidity cycles for adhesion testing.
    bias_voltage : float
        Electrical bias (V) applied during CAF/SIR testing.
    """

    frequency: float = 1e9
    temperature: float = 25.0
    humidity: float = 0.5
    stress_cycles: int = 100
    bias_voltage: float = 3.3


@dataclass
class ModuleOutputs:
    """Holds the outputs of each OTV module.

    Attributes correspond to Tier 0–2 metrics as described in OBF.
    """

    dk_df: float  # Effective dielectric constant & loss metric
    moisture_uptake: float  # Moisture uptake after exposure
    adhesion_retention: float  # Fraction of initial adhesion remaining
    via_chain_yield: float  # Yield of via chain module
    sir: float  # Surface insulation resistance (log scale)
    caf_failures: int  # Count of CAF failures


class OTVPhysicsSimulator:
    """Simulate physics‑based metrics for OBF OTV modules.

    The simulator exposes a single ``run`` method that accepts a list
    of ``Material`` objects and returns a pandas ``DataFrame`` with
    one row per material and columns for each module's outputs.  Each
    module calculation is implemented as a separate method and can be
    overridden or extended for more accurate modelling.
    """

    def __init__(self, conditions: OperatingConditions = None, random_state: int = 42) -> None:
        self.conditions = conditions or OperatingConditions()
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, materials: List[Material]) -> pd.DataFrame:
        """Run the physics simulation for a list of materials.

        Parameters
        ----------
        materials : List[Material]
            Materials to simulate.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the simulated module outputs and
            intrinsic properties for each material.
        """
        records: List[Dict[str, float]] = []
        for mat in materials:
            outputs = self.simulate_material(mat)
            record = {
                'material': mat.name,
                'dielectric_constant': mat.dielectric_constant,
                'loss_tangent': mat.loss_tangent,
                'moisture_uptake_equilibrium': mat.moisture_uptake,
                'adhesion_energy': mat.adhesion_energy,
                'via_yield': mat.via_yield,
                'ion_content': mat.ion_content,
                'dk_df_metric': outputs.dk_df,
                'moisture_uptake_metric': outputs.moisture_uptake,
                'adhesion_retention_metric': outputs.adhesion_retention,
                'via_chain_yield_metric': outputs.via_chain_yield,
                'sir_metric': outputs.sir,
                'caf_failures': outputs.caf_failures,
            }
            records.append(record)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Core simulation methods
    # ------------------------------------------------------------------
    def simulate_material(self, mat: Material) -> ModuleOutputs:
        """Simulate all OTV modules for a single material.

        The sequence follows the OTV module numbering described in the
        user instructions: M1 (Dk/Df & loss), M2 (Moisture uptake), M4
        (Adhesion retention), M2 (Via chain), M3 (CAF/SIR).  The
        implementation here uses simple formulas that relate intrinsic
        properties to measured outputs.  Replace these formulas with
        more sophisticated models (e.g. FEM solvers, empirical fits)
        as needed.

        Parameters
        ----------
        mat : Material
            The material being simulated.

        Returns
        -------
        ModuleOutputs
            The simulated outputs for the material.
        """
        dk_df = self.module_m1_dk_df(mat)
        mu = self.module_m2_moisture(mat)
        adhesion = self.module_m4_adhesion(mat)
        via_yield = self.module_m2_via_chain(mat)
        sir, caf = self.module_m3_caf_sir(mat)
        return ModuleOutputs(dk_df, mu, adhesion, via_yield, sir, caf)

    # ------------------------------------------------------------------
    # Module implementations
    # ------------------------------------------------------------------
    def module_m1_dk_df(self, mat: Material) -> float:
        """Compute a composite metric representing Dk/Df & loss.

        We approximate the effective loss metric as the product of the
        dielectric constant, loss tangent and a frequency factor.  A
        lower value is better.  The formula is deliberately simple:

        ``dk_df_metric = dk * df * log10(frequency / 1e6)``

        where ``dk`` is the dielectric constant, ``df`` is the loss
        tangent and ``frequency`` is the measurement frequency.

        Returns
        -------
        float
            The Dk/Df composite metric.
        """
        frequency = self.conditions.frequency
        # Avoid log(0); frequency must be > 0
        freq_factor = np.log10(max(frequency, 1e3) / 1e6)
        return mat.dielectric_constant * mat.loss_tangent * freq_factor

    def module_m2_moisture(self, mat: Material) -> float:
        """Compute moisture uptake after exposure.

        We model moisture uptake as equilibrium uptake scaled by a
        humidity and temperature factor.  At higher humidity and
        temperature the uptake increases.  A noise term is added to
        represent variability.

        Returns
        -------
        float
            Moisture uptake fraction after exposure.
        """
        temp_factor = (self.conditions.temperature + 273.15) / 298.15
        humidity_factor = self.conditions.humidity
        uptake = mat.moisture_uptake * humidity_factor * temp_factor
        # Add small noise to simulate variability
        uptake += self.rng.normal(scale=0.01 * mat.moisture_uptake)
        return max(uptake, 0.0)

    def module_m4_adhesion(self, mat: Material) -> float:
        """Simulate adhesion retention after stress cycling.

        Adhesion energy decays exponentially with the number of
        stress cycles and is also influenced by moisture exposure.  We
        compute retention as:

        ``retention = exp(-alpha * cycles) * exp(-beta * humidity)``

        where ``alpha`` and ``beta`` are derived from intrinsic
        adhesion energy and moisture uptake, respectively.  The result
        is clipped between 0 and 1.

        Returns
        -------
        float
            Fraction of initial adhesion energy remaining.
        """
        cycles = self.conditions.stress_cycles
        alpha = 1.0 / max(mat.adhesion_energy, 1e-6)
        beta = mat.moisture_uptake
        retention = np.exp(-alpha * cycles) * np.exp(-beta * self.conditions.humidity * 10)
        return float(np.clip(retention, 0.0, 1.0))

    def module_m2_via_chain(self, mat: Material) -> float:
        """Compute via chain yield for the material.

        The yield is reduced by moisture uptake and loss tangent (more
        polar materials may reduce via quality).  A small random term
        is introduced to reflect process variability.

        Returns
        -------
        float
            Fraction of vias that remain conductive.
        """
        base_yield = mat.via_yield
        reduction = 0.1 * mat.moisture_uptake + 0.05 * mat.loss_tangent
        yield_metric = base_yield * (1.0 - reduction)
        noise = self.rng.normal(scale=0.02)
        yield_metric += noise
        return float(np.clip(yield_metric, 0.0, 1.0))

    def module_m3_caf_sir(self, mat: Material) -> Tuple[float, int]:
        """Simulate CAF/SIR behaviour.

        We model the surface insulation resistance (SIR) on a log scale
        as inversely proportional to ion content and moisture uptake,
        and apply a bias voltage factor.  The number of CAF failures
        is modelled as a Poisson random variable with mean scaling with
        ion content and humidity.

        Returns
        -------
        (float, int)
            A tuple of (SIR metric, number of CAF failures).
        """
        # SIR metric (log10 ohms): high is good
        sir = 12.0 - 2.0 * mat.ion_content - 1.0 * mat.moisture_uptake + 0.1 * self.conditions.bias_voltage
        sir += self.rng.normal(scale=0.2)  # variability
        # CAF failures: Poisson with mean scaled by ion content and humidity
        lam = max(mat.ion_content * 10 * self.conditions.humidity, 0.01)
        caf_failures = self.rng.poisson(lam)
        return float(sir), int(caf_failures)


if __name__ == '__main__':
    # Example usage: simulate three materials
    mats = [
        Material(3.2, 0.012, 0.03, 5.0, 0.95, 0.01, name='MaterialA'),
        Material(3.8, 0.016, 0.04, 4.0, 0.90, 0.02, name='MaterialB'),
        Material(2.9, 0.009, 0.02, 6.0, 0.98, 0.005, name='MaterialC'),
    ]
    sim = OTVPhysicsSimulator()
    df = sim.run(mats)
    print(df.head())
https://github.com/MattMessinger1/aimdesign/tree/main/docs/lab_simulators
