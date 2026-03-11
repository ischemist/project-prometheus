"""CalcFlow: Quantum Chemistry Calculation I/O Done Right.

This package provides a robust, Pythonic interface for preparing inputs and parsing
outputs for quantum chemistry software like Q-Chem and ORCA.
"""

from calcflow._version import __version__
from calcflow.common.input import CalculationInput
from calcflow.geometry.static import Geometry
from calcflow.geometry.topology import classify_bond, detect_molecules
from calcflow.geometry.trajectory import Trajectory
from calcflow.io.orca import parse_orca_output
from calcflow.io.qchem import parse_qchem_multi_job_output, parse_qchem_output
from calcflow.postprocess import (
    lorentzian_broadening,
    make_energy_grid,
    opa_spectrum_from_adc_states,
    spectrum_from_excited_states,
    tpa_spectrum_from_adc_states,
)

__all__ = [
    "__version__",
    "CalculationInput",
    "Geometry",
    "Trajectory",
    "classify_bond",
    "detect_molecules",
    "lorentzian_broadening",
    "make_energy_grid",
    "opa_spectrum_from_adc_states",
    "parse_orca_output",
    "parse_qchem_output",
    "parse_qchem_multi_job_output",
    "spectrum_from_excited_states",
    "tpa_spectrum_from_adc_states",
]
