"""CalcFlow: Quantum Chemistry Calculation I/O Done Right.

This package provides a robust, Pythonic interface for preparing inputs and parsing
outputs for quantum chemistry software like Q-Chem and ORCA.
"""

from importlib.metadata import version

from calcflow.common.input import CalculationInput
from calcflow.geometry.static import Geometry
from calcflow.geometry.trajectory import Trajectory
from calcflow.io.orca import parse_orca_output
from calcflow.io.qchem import parse_qchem_multi_job_output, parse_qchem_output
from calcflow.postprocess import lorentzian_spectrum, make_energy_grid

__version__ = version("calcflow")

__all__ = [
    "__version__",
    "CalculationInput",
    "Geometry",
    "Trajectory",
    "lorentzian_spectrum",
    "make_energy_grid",
    "parse_orca_output",
    "parse_qchem_output",
    "parse_qchem_multi_job_output",
]
