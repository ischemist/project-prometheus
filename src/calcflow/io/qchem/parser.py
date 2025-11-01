"""
Main entry point for parsing QChem output files.
"""

from collections.abc import Sequence

from calcflow.common.models import CalculationResult
from calcflow.io.core import BlockParser, core_parse
from calcflow.io.qchem.blocks.charges import ChargesParser
from calcflow.io.qchem.blocks.geometry import GeometryParser
from calcflow.io.qchem.blocks.metadata import MetadataParser
from calcflow.io.qchem.blocks.multipole import MultipoleParser
from calcflow.io.qchem.blocks.orbitals import OrbitalsParser
from calcflow.io.qchem.blocks.scf import ScfParser
from calcflow.io.qchem.blocks.tddft.excitations import ExcitationsParser
from calcflow.io.qchem.blocks.tddft.gs_ref import GroundStateRefParser
from calcflow.io.qchem.blocks.tddft.nto import NTOParser
from calcflow.io.qchem.blocks.tddft.trans_dm import TransitionDensityMatrixParser
from calcflow.io.qchem.blocks.tddft.unrel_dm import UnrelaxedDensityMatrixParser
from calcflow.io.qchem.blocks.timing import TimingParser

# The ordered registry of parsers for a standard QChem calculations.
PARSER_REGISTRY_SP: Sequence[BlockParser] = [
    MetadataParser(),
    GeometryParser(),
    ScfParser(),
    ChargesParser(),
    OrbitalsParser(),
    MultipoleParser(),
    ExcitationsParser(),
    NTOParser(),
    GroundStateRefParser(),
    UnrelaxedDensityMatrixParser(),
    TransitionDensityMatrixParser(),
    TimingParser(),
]


def parse_qchem_output(output: str) -> CalculationResult:
    """
    Parses the text output of a QChem calculation.

    Args:
        output: The string content of the QChem output file.

    Returns:
        A CalculationResult object containing the parsed results.
    """
    return core_parse(output, PARSER_REGISTRY_SP)
