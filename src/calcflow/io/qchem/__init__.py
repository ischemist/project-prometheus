"""
Main entry point for parsing QChem output files.
"""

from collections.abc import Sequence

from calcflow.common.models import CalculationResult
from calcflow.io.core import BlockParser, core_parse

# from calcflow.io.qchem.blocks.finalization import TerminationParser
from calcflow.io.qchem.blocks.charges import ChargesParser
from calcflow.io.qchem.blocks.geometry import GeometryParser
from calcflow.io.qchem.blocks.metadata import MetadataParser
from calcflow.io.qchem.blocks.multipole import MultipoleParser
from calcflow.io.qchem.blocks.orbitals import OrbitalsParser
from calcflow.io.qchem.blocks.scf import ScfParser

# The ordered registry of parsers for a standard QChem Single Point calculation.
# Order can be important. Metadata/REM should usually come first.
PARSER_REGISTRY_SP: Sequence[BlockParser] = [
    # RemBlockParser(),
    MetadataParser(),
    GeometryParser(),
    ScfParser(),
    ChargesParser(),
    OrbitalsParser(),
    MultipoleParser(),
    # TerminationParser(),
]


def parse_qchem_sp(output: str) -> CalculationResult:
    """
    Parses the text output of a QChem single-point calculation.

    Args:
        output: The string content of the QChem output file.

    Returns:
        A CalculationResult object containing the parsed results.
    """
    return core_parse(output, PARSER_REGISTRY_SP)
