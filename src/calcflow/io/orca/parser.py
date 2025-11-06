"""
Main entry point for parsing ORCA output files.
"""

from collections.abc import Sequence

from calcflow.common.results import CalculationResult
from calcflow.io.core import BlockParser, core_parse
from calcflow.io.orca.blocks.charges import ChargesParser
from calcflow.io.orca.blocks.dipole import DipoleParser
from calcflow.io.orca.blocks.dispersion import DispersionParser
from calcflow.io.orca.blocks.finalization import FinalEnergyParser, TerminationParser
from calcflow.io.orca.blocks.geometry import GeometryParser
from calcflow.io.orca.blocks.orbitals import OrbitalsParser
from calcflow.io.orca.blocks.scf import ScfParser
from calcflow.io.orca.blocks.timing import TimingParser

# The ordered registry of parsers for a standard ORCA Single Point calculation.
PARSER_REGISTRY_SP: Sequence[BlockParser] = [
    GeometryParser(),
    ScfParser(),
    OrbitalsParser(),
    ChargesParser(),
    DipoleParser(),
    DispersionParser(),
    FinalEnergyParser(),
    TerminationParser(),
    TimingParser(),
]


def parse_orca_output(output: str) -> CalculationResult:
    """
    Parses the text output of an ORCA calculation.

    Args:
        output: The string content of the ORCA output file.

    Returns:
        A CalculationResult object containing the parsed results.
    """
    return core_parse(output, PARSER_REGISTRY_SP)
