"""
Main entry point for parsing QChem output files.
"""

import re
from collections.abc import Sequence

from calcflow.common.exceptions import ParsingError
from calcflow.common.results import CalculationResult
from calcflow.io.core import BlockParser, core_parse
from calcflow.io.qchem.blocks.charges import ChargesParser
from calcflow.io.qchem.blocks.finalization import TerminationParser
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
from calcflow.utils import logger

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
    TerminationParser(),
]


def parse_qchem_output(output: str) -> CalculationResult:
    """
    Parses the text output of a single-job QChem calculation.

    Args:
        output: The string content of the QChem output file.

    Returns:
        A CalculationResult object containing the parsed results.
    """
    return core_parse(output, PARSER_REGISTRY_SP)


def parse_qchem_multi_job_output(output: str, num_jobs: int | None = None) -> Sequence[CalculationResult]:
    """
    Parses multi-job QChem output files (e.g., MOM, XAS calculations).

    QChem can run multiple sequential jobs in a single output file, with each job
    separated by "Running Job X of Y" markers. This function splits the output by
    these job boundaries and parses each job independently using the standard parser.

    Args:
        output: The string content of the QChem multi-job output file.
        num_jobs: Optional. If specified, only parse the first N jobs and ignore the rest.
                  Useful for contract tests that only need the first job(s).

    Returns:
        A list of CalculationResult objects, one per job (or up to num_jobs if specified), in sequential order.

    Raises:
        ParsingError: If job markers are not found or job structure is invalid.
    """
    # Pattern: "Running Job 1 of 2 filename.in"
    job_pattern = re.compile(r"^Running Job (\d+) of (\d+)", re.MULTILINE)

    # Find all job markers and their positions
    matches = list(job_pattern.finditer(output))

    if not matches:
        raise ParsingError(
            "No 'Running Job X of Y' markers found. This may be a single-job file. Use parse_qchem_output() instead."
        )

    # Extract job count from first match
    expected_job_count = int(matches[0].group(2))
    logger.debug(f"Found multi-job file with {expected_job_count} jobs")

    # Validate we found the expected number of jobs
    if len(matches) != expected_job_count:
        logger.warning(
            f"Expected {expected_job_count} jobs but found {len(matches)} job markers. Proceeding with found jobs."
        )

    # Limit matches if num_jobs is specified
    if num_jobs is not None:
        if num_jobs < 1:
            raise ParsingError(f"num_jobs must be at least 1, got {num_jobs}")
        if num_jobs > len(matches):
            logger.warning(f"num_jobs={num_jobs} but only {len(matches)} jobs found. Parsing all available jobs.")
        matches = matches[:num_jobs]

    # Split output into job chunks
    job_chunks: list[str] = []
    for i, match in enumerate(matches):
        start_pos = match.start()
        # End position is the start of next job, or end of file for last job
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        job_chunks.append(output[start_pos:end_pos])

    # Validate job indices are sequential
    for i, match in enumerate(matches, start=1):
        job_idx = int(match.group(1))
        if job_idx != i:
            logger.warning(f"Job indices are not sequential: expected {i}, found {job_idx}")

    # Parse each job independently
    results: list[CalculationResult] = []
    for i, chunk in enumerate(job_chunks, start=1):
        logger.debug(f"Parsing job {i}/{len(job_chunks)}")
        try:
            result = core_parse(chunk, PARSER_REGISTRY_SP)
            results.append(result)
        except Exception as e:
            raise ParsingError(f"Failed to parse job {i}: {e}") from e

    logger.info(f"Successfully parsed {len(results)} jobs from multi-job output")
    return results
