"""
The unified core parsing engine for calcflow.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol

from calcflow.common.exceptions import ParsingError
from calcflow.common.models import CalculationResult
from calcflow.io.state import ParseState
from calcflow.utils import logger


class BlockParser(Protocol):
    """Protocol for all section/block parsers."""

    def matches(self, line: str, state: ParseState) -> bool: ...
    def parse(self, iterator: Iterator[str], start_line: str, state: ParseState) -> None: ...


def core_parse(output_text: str, parser_registry: Sequence[BlockParser]) -> CalculationResult:
    """
    Orchestrates the parsing of a full output file using a registry of block parsers.

    Args:
        output_text: The entire content of the calculation output file.
        parser_registry: A list of `BlockParser` instances to use for parsing.

    Returns:
        An immutable `CalculationResult` object.
    """
    state = ParseState(raw_output=output_text)
    line_iterator = iter(output_text.splitlines())

    current_line_num = 0
    while True:
        try:
            if state.buffered_line is not None:
                line = state.buffered_line
                state.buffered_line = None
            else:
                line = next(line_iterator)
                current_line_num += 1

            parser_found = False
            for parser in parser_registry:
                if parser.matches(line, state):
                    logger.debug(f"Line {current_line_num}: Matched {type(parser).__name__}")
                    parser.parse(line_iterator, line, state)
                    parser_found = True
                    break

            if parser_found:
                continue

        except StopIteration:
            logger.debug("Core parser reached end of file.")
            break
        except ParsingError:
            state.termination_status = "ERROR"
            raise
        except Exception as e:
            logger.critical(f"Unhandled exception in core parser at line ~{current_line_num}", exc_info=True)
            state.termination_status = "ERROR"
            raise ParsingError("An unexpected critical error occurred during parsing.") from e

    # --- Finalization ---
    # Finalize termination status if it wasn't explicitly set by a parser
    if state.termination_status == "UNKNOWN":
        logger.warning("Termination status was not explicitly found. Assuming ERROR.")
        state.termination_status = "ERROR"

    # Finalize energy if not explicitly set by a dedicated parser
    if state.final_energy is None and state.scf is not None:
        state.final_energy = state.scf.energy
        logger.debug("Using final SCF energy as the calculation's final_energy.")

    return CalculationResult.model_validate(state.model_dump())
