# Agent Guidelines for CalcFlow

## Build/Test/Lint Commands
- **Run all tests**: `uv run pytest`
- **Run single test**: `uv run pytest tests/path/to/test_file.py::test_function_name -v`
- **Run tests by marker**: `uv run pytest -m unit` (markers: unit, contract, integration, regression)
- **Run with coverage**: `uv run pytest --cov=calcflow --cov-report=term-missing`
- **Lint**: `uvx ruff check` (add `--fix` to auto-fix)
- **Format**: `uvx ruff format` (add `--check` to verify without writing)
- **Type check**: `uvx ty check .`

## Code Style & Standards
- **Python version**: 3.13+
- **Line length**: 120 characters
- **Imports**: Organized by ruff (stdlib, third-party, local). Use `from collections.abc` not `typing` for Sequence/Mapping
- **Types**: Fully type-annotated. Use modern syntax (`float | None` not `Optional[float]`)
- **Models**: Standard library dataclasses with `frozen=True` for immutability
- **Naming**: lowercase_with_underscores for functions/variables/modules, CapitalCase for classes
- **Docstrings**: concise. Focus on "why" not "what"
- **Error handling**: Use custom exceptions from `calcflow.common.exceptions` (ParsingError, ConfigurationError, etc.)
- **Zero dependencies**: Never add external dependencies to main package (pydantic only)
- **Fluent API**: Return new immutable instances for modifications (e.g., `job.set_basis(...)` returns new job)

## Testing Philosophy (see docs/testing-spec.md)
- **unit**: fast (<1ms), isolated functions, no external deps, pure logic
- **contract**: fast (<10ms), parser produces correct data *structure* with small fixtures
- **integration**: moderate (10-500ms), multiple components working together, full files
- **regression**: slow, high-precision numerical checks, catches unintended changes
- **Guiding principles**: fixtures are king, use `parametrize` for edge cases, separate integration from regression

## Parser Design (see docs/parser-spec.md)
- **Strategy pattern**: CoreParser iterates lines, mutable ParseState holds results, BlockParser registry handles specific sections
- **BlockParser protocol**: `matches()` checks if parser handles line (fast, no state mutation), `parse()` consumes lines and updates state
- **ParseState contract**: single mutable scratchpad, parsers write results to it, control flags prevent duplicate parsing, `buffered_line` for over-reading
- **Adding parser**: (1) identify block markers, (2) create pydantic model in `common/models.py`, (3) implement BlockParser with `matches()`/`parse()`, (4) register in parser_registry, (5) write contract test
- **Critical**: parsers must set completion flags (e.g., `state.parsed_scf = True`), handle `buffered_line` if over-reading, raise ParsingError for critical issues
