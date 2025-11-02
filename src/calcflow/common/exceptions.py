"""centralized custom exceptions for the calcflow library."""


class CalcflowError(Exception):
    """base class for exceptions in the calcflow package."""

    pass


class NotSupportedError(CalcflowError):
    """exception raised for features that are not yet implemented or supported."""

    pass


class InputGenerationError(CalcflowError):
    """exception raised for errors during input file generation."""

    pass


class ConfigurationError(CalcflowError):
    """exception raised for configuration-related errors (e.g., inconsistent settings)."""

    pass


class ParsingError(CalcflowError):
    """exception raised for errors during file parsing."""

    pass


class InternalCodeError(CalcflowError):
    """exception raised for logic errors that indicate a bug in the library itself."""

    pass


class ValidationError(CalcflowError):
    """exception raised for validation errors in data models."""

    pass
