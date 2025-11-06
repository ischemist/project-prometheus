"""tests for the get_api_docs method."""

from calcflow.common.input import CalculationInput


def test_get_api_docs_returns_string():
    """test that get_api_docs returns a non-empty string."""
    docs = CalculationInput.get_api_docs()
    assert isinstance(docs, str)
    assert len(docs) > 1000


def test_get_api_docs_contains_key_sections():
    """test that documentation contains all major sections."""
    docs = CalculationInput.get_api_docs()

    # check for main sections
    assert "CalculationInput API Reference" in docs
    assert "DESCRIPTION" in docs
    assert "CONSTRUCTOR" in docs
    assert "FLUENT API METHODS" in docs
    assert "USAGE EXAMPLES" in docs
    assert "VALIDATION" in docs


def test_get_api_docs_describes_core_methods():
    """test that all core fluent api methods are documented."""
    docs = CalculationInput.get_api_docs()

    # check for key methods
    assert ".set_tddft(" in docs
    assert ".set_solvation(" in docs
    assert ".set_mom(" in docs
    assert ".set_optimization(" in docs
    assert ".run_frequency_after_opt(" in docs
    assert ".set_level_of_theory(" in docs
    assert ".set_basis_set(" in docs
    assert ".set_unrestricted(" in docs
    assert ".export(" in docs
    assert ".to_json(" in docs
    assert ".from_json(" in docs


def test_get_api_docs_contains_usage_examples():
    """test that documentation includes practical usage examples."""
    docs = CalculationInput.get_api_docs()

    # check for import statements in examples
    assert "from calcflow.common.input import CalculationInput" in docs
    assert "from calcflow.geometry.static import Geometry" in docs

    # check for example patterns
    assert "Geometry.from_xyz_file" in docs
    assert ".export(" in docs
    assert "with open(" in docs


def test_get_api_docs_describes_all_specs():
    """test that all spec classes are documented."""
    docs = CalculationInput.get_api_docs()

    assert "TddftSpec" in docs
    assert "SolvationSpec" in docs
    assert "OptimizationSpec" in docs
    assert "MomSpec" in docs


def test_get_api_docs_includes_mom_transition_examples():
    """test that mom transition notation examples are documented."""
    docs = CalculationInput.get_api_docs()

    # check for various mom transition notations
    assert "HOMO->LUMO" in docs
    assert "HOMO-1->LUMO" in docs
    assert "HOMO->vac" in docs  # ionization example
    assert "5->6" in docs  # numeric notation


def test_get_api_docs_includes_validation_info():
    """test that validation requirements are documented."""
    docs = CalculationInput.get_api_docs()

    assert "spin_multiplicity must be >= 1" in docs
    assert "mom requires unrestricted=True" in docs
