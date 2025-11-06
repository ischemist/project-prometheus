"""tests for the get_api_docs method."""

from calcflow.common.input import CalculationInput
from calcflow.common.results import CalculationResult


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


# --- CalculationResult tests ---


def test_results_get_api_docs_returns_string():
    """test that get_api_docs returns a non-empty string."""
    docs = CalculationResult.get_api_docs()
    assert isinstance(docs, str)
    assert len(docs) > 2000


def test_results_get_api_docs_contains_key_sections():
    """test that documentation contains all major sections."""
    docs = CalculationResult.get_api_docs()

    # check for main sections
    assert "CalculationResult API Reference" in docs
    assert "DESCRIPTION" in docs
    assert "TOP-LEVEL MODEL: CalculationResult" in docs
    assert "FUNDAMENTAL MODELS" in docs
    assert "SCF (SELF-CONSISTENT FIELD) MODELS" in docs
    assert "MOLECULAR ORBITALS MODEL" in docs
    assert "PROPERTY MODELS" in docs
    assert "TDDFT & EXCITED STATE MODELS" in docs
    assert "SERIALIZATION METHODS" in docs
    assert "USAGE EXAMPLES" in docs
    assert "IMPORTANT NOTES" in docs


def test_results_get_api_docs_describes_core_models():
    """test that all core result models are documented."""
    docs = CalculationResult.get_api_docs()

    # fundamental models
    assert "Atom:" in docs
    assert "Orbital:" in docs

    # scf models
    assert "ScfIteration:" in docs
    assert "ScfEnergyComponents:" in docs
    assert "ScfResults:" in docs

    # orbital model
    assert "OrbitalsSet:" in docs

    # property models
    assert "AtomicCharges:" in docs
    assert "DipoleMoment:" in docs
    assert "MultipoleResults:" in docs
    assert "DispersionCorrection:" in docs
    assert "SmdResults:" in docs
    assert "TimingResults:" in docs

    # tddft models
    assert "ExcitedState:" in docs
    assert "OrbitalTransition:" in docs
    assert "NTOContribution:" in docs
    assert "NTOStateAnalysis:" in docs
    assert "TddftResults:" in docs
    assert "ExcitonAnalysis:" in docs
    assert "UnrelaxedDensityMatrix:" in docs
    assert "TransitionDensityMatrix:" in docs

    # metadata
    assert "CalculationMetadata:" in docs


def test_results_get_api_docs_describes_serialization_methods():
    """test that all serialization methods are documented."""
    docs = CalculationResult.get_api_docs()

    assert ".to_dict()" in docs
    assert ".from_dict(" in docs
    assert ".to_json(" in docs
    assert ".from_json(" in docs


def test_results_get_api_docs_contains_usage_examples():
    """test that documentation includes practical usage examples."""
    docs = CalculationResult.get_api_docs()

    # check for import statements
    assert "from calcflow.io.qchem import parse_qchem_output" in docs
    assert "from calcflow.io.orca import parse_orca_output" in docs

    # check for common parsing patterns
    assert "parse_qchem_output" in docs
    assert "parse_orca_output" in docs
    assert 'termination_status == "NORMAL"' in docs

    # check for result access patterns
    assert "result.scf" in docs
    assert "result.orbitals" in docs
    assert "result.tddft" in docs
    assert "result.final_energy" in docs


def test_results_get_api_docs_describes_field_types():
    """test that field types are documented."""
    docs = CalculationResult.get_api_docs()

    # check for type annotations
    assert "Literal[" in docs
    assert "Sequence[" in docs
    assert "Mapping[" in docs
    assert "| None" in docs or "None" in docs
    assert "float" in docs
    assert "int" in docs
    assert "str" in docs
    assert "bool" in docs


def test_results_get_api_docs_mentions_units():
    """test that units are documented."""
    docs = CalculationResult.get_api_docs()

    assert "Hartree" in docs
    assert "Angstrom" in docs
    assert "Debye" in docs
    assert "eV" in docs


def test_results_get_api_docs_mentions_raw_output_exclusion():
    """test that raw_output exclusion is documented."""
    docs = CalculationResult.get_api_docs()

    assert "raw_output" in docs
    assert "exclude" in docs.lower() or "excluded" in docs.lower()


def test_results_get_api_docs_includes_tddft_examples():
    """test that tddft analysis examples are documented."""
    docs = CalculationResult.get_api_docs()

    assert "tddft_states" in docs
    assert "excited" in docs.lower()
    assert "excitation_energy_ev" in docs
    assert "oscillator_strength" in docs


def test_results_get_api_docs_includes_orbital_examples():
    """test that orbital analysis examples are documented."""
    docs = CalculationResult.get_api_docs()

    assert "HOMO" in docs
    assert "LUMO" in docs
    assert "alpha_orbitals" in docs


def test_results_get_api_docs_includes_important_notes():
    """test that important notes about usage are documented."""
    docs = CalculationResult.get_api_docs()

    assert "immutable" in docs.lower() or "frozen" in docs.lower()
    assert "0-based" in docs  # for indices
    assert "1-based" in docs  # for state numbers
