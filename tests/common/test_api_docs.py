"""tests for the auto-generated api documentation methods."""

from calcflow.common.input import CalculationInput
from calcflow.common.results import CalculationResult

# =============================================================================
# CalculationInput.get_quick_ref()
# =============================================================================


def test_get_quick_ref_returns_string():
    ref = CalculationInput.get_quick_ref()
    assert isinstance(ref, str)
    assert len(ref) > 200


def test_get_quick_ref_has_constructor_section():
    ref = CalculationInput.get_quick_ref()
    assert "CONSTRUCTOR" in ref
    # required fields must appear
    assert "charge" in ref
    assert "spin_multiplicity" in ref
    assert "task" in ref
    assert "level_of_theory" in ref
    assert "basis_set" in ref


def test_get_quick_ref_has_fluent_methods_section():
    ref = CalculationInput.get_quick_ref()
    assert "FLUENT METHODS" in ref
    # all key methods must be listed
    for method in (
        "set_tddft",
        "set_solvation",
        "set_mom",
        "set_optimization",
        "run_frequency_after_opt",
        "set_level_of_theory",
        "set_basis_set",
        "set_unrestricted",
        "set_cores",
        "set_memory_per_core",
        "enable_ri_for_orca",
    ):
        assert method in ref, f"method '{method}' missing from get_quick_ref()"


def test_get_quick_ref_has_export_section():
    ref = CalculationInput.get_quick_ref()
    assert "export" in ref
    assert "to_json" in ref
    assert "from_json" in ref


def test_get_quick_ref_has_minimal_example():
    ref = CalculationInput.get_quick_ref()
    assert "Geometry.from_xyz_file" in ref
    assert "CalculationInput(" in ref


# =============================================================================
# CalculationInput.get_method_docs()
# =============================================================================


def test_get_method_docs_catalogue():
    """calling with no argument returns a method catalogue."""
    catalogue = CalculationInput.get_method_docs()
    assert isinstance(catalogue, str)
    assert "set_tddft" in catalogue
    assert "set_solvation" in catalogue
    assert "set_mom" in catalogue


def test_get_method_docs_specific_method():
    """calling with a method name returns that method's full docs."""
    docs = CalculationInput.get_method_docs("set_tddft")
    assert "set_tddft" in docs
    # should contain signature info
    assert "nroots" in docs
    # should contain docstring content
    assert "singlets" in docs or "triplets" in docs


def test_get_method_docs_mom_includes_transition_notation():
    docs = CalculationInput.get_method_docs("set_mom")
    assert "HOMO->LUMO" in docs
    assert "HOMO->vac" in docs
    assert "5->6" in docs


def test_get_method_docs_unknown_method():
    result = CalculationInput.get_method_docs("nonexistent_method")
    assert "no method" in result.lower() or "nonexistent" in result


def test_get_method_docs_includes_validation_context():
    """set_mom docstring covers unrestricted requirement."""
    docs = CalculationInput.get_method_docs("set_mom")
    assert "unrestricted" in docs


# =============================================================================
# CalculationInput.get_api_docs() — compatibility alias
# =============================================================================


def test_get_api_docs_is_nonempty_string():
    docs = CalculationInput.get_api_docs()
    assert isinstance(docs, str)
    assert len(docs) > 500


def test_get_api_docs_covers_key_methods():
    """alias output still mentions all fluent methods."""
    docs = CalculationInput.get_api_docs()
    for method in ("set_tddft", "set_solvation", "set_mom", "export"):
        assert method in docs, f"'{method}' missing from get_api_docs()"


# =============================================================================
# CalculationResult.get_schema()
# =============================================================================


def test_get_schema_returns_string():
    schema = CalculationResult.get_schema()
    assert isinstance(schema, str)
    assert len(schema) > 500


def test_get_schema_has_parse_functions():
    schema = CalculationResult.get_schema()
    assert "parse_qchem_output" in schema
    assert "parse_orca_output" in schema
    assert "parse_qchem_multi_job_output" in schema


def test_get_schema_has_top_level_fields():
    schema = CalculationResult.get_schema()
    for field in (
        "termination_status",
        "final_energy",
        "scf",
        "orbitals",
        "tddft",
        "dispersion",
        "atomic_charges",
        "raw_output",
    ):
        assert field in schema, f"field '{field}' missing from get_schema()"


def test_get_schema_recurses_into_nested_models():
    schema = CalculationResult.get_schema()
    # ScfResults fields should appear
    assert "n_iterations" in schema or "converged" in schema
    # ExcitedState fields
    assert "excitation_energy_ev" in schema
    assert "oscillator_strength" in schema
    # Atom fields
    assert "symbol" in schema


def test_get_schema_shows_types():
    schema = CalculationResult.get_schema()
    assert "float" in schema
    assert "int" in schema
    assert "str" in schema
    assert "bool" in schema
    assert "None" in schema


def test_get_schema_mentions_units():
    schema = CalculationResult.get_schema()
    assert "Hartree" in schema
    assert "Angstrom" in schema
    assert "Debye" in schema
    assert "eV" in schema


def test_get_schema_mentions_raw_output_exclusion():
    schema = CalculationResult.get_schema()
    assert "raw_output" in schema
    # should note it's excluded from serialization
    assert "excludes" in schema.lower() or "excluded" in schema.lower()


def test_get_schema_mentions_index_conventions():
    schema = CalculationResult.get_schema()
    assert "0-based" in schema
    assert "1-based" in schema


def test_get_schema_has_orbital_fields():
    schema = CalculationResult.get_schema()
    assert "alpha_orbitals" in schema or "alpha_homo_index" in schema


# =============================================================================
# CalculationResult.get_api_docs() — compatibility alias
# =============================================================================


def test_results_get_api_docs_is_alias_for_get_schema():
    assert CalculationResult.get_api_docs() == CalculationResult.get_schema()


def test_results_get_api_docs_nonempty():
    docs = CalculationResult.get_api_docs()
    assert isinstance(docs, str)
    assert len(docs) > 500
