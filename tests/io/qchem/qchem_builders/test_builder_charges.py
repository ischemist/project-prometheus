"""Charges / population analysis tests for Q-Chem builder.

Tests for ChargesSpec in Q-Chem input generation.

Test Structure:
- Unit tests: REM variable generation for each charge scheme
- Contract tests: correct keys present / absent, CM5 auto-enables Hirshfeld
- Integration tests: fluent API workflows with .set_charges()
- Regression tests: semantic validation, MOM two-job structure
"""

from __future__ import annotations

import pytest

from calcflow.common.exceptions import ConfigurationError
from calcflow.common.input import CalculationInput, ChargesSpec
from tests.io.qchem.qchem_builders.conftest import (
    assert_rem_value,
    assert_two_job_structure,
    extract_job,
    parse_qchem_input,
)

# =============================================================================
# UNIT TESTS: REM variable generation
# =============================================================================


@pytest.mark.unit
def test_hirshfeld_only_emits_hirshfeld_true(qchem_builder, h2o_geometry):
    """HIRSHFELD TRUE should appear when hirshfeld=True."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(hirshfeld=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)


@pytest.mark.unit
def test_cm5_emits_both_hirshfeld_and_cm5(qchem_builder, h2o_geometry):
    """CM5 TRUE and HIRSHFELD TRUE should both appear when cm5=True."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(cm5=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)
    assert_rem_value(parsed.rem_block, "CM5", True)


@pytest.mark.unit
def test_hirshiter_emits_hirshiter_true(qchem_builder, h2o_geometry):
    """HIRSHITER TRUE should appear when hirshiter=True."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(hirshfeld=True, hirshiter=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHITER", True)


@pytest.mark.unit
def test_hirshiter_emits_default_threshold(qchem_builder, h2o_geometry):
    """HIRSHITER_THRESH should default to 5 when hirshiter=True."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(hirshfeld=True, hirshiter=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHITER_THRESH", 5)


@pytest.mark.unit
def test_hirshiter_emits_custom_threshold(qchem_builder, h2o_geometry):
    """HIRSHITER_THRESH should follow explicit hirshiter_thresh."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(hirshfeld=True, hirshiter=True, hirshiter_thresh=9),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHITER_THRESH", 9)


@pytest.mark.unit
def test_hirshiter_threshold_not_emitted_without_hirshiter(qchem_builder, h2o_geometry):
    """HIRSHITER_THRESH should not appear when hirshiter=False."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(hirshfeld=True, hirshiter=False, hirshiter_thresh=9),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert "hirshiter_thresh" not in parsed.rem_block.lower()


@pytest.mark.unit
def test_mulliken_false_emits_pop_mulliken_0(qchem_builder, h2o_geometry):
    """POP_MULLIKEN 0 should appear when mulliken=False."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(mulliken=False, hirshfeld=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "POP_MULLIKEN", 0)


@pytest.mark.unit
def test_no_charges_spec_emits_no_charge_rem_vars(qchem_builder, h2o_geometry, minimal_spec):
    """Without ChargesSpec, no charge-related REM vars should appear."""
    result = qchem_builder.build(minimal_spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    rem_lower = parsed.rem_block.lower()
    assert "hirshfeld" not in rem_lower
    assert "cm5" not in rem_lower
    assert "pop_mulliken" not in rem_lower
    assert "hirshiter" not in rem_lower


# =============================================================================
# CONTRACT TESTS: ChargesSpec auto-enable and validation
# =============================================================================


@pytest.mark.contract
def test_cm5_without_hirshfeld_auto_enables_hirshfeld():
    """ChargesSpec should auto-enable hirshfeld when cm5=True."""
    spec = ChargesSpec(cm5=True)
    assert spec.hirshfeld is True
    assert spec.cm5 is True


@pytest.mark.contract
def test_hirshiter_without_hirshfeld_auto_enables_hirshfeld():
    """ChargesSpec should auto-enable hirshfeld when hirshiter=True."""
    spec = ChargesSpec(hirshiter=True)
    assert spec.hirshfeld is True
    assert spec.hirshiter is True


@pytest.mark.contract
def test_cm5_with_explicit_hirshfeld_unchanged():
    """ChargesSpec with cm5=True and hirshfeld=True should be stable."""
    spec = ChargesSpec(hirshfeld=True, cm5=True)
    assert spec.hirshfeld is True
    assert spec.cm5 is True


@pytest.mark.contract
def test_hirshfeld_default_is_false():
    """Default ChargesSpec should not enable any non-standard analyses."""
    spec = ChargesSpec()
    assert spec.mulliken is True
    assert spec.hirshfeld is False
    assert spec.cm5 is False
    assert spec.hirshiter is False
    assert spec.hirshiter_thresh == 5


@pytest.mark.contract
def test_charged_system_hirshfeld_raises(qchem_builder, h2o_geometry):
    """Standard Hirshfeld on charged systems should raise ConfigurationError."""
    spec = CalculationInput(
        charge=1,
        spin_multiplicity=2,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        charges=ChargesSpec(hirshfeld=True),
    )
    with pytest.raises(ConfigurationError, match="hirshiter=True"):
        qchem_builder.build(spec, h2o_geometry)


@pytest.mark.contract
def test_charged_system_hirshiter_does_not_raise(qchem_builder, h2o_geometry):
    """Hirshfeld-I (hirshiter=True) on charged system should not raise."""
    spec = CalculationInput(
        charge=1,
        spin_multiplicity=2,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        unrestricted=True,
        charges=ChargesSpec(hirshfeld=True, hirshiter=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    assert "$rem" in result


@pytest.mark.contract
def test_charges_spec_does_not_affect_unrelated_rem(qchem_builder, h2o_geometry):
    """ChargesSpec should not affect METHOD, BASIS, JOBTYPE, or UNRESTRICTED."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="wb97x-d3",
        basis_set="def2-tzvp",
        charges=ChargesSpec(hirshfeld=True, cm5=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "METHOD", "wb97x-d3")
    assert_rem_value(parsed.rem_block, "BASIS", "def2-tzvp")
    assert_rem_value(parsed.rem_block, "JOBTYPE", "sp")


# =============================================================================
# INTEGRATION TESTS: Fluent API
# =============================================================================


@pytest.mark.integration
def test_set_charges_hirshfeld_cm5_workflow(h2o_geometry):
    """set_charges(hirshfeld=True, cm5=True) should produce correct REM vars."""
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
    ).set_charges(hirshfeld=True, cm5=True)

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)
    assert_rem_value(parsed.rem_block, "CM5", True)


@pytest.mark.integration
def test_set_charges_cm5_only_auto_enables_hirshfeld(h2o_geometry):
    """set_charges(cm5=True) alone should auto-enable hirshfeld in the output."""
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
    ).set_charges(cm5=True)

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)
    assert_rem_value(parsed.rem_block, "CM5", True)


@pytest.mark.integration
def test_set_charges_chaining_with_solvation(h2o_geometry):
    """set_charges() should chain cleanly with other fluent methods."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="wb97x-d3",
            basis_set="def2-tzvp",
        )
        .set_unrestricted()
        .set_solvation("smd", "water")
        .set_charges(hirshfeld=True, cm5=True)
    )

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)
    assert_rem_value(parsed.rem_block, "CM5", True)
    assert_rem_value(parsed.rem_block, "UNRESTRICTED", True)


@pytest.mark.integration
def test_set_charges_with_hirshiter_threshold(h2o_geometry):
    """set_charges(hirshiter_thresh=...) should emit HIRSHITER_THRESH when hirshiter is enabled."""
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
    ).set_charges(hirshfeld=True, hirshiter=True, hirshiter_thresh=9)

    result = calc.export("qchem", h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHITER", True)
    assert_rem_value(parsed.rem_block, "HIRSHITER_THRESH", 9)


# =============================================================================
# REGRESSION TESTS: MOM two-job structure
# =============================================================================


@pytest.mark.regression
def test_charges_rem_in_both_mom_jobs(h2o_geometry):
    """ChargesSpec should produce charge REM vars in both MOM jobs."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="wb97x-d3",
            basis_set="def2-tzvp",
            unrestricted=True,
        )
        .set_mom("HOMO->LUMO")
        .set_charges(hirshfeld=True, cm5=True)
    )

    result = calc.export("qchem", h2o_geometry)
    assert_two_job_structure(result)

    job1_text = extract_job(result, 1)
    job2_text = extract_job(result, 2)

    job1 = parse_qchem_input(job1_text)
    job2 = parse_qchem_input(job2_text)

    assert_rem_value(job1.rem_block, "HIRSHFELD", True)
    assert_rem_value(job1.rem_block, "CM5", True)
    assert_rem_value(job2.rem_block, "HIRSHFELD", True)
    assert_rem_value(job2.rem_block, "CM5", True)


@pytest.mark.regression
def test_hirshfeld_cm5_mulliken_together(qchem_builder, h2o_geometry):
    """All three schemes enabled together should emit all relevant REM vars."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(mulliken=True, hirshfeld=True, cm5=True),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)
    assert_rem_value(parsed.rem_block, "CM5", True)
    # mulliken=True means POP_MULLIKEN should NOT be suppressed
    rem_lower = parsed.rem_block.lower()
    assert "pop_mulliken" not in rem_lower


@pytest.mark.regression
def test_hirshfeld_only_no_cm5_key(qchem_builder, h2o_geometry):
    """Hirshfeld without CM5 should not emit CM5 key."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="6-31g",
        charges=ChargesSpec(hirshfeld=True, cm5=False),
    )
    result = qchem_builder.build(spec, h2o_geometry)
    parsed = parse_qchem_input(result)
    assert_rem_value(parsed.rem_block, "HIRSHFELD", True)
    assert "cm5" not in parsed.rem_block.lower()
