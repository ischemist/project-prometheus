---
icon: lucide/flask-conical
---

# Testing

CalcFlow uses four test levels, each targeting a distinct layer of complexity: pure logic, parser contracts, component integration, and numerical regression.

## The Four Levels

| Marker | Scope | Speed | Asserts |
| :--- | :--- | :--- | :--- |
| `unit` | Single function or method | <1 ms | Correctness of pure logic |
| `contract` | Single `BlockParser` + minimal fixture | <10 ms | Data *structure* (not precision) |
| `integration` | Multiple components, real output files | 10–500 ms | Functional correctness |
| `regression` | Same scope as integration | Slow | Exact numerical values |

### Unit tests

Test individual functions in isolation. No file I/O, no parsers, no external state. Input and expected output are constructed inline.

```python
@pytest.mark.unit
def test_atom_validates_symbol():
    from calcflow.common.models import Atom
    with pytest.raises(ValueError):
        Atom(symbol="Xx", x=0.0, y=0.0, z=0.0)
```

### Contract tests

Verify that a `BlockParser` produces the correct data *structure* from a minimal fixture. The fixture is the smallest possible snippet of output text that exercises the parser — not a real output file.

```python
MULLIKEN_FIXTURE = """\
 Mulliken charges:
               1
     1  O   -0.8476
     2  H    0.4238
     3  H    0.4238
 Sum of Mulliken charges =   0.00000
"""

@pytest.mark.contract
def test_mulliken_charges_parsed():
    result = parse_qchem_output(MULLIKEN_FIXTURE)
    mulliken = result.get_charges("Mulliken")
    assert mulliken is not None
    assert len(mulliken.charges) == 3
    assert isinstance(mulliken.charges[0], float)
```

Contract tests are the workhorses of the test suite. Write one for every `BlockParser` you add.

### Integration tests

Test the full parse pipeline against real output files from `tests/testing_data/`. Assert on functional correctness — the right fields are populated, the right states are found — but do not assert on exact floating-point values.

```python
@pytest.mark.integration
def test_qchem_tddft_sp_parses_excited_states(testing_data_path):
    text = (testing_data_path / "qchem" / "h2o" / "tddft_sp.out").read_text()
    result = parse_qchem_output(text)
    assert result.termination_status == "NORMAL"
    assert result.tddft is not None
    assert len(result.tddft.tddft_states) > 0
    assert result.tddft.tddft_states[0].excitation_energy_ev > 0
```

### Regression tests

Same scope as integration, but assert on exact values using `pytest.approx`. These tests catch unintended numerical changes — a parser refactor that accidentally changes a parsed energy value.

```python
@pytest.mark.regression
def test_qchem_tddft_sp_first_state_energy(testing_data_path):
    text = (testing_data_path / "qchem" / "h2o" / "tddft_sp.out").read_text()
    result = parse_qchem_output(text)
    state = result.tddft.tddft_states[0]
    assert state.excitation_energy_ev == pytest.approx(7.654, abs=1e-3)
    assert state.oscillator_strength  == pytest.approx(0.0821, abs=1e-4)
```

## Directory Structure

The test directory mirrors the source layout:

```
tests/
    conftest.py              # testing_data_path fixture (shared)
    smoke_test.py            # end-to-end smoke tests

    common/
        test_api_docs.py
        test_input_serialization.py
        test_results_serialization.py

    geometry/
        test_static.py
        test_topology.py
        test_trajectory.py
        test_annotated.py

    io/
        test_peekable.py
        orca/
            orca_builders/
            orca_parsers/
        qchem/
            qchem_builders/
            qchem_parsers/
                tddft/
                adc/

    postprocess/
        test_postprocess.py

    testing_data/
        geometries/           # .xyz files
        orca/h2o/             # .out files
        qchem/h2o/            # .out files
```

Every test file lives in a directory that mirrors its corresponding source module. Parser tests live under `io/<program>/qchem_parsers/`, builder tests under `io/<program>/<program>_builders/`.

## Fixtures

### `testing_data_path`

Defined in `tests/conftest.py`. Returns a `Path` to `tests/testing_data/`:

```python
@pytest.fixture
def testing_data_path():
    return Path(__file__).parent / "testing_data"
```

Use it in integration and regression tests:

```python
def test_something(testing_data_path):
    text = (testing_data_path / "qchem" / "h2o" / "sp.out").read_text()
```

### Shared builder fixtures

For builder tests, define common fixtures in a `conftest.py` at the appropriate level:

```python
@pytest.fixture
def water_geometry():
    return Geometry.from_lines([
        "O    0.000000    0.000000    0.119748",
        "H    0.000000    0.756950   -0.478993",
        "H    0.000000   -0.756950   -0.478993",
    ], source="water")

@pytest.fixture
def minimal_spec():
    return CalculationInput(
        charge=0, spin_multiplicity=1,
        task="energy", level_of_theory="B3LYP", basis_set="def2-SVP"
    )
```

## Testing Input Builders

Builder tests require a different strategy from parser tests. The output is structured text — exact string comparison is brittle because whitespace and keyword ordering may change without affecting correctness.

**The correct approach**: assert on *semantics*, not on exact text.

=== "Do this"

    ```python
    @pytest.mark.contract
    def test_tddft_keywords_present(minimal_spec, water_geometry):
        calc   = minimal_spec.set_tddft(nroots=5, singlets=True, triplets=False)
        output = calc.export("qchem", water_geometry)

        assert "cis_n_roots" in output.lower()
        assert "5" in output
        assert "$molecule" in output
        assert "$end" in output
    ```

=== "Don't do this"

    ```python
    # Brittle — breaks on any formatting change
    @pytest.mark.contract
    def test_tddft_exact_output(minimal_spec, water_geometry):
        calc   = minimal_spec.set_tddft(nroots=5, singlets=True, triplets=False)
        output = calc.export("qchem", water_geometry)
        assert output == EXPECTED_EXACT_STRING
    ```

For regression tests on builders, the most robust approach is **round-tripping**: generate the input, parse it back into structured data, and assert on the parsed values.

## Running Tests

```bash
# All tests
uv run pytest

# By marker
uv run pytest -m unit
uv run pytest -m contract
uv run pytest -m "integration or regression"

# Single file
uv run pytest tests/io/qchem/qchem_parsers/test_qchem_sp_scf.py -v

# Single test
uv run pytest tests/io/qchem/qchem_parsers/test_qchem_sp_scf.py::test_scf_converged -v

# With coverage
uv run pytest --cov=calcflow --cov-report=term-missing
```

## The Testing Pyramid

Aim for this shape:

```
         ▲
        / \
       /   \   regression (few, expensive)
      /─────\
     /       \  integration (moderate)
    /─────────\
   /           \ contract (many, fast)
  /─────────────\
 /               \ unit (most, fastest)
/─────────────────\
```

Most of the safety net comes from unit and contract tests. Integration and regression tests are the final check that everything works end-to-end with real data.
