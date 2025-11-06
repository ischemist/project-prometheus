"""
Tests for the slurm job submission script generation module.

Following the CalcFlow testing philosophy:
- unit: test individual methods in isolation (<1ms)
- contract: test build() produces correct structure (<10ms)
- integration: test full workflow with CalculationInput (10-100ms)
- regression: semantic validation of known outputs
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from calcflow.common.input import CalculationInput
from calcflow.common.results import Atom
from calcflow.geometry.static import Geometry
from calcflow.io.orca.builder import OrcaBuilder
from calcflow.io.qchem.builder import QchemBuilder
from calcflow.slurm import SlurmJob

# --- Fixtures ---


@pytest.fixture
def h2o_geometry() -> Geometry:
    """Minimal water molecule for testing."""
    return Geometry(
        comment="water molecule",
        atoms=(
            Atom(symbol="O", x=0.0, y=0.0, z=0.0),
            Atom(symbol="H", x=0.0, y=0.0, z=1.0),
            Atom(symbol="H", x=0.0, y=1.0, z=0.0),
        ),
    )


@pytest.fixture
def minimal_spec() -> CalculationInput:
    """Bare minimum spec for unit testing."""
    return CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
    )


@pytest.fixture
def minimal_slurm_job() -> SlurmJob:
    """Bare minimum slurm job for unit testing."""
    return SlurmJob(
        job_name="test_job",
        time="01:00:00",
        n_cores=4,
    )


@pytest.fixture
def orca_builder() -> OrcaBuilder:
    """Orca builder instance."""
    return OrcaBuilder()


@pytest.fixture
def qchem_builder() -> QchemBuilder:
    """Q-Chem builder instance."""
    return QchemBuilder()


# --- Unit Tests: SlurmJob.build() ---


@pytest.mark.unit
def test_build_minimal_script(minimal_slurm_job):
    """Minimal job should generate basic sbatch directives."""
    script = minimal_slurm_job.build([], "echo test")

    # Check shebang
    assert script.startswith("#!/bin/sh\n")

    # Check required sbatch directives
    assert "#SBATCH -J test_job" in script
    assert "#SBATCH --output sbatch.out" in script
    assert "#SBATCH --error sbatch.err" in script
    assert "#SBATCH --time 01:00:00" in script

    # Check launch command
    assert "echo test" in script
    assert script.strip().endswith("echo test")


@pytest.mark.unit
def test_build_with_all_optional_fields():
    """Job with all optional fields should include all directives."""
    job = SlurmJob(
        job_name="complete_job",
        time="02:30:00",
        n_cores=16,
        memory_mb=64000,
        partition="gpu",
        constraint="v100",
        account="myproject",
        queue="high",
    )
    script = job.build([], "sleep 10")

    assert "#SBATCH -J complete_job" in script
    assert "#SBATCH --time 02:30:00" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --qos=high" in script
    assert "#SBATCH --account=myproject" in script
    assert "#SBATCH --constraint=v100" in script
    assert "#SBATCH --mem=64000" in script


@pytest.mark.unit
@pytest.mark.parametrize("memory", [1000, 8000, 32000, 128000])
def test_build_memory_directive(memory):
    """Memory directive should be included when memory_mb is set."""
    job = SlurmJob(
        job_name="mem_test",
        time="01:00:00",
        n_cores=4,
        memory_mb=memory,
    )
    script = job.build([], "echo test")
    assert f"#SBATCH --mem={memory}" in script


@pytest.mark.unit
def test_build_without_memory(minimal_slurm_job):
    """Script without memory_mb should not have memory directive."""
    script = minimal_slurm_job.build([], "echo test")
    assert "--mem=" not in script


@pytest.mark.unit
@pytest.mark.parametrize("partition", ["normal", "gpu", "high-mem"])
def test_build_partition_directive(partition):
    """Partition directive should be included when set."""
    job = SlurmJob(
        job_name="part_test",
        time="01:00:00",
        n_cores=4,
        partition=partition,
    )
    script = job.build([], "echo test")
    assert f"#SBATCH --partition={partition}" in script


@pytest.mark.unit
def test_build_without_partition(minimal_slurm_job):
    """Script without partition should not have partition directive."""
    script = minimal_slurm_job.build([], "echo test")
    assert "--partition=" not in script


@pytest.mark.unit
@pytest.mark.parametrize("queue", ["normal", "high", "long"])
def test_build_qos_directive(queue):
    """QOS directive should be included when queue is set."""
    job = SlurmJob(
        job_name="qos_test",
        time="01:00:00",
        n_cores=4,
        queue=queue,
    )
    script = job.build([], "echo test")
    assert f"#SBATCH --qos={queue}" in script


@pytest.mark.unit
def test_build_without_qos(minimal_slurm_job):
    """Script without queue should not have qos directive."""
    script = minimal_slurm_job.build([], "echo test")
    assert "--qos=" not in script


@pytest.mark.unit
def test_build_with_modules():
    """Job with modules should include module load commands."""
    job = SlurmJob(
        job_name="module_test",
        time="01:00:00",
        n_cores=4,
        modules=["gcc/11.2.0", "openmpi/4.1.1"],
    )
    script = job.build([], "mpirun ./program")

    assert "module load gcc/11.2.0" in script
    assert "module load openmpi/4.1.1" in script


@pytest.mark.unit
def test_build_without_modules(minimal_slurm_job):
    """Script without modules should not have module load commands."""
    script = minimal_slurm_job.build([], "echo test")
    assert "module load" not in script


@pytest.mark.unit
def test_build_program_directives():
    """Program-specific directives should be included in output."""
    job = SlurmJob(
        job_name="directives_test",
        time="01:00:00",
        n_cores=8,
    )
    program_directives = [
        "#SBATCH --ntasks=8",
        "#SBATCH --nodes=1",
    ]
    script = job.build(program_directives, "orca input.inp")

    assert "#SBATCH --ntasks=8" in script
    assert "#SBATCH --nodes=1" in script


@pytest.mark.unit
def test_build_output_structure(minimal_slurm_job):
    """Output should have correct section structure."""
    script = minimal_slurm_job.build(
        ["#SBATCH --ntasks=4"],
        "echo launch",
    )
    lines = script.split("\n")

    # Find shebang
    assert lines[0] == "#!/bin/sh"

    # Find directives (should come first)
    directive_lines = [line for line in lines if line.startswith("#SBATCH")]
    assert len(directive_lines) >= 4  # job, output, error, time

    # Find blank line (separator)
    non_empty_lines = [line for line in lines if line.strip()]
    # Last line should be the launch command
    assert non_empty_lines[-1] == "echo launch"


@pytest.mark.unit
def test_build_launch_command_preserved():
    """Launch command should appear exactly as provided."""
    commands = [
        "echo hello",
        "$(which orca) input.inp > output.out",
        "qchem -nt 16 h2o.in h2o.out",
        "python script.py --arg=value",
    ]
    for cmd in commands:
        job = SlurmJob(
            job_name="cmd_test",
            time="01:00:00",
            n_cores=4,
        )
        script = job.build([], cmd)
        assert cmd in script
        # Verify it's the last non-empty line
        assert script.strip().endswith(cmd)


# --- Unit Tests: SlurmJob Fluent API ---


@pytest.mark.unit
def test_set_partition_returns_new_instance(minimal_slurm_job):
    """set_partition() should return new instance."""
    updated_job = minimal_slurm_job.set_partition("gpu")

    # Check new instance has updated value
    assert updated_job.partition == "gpu"

    # Check original is unchanged
    assert minimal_slurm_job.partition is None

    # Check they are different objects
    assert updated_job is not minimal_slurm_job


@pytest.mark.unit
def test_set_partition_updates_script():
    """Updated job should produce script with new partition."""
    job = SlurmJob(
        job_name="test",
        time="01:00:00",
        n_cores=4,
    )
    updated = job.set_partition("high-mem")
    script = updated.build([], "echo test")

    assert "#SBATCH --partition=high-mem" in script
    assert "#SBATCH --partition=" not in job.build([], "echo test")


@pytest.mark.unit
def test_add_modules_returns_new_instance(minimal_slurm_job):
    """add_modules() should return new instance."""
    updated_job = minimal_slurm_job.add_modules(["gcc/11.2.0"])

    # Check new instance has updated value
    assert updated_job.modules == ["gcc/11.2.0"]

    # Check original is unchanged
    assert minimal_slurm_job.modules is None

    # Check they are different objects
    assert updated_job is not minimal_slurm_job


@pytest.mark.unit
def test_add_modules_updates_script():
    """Updated job should produce script with modules."""
    job = SlurmJob(
        job_name="test",
        time="01:00:00",
        n_cores=4,
    )
    updated = job.add_modules(["python/3.10", "gcc/11.2.0"])
    script = updated.build([], "python script.py")

    assert "module load python/3.10" in script
    assert "module load gcc/11.2.0" in script
    assert "module load" not in job.build([], "python script.py")


@pytest.mark.unit
def test_fluent_api_chaining():
    """Multiple set operations should chain correctly."""
    job = SlurmJob(
        job_name="chain_test",
        time="01:00:00",
        n_cores=8,
    )
    updated = job.set_partition("gpu").add_modules(["cuda/11.2"])

    assert updated.partition == "gpu"
    assert updated.modules == ["cuda/11.2"]
    assert job.partition is None
    assert job.modules is None


# --- Unit Tests: Builder get_slurm_directives() ---


@pytest.mark.unit
@pytest.mark.parametrize("n_cores", [1, 4, 8, 16, 32])
def test_orca_get_slurm_directives_cores(orca_builder, minimal_spec, n_cores):
    """ORCA directives should scale with core count."""
    spec = replace(minimal_spec, n_cores=n_cores)
    directives = orca_builder.get_slurm_directives(spec)

    # ORCA uses MPI (ntasks)
    assert any(f"--ntasks={n_cores}" in d for d in directives)
    assert any("--nodes=1" in d for d in directives)


@pytest.mark.unit
def test_orca_get_slurm_directives_single_core(orca_builder, minimal_spec):
    """ORCA with single core should have ntasks=1."""
    spec = replace(minimal_spec, n_cores=1)
    directives = orca_builder.get_slurm_directives(spec)

    assert any("--ntasks=1" in d for d in directives)


@pytest.mark.unit
def test_qchem_get_slurm_directives_openmp(qchem_builder, minimal_spec):
    """Q-Chem with OpenMP should use cpus-per-task."""
    spec = replace(minimal_spec, n_cores=8)
    spec = replace(spec, program_options={"parallelism": "openmp"})
    directives = qchem_builder.get_slurm_directives(spec)

    assert any("--ntasks=1" in d for d in directives)
    assert any("--cpus-per-task=8" in d for d in directives)


@pytest.mark.unit
def test_qchem_get_slurm_directives_mpi(qchem_builder, minimal_spec):
    """Q-Chem with MPI should use ntasks."""
    spec = replace(minimal_spec, n_cores=16)
    spec = replace(spec, program_options={"parallelism": "mpi"})
    directives = qchem_builder.get_slurm_directives(spec)

    assert any("--ntasks=16" in d for d in directives)


@pytest.mark.unit
def test_qchem_default_parallelism_is_openmp(qchem_builder, minimal_spec):
    """Q-Chem without explicit parallelism should default to OpenMP."""
    spec = replace(minimal_spec, n_cores=8)
    directives = qchem_builder.get_slurm_directives(spec)

    # Default is OpenMP (cpus-per-task)
    assert any("--ntasks=1" in d for d in directives)
    assert any("--cpus-per-task=8" in d for d in directives)


# --- Unit Tests: Builder get_launch_command() ---


@pytest.mark.unit
def test_orca_get_launch_command(orca_builder, minimal_spec):
    """ORCA launch command should use $(which orca)."""
    cmd = orca_builder.get_launch_command(minimal_spec, "h2o.inp", "h2o.out")

    assert "$(which orca)" in cmd
    assert "h2o.inp" in cmd
    assert "h2o.out" in cmd
    assert ">" in cmd


@pytest.mark.unit
def test_orca_launch_command_preserves_filenames(orca_builder, minimal_spec):
    """ORCA launch command should preserve input and output filenames."""
    filenames = [
        ("water.inp", "water.out"),
        ("caffeine_opt.inp", "caffeine_opt.out"),
        ("tddft-sp.inp", "tddft-sp.out"),
    ]
    for inp, out in filenames:
        cmd = orca_builder.get_launch_command(minimal_spec, inp, out)
        assert inp in cmd
        assert out in cmd


@pytest.mark.unit
def test_qchem_launch_command_openmp(qchem_builder, minimal_spec):
    """Q-Chem OpenMP launch command should use -nt flag."""
    spec = replace(minimal_spec, n_cores=8)
    spec = replace(spec, program_options={"parallelism": "openmp"})
    cmd = qchem_builder.get_launch_command(spec, "h2o.in", "h2o.out")

    assert "qchem" in cmd
    assert "-nt 8" in cmd
    assert "h2o.in" in cmd
    assert "h2o.out" in cmd


@pytest.mark.unit
def test_qchem_launch_command_mpi(qchem_builder, minimal_spec):
    """Q-Chem MPI launch command should use -np flag."""
    spec = replace(minimal_spec, n_cores=16)
    spec = replace(spec, program_options={"parallelism": "mpi"})
    cmd = qchem_builder.get_launch_command(spec, "h2o.in", "h2o.out")

    assert "qchem" in cmd
    assert "-np 16" in cmd
    assert "h2o.in" in cmd
    assert "h2o.out" in cmd


@pytest.mark.unit
def test_qchem_launch_command_default_openmp(qchem_builder, minimal_spec):
    """Q-Chem without explicit parallelism should use -nt (OpenMP)."""
    spec = replace(minimal_spec, n_cores=4)
    cmd = qchem_builder.get_launch_command(spec, "mol.in", "mol.out")

    assert "qchem" in cmd
    assert "-nt 4" in cmd


# --- Contract Tests: SlurmJob.export() ---


@pytest.mark.contract
def test_export_qchem_basic_structure(minimal_spec, h2o_geometry):
    """Q-Chem export should produce valid slurm script with qchem launch."""
    # Note: spec's n_cores is used for builder directives, not SlurmJob's n_cores
    spec = replace(minimal_spec, n_cores=8)
    job = SlurmJob(
        job_name="h2o_sp",
        time="01:00:00",
        n_cores=8,
    )
    script = job.export(spec, program="qchem", input_filename="h2o.in", output_filename="h2o.out")

    # Check slurm structure
    assert script.startswith("#!/bin/sh\n")
    assert "#SBATCH -J h2o_sp" in script

    # Check Q-Chem OpenMP directives (default, from spec's n_cores)
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=8" in script

    # Check launch command
    assert "qchem" in script
    assert "-nt 8" in script
    assert "h2o.in" in script
    assert "h2o.out" in script


@pytest.mark.contract
def test_export_orca_basic_structure(minimal_spec, h2o_geometry):
    """ORCA export should produce valid slurm script with orca launch."""
    # Note: spec's n_cores is used for builder directives, not SlurmJob's n_cores
    spec = replace(minimal_spec, n_cores=4)
    job = SlurmJob(
        job_name="h2o_sp",
        time="01:00:00",
        n_cores=4,
    )
    script = job.export(spec, program="orca", input_filename="h2o.inp", output_filename="h2o.out")

    # Check slurm structure
    assert script.startswith("#!/bin/sh\n")
    assert "#SBATCH -J h2o_sp" in script
    assert "#SBATCH --time 01:00:00" in script

    # Check ORCA-specific directives (from spec's n_cores)
    assert "#SBATCH --ntasks=4" in script
    assert "#SBATCH --nodes=1" in script

    # Check launch command
    assert "$(which orca)" in script
    assert "h2o.inp" in script
    assert "h2o.out" in script


@pytest.mark.contract
def test_export_with_modules(minimal_spec, h2o_geometry):
    """Export with modules should include module load commands."""
    job = SlurmJob(
        job_name="h2o_tddft",
        time="02:00:00",
        n_cores=4,
    ).add_modules(["orca/5.0", "openmpi/4.1.1"])
    script = job.export(minimal_spec, program="orca", input_filename="h2o.inp", output_filename="h2o.out")

    assert "module load orca/5.0" in script
    assert "module load openmpi/4.1.1" in script


@pytest.mark.contract
def test_export_with_all_optional_fields(minimal_spec, h2o_geometry):
    """Export with all optional fields should include all directives."""
    job = SlurmJob(
        job_name="complex_job",
        time="04:00:00",
        n_cores=16,
        memory_mb=64000,
        partition="gpu",
        account="myproject",
        queue="high",
        modules=["cuda/11.2"],
    )
    script = job.export(minimal_spec, program="orca", input_filename="calc.inp", output_filename="calc.out")

    # All directives present
    assert "#SBATCH -J complex_job" in script
    assert "#SBATCH --time 04:00:00" in script
    assert "#SBATCH --mem=64000" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --account=myproject" in script
    assert "#SBATCH --qos=high" in script
    assert "module load cuda/11.2" in script


@pytest.mark.contract
def test_export_unsupported_program_raises(minimal_spec, h2o_geometry):
    """Export with unsupported program should raise NotImplementedError."""
    job = SlurmJob(
        job_name="test",
        time="01:00:00",
        n_cores=4,
    )
    with pytest.raises(NotImplementedError, match="program 'unsupported' not supported"):
        job.export(minimal_spec, program="unsupported", input_filename="test.inp", output_filename="test.out")


# --- Integration Tests ---


@pytest.mark.integration
def test_orca_tddft_workflow(h2o_geometry):
    """Complete ORCA TDDFT workflow with slurm."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="cam-b3lyp",
            basis_set="def2-tzvp",
            n_cores=8,
        )
        .set_tddft(nroots=10, singlets=True, triplets=False)
        .set_solvation("smd", "water")
    )

    job = (
        SlurmJob(
            job_name="h2o_tddft",
            time="02:00:00",
            n_cores=8,
            memory_mb=16000,
        )
        .set_partition("normal")
        .add_modules(["orca/5.0"])
    )

    script = job.export(calc, program="orca", input_filename="h2o.inp", output_filename="h2o.out")

    # Verify structure
    assert "#SBATCH -J h2o_tddft" in script
    assert "#SBATCH --partition=normal" in script
    assert "#SBATCH --mem=16000" in script
    assert "module load orca/5.0" in script
    assert "$(which orca)" in script
    assert "#SBATCH --ntasks=8" in script


@pytest.mark.integration
def test_qchem_mom_workflow(h2o_geometry):
    """Complete Q-Chem MOM workflow with slurm."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="energy",
            level_of_theory="b3lyp",
            basis_set="6-31g",
            n_cores=16,
        )
        .set_unrestricted(True)
        .set_mom("HOMO->LUMO")
        .set_solvation("smd", "acetonitrile")
    )

    job = SlurmJob(
        job_name="h2o_mom",
        time="04:00:00",
        n_cores=16,
        memory_mb=32000,
        partition="normal",
        account="chem_group",
    ).add_modules(["qchem/6.2", "intel/2021.3"])

    script = job.export(calc, program="qchem", input_filename="h2o.in", output_filename="h2o.out")

    # Verify structure
    assert "#SBATCH -J h2o_mom" in script
    assert "#SBATCH --partition=normal" in script
    assert "#SBATCH --account=chem_group" in script
    assert "module load qchem/6.2" in script
    assert "module load intel/2021.3" in script
    assert "qchem -nt 16 h2o.in h2o.out" in script


@pytest.mark.integration
def test_qchem_mpi_workflow(h2o_geometry):
    """Q-Chem MPI workflow with slurm."""
    calc = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="pbe0",
        basis_set="6-311++g(d,p)",
        n_cores=32,
    ).set_options(parallelism="mpi")

    job = SlurmJob(
        job_name="large_calc",
        time="08:00:00",
        n_cores=32,
        memory_mb=128000,
        partition="gpu",
    )

    script = job.export(calc, program="qchem", input_filename="mol.in", output_filename="mol.out")

    # Verify MPI directives
    assert "#SBATCH --ntasks=32" in script
    assert "--cpus-per-task" not in script
    # Verify MPI launch command
    assert "qchem -np 32" in script


@pytest.mark.integration
def test_orca_optimization_workflow(h2o_geometry):
    """Complete ORCA geometry optimization workflow with slurm."""
    calc = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="b3lyp",
            basis_set="def2-svp",
            n_cores=4,
        )
        .set_optimization(calc_hess_initial=True)
        .run_frequency_after_opt()
    )

    job = SlurmJob(
        job_name="h2o_opt",
        time="12:00:00",
        n_cores=4,
        memory_mb=8000,
    )

    script = job.export(calc, program="orca", input_filename="h2o.inp", output_filename="h2o.out")

    # Verify job configuration
    assert "#SBATCH -J h2o_opt" in script
    assert "#SBATCH --time 12:00:00" in script
    assert "#SBATCH --ntasks=4" in script


# --- Regression Tests ---


def _extract_sbatch_directives(script: str) -> dict[str, str]:
    """Parse sbatch directives from script into dict."""
    directives = {}
    for line in script.split("\n"):
        if line.startswith("#SBATCH"):
            # Handle various formats: #SBATCH -J name, #SBATCH --key=value, #SBATCH --key value
            if " -J " in line:
                parts = line.split(" -J ", 1)
                directives["job_name"] = parts[1].strip()
            elif "=" in line:
                parts = line.split("=", 1)
                key = parts[0].replace("#SBATCH --", "").strip()
                directives[key] = parts[1].strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[1].lstrip("-").strip("=")
                    value = parts[2] if len(parts) > 2 else ""
                    if value:
                        directives[key] = value
    return directives


def _extract_modules(script: str) -> list[str]:
    """Extract module load commands from script."""
    modules = []
    for line in script.split("\n"):
        if line.startswith("module load "):
            modules.append(line.replace("module load ", "").strip())
    return modules


def _extract_launch_command(script: str) -> str:
    """Extract the launch command (last non-empty, non-sbatch line)."""
    lines = script.split("\n")
    non_directive_lines = [line for line in lines if line.strip() and not line.startswith("#")]
    return non_directive_lines[-1] if non_directive_lines else ""


@pytest.mark.regression
def test_orca_sp_semantic_regression(h2o_geometry):
    """ORCA single-point script should have semantic correctness."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="b3lyp",
        basis_set="def2-svp",
        n_cores=4,
    )

    job = SlurmJob(
        job_name="h2o_sp",
        time="01:00:00",
        n_cores=4,
        memory_mb=8000,
    )

    script = job.export(spec, program="orca", input_filename="h2o.inp", output_filename="h2o.out")

    # Parse and check semantic content
    directives = _extract_sbatch_directives(script)

    assert directives.get("job_name") == "h2o_sp"
    assert directives.get("time") == "01:00:00"
    assert directives.get("ntasks") == "4"
    assert directives.get("mem") == "8000"

    # Check launch command
    launch_cmd = _extract_launch_command(script)
    assert "$(which orca)" in launch_cmd
    assert "h2o.inp" in launch_cmd
    assert "h2o.out" in launch_cmd


@pytest.mark.regression
def test_qchem_tddft_semantic_regression(h2o_geometry):
    """Q-Chem TDDFT script should have semantic correctness."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="cam-b3lyp",
        basis_set="6-31g*",
        n_cores=8,
    ).set_tddft(nroots=10, singlets=True, triplets=False)

    job = SlurmJob(
        job_name="h2o_tddft",
        time="02:00:00",
        n_cores=8,
        memory_mb=16000,
        partition="gpu",
    )

    script = job.export(spec, program="qchem", input_filename="h2o.in", output_filename="h2o.out")

    # Parse and check semantic content
    directives = _extract_sbatch_directives(script)

    assert directives.get("job_name") == "h2o_tddft"
    assert directives.get("partition") == "gpu"
    assert directives.get("mem") == "16000"

    # Q-Chem OpenMP by default
    assert directives.get("ntasks") == "1"
    assert directives.get("cpus-per-task") == "8"

    # Check launch command
    launch_cmd = _extract_launch_command(script)
    assert "qchem" in launch_cmd
    assert "-nt 8" in launch_cmd
    assert "h2o.in" in launch_cmd
    assert "h2o.out" in launch_cmd


@pytest.mark.regression
def test_slurm_script_with_modules_semantic_regression(h2o_geometry):
    """Script with modules should have correct semantic structure."""
    spec = CalculationInput(
        charge=0,
        spin_multiplicity=1,
        task="energy",
        level_of_theory="hf",
        basis_set="sto-3g",
        n_cores=4,
    )

    job = SlurmJob(
        job_name="test_job",
        time="01:00:00",
        n_cores=4,
    ).add_modules(["gcc/11.2.0", "openmpi/4.1.1"])

    script = job.export(spec, program="orca", input_filename="test.inp", output_filename="test.out")

    # Parse and check modules
    modules = _extract_modules(script)
    assert "gcc/11.2.0" in modules
    assert "openmpi/4.1.1" in modules
    assert len(modules) == 2

    # Check that modules appear before launch command
    module_lines = [i for i, line in enumerate(script.split("\n")) if "module load" in line]
    launch_line = script.find("$(which orca)")
    assert module_lines and launch_line > 0


@pytest.mark.regression
def test_slurm_script_multifeature_semantic_regression(h2o_geometry):
    """Complex script with multiple features should maintain semantic correctness."""
    spec = (
        CalculationInput(
            charge=0,
            spin_multiplicity=1,
            task="geometry",
            level_of_theory="wb97x-d3",
            basis_set="6-311++g(d,p)",
            n_cores=16,
        )
        .set_unrestricted(True)
        .set_tddft(nroots=5)
        .set_solvation("smd", "water")
    )

    job = (
        SlurmJob(
            job_name="complex_calc",
            time="06:00:00",
            n_cores=16,
            memory_mb=32000,
            partition="gpu",
            account="research",
            queue="high",
            modules=["cuda/11.2", "gcc/11.2.0"],
        ).set_partition("gpu")  # overwrite
    )

    script = job.export(spec, program="qchem", input_filename="mol.in", output_filename="mol.out")

    # Comprehensive semantic checks
    directives = _extract_sbatch_directives(script)
    assert directives.get("job_name") == "complex_calc"
    assert directives.get("time") == "06:00:00"
    assert directives.get("partition") == "gpu"
    assert directives.get("account") == "research"
    assert directives.get("qos") == "high"
    assert directives.get("mem") == "32000"

    # Check modules
    modules = _extract_modules(script)
    assert "cuda/11.2" in modules
    assert "gcc/11.2.0" in modules

    # Check launch command
    launch_cmd = _extract_launch_command(script)
    assert "qchem" in launch_cmd
