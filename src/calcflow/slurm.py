from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from calcflow.common.input import CalculationInput

T_SlurmJob = TypeVar("T_SlurmJob", bound="SlurmJob")


# --- User-Facing API ---
@dataclass(frozen=True)
class SlurmJob:
    """the fluent, user-facing api for building a slurm job."""

    job_name: str
    time: str
    n_cores: int
    memory_mb: int | None = None
    partition: str | None = None
    constraint: str | None = None
    account: str | None = None
    queue: str | None = None  # often called QOS
    modules: list[str] | None = None

    def set_partition(self: T_SlurmJob, partition: str) -> T_SlurmJob:
        return replace(self, partition=partition)

    def add_modules(self: T_SlurmJob, modules: list[str]) -> T_SlurmJob:
        return replace(self, modules=modules)

    def export(
        self,
        calculation: CalculationInput,
        *,
        program: str,
        input_filename: str,
        output_filename: str,
    ) -> str:
        """
        exports the slurm script by orchestrating the builders.

        args:
            calculation: the `CalculationInput` object defining the qc task.
            program: the name of the program ("orca", "qchem").
            input_filename: the name of the input file to be generated.
            output_filename: the name of the output file to be generated.
        """
        from calcflow.common.input import BUILDERS

        program_lower = program.lower()
        if program_lower not in BUILDERS:
            raise NotImplementedError(f"program '{program}' not supported.")

        program_builder = BUILDERS[program_lower]

        # query the program builder for the expert information
        directives = program_builder.get_slurm_directives(calculation)
        launch_cmd = program_builder.get_launch_command(calculation, input_filename, output_filename)

        return self.build(directives, launch_cmd)

    def build(self, program_directives: list[str], launch_command: str) -> str:
        """
        generates the slurm script content.

        args:
            spec: the slurm job specification.
            program_directives: program-specific #sbatch lines (e.g., for mpi/openmp).
            launch_command: the shell command to execute the calculation.
        """
        lines = [
            "#!/bin/sh",
            f"#SBATCH -J {self.job_name}",
            "#SBATCH --output sbatch.out",
            "#SBATCH --error sbatch.err",
            f"#SBATCH --time {self.time}",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.queue:
            lines.append(f"#SBATCH --qos={self.queue}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.memory_mb:
            lines.append(f"#SBATCH --mem={self.memory_mb}")

        # add program-specific directives for parallelism
        lines.extend(program_directives)
        lines.append("")

        if self.modules:
            for module in self.modules:
                lines.append(f"module load {module}")
            lines.append("")

        lines.append(launch_command)
        return "\n".join(lines) + "\n"
