from calcflow.common.input import CalculationInput
from calcflow.geometry.static import Geometry
from calcflow.slurm import SlurmJob

# from calcflow.jobs.slurm import SlurmJob

# 1. define the scientific calculation
calc = (
    CalculationInput(
        charge=0, spin_multiplicity=1, task="geometry", level_of_theory="wb97x-d3", basis_set="def2-svp", n_cores=16
    )
    .enable_ri_for_orca("RIJCOSX", "def2/j")
    .run_frequency_after_opt()
)

# 2. define the compute job
slurm_job = SlurmJob(job_name="h2o_sp", time="01:00:00", n_cores=16, account="m410", queue="premium").add_modules(
    ["orca"]
)

# 3. get geometry
water = Geometry.from_xyz_file("tests/testing_data/geometries/1h2o.xyz")

# 4. export files
input_file_content = calc.export("orca", water)
slurm_script_content = slurm_job.export(calc, program="orca", input_filename="h2o.inp", output_filename="h2o.out")

# 5. write to disk
with open("h2o.inp", "w") as f:
    f.write(input_file_content)

with open("submit.sh", "w") as f:
    f.write(slurm_script_content)

# 6. save calculation spec as json for reproducibility
# this allows you to see exactly what parameters were used without parsing the input file
with open("h2o_calc_spec.json", "w") as f:
    f.write(calc.to_json())

# 7. later, you can load the spec back from json
# loaded_calc = CalculationInput.from_json(open("h2o_calc_spec.json").read())
# assert loaded_calc == calc  # perfect roundtrip!
