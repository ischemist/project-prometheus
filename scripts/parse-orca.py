from pathlib import Path

from calcflow import parse_orca_output, parse_qchem_output

base_dir = Path(__file__).resolve().parents[1]
test_dir = base_dir / "tests" / "testing_data"

# 1. parse output file
orca_output_path = test_dir / "orca" / "h2o" / "sp.out"
result = parse_orca_output(orca_output_path.read_text())

# 2. save result to json (excludes raw_output automatically)
# this is useful for caching - next time you can load from json instead of re-parsing
with open("orca_result.json", "w") as f:
    f.write(result.to_json())

result = parse_qchem_output((test_dir / "qchem" / "h2o" / "6.2-mom-xas-smd.out").read_text())
with open("qchem_result.json", "w") as f:
    f.write(result.to_json())

# 3. later, load result from json (much faster than re-parsing)
# loaded_result = CalculationResult.from_json(open("orca_result.json").read())
# assert loaded_result.final_energy == result.final_energy  # all data preserved except raw_output
