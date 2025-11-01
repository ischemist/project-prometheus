from pathlib import Path

from calcflow.io.orca import parse_orca_output
from calcflow.io.qchem import parse_qchem_output

base_dir = Path(__file__).resolve().parents[1]
test_dir = base_dir / "tests" / "testing_data"
orca_sp_path = test_dir / "orca" / "h2o" / "sp.out"
qchem_sp_path = test_dir / "qchem" / "h2o" / "6.2-sp-smd.out"

parse_orca_output(orca_sp_path.read_text())
o = parse_qchem_output(qchem_sp_path.read_text())
print(o)
# print(parse_qchem_multi_job_output(( test_dir / "qchem" / "h2o" / "6.2-mom-sp-smd.out").read_text()))
