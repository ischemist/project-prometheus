from pathlib import Path

from calcflow.io.orca import parse_orca_sp

base_dir = Path(__file__).resolve().parents[1]

out_path = base_dir / "tests" / "testing_data" / "orca" / "h2o" / "sp.out"


o = parse_orca_sp(out_path.read_text())
print(o)
