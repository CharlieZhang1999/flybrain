"""Sanity check for the day-1 setup. Run: .venv/bin/python scripts/check_setup.py"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = True

print("python", sys.version.split()[0], sys.executable)
for mod in ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "matplotlib",
            "brian2", "jupyter", "navis", "neuprint", "openai"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  FAIL {mod}: {e}")

import pandas as pd  # noqa: E402

con_path = ROOT / "data/flywire_v783/Connectivity_783.parquet"
comp_path = ROOT / "data/flywire_v783/Completeness_783.csv"
if con_path.exists() and comp_path.exists():
    con = pd.read_parquet(con_path)
    comp = pd.read_csv(comp_path, index_col=0)
    print(f"FlyWire v783: {len(comp):,} neurons, {len(con):,} edges")
else:
    ok = False
    print("FAIL FlyWire v783 files missing; run scripts/setup.sh")

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if v.strip():
                os.environ.setdefault(k.strip(), v.strip())

token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
if not token:
    print("neuPrint token: NOT SET (needed for MaleCNS, week 2)")
else:
    try:
        from neuprint import Client
        c = Client(os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org"),
                   dataset=os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0"), token=token)
        n = c.fetch_custom("MATCH (n:Neuron) RETURN count(n) AS n").iloc[0, 0]
        print(f"neuPrint {c.dataset}: OK, {n:,} Neuron nodes")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"FAIL neuPrint: {e}")

sys.exit(0 if ok else 1)
