import sys
from pathlib import Path

# make `import sim` and `import synth` work without installing the repo
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'tests'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
