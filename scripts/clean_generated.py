from __future__ import annotations

import shutil
from pathlib import Path

for path in (Path("data/demo"), Path("artifacts")):
    if path.exists():
        shutil.rmtree(path)
        print(f"Removed {path}")
