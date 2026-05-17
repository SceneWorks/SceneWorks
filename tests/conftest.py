from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package_path in (ROOT / "apps" / "api", ROOT / "apps" / "worker"):
    sys.path.insert(0, str(package_path))
