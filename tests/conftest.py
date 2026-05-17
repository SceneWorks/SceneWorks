from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for package_path in (ROOT / "apps" / "api", ROOT / "apps" / "worker", ROOT / "packages" / "shared"):
    sys.path.insert(0, str(package_path))
