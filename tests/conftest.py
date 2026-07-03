from __future__ import annotations

import sys
from pathlib import Path


# The retired Python worker's test suite (16 `tests/test_worker_*.py` /
# `test_*_adapters.py` files + `worker_runtime_shared.py`) was deleted in sc-8863
# once its live gates had been re-expressed against the Rust worker (sc-8861 e2e
# pure-HTTP client + sc-9513 engine-wiring guards), so `apps/worker` is no longer
# imported by any test. The surviving tests (Rust-API contract/smoke, manifest
# audits, the convert-script unit test) only need `packages/shared` on the path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "shared"))
