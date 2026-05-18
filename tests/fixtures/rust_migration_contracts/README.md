# Rust Migration Contract Fixtures

These fixtures preserve the migration-era backend contract for HTTP route
coverage, worker queue protocol strings, SSE event names, and persisted project
sidecar shapes.

The live API contract is now enforced by
`tests/test_rust_api_contract_snapshots.py`, which exercises the Rust API and
compares normalized responses to committed snapshots. When the public contract
changes intentionally, regenerate the snapshots with `UPDATE_SNAPSHOTS=1` and
review the diff.

