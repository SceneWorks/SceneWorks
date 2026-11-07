# Windows runner storage cleanup

Register this local JavaScript action immediately after checkout and **before**
`Swatinem/rust-cache`. Its `always()` post action runs after later cache post
actions and normal artifact uploads. Failed/cancelled jobs also attempt cleanup;
a hard-killed runner leaves storage for a later job. It never stops a process.

The action is vendored identically in SceneWorks and inference so neither
repository's CI depends on an unmerged or floating cross-repository action.
Keep the action files and `scripts/runner-storage-cleanup.test.mjs` synchronized.
The MOSS workflow checks out this action sparsely from its workflow SHA, separately
from the requested product SHA, so older product revisions remain buildable.

## Cargo targets

Remove only `target` directly under the current listener's checkout, `source`
checkout, or `inference` checkout, plus the current MOSS run/attempt's `target`.
These directories are disposable once cache uploads and job steps have finished.
Rust-cache still saves according to the workflow's existing policy. Cleanup can
make a subsequent cache-miss build cold; it bounds retained local build output.

Deletion requires a Cargo marker, no tracked files in ordinary checkout targets,
ordinary resolved ancestors/descendants, no process referencing the directory,
and exclusive access to every existing `.cargo-lock`. The lock handles prevent
new Cargo read/write opens during deletion. A held lock, link/reparse point,
unknown checkout, or unreadable ownership check retains the target.

`CARGO_HOME`, `sccache`, external/shared `CARGO_TARGET_DIR` values, primary
developer checkouts and worktrees are outside the deletion scope.

## Retired private MOSS copies (`moss: 'true'`)

The current MOSS smoke reads its complete input cache directly; it creates no
new private model projection. Historical campaigns left private `hf` copies and
build targets under this listener's `moss-sc-26752` directory.

Sweep only exact `moss-build-RUN-ATTEMPT-cuda` and
`moss-execute-RUN-ATTEMPT-cuda` owners verified through the read-only Actions API.
Private `hf` ownership must also match the original runtime's root, HF path,
backend, and source pair. Check both the original attempt and the latest run;
queued/in-progress retries preserve the original cache. Keep the two newest
completed owners and at least 24 hours after the run's latest update. Recheck
completion and the recovery window before removal.

Preserve explicit input caches and every HF source referenced by a retained
MOSS bundle, including bundles on sibling/offline listener installations.
Discover installations from `.runner` configuration and running listeners rather
than a remembered runner count. Unreadable input manifests, API failures and
unknown owners skip the historical sweep. The sweep deletes only `hf` and
`target`; runtime records, images and diagnostics remain.

Before removing model copies, also require no other Actions worker anywhere on
the host and no build/model process. Recheck local activity under a host mutex
for each leaf. Starting jobs pass the same short mutex barrier before using
models. No GPU scheduling, device restrictions, workflow concurrency, runner
services, shared model inputs, or historical evidence are changed.

The action emits JSON results (`removed`, `skipped`, and reasons) in its post-job
log. The broad activity checks deliberately prefer retaining uncertain data.
No production deletion runs from a developer shell. The PowerShell library mode
exists for fixture tests and read-only inspection; tests use temporary folders.

Run `node --test scripts/runner-storage-cleanup.test.mjs` for retention/API tests
and, on Windows, actual lock, junction and filesystem deletion tests.
