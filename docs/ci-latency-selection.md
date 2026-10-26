# Check job selection and explicit acceptance

`check.yml` keeps the required `web`, `parity`, and `candle` job names and an
unfiltered pull-request trigger. Branch pushes under `codex/**` no longer create
a second Check run; opening or updating the PR validates the branch. Main pushes,
merge groups, and `workflow_dispatch` run every Check job. To request full Check
acceptance for a story or epic branch:

```sh
gh workflow run check.yml --ref <branch>
```

For pull requests, `scripts/check-job-relevance.mjs` reads an exhaustive local
three-dot diff with NUL-delimited names and rename detection disabled, so both
sides of a rename count. Failed/missing/empty diffs, unclassified inputs, and
shared build/controller changes select every job. Native/data/tooling roots
remain conservative because Node, Python, Rust and web contracts read across
those boundaries. Rust's literal `include_str!`/`include_bytes!` inputs are also selected for
native compilation, including embedded web and documentation files. Runtime doc
readers include the workflow-share contract and MiniMax hosting decision. Web tests' external Rust,
manifest, fixture, schema, license and bundled-document inputs remain selected.
Ordinary contributor prose skips the expensive jobs; actual consumed docs and
calibration evidence still select their readers.

`parity` always runs and grades the exact expected dependency set, the selector's
successful result and boolean outputs, the unconditional advisory gate, and each
job result. A skipped job is accepted only with an explicit `false` selection;
failed, cancelled, missing or unexpectedly skipped jobs fail. Jobs also run
conservatively if selection itself fails. Docker's step gate consumes that same
selection; it includes native/web image source, toolchain, vendored dependencies,
configuration and embedded calibration inputs.

`npm run check` remains the complete local test inventory. The portable and
historical CI invocations partition its existing complete test files and preserve
its original serial command barriers. Historical integration includes the anchor
closure, memory-catalog, memory-declaration and StarVector producer suites. They
run for native/loader, pin, anchor/evidence and tooling changes, or full acceptance;
none of their source-dependent assertions becomes optional on CI.

The historical cache contains a bare Git object database with a named retained
ref for the current consumer pin, every measurement/attestation revision emitted
by `--anchor-revisions`, and the sealed StarVector contract and validator base.
The key binds that exact revision set and preparation code. Only main saves
shared caches. Restores verify every retained ref and tree plus Git connectivity;
missing refs are fetched by exact SHA. Fresh per-run checkouts share only the
verified object database, so temporary fixture worktrees never enter the cache.
The deliberately disabled pin-keyed measurement gates remain disabled.

The imported NVFP4 worker smoke is explicit acceptance through
`windows-candle.yml`'s `run_imported_nvfp4_worker_smoke=true` dispatch input. The
complete ignored release-mode test, checkpoint identity, real-weight labels,
fixed real-weight concurrency group and physical GPU1 mapping remain in place.
Its preceding candle-worker dispatch also selects physical GPU1 and the shared
GPU lock. Capability-facts-only dispatches reject combining this smoke. It does
not run automatically on a PR or push.

```sh
gh workflow run windows-candle.yml --ref <branch> \
  -f run_imported_nvfp4_worker_smoke=true
```

The hosted `macos-checks` job sets `line-tables-only` for dev/test and both build
override profiles in its workflow environment. Root Cargo profiles and release
artifacts are unchanged. Full epic acceptance still requires all applicable
platform lanes and explicit real-weight evidence; a selected PR run alone does
not establish that evidence. The existing platform workflows' terminal
feature-to-main full-validation behavior remains intact.
