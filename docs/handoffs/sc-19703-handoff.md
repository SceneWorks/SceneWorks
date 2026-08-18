# Epic 19703 — Resolved-model hot cache: session handoff

**Written 2026-08-17.** Session ended on a usage limit, not on a problem. Every branch below is
**pushed**; nothing is stranded in a worktree. Written for a session with **zero context**.

Epic: https://app.shortcut.com/trefry/epic/19703 ·
Integration branch: `feature/sc-19703-resolved-model-hot-cache` @ **`833a1df2c`** ·
**SceneWorks repo only — zero inference scope, no pin bump, no measurement campaign.**

---

## 1. The architecture bar — read this before touching any code

Codex's first attempt at sc-19708 was ~2,600 lines of **route-by-route, model-by-model**
availability wiring. Michael rejected it outright. The binding design:

1. Jobs carry a **generic list of selected artifact requirements**.
2. Manifests **declaratively** describe each artifact closure.
3. **ONE pre-loader resolver** handles every requirement uniformly.
4. **ONE worker guard** chooses local_ready / external_ready / installed_external_unavailable /
   incomplete / missing.
5. Existing loaders receive **resolved paths**; they contain no cache policy.
6. Conditional adapters/components are **data in the request**, never Rust model-name branches.

Model count must affect **manifest data and fixtures only** — never the volume of cache-control
code. A `model.id == "..."` branch in availability/preference logic is a defect, not a shortcut.
The rejected work is preserved verbatim as audit commit `babcb8a10` on
`codex/sc-19708-external-library-state` — **do not build on it.**

---

## 2. Story state

| Story | State | PR / branch | Notes |
|---|---|---|---|
| sc-19704 typed two-tier contract | **Done** | #2366 | pre-rescue |
| sc-19705 policy/journal/leases | **Done** | #2387 | pre-rescue |
| sc-20305 Windows CI safety | **Done** | #2389 | pre-rescue |
| sc-19706 materialization | **REOPENED** (In Progress) | #2396 merged; `claude/sc-19706-promotion-activation` | mechanism merged but **never called** — §4 |
| sc-19708 generic availability seam | **Done** | #2403 merged | the rescue's centrepiece |
| sc-19710 retention/eviction/pinning | **Done** | #2405 merged | |
| sc-19707 local-tier preference | In Progress | **#2407** `claude/sc-19707-local-artifact-preference` | **2 blockers, HEAD DOES NOT COMPILE** — §3 |
| sc-19709 reconnect/relocate UX | In Progress | **#2406** `claude/sc-19709-reconnect-ux` | batch pushed, needs re-review — §5 |
| sc-19711 Settings + Model Manager | In Progress | `claude/sc-19711-cache-controls-ui` (no PR) | no web tests yet — §6 |
| sc-19712 end-to-end validation | Backlog | — | terminal; run against the **real** drive — §8 |

**Accounting this session: stories closed 3 (19706 reopened) │ stories filed 0.**

---

## 3. 🔴 sc-19707 / PR #2407 — TWO BLOCKERS, HEAD IS BROKEN

**`origin/claude/sc-19707-local-artifact-preference` @ `0cc28bc4b` DOES NOT COMPILE.**
Last fully green commit: **`3de7d30cf`** (all CI lanes passed there). PR is **DIRTY/CONFLICTING**
with the feature branch.

Compile errors — all one mechanical cause: the registry now stores `HeldEntry { entry, holders }`
but `local_snapshot`, `unique_local_snapshot` and `redirect_source_library_path`
(`crates/sceneworks-core/src/model_artifacts/local_preference.rs`, ~lines 285-350) still read
`.repository` / `.revision` / `.snapshot_root` off the iterated item — they need `held.entry.*`.
Also `prefer_local_artifacts` was renamed/retyped to
`prefer_local_snapshots(Vec<LocalArtifactOverlayEntry>)`, so its callers are stale:
`crates/sceneworks-worker/src/external_library_runtime.rs::admit_local_artifact`, several call
sites in `local_preference_tests.rs`, and `a_narrower_serving_bundle_is_never_adopted` still
references the now-deleted `ensure_covers`.

### Blocker 1 — overlay keyed on `(repository, revision)` with no file-coverage check
`local_snapshot` matches on `(repository, revision)` + `snapshot_root.is_dir()` and nothing else,
and hands that bundle path to **every** caller asking for the pair. The *decision* layer
(`resolve_model_availability`) is keyed on the full selected closure **including variant/tier**, so
the two layers disagree: a job carrying `owner/matrix` q4 (bundled) and q8 (not bundled) admits q4
as LocalReady, installs the overlay for `owner/matrix@REV_A`, and then the **q8 load resolves into
the q4 bundle root**, which has no `q8/` subtree. Tell that proves the omission:
`redirect_source_library_path` *does* verify `redirected.exists()`; the snapshot resolver verifies
only `is_dir()`.

### Blocker 2 — the fail-closed fallback does not close
When `admit_local_artifact` returns `None`, the model is re-resolved without local candidates —
but **the narrower serving bundle is still installed in the process-wide overlay**, so the
re-resolved source-tier path still resolves into it. The test that claims to cover this,
`a_narrower_serving_bundle_is_never_adopted`, calls `drop(scope)` **before** asserting — so it
never asks its own question. Same defect in `a_wrong_revision_or_wrong_tier_bundle_is_not_used`
(`drop(guard)` before evaluating the q8 selection), which is why wrong-tier was verified only at
the availability layer and never at the path layer — exactly where the bug lives.

### The implementer's intended fix (adopt it; it is sound)
Make the decision **at the guard**, not at the lookup. A directory-level seam cannot re-ask "does
this bundle hold what this caller needs" after handing out a path. Rule:

> A `(repository, revision)` pair is served locally **iff** one leased bundle holds a **superset of
> the union of file requirements every model entry in the job needs from that pair**; otherwise the
> pair is served to nobody and every model needing it re-resolves against the source tier.

Three-pass shape in `RuntimeSourceGuard::begin`: (1) resolve every entry with local candidates,
keeping `selected.requirements` + candidate artifact; (2) build `(repo, revision) -> union of
required files` across **all** resolutions, local *and* non-local (the non-local q8 sibling is what
poisons the pair), then pick a serving entry only if `entry_covers(entry, union)` — treat a
requirement with `revision: None` as making the whole repository unserveable; (3) a model is
LocalReady only if every pair in its closure was selected, all others re-resolve with `&[]`.
Install ONE scope with exactly the selected entries. **Deliberate consequence:** in a q4+q8 job the
q4 model *also* falls back to source. That is the correct conservative answer — state it in the
test name, do not engineer around it.

`pub fn entry_covers(entry, files)` (the subset primitive) is already written and pushed.
Item 3 (adopt/drop refcount asymmetry) is essentially done once it compiles.

### Remaining review items on #2407 (none started)
- **major** `valid_local_artifacts` drops layout-unsupported entries with a bare `continue` before
  the guard sees them, making `resolved_cache_local_tier_unsupported` unreachable for the case it
  names. Fix: return `LocalArtifactScan { artifacts, rejections }` so rejections reach the guard;
  the coverage/selection failure needs a *different* event, not the unsupported label.
- **major** no test asserts emission of `resolved_cache_local_tier_unsupported` or
  `model_source_tier_selected`, though both are presented as satisfying ACs.
- **minor** `FileExt::lock_shared` is blocking and `RuntimeSourceGuard::begin` runs directly in the
  job's async block (`worker/src/lib.rs:1642`) — wrap in `spawn_blocking` rather than changing
  `acquire_complete`'s lock semantics (shared with sc-19710).
- **minor** `valid_local_artifacts` swallows every failure with no diagnostic.
- **minor** worker `with_local_cache` tests take no overlay serialization mutex — `.cargo/config.toml`
  forces `RUST_TEST_THREADS=1`, so a green run proves nothing here.

---

## 4. 🔴 The gap that reopened sc-19706 — production never populates the cache

**Nothing in production calls `schedule` / `run_next_if_idle`, and nothing builds a
`ResolvedBundleClosure` for a real manifest entry.** `<data_dir>/models/resolved/` is filled only
by tests. The epic's entire user-visible value does not exist in a real install.

It fell between two stories, and **both passed adversarial review inside their own scope**:
sc-19706 said "generic seam; production activation is 19707's scope", and sc-19707's ACs cover
*preferring* artifacts that exist, not *producing* them. That is a scoping defect, not an
implementation one, and it is the same horizontal-slicing pathology that made this epic need a
rescue.

### Branch `claude/sc-19706-promotion-activation` @ `e02492801` (no PR yet)
Compiles; `promotion` 7/7 and `resolved_cache` 59/59 pass. **Done:**
`crates/sceneworks-core/src/model_artifacts/promotion.rs` with
`promotion_candidate_for_requirements(...)` (consumes sc-19708's already-computed closure — no
second closure computation, no per-model logic — pins immutable snapshots, mints the candidate only
through sc-19704's `mark_success` boundary), `hub_cache_member_destination(...)`,
`resolve_requirement_revision`, plus policy admission in `resolved_cache/materialization.rs`
(`PromotionDeclineReason::{ExceedsSizeLimit, SourceUnavailable}`,
`IdlePromotionOutcome::{AlreadyComplete, Declined}`, `scheduler.admit()` called from
`run_next_if_idle`), and a core round-trip test that materializes a real manifest entry and
re-opens the bundle as a valid `ArtifactSourceLibrary`.

**NOT done — this is most of the user-visible work:**
- **Trigger**: nothing calls the producer. Intended seam is
  `worker/src/external_library_runtime.rs::RuntimeSourceGuard::finish_success` (~:183), UNTOUCHED.
  (Note: no production code calls `acquire_runtime_lease` at all on the feature branch;
  `finish_success` is where a job that actually loaded and ran lands.) Only `ExternalReady`
  resolutions should produce candidates.
- **Idle drain**: worker poll loop UNTOUCHED; `crates/sceneworks-worker/src/resolved_cache_promotion.rs`
  planned, not written.
- **Zero tests** for the admission code (`ExceedsSizeLimit`, `SourceUnavailable`, `AlreadyComplete`)
  and no mutation evidence — the author flags this as the biggest gap in what exists.

**Design decisions to preserve:** two-stage queue is deliberate — `finish_success` pushes an
**I/O-free** `PendingPromotion` into a bounded intake; the idle drain (on `spawn_blocking`) builds
the closure (dozens of stats/canonicalizes) and calls `schedule()` then `run_next_if_idle(true)`.
Building inline would put source-library I/O on the job path, which is forbidden. Size rule is
`protected_bytes + candidate_bytes > max_bytes` declines, where protected = Complete AND
`effective_pin` — sizing against *pinned* bytes is the precise anti-thrash rule; total bytes would
refuse promotions the cache can hold. `IdlePromotionOutcome::AlreadyComplete` is distinct from
`Materialized(AlreadyComplete)` on purpose: it skips **without** taking the exclusive artifact lock
that a live local-tier load holds shared.

**Merge hazard:** `hub_cache_member_destination` is duplicated byte-identically with #2407's copy
in `local_preference.rs`. At merge, delete one and re-export — call it out in the PR body.
**Also:** `worker/src/tests.rs:186` asserts on poll-loop source text via
`split_once("let mut retention_task =")`; the single-maintenance-slot design renames that local, so
update that test rather than working around it.

**Ordering that matters:** #2407's blockers must land **before** promotion activation merges —
promotion is what turns them from latent into live.

---

## 5. sc-19709 / PR #2406 — COMPLETE, one lane from mergeable. **Start here.**

Head **`45d7fe30f`**. **Every review item and the restart work landed**; verified green locally
*after* merging the moved base (sc-19710 landed mid-batch): `npm run rust:check` exit 0,
`npm run check` 412 pass, web suite 3694 pass / 250 files, eslint clean,
`cargo check -p sceneworks-desktop` clean, contract snapshots 14/14 (regenerated for the new
`modelResolution.libraryPresent` field). CI on the current head: build-windows, candle,
check-linux, check-macos, web, parity, parity-rust, parity-scaffold, parity-digests, parity-docker
all **SUCCESS**; **only "macOS build, lint and workspace tests (hosted)" is unconfirmed** — still
running when the session ended. The previous push (`d499b6d6b`) was fully green on every lane.

Landed: the **restart dialog** ("The generation you started was not queued. Submit it again after
SceneWorks restarts.", Restart now / Later) via a `restart_app` command → `setup::begin_restart`
(graceful SIGTERM + grace teardown, **not** the auto-update force-kill), fired once through a ref
guard, **withheld while a job is running**, degrading to guidance off-desktop; both overlays hoisted
into `modelLibraryOverlays` and rendered in the **Simple** branch (verified: the new Simple-shell
test fails without it); `relocate_binding` judging the **same evidence union availability uses**
(validated-closure ledger **and** download receipts from
`<data_dir>/models/*/.sceneworks-download-complete.json`), refusing with typed `no_installed_models`
when there is none; validate(`dryRun`) → persist → adopt with the persist **undone** if adopt fails
(distinct messages for undo-succeeded vs undo-failed); 11 new core cases in
`external_library_tests.rs`; the four minors; **loopback-only** relocate endpoint, fail-closed on
unknown peer; and the path-aliasing fix (§7).

**🔴 Design note that is easy to get wrong:** relaxing `probe_binding` alone is **not sufficient
and looks sufficient**. `ModelResolution::validate` requires
`binding.configured_path == resolution.configured_library_path`, and the worker's pre-loader guard
calls it — so an aliased resolution would pass the probe and then be **rejected at job start**.
`bind_or_probe_validated` therefore **re-aliases the ledger** (`realias_unlocked`) once the probe
has proven both names are the same library; canonical path, identity and `bound_at` are untouched.
The gate also deliberately **drops** the pending action on relocation (like cancel) rather than
resuming — resuming would only be refused again until relaunch, which is why the dropped-action
wording exists in both dialog and notice. `library_present` is `#[serde(default)]` on purpose so
resolutions stamped into jobs before this field still deserialize under `deny_unknown_fields`.

**Next:** confirm that one macOS hosted lane reached SUCCESS (`gh pr checks 2406`); if it failed,
read the job log before assuming anything — everything it runs passed locally on the same commit.
Then re-review the delta and merge.

### Environment gotchas this PR paid for (they will bite again)
- **`apps/desktop/capabilities/default.json` is a THIRD registration site** beyond `main.rs` and
  `build.rs`. A command missing there is **ACL-rejected at runtime** at the shell's remote
  `127.0.0.1:<port>` origin, and web tests mock `tauriInvoke` so **nothing catches it**.
- `cargo check -p sceneworks-desktop` fails in a fresh worktree until you run
  `node apps/desktop/scripts/stage-test-sidecars.mjs`.
- A worktree needs `node_modules` symlinked from the main checkout for vitest.
- Contract snapshots need a venv with `pytest==9.0.2 httpx==0.28.1 jsonschema==4.25.1`; regenerate
  with `UPDATE_SNAPSHOTS=1 python -m pytest tests/test_rust_api_contract_snapshots.py` —
  **unfiltered**, because a filtered update run will not delete stale entries.
- Any new file calling `model_source_library(` must be classified in
  `crates/sceneworks-worker/src/model_artifact_inventory.rs` or that guard fires.
- App-level web tests need `ui-preferences` to return `workflowEmbedNoticeSeen: true`, or the
  workflow-embed modal silently blocks the first generation and the test sees zero POSTs.

---

## 6. sc-19711 — `claude/sc-19711-cache-controls-ui` @ `ef845a5f7` (no PR)

Rust compiles; `resolved_cache` 60 pass, `model_cache` API 6 pass, existing web suites unaffected.
New `apps/rust-api/src/model_cache.rs` (status / removal-preview / remove / pin), core
`inspect()` + `open_for_inspection()`, Model Manager typed badges + Keep-locally / Remove-local
wired through a real preview-then-confirm, Settings "Local model copies" card.

**Fixed in passing (real perf defect):** `validate_metadata_shape` ran a **full content re-hash**
inside `read_metadata_unlocked`, so every journal read of a complete entry — catalog GET, every pin
read, every listing — hashed the whole bundle. Slot selection is now paths-and-sizes; the load path
and all writes still validate at full strength. Pinned by a new test.

**NOT done:** all new web tests, `npm run rust:check`, full `npm run check`, base merge + rebuild,
PR creation. Detailed next steps (including the exact test files and the SettingsScreen harness
pattern) are in the branch's Shortcut WIP comment on sc-19711.

**Design note:** the cache policy is **desktop-shell-owned and restart-bound** —
`apps/desktop/src/settings.rs::set_resolved_cache_policy` persists it and `setup.rs` injects it as
env at sidecar spawn, so the API *reports* the running policy rather than persisting it; the UI
compares persisted-vs-effective to say "restart to apply" honestly.

---

## 7. Michael's machine, and a merged-code defect it exposed

```
~/.cache/huggingface/hub  ->  /Volumes/Models/huggingface/hub   (external, 98 repos)
VolumeUUID CD01D2AD-5CB6-480B-8A7E-322816A804BD
```

**Verified working on merged code** (do not re-investigate): a directly-configured `/Volumes/...`
root binds the real volume UUID (`df -P` → `diskutil info -plist` → `VolumeUUID`); eject → typed
`installed_external_unavailable` with no ENOENT and no redownload; reconnect restores; a different
drive at the same mount point fails closed; HF blob symlinks (`snapshots/* -> ../../blobs/<sha>`)
are explicitly handled by `validate_source_file`, which returns the canonical blob path so
sc-19706's `O_NOFOLLOW` confinement passes. `hub.pre-volume-remnant` is a *sibling* of `hub` and is
never enumerated.

**The defect:** `probe_binding` (`external_library.rs` ~:715-725) compares the **literal configured
path** first and returns `IdentityMismatch` before canonical path or volume UUID are consulted, and
`bind_or_probe_validated` never replaces a binding. Michael's stated intent is for users to
reference the external drive **directly** — that transition (same drive, same UUID, same bytes, new
path string) reads as a different library and wedges everything: receipt-backed models pin to
`installed_external_unavailable` forever, non-receipt ones fall to `Missing` and re-enter the
download path for all 98 repos. **Fix is in #2406** (`45d7fe30f`): when the lexical path differs but
canonical path *and* physical identity match, return `Available`; plus surface the relocation prompt
on `IdentityMismatch` with a readable root so the wedge is user-recoverable.

---

## 8. sc-19712 — how to close the epic honestly

Run it **last**, after every code story merges. It is the only story positioned to catch the
between-story integration gaps that this epic keeps producing (§4 is the proof). **If it comes back
clean on the first pass, distrust the validation, not the code.**

It must include a **real external-volume** cycle against `/Volumes/Models`, with the library
configured **directly** (not via Michael's symlink): promote → eject → load from the local tier →
reconnect → verify no redownload, no receipt mutation, installed state preserved. Coordinate the
eject — this Mac is also the `nax-macos` CI runner and that drive holds live weights, so nothing
else may be running in `gpu-local`/`nax-ci`.

---

## 9. Open items for Michael

1. **Licensing (unresolved, needs a decision).** Declaring the face stack as a catalog entry brought
   SCRFD + ArcFace (InsightFace-derived, staged as `SceneWorks/instantid-mlx`) under the
   license-coverage gate. Those **weights are non-commercial research use only** — InsightFace's MIT
   licence covers their *code*, not the weights, and commercial use needs a separate licence
   (contact@insightface.ai). Affects face analysis, KPS, likeness and InstantID lanes. The exposure
   **predates the epic**; sc-19708 only made it visible. The recorded notice is verbatim upstream
   terms, not a placeholder.
2. **Confirmed design carve-out.** `reconcile_removed_source` removes **pinned** entries on model
   uninstall — the single exception to never-remove-pinned, on the reasoning that keeping bytes for
   a model the user deleted is stranded state, not a kept promise. Automatic retention never does
   this. Reversible if you disagree.
3. **Pre-existing flake, surfaced not silenced.**
   `tests::dataset_catalogs::cancellation_while_waiting_for_busy_lease_is_restartable`
   (`apps/rust-api`, untouched by this epic) failed once under full-suite load, then passed 5/5 in
   isolation and on every later full run. Lease-cancellation code; main has had recent flake work in
   that area. Treat a *moving* failing set there as a leaked shared lock, not true flakiness.

---

## 10. Recommended order for the next session

1. **Fix #2407** (§3) — make it compile first (`held.entry.*` + stale callers), then the three-pass
   guard selection, then the remaining review items. Rewrite the two tests that drop the guard
   before asserting. Merge the feature branch (PR is DIRTY) and rebuild.
1b. **Re-review and merge #2406** (§5) — COMPLETE and one unconfirmed lane from mergeable; it carries the
   path-aliasing fix Michael's own configuration needs.
3. **Finish promotion activation** (§4) — trigger, idle drain, admission tests with per-guard
   mutation evidence, the one real end-to-end worker test. Open its PR **after** #2407 merges.
4. **Finish sc-19711** (§6) — web tests, checks, PR.
5. **sc-19712** (§8) — real external-drive campaign, then close the epic.

Per-story detail beyond this doc lives in the Shortcut story comments (all lead with
`[author: claude]`) and in each PR body.
