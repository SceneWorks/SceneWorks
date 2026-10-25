# SceneWorks feature development workflow

How a substantial feature goes from a Shortcut epic to `main` in SceneWorks
(and, when the feature touches engines, in the paired `SceneWorks/inference`
repository) without incomplete work reaching `main` or `release/next`.

This document is the **process contract**. The executable procedure lives in the
skills, which are the authority when the two disagree:

- `shortcut-plan` — slice the epic (strict numbered requirements on the epic,
  1–3 story-local acceptance bullets per story, intentional slices, one terminal
  integration story).
- `sceneworks-epic` — drive the epic (parallel local work, reviewed publication batches,
  one terminal requirements/code review per repository, one final pin, one campaign).
- `shortcut-story` — work a single story.

Release and hotfix work is governed by [RELEASING.md](RELEASING.md). Promoting a
completed feature from `main` into `release/next` is part of that process, not
this one.

## Cost model — why the procedure is shaped this way

Story tracking and publication are separate units. Small related stories share
one canonical story branch, one reviewed PR, and one CI cycle. Keep every story's
scope, acceptance criteria, state, and closeout individually visible in Shortcut.
Do not batch unrelated changes or let a large story stall otherwise independent
work.

> parallel isolated implementation → local integration → independent batch
> review against every included story's AC → batched fixes and focused checks →
> one push and PR → green final-head CI → individual story closeouts.

The coordinator owns publication. Implementers and reviewers work locally in
isolated worktrees; they do not independently push or open overlapping PRs.
Epic-wide acceptance runs at feature end, with one combined requirements and
whole-delta code review per repository. Follow-up review and verification target
actual fixes and invalidated evidence rather than restarting the whole process.

## Core invariants

1. Every substantial feature has a complete Shortcut epic and implementation
   plan. A standalone planning request uses `shortcut-plan`'s proposal workflow;
   an instruction to execute the epic authorizes necessary plan/story updates
   throughout delivery without repeated approval.
2. Each epic has one SceneWorks integration branch
   `feature/sc-<epic-id>-<epic-slug>` created from `origin/main`; if the epic
   changes inference, inference has a branch with the **identical name** from its
   own `origin/main`.
3. Publication branches start from the owning feature branch. Small related
   stories share one canonical story branch and PR targeting that feature branch.
   Each merged batch leaves the feature branch buildable and validated.
4. No incomplete feature branch merges into `main`; feature branches never merge
   directly into `release/next`; one feature branch never merges into another.
5. A SceneWorks branch on `main` pins an inference commit reachable from
   inference `main`, never a deleted feature branch. The final pin names the
   landed inference `main` revision. Necessary interim feature pins are coalesced
   into consumer batches to keep them buildable; never create a pin-only helper PR.
6. Always work in an isolated, owned worktree or clone; never commit on `main`.
   The coordinator may prepare terminal fixes, relevant main synchronization,
   and the final pin on an isolated checkout of the feature branch, then push
   it directly by fast-forward. Never force-push shared feature branches.

## Branch model

| Ref | Purpose | Lifetime |
| --- | --- | --- |
| `main` | Completed features and normal development | Permanent, protected |
| `feature/sc-<epic-id>-<epic-slug>` | Combined implementation of one epic | Epic start through verified main merge |
| `story/sc-<story-id>-epic-<epic-id>-<epic-slug>` | One reviewed publication batch; primary story names the branch | Story start through feature-branch merge |
| `release/next` | Candidate for the next release | Governed by `RELEASING.md` |

Lower-case kebab-case slugs. The story branch repeats the owning epic id and the
feature branch's slug exactly (`feature/sc-123-name` ⇒
`story/sc-456-epic-123-name`); a Shortcut-generated branch name is not
sufficient. Do not nest (`feature/sc-123-name/sc-456`): Git refs cannot be both a
branch and a directory. inference enforces this topology in CI
(`scripts/ci/feature_epic_policy.py`); SceneWorks relies on the same convention.
A batch that changes both repositories uses the same primary-story branch name
in both. Its PRs list every included story ID and map each local AC to changes
and validation.

## Current repository state (verified 2026-10-24 — re-query before relying on it)

| | SceneWorks | inference |
| --- | --- | --- |
| `main` rulesets | `Require MR` 17708030 + `Main Ruleset` 20886480 | `Require MR` 20481541 |
| `main` required checks | `web`, `parity`, `candle`, `build-windows`, `check-linux`, `check-macos`, `macOS build, lint and workspace tests (hosted)` | `CI gate` |
| `main` up-to-date required | **no** (`strict: false`) | **no** (`strict: false` since 2026-08-23; it drifted back to `strict: true` twice — do not re-enable) |
| `main` merge methods | merge / squash / rebase | merge commit only |
| `feature/*` rulesets | **none** — all removed, including deletion guard | **none** — all removed, including deletion guard |
| merge queue | **none** (removed 2026-08-11) | **none** (removed 2026-08-11) |

Consequences:

- Use merge commits for feature-target batch PRs to preserve branch history.
  Feature branches allow direct fast-forward updates for terminal preparation;
  this does not remove the workflow requirement to review and validate batches.
- A story PR whose base advanced is still mergeable (`strict: false`); update the
  branch only when the base change can affect the story. The same holds for an
  inference feature→`main` PR (`strict: false` since 2026-08-23); merge `main`
  in yourself when the base moved in a way that can affect the branch.
- Neither repository has feature protection or a feature deletion guard. Do
  not create per-epic rulesets or temporary deletion bypasses. Normal
  fast-forward publication and no force-push remain workflow practices.
- The `Main Ruleset` requires code-owner review, but the repository has no
  `CODEOWNERS` file, so it currently requires nothing.
- `release/next` is covered by no ruleset in either repository.

### CI triggers that matter

A feature branch having no GitHub-required contexts does not make its batch CI
gate empty. Wait for inference `CI gate` and all applicable declared SceneWorks
workflow lanes on the final PR head. Expected irrelevant-path skips are allowed
as designed; missing or pending applicable checks are not passing evidence. Main
required-check policy is unchanged.

- Inspect workflow triggers before publication: a `codex/**` push currently
  triggers SceneWorks `check.yml` even without a PR, and an open PR can add a
  second run. Use canonical `story/*` publication branches for epic batches;
  do not describe arbitrary pushes as free. Local agent branches need no push.
- A PR head update can fan out into expensive platform workflows. Complete
  implementation, independent review, fixes, and scope-appropriate local checks
  before the first push/open. Accumulate subsequent necessary fixes locally and
  publish them together. Preserve green checks on the actual final PR head.
- `macos-mlx.yml` runs its hosted half on PRs to `main` and `feature/*`; the
  self-hosted `nax-worker` job runs only on PRs whose base is `main` (the final
  feature PR) — not per story.
- `windows-candle.yml` runs automatically on relevant PRs to `main` and pushes
  to `main`, and also supports manual dispatch. It is not a required check. Reuse
  the relevant final-PR run instead of dispatching the same lane separately.
- Required workflows keep `merge_group:` (costs nothing; keeps a queue a
  one-line re-enable) and use a `changes` gate job instead of a workflow-level
  `paths:` filter (a path-filtered required check stays Pending forever).

### Gate teardown (2026-08-15/16) — binding on agents

Measurement currency is advisory in CI as well as at runtime (sc-23692). A pin
bump, a changed loader-closure digest, or an attestation naming an older revision
must not fail CI, require attestation renewal, or demand remeasurement. Retain the
original measurement and review provenance; warnings may report the difference.
Tests of currency reporting must use controlled fixtures rather than require the
packaged measurements to remain current. Measurement integrity checks still apply.

The pin-keyed verification gates were deliberately dismantled (sc-19758
`68670a3ee`: four `check.yml` steps `if: false`; sc-19751: license coverage
reports and exits 0; `e14171984`: a pin bump no longer invalidates capability
dumps). The jobs and scripts are retained on purpose.

- A script that exists in `scripts/` but is disabled or unwired is the designed
  state. Do not wire it up, re-enable it, treat it as a requirement, or file a
  story about it.
- Measurement campaigns — capability dumps, memory-matrix regeneration,
  calibration captures, VRAM/canary runs — run **once, at the end of an epic**,
  after the stable final pin and source review, or on explicit request. Never per story,
  never per pin movement. Records reported stale during development is the
  accepted state: measurement never gates, and staleness never changes runtime
  behaviour (sc-22738) — the runtime keeps using a stale record, anchor, curve or
  bound exactly as if it were current; "stale" is only the probe tooling's cue
  to re-capture.
- Story "done" = complete local AC, scoped validation, independent review of
  its publication batch, and green applicable CI on that batch's final PR head
  merged into the feature branch. A separate PR per story is unnecessary.
- Tests over measured corpora assert shape and invariants, never exact
  populations or pinned counts. A re-capture that trips a pinned count means the
  test is wrong; rewrite it to shape in the same PR.

## 1. Plan the epic

Executing an epic is autonomous through validated closeout. The agent owns
implementation decisions, architecture, necessary dependency/build/CI repairs,
tracker updates, review fixes and normal PR merges. Do not ask the user how a
story should be implemented. Inspect requirements, current architecture and
established patterns, choose the correct complete solution and record important
rationale. When a pattern is demonstrably flawed, repair the cause and affected
contracts rather than preserving the defect or introducing a local workaround.
Do not trade implementation correctness or capability coverage for speed.

Only an actual system wall justifies asking for intervention: enforced tool or
policy permission, missing credentials/access, or unavailable required external
resources after feasible recovery paths fail. Report the blocked operation,
evidence, attempted recovery and minimum user action. Keep independent work
moving. Engineering ambiguity, tradeoffs, test failures and review findings are
work to resolve, not routine approval gates. This does not authorize unrelated
external actions or bypass system controls.

Use `shortcut-plan`. The epic description carries the numbered requirements
(E1…En) and the epic acceptance tests, written once. Each story carries 1–3
story-local observable bullets plus `Inherits epic requirements: E…`. Slice by
mechanism (N parameter cells = one story with a table); keep slices intentional; walking
skeleton first; one terminal integration story owns the final pin, the single
measurement campaign, the combined requirements/code review, and feature→`main`.

Do not define a per-story validation matrix. Epic-wide acceptance runs at feature
end. For a new model's inference memory-fit matrix, measure the lowest-memory
and highest-memory supported fits and estimate intermediate entries; do not
execute every Cartesian combination. Select the two anchors from actual
model/runtime settings and the target hardware envelope, record revisions,
settings, device and observed peak memory, and calibrate the established memory
estimator with that evidence. Mark rows measured or estimated with assumptions,
uncertainty and provenance. Estimates do not certify execution or device support.
Investigate endpoint failures or estimator inconsistencies with focused probes,
not an exhaustive grid. Preserve full implementation and meaningful correctness,
contract and integration checks, plus required CI.

Record in the epic: the intended feature branch name, whether inference is in
scope, and the current SceneWorks inference pin.

## Host and runner infrastructure

Before assigning resource lanes or reporting capacity, read
[Host and Runner Guidance](docs/host-and-runner-guidance.md). Discover both
organization and repository runner scopes plus group/repository/workflow access.
Share one relevant live inventory with agents; do not repeat discovery per story.
Keep GitHub-hosted capacity separate from self-hosted listeners and physical
GPU/memory budgets. Busy compatible runners queue ready work rather than block it.

## 2. Create the integration branches

```bash
git fetch origin
git switch -c feature/sc-<epic-id>-<epic-slug> origin/main
git push -u origin feature/sc-<epic-id>-<epic-slug>
```

Repeat with the identical name in inference only when inference changes are in
scope. Confirm the remote ref exists (`git ls-remote origin feature/sc-<epic-id>-*`)
and add the branch URL(s) and starting `main` SHA(s) to the epic. That is the whole
step: feature rulesets require no setup, and a red `main` at branch time is
not a stop condition. Refresh from `main` only for a relevant change, an actual
conflict, or an enforced strict-base policy; base movement alone is insufficient.

## 3. Deliver a reviewed batch

1. Re-fetch and revalidate each candidate story. Keep complete scope and 1–3
   local AC plus inherited epic requirements; move actionable stories through
   Ready to In Progress. Small related stories may share a publication batch.
2. Assign independent implementation to parallel agents in isolated local
   worktrees, bounded by hardware/build resources. Name one coordinator as the
   only owner of integration, pushes, PRs, and CI. Record the batch's primary
   story and included story IDs; use the primary story's canonical branch name
   from the owning feature branch.
3. Integrate the complete changes locally. Run checks appropriate to affected
   behavior and contracts; do not reproduce the full platform CI matrix locally.
   Run explicitly required repository gates, without repeating unchanged checks.
4. A fresh independent agent reviews the integrated batch against every included
   story's local AC and the complete batch diff. Fix findings together and run
   focused verification before the first push. If required evidence is available
   only on a remote runner, the coordinator may publish after available local
   checks, record why, and overlap review; all findings and evidence must be
   resolved before merge. Re-review only unresolved findings
   or materially changed behavior; a fixed item does not restart blanket review.
5. Push once and open one PR targeting the feature branch with authenticated
   `gh pr create`. List all story IDs and map every AC to implementation and
   evidence. Wait with `gh run watch`, then verify applicable checks and the
   current PR head. Batch any CI-driven fixes before another push; never force-push.
6. Merge with a merge commit when the final head's applicable CI is green and
   review findings are resolved. Do not update only because the base moved.
7. Close each included story separately with `author: codex`, its shared PR link,
   outcome and validation. Move it to Done and read back. Done means accepted into
   the feature branch; the terminal story owns epic acceptance and main delivery.

Delete merged publication branches. Do not rerun each story's matrix or AC after
merge; the terminal review covers the combined delta once.

### Cross-repository batches

Publish and merge the inference batch first, then its consumer batch. Use the
same primary-story branch name in both repositories. If the consumer needs the
new engine, coalesce the necessary feature pin with its implementation using
`node scripts/bump-inference.mjs --sha <inference-feature-sha40>` and regenerate
required pin-derived artifacts. Reuse that pin until another engine change is
actually needed. Do not publish standalone pin-only PRs or defer buildability to
the end of the epic. Record the tested engine revision in the consumer PR.

The inference feature branch must remain reachable until the final consumer pin
names inference main and SceneWorks has merged to main. Never delete or rewrite
it while anything depends on it.

## 4. Keep long-running branches current

Refresh only when a main change affects the work, causes conflicts, or an enforced
strict-base policy requires it. Do not sync on a schedule or on every base move.
Feature branches are never rebased or force-pushed. In either repository the
coordinator prepares a needed main merge on an isolated, owned feature checkout,
resolves and locally reviews conflicts, and fast-forwards the remote feature ref.
Coalesce synchronization with pending batch/terminal changes when practical.
There is no sync helper PR or extra CI cycle; the next publication PR validates
the resulting tree. An unrelated inference main sync alone needs no consumer pin.

## 5. Finish the epic (the terminal story)

Ordered because SceneWorks consumes inference by exact commit:

1. Freeze new batch merges. Check whether main changes are relevant or create
   conflicts (§4); do not automatically refresh merely because main moved. On
   isolated owned feature checkouts, complete terminal integration and run
   scope-appropriate local checks. Do not clone the full CI matrix locally or
   open a PR merely to discover predictable source failures.
2. Perform one combined review per repository of the epic requirements,
   acceptance evidence, and whole feature-to-main code delta. Independent agents
   may inspect disjoint areas in parallel; one coordinator consolidates findings.
   Prepare fixes together on the owned feature checkout and fast-forward directly
   to the feature ref. No terminal helper branch or helper PR. Follow-up review
   and checks cover fixes and affected contracts, without repeating the blanket
   review for untouched areas.
3. Open one inference feature-to-main PR, if inference is in scope, after local
   review and fixes. Wait for required CI on its actual final head and merge.
   Record the resulting inference main merge commit; an open PR is not a landed
   dependency. Preserve main protections, and refresh only for demonstrated
   relevance or an enforced strict-base requirement.
4. On the owned SceneWorks feature checkout, set the final pin with
   `node scripts/bump-inference.mjs --sha <inference-main-sha40>` and regenerate
   required artifacts. Complete any remaining consumer integration before final
   acceptance. Review the final consumer delta introduced by the landed pin and
   terminal fixes, with focused checks. Fast-forward directly to the feature ref;
   do not create a pin-only or terminal helper PR, and do not repeat the complete
   epic review solely because the pin hash changed.
5. Run the required epic acceptance and measurement campaign once, after the
   final pin and reviewed implementation are stable. A later hash-only change
   does not invalidate provenance or require replacement. Replace only evidence
   invalidated by an actual measurement-contract/provider-behavior change, or
   on explicit request. Retain complete advertised capability acceptance.
6. Open one SceneWorks feature-to-main PR. Include the epic, all stories, final
   inference pin and landed inference PR, acceptance evidence, migrations and
   compatibility decisions, and limitations. The final PR triggers its applicable
   platform lanes once after local review; do not separately dispatch those same
   lanes or run a duplicate local matrix. Batch necessary CI repairs into one
   update where practical and verify green required checks on the final head.
   Refresh for relevant main changes or enforced policy, never base movement alone.
7. After merge, re-fetch and verify the remote merge commit, close the terminal
   story and epic with acceptance evidence, read their states back, and delete
   feature branches only when no consumers still pin them. Remove owned worktrees.

Delete merged, unpinned feature branches normally: there are no deletion guards
or temporary bypass steps. Verify no consumer still pins the inference feature
branch before deleting it. A retained merged branch is inert; deletion is hygiene,
not a gate.

Merging the feature into `main` does not change `release/next` and does not
publish a release. If the epic also publishes a release, its terminal story and
epic close only after the mandatory post-release `release/next` to main merge
and released-ancestry/version-metadata verification in `RELEASING.md`. This
includes hotfix, patch and version-only releases; no additional approval is needed.

## Failure, abandonment, and scope changes

- Keep a red feature branch open and tracked until repaired; never merge an
  incomplete epic to make the branch disappear.
- Fix defects inside the epic's subject and necessary pre-existing dependency,
  build or CI blockers in the current delivery without routine approval. Track
  material scope changes and protect unrelated user work. Keep unrelated work
  separate and continue independent work while an actual system wall is resolved.
- If the epic is abandoned, record the decision and retained work in Shortcut,
  verify no SceneWorks ref pins its inference branch, then delete both branches.
  Extract reusable work through a separately scoped PR to `main`; never merge a
  partial epic wholesale.
- If requirements change enough to invalidate the plan, update the epic and
  stories before continuing.

## Appendix — CI contract for feature branches

These are the properties the workflows and rulesets must keep satisfying;
`scripts/platform-review-contracts.test.mjs` enforces the workflow half.

- Feature refs have no rulesets, required PRs, check contexts, or deletion
  guards. Normal fast-forward publication and no force-push remain workflow
  practices. Main required-check and PR rules remain unchanged. Reviewed batch
  PRs still require green applicable final-head CI under this workflow.
- Required workflows report their contexts for supported main/feature PR targets
  and keep `merge_group:`. Avoid workflow-level `paths:` filters for required
  contexts; use an explicit changes gate so irrelevant work reports its designed
  result instead of leaving a required context permanently pending.
- `macos-mlx.yml` targets `pull_request: branches: [main, "feature/*"]`;
  `nax-worker` is gated to `base.ref == main` and same-repo heads.
- `windows-candle.yml` stays limited to PRs targeting `main`, keeps its path
  filter, and stays out of the required set.
- Concurrency groups are keyed so one PR run cannot cancel another's verdict.
- Merge queues were removed from both repositories on 2026-08-11 after 103
  groups caught zero integration failures at a median +21 minutes per merge
  (`min_entries_to_merge: 1` formed a group per PR, so nothing batched). Do not
  re-add one without new evidence; if ever reintroduced, a wildcard ruleset
  cannot carry `merge_queue`, so it needs an exact-branch ruleset staged
  disabled, the ref created, then activated. The `merge_group:` triggers are
  retained so that remains a ruleset-only change.
