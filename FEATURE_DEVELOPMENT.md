# SceneWorks feature development workflow

How a substantial feature goes from a Shortcut epic to `main` in SceneWorks
(and, when the feature touches engines, in the private `SceneWorks/inference`
repository) without incomplete work reaching `main` or `release/next`.

This document is the **process contract**. The executable procedure lives in the
skills, which are the authority when the two disagree:

- `shortcut-plan` — slice the epic (strict numbered requirements on the epic,
  1–3 story-local acceptance bullets per story, 5–12 stories, one terminal
  integration story).
- `sceneworks-epic` — drive the epic (waves, lanes, one review + one fix pass
  per story, one feature-end review, one pin bump, one measurement campaign).
- `shortcut-story` — work a single story.

Release and hotfix work is governed by [RELEASING.md](RELEASING.md). Promoting a
completed feature from `main` into `release/next` is part of that process, not
this one.

## Cost model — why the procedure is shaped this way

Measured 2026-08-23 (epics 17137 and 20762): over 90% of epic wall-clock and
token spend was process — repeated reviews, epic-wide acceptance criteria
re-litigated on every story, per-story evidence gathering, pin bumps that shipped
no behaviour, closeout narratives — not code. The procedure below is built for
this per-story budget and nothing more:

> one implementation → one adversarial review against the story's **local** AC →
> one self-verifying fix pass → merge on green required CI → one-line closeout.

Everything epic-wide (every advertised mode, exact identity, fail-closed
behaviour, telemetry agreement, real-weight evidence) is checked **once**, at
feature end. Anything in this document that appears to ask for more than that per
story is a mistake in the document.

## Core invariants

1. Every substantial feature starts with an approved Shortcut epic planned with
   `shortcut-plan`.
2. Each epic has one SceneWorks integration branch
   `feature/sc-<epic-id>-<epic-slug>` created from `origin/main`; if the epic
   changes inference, inference has a branch with the **identical name** from its
   own `origin/main`.
3. Story branches start from the owning feature branch and merge back only
   through PRs targeting that feature branch. Every story merge leaves the
   feature branch buildable and green. Incomplete does not mean broken.
4. No incomplete feature branch merges into `main`; feature branches never merge
   directly into `release/next`; one feature branch never merges into another.
5. A SceneWorks branch on `main` pins an inference commit reachable from
   inference `main`, never a deleted feature branch. The epic lands **exactly one**
   pin bump, at the end, to an inference `main` revision.
6. Always work in an isolated worktree or clone. Never commit on `main` or on a
   `feature/*` branch directly.

## Branch model

| Ref | Purpose | Lifetime |
| --- | --- | --- |
| `main` | Completed features and normal development | Permanent, protected |
| `feature/sc-<epic-id>-<epic-slug>` | Combined implementation of one epic | Epic start through verified main merge |
| `story/sc-<story-id>-epic-<epic-id>-<epic-slug>` | One story in the epic | Story start through feature-branch merge |
| `sync/sc-<epic-id>-main-<date>` | Merge of `main` into a feature branch | Delete after merge |
| `release/next` | Candidate for the next release | Governed by `RELEASING.md` |

Lower-case kebab-case slugs. The story branch repeats the owning epic id and the
feature branch's slug exactly (`feature/sc-123-name` ⇒
`story/sc-456-epic-123-name`); a Shortcut-generated branch name is not
sufficient. Do not nest (`feature/sc-123-name/sc-456`): Git refs cannot be both a
branch and a directory. inference enforces this topology in CI
(`scripts/ci/feature_epic_policy.py`); SceneWorks relies on the same convention.
A story that changes both repositories uses the same story branch name in both.

## Current repository state (verified 2026-08-23 — re-query before relying on it)

| | SceneWorks | inference |
| --- | --- | --- |
| `main` rulesets | `Require MR` 17708030 + `Main Ruleset` 20886480 | `Require MR` 20481541 |
| `main` required checks | `web`, `parity`, `candle`, `build-windows`, `check-linux`, `check-macos`, `macOS build, lint and workspace tests (hosted)` | `CI gate` |
| `main` up-to-date required | **no** (`strict: false`) | **yes** (`strict: true`) |
| `main` merge methods | merge / squash / rebase | merge commit only |
| `feature/*` base policy | 20638194 — same 7 checks, `strict: false`, **merge commit only** | 20638200 — `CI gate`, `strict: false`, merge commit only |
| `feature/*` deletion guard | 20638197 (deletion rule only, no bypass actors) | 20638201 (same) |
| merge queue | **none** (removed 2026-08-11) | **none** (removed 2026-08-11) |

Consequences:

- `gh pr merge --squash` fails on a feature-target PR in either repository. Use
  `--merge`.
- A story PR whose base advanced is still mergeable (`strict: false`); update the
  branch only when the base change can affect the story. An inference
  feature→`main` PR must be up to date (`strict: true`).
- `feature/*` protection is ruleset-managed; the legacy branch-protection
  endpoint returns 404 for a feature branch even though it is protected. Both
  wildcard rulesets are permanent — there is no per-epic ruleset to stage,
  verify, or recover.
- The `Main Ruleset` requires code-owner review, but the repository has no
  `CODEOWNERS` file, so it currently requires nothing.
- `release/next` is covered by no ruleset in either repository.

### CI triggers that matter

- A push to a `story/*` or `sync/*` branch with **no open PR fires nothing** in
  either repository. Push as soon as work is committable. The PR is what costs a
  CI cycle: in SceneWorks five workflows bounded by the ~37-minute macOS lane on
  every PR-head update; in inference the ~20–40-minute `CI gate`.
- `macos-mlx.yml` runs its hosted half on PRs to `main` and `feature/*`; the
  self-hosted `nax-worker` job runs only on PRs whose base is `main` (the final
  feature PR) — not per story.
- `windows-candle.yml` runs only on PRs to `main` whose diff touches its path
  anchor. It is not a required check. The final feature PR triggers it when the
  diff is candle-relevant; nothing else does.
- Required workflows keep `merge_group:` (costs nothing; keeps a queue a
  one-line re-enable) and use a `changes` gate job instead of a workflow-level
  `paths:` filter (a path-filtered required check stays Pending forever).

### Gate teardown (2026-08-15/16) — binding on agents

The pin-keyed verification gates were deliberately dismantled (sc-19758
`68670a3ee`: four `check.yml` steps `if: false`; sc-19751: license coverage
reports and exits 0; `e14171984`: a pin bump no longer invalidates capability
dumps). The jobs and scripts are retained on purpose.

- A script that exists in `scripts/` but is disabled or unwired is the designed
  state. Do not wire it up, re-enable it, treat it as a requirement, or file a
  story about it.
- Measurement campaigns — capability dumps, memory-matrix regeneration,
  calibration captures, VRAM/canary runs — run **once, at the end of an epic**,
  immediately after its single pin bump, or on explicit request. Never per story,
  never per pin movement. Records demoted to floors during development is the
  accepted state.
- Story "done" = the code, its tests, one adversarial review + one fix pass, and
  green required CI on the PR. Nothing more unless the story *is* a measurement
  story.
- Tests over measured corpora assert shape and invariants, never exact
  populations or pinned counts. A re-capture that trips a pinned count means the
  test is wrong; rewrite it to shape in the same PR.

## 1. Plan the epic

Use `shortcut-plan`. The epic description carries the numbered requirements
(E1…En) and the epic acceptance tests, written once. Each story carries 1–3
story-local observable bullets plus `Inherits epic requirements: E…`. Slice by
mechanism (N parameter cells = one story with a table); target 5–12 stories; walking
skeleton first; one terminal integration story owns the pin bump, the single
measurement campaign, the feature-end review, and feature→`main`.

Do not define a per-story validation matrix. The epic's acceptance tests are the
matrix, and they run once at feature end.

Record in the epic: the intended feature branch name, whether inference is in
scope, and the current SceneWorks inference pin.

## 2. Create the integration branches

```bash
git fetch origin
git switch -c feature/sc-<epic-id>-<epic-slug> origin/main
git push -u origin feature/sc-<epic-id>-<epic-slug>
```

Repeat with the identical name in inference only when inference changes are in
scope. Confirm the remote ref exists (`git ls-remote origin feature/sc-<epic-id>-*`)
and add the branch URL(s) and starting `main` SHA(s) to the epic. That is the whole
step: the wildcard rulesets apply automatically, and a red `main` at branch time is
not a stop condition — the feature branch will re-merge `main` before it lands.

## 3. Deliver a story

1. Re-fetch the story; revalidation is the implementer's first step, not a
   separate pass or a comment. Move the story to **In Progress**.
2. If the story's AC restates epic invariants, contains process bullets, or gates
   on terminal evidence, normalise it to 1–3 local bullets +
   `Inherits E…` before briefing anyone (`shortcut-plan` rules; one comment).
3. In an isolated worktree, create
   `story/sc-<story-id>-epic-<epic-id>-<epic-slug>` from the latest remote
   feature branch.
4. Implement the complete story — code, focused tests, generated artifacts
   current CI requires. Run the repository gate (`npm run rust:check` in
   SceneWorks; the applicable lane commands in inference). **Push as soon as it
   is committable** (free); open the PR when it is green locally.
5. **One** adversarial review by a fresh agent against the story's local AC.
   **One** fix pass that self-verifies (mutation per touched assertion). No
   re-review unless a blocker is still `partial` — then one more pass, then
   surface it.
6. Open the PR against the feature branch:

   ```bash
   gh pr create \
     --base feature/sc-<epic-id>-<epic-slug> \
     --head story/sc-<story-id>-epic-<epic-id>-<epic-slug>
   ```

7. Merge (merge commit) when every required check on the PR head is green.
   Poll until nothing is pending; do not rely on `gh pr checks --watch`. Do not
   update the branch merely because the base moved.
8. Closeout comment (`[author: claude]`): PR link, one line of what shipped,
   test count. Move to **Done** and read the state back. Done means accepted
   into the feature branch; it does not mean the feature reached `main`.
   **Terminal evidence never holds a code story open** — pin bumps, campaigns,
   real-weight runs, and feature→`main` belong to the terminal story.

Delete the merged story branch. Do not re-validate the story's AC against the
merged feature head or rerun any matrix; the feature-end review covers the
combined branch once.

### Cross-repository stories

A story that changes inference and SceneWorks is **two PRs in sequence**,
inference first. The SceneWorks side **does not bump the pin**:

- To build SceneWorks against the new engine for discovery, run
  `node scripts/bump-inference.mjs --sha <inference-feature-sha40>` in the
  worktree, build and test, then revert every pin site
  (`git checkout -- Cargo.toml Cargo.lock crates/sceneworks-worker/Cargo.toml crates/sceneworks-memory-adapter/Cargo.toml`)
  and commit only the story. Report what was verified against in the PR.
- Schedule SceneWorks stories that need the new engine **after** the epic's
  single pin bump (§5), where they get real CI against the landed pin.
- A mid-epic bump costs a full SceneWorks CI cycle for no behaviour, a
  `Cargo.lock` edit every in-flight branch must merge, staled calibration
  records, and pins a SHA that may never reach inference `main`. If the schedule
  seems to need one, the schedule is wrong.

The inference feature branch must remain reachable until the SceneWorks feature
is on `main`. Never delete or rewrite it during the epic.

## 4. Keep long-running branches current

Feature branches are never rebased or force-pushed. Merge `main` in through a PR
(the ruleset requires one):

```bash
git fetch origin
git switch -c sync/sc-<epic-id>-main-<date> origin/feature/sc-<epic-id>-<epic-slug>
git merge origin/main
git push -u origin sync/sc-<epic-id>-main-<date>
gh pr create --base feature/sc-<epic-id>-<epic-slug> --head sync/sc-<epic-id>-main-<date>
```

Green required CI **is** the review of a sync PR; no adversarial review, no
matrix rerun. Synchronize when a story actually depends on something that landed
on `main`, when a hotfix on `main` conflicts with the branch, and once before
feature end (§5). Not on a schedule, not "whenever drift increases". Resolve
conflicts toward the epic's architecture. If inference is in scope, synchronize
inference `main` into its feature branch the same way; do not bump the SceneWorks
pin for it.

## 5. Finish the epic (the terminal story)

Ordered because SceneWorks consumes inference by exact commit:

1. Freeze new story merges in both feature branches. Synchronize both from their
   respective `main` (§4).
2. **Feature-end review**, once, after the last code story merges: the full
   `feature/*` vs `main` diff in each repository against the epic's numbered
   requirements and acceptance tests, at the session model. Fix → re-review
   loops allowed here, cap 3, each round one batched PR into the feature branch.
   Stories that land after the pin bump (step 4) get their ordinary per-story
   review; the feature-end review is not repeated for them.
3. Merge inference `feature/sc-<epic-id>-<slug>` → inference `main` through `CI
   gate` (`strict: true`: the branch must contain current `main`). Record the
   resulting inference `main` merge commit. An open PR is not a merged dependency.
4. **The epic's one pin bump**: on a `story/*` branch off the SceneWorks feature
   branch, `node scripts/bump-inference.mjs --sha <inference-main-sha40>`,
   regenerate only what current CI on the PR requires, merge through the feature
   branch. Then run any SceneWorks stories that were waiting on the new engine.
5. **The epic's one measurement campaign**, if the epic calls for one,
   immediately after the bump (the bump stales the records; bump-then-capture is
   the only ordering that yields current evidence).
6. Open one PR from the SceneWorks feature branch to `main`. Body: epic link and
   story list, final inference pin and the inference `main` PR, migrations and
   compatibility decisions, known limitations. The PR itself runs `nax-worker`
   and (when the diff is candle-relevant) `windows-candle.yml`; do not dispatch
   them separately beforehand. If `main` advanced while the PR sat open, merge
   it in and let the checks re-run.
7. After the merge: re-fetch and verify the remote merge commit, add the epic
   closeout comment, move the epic to **Done**, delete the SceneWorks feature
   branch, delete the inference feature branch once nothing pins it, remove the
   epic's worktrees.

Deleting a `feature/*` branch requires a temporary bypass actor on the
deletion-guard ruleset (it holds only the deletion rule, so the bypass weakens
nothing else); restore `bypass_actors: []` afterwards. A merged, unpinned
feature branch left in place is inert — deletion is hygiene, not a gate.

Merging the feature into `main` does not change `release/next` and does not
publish a release.

## Failure, abandonment, and scope changes

- Keep a red feature branch open and tracked until repaired; never merge an
  incomplete epic to make the branch disappear.
- Discovered work inside the epic's subject: fix it in the same PR or a second PR
  the same session. Outside the subject: still fix it, batched into one PR. File
  a story only for the three genuine blockers (hardware, a credential only the
  owner can set, a decision only the owner can make).
- If the epic is abandoned, record the decision and retained work in Shortcut,
  verify no SceneWorks ref pins its inference branch, then delete both branches.
  Extract reusable work through a separately scoped PR to `main`; never merge a
  partial epic wholesale.
- If requirements change enough to invalidate the plan, update the epic and
  stories before continuing.

## Appendix — CI contract for feature branches

These are the properties the workflows and rulesets must keep satisfying;
`scripts/platform-review-contracts.test.mjs` enforces the workflow half.

- SceneWorks `feature/*` requires the same seven contexts as `main`; inference
  `feature/*` requires `CI gate`. Both base policies: pull request required,
  merge commits only, non-fast-forward blocked, no bypass actors.
- Every required workflow triggers on `pull_request` for any base and keeps
  `merge_group:`; none uses a workflow-level `paths:` filter (use a `changes`
  gate job that reports success when irrelevant).
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
