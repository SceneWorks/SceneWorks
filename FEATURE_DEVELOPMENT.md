# SceneWorks feature development workflow

This is the official process for implementing a substantial SceneWorks feature
from a Shortcut epic while keeping incomplete work isolated from `main` and
`release/next`. It supports multiple concurrent feature epics and coordinated
changes in the private `SceneWorks/inference` repository.

This document defines the intended process and the CI/ruleset work required to
enforce it. Do not treat feature branches as fully protected until the
implementation checklist in [CI and repository configuration](#ci-and-repository-configuration)
has been completed and verified in both repositories.

Release and hotfix work remains governed by [RELEASING.md](RELEASING.md).
Promoting completed features from `main` into `release/next` and cutting a minor
release is a separate process and is not defined here.

## Core invariants

1. Every substantial feature starts with an approved Shortcut epic containing
   implementation-ready requirements, an implementation plan, and executable
   stories.
2. Each epic has one protected SceneWorks integration branch named
   `feature/sc-<epic-id>-<epic-slug>` created from current `origin/main`.
3. If the epic changes inference, the inference repository has a branch with
   the exact same name, also created from its current `origin/main`.
4. Story branches start from the owning feature branch and merge back only
   through reviewed PRs targeting that feature branch.
5. Every story merge leaves the feature branch buildable, internally
   consistent, and green. Incomplete does not mean broken.
6. No incomplete feature branch merges into `main`.
7. A SceneWorks feature that uses inference permanently pins a commit reachable
   from inference `main`, never a deleted feature branch.
8. The epic remains open until the final SceneWorks feature PR is merged into
   `main` and post-merge state and CI are verified.
9. Feature branches never merge directly into `release/next`. Release promotion
   happens only after the feature is complete on `main`.
10. Use isolated worktrees or clones. Never disturb an unrelated dirty checkout.

## Branch model

| Ref | Purpose | Lifetime |
| --- | --- | --- |
| `main` | Completed features and normal development | Permanent, protected |
| `feature/sc-<epic-id>-<epic-slug>` | Combined implementation of one epic | Epic start through verified main merge |
| `story/sc-<story-id>-epic-<epic-id>-<epic-slug>` | One story in the epic | Story start through verified feature-branch merge |
| `sync/sc-<epic-id>-main-<date>` | Reviewed synchronization of `main` into a feature branch | Delete after merge |
| `release/next` | Candidate for the next release | Governed by `RELEASING.md` |

Use lower-case kebab-case slugs. The story branch must repeat the owning epic id
and the feature branch's canonical slug exactly. For example,
`feature/sc-123-name` accepts `story/sc-456-epic-123-name`; a Shortcut-generated
branch name or a story-title slug is not sufficient unless it already matches
this repository policy. Do not try to nest story branches below the exact
feature branch name, such as `feature/sc-123-name/sc-456-story`: Git refs cannot
contain both a branch and a directory at the same path.

If one story changes both repositories, use the same story branch name in
SceneWorks and inference. A branch name records ownership; the Shortcut epic
and story remain the requirements and status sources of truth.

## 1. Prepare the epic

Before creating branches:

1. Re-fetch the live Shortcut epic and every proposed story.
2. Reconcile the requirements and plan with current SceneWorks and inference
   source, open PRs, current pins, and platform constraints.
3. Ensure every implementation-plan item is represented by a story or task.
4. Record dependencies between stories and between repositories.
5. Add an explicit final integration/acceptance story covering the whole epic,
   cross-repository pins, documentation, migrations, and runtime evidence.
6. Define the validation matrix, including required MLX, candle/CUDA, desktop,
   server, real-weight, migration, and compatibility evidence. Matrix evidence
   is gathered once, at epic completion — not re-proven per story (see *Gate
   teardown* under **CI and repository configuration**).
7. Record the intended feature branch name in the epic.

Do not create an epic branch as a substitute for incomplete requirements.

## 2. Create the integration branches

Use current remote state and a clean checkout.

Run the complete read-only preflight for every applicable repository before the
first mutation: capture the current `origin/main` SHA; verify both wildcard
ruleset layers; and verify every configured required context on that exact
commit is terminal `success`, `skipped`, or `neutral` from the expected GitHub
Actions app. Pending, missing, wrong-app, or failed contexts are a stop
condition. Persist the plan before mutating anything.

Neither repository has a merge queue, so there is no per-branch queue ruleset to
stage and no ordering constraint on ref creation. Create the branch at the
captured immutable SHA under the active wildcard base and deletion guards:

```bash
git fetch origin
git switch -c feature/sc-<epic-id>-<epic-slug> origin/main
git push -u origin feature/sc-<epic-id>-<epic-slug>
```

Create the inference branch only when inference changes are part of the approved
scope, using the identical commands and branch name against that remote.

Then verify that **two** layers aggregate in each applicable repository: the
wildcard base policy and the wildcard deletion guard. Do not create a per-branch
queue ruleset in either repository; both had one and both had it removed after
measurement showed it bought nothing (see *Current state* under **CI and
repository configuration**).

Only after every applicable repository reaches that verified state may bootstrap
create a workspace or declare success. Do not loosen
`do_not_enforce_on_create=false` or silently choose an older commit to make
branch creation succeed. An unexpected ref SHA, duplicate matching ref or
ruleset, payload drift, or an unrecognized partial state fails closed. A partial
success in one repository is resumed after revalidation; it is not force-deleted
or rewritten. Do not begin story merges while a branch allows unreviewed direct
pushes, lacks required checks, or can be deleted.

Add the following to the Shortcut epic:

- SceneWorks branch URL and starting main SHA;
- inference branch URL and starting main SHA, when present;
- current SceneWorks inference pin;
- required CI/runtime lanes; and
- the final integration story.

## 3. Deliver a story

Treat each story as a small PR even though its base is an epic branch.

1. Re-fetch the live story, current feature head, related PRs, and current code.
2. Revalidate that the story is still required and that its dependencies are
   already integrated or explicitly ordered.
3. Move the story to In Progress immediately before editing.
4. Create `story/sc-<story-id>-epic-<epic-id>-<epic-slug>` from the latest remote
   feature branch in an isolated worktree or clone. The epic id and slug must
   match the target feature branch exactly.
5. Implement the complete story, focused regression tests, generated artifacts,
   and required cross-repository work. Do not silently defer required capability.
6. Run focused checks and the complete applicable repository gates.
7. Perform a fresh adversarial review and resolve every valid finding.
8. Open the PR against the feature branch, not `main`:

   ```bash
   gh pr create \
     --base feature/sc-<epic-id>-<epic-slug> \
     --head story/sc-<story-id>-epic-<epic-id>-<epic-slug>
   ```

9. Merge through the feature branch's required checks. Neither repository has a
   merge queue; `feature/*` requires the branch to be up to date with its base
   (`strict: true`), so merge the feature head in and let the checks re-run
   rather than waiting for a queue to do it for you. Verify the remote merge
   rather than treating a green check as completion.
10. Validate the acceptance criteria against the new combined feature head.
11. Add a Shortcut closeout comment containing the PR, merge commit, tests,
    runtime evidence, limitations, and tracked follow-ups.
12. Move the story to Done only when its complete acceptance criteria are
    satisfied on the integrated feature branch. Story Done means accepted into
    the epic; it does not mean the feature has reached `main` or a release.

Delete only the merged story branch. The feature branch remains active.

### Cross-repository stories

For a story that changes inference:

1. Branch from the inference feature branch and merge the inference story PR
   back into that branch.
2. Record the exact merged inference feature SHA.
3. On the SceneWorks story branch, update the pin with the repository-owned
   script:

   ```bash
   node scripts/bump-inference.mjs --sha <inference-feature-sha40>
   ```

4. Regenerate the pin-derived artifacts that `bump-inference.mjs` and current
   CI actually require — as of the 2026-08-15/16 gate teardown that is a short
   list, and a pin bump no longer invalidates capability dumps or obligates any
   measurement work (see *Gate teardown* under **CI and repository
   configuration**).
5. Merge the SceneWorks story PR only after inference and SceneWorks validation
   both pass at the exact paired revisions.

The inference feature branch must remain reachable while SceneWorks points to
it. Never delete or rewrite it during the epic.

## 4. Keep long-running branches current

Shared feature branches are never rebased or force-pushed. Synchronize them by
merging current `main` through a reviewed PR:

```bash
git fetch origin
git switch -c sync/sc-<epic-id>-main-<date> \
  origin/feature/sc-<epic-id>-<epic-slug>
git merge origin/main
git push -u origin sync/sc-<epic-id>-main-<date>
gh pr create \
  --base feature/sc-<epic-id>-<epic-slug> \
  --head sync/sc-<epic-id>-main-<date>
```

Synchronize:

- after a relevant hotfix reaches `main`;
- before starting a story whose dependencies changed on `main`;
- whenever drift materially increases conflict risk;
- before whole-epic acceptance testing; and
- immediately before the final PR to `main`.

Resolve conflicts against the feature's approved architecture and rerun the
combined validation matrix. If inference is involved, synchronize inference
`main` into its feature branch first, update the SceneWorks pin to that new
feature head, and validate the pair again.

Never merge one feature branch directly into another. If epic B depends on epic
A, complete A into `main`, then synchronize `main` into B. Record the dependency
in both epics.

## 5. Complete an inference-backed epic

Cross-repository finalization is ordered because SceneWorks consumes inference
by exact commit SHA:

1. Freeze new story merges in both feature branches.
2. Synchronize inference `main` into the inference feature branch.
3. Synchronize SceneWorks `main` into the SceneWorks feature branch.
4. Update SceneWorks to the final inference feature head and run the complete
   paired validation and final adversarial review.
5. Open the inference feature PR against inference `main` and merge it through
   the required CI. inference `main` is `strict: false` and has no queue, so if
   inference `main` moved after those checks went green, merge it in and let them
   re-run before merging.
6. Record the exact resulting inference `main` commit. An open PR is not a merged
   dependency.
7. Update the SceneWorks feature branch to that inference-main commit with
   `bump-inference.mjs` and regenerate the derived artifacts current CI
   requires at the final pin. The epic's single measurement campaign, if the
   epic calls for one, runs here — after this bump, never earlier (see *Gate
   teardown* under **CI and repository configuration**).
8. Rerun the complete SceneWorks validation matrix and final review. The pin
   change after the inference merge invalidates earlier final-CI claims.
9. Proceed to the SceneWorks final merge only when inference `main`, the
   SceneWorks pin, generated evidence, and validated source all agree.

Keep the inference feature branch until the SceneWorks feature is verified on
`main`, even though the permanent pin is now reachable from inference `main`.

## 6. Merge the completed feature into main

Open one PR from the SceneWorks feature branch to `main`. Its body must contain:

- the Shortcut epic and complete story list;
- requirements and acceptance-criterion mapping;
- SceneWorks and inference base/final SHAs;
- the final inference pin and inference main PR, when applicable;
- migrations and compatibility decisions;
- complete local, hosted, platform, and real-weight evidence;
- independent-review verdicts; and
- explicit limitations and separately tracked follow-ups.

Before merging, verify:

- every required story is Done and no epic blocker remains;
- both feature branches include current respective `main`;
- the final integration/acceptance story passed against the combined head;
- the SceneWorks pin is reachable from inference `main`;
- required CI reports on the actual feature-to-main PR head; and
- no unrelated changes entered through synchronization.

Merge through the protected `main` ruleset. SceneWorks `main` has **no merge
queue** and does **not** require the branch to be up to date (`strict: false`),
so nothing re-verifies this PR against a `main` that moved after its checks went
green — if `main` advanced while the final PR sat open, merge it in and let the
required checks re-run before merging. Afterward, re-fetch and verify the exact
remote merge commit, post-merge state, and any required runtime or deployment
evidence. Only then:

1. add the epic closeout comment;
2. move the epic to Done;
3. delete the SceneWorks feature branch through the authorized cleanup path;
4. delete the inference feature branch if nothing still pins it; and
5. remove local worktrees owned by the epic.

Merging the feature into `main` does not change `release/next` and does not
publish a release.

## Failure, abandonment, and scope changes

- Keep a red feature branch open and tracked until it is repaired; do not merge
  an incomplete epic to make the branch disappear.
- If the epic is abandoned, document the decision and retained work in
  Shortcut, verify that no SceneWorks ref pins its inference branch, then delete
  both branches through the authorized cleanup path.
- Extract reusable work through a separately scoped story and PR to `main`.
  Never merge a partial epic wholesale.
- Add newly discovered required work to the epic immediately. Do not hide it in
  a TODO or declare the epic complete around it.
- If requirements change enough to invalidate the implementation plan, update
  the epic and stories before continuing.

## CI and repository configuration

### Current state verified on 2026-08-11

#### Gate teardown (2026-08-15/16) — read before regenerating anything

The pin-keyed verification gates are **deliberately dismantled**:

- sc-19758 (`68670a3ee`): four `check.yml` steps are switched off with `if: false`
  — the per-lane inference closure-digest verification, the NC-weights source
  scan, the About→Licenses guard, and the gen-core version-skew guard. The jobs
  and their scripts are retained **on purpose** (the `parity` aggregator asserts
  a member floor, and re-enabling any of them is deleting one line).
  `release.yml` still scans built bundles for NC weights.
- sc-19751: `check-license-coverage.mjs` reports and exits 0; `--strict` gates
  only during a deliberate compliance pass.
- `e14171984` (PR #2360): an inference pin bump **no longer invalidates
  capability dumps**. A dump is stale only when a provider's declared
  capabilities actually change; a media/audio revision disagreement is recorded
  (`audioInferenceRevision`), not refused, and `bump-inference.mjs` no longer
  fails a bump on a stale facts file.

Consequences, binding on agents:

- **A script that exists in `scripts/` but is disabled or unwired is the
  designed state, not a blocker.** Do not wire it up, re-enable an `if: false`,
  treat its existence as a requirement, or file a story about it.
- **Measurement campaigns — capability dumps, memory-matrix regeneration,
  calibration captures, VRAM/canary runs — run once, at the end of an epic**
  (immediately after its single pin bump), or when explicitly requested. Never
  per story and never per pin movement. Calibration records demoted to floors
  during development is the accepted state.
- Where older prose in this document says to regenerate or validate "every"
  pin-derived artifact, the operative authority is **what current CI on the
  actual PR requires**, which after the teardown is much less than the lists
  suggest.
- Story "done" = the code, its tests, adversarial review, and green required CI
  on the PR. Nothing more unless the story itself is a measurement story.
- **Tests over measured corpora** (calibration records and evidence, capability
  dumps, session logs, survey populations) **assert shape and invariants —
  schema validity, non-emptiness, resolving cross-references, per-item
  properties — never exact populations, pinned ids, or historical counts.** A
  pinned count over a corpus that is supposed to churn cannot fail on a bug and
  is guaranteed to fail on every legitimate re-capture; it is a gate on
  measurement wearing a test's clothes. When a re-capture trips one, the defect
  is in the test: rewrite that assertion and its siblings to shape in the same
  PR rather than hand-updating the numbers. (Golden/parity fixtures that pin a
  fixed input's output are a different, legitimate class.)

#### Merge queues

**Neither repository has a merge queue.** Both were removed on 2026-08-11 —
SceneWorks first, inference the same day — and the two are now structurally
identical.

| | SceneWorks | inference |
| --- | --- | --- |
| `main` ruleset | `Require MR` (17708030) | `Require MR` (20481541) |
| queue on `main` | **none** | **none** |
| up-to-date required on `main` | no (`strict: false`) | no (`strict: false`) |
| `feature/*` base policy | 20638194 | 20638200 |
| queue on `feature/*` | **none** | **none** |
| `feature/*` up-to-date | yes (`strict: true`) | yes (`strict: true`) |
| deletion guard | 20638197 | 20638201 |

Both `main` rulesets carry `deletion`, `non_fast_forward`, `pull_request`, and
required status checks. Both `feature/*` base policies carry `non_fast_forward`,
`pull_request`, and required status checks at `strict: true`.

The deliberate remaining differences are policy, not structure: SceneWorks
requires code-owner review and allows merge/squash/rebase; inference requires no
code-owner review and allows merge commits only. Required contexts differ by
repository — SceneWorks requires `web`, `parity`, `candle`, `build-windows`,
`check-linux`, `check-macos`, and `macOS build, lint and workspace tests
(hosted)`; inference requires `CI gate`. inference also retains a **disabled**
`No Force-Push` ruleset (18886583) with no SceneWorks equivalent.

#### Why there is no queue

Removed after measurement, not preference. Over 5.4 days the SceneWorks queue
produced **103 merge groups and caught zero integration failures**, while adding
a median 21m (mean 35m, p90 43m) to every merge. Every group contained exactly
one PR — `min_entries_to_merge: 1` forms a group the instant one entry arrives,
so `max_entries_to_build` never engaged, `min_entries_to_merge_wait_minutes` was
inert, nothing was ever amortized, and the entire cost was additive. Its only
observable effects were two evictions, on PRs that then took 1h32m and 5h33m.
inference ran the identical configuration, including the same
`min_entries_to_merge: 1`, and was removed to match.

Do not re-add a queue to either repository without new evidence. A merge queue
earns its cost through batching under concurrency; at these merge rates it
batched nothing. The `merge_group:` triggers remain in the workflows deliberately
so re-enabling is a one-line ruleset change — and a required lane *without* that
trigger strands a queued group until the response timeout evicts its entries.

#### Consequences for this document

- Every step that stages, activates, or recovers an exact per-branch queue
  ruleset is obsolete in **both** repositories. Feature branches have **two**
  layers, not three, and ref creation needs no transaction in either repo.
- `release/next` is covered by **no ruleset in either repository** — no required
  checks, no PR requirement, no force-push or deletion guard. It is the least
  protected branch in the hotfix path.

Workflow triggers, verified the same day:

- SceneWorks `check.yml`, `desktop-macos-check.yml`, `desktop-linux-check.yml`,
  and `desktop-windows.yml` trigger for arbitrary pull-request bases and
  `merge_group`.
- SceneWorks `macos-mlx.yml` now targets `pull_request: branches: [main,
  "feature/*"]` with no PR path filter, so its required status reports on
  feature-target PRs. `platform-review-contracts.test.mjs` enforces this.
- SceneWorks `windows-candle.yml` still restricts ordinary PRs to `main`, keeps
  a PR path filter, and has no `merge_group` trigger. It is deliberately not a
  required check; a contract test asserts it stays out of the required set.
- Inference `ci.yml` supports arbitrary pull requests and `merge_group`, and its
  `CI gate` is required on `feature/*`.

Re-query live rules and workflow files before implementation; this section is a
dated snapshot, not a substitute for current inspection.

### Ruleset and workflow requirements

These layers already exist in both repositories (see *Current state* above); this
section is the contract they must keep satisfying, and the recipe for any future
repository or feature branch.

#### 1. Feature-branch rulesets

GitHub rulesets use `fnmatch`, so the durable wildcard layers target `feature/*`,
which intentionally matches the one-slash branch format defined above.

**Two layers in each repository** — the wildcard base policy and the wildcard
deletion guard. There is no queue layer in either repository, so there is no
exact-branch ruleset, no staging transaction, and no per-epic ruleset lifecycle
to manage.

Should a queue ever be reintroduced, the constraint that shaped the old
three-layer design still applies and is worth recording: GitHub's ruleset API
rejects `merge_queue` when the condition contains a wildcard (`Invalid rule
'merge_queue': Wildcard ref names are not supported when merge queue is
enabled`), and an active exact queue rule rejects initial creation of its
matching ref with `Changes must be made through the merge queue`. A queue
therefore cannot live in the wildcard base policy, and its exact ruleset and ref
have to be staged disabled, created, then activated as one recoverable
transaction. Do not reintroduce that machinery without first fixing the batching
parameters that made the queue worthless.

SceneWorks must require at least the same integration contexts currently
required on `main`:

- `web`
- `parity`
- `candle`
- `build-windows`
- `check-linux`
- `check-macos`
- `macOS build, lint and workspace tests (hosted)`

Inference must require `CI gate`.

The wildcard base-policy ruleset in each repository must have no bypass actors and
must:

- require pull requests;
- block force pushes and non-fast-forward updates;
- require code-owner/review behavior consistent with each repository's `main`;
- require current-head status checks;
- use merge commits so story and synchronization provenance is retained.

Note the batch-size trap that removed both repositories' queues:
`min_entries_to_merge: 1` forms a group the instant a single entry arrives, so
`max_entries_to_build` never engages, `min_entries_to_merge_wait_minutes` is
inert, and the queue amortizes nothing while charging a full verification pass
per PR. Both repositories ran exactly that configuration. If a queue is ever
reintroduced, set the batching parameters so a group can actually hold several
entries, and measure group sizes before trusting it — do not pay for a group of
one.

The wildcard deletion-guard ruleset must contain only the deletion rule and
normally have no bypass actors. After every non-cleanup assertion passes on a
disposable proof branch, or after the post-merge checklist succeeds, freeze the
target branch and require no open PR. With no queue in either repository,
deletion is gated only by the deletion guard itself.

An administrator may then temporarily add one closed maintainer team or cleanup
automation as the deletion guard's exact cleanup actor. If neither exists, one
explicitly named repository administrator may be the temporary actor; record
its immutable actor ID, the approved proof-or-completed branch names, and the
start/end of the cleanup window. Delete only those branches and restore
`bypass_actors: []` immediately in a finally-style cleanup step, whether the
deletion succeeds or fails. If deletion fails, keep the branch explicitly unsafe,
expose no workspace, admit no PR, and record an incident for recovery. Never
leave deletion authority active during an epic. Bypass is ruleset-wide: never put
the deletion rule and its cleanup bypass in the base policy, because that would
also let the cleanup actor bypass pull-request, status-check, or
non-fast-forward protections. If the ref is absent, audit the restored guard. Use
this same bounded cleanup only after an explicit abandonment decision.

Evaluate mode, when available, is an optional preview of matched refs and rule
evaluations; it cannot prove enforced denials. Activate both layers on disposable
branches before running those enforcement tests, and do not create the real epic
branches until the complete proof matrix passes.
GitHub's ruleset pattern and bypass behavior is documented in
[Creating rulesets for a repository](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository).

#### 2. Make every required SceneWorks check report

At minimum:

1. Keep `macos-mlx.yml` ordinary PR targeting on both `main` and `feature/*`.
   Repository contract tests must reject a required workflow whose base filter
   would prevent its status from reporting on a feature-target PR.
2. Audit every required workflow and job condition for assumptions about
   `main`, `pull_request.base.sha`, or `github.event.before`. A merge-group run
   must use `github.event.merge_group.base_sha` where a base is required.
3. Ensure path-irrelevant required jobs report success through an always-created
   change-selection job. Do not use a workflow-level `paths` filter for a
   required check, because an absent workflow leaves the requirement pending.
4. Keep `merge_group` on every required workflow even though neither repository
   has a queue. The trigger costs nothing while no such event fires, and it keeps
   re-enabling a queue a one-line ruleset change; a required lane without it
   strands a queued group until `check_response_timeout_minutes` evicts its
   entries, which re-orders the queue silently rather than failing.
   GitHub requires that trigger for checks used by a merge queue; see
   [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#merge_group).
5. Keep concurrency keyed so one PR or merge-group run cannot cancel another
   required verdict.

Do not add duplicate `push: feature/*` builds merely because the branches are
new. With no queue in either repository, the integration verdict is the
required-check run on the PR head itself, and `strict: true` on `feature/*`
guarantees that head contains the current base. Add post-merge feature pushes
only when they prove a separate property.

#### 3. Decide the privileged runtime policy

`windows-candle.yml` and `macos-mlx.yml`'s `nax-worker` supply authoritative
CUDA-linked and Apple matrix-unit runtime coverage, but they run on scarce,
self-hosted hardware and are not feature-branch required checks. Use this
policy:

- feature-target story and synchronization PRs run the required hosted checks,
  including `macOS build, lint and workspace tests (hosted)`;
- `nax-worker` does not auto-run for PRs targeting `feature/*`, nor for merge
  groups should a queue ever be reintroduced; it continues to run for the final
  same-repo integration PR targeting `main`;
- `windows-candle.yml` remains limited to ordinary PRs targeting `main`, keeps
  its PR path filter, and carries no `merge_group` trigger, so it stays out of
  the required set; and
- at the frozen final feature head, explicitly dispatch both privileged
  workflows and record their exact head SHA and successful run URLs in the epic
  before opening the final PR to `main`.

Apply the same rule to privileged inference real-weight lanes: ordinary CI may
select their impact without having authority or hardware to execute them. An
epic that requires real-weight evidence cannot close on a compile-only lane.

#### 4. Validate inference feature branches

After adding the inference ruleset:

- prove that a story PR to `feature/*` produces `CI gate`;
- prove that an out-of-date story PR is blocked until its base is merged in and
  `CI gate` re-runs (`strict: true`, no queue);
- prove that a SceneWorks PR can fetch an exact commit reachable only from the
  active private inference feature branch;
- prove that `bump-inference.mjs` regenerates and validates all pin-derived
  artifacts at that revision; and
- prove that deletion is blocked until authorized cleanup.

#### 5. Add policy automation

Implement small fail-closed automation rather than relying only on memory:

- a feature bootstrap command or agent skill that creates and records mirrored
  branches from exact main SHAs and verifies both wildcard layers in each
  repository;
- a PR policy check that rejects a feature story targeting `main` or the wrong
  epic branch;
- a check that rejects a final SceneWorks feature PR while its inference pin is
  reachable only from `feature/*`;
- an epic closeout audit covering story state, open PRs, current-main ancestry,
  CI conclusions, exact pins, and branch cleanup eligibility; and
- a future minor-release workflow that promotes completed `main` into
  `release/next` without conflating feature completion with release publication.

### CI acceptance test

Before declaring the feature process operational, create disposable mirrored
feature branches and exercise all of the following:

1. A docs-only SceneWorks story PR reports every required status without a
   permanently pending path-filtered check.
2. A web/Rust story PR runs the expected hosted matrix.
3. An MLX-sensitive PR runs the required macOS/MLX context.
4. A candle-sensitive PR produces the hosted candle verdict and the chosen CUDA
   runtime evidence.
5. An inference story PR produces `CI gate`, merges, and can be pinned by the
   SceneWorks feature branch.
6. A story PR whose base has moved is blocked as out-of-date (`strict: true` on
   `feature/*`) in both repositories, and merges once the base is merged in and
   the required checks re-run. With no queue validating a speculative commit,
   this is the only integration verdict.
7. Creating a feature ref succeeds directly under the two wildcard layers, in
   both repositories, with no queue ruleset staged.
8. Direct and force pushes to feature branches fail.
9. An unauthorized deletion fails, while the authorized post-epic cleanup path
   succeeds without bypassing the base policy.
10. The final inference-first merge and SceneWorks pin transition succeed without
   leaving `main` dependent on a feature-only inference commit.

Capture the disposable PRs and run URLs in the CI implementation PR. Only after
this test passes should agents treat the feature process as fully automated.

## Success criteria

A feature epic is complete only when:

- all required stories are accepted on the combined feature branch;
- the final integration story and full validation matrix pass;
- inference changes are merged to inference `main` and SceneWorks pins that
  exact reachable revision;
- the SceneWorks feature PR is merged to `main` through required CI;
- live post-merge state is verified;
- the Shortcut epic is closed with evidence; and
- mirrored feature branches are removed through the authorized cleanup path.
