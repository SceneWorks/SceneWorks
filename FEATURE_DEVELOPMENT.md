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
4. Story branches start from the owning feature branch, carry both the story
   and epic ids, and merge back only through reviewed PRs targeting that
   feature branch.
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
| `sync/sc-<epic-id>-main-<date>[-<sequence>]` | Reviewed synchronization of `main` into a feature branch | Delete after merge |
| `release/next` | Candidate for the next release | Governed by `RELEASING.md` |

Use lower-case kebab-case slugs. The epic id in every story branch makes the
intended integration base mechanically derivable before a PR is allowed to
run. Do not try to nest story branches below the exact feature branch name, such as
`feature/sc-123-name/sc-456-story`: Git refs cannot contain both a branch and a
directory at the same path.

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
   server, real-weight, migration, and compatibility evidence.
7. Record the intended feature branch name in the epic.

Do not create an epic branch as a substitute for incomplete requirements.

## 2. Create the integration branches

Use current remote state and a clean checkout. GitHub rejects a merge-queue rule
whose condition is the wildcard `feature/*`, while an active exact queue rule
rejects creation of its not-yet-existing ref. The bootstrap command must
therefore treat the applicable exact queue ruleset and ref as one durable,
recoverable transaction; when inference is in scope, that transaction covers
both repositories' queue rulesets and refs.

Run the complete read-only preflight for every applicable repository before the
first mutation: capture the current `origin/main` SHA; verify both wildcard
ruleset layers; and verify every configured required context on that exact
commit is terminal `success`, `skipped`, or `neutral` from the expected GitHub
Actions app. Pending, missing, wrong-app, or failed contexts are a stop
condition. Planning and bootstrap exhaust every ruleset and check-run page;
page-one-only evidence is not sufficient. Bootstrap completes the exhaustive
ruleset summary and detail reads, then repeats the exact planned main-SHA check
matrix immediately before its first journaled mutation. Once a mutation intent
is durable, recovery uses the frozen SHAs and policy digests instead of
re-evaluating later main or check state. Persist the plan, then for each
repository:

1. create or verify the exact queue ruleset in disabled mode and record its ID
   and payload digest;
2. create the feature ref at the captured immutable SHA under the active
   wildcard base and deletion guards;
3. immediately activate only the recorded exact queue ruleset; and
4. query the effective rules and require the five-rule aggregate.

Only after every repository reaches that state may bootstrap create a workspace
or declare success. Do not loosen `do_not_enforce_on_create=false` or silently
choose an older commit to make branch creation succeed.

Recovery resumes recognized durable states instead of blindly rolling back:

- missing queue/no ref: create the recorded policy disabled, then continue;
- disabled queue/no ref: create the ref, then activate;
- disabled queue/ref at the recorded SHA: activate;
- active queue/ref at the recorded SHA: verify the aggregate;
- active queue/no ref: first disable that exact recorded, still-matching policy,
  then create the ref and reactivate it; and
- missing queue/ref at the recorded SHA: create the recorded policy disabled,
  then activate it.

An unexpected ref SHA, duplicate matching ref or ruleset, payload drift, or an
unrecognized partial state fails closed. A partial success in one repository is
resumed after revalidation; it is not force-deleted or rewritten. A ref whose
queue rule is missing or disabled remains protected by the wildcard layers but
is unsafe: expose no workspace and admit no story PR until recovery completes.

Run the fail-closed automation from a clean checkout with an explicit external
state path and an already-created empty workspace parent. Export a read-only
Shortcut API token as `SHORTCUT_API_TOKEN`; the token is sent to `curl` through
stdin and is never placed in the process argument list:

```bash
mkdir -p /absolute/path/to/sc-<epic-id>-workspaces
npm run feature:epic -- plan \
  --epic sc-<epic-id> \
  --slug <epic-slug> \
  --stories sc-<story-id>,sc-<story-id> \
  --workspace /absolute/path/to/sc-<epic-id>-workspaces \
  --state /absolute/path/to/sc-<epic-id>-state.json
npm run feature:epic -- bootstrap \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --apply
```

`plan` accepts only the fixed `SceneWorks/SceneWorks` and
`SceneWorks/inference` pair. Before writing the durable transaction state it
requires the supplied story ids to equal the complete live Shortcut epic
inventory. It checks the epic-reported total against the returned story count,
rejects duplicate or cross-epic ids, and resolves every story workflow state
from the exhaustive workspace workflow map. It then requires both live default
branches to be `main`, resolves their exact heads,
and parses every real Cargo dependency table, per-dependency table, target,
workspace dependency, and patch entry to prove that each
`SceneWorks/inference` git dependency uses the canonical repository URL and one
consistent 40-hex revision reachable from inference `main`. Comments and
metadata are not dependencies; alternate URL spellings, `branch`, and `tag`
selectors fail closed. `plan` does not create a ref or clone.

Before its first durable mutation intent, `bootstrap --apply` re-resolves both
main heads and refuses a stale plan. Before any branch ref exists, it verifies
the two active wildcard feature policies in each repository and creates one
exact-ref queue ruleset per repository with enforcement disabled. After that
global staging barrier, it completes each repository in order: create and verify
that repository's ref, activate only its recorded exact ruleset id, and verify
that all three policy layers are effective on that branch. It provisions clean
state-owned clones only after both repositories pass that sequence.
This ordering is mandatory: an active exact merge queue blocks creation of its
target ref with HTTP 422. Every ruleset, ref, and activation intent is journaled
before the API call and reconciled afterward. The automation never updates,
force-pushes, or deletes a ref, and never updates an unknown or pre-existing
ruleset. Main-head freshness is a pre-transaction check: once the first durable
mutation intent is journaled, recovery keeps using the frozen planned SHAs and
policy digests even if either live `main` advances. A matching pair is
idempotent; a duplicate, divergent, or unrecorded partial fails closed. The
inference branch is mandatory for this cross-repository workflow and uses the
exact same feature branch name.

### Adopt an existing protected train

When both canonical feature refs and their exact queue rulesets already exist,
do not run create mode or reconstruct their starting points from current
`main`. Use the explicit read-only adoption mode:

```bash
npm run feature:epic -- plan \
  --epic sc-<epic-id> \
  --slug <epic-slug> \
  --stories sc-<story-id>,sc-<story-id> \
  --workspace /absolute/path/to/sc-<epic-id>-workspaces \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --adopt-existing
npm run feature:epic -- bootstrap \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --apply
```

Adoption requires both mirrored refs. The plan records separate immutable
planned-main and adopted-feature SHAs, proves each planned main is an ancestor
of its adopted feature, captures the exact trusted required-check runs on both
adopted heads, verifies the recorded wildcard and exact active/effective policy
IDs and digests, and binds the SceneWorks inference pin to both inference
baselines by ancestry. Adoption never creates or updates a ref or ruleset;
`bootstrap` only revalidates that frozen evidence and clones the exact adopted
heads. Any drift requires a new plan.

Record every already-merged PR needed to explain pre-automation history with an
explicit immutable disposition:

```bash
npm run feature:epic -- adopt-pr \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --repo sceneworks \
  --number <pr-number> \
  --disposition precanonical-story|historical-train|governance-proof
```

`precanonical-story` is limited to `story/sc-N-<slug> -> main` and must be in
both planned-main and canonical-feature history. `historical-train` accepts
only the strict current story or sync topology and must be in canonical-feature
history. `governance-proof` accepts only
`story/sc-N-epic-N-ruleset-proof -> feature/ruleset-proof-sc-N`; it resolves
that exact exceptional topology but never satisfies story coverage. Every
disposition requires an already-merged same-repository PR, immutable head and
merge SHAs, exact trusted required checks, and a timestamped merge-queue event
at or before the merge. Open, unmerged, non-ancestral, mismatched, or waiver
evidence fails closed.

If the live epic grows before final integration begins, adopt only the additive
scope explicitly:

```bash
npm run feature:epic -- refresh-stories \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --apply
```

The refresh journals the previous and next inventory digests, every added or
removed id, and every workflow-state-map change. Removals and cross-epic moves
fail closed. Any live or recorded PR from the feature branch freezes the
inventory permanently; post-freeze drift remains visible as a failed audit
gate rather than being adopted.

Every command that reads or writes the state/report transaction holds an
exclusive sibling lock for its full lifetime. A crashed same-host lock is
reclaimed only when its recorded PID is provably dead. A missing, corrupt, or
foreign-host owner is never guessed stale; after external verification, an
operator may explicitly provide
`--recover-stale-lock-after-seconds <N>` (minimum 300). A live same-host owner
is never displaced regardless of age.

Immediately verify that all three applicable active layers aggregate: the wildcard
base policy, the wildcard deletion guard, and the exact-branch queue policy. Do
not begin story merges while the branch allows unreviewed direct pushes, lacks
required checks, can be deleted, or has no merge queue.

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
4. Create `story/sc-<story-id>-epic-<epic-id>-<epic-slug>` from the latest
   remote feature branch in an isolated worktree or clone. The epic id and slug
   must match the target feature branch.
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

   Record the immutable live head/base evidence in the bootstrap state after
   the PR exists:

   ```bash
   npm run feature:epic -- record-pr \
     --state /absolute/path/to/sc-<epic-id>-state.json \
     --repo sceneworks \
     --number <pr-number>
   ```

9. Merge through the feature branch's required checks and merge queue. Verify
   the remote merge rather than treating queue entry as completion.
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

4. Regenerate and validate every pin-derived lockfile, provenance, license,
   compatibility, capability, memory, calibration, and closure artifact
   required by the script and current CI.
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
   the required CI and merge queue.
6. Record the exact resulting inference `main` commit. A queue or open PR is not
   a merged dependency.
7. Update the SceneWorks feature branch to that inference-main commit with
   `bump-inference.mjs` and regenerate every derived artifact at the final pin.
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

Before enqueueing, verify:

- every required story is Done and no epic blocker remains;
- both feature branches include current respective `main`;
- the final integration/acceptance story passed against the combined head;
- the SceneWorks pin equals the landed inference final-PR merge commit and that
  commit remains reachable from inference `main`;
- required CI reports on the actual feature-to-main PR head; and
- no unrelated changes entered through synchronization.

Record each final integration PR at its immutable source head. Before the PR
enters the merge queue, record every required privileged workflow dispatch; the
command queries the live object and refuses an unsuccessful, non-dispatch,
unexpected, queued, merged, or inexact source-head receipt:

```bash
npm run feature:epic -- record-pr \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --repo sceneworks \
  --number <final-pr-number> \
  --run sceneworks:<windows-candle-run-id> \
  --run sceneworks:<macos-mlx-run-id>
```

After the PR has landed, record deployment evidence in a separate invocation.
The deployment must name the actual landed merge commit; an open PR's synthetic
test-merge SHA and the pre-merge source head are both rejected:

```bash
npm run feature:epic -- record-pr \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --repo sceneworks \
  --number <final-pr-number> \
  --deployment sceneworks:<deployment-id>
```

Omit `--deployment` only when the plan did not declare
`--deployment-required`. Inference run receipts use `inference:<run-id>` and
must name a privileged inference workflow declared by the plan.

Before requesting cleanup, generate a fresh live report:

```bash
npm run feature:epic -- audit \
  --state /absolute/path/to/sc-<epic-id>-state.json \
  --report /absolute/path/to/sc-<epic-id>-audit.json
```

The report re-fetches the complete live Shortcut epic, story inventory, and
workflow-state map. It separately requires exact inventory equality, exact
equality with the frozen workflow-state map, every
story completed in a `done` state, no blocked story, and no unresolved blocker;
the epic itself remains open until the documented post-cleanup closeout step.
It also exhaustively paginates all pull requests targeting each feature branch,
all pull requests from each feature branch, and every repository-wide pull
request. It then fails cleanup for an open train PR, an unrecorded merged train
PR, a recorded receipt missing from live discovery, or a live topology violation.
These live gates remain separate from per-story merged-PR coverage.

The report also keeps these claims separate: each canonical or adopted PR's merged state,
exact recorded head, required-context conclusion, and merge-queue evidence; current-main
ancestry; inclusion of the immutable planned/adopted baselines; exact adopted-head
checks and paired pin evidence; the final inference pin and its reachability from live inference
`main`, plus exact equality with the landed inference final-PR merge commit;
pre-queue privileged runtime dispatches at the exact frozen source head;
post-merge deployment evidence at the landed merge commit; and cleanup
eligibility. Cleanup additionally requires every recorded PR to be merged and
each live feature ref to equal its immutable merged final-PR source head. A
merged PR does not excuse later feature-ref drift. Check-run and PR-timeline
evidence is exhaustively paginated, including queue events after the first 100
timeline items. A merge-queue event proves the recorded head only when it has a
valid `created_at` at or after that PR receipt's `recordedAt`; older or
untimestamped events do not satisfy cleanup.
`audit` never deletes a branch or opens a
ruleset bypass window. A failed dimension remains explicit
instead of being collapsed into a single completion label.

Merge through the protected `main` merge queue. Afterward, re-fetch and verify
the exact remote merge commit, post-merge state, and any required runtime or
deployment evidence. Only then:

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
- If bootstrap creates one remote ref and the other create-ref fails, do not
  improvise an update or delete. The journal records `recovery-required`; after
  resolving the external failure run:

  ```bash
  npm run feature:epic -- recover \
    --state /absolute/path/to/sc-<epic-id>-state.json \
    --complete
  ```

  Recovery first reconciles or creates the complete state-owned exact queue
  pair, then globally disables each recorded active queue whose ref is missing.
  It next completes each repository's missing ref at its recorded main SHA,
  activates only that repository's recorded exact ruleset id, and verifies its
  effective policy before continuing. If recovery finds
  a missing ref whose matching exact queue was already activated by this same
  transaction, it first updates only that recorded id back to the disabled
  staging payload, creates the missing ref, and immediately reactivates it. A
  partial policy failure occurs before refs, so recovery can finish the disabled
  policy pair and then process both planned repositories. A partial or ambiguous
  activation can occur after one repository is fully protected and before the
  second is complete; recovery distinguishes the live disabled and active
  semantic payload digests before issuing any PUT. If
  matching policies and refs already exist because an API response was
  ambiguous or only local cloning failed, it performs no duplicate remote
  mutation and idempotently provisions the missing clone. Dirty, unrecorded,
  reused, or conflicting workspace paths are refused. Failed clone staging
  directories are retained and reported rather than silently deleted.
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

### Current state observed on 2026-08-08

The process above is not yet fully enforced:

- SceneWorks ruleset `Require MR` targets only the default branch.
- Inference ruleset `Require MR` targets only the default branch.
- Therefore `feature/*` branches in both repositories currently lack the PR,
  required-check, merge-queue, non-fast-forward, and deletion rules applied to
  `main`.
- SceneWorks `check.yml`, `desktop-macos-check.yml`,
  `desktop-linux-check.yml`, and `desktop-windows.yml` already trigger for
  arbitrary pull-request bases and `merge_group` events.
- SceneWorks `macos-mlx.yml` supports `merge_group` but restricts ordinary PRs
  to `main`; its required macOS/MLX status would not report on a story PR whose
  base is `feature/*`.
- SceneWorks `windows-candle.yml` restricts ordinary PRs to `main` and does not
  currently support `merge_group`; it cannot be made a feature-branch required
  check without restructuring its triggers and path selection.
- Inference `ci.yml` already supports arbitrary pull requests and
  `merge_group`; its workflow trigger is suitable for feature branches, but no
  feature-branch ruleset currently requires its `CI gate`.

Re-query live rules and workflow files before implementation; this section is a
dated baseline, not a substitute for current inspection.

### Required before the first feature branch

#### 1. Add feature-branch rulesets

Create three layers of `Feature integration` rulesets in each repository.
GitHub rulesets use `fnmatch`, so the two durable wildcard layers target
`feature/*`, which intentionally matches the one-slash branch format defined
above. The third layer targets one exact feature branch.

GitHub's ruleset API rejects `merge_queue` when the condition contains a
wildcard (`Invalid rule 'merge_queue': Wildcard ref names are not supported
when merge queue is enabled`). An active exact queue rule also rejects initial
creation of its matching ref with `Changes must be made through the merge
queue`. Consequently the wildcard base policy must not contain the queue rule.
Every bootstrap must create or verify a separate disabled exact queue ruleset
for `feature/sc-<epic-id>-<epic-slug>`, create the checked ref under the active
wildcard guards, immediately activate the exact rule, and verify the aggregate.
A wildcard queue payload or a feature ref with a missing/disabled exact queue
rule is a configuration error, not a degraded mode.

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

The exact-branch queue ruleset must also have no bypass actors. It contains the
`merge_queue` rule for that one feature ref, uses merge commits, and uses merge
groups of one entry unless deliberate batch validation is accepted. Persist its
ID and payload digest while it is disabled, then activate only that exact
recorded policy after ref creation. If the disposable proof shows that GitHub
does not aggregate a queue-only exact ruleset with the wildcard pull-request
and status rules, repeat those rules in the exact ruleset; never remove them
from the wildcard layer. Record the final active state with the epic bootstrap
evidence.

The wildcard deletion-guard ruleset must contain only the deletion rule and
normally have no bypass actors. After every non-cleanup assertion passes on a
disposable proof branch, or after the post-merge checklist succeeds, freeze the
target branch and require no open PR or queue entry. First attempt the proof
deletion with the exact queue still active. If GitHub's queue rule rejects
deletion, disable only the recorded exact queue policy and re-verify that the
wildcard base and deletion layers remain active before continuing.

An administrator may then temporarily add one closed maintainer team or cleanup
automation as the deletion guard's exact cleanup actor. If neither exists, one
explicitly named repository administrator may be the temporary actor; record
its immutable actor ID, the approved proof-or-completed branch names, and the
start/end of the cleanup window. Delete only those branches and restore
`bypass_actors: []` immediately in a finally-style cleanup step, whether the
deletion succeeds or fails. If deletion failed after the exact queue was
disabled, reactivate that recorded queue policy and verify all five effective
rules before releasing the freeze. If reactivation also fails, keep the branch
explicitly unsafe, expose no workspace, admit no PR, and record an incident for
recovery. Never leave deletion authority active during an epic. Bypass is
ruleset-wide: never put the deletion rule and its cleanup bypass in the base
policy or exact queue ruleset, because that would also let the cleanup actor
bypass pull-request, status-check, non-fast-forward, or queue protections. If
the ref is absent, audit the restored guard and then delete the obsolete exact
queue ruleset. If activation failed during bootstrap, keep the disabled-queue
ref as a fail-closed resumable state; use this same bounded cleanup only after
an explicit abandonment decision.

For a new epic, create each exact queue with enforcement `disabled`. GitHub
rejects creation of a missing target ref when its exact merge queue is already
active. Stage both exact queues disabled before creating either ref. Then, for
each repository, resolve its create-only ref to the recorded main SHA, update
that recorded exact ruleset id to the active payload, and verify the effective
branch-rule aggregate contains the wildcard base, wildcard deletion guard, and
exact merge queue before proceeding to the other repository. Do not expose
either clone before both aggregates verify.
Recovery may repeat this transition only for ids owned by the durable
transaction journal. If such an owned queue is active while its ref is missing,
recovery journals a bounded active-to-disabled-to-active repair around the
create-only ref operation. An ambiguous disable or activation is reconciled
against the live semantic payload before any retry. Recovery never disables a
pre-existing queue, adopts an unknown id, or leaves a recovered queue disabled.

Evaluate mode, when available, is an optional preview of matched refs and rule
evaluations; it cannot prove enforced denials or merge-queue admission. Activate
the three layers on disposable branches before running those enforcement tests,
and do not create the real epic branches until the complete proof matrix passes.
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
4. Ensure every required workflow includes `merge_group`. GitHub explicitly
   requires that trigger for checks used by a merge queue; see
   [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#merge_group).
5. Keep concurrency keyed so one PR or merge-group run cannot cancel another
   required verdict.

Do not add duplicate `push: feature/*` builds merely because the branches are
new. If the merge queue tests the exact speculative commit that lands, that run
is the integration verdict. Add post-merge feature pushes only when they prove a
separate property.

#### 3. Decide the privileged runtime policy

`windows-candle.yml` and `macos-mlx.yml`'s `nax-worker` supply authoritative
CUDA-linked and Apple matrix-unit runtime coverage, but they run on scarce,
self-hosted hardware and are not feature-branch required checks. Use this
policy:

- feature-target story and synchronization PRs run the required hosted checks,
  including `macOS build, lint and workspace tests (hosted)`;
- `nax-worker` does not auto-run for PRs targeting `feature/*` or for merge
  groups; it continues to run for the final same-repo integration PR targeting
  `main`;
- `windows-candle.yml` remains limited to ordinary PRs targeting `main` and
  stays out of the merge queue; and
- at the frozen final feature head, explicitly dispatch both privileged
  workflows and record their exact head SHA and successful run URLs in the epic
  before opening the final PR to `main`.

Apply the same rule to privileged inference real-weight lanes: ordinary CI may
select their impact without having authority or hardware to execute them. An
epic that requires real-weight evidence cannot close on a compile-only lane.

#### 4. Validate inference feature branches

After adding the inference ruleset:

- prove that a story PR to `feature/*` produces `CI gate`;
- prove that its merge queue produces and accepts the `merge_group` verdict;
- prove that a SceneWorks PR can fetch an exact commit reachable only from the
  active private inference feature branch;
- prove that `bump-inference.mjs` regenerates and validates all pin-derived
  artifacts at that revision; and
- prove that deletion is blocked until authorized cleanup.

#### 5. Add policy automation

SceneWorks owns `scripts/feature-epic.mjs` and its pure, injectable library.
The `plan`, `bootstrap --apply`, `recover --complete`, `adopt-pr`, `record-pr`, and `audit`
commands implement the stateful operator workflow described above.

The unconditional `parity` path in `check.yml` runs `ci-policy` for every event.
It enforces these topology rules without relying on a mutable epic database:

- `story/sc-<story-id>-epic-<epic-id>-<slug>` targets exactly
  `feature/sc-<epic-id>-<slug>`;
- `sync/sc-<epic-id>-main-<date>[-<sequence>]` targets a feature branch with that epic id; use the positive sequence suffix when more than one sync is required on the same date;
- `feature/sc-<epic-id>-<slug>` targets `main` only;
- feature-train PR heads come from the same repository; and
- protected PRs and feature-target merge groups resolve exactly one live
  canonical `feature/sc-<epic-id>-*` branch from that checkout's fixed `origin`;
  zero, duplicate, divergent, or wrong-slug candidates fail; and
- malformed feature-to-feature, feature-to-`release/next`, story-to-`main`, and
  unowned feature-target PRs fail.

Only `story/sc-<story-id>-epic` names are reserved for the train, so ordinary
`story/sc-<story-id>-<slug>` PRs, including forks, remain unaffected. Final pin
policy runs only for a feature-to-main PR, a merge group targeting `main`, or a
push to `main`. At those boundaries it
uses `git ls-files` to enumerate every tracked `Cargo.toml`, parses Cargo's
dependency forms instead of matching lines, requires every real inference git
dependency to use the canonical URL and the same exact 40-hex `rev`, resolves
live inference `main`, and fails unless GitHub proves the pin is its ancestor.
Story PRs and feature-target merge groups do not make that permanent-pin claim.

The future minor-release workflow remains separate work: feature completion
must not be conflated with promotion into `release/next` or publication.

### CI acceptance test

Before declaring the feature process operational, create disposable mirrored
feature branches and exercise all of the following:

1. A docs-only SceneWorks story PR reports every required status without a
   permanently pending path-filtered check.
2. A web/Rust story PR runs the expected hosted matrix.
3. An MLX-sensitive PR runs the required macOS/MLX context.
4. A candle-sensitive PR produces the hosted candle verdict and the chosen CUDA
   runtime evidence.
5. An inference story PR produces `CI gate`, merges through its queue, and can be
   pinned by the SceneWorks feature branch.
6. SceneWorks and inference merge queues validate their speculative commits and
   do not cancel one another.
7. The API refuses a wildcard queue payload and an active queue rejects initial
   ref creation; disabled staging followed by ref creation and immediate queue
   activation produces all five effective rules.
8. Direct and force pushes to feature branches fail.
9. An unauthorized deletion fails, while the authorized post-epic cleanup path
   succeeds without bypassing the base or exact queue policies.
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
