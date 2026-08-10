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
| `story/sc-<story-id>-<story-slug>` | One story in the epic | Story start through verified feature-branch merge |
| `sync/sc-<epic-id>-main-<date>` | Reviewed synchronization of `main` into a feature branch | Delete after merge |
| `release/next` | Candidate for the next release | Governed by `RELEASING.md` |

Use lower-case kebab-case slugs. Shortcut story ids are globally unique, so a
story branch does not need the epic id in its name. Do not try to nest story
branches below the exact feature branch name, such as
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

Use current remote state and a clean checkout:

```bash
git fetch origin
git switch -c feature/sc-<epic-id>-<epic-slug> origin/main
git push -u origin feature/sc-<epic-id>-<epic-slug>
```

Create the inference branch only when inference changes are part of the approved
scope. Use the identical command and branch name in that repository.

Immediately verify that the appropriate feature-branch ruleset applies. Do not
begin story merges while the branch allows unreviewed direct pushes or lacks
required checks.

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
4. Create `story/sc-<story-id>-<story-slug>` from the latest remote feature
   branch in an isolated worktree or clone.
5. Implement the complete story, focused regression tests, generated artifacts,
   and required cross-repository work. Do not silently defer required capability.
6. Run focused checks and the complete applicable repository gates.
7. Perform a fresh adversarial review and resolve every valid finding.
8. Open the PR against the feature branch, not `main`:

   ```bash
   gh pr create \
     --base feature/sc-<epic-id>-<epic-slug> \
     --head story/sc-<story-id>-<story-slug>
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
- the SceneWorks pin is reachable from inference `main`;
- required CI reports on the actual feature-to-main PR head; and
- no unrelated changes entered through synchronization.

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

Create two layered `Feature integration` branch rulesets in each repository,
both targeting `feature/*`. GitHub rulesets use `fnmatch`; this pattern
intentionally matches the one-slash branch format defined above.

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

The core merge-policy ruleset in each repository must have no bypass actors and
must:

- require pull requests;
- block force pushes and non-fast-forward updates;
- require code-owner/review behavior consistent with each repository's `main`;
- require current-head status checks;
- use merge commits so story and synchronization provenance is retained;
- use a merge queue after every required workflow supports `merge_group`;
- use merge groups of one entry unless deliberate batch validation is accepted.

The second ruleset must contain only the deletion rule. Give only a closed
maintainer team or cleanup automation a bypass on that deletion-only ruleset for
branch removal after verified epic completion. Bypass is ruleset-wide: never
put the deletion rule and its cleanup bypass in the core merge-policy ruleset,
because that would also let the cleanup actor bypass pull-request, status-check,
and non-fast-forward protections. Matching rulesets aggregate, so the no-bypass
core policy remains enforced while the authorized actor deletes a completed
feature branch.

Start the rulesets in Evaluate mode if available, prove them with disposable
branches, and activate them only after the check matrix is complete. GitHub's
ruleset pattern and bypass behavior is documented in
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

Implement small fail-closed automation rather than relying only on memory:

- a feature bootstrap command or agent skill that creates and records mirrored
  branches from exact main SHAs;
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
5. An inference story PR produces `CI gate`, merges through its queue, and can be
   pinned by the SceneWorks feature branch.
6. SceneWorks and inference merge queues validate their speculative commits and
   do not cancel one another.
7. Direct and force pushes to feature branches fail.
8. An unauthorized deletion fails, while the authorized post-epic cleanup path
   succeeds.
9. The final inference-first merge and SceneWorks pin transition succeed without
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
