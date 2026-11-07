# SceneWorks Agent Notes

## Releases

- Follow [RELEASING.md](RELEASING.md) for the official release, hotfix,
  inference-pin, publication, and failed-candidate recovery workflow.
- Every published release, including a hotfix, patch or version-only release,
  finishes by merging `release/next` back into `main` through a protected PR
  and merge commit. Verify released ancestry and version metadata on remote
  main before closeout. This is part of release delivery, with no extra approval;
  preserve published tags and versioned branches unchanged.

## Feature epics

- An instruction to run an epic authorizes autonomous end-to-end delivery,
  including necessary tracker/plan updates, dependency and CI repairs, review
  fixes, PRs, merges and closeout. Do not ask how to implement a story or request
  routine approvals. Resolve decisions from the greater architecture and established
  patterns; repair demonstrably wrong patterns and affected contracts. Escalate
  only an actual system wall such as missing access, credentials, required external
  resources or an enforced tool permission. Keep independent work moving.
- For a new model's inference memory-fit matrix, measure the lowest-memory and
  highest-memory supported fits and estimate intermediate entries using the
  established estimator. Record exact settings/device and measured provenance;
  label estimates and uncertainty explicitly. Do not execute the Cartesian grid.
  This sampling exception does not reduce implementation scope, meaningful
  correctness tests, or required CI.
- Follow [FEATURE_DEVELOPMENT.md](FEATURE_DEVELOPMENT.md) for epic integration
  branches, story PR targets, mirrored inference work, final merge ordering, and
  the CI protections required before using this workflow.
- Its *Current repository state* table is the branch-protection reference.
  Neither repository has a merge queue; both `main` branches use
  `strict: false`. All feature rulesets, including deletion guards, are removed
  in both repositories. Fast-forward updates and no force-push remain workflow
  practices. Main protections are unchanged.
  The skills (`shortcut-plan`, `sceneworks-epic`, `shortcut-story`) are the
  executable procedure; the document is the contract.
  Track every story separately, but batch small related stories into one canonical
  story branch, one independently reviewed PR, and one CI cycle. Parallel agents
  implement locally in isolation; one coordinator owns all publication. Complete
  review, batched fixes, and focused local validation before the first push.
  Terminal fixes, relevant main sync, and the final pin fast-forward directly
  from an owned feature checkout: no terminal helper PR. Keep required final-head
  CI and full epic acceptance. Feature batches still wait for inference `CI gate`
  and applicable SceneWorks workflow lanes, even without required contexts;
  missing or pending checks do not pass. Review the whole delta once per repository and
  follow up only on changed areas. Never force-push shared feature branches.
- The pin-keyed verification gates were **deliberately dismantled** on
  2026-08-15/16 (sc-19758, sc-19751, `e14171984`). A script that exists but is
  disabled or unwired is the designed state — never a blocker, never something
  to wire up or re-enable. Measurement work (capability dumps, calibration,
  memory matrix, canaries) runs once at epic end or on explicit request, never
  per code change. See the *Gate teardown* note in FEATURE_DEVELOPMENT.md's
  *Current state* section.

## Pull Requests

- For this repository, create pull requests with authenticated `gh pr create` directly.
- Do not try the GitHub connector PR creation first; it repeatedly fails with `Resource not accessible by integration` and then requires falling back to `gh` anyway.

## SceneWorks runner discovery

For SceneWorks, inference, and ChatWorks, discover the live Actions pool rather than using a remembered runner count. Query both registration scopes with pagination:

```sh
gh api --paginate orgs/SceneWorks/actions/runners
gh api --paginate repos/SceneWorks/REPOSITORY/actions/runners
gh api --paginate orgs/SceneWorks/actions/runner-groups
```

For each relevant organization group, check its runners and repository access (`orgs/SceneWorks/actions/runner-groups/GROUP_ID/runners` and, for selected visibility, `.../repositories`). Combine organization runners accessible to the target repository with repository runners, keep each runner's registration scope with its ID, then match every workflow `runs-on` label and any group/workflow restrictions. Either inventory alone can undercount the usable pool. Surface failed inventory/access queries instead of treating them as zero runners.

Report registered/online, label-compatible, and currently idle counts separately. `busy` is transient; submit ready work and let Actions queue it. A `cuda` label does not imply `real-weights`, a runner name is not automatically a label, and multiple listeners do not imply multiple physical GPUs. Preserve existing device restrictions and required workflow concurrency.

The CUDA pool was consolidated into the organization `self-hosted-gpu` group on
2026-10-24: `cuda-windows`, `cuda-windows-2`, `cuda-windows-3`, and
`cuda-windows-4`, accessible to SceneWorks, inference, and ChatWorks. Only the
first two carry `real-weights`. Treat this as a dated baseline and verify live
inventory before reporting capacity.

Read [Host and Runner Guidance](docs/host-and-runner-guidance.md) for the complete
dated inventory, physical host relationships, cache/shell details, and hosted
versus self-hosted execution lanes. Refresh changing facts before use.
