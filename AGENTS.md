# SceneWorks Agent Notes

## Releases

- Follow [RELEASING.md](RELEASING.md) for the official release, hotfix,
  inference-pin, publication, and failed-candidate recovery workflow.

## Feature epics

- Follow [FEATURE_DEVELOPMENT.md](FEATURE_DEVELOPMENT.md) for epic integration
  branches, story PR targets, mirrored inference work, final merge ordering, and
  the CI protections required before using this workflow.
- Its *Current repository state* table is the branch-protection reference.
  Neither repository has a merge queue; `feature/*` is `strict: false` and
  merge-commit only in both; SceneWorks `main` is `strict: false`, inference
  `main` is `strict: true`. The skills (`shortcut-plan`, `sceneworks-epic`,
  `shortcut-story`) are the executable procedure; the document is the contract.
  Per story: one implementation, one adversarial review against story-local AC,
  one fix pass, merge on green, one-line closeout — nothing more.
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
