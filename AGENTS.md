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
