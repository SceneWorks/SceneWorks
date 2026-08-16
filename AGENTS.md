# SceneWorks Agent Notes

## Releases

- Follow [RELEASING.md](RELEASING.md) for the official release, hotfix,
  inference-pin, publication, and failed-candidate recovery workflow.

## Feature epics

- Follow [FEATURE_DEVELOPMENT.md](FEATURE_DEVELOPMENT.md) for epic integration
  branches, story PR targets, mirrored inference work, final merge ordering, and
  the CI protections required before using this workflow.
- Read its *Current state* section before assuming anything about branch
  protection. As of 2026-08-11 neither SceneWorks nor inference has a merge
  queue; `main` is `strict: false` and `feature/*` is `strict: true` in both.
  Queue staging/activation/recovery procedures are obsolete in both repositories.
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
