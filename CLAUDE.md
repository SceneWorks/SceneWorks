# SceneWorks

Agent instructions for this repository live in **[AGENTS.md](AGENTS.md)** — read it
first. This file exists so the `CLAUDE.md` convention resolves; AGENTS.md is the
source of truth, and anything beyond the pointers below belongs there rather than
here.

## Workflows

- **[FEATURE_DEVELOPMENT.md](FEATURE_DEVELOPMENT.md)** — epic integration
  branches, story PR targets, mirrored inference work, final merge ordering, and
  the current ruleset/CI state.

  Read its *Current repository state* table before acting on any branch
  protection assumption. **Neither SceneWorks nor inference has a merge queue**
  (removed 2026-08-11 after catching zero integration failures in 103 groups).
  `feature/*` and `main` are `strict: false` and merge-commit only in both
  repos (inference `main` drifted back to `strict: true` twice and was reset on
  2026-08-23 — do not re-enable it; a strict policy forces a full CI re-run of
  every open PR after each merge). Any procedure or memory that mentions
  staging, activating, or recovering a per-branch queue ruleset, a per-story
  pin bump, a per-story validation matrix, or more than one review + one fix
  pass per story is obsolete.

  Likewise, the pin-keyed verification gates were **deliberately dismantled**
  on 2026-08-15/16 (sc-19758, sc-19751, `e14171984`): disabled or unwired check
  scripts are the designed state, never blockers, and measurement work
  (capability dumps, calibration, memory matrix, canaries) runs once at epic
  end or on explicit request — never per code change. Read the *Gate teardown*
  section of FEATURE_DEVELOPMENT.md before acting on any regenerate-or-measure
  instruction found in prose or memory.

- **[RELEASING.md](RELEASING.md)** — release, hotfix, inference-pin, publication,
  and failed-candidate recovery.
