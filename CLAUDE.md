# SceneWorks

Agent instructions for this repository live in **[AGENTS.md](AGENTS.md)** — read it
first. This file exists so the `CLAUDE.md` convention resolves; AGENTS.md is the
source of truth, and anything beyond the pointers below belongs there rather than
here.

## Workflows

- **[FEATURE_DEVELOPMENT.md](FEATURE_DEVELOPMENT.md)** — epic integration
  branches, story PR targets, mirrored inference work, final merge ordering, and
  the current ruleset/CI state.

  Read its *Current state* section before acting on any branch protection
  assumption. As of 2026-08-11 **neither SceneWorks nor inference has a merge
  queue** — both were removed the same day after the queue was measured catching
  zero integration failures in 103 groups while adding ~21m per merge. `main` is
  `strict: false` in both; `feature/*` is `strict: true` in both. Any procedure
  or memory that mentions staging, activating, or recovering a per-branch queue
  ruleset is obsolete.

- **[RELEASING.md](RELEASING.md)** — release, hotfix, inference-pin, publication,
  and failed-candidate recovery.
