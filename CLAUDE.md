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
  assumption. As of 2026-08-11 the two repositories deliberately differ:
  **SceneWorks has no merge queue anywhere**, while **inference still queues
  `main` and every feature branch**. Procedures that mention staging, activating,
  or recovering a per-branch queue ruleset apply to inference only.

- **[RELEASING.md](RELEASING.md)** — release, hotfix, inference-pin, publication,
  and failed-candidate recovery.
