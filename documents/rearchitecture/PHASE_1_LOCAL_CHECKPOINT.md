# Phase 1 Local History-Import Checkpoint

> **Status:** Local import complete; remote repository creation blocked by invalid
> GitHub CLI API authentication.
>
> **Local repository:** `/Users/zakkeown/Code/SceneWorks/inference`
>
> **Assembled HEAD:** `2555926f`

## Completed

- Installed `git-filter-repo` 2.47.0.
- Bootstrapped the provisional inference repository.
- Imported filtered default-branch histories for `core-llm`, `mlx-llm`,
  `candle-llm`, `mlx-gen`, and `candle-gen` under `imports/`.
- Preserved exact old-to-new commit maps and ref maps.
- Created annotated baseline tags for all five imports.
- Preserved the four-commit-newer `candle-gen` tracking history under
  `history/candle-gen-tracking-main` without merging it into baseline `main`.
- Verified all five imported baseline subtree IDs exactly match the recorded source
  tree IDs.
- Verified the preserved newer Candle subtree also exactly matches its source.
- Verified root `cargo metadata`, `git fsck`, and `git diff --check`.
- Published the Phase 0 SceneWorks branch
  `codex/multi-repo-rearchitecture-phase-0`.

## Local inference first-parent history

```text
2555926f docs: record history import equivalence
7a39123d docs: preserve filtered history commit maps
7c7979c2 chore: import candle-gen history
40953c15 chore: import mlx-gen history
1adf357a chore: import candle-llm history
a3afe0e7 chore: import mlx-llm history
10107594 chore: import core-llm history
d28ea14e chore: bootstrap unified inference repository
```

## Publication blocker

`gh auth status` reports the active `zakkeown` GitHub CLI token is invalid. Normal
Git HTTPS credentials remain valid, as proven by pushing the SceneWorks Phase 0
branch, but repository creation and PR creation use the GitHub API token.

Required operator action:

```sh
gh auth login -h github.com
```

After authentication, migration resumes with:

1. Create public `SceneWorks/inference` with no generated initial files.
2. Add it as the local inference repository's `origin`.
3. Push `main` and migration/history tags.
4. Verify remote commits and tree IDs.
5. Create the SceneWorks Phase 0/1 baseline PR.

