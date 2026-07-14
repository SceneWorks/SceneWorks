# Phase 1 Local History-Import Checkpoint

> **Status:** Complete and published. The temporary GitHub authentication blocker
> was resolved; canonical `main` and every migration/history ref are present in
> the private `SceneWorks/inference` repository.
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

## Publication outcome

GitHub CLI authentication was restored. The empty canonical repository was
created, configured as `origin`, and received `main`, all five
`migration-baseline/*` tags, and `history/candle-gen-tracking-main` without a
force push. Remote refs were verified after publication. Its actual private
visibility is preserved; the earlier public recommendation was not adopted.
