# SC-18663 MiniMax-H3 terminal capture

Run this against the permanent inference pin `28f0563baa03640ade1635356d2d54fe8a477f1a` and require
the loaded generator to publish the exact memory-strategy contract for every selected case. The
harness must fail closed only when that publication is actually missing or mismatched; never
manufacture a receipt. It selects the 72 checked-in MiniMax-H3 rows (three tiers, four implemented rungs, two
areas, three legal `17n+5` frame counts) and writes every mutable artifact outside either checkout.

```bash
SCENEWORKS_REPO=/absolute/path/to/SceneWorks
INFERENCE_REPO=/absolute/path/to/inference-at-the-final-pin
CAPTURE_ROOT=/Volumes/Data/sc-17137-minimax-h3-terminal

mkdir -p "$CAPTURE_ROOT"
jq '{providers: [.providers[] | select(.target.modelId == "minimax_h3")]}' \
  "$SCENEWORKS_REPO/config/memory-calibration-plan.json" \
  > "$CAPTURE_ROOT/minimax-h3-calibration-plan.json"

node "$SCENEWORKS_REPO/scripts/memory-calibration-harness.mjs" run \
  --config "$CAPTURE_ROOT/minimax-h3-calibration-plan.json" \
  --backend mlx \
  --fresh-per-case \
  --provider-command "[\"$SCENEWORKS_REPO/target/release/memory-mlx-adapter\"]" \
  --sceneworks-repo "$SCENEWORKS_REPO" \
  --inference-repo "$INFERENCE_REPO" \
  --resume "$CAPTURE_ROOT/minimax-h3-evidence.json" \
  --output "$CAPTURE_ROOT/minimax-h3-evidence.json" \
  --raw-log-dir "$CAPTURE_ROOT/raw" \
  --source-path-prefix docs/calibration/sc-18663
```

Do not copy `$CAPTURE_ROOT/minimax-h3-evidence.json` into the repository by hand. The terminal
coordinator owns receipt review, temporal fitting, generated-artifact refresh, and the resulting
out-of-matrix `memoryCharacterization` update.
