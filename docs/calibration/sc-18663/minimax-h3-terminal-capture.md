# SC-18663 MiniMax-H3 terminal capture

Run this against the permanent inference pin `28f0563baa03640ade1635356d2d54fe8a477f1a` and require
the loaded generator to publish the exact memory-strategy contract for every captured anchor. The
harness must fail closed only when that publication is actually missing or mismatched; never
manufacture a receipt.

sc-22514 collapsed `config/memory-calibration-plan.json` to an ANCHOR plan, so this is **three
independent captures of three anchor keys** — `minimax_h3:{q4,q8,bf16}:mlx`, each 576×320 × 124
frames, `eager_materialization`, fingerprint `minimax-h3-mlx-staged-joint-av-eager-abi3-v1`. There is
no plan slicing step (nothing to filter — `--anchor` names the one key), no `--resume` (each command
writes one record to its own file and either succeeds or is re-run whole) and no 72-row selection
(the rung/geometry/frame grid it enumerated is now derived analytically from the anchor by
`crates/sceneworks-core/src/memory_anchor.rs`).

Every mutable artifact is written outside both checkouts.

```bash
SCENEWORKS_REPO=/absolute/path/to/SceneWorks
INFERENCE_REPO=/absolute/path/to/inference-at-the-final-pin
CAPTURE_ROOT=/Volumes/Data/sc-17137-minimax-h3-terminal

mkdir -p "$CAPTURE_ROOT"

for TIER in q4 q8 bf16; do
  node "$SCENEWORKS_REPO/scripts/memory-calibration-harness.mjs" capture \
    --anchor "minimax_h3:$TIER:mlx" \
    --provider-command "[\"$SCENEWORKS_REPO/target/release/memory-mlx-adapter\"]" \
    --sceneworks-repo "$SCENEWORKS_REPO" \
    --inference-repo "$INFERENCE_REPO" \
    --raw-log-dir "$CAPTURE_ROOT/raw/$TIER" \
    --source-path-prefix docs/calibration/sc-18663 \
    --output "$CAPTURE_ROOT/minimax-h3-$TIER-evidence.json"
done
```

`--plan` is omitted deliberately: it defaults to `config/memory-calibration-plan.json`, which is
where the three `minimax_h3` anchors are declared. `--raw-log-dir` and `--source-path-prefix` are a
pair — supplying one without the other is refused.

Validate each file with
`node "$SCENEWORKS_REPO/scripts/memory-calibration-harness.mjs" check --input <file>`.

Do not copy any `$CAPTURE_ROOT/minimax-h3-*-evidence.json` into the repository by hand. The terminal
coordinator owns receipt review, temporal fitting, generated-artifact refresh, and the resulting
out-of-matrix `memoryCharacterization` update.
