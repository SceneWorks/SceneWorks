# SC-19059 synthetic Candle promotion rehearsal

This receipt preserves the CPU-only rehearsal without preserving, publishing, or binding any
synthetic calibration data. It proves only the producer, lane replacement, source catalog,
packaging, generated-doc, and cleanup plumbing. It is not Candle measurement evidence and cannot
authorize a runtime curve.

## Frozen identity and initial state

- SceneWorks commit: `5a6b46762a238df26b97e27027306f5249f33f4a`
- SceneWorks tree: `6116e27850d543d77461b98a5bec4f7b638289f2`
- Inference pin: `4013049764172ee7dc707101c7da8c83c1483f2d`
- Initial committed curve SHA-256:
  `b0c51f9bc0adf29155983a9b429735c25082d9132191c64f8e1a585f9877e125`
- Initial Candle inventory SHA-256:
  `f8efe85b6172c843369e313c4e9e66c72f2770d762cc1d381d22c507720b03e2`
- Initial Rust decision SHA-256:
  `2aecbc27a99e467d78488df09441452434d72ddcf0cb432613fb611581125bc9`
- Initial Candle inventory Markdown SHA-256:
  `fdf20aad199f362b82526fa8e0c8e0a4945a749160637deb2aa0f8a72e70a929`

Before the rehearsal, the branch already carried the 11 real SC-19059 changes listed in
`07-pre-rehearsal-status.log`. Their binary patch SHA-256 was
`1e70b0d6eef738fb98a910f59202cacdd1334b97a79608c85ba53e11e30871a7`; the aggregate SHA-256 over
every tracked file was `48a8b4ff2de67c5a4c9b642087008d01c01b7769bfceccc395a7d079918d23e1`.

## Synthetic constructor and exact command sequence

The transient constructor was 2,659 bytes with SHA-256
`96240da5f383ff8d9cd5bfc96b393ba58d23329be64ba97dba452bbb2b032540`. It cloned the committed LTX
records onto every provider in `docs/calibration/sc-19057/wan-candle-video-capture-plan.json`, changed
the lane to `candle`, set the exact Wan provider/strategy/geometry from the plan, assigned a visibly
synthetic closure (`b4a29108bbbb...abcdef`), and generated deterministic affine phase bytes. It also
set `status = runtime_complete` and both repository dirty flags to false so the real producer's
record-terminal validation, not an invented driver log, was exercised. The script and everything it
wrote were deleted after the run.

With `EVIDENCE_DIR` naming an outside-repository temporary directory, the promotion and validation
commands were:

```sh
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
git status --porcelain=v2

git diff --binary > "$EVIDENCE_DIR/pre-rehearsal-actual.patch"
( git ls-files -z | xargs -0 shasum -a 256 ) | shasum -a 256
git status --porcelain=v2 > "$EVIDENCE_DIR/07-pre-rehearsal-status.log"

node scripts/sc19059-synthetic-candle-promotion.mjs
node scripts/fit-ltx-temporal-form.mjs \
  --story sc-19057 \
  --dataset docs/generated/wan-candle-video-sc-19057.json \
  --plan docs/calibration/sc-19057/wan-candle-video-capture-plan.json \
  --record-terminals \
  --write docs/generated/wan-temporal-form-fit-sc-19057.json \
  --source-fit docs/generated/wan-temporal-form-fit-sc-19057.json

npm run generate:memory-matrix
npm run generate:candle-admission
node --test scripts/fit-ltx-temporal-form.test.mjs \
  scripts/generate-candle-admission-inventory.test.mjs \
  scripts/platform-review-contracts.test.mjs
cargo test -p sceneworks-core video_memory_curves
cargo test -p sceneworks-worker --features backend-candle video_admission
npm run check

git diff --binary > "$EVIDENCE_DIR/rehearsal-full.patch"
git status --short > "$EVIDENCE_DIR/19-rehearsal-final-status-before-revert.log"
```

The first producer attempt deliberately omitted both terminal modes and failed with the expected
provenance error: a captured record had no driver-log `OK` terminal. Adding `--record-terminals`
cleared it only because every transient record passed the producer's `runtime_complete` and
clean-repository guards.

## Two-lane result and tripwires

The transient bundle contained exactly two curves and two source-catalog entries:

| Lane | Provider | Tier | Source fit | Closure |
| --- | --- | --- | --- | --- |
| MLX | `ltx_2_3` | q8 | `docs/generated/ltx-temporal-form-fit-sc-18810.json` | `87a27d5dcab7...` (real committed source) |
| Candle | `wan2_2_ti2v_5b` | q4 | `docs/generated/wan-temporal-form-fit-sc-19057.json` | `b4a29108bbbb...` (synthetic, deliberately non-live) |

Promoting the second lane exposed the three expected red tripwires. They were cleared without
weakening their assertions:

1. The Rust Docker embed test required both builder stages to copy the newly included evidence
   source; the rehearsal added both `COPY` lines.
2. The Candle inventory's MLX-only assertion required retiring the zero-Candle-curve known gap and
   deriving lane counts from the two-lane bundle.
3. The committed-artifact round trip required regenerating the inventory after its source closure
   changed.

After those clears, focused JavaScript was 155/155, `sceneworks-core` curve tests were 11/11,
`sceneworks-worker` Candle video-admission tests were 42/42, and the full `npm run check` suite was
734/734. The promoted artifact hashes are recorded in the adjacent checksum manifest. No test or
runtime route bound the synthetic curve: its deliberately non-live closure kept the bundle at
packaging/provenance evidence only.

## Revert and residue proof

Immediately before cleanup, the rehearsal-only delta comprised:

- `crates/sceneworks-core/src/video_memory_curves.rs` and both Docker builder stages;
- the synthetic Wan evidence file and fit report;
- the transient two-lane curve bundle and regenerated matrix/inventory/baseline; and
- `scripts/sc19059-synthetic-candle-promotion.mjs`.

Cleanup restored the worktree, reapplied only `pre-rehearsal-actual.patch`, and repeated the hashes:

```sh
# Delete exactly these three untracked rehearsal files with the recorded patch operation:
# scripts/sc19059-synthetic-candle-promotion.mjs
# docs/generated/wan-candle-video-sc-19057.json
# docs/generated/wan-temporal-form-fit-sc-19057.json
git restore --worktree -- .
git apply "$EVIDENCE_DIR/pre-rehearsal-actual.patch"
test ! -e scripts/sc19059-synthetic-candle-promotion.mjs
test ! -e docs/generated/wan-candle-video-sc-19057.json
test ! -e docs/generated/wan-temporal-form-fit-sc-19057.json
( git ls-files -z | xargs -0 shasum -a 256 ) | shasum -a 256
git status --porcelain=v2 > "$EVIDENCE_DIR/20-post-revert-status.log"
git diff --binary | shasum -a 256
shasum -a 256 "$EVIDENCE_DIR/pre-rehearsal-actual.patch"
git diff --check
```

The tracked aggregate returned to
`48a8b4ff2de67c5a4c9b642087008d01c01b7769bfceccc395a7d079918d23e1`; both the restored diff and
saved pre-rehearsal patch were
`1e70b0d6eef738fb98a910f59202cacdd1334b97a79608c85ba53e11e30871a7`. The pre/post status logs
are byte-identical (`a49646bdfd4d77a14deb74371e583185c053217269a9f2d7bfdab0c1745283b9`). The synthetic constructor,
Wan evidence, Wan fit, two-lane curve, and packaging edits are absent from the committed result.

The full transient patch SHA-256 was
`344689b58e927baa47caaa68c4a7e7db0b115b135df35a7761df0daedbdc368d`. The outside-repository
`SHA256SUMS` manifest verified every retained raw log/receipt and had SHA-256
`67f0f544afae0f4754c986810f1bdfda7a9651062688b2a396034bc30aa74c97`. Its sanitized,
repository-relative copy is
`docs/calibration/sc-19059/synthetic-candle-promotion-rehearsal.checksums.json`.
