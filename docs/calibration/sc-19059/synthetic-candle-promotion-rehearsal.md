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

Promoting the second lane exposed the three expected red tripwires. The first rehearsal retained
the final clear but not the individual red transcripts; the bounded falsification replay below
retains a distinct command, exit status, and output hash for every red and green gate. The transient
clear did not weaken their assertions:

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

## Packaging-tripwire falsification replay

This CPU-only replay started from PR repair head
`3427e5020a3b08f68c8dc3b83cb7ba1ed7718c29`, tree
`7994dbd913819121917ae1d0c5cb2f3951712691`, with empty status and patch. The aggregate SHA-256
over all tracked file hashes was
`a34fc068cb6398a61f23984eb011c2c4081b29db82546ad88d3e84c0650e6eff`; the empty binary patch
hashed to `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The same deterministic constructor and real fit producer created the transient Wan Candle source
and two-lane bundle. Before making the three required promotion edits, these tests were run and
captured separately:

```sh
node --test --test-name-pattern \
  "Rust Docker builders copy every production generated embed from sceneworks-core" \
  scripts/platform-review-contracts.test.mjs

node --test --test-name-pattern \
  "the packaged video curves are MLX-only" \
  scripts/generate-candle-admission-inventory.test.mjs

node --test --test-name-pattern \
  "the committed artifacts match a fresh build" \
  scripts/generate-candle-admission-inventory.test.mjs
```

All three red commands exited 1 for the intended reason:

| Gate | Sanitized failure | Red output SHA-256 |
| --- | --- | --- |
| Docker embed | Wan Candle source was present in 0 builder contexts; expected 2 | `459a810545a89936e1c12d62cd978aac13e992e0f567bf643c5cf374ff613293` |
| Recorded gap | Candle lane count was 1; the recorded zero-curve gap expected 0 | `1f2ce7641f88e819d3ccb52ad5e51effeef3663b30c1e6e08d7be978867cc228` |
| Artifact freshness | Fresh source/tree hashes and generated content differed from the committed inventory | `887057205585c8e1b06f003c397261b182eef67be831ca595cb8b0f3b0bd34b2` |

The transient clear added the Wan source `COPY` to both Rust builder contexts, retired the
zero-Candle-curve known gap while preserving a one-MLX/one-Candle lane assertion and zero runtime
route bindings, and regenerated the Candle-admission artifacts. Re-running the Docker embed,
replacement known-gap, and committed-artifact tests separately produced one pass and zero failures
each, all with exit 0. Their output SHA-256 values were respectively
`bd124e465267d381833aaf0f25417865b980f9e64379ecdeae0d10a9cc19af8e`,
`7935eeeb2d3d78bc35c1b85b9d42d1db52e29f2e1759d15ef78e8312630e9607`, and
`65f4cfe72e1c5a0107f2121499ba82e582772dcb434596369b2e730ba2939689`.

After capture, only the three named untracked rehearsal files were deleted and every tracked
mutation was restored. The post-replay head, tree, status, binary patch, and tracked aggregate are
byte-identical to the pre-replay values above; all three rehearsal files are absent. The external
raw-log `SHA256SUMS` manifest (repository-relative basenames only) is
`eb721bde48c29e8f6ef8dbff1302f0a6a65a1a546ac6af18af53482fb008d37f`. The adjacent checksum
JSON records the individual command, exit, output, transient-artifact, and cleanup-proof hashes.
