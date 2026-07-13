# Inference Binary Fixture Inventory and Policy

> **Status:** Phase 0 recommendation for MRP-004. No fixture has been moved.
>
> **Threshold:** Tracked files at least 500,000 bytes in the five inference source
> repositories at the SHAs in `baseline/release-set.toml`.

## Inventory summary

| Repository | Files ≥500 KB | Current-tree bytes | Git pack size |
|---|---:|---:|---:|
| `core-llm` | 0 | 0 | 210.77 KiB |
| `mlx-llm` | 3 | 3,866,923 | 2.70 MiB |
| `candle-llm` | 3 | 3,866,923 | 1.36 MiB |
| `mlx-gen` | 66 | 378,755,084 | 172.20 MiB |
| `candle-gen` | 12 | 58,115,092 | 28.51 MiB |
| **Total** | **84** | **444,604,022** | **about 205 MiB** |

The current-tree total is not the incremental size of the proposed repository;
Git pack compression and shared objects reduce it. It does identify future growth
and checkout pressure.

## Largest tracked artifacts

| Bytes | Repository path | Classification |
|---:|---|---|
| 52,964,380 | `mlx-gen/mlx-gen-sensenova/tests/fixtures/it2i_golden.safetensors` | Large test golden |
| 52,832,528 | `mlx-gen/mlx-gen-sensenova/tests/fixtures/t2i_golden.safetensors` | Large test golden |
| 52,718,748 | `mlx-gen/mlx-gen-sensenova/tests/fixtures/interleave_golden.safetensors` | Large test golden |
| 52,718,544 | `mlx-gen/mlx-gen-sensenova/tests/fixtures/vqa_golden.safetensors` | Large test golden |
| 12,714,152 | `mlx-gen/mlx-gen-sana/tests/fixtures/dcae_decode_golden.safetensors` | Large test golden |
| 12,584,438 | `mlx-gen/mlx-gen-ltx/tests/fixtures/ltx_connector_golden.safetensors` | Large test golden |
| 11,422,654 | `mlx-gen/mlx-gen-anima/assets/qwen_tokenizer.json` | Runtime/test asset, duplicated in Candle |
| 11,422,654 | `candle-gen/candle-gen-anima/assets/qwen_tokenizer.json` | Runtime/test asset, duplicated in MLX |
| 10,308,176 each | Three `candle-gen-sd3/tests/parity/reference/*.safetensors` files | Large test goldens |

The remaining files are mostly sub-10-MiB safetensors parity fixtures, tokenizer
assets, negative embeddings, and LLM JSON oracle data.

## Exact duplicate groups

There are **11** exact duplicate groups among files at least 500 KB, representing
**33,481,987 reclaimable current-tree bytes** if each group is stored once:

| Copies | Bytes each | Content | Current locations |
|---:|---:|---|---|
| 2 | 11,422,654 | Qwen tokenizer | MLX/Candle Anima assets |
| 4 | 2,424,500 | T5 tokenizer | MLX/Candle Anima and Chroma assets |
| 2 | 3,642,170 | CLIP tokenizer | MLX/Candle Flux assets |
| 2 | 3,459,344 | Krea DiT golden | MLX/Candle Krea fixtures |
| 2 | 1,227,752 | Krea text-encoder golden | MLX/Candle Krea fixtures |
| 2 | 1,127,860 | Krea text-fusion golden | MLX/Candle Krea fixtures |
| 2 | 867,776 | Krea single-block golden | MLX/Candle Krea fixtures |
| 2 | 594,008 | SeedVR2 negative embedding | MLX/Candle SeedVR2 data |
| 2 | 1,577,574 | Qwen3-VL vision oracle | MLX/Candle LLM testdata |
| 2 | 1,407,521 | Qwen3.5 preprocess oracle | MLX/Candle LLM testdata |
| 2 | 881,828 | Qwen3.5 vision oracle | MLX/Candle LLM testdata |

Hashes were calculated with SHA-256. Exact hash details remain reproducible from
the recorded source trees and should be emitted into the destination fixture
manifest when deduplication occurs.

## Recommended policy

### History import

Preserve the recorded source trees byte-for-byte during the first filtered-history
import. Do **not** combine Git LFS conversion, artifact removal, or deduplication
with source-history import. This keeps tree-equivalence validation simple and makes
any migration problem attributable to path/history handling rather than fixture
storage changes.

The resulting initial inference Git pack is expected to be roughly the combined
205-MiB source pack plus import/merge overhead. That is acceptable as the cost of a
safe first import.

### Destination tree after equivalence is proven

1. **Files below 5 MiB:** keep directly in Git unless they are generated or
   license-restricted.
2. **Runtime-required tokenizer/config assets:** keep available to Cargo/package
   builds without network access. Deduplicate them into a family-level shared asset
   crate/directory; do not place required runtime assets behind optional CI fetches.
3. **Test-only files from 5–10 MiB:** keep in Git initially, then evaluate LFS only
   if repository growth remains material.
4. **Test-only files above 10 MiB:** move to immutable checksummed release assets or
   an artifact bucket after the family has a reliable fetch/cache helper.
5. **The four ~53-MiB SenseNova goldens:** highest-priority externalization target.
6. **Exact MLX/Candle duplicates:** store once under the model-first family fixture
   directory after Phase 6 relocation.
7. **New large fixtures:** reject direct Git additions above 10 MiB unless an ADR
   documents why offline availability outweighs clone cost.

### External fixture manifest requirements

Every externally stored fixture records:

- Logical fixture ID and owning tests.
- Immutable URL/release asset.
- SHA-256 and byte size.
- Generator/reference repository and revision.
- License and redistribution status.
- Cache path/key.
- Clear offline/absent-fixture diagnostic.
- Whether the test is required on PRs, nightly, or release candidates.

### Why artifact storage is preferred over broad Git LFS adoption

- The largest candidates are test-only goldens, not runtime package inputs.
- Release/nightly GPU lanes already have specialized caches and prerequisites.
- Artifact manifests give explicit revisions, licensing, and checksums.
- Required runtime tokenizers must remain usable in offline Cargo/package builds.
- LFS can remain an option for moderate redistributable fixtures, but adopting it
  for every binary would make ordinary source checkout depend on a second storage
  service without eliminating the need for fixture metadata.

## Phase 0 decision requested

- [ ] Approve byte-preserving first import.
- [ ] Approve direct Git for files below 5 MiB by default.
- [ ] Approve artifact storage for test-only files above 10 MiB.
- [ ] Approve family-level deduplication only after import equivalence passes.

