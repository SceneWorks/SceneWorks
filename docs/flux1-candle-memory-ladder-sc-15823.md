# FLUX.1 Candle/CUDA shared image memory ladder evidence (SC-15823)

## Certification boundary

SC-15823 certifies the five shared image memory rungs for the exact base-only coordinates below:

- `flux_schnell` / runtime provider `flux1_schnell`, q4, text-to-image, 1024×1024, batch 1, one frame
- `flux_dev` / runtime provider `flux1_dev`, q4, text-to-image, 1024×1024, batch 1, one frame
- Candle/CUDA at inference revision `5b6d6aa02f9d85503d26e66a0732fcb1b841fa5b`
- SceneWorks pre-evidence revision `002e19717c322ba0520ebff28c96c71d0dea21eb` and matrix source `source-tree:8006080e1f8ee1c5907cbae552797c40dffdd5eb7ba929cf52d7f25cf41609a2`

These records do **not** certify q8/bf16, another geometry or mode, LoRA, FLUX IP-Adapter identity, FLUX ControlNet, or PuLID. The manifest has no calibration binding for those coordinates, so they retain the fail-closed resident fallback. In particular, declared identity/control runtime capabilities are not evidence that an overlay was executed.

The records use `runtime_complete`, not Full `complete`. This status makes the exact base coordinate eligible for production selection while preserving the actual test boundary: exact-fit, unknown-budget, stale-evidence, loadability, memory, and output parity passed; same-process warm repeat, cancellation cleanup, fault cleanup, and a measured negative mutation were not part of these physical sessions and remain explicitly `not_run`/`null`. No sibling Full calibration is inferred.

## Hardware and method

All ten accepted runs were fresh processes on device 0 with `CUDA_VISIBLE_DEVICES=0`:

| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Dedicated memory | 102,641,958,912 bytes (97,887 MiB) |
| Compute capability | 12.0 |
| Driver | 596.36 |
| CUDA | toolkit/runtime 12.9 (`nvcc V12.9.41`) |
| OS | Microsoft Windows NT 10.0.26200.0 |
| Fixture | fixed prompt and seed 42; 1024×1024 RGB8, batch 1, one frame |
| Steps | schnell 4; dev 8 |

The inference harness emitted `MEMORY_EVIDENCE_V1` after a real render. Its request live-allocation high-water is the authoritative byte value. Because that record is an aggregate request high-water rather than phase telemetry, the evidence schema's conditioning/denoise/decode phase slots conservatively repeat that aggregate; they must not be read as independently measured phase peaks. The harness also printed a rounded driver overall peak, retained only as a diagnostic cross-check.

The same artifact snapshot, prompt, seed, geometry, step count, precision, and conditioning were used for resident comparison. Resident and staged outputs were exact. Each model's three bounded-rung outputs were mutually identical and were compared to its resident RGB8 output under the recorded tolerance contract. The temporary harness `exact/not_run` label was not imported as the quality verdict; the measured comparison statistics below are authoritative.

The first schnell bounded-transformer attempt failed closed before rendering because the sandbox denied both model-adjacent and `LOCALAPPDATA` device-format cache writes. It was rejected from evidence. The accepted bounded-transformer runs set `SCENEWORKS_CANDLE_DEVICE_CACHE_DIR` to a workspace-local temporary sidecar cache. The content-addressed cache held 990 files / 14,793,250,384 bytes after both models. The committed records and bindings contain no dependency on that local path.

## Artifact identity

| Model | Repository | Revision | Variant | Inventory SHA-256 |
|---|---|---|---|---|
| schnell | `SceneWorks/flux1-schnell-mlx` | `bba3ae01dfd94089f173c05edd4e1a4c551f2599` | q4 | `3157f5cdd80246daf0dd5f7c07694e8e8ee2845ec01bec8b9edb8a02b4bd8f62` |
| dev | `SceneWorks/flux1-dev-mlx` | `323fd12d79f78ad444e882e8d8e871914584f2b9` | q4 | `38892dfdd0177068e4996834a3ef6666309db5a0034f2548bd13f814e742b341` |

## Memory results

The driver column reproduces the harness's rounded display and is not substituted for the exact live-allocation value.

| Model | Rung | Load shape | Authoritative peak bytes | Driver diagnostic |
|---|---|---|---:|---:|
| schnell | resident | eager | 17,642,573,594 | 27.1 GB |
| schnell | staged residency | eager | 14,588,695,430 | 20.8 GB |
| schnell | bounded decode | eager | 9,608,827,400 | 10.4 GB |
| schnell | bounded attention | eager | 8,234,088,968 | 9.1 GB |
| schnell | bounded transformer residency | deferred | 3,843,456,916 | 4.5 GB |
| dev | resident | eager | 17,651,074,970 | 22.4 GB |
| dev | staged residency | eager | 14,597,196,806 | 19.1 GB |
| dev | bounded decode | eager | 9,858,108,040 | 11.7 GB |
| dev | bounded attention | eager | 8,272,605,832 | 9.1 GB |
| dev | bounded transformer residency | deferred | 3,843,457,940 | 4.5 GB |

## Output parity

Metrics are raw RGB8 channel-space differences versus the corresponding resident output. Thresholds are pinned to the accepted measured envelope, so any later increase fails closed.

| Model | Rungs | Maximum RGB8 | Mean RGB8 | RMSE | Contract |
|---|---|---:|---:|---:|---|
| schnell | resident, staged | 0 | 0 | 0 | exact |
| schnell | bounded decode, attention, transformer | 29 | 0.2289120356241862 | 0.5452312653086621 | tolerance |
| dev | resident, staged | 0 | 0 | 0 | exact |
| dev | bounded decode, attention, transformer | 21 | 0.26471201578776044 | 0.5180919471938068 | tolerance |

Within schnell, all three bounded rungs produced SHA-256 `b84dd88f02854e15df837216d6f67a37e2d538f9c221ca5c1a14a741bcffcfef`; resident/staged produced `fe0e5ff1a022da5cefe4be7dfeb2bac51bc3870bbfd9bb36949ab8381ba3d9e0`. Within dev, the bounded digest was `3727e3e9c323be1be2b25ce4237c5280876832381c5e8bc64e1e36b2937348d4`, and resident/staged was `009e11f9bbcaca6edebe3658e221589bd7404d7a4c9db8ee251c0eae57801964`.

The authoritative machine-readable records are in `docs/generated/memory-calibration-evidence.json`; exact manifest bindings are in `config/manifests/builtin.models.jsonc`.
