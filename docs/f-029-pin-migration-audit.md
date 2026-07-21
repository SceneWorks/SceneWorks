# F-029 pin-migration audit — the manifest `revision` is the download pin authority

**Story:** sc-13685 (epic 13678). **Gates:** inference sc-13665 (pin-constant deletion).

## What F-029 means now

Under epic 13657 / sc-13591, inference no longer self-fetches any model component: every
primary weight **and** every co-requisite is provisioned by SceneWorks (env-correct
`downloads.rs`) and passed into inference as an already-resolved local path. The immutable
commit that used to live in an inference `HUB_REVISION`/`PINNED_HUB_REVISIONS` constant now
lives in the SceneWorks manifest as the download entry's `revision` (a full 40-hex SHA,
`^[0-9a-f]{40}$`, enforced by `model-manifest.schema.json` + the Python/Rust manifest audits).

**The manifest `revision` is the F-029 supply-chain pin authority.** Inference keeps no pin.
This is the accounting gate that makes inference sc-13665 (deletion of the dead
`HUB_REPO`/`HUB_REVISION` pin constants) safe: a pin may not be deleted from inference until
this table shows it has a pinned manifest home.

## Audit method

- Inference clone swept for every real repo-pin constant across the rehomed audio (11) +
  mmaudio (3 repos) + SDXL (3) crates: `grep -rhoE '"[0-9a-f]{40}"'` over
  `crates/audio/*/src/*.rs` + `crates/media/candle-gen/candle-gen-sdxl/src/*.rs` yields
  **18 distinct real repo pins** (a 19th 40-hex literal, `91b3b1eb…`, is only a `#[cfg(test)]`
  fixture in `candle-audio/src/hub.rs` and is deleted with `hub.rs` itself — not a pin).
- Each manifest `revision` compared byte-for-byte against the inference constant.
- The 5 previously-unpinned primaries pinned in this story were additionally confirmed as real
  commits via the HF API (`GET /api/models/<repo>/revision/<sha>` → HTTP 200).

## Cross-check table (every inference pin → its manifest home)

| # | Inference pin constant (crate / const) | Repo | Inference SHA | SceneWorks manifest entry — field | Manifest `revision` | Match |
|---|----------------------------------------|------|---------------|-----------------------------------|---------------------|-------|
| 1 | candle-audio-chatterbox · `HUB_REVISION` | ResembleAI/chatterbox | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | `chatterbox_tts` — primary download (**pinned here, sc-13685**) | `5bb1f6ee…6e18` | ✅ |
| 2 | candle-audio-chatterbox · `PERTH_HUB_REVISION` | SceneWorks/perth-implicit | `80b60f9caead09b8d3b512bda0b24038f28c08ec` | `chatterbox_tts` — `perth` coRequisite | `80b60f9c…08ec` | ✅ |
| 3 | candle-audio-chatterbox-ve · `HUB_REVISION` | ResembleAI/chatterbox | `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | `chatterbox_tts` `voice_embedding` coReq + `chatterbox_ve` primary | `5bb1f6ee…6e18` | ✅ |
| 4 | candle-audio-openvoice · `HUB_REVISION` | myshell-ai/OpenVoiceV2 | `f36e7edfe1684461a8343844af60babc2efbb727` | `openvoice_v2` — primary download (**pinned here, sc-13685**) | `f36e7edf…b727` | ✅ |
| 5 | candle-audio-moss-tts · `HUB_REVISION` | OpenMOSS-Team/MOSS-TTSD-v0.5 | `8527b9136b6afefe2252ae597cecea2e80e7ebeb` | `moss_ttsd_v05` — primary download | `8527b913…7ebeb`… | ✅ |
| 6 | candle-audio-moss-tts · `CODEC_HUB_REVISION` | OpenMOSS-Team/XY_Tokenizer_TTSD_V0 | `c83433728e698ed0698e88cb5096bc221fb8f8c5` | `moss_ttsd_v05` — `codec` coRequisite | `c8343372…f8c5` | ✅ |
| 7 | candle-audio-moss-tts-realtime · `HUB_REVISION` | OpenMOSS-Team/MOSS-TTS-Realtime | `6acbc7f161a0db71c291f2d0aaa9eee59334cab2` | `moss_tts_realtime` — primary download | `6acbc7f1…4cab2`… | ✅ |
| 8 | candle-audio-moss-tts-realtime · `CODEC_HUB_REVISION` | OpenMOSS-Team/MOSS-Audio-Tokenizer | `3cd226ba2947efa357ef453bcad111b6eafba782` | `moss_tts_realtime` — `codec` coRequisite | `3cd226ba…b782` | ✅ |
| 9 | candle-audio-moss-sfx · `HUB_REVISION` | OpenMOSS-Team/MOSS-SoundEffect-v2.0 | `e35df4d82fbe87fcd5d14e5d100e349c0c3c076d` | `moss_sfx_v2` — primary download (**pinned here, sc-13685**) | `e35df4d8…076d` | ✅ |
| 10 | candle-audio-kokoro · `HUB_REVISION` | hexgrad/Kokoro-82M | `f3ff3571791e39611d31c381e3a41a3af07b4987` | `kokoro_82m` — primary download (**pinned here, sc-13685**) | `f3ff3571…4987` | ✅ |
| 11 | candle-audio-acestep · `HUB_REVISION` | ACE-Step/acestep-v15-xl-turbo-diffusers | `200ba991ae448051e14b0183157e35c2d27c9fb0` | `acestep_v15_turbo` — primary download (**pinned here, sc-13685**) | `200ba991…9fb0` | ✅ |
| 12 | candle-audio-whisper · `HUB_REVISION` | openai/whisper-base | `e37978b90ca9030d5170a5c07aadb050351a65bb` | `whisper_base` — primary download | `e37978b9…65bb` | ✅ |
| 13 | candle-audio-clap · `HUB_REVISION` | laion/clap-htsat-unfused | `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` | `clap_htsat_unfused` — primary download | `8fa0f1c6…da8a` | ✅ |
| 14 | candle-audio-mmaudio · `HUB_REVISION` (model/mmdit/output) | hkchengrex/MMAudio | `eb13a1a98fdbec91753775c57b074ccdfc60587c` | `mmaudio_small_16k` + `mmaudio_large_44k` — primary + `synchformer`/`dit`/`vae`/`vocoder` coReqs | `eb13a1a9…587c` | ✅ |
| 15 | candle-audio-mmaudio · `BIGVGAN_V2_HUB_REVISION` | nvidia/bigvgan_v2_44khz_128band_512x | `95a9d1dcb12906c03edd938d77b9333d6ded7dfb` | `mmaudio_large_44k` — `vocoder` coRequisite | `95a9d1dc…7dfb` | ✅ |
| 16 | candle-audio-mmaudio · `CLIP_HUB_REVISION` | apple/DFN5B-CLIP-ViT-H-14-384 | `01b771ed0d1395ca5ffdd279897d665ebe00dfd2` | `mmaudio_small_16k` + `mmaudio_large_44k` — `clip` coRequisite | `01b771ed…dfd2` | ✅ |
| 17 | candle-gen-sdxl · `PINNED_HUB_REVISIONS[VAE_FIX_REPO]` | madebyollin/sdxl-vae-fp16-fix | `207b116dae70ace3637169f1ddd2434b91b3a8cd` | `sdxl`, `realvisxl`, `realvisxl_lightning`, `illustrious_xl_v1`, `illustrious_xl_v2`, `instantid_realvisxl` — `vae_fp16_fix` coReq | `207b116d…a8cd` | ✅ |
| 18 | candle-gen-sdxl · `PINNED_HUB_REVISIONS` (CLIP-L) | openai/clip-vit-large-patch14 | `32bd64288804d66eefd0ccbe215aa642df71cc41` | SDXL family — `tokenizer_clip_l` coReq + `clip_vit_l14` utility primary | `32bd6428…cc41` | ✅ |
| 19 | candle-gen-sdxl · `PINNED_HUB_REVISIONS` (bigG) | laion/CLIP-ViT-bigG-14-laion2B-39B-b160k | `743c27bd53dfe508a0ade0f50698f99b39d03bec` | SDXL family — `tokenizer_clip_bigg` coRequisite | `743c27bd…03bec`… | ✅ |

(Rows 1–3 share the single repo `ResembleAI/chatterbox`; rows 14/17/18 fan out across multiple
manifest models. 18 distinct repo pins across 19 constant sites — every one now carries a pinned
manifest `revision`. **Zero unaccounted pins.**)

## Cache-read class — NOT pins (no migration owed)

These sc-13591 inventory items were **HF-cache-scan fallbacks** (`~/.cache/huggingface/hub`
repo-name scans), never pinned-SHA constants. They were removed by inference **sc-13664**
(a separate, already-merged story), not sc-13665, so they carry no SHA to migrate and do not
gate the pin-constant deletion:

| sc-13591 item | Inference repo(s) (scan target) | Had a SHA pin? | Disposition |
|---------------|----------------------------------|----------------|-------------|
| sensenova distill LoRA | `sensenova/SenseNova-U1-8B-MoT-LoRAs` | No (repo scan) | cache-scan removed by sc-13664; SceneWorks provisions the fast-tier LoRA via its own download machinery |
| LTX gemma text-encoder | `mlx-community/gemma-3-12b-it-bf16`, `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit` | No (repo scan) | cache-scan removed by sc-13664; LTX bundle provisions gemma as a manifest coRequisite (sc-13683, `01df27d3…`) |
| PuLID adapter | `guozinan/PuLID` | No (repo scan) | cache-scan removed by sc-13664; PuLID identity weights provisioned by sc-13683 |

## Grandfathered co-requisite migration allowlist — unaffected by this story

The `COREQUISITE_REVISION_MIGRATION_PENDING` allowlist (Rust
`crates/sceneworks-core/src/builtin_manifests.rs` + Python
`tests/test_builtin_manifest_audit.py`, kept in lockstep) still holds 8 entries:
`qwen_image` ControlNet, `ltx_2_3_eros`, `wan_2_2_t2v_14b`/`wan_2_2_i2v_14b` Lightning, and
`pid_qwenimage`/`pid_flux`/`pid_flux2`/`pid_sdxl` gemma-2-2b-it. **None of these repos is an
inference self-fetch pin** — they are grandfathered under their own per-family migration
stories, their SHAs are not in the sc-13591 inference inventory, and inference sc-13665 does not
delete any pin for them. The 5 primaries pinned by this story are **not** co-requisites
(`coRequisite` absent), so the allowlist required no change and its self-cleaning audit stays green.

## sc-13665 gate status

**UNBLOCKED.** Every inference audio/mmaudio/SDXL repo-pin constant (18 distinct pins, 19 sites)
maps to a pinned manifest `revision` carrying the identical SHA. sc-13665 may delete the
`HUB_REPO`/`HUB_REVISION`/`PINNED_HUB_REVISIONS` pin constants; the F-029 supply-chain pin now
lives in the SceneWorks manifest. (The inference-side `docs/migration/` note pointing at the
manifest as the new pin authority is authored by sc-13665's landing, not this PR.)
