//! Per-family tier-completeness predicates for the MLX turnkeys that ship NO `model_index.json`
//! (Anima / Boogu / SANA / Wan / SenseNova-U1 / MiniMax-H3).
//!
//! These families lay their weights out in a bespoke tree rather than a diffusers `model_index.json`
//! pipeline, so the generic `<tier>/*` presence checks — the worker's `tier_components_present`
//! (reads the tier's own `model_index.json`) and rust-api's coarse glob / `tier_subdir_has_weights`
//! — are a no-op for them: a TORN tier (backbone present, text-encoder / VAE / tokenizer missing)
//! passes the coarse check yet fails to load. Both the worker's tier resolvers (which pick a loadable
//! tier) AND rust-api's catalog completeness (which reports `installed` vs `incomplete`) need the SAME
//! concrete per-family predicate, so it lives here in the shared crate — a single source of truth that
//! the two consumers cannot drift apart (sc-13513). Hidden AppleDouble `._*` sidecars never count
//! (SceneWorks#1333).

use std::path::Path;

/// The component names a tier's own `model_index.json` declares, or `None` when it ships none.
///
/// A diffusers `model_index.json` maps a component name to a `[library, class]` pair; every other
/// value is configuration. Keys starting with `_` are metadata (`_class_name`, `_diffusers_version`,
/// and sc-14980's `_sceneworks_tier` / `_sceneworks_shared_components`) and are never components.
///
/// Shared (sc-13513 / sc-14980) so the worker's tier resolver, rust-api's catalog completeness, and
/// the training-base gate all read a tier's declared component set the same way — a split tier that
/// ships only the components it owns must look identical to all three.
pub fn tier_declared_components(dir: &Path) -> Option<Vec<String>> {
    let raw = std::fs::read_to_string(dir.join("model_index.json")).ok()?;
    let index: serde_json::Value = serde_json::from_str(&raw).ok()?;
    Some(
        index
            .as_object()?
            .iter()
            .filter(|(key, value)| !key.starts_with('_') && is_component_entry(value))
            .map(|(key, _)| key.clone())
            .collect(),
    )
}

/// Whether a `model_index.json` value names a COMPONENT — the diffusers `[library, class]` pair.
///
/// Three things in a real index are NOT components, and all three must be rejected here (each is
/// live in a shipping turnkey, so a laxer test fails a good tier):
/// - `[null, null]` marks an ABSENT optional component — `SceneWorks/realvisxl-mlx` declares
///   `feature_extractor` and `image_encoder` this way and ships neither dir.
/// - config arrays — `SceneWorks/krea-2-raw-mlx` declares `text_encoder_select_layers: [2, 5, 8, …]`.
/// - config scalars — krea's `patch_size: 2`, realvisxl's `force_zeros_for_empty_prompt: true`.
fn is_component_entry(value: &serde_json::Value) -> bool {
    value
        .as_array()
        .is_some_and(|pair| pair.len() == 2 && pair.iter().all(serde_json::Value::is_string))
}

/// Whether `dir` holds at least one non-hidden file whose name ends with `suffix` (e.g. `.safetensors`
/// / `.index.json`). The building block for the per-family completeness checks: a hidden AppleDouble
/// `._*` sidecar never counts (SceneWorks#1333).
pub fn dir_has_visible_file_ending(dir: &Path, suffix: &str) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.flatten().any(|entry| {
            !crate::lora_family::is_hidden_file(&entry.path())
                && entry.file_name().to_string_lossy().ends_with(suffix)
        })
    })
}

/// Whether an Anima tier `dir` is COMPLETE and loadable (issue #850 class, generalized). Anima ships no
/// `model_index.json`, so the shared presence guards are a no-op for it and a torn tier (DiT present,
/// text-encoder/VAE absent) reached the loader, which then died with a raw "No such file or directory"
/// (mlx-gen-anima `load_text_phase` / `load_vae`). A tier is complete when all three concrete inputs are
/// present: the DiT (any visible `.safetensors` in `diffusion_models/`), the dense text encoder, and the
/// VAE. Tokenizers are vendored into the binary, so there is no tokenizer file to check here.
pub fn anima_tier_complete(dir: &Path) -> bool {
    dir_has_visible_file_ending(&dir.join("diffusion_models"), ".safetensors")
        && dir
            .join("text_encoders/qwen_3_06b_base.safetensors")
            .is_file()
        && dir.join("vae/qwen_image_vae.safetensors").is_file()
}

/// Whether a Boogu tier `dir` is COMPLETE and loadable. Boogu ships no `model_index.json`, so the shared
/// guard is a no-op; the loader crashes FIRST on a missing `mllm/tokenizer.json`, then on an absent
/// `mllm`/`transformer`/`vae` weight. A tier is complete when the transformer (packed single-file OR
/// sharded index) plus its `config.json`, the Qwen3-VL `mllm/` (weights + `tokenizer.json`), and the VAE
/// weights are all present.
pub fn boogu_tier_complete(dir: &Path) -> bool {
    let transformer = dir.join("transformer");
    let transformer_weights = transformer
        .join("diffusion_pytorch_model.safetensors")
        .is_file()
        || transformer
            .join("diffusion_pytorch_model.safetensors.index.json")
            .is_file();
    transformer_weights
        && transformer.join("config.json").is_file()
        && dir.join("mllm/tokenizer.json").is_file()
        && dir_has_visible_file_ending(&dir.join("mllm"), ".safetensors")
        && dir_has_visible_file_ending(&dir.join("vae"), ".safetensors")
}

/// Whether a SANA tier `dir` is COMPLETE and loadable. The `SceneWorks/Sana_*_mlx` turnkeys ship no
/// `model_index.json`, so the standard-tier guard is a no-op for SANA specifically (flux/qwen/etc. DO
/// ship one and stay protected by the `model_index.json` check); the SANA loader dies with a raw OS
/// error on an absent Gemma text encoder or its tokenizer (mlx-gen-sana `SanaTextEncoder::from_snapshot`).
/// A tier is complete when the Linear-DiT transformer, the DC-AE VAE, and the Gemma-2 text encoder + its
/// tokenizer (both bundled inside `text_encoder/`) are all present.
pub fn sana_tier_complete(dir: &Path) -> bool {
    dir_has_visible_file_ending(&dir.join("transformer"), ".safetensors")
        && dir_has_visible_file_ending(&dir.join("vae"), ".safetensors")
        && dir.join("text_encoder/gemma-2-2b-it.safetensors").is_file()
        && dir.join("text_encoder/tokenizer.json").is_file()
}

/// Whether a Wan2.2 MLX turnkey tier `dir` is COMPLETE and loadable by the native MLX Wan trainer
/// (sc-13878). The `SceneWorks/wan2.2-{ti2v-5b,t2v-a14b,i2v-a14b}-mlx` turnkeys ship a FLAT MLX layout
/// (a `config.json` + top-level `*.safetensors`, NOT a diffusers `model_index.json` / `transformer/`
/// component tree), so both the shared `model_index.json` guard and rust-api's diffusers-shaped
/// training-base checks (`bf16_component_tree_present`, which probes `transformer/`|`unet/` dirs) are a
/// no-op for them: a torn tier clears the coarse `<tier>/*` glob yet the trainer
/// (mlx-gen-wan `build_trainer_concrete`) dies on the first absent flat file. A tier is complete when the
/// UMT5 T5 encoder, the Wan VAE, the tokenizer, the `config.json`, and the DiT are all present — the DiT
/// being either the single-expert `model.safetensors` (dense TI2V-5B) OR both MoE experts
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` (A14B T2V/I2V). Exact-path `.is_file()`
/// checks are AppleDouble-safe by construction — a hidden `._name` sidecar has a different name
/// (SceneWorks#1333) — and pin the exact filenames the trainer's `LoadSpec` dir must contain.
pub fn wan_tier_complete(dir: &Path) -> bool {
    let shared = dir.join("config.json").is_file()
        && dir.join("t5_encoder.safetensors").is_file()
        && dir.join("vae.safetensors").is_file()
        && dir.join("tokenizer.json").is_file();
    let dense_expert = dir.join("model.safetensors").is_file();
    let dual_expert = dir.join("low_noise_model.safetensors").is_file()
        && dir.join("high_noise_model.safetensors").is_file();
    shared && (dense_expert || dual_expert)
}

/// The sibling tier dirs a SenseNova tier may borrow a `tokenizer.json` from, in the order the engine
/// consults them. Mirrors `SIBLING_TIER_DIRS` in inference's `candle-gen-sensenova` / `mlx-gen-sensenova`
/// `resolve_tokenizer_path` (sc-14432) so this predicate accepts exactly the installs those loaders
/// accept — no more (a false `installed`) and no less (a false `incomplete` on a tier that loads).
const SENSENOVA_SIBLING_TIER_DIRS: [&str; 3] = ["q8", "bf16", "q4"];

/// Whether a SenseNova-U1 tier `dir` can supply the fast `tokenizer.json` its loader needs: its own, or
/// — since the tokenizer is model-wide and byte-identical across quant tiers — a sibling tier's under the
/// same snapshot parent. Only the fixed [`SENSENOVA_SIBLING_TIER_DIRS`] names are consulted, never an
/// arbitrary dir or a path above the snapshot, so this can't be satisfied by a different model's copy.
fn sensenova_tokenizer_resolvable(dir: &Path) -> bool {
    if dir.join("tokenizer.json").is_file() {
        return true;
    }
    dir.parent().is_some_and(|parent| {
        SENSENOVA_SIBLING_TIER_DIRS
            .iter()
            .any(|tier| parent.join(tier).join("tokenizer.json").is_file())
    })
}

/// Whether EVERY shard the safetensors index `index_file` in `dir` names is on disk, resolved against
/// `dir` itself (the same rule `verify_model_snapshot.py` applies to an `expected_files` index).
///
/// The shard set is read from the index rather than settling for "some `.safetensors` is here": the
/// MLX loaders merge whatever shards they find WITHOUT consulting the index, then fail much later on
/// missing tensor keys — so a download interrupted after 1 of N shards would otherwise read complete.
/// An index naming no shards, or a malformed / unreadable one, is treated as incomplete.
fn index_shards_present(dir: &Path, index_file: &str) -> bool {
    let Ok(contents) = std::fs::read_to_string(dir.join(index_file)) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return false;
    };
    let Some(weight_map) = parsed.get("weight_map").and_then(|map| map.as_object()) else {
        return false;
    };
    let mut shards = weight_map
        .values()
        .filter_map(serde_json::Value::as_str)
        .peekable();
    shards.peek().is_some() && shards.all(|shard| dir.join(shard).is_file())
}

/// Whether the SenseNova backbone under `dir` is fully on disk: either the packed single-file
/// `model.safetensors` (every q4/q8 tier, and the `*-fast-mlx` bf16, which the packer emits pre-merged
/// as one file) or a sharded `model.safetensors.index.json` with EVERY shard its `weight_map` names
/// (the base/infographic bf16 tiers ship 8 shards / ~35 GB), per [`index_shards_present`].
fn sensenova_backbone_present(dir: &Path) -> bool {
    dir.join("model.safetensors").is_file()
        || index_shards_present(dir, "model.safetensors.index.json")
}

/// Whether a SenseNova-U1 tier `dir` is COMPLETE and loadable. The `SceneWorks/sensenova-u1-8b*-mlx`
/// turnkeys ship a FLAT unified layout (a `config.json` + the backbone at the tier root, NO diffusers
/// `model_index.json` and no component subdirs), so `tier_components_present` and the coarse
/// `<tier>/*` glob are both a no-op for them: a torn tier reads `installed` in the catalog and
/// short-circuits the worker's tier chain, then dies at load (sc-14432 — the "complete but unloadable
/// tier" class). A tier is complete when the NEO-unify backbone is fully present
/// ([`sensenova_backbone_present`]), its `config.json` is there (`NeoChatConfig::from_dir` errors
/// without it), and a `tokenizer.json` is resolvable ([`sensenova_tokenizer_resolvable`]: its own, else
/// a sibling tier's, exactly as the engine resolves it).
///
/// This is the BASE-variant predicate; the distilled `_fast` ids additionally need
/// [`sensenova_fast_tier_complete`].
pub fn sensenova_tier_complete(dir: &Path) -> bool {
    sensenova_backbone_present(dir)
        && dir.join("config.json").is_file()
        && sensenova_tokenizer_resolvable(dir)
}

/// The provenance marker a pre-merged `*-fast-mlx` tier carries (`DISTILL_MERGED_MARKER` in inference's
/// `mlx-gen-sensenova` / `candle-gen-sensenova` `distill`), and the 8-step distill LoRA filename the
/// loader falls back to when the merge still has to run (`DISTILL_LORA_FILE`).
const SENSENOVA_DISTILL_MERGED_MARKER: &str = "distill_merged.json";
const SENSENOVA_DISTILL_LORA_FILE: &str = "SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors";

/// Whether a DISTILLED (`sensenova_u1_8b*_fast`) tier `dir` is COMPLETE and loadable. On top of
/// [`sensenova_tier_complete`], the fast ids need the 8-step distill accounted for: `load_fast` skips
/// resolve+merge only when the tier carries the `distill_merged.json` marker, and otherwise resolves
/// the distill LoRA — erroring if it cannot. The marker is a ~200-byte non-LFS file that every shipped
/// fast tier carries, which makes it exactly the file a torn download loses while the 12 GB backbone
/// lands; without it a packed tier cannot even merge (the engine refuses to fold a LoRA into a packed
/// projection) and dies at load.
///
/// The co-located LoRA is accepted as the alternative because that is the loader's own fallback
/// (`resolve_distill_lora`). It is the deliberately permissive direction: on a PACKED tier the engine
/// would resolve that LoRA and then still refuse to fold it in, but a packed tier is not distinguishable
/// from a dense fast bf16 by filename (both are `model.safetensors`), and a false `incomplete` hides an
/// installed model while a false `complete` only reproduces today's load error. No shipping repo
/// produces that shape — all three `*-fast-mlx` repos have carried the marker in every tier since their
/// only content commit. A LoRA supplied as a `distill_lora` *component* from another snapshot is
/// invisible to a path predicate, and nothing in SceneWorks uses that route.
pub fn sensenova_fast_tier_complete(dir: &Path) -> bool {
    sensenova_tier_complete(dir)
        && (dir.join(SENSENOVA_DISTILL_MERGED_MARKER).is_file()
            || dir.join(SENSENOVA_DISTILL_LORA_FILE).is_file())
}

/// The SceneWorks ids served by the SenseNova-U1 turnkeys — base + Infographic-V2/V3, each with its
/// distilled `_fast` twin. All six share the flat unified tier layout and the `sensenova-u1` manifest
/// family.
///
/// Listed exhaustively rather than prefix-matched so a future `sensenova_u1_*` id with a different
/// on-disk shape cannot be silently mis-judged, and it lives HERE, beside the predicates, because both
/// the worker's tier resolvers and rust-api's catalog gate on it — a per-crate copy is exactly the drift
/// that leaves one of the two gates un-wired (sc-14432).
pub const SENSENOVA_MODELS: [&str; 6] = [
    "sensenova_u1_8b",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v3",
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3_fast",
];

/// The per-tier completeness predicate for a SenseNova-U1 `model_id`, or `None` when `model_id` is not
/// one of [`SENSENOVA_MODELS`] — [`sensenova_tier_complete`] for the base ids, the distill-aware
/// [`sensenova_fast_tier_complete`] for the `_fast` twins.
pub fn sensenova_tier_predicate(model_id: &str) -> Option<fn(&Path) -> bool> {
    if !SENSENOVA_MODELS.contains(&model_id) {
        return None;
    }
    Some(if model_id.ends_with("_fast") {
        sensenova_fast_tier_complete
    } else {
        sensenova_tier_complete
    })
}

/// The DiT partition directory each MiniMax-H3 catalog entry owns inside a `SceneWorks/minimax-h3-mlx`
/// tier (sc-19078). `minimax_h3` (t2va + fl2va) is the base `transformer/` checkpoint; `minimax_h3_ref`
/// (Ref2VA) is `transformer_ref/`. The names mirror `DIT_COMPONENT` / `REFERENCE_DIT_PARTITION` in
/// inference's `mlx-gen-minimax-h3::model`, which is what the loader actually opens.
///
/// The two partitions are the ONLY things a tier of that repo holds: the Qwen3-VL-32B text encoder,
/// both VAEs and the tokenizer are dense in every tier and ship as shared co-requisites from upstream
/// `MiniMaxAI/MiniMax-H3`, so they are deliberately NOT part of this predicate — the catalog gates them
/// separately as co-requisites (`install_state_for`), which is what keeps a q4 install from demanding
/// the bf16 tier's floor.
pub const MINIMAX_H3_PARTITIONS: [(&str, &str); 2] = [
    ("minimax_h3", "transformer"),
    ("minimax_h3_ref", "transformer_ref"),
];

/// The sharded-weight index inference's `mlx-gen-minimax-h3::convert` writes into every MiniMax-H3 DiT
/// partition. It is the file that makes a 14-shard partition checkable at all: without it the predicate
/// would need fourteen hand-maintained shard names per partition and would silently stop covering a
/// re-shard.
const MINIMAX_H3_DIT_INDEX: &str = "diffusion_pytorch_model.safetensors.index.json";

/// Whether ONE MiniMax-H3 DiT `partition` (`transformer` / `transformer_ref`) under tier dir `dir` is
/// COMPLETE and loadable. `SceneWorks/minimax-h3-mlx` ships no `model_index.json` at either the tier
/// root or inside a partition, so rust-api's coarse `<tier>/<partition>/*` glob is satisfied by a
/// SINGLE landed file: a download interrupted after one of fourteen shards read `installed` and then
/// died at load. A partition is complete when its `config.json` is present (the engine's config parse
/// errors without it) and every shard its [`MINIMAX_H3_DIT_INDEX`] names is on disk.
fn minimax_h3_partition_complete(dir: &Path, partition: &str) -> bool {
    let partition = dir.join(partition);
    partition.join("config.json").is_file()
        && index_shards_present(&partition, MINIMAX_H3_DIT_INDEX)
}

/// Whether a `minimax_h3` (t2va + fl2va) tier `dir` is complete — its `transformer/` partition.
pub fn minimax_h3_tier_complete(dir: &Path) -> bool {
    minimax_h3_partition_complete(dir, MINIMAX_H3_PARTITIONS[0].1)
}

/// Whether a `minimax_h3_ref` (Ref2VA) tier `dir` is complete — its `transformer_ref/` partition.
///
/// Deliberately independent of [`minimax_h3_tier_complete`]: the two partitions are separate catalog
/// entries with disjoint `files` predicates, so a user who installed only the reference checkpoint has
/// a complete `minimax_h3_ref` tier and NO `transformer/` at all. Requiring both would report that
/// working install as torn and prompt an endless repair.
pub fn minimax_h3_ref_tier_complete(dir: &Path) -> bool {
    minimax_h3_partition_complete(dir, MINIMAX_H3_PARTITIONS[1].1)
}

/// The per-tier completeness predicate for a MiniMax-H3 `model_id`, or `None` for an id outside
/// [`MINIMAX_H3_PARTITIONS`]. Dispatched on the id rather than the shared `minimax-h3` family because
/// the two entries own DIFFERENT partition dirs of the same repo — a family-only predicate could not
/// tell them apart and would have to demand both, which is exactly the false-torn case above.
pub fn minimax_h3_tier_predicate(model_id: &str) -> Option<fn(&Path) -> bool> {
    match model_id {
        id if id == MINIMAX_H3_PARTITIONS[0].0 => Some(minimax_h3_tier_complete),
        id if id == MINIMAX_H3_PARTITIONS[1].0 => Some(minimax_h3_ref_tier_complete),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// sc-19573 — the SHARED (non-DiT) half of what `mlx-gen-minimax-h3::load` opens.
//
// Everything below is ONE list, and that is the point. Before sc-19573 the probe list existed twice:
// once as a hard-coded array inside the worker's `minimax_h3_shared_is_complete`, and once
// implicitly as whatever rows happened to be in the catalog entry's `downloads`. The two disagreed —
// `tokenizer/` and the `FL2VA/audio_vae/` constructor documents were probed and never declared — so
// the manifest described an install that could not load. A manifest guard written against a SECOND
// hand-copied list would have reproduced exactly that failure mode, which is why the guard, the
// runtime completeness predicate and the resolver all read these constants and nothing else.
//
// The engine itself lives in the pinned `inference` workspace and cannot be imported here, so this
// is a mirror rather than a derivation. What makes it a HONEST mirror is that it is the ONLY mirror:
// the check that stands in front of the loader ([`minimax_h3_shared_is_complete`]) and the check that
// stands in front of the download ([`crate::builtin_manifests`]' guard) cannot disagree with each
// other, so a wrong entry here goes red in one place and is fixed in one place.
//
// ⚠️ NO TRIPWIRE BINDS THIS TO THE ENGINE. Being the only mirror keeps the two SceneWorks-side
// readers consistent with EACH OTHER; nothing here can go red when the engine's probe list changes
// upstream, because at the current pin (`014134e3`) `mlx-gen-minimax-h3` is not in the tree at all
// and there is no symbol to import or construct. Every constant below was transcribed by reading
// the engine source, and it stays a transcription until the pin carries the crate.
//
// The obligation therefore rides the PIN BUMP: sc-18650 must re-read
// `mlx-gen-minimax-h3::model::load` and reconcile it against these constants, and — once the crate
// is linkable — bind them to it (importing `DIT_COMPONENT` / `TEXT_ENCODER_COMPONENT` directly, and
// asserting the shared-dir list against a real `load` of a fixture missing each path in turn) so
// this stops being a mirror. Recorded on sc-18650 rather than only here, because a comment in this
// file is not something a pin-bump session is guaranteed to open.
// ---------------------------------------------------------------------------

/// The shared component DIRECTORIES a MiniMax-H3 load opens under the snapshot it is handed as
/// `spec.weights` — the upstream `MiniMaxAI/MiniMax-H3` root.
///
/// The text encoder is deliberately NOT here: sc-19120 made it the one shared component whose ROOT
/// depends on the tier (packed under `<tier>/text_encoder` in the rehost at q4/q8, dense under the
/// upstream root at bf16), so it is resolved by [`minimax_h3_text_encoder_dir`] instead of probed at
/// a fixed path. The two DiT partitions are not here either — they are [`MINIMAX_H3_PARTITIONS`],
/// under the TIER root.
pub const MINIMAX_H3_SHARED_PROBED_DIRS: [&str; 3] = ["tokenizer", "vae", "audio_vae"];

/// The directory holding the audio VAE's CONSTRUCTOR DOCUMENTS, relative to the same snapshot root.
///
/// Separate from `audio_vae/` (which carries the weights) because the repackaged root config carries
/// none of the audio VAE's constructor arguments — the engine reads them from here.
pub const MINIMAX_H3_AUDIO_VAE_CONFIG_DIR: &str = "FL2VA/audio_vae";

/// The three files the engine reads out of [`MINIMAX_H3_AUDIO_VAE_CONFIG_DIR`].
///
/// Named individually rather than globbed as `FL2VA/audio_vae/*`: the real directory also holds a
/// 605 MB `model.safetensors` that is a byte-for-byte duplicate of the `audio_vae/` component
/// already downloaded, plus nine `.py` files no Rust engine can execute. A glob would double the
/// audio VAE's bytes for three documents totalling 2,504 B.
pub const MINIMAX_H3_AUDIO_VAE_CONFIG_FILES: [&str; 3] =
    ["config.json", "config.yaml", "metadata.json"];

/// The component directory name the Qwen3-VL-32B text encoder is read from, under whichever root
/// [`minimax_h3_text_encoder_dir`] selects.
pub const MINIMAX_H3_TEXT_ENCODER_DIR: &str = "text_encoder";

/// Where a MiniMax-H3 load reads its text encoder from, for `tier` (sc-19120).
///
/// * `q4` / `q8` → `<tier_root>/<tier>/text_encoder`, the PACKED encoder in
///   `SceneWorks/minimax-h3-mlx`. It does not exist upstream in any form.
/// * anything else (`bf16`) → `<base_root>/text_encoder`, the dense upstream component.
///
/// This function is the reason sc-19573 exists at all in the worker: the pre-sc-19573 completeness
/// check probed `<base_root>/text_encoder` UNCONDITIONALLY, and a q4 or q8 install never has that
/// path — its co-requisite row is `variant`-scoped to bf16 and is never fetched. So every packed-tier
/// install was refused before it reached the engine, with an error blaming the shared floor for a
/// file that tier is correct not to have.
pub fn minimax_h3_text_encoder_dir(
    tier_root: &Path,
    base_root: &Path,
    tier: &str,
) -> std::path::PathBuf {
    match tier {
        "q4" | "q8" => tier_root.join(tier).join(MINIMAX_H3_TEXT_ENCODER_DIR),
        _ => base_root.join(MINIMAX_H3_TEXT_ENCODER_DIR),
    }
}

/// Whether the SHARED components a MiniMax-H3 load probes are all on disk: the three
/// [`MINIMAX_H3_SHARED_PROBED_DIRS`] and the three [`MINIMAX_H3_AUDIO_VAE_CONFIG_FILES`] under
/// `base_root`, plus the tier's `text_encoder` at whatever root
/// [`minimax_h3_text_encoder_dir`] resolved.
///
/// Every one of these is a `coRequisite` download from a repo the tiers do not come from, so a user
/// can genuinely have a complete `q4/` and no shared floor at all.
///
/// The audio-VAE documents are checked as FILES rather than as a directory: the directory exists as
/// soon as any one of its thirteen entries lands, and the three the engine actually opens are the
/// three smallest.
pub fn minimax_h3_shared_is_complete(base_root: &Path, text_encoder_dir: &Path) -> bool {
    MINIMAX_H3_SHARED_PROBED_DIRS
        .iter()
        .all(|component| base_root.join(component).is_dir())
        && MINIMAX_H3_AUDIO_VAE_CONFIG_FILES.iter().all(|file| {
            base_root
                .join(MINIMAX_H3_AUDIO_VAE_CONFIG_DIR)
                .join(file)
                .is_file()
        })
        && text_encoder_dir.is_dir()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn touch(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, b"x").unwrap();
    }

    /// Write a COMPLETE Anima tier tree, then return its dir. Callers delete one component to probe that
    /// each is load-bearing (a mutation check — a torn tier must go RED, not pass on the backbone alone).
    fn seed_anima(dir: &Path) {
        touch(&dir.join("diffusion_models/anima-base-v1.0.safetensors"));
        touch(&dir.join("text_encoders/qwen_3_06b_base.safetensors"));
        touch(&dir.join("vae/qwen_image_vae.safetensors"));
    }

    fn seed_boogu(dir: &Path) {
        touch(&dir.join("transformer/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("transformer/config.json"));
        touch(&dir.join("mllm/model.safetensors"));
        touch(&dir.join("mllm/tokenizer.json"));
        touch(&dir.join("vae/diffusion_pytorch_model.safetensors"));
    }

    fn seed_sana(dir: &Path) {
        touch(&dir.join("transformer/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("vae/diffusion_pytorch_model.safetensors"));
        touch(&dir.join("text_encoder/gemma-2-2b-it.safetensors"));
        touch(&dir.join("text_encoder/tokenizer.json"));
    }

    /// Write a COMPLETE dense (single-expert, TI2V-5B) Wan MLX tier tree.
    fn seed_wan_dense(dir: &Path) {
        touch(&dir.join("config.json"));
        touch(&dir.join("t5_encoder.safetensors"));
        touch(&dir.join("vae.safetensors"));
        touch(&dir.join("tokenizer.json"));
        touch(&dir.join("model.safetensors"));
    }

    /// Write a COMPLETE dual-expert (A14B MoE) Wan MLX tier tree.
    fn seed_wan_moe(dir: &Path) {
        touch(&dir.join("config.json"));
        touch(&dir.join("t5_encoder.safetensors"));
        touch(&dir.join("vae.safetensors"));
        touch(&dir.join("tokenizer.json"));
        touch(&dir.join("low_noise_model.safetensors"));
        touch(&dir.join("high_noise_model.safetensors"));
    }

    #[test]
    fn anima_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_anima(&dir);
        assert!(anima_tier_complete(&dir), "fully-seeded tier is complete");

        // Mutation check: removing ANY single component must flip the verdict to incomplete.
        for component in [
            "diffusion_models/anima-base-v1.0.safetensors",
            "text_encoders/qwen_3_06b_base.safetensors",
            "vae/qwen_image_vae.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_anima(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !anima_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn anima_wrong_text_encoder_filename_is_incomplete() {
        // The predicate pins the EXACT dense-TE filename; a differently-named `.safetensors` in
        // `text_encoders/` does not satisfy it (guards a typo'd rename from silently passing).
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_anima(&dir);
        fs::remove_file(dir.join("text_encoders/qwen_3_06b_base.safetensors")).unwrap();
        touch(&dir.join("text_encoders/some_other.safetensors"));
        assert!(!anima_tier_complete(&dir));
    }

    #[test]
    fn anima_ignores_appledouble_dit_sidecar() {
        // A hidden `._` DiT sidecar is not a real weight (SceneWorks#1333).
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_anima(&dir);
        fs::remove_file(dir.join("diffusion_models/anima-base-v1.0.safetensors")).unwrap();
        touch(&dir.join("diffusion_models/._anima-base-v1.0.safetensors"));
        assert!(!anima_tier_complete(&dir));
    }

    #[test]
    fn boogu_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("base");
        seed_boogu(&dir);
        assert!(boogu_tier_complete(&dir));

        for component in [
            "transformer/diffusion_pytorch_model.safetensors",
            "transformer/config.json",
            "mllm/model.safetensors",
            "mllm/tokenizer.json",
            "vae/diffusion_pytorch_model.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_boogu(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !boogu_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn boogu_accepts_sharded_transformer_index() {
        // A sharded transformer (index.json, no single-file weight) is loadable too.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("base");
        seed_boogu(&dir);
        fs::remove_file(dir.join("transformer/diffusion_pytorch_model.safetensors")).unwrap();
        touch(&dir.join("transformer/diffusion_pytorch_model.safetensors.index.json"));
        assert!(boogu_tier_complete(&dir));
    }

    #[test]
    fn sana_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_sana(&dir);
        assert!(sana_tier_complete(&dir));

        for component in [
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/gemma-2-2b-it.safetensors",
            "text_encoder/tokenizer.json",
        ] {
            let torn = tmp.path().join("torn");
            seed_sana(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !sana_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn wan_dense_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_wan_dense(&dir);
        assert!(
            wan_tier_complete(&dir),
            "fully-seeded dense tier is complete"
        );

        // Mutation check: removing ANY single component must flip the verdict to incomplete.
        for component in [
            "config.json",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
            "model.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_wan_dense(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !wan_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn wan_moe_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_wan_moe(&dir);
        assert!(wan_tier_complete(&dir), "fully-seeded MoE tier is complete");

        // Mutation check: BOTH experts are load-bearing (a single-expert dir is not a valid A14B tier)
        // alongside every shared component.
        for component in [
            "config.json",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
            "low_noise_model.safetensors",
            "high_noise_model.safetensors",
        ] {
            let torn = tmp.path().join("torn");
            seed_wan_moe(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !wan_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn wan_ignores_appledouble_expert_sidecar() {
        // A hidden `._` DiT sidecar is not a real weight (SceneWorks#1333); exact-path checks reject it.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_wan_dense(&dir);
        fs::remove_file(dir.join("model.safetensors")).unwrap();
        touch(&dir.join("._model.safetensors"));
        assert!(!wan_tier_complete(&dir));
    }

    /// Write a COMPLETE packed (q4/q8) SenseNova-U1 tier tree: flat backbone + config + its own
    /// tokenizer.
    fn seed_sensenova_packed(dir: &Path) {
        touch(&dir.join("model.safetensors"));
        touch(&dir.join("config.json"));
        touch(&dir.join("tokenizer.json"));
    }

    #[test]
    fn sensenova_packed_complete_true_each_component_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_sensenova_packed(&dir);
        assert!(
            sensenova_tier_complete(&dir),
            "fully-seeded tier is complete"
        );

        // Mutation check: with NO sibling tier to borrow from, removing ANY component reads incomplete.
        for component in ["model.safetensors", "config.json", "tokenizer.json"] {
            let torn = tmp.path().join("torn").join("q8");
            seed_sensenova_packed(&torn);
            fs::remove_file(torn.join(component)).unwrap();
            assert!(
                !sensenova_tier_complete(&torn),
                "removing {component} must read incomplete"
            );
            fs::remove_dir_all(tmp.path().join("torn")).ok();
        }
    }

    /// Write a sharded bf16 backbone: an index naming `shards` files, plus the shards in `present`.
    fn seed_sensenova_sharded(dir: &Path, shards: &[&str], present: &[&str]) {
        let weight_map = shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                (
                    format!("layer.{index}.weight"),
                    serde_json::Value::String((*shard).to_owned()),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        fs::create_dir_all(dir).unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::json!({ "weight_map": weight_map }).to_string(),
        )
        .unwrap();
        for shard in present {
            touch(&dir.join(shard));
        }
        touch(&dir.join("config.json"));
        touch(&dir.join("tokenizer.json"));
    }

    #[test]
    fn sensenova_sharded_bf16_backbone_needs_every_shard_the_index_names() {
        // The base/infographic bf16 tiers ship the backbone SHARDED (8 × `model-0000n-of-00008` + an
        // index), not as the single packed `model.safetensors` the quant tiers carry. The engine merges
        // whatever shards it finds WITHOUT reading the index and only then fails on missing tensors, so
        // "the index plus SOME shard" must not read complete — every shard it names has to be on disk.
        let shards = [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ];
        let tmp = tempfile::tempdir().unwrap();

        let complete = tmp.path().join("bf16");
        seed_sensenova_sharded(&complete, &shards, &shards);
        assert!(sensenova_tier_complete(&complete));

        let torn = tmp.path().join("torn").join("bf16");
        seed_sensenova_sharded(&torn, &shards, &shards[..1]);
        assert!(
            !sensenova_tier_complete(&torn),
            "1 of 3 shards on disk is a torn tier, not a complete one"
        );

        // A malformed / unreadable index is incomplete, not a pass-by-default.
        let malformed = tmp.path().join("malformed").join("bf16");
        seed_sensenova_sharded(&malformed, &shards, &shards);
        fs::write(malformed.join("model.safetensors.index.json"), b"{oops").unwrap();
        assert!(!sensenova_tier_complete(&malformed));
    }

    #[test]
    fn sensenova_fast_tier_needs_the_distill_merge_accounted_for() {
        // sc-14432: `load_fast` skips the distill merge ONLY on the `distill_merged.json` marker, and a
        // packed tier cannot merge a LoRA at all — so a fast tier that lost the ~200-byte marker while
        // the 12 GB backbone landed is unloadable and must not read complete.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_sensenova_packed(&dir);
        assert!(
            sensenova_tier_complete(&dir),
            "the BASE predicate is satisfied — the marker is a fast-only requirement"
        );
        assert!(
            !sensenova_fast_tier_complete(&dir),
            "a fast tier with no marker and no co-located LoRA must read incomplete"
        );

        touch(&dir.join("distill_merged.json"));
        assert!(sensenova_fast_tier_complete(&dir));

        // The loader's own fallback — a co-located distill LoRA — also satisfies it.
        let lora_only = tmp.path().join("lora-only").join("q4");
        seed_sensenova_packed(&lora_only);
        touch(&lora_only.join("SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors"));
        assert!(sensenova_fast_tier_complete(&lora_only));

        // The fast predicate still subsumes the base one: a torn backbone fails even with the marker.
        fs::remove_file(dir.join("model.safetensors")).unwrap();
        assert!(!sensenova_fast_tier_complete(&dir));
    }

    #[test]
    fn sensenova_borrows_a_sibling_tiers_tokenizer_but_only_a_sibling_tier() {
        // sc-14432: a tier that ships no `tokenizer.json` still LOADS when a sibling tier has one (the
        // engine's `resolve_tokenizer_path` borrows it), so it must read complete, not incomplete.
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = tmp.path().join("snapshot");
        let q4 = snapshot.join("q4");
        touch(&q4.join("model.safetensors"));
        touch(&q4.join("config.json"));
        assert!(
            !sensenova_tier_complete(&q4),
            "no tokenizer anywhere in the snapshot → incomplete"
        );

        // A tokenizer in the snapshot ROOT (not a tier dir) is NOT borrowable — the engine consults
        // only the fixed tier names, so accepting it here would report a tier that cannot load.
        touch(&snapshot.join("tokenizer.json"));
        assert!(!sensenova_tier_complete(&q4));

        // A tokenizer in a NON-tier sibling dir is likewise not borrowable.
        touch(&snapshot.join("original").join("tokenizer.json"));
        assert!(!sensenova_tier_complete(&q4));

        // The q8 sibling's copy IS borrowable — this is the shipping base-repo shape.
        touch(&snapshot.join("q8").join("tokenizer.json"));
        assert!(sensenova_tier_complete(&q4));
    }

    #[test]
    fn sensenova_ignores_appledouble_shard_sidecar() {
        // A hidden `._` shard is not a real weight (SceneWorks#1333). The index names the REAL shard, so
        // the exact-path check rejects the sidecar standing in for it rather than counting it.
        let shards = ["model-00001-of-00001.safetensors"];
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_sensenova_sharded(&dir, &shards, &[]);
        touch(&dir.join("._model-00001-of-00001.safetensors"));
        assert!(!sensenova_tier_complete(&dir));

        // Mutation check: the REAL shard alongside it flips the verdict, so the assertion above is
        // discriminating on the sidecar, not merely on an empty tier.
        touch(&dir.join(shards[0]));
        assert!(sensenova_tier_complete(&dir));
    }

    #[test]
    fn sensenova_predicate_dispatch_is_keyed_on_the_shared_id_list() {
        // The one dispatcher both the worker's tier resolver and rust-api's catalog gate on. Probe it
        // BEHAVIOURALLY (fn pointers don't compare reliably): a tier that satisfies the base contract but
        // ships no distill marker separates the two predicates.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_sensenova_packed(&dir);

        for base in [
            "sensenova_u1_8b",
            "sensenova_u1_8b_infographic_v2",
            "sensenova_u1_8b_infographic_v3",
        ] {
            let complete = sensenova_tier_predicate(base).expect("a known base id has a predicate");
            assert!(complete(&dir), "{base} must take the BASE predicate");
        }
        for fast in [
            "sensenova_u1_8b_fast",
            "sensenova_u1_8b_infographic_v2_fast",
            "sensenova_u1_8b_infographic_v3_fast",
        ] {
            let complete = sensenova_tier_predicate(fast).expect("a known fast id has a predicate");
            assert!(
                !complete(&dir),
                "{fast} must take the DISTILL-aware predicate"
            );
        }

        // An id outside the list gets no predicate at all, rather than a guessed one — so a future
        // `sensenova_u1_*` with a different on-disk shape is left alone in BOTH consumers, not
        // mis-tightened in one of them.
        assert!(sensenova_tier_predicate("sensenova_u1_30b_a3b").is_none());
        assert!(sensenova_tier_predicate("sana_1600m").is_none());
    }

    /// Write ONE MiniMax-H3 DiT partition under tier dir `dir`: `config.json` plus a shard index naming
    /// `shards`, of which only `present` actually land. The real repo ships 14 shards per partition; the
    /// fixture keeps the COUNT small and the index-vs-disk relationship exact, which is what the
    /// predicate reads.
    fn seed_minimax_partition(dir: &Path, partition: &str, shards: &[&str], present: &[&str]) {
        let partition = dir.join(partition);
        let weight_map = shards
            .iter()
            .enumerate()
            .map(|(index, shard)| {
                (
                    format!("blocks.{index}.attn.to_q.weight"),
                    serde_json::Value::String((*shard).to_owned()),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        fs::create_dir_all(&partition).unwrap();
        fs::write(
            partition.join(MINIMAX_H3_DIT_INDEX),
            serde_json::json!({ "weight_map": weight_map }).to_string(),
        )
        .unwrap();
        touch(&partition.join("config.json"));
        for shard in present {
            touch(&partition.join(shard));
        }
    }

    const MINIMAX_SHARDS: [&str; 3] = [
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
    ];

    /// Write a COMPLETE `minimax_h3` tier (the base `transformer/` partition only).
    fn seed_minimax_h3(dir: &Path) {
        seed_minimax_partition(dir, "transformer", &MINIMAX_SHARDS, &MINIMAX_SHARDS);
    }

    /// Write a COMPLETE `minimax_h3_ref` tier (the `transformer_ref/` partition only).
    fn seed_minimax_h3_ref(dir: &Path) {
        seed_minimax_partition(dir, "transformer_ref", &MINIMAX_SHARDS, &MINIMAX_SHARDS);
    }

    #[test]
    fn minimax_h3_complete_true_each_file_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_minimax_h3(&dir);
        assert!(
            minimax_h3_tier_complete(&dir),
            "fully-seeded transformer partition is complete"
        );

        // Mutation check, applied to EACH file on its own: the config, the index, and every one of the
        // shards it names must independently flip the verdict. A tier that clears the coarse
        // `q4/transformer/*` glob on a single landed file is exactly the "installed but unloadable"
        // shape this predicate exists to catch.
        let mut files = vec!["config.json".to_owned(), MINIMAX_H3_DIT_INDEX.to_owned()];
        files.extend(MINIMAX_SHARDS.iter().map(|shard| (*shard).to_owned()));
        for file in files {
            let torn = tmp.path().join("torn");
            seed_minimax_h3(&torn);
            fs::remove_file(torn.join("transformer").join(&file)).unwrap();
            assert!(
                !minimax_h3_tier_complete(&torn),
                "removing transformer/{file} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn minimax_h3_ref_complete_true_each_file_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_minimax_h3_ref(&dir);
        assert!(minimax_h3_ref_tier_complete(&dir));

        let mut files = vec!["config.json".to_owned(), MINIMAX_H3_DIT_INDEX.to_owned()];
        files.extend(MINIMAX_SHARDS.iter().map(|shard| (*shard).to_owned()));
        for file in files {
            let torn = tmp.path().join("torn");
            seed_minimax_h3_ref(&torn);
            fs::remove_file(torn.join("transformer_ref").join(&file)).unwrap();
            assert!(
                !minimax_h3_ref_tier_complete(&torn),
                "removing transformer_ref/{file} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
    }

    #[test]
    fn minimax_h3_partitions_are_judged_independently_of_each_other() {
        // The two catalog entries own DISJOINT partition dirs of ONE repo, so each tier verdict must
        // read only its own partition. A user who installed just the reference checkpoint has no
        // `transformer/` at all; demanding both would report that working install as torn forever.
        let tmp = tempfile::tempdir().unwrap();
        let base_only = tmp.path().join("base-only");
        seed_minimax_h3(&base_only);
        assert!(minimax_h3_tier_complete(&base_only));
        assert!(!minimax_h3_ref_tier_complete(&base_only));

        let ref_only = tmp.path().join("ref-only");
        seed_minimax_h3_ref(&ref_only);
        assert!(minimax_h3_ref_tier_complete(&ref_only));
        assert!(!minimax_h3_tier_complete(&ref_only));

        // Both installed side by side in one tier — the shipped shape once a user owns both entries.
        let both = tmp.path().join("both");
        seed_minimax_h3(&both);
        seed_minimax_h3_ref(&both);
        assert!(minimax_h3_tier_complete(&both));
        assert!(minimax_h3_ref_tier_complete(&both));
    }

    #[test]
    fn minimax_h3_partition_needs_every_shard_the_index_names() {
        // 2 of 3 shards landed: the coarse glob is satisfied, the index is not.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("bf16");
        seed_minimax_partition(&dir, "transformer", &MINIMAX_SHARDS, &MINIMAX_SHARDS[..2]);
        assert!(!minimax_h3_tier_complete(&dir));

        // Mutation check: landing the last shard flips it, so the assertion above discriminates on the
        // MISSING shard rather than on some unrelated absence.
        touch(&dir.join("transformer").join(MINIMAX_SHARDS[2]));
        assert!(minimax_h3_tier_complete(&dir));
    }

    #[test]
    fn minimax_h3_ignores_appledouble_shard_sidecars_and_empty_indexes() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q8");
        seed_minimax_partition(&dir, "transformer", &MINIMAX_SHARDS, &MINIMAX_SHARDS[..2]);
        // A hidden `._` sidecar has a different name than the shard the index maps to, so an exact-path
        // check rejects it (SceneWorks#1333).
        touch(
            &dir.join("transformer")
                .join(format!("._{}", MINIMAX_SHARDS[2])),
        );
        assert!(!minimax_h3_tier_complete(&dir));

        // An index whose `weight_map` names NOTHING is incomplete rather than vacuously complete: a
        // truncated/placeholder index must never certify a partition with no weights at all.
        let empty = tmp.path().join("empty-index");
        seed_minimax_partition(&empty, "transformer", &[], &[]);
        assert!(!minimax_h3_tier_complete(&empty));
    }

    #[test]
    fn minimax_h3_predicate_dispatch_is_keyed_on_the_entry_id() {
        // Behavioural probe (fn pointers don't compare reliably): a tier holding ONLY the base
        // partition separates the two predicates the dispatcher can return.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("q4");
        seed_minimax_h3(&dir);

        let base = minimax_h3_tier_predicate("minimax_h3").expect("base id has a predicate");
        assert!(base(&dir), "minimax_h3 must take the transformer predicate");
        let reference =
            minimax_h3_tier_predicate("minimax_h3_ref").expect("reference id has a predicate");
        assert!(
            !reference(&dir),
            "minimax_h3_ref must take the transformer_ref predicate"
        );

        // Ids outside the pair get no predicate, so a future MiniMax entry with a different on-disk
        // shape is left alone rather than mis-tightened. `minimax_h3_ref` is a PREFIX extension of
        // `minimax_h3`, so a prefix match here would silently mis-dispatch — assert exact keying.
        assert!(minimax_h3_tier_predicate("minimax_h3_2k").is_none());
        assert!(minimax_h3_tier_predicate("minimax_h3_refiner").is_none());
        assert!(minimax_h3_tier_predicate("wan_2_2").is_none());
    }

    /// Write the COMPLETE shared floor under `base_root` — every path
    /// [`minimax_h3_shared_is_complete`] probes there, and nothing else.
    fn seed_minimax_shared(base_root: &Path) {
        for component in MINIMAX_H3_SHARED_PROBED_DIRS {
            touch(&base_root.join(component).join("placeholder"));
        }
        for file in MINIMAX_H3_AUDIO_VAE_CONFIG_FILES {
            touch(&base_root.join(MINIMAX_H3_AUDIO_VAE_CONFIG_DIR).join(file));
        }
    }

    #[test]
    fn minimax_h3_shared_floor_needs_every_probed_path_and_each_is_load_bearing() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("base");
        let text_encoder = base.join(MINIMAX_H3_TEXT_ENCODER_DIR);
        seed_minimax_shared(&base);
        touch(&text_encoder.join("config.json"));
        assert!(
            minimax_h3_shared_is_complete(&base, &text_encoder),
            "a fully-seeded shared floor is complete"
        );

        // Mutation check, applied to EACH probed path on its own — the set passing together proves
        // nothing about whether any individual member is read.
        for component in MINIMAX_H3_SHARED_PROBED_DIRS {
            let torn = tmp.path().join("torn");
            seed_minimax_shared(&torn);
            let torn_te = torn.join(MINIMAX_H3_TEXT_ENCODER_DIR);
            touch(&torn_te.join("config.json"));
            fs::remove_dir_all(torn.join(component)).unwrap();
            assert!(
                !minimax_h3_shared_is_complete(&torn, &torn_te),
                "removing {component}/ must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }
        for file in MINIMAX_H3_AUDIO_VAE_CONFIG_FILES {
            let torn = tmp.path().join("torn");
            seed_minimax_shared(&torn);
            let torn_te = torn.join(MINIMAX_H3_TEXT_ENCODER_DIR);
            touch(&torn_te.join("config.json"));
            fs::remove_file(torn.join(MINIMAX_H3_AUDIO_VAE_CONFIG_DIR).join(file)).unwrap();
            assert!(
                !minimax_h3_shared_is_complete(&torn, &torn_te),
                "removing {MINIMAX_H3_AUDIO_VAE_CONFIG_DIR}/{file} must read incomplete"
            );
            fs::remove_dir_all(&torn).ok();
        }

        // The text encoder is the sixth load-bearing path, and it is checked at the root the CALLER
        // resolved rather than under `base_root`.
        let no_te = tmp.path().join("no-te");
        seed_minimax_shared(&no_te);
        assert!(!minimax_h3_shared_is_complete(
            &no_te,
            &no_te.join(MINIMAX_H3_TEXT_ENCODER_DIR)
        ));

        // The audio-VAE documents are files, not a directory: a `FL2VA/audio_vae/` that exists but
        // holds only the 605 MB weight duplicate is NOT the constructor documents the engine opens.
        let weights_only = tmp.path().join("weights-only");
        seed_minimax_shared(&weights_only);
        let weights_te = weights_only.join(MINIMAX_H3_TEXT_ENCODER_DIR);
        touch(&weights_te.join("config.json"));
        for file in MINIMAX_H3_AUDIO_VAE_CONFIG_FILES {
            fs::remove_file(
                weights_only
                    .join(MINIMAX_H3_AUDIO_VAE_CONFIG_DIR)
                    .join(file),
            )
            .unwrap();
        }
        touch(
            &weights_only
                .join(MINIMAX_H3_AUDIO_VAE_CONFIG_DIR)
                .join("model.safetensors"),
        );
        assert!(
            !minimax_h3_shared_is_complete(&weights_only, &weights_te),
            "a present FL2VA/audio_vae/ dir with no constructor documents must read incomplete"
        );
    }

    #[test]
    fn minimax_h3_text_encoder_root_is_tier_dependent() {
        // sc-19120 + sc-19573: the PACKED q4/q8 encoders live beside the DiT tiers in the rehost and
        // do not exist upstream at all, so a q4 install legitimately has NO `text_encoder/` under the
        // upstream snapshot. Resolving both tiers to the same root is what refused every packed-tier
        // install before sc-19573.
        let tier_root = Path::new("/tiers");
        let base_root = Path::new("/base");
        assert_eq!(
            minimax_h3_text_encoder_dir(tier_root, base_root, "q4"),
            Path::new("/tiers/q4/text_encoder")
        );
        assert_eq!(
            minimax_h3_text_encoder_dir(tier_root, base_root, "q8"),
            Path::new("/tiers/q8/text_encoder")
        );
        assert_eq!(
            minimax_h3_text_encoder_dir(tier_root, base_root, "bf16"),
            Path::new("/base/text_encoder")
        );
    }

    /// A q4 install has NO `text_encoder/` under the upstream snapshot and MUST still read complete.
    /// This is the exact shape sc-19573 fixed; it is asserted against the shipped tier names rather
    /// than a synthetic one so a future tier rename cannot leave it vacuously green.
    #[test]
    fn minimax_h3_packed_tier_install_is_complete_without_an_upstream_text_encoder() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("MiniMaxAI--MiniMax-H3");
        let tiers = tmp.path().join("SceneWorks--minimax-h3-mlx");
        seed_minimax_shared(&base);
        for tier in ["q4", "q8"] {
            touch(
                &tiers
                    .join(tier)
                    .join(MINIMAX_H3_TEXT_ENCODER_DIR)
                    .join("config.json"),
            );
            let resolved = minimax_h3_text_encoder_dir(&tiers, &base, tier);
            assert!(
                minimax_h3_shared_is_complete(&base, &resolved),
                "the {tier} packed text encoder must satisfy the shared floor"
            );
        }
        // bf16 still reads the upstream component, and is genuinely incomplete without it.
        assert!(!minimax_h3_shared_is_complete(
            &base,
            &minimax_h3_text_encoder_dir(&tiers, &base, "bf16")
        ));
        touch(&base.join(MINIMAX_H3_TEXT_ENCODER_DIR).join("config.json"));
        assert!(minimax_h3_shared_is_complete(
            &base,
            &minimax_h3_text_encoder_dir(&tiers, &base, "bf16")
        ));
    }

    #[test]
    fn missing_tier_dir_is_incomplete_for_every_family() {
        // A tier subdir that does not exist at all (never converted / never downloaded) is incomplete,
        // not a panic — the catalog treats it as a clean "missing".
        let tmp = tempfile::tempdir().unwrap();
        let absent = tmp.path().join("nope");
        assert!(!anima_tier_complete(&absent));
        assert!(!boogu_tier_complete(&absent));
        assert!(!sana_tier_complete(&absent));
        assert!(!wan_tier_complete(&absent));
        assert!(!sensenova_tier_complete(&absent));
        assert!(!minimax_h3_tier_complete(&absent));
        assert!(!minimax_h3_ref_tier_complete(&absent));
    }
}
