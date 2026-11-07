/// A base model whose upstream snapshot omits the HF **fast** `tokenizer.json` the in-process Rust
/// generators/trainers require, paired with the small public `SceneWorks/*` repo that hosts the
/// derived `tokenizer.json` to overlay at install. Materializing once + hosting (rather than running
/// a slow→fast conversion locally) keeps a Python-free desktop working out-of-box.
struct DerivedTokenizerOverlay {
    /// Upstream base-model repo whose snapshot ships only a *slow* tokenizer.
    base_repo: &'static str,
    /// SceneWorks-hosted repo holding the materialized fast `tokenizer.json`.
    tokenizer_repo: &'static str,
    /// Pinned commit of [`tokenizer_repo`] to fetch (sc-9879, F-077 follow-up). The tokenizer repo is a
    /// fixed SceneWorks-hosted const here (no manifest/payload override reaches this overlay), so pulling
    /// the mutable `main` branch would let an upstream re-push silently swap the derived `tokenizer.json`
    /// the in-process generator/trainer constructs from. Pin the exact commit for defense-in-depth
    /// (mirrors sc-8879/sc-9682). Per-entry because entries may share a repo (Qwen-Image / -2512) but the
    /// pin must be resolved per distinct repo. The standard resolve+download path still verifies each
    /// file's reported `lfs.oid`.
    tokenizer_revision: &'static str,
}

/// The single overlaid file (in every case the HF fast serialization) and its in-snapshot subdir.
const DERIVED_TOKENIZER_FILE: &str = "tokenizer.json";

/// Base models that need a derived fast-tokenizer overlay. Keep entries narrowly scoped to the exact
/// base repo — sibling repos (IP-Adapter, Edit variants that ship their own `tokenizer.json`, etc.)
/// must NOT match.
const DERIVED_TOKENIZER_OVERLAYS: &[DerivedTokenizerOverlay] = &[
    // Kolors (sc-4764): conditions on ChatGLM3-6B, which ships only the **slow** SentencePiece
    // tokenizer (`tokenizer.model`). The Rust Kolors generator (sc-3875) + LoRA/LoKr trainer
    // (sc-4732) construct via `KolorsTokenizer::from_dir`, which reads only `tokenizer/tokenizer.json`.
    // Derived via mlx-gen `tools/build_kolors_tokenizer.py` (content-id-parity validated).
    DerivedTokenizerOverlay {
        base_repo: "Kwai-Kolors/Kolors-diffusers",
        tokenizer_repo: "SceneWorks/kolors-chatglm3-tokenizer",
        tokenizer_revision: "4001e09f10ef05845457b976bbf1d28d54319886",
    },
    // Qwen-Image (sc-6570): ships the Qwen2 BPE tokenizer as `vocab.json` + `merges.txt` only. The MLX
    // `qwen-image` provider's `load_tokenizer` reads `tokenizer/tokenizer.json`. Derived via mlx-gen
    // `tools/build_qwen_tokenizer.py` (byte-identical to the fork's runtime fast tokenizer). Note:
    // `Qwen/Qwen-Image-Edit-2511` already ships its own `tokenizer.json`, so it needs no overlay.
    DerivedTokenizerOverlay {
        base_repo: "Qwen/Qwen-Image",
        tokenizer_repo: "SceneWorks/qwen-image-tokenizer",
        tokenizer_revision: "b0178292f0f4b1be9b5bfdda2b6e97fda0e195c3",
    },
    // Qwen-Image-2512 (sc-8271): the Dec-2025 base refresh is architecturally identical to
    // `Qwen/Qwen-Image` and reuses the same Qwen2 BPE tokenizer (unchanged across the line),
    // so it points at the same hosted overlay repo.
    DerivedTokenizerOverlay {
        base_repo: "Qwen/Qwen-Image-2512",
        tokenizer_repo: "SceneWorks/qwen-image-tokenizer",
        tokenizer_revision: "b0178292f0f4b1be9b5bfdda2b6e97fda0e195c3",
    },
];

/// The hosted tokenizer repo + its pinned revision + overlay dest for a just-downloaded `repo`, or
/// `None` when `repo` is not a base model that needs the overlay (a no-op for every other model). Pure so
/// the repo guard + pinned revision + target path are unit-testable without a download.
pub(crate) fn derived_tokenizer_overlay(
    repo: &str,
    snapshot_dir: &Path,
) -> Option<(&'static str, &'static str, PathBuf)> {
    let trimmed = repo.trim();
    DERIVED_TOKENIZER_OVERLAYS
        .iter()
        .find(|overlay| overlay.base_repo == trimmed)
        .map(|overlay| {
            (
                overlay.tokenizer_repo,
                overlay.tokenizer_revision,
                snapshot_dir.join("tokenizer").join(DERIVED_TOKENIZER_FILE),
            )
        })
}

/// After a base-model download, overlay the derived `tokenizer.json` from its SceneWorks tokenizer
/// repo into the snapshot's `tokenizer/` dir, so the in-process generator/trainer can construct (they
/// read only `tokenizer/tokenizer.json`, which these upstreams omit). No-op for any other repo;
/// idempotent — skips when the file is already present (a re-install, or a snapshot that already
/// shipped it). Reuses the standard HF resolve + download path, so the SceneWorks repo's
/// auth/size/resume handling matches every other download.
pub(crate) async fn overlay_derived_tokenizer(
    api: &ApiClient,
    settings: &Settings,
    http_client: &reqwest::Client,
    job_id: &str,
    repo: &str,
    snapshot_dir: &Path,
) -> WorkerResult<()> {
    let Some((tokenizer_repo, tokenizer_revision, dest)) =
        derived_tokenizer_overlay(repo, snapshot_dir)
    else {
        return Ok(());
    };
    if dest.exists() {
        return Ok(());
    }
    let tokenizer_dir = dest
        .parent()
        .expect("overlay dest always has a tokenizer/ parent");
    let snapshot = HuggingFaceSnapshot::resolve(
        http_client,
        settings,
        tokenizer_repo,
        tokenizer_revision,
        &[DERIVED_TOKENIZER_FILE.to_owned()],
    )
    .await?;
    let mut progress = DownloadProgress::new(
        tokenizer_repo,
        0,
        snapshot.total_bytes(),
        progress_report_interval(settings),
    );
    download_snapshot(
        &DownloadContext {
            api,
            client: http_client,
            settings,
            job_id,
            cancel_message: "Derived tokenizer overlay canceled by user.",
            fresh_download: false,
        },
        tokenizer_dir,
        &snapshot,
        &mut progress,
    )
    .await
}

/// Native Rust/MLX FLUX.2-klein single-file → diffusers convert (sc-3136), replacing the
/// retired Python `mlx_flux_convert.py` (sc-3032). Loads the wikeeyang single-file
/// transformer, remaps it to the diffusers layout (fused-qkv 1→3 split + the load-bearing
/// `norm_out` scale/shift swap), validates against the base, and assembles a local diffusers
/// dir whose borrowed vae/text-encoder/tokenizer are absolute symlinks (so they survive the
/// worker's temp→final atomic rename). Runs MLX on macOS; the candle (Windows/CUDA) lane runs the
/// byte-equivalent candle twin (sc-7459) so `flux2_klein_9b_true_v2` installs + converts off-Mac too.
#[cfg(target_os = "macos")]
fn convert_flux2_klein_diffusers(
    source_file: &Path,
    base_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    runtime_macos::providers::flux2::convert_and_assemble(source_file, base_dir, out_dir)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

/// Candle (Windows/CUDA) twin of the macOS converter (sc-7459): the same single-file → diffusers key
/// remap (fused-qkv split + the `norm_out` scale/shift swap), implemented on candle CPU tensors, so the
/// `flux2_klein_9b_true_v2` weight variant has a real install-time convert lane on the candle worker
/// (the borrowed components are hardlinked, not symlinked — Windows symlinks fail to read; see the
/// crate's `convert` module). Reached only on the `backend-candle` build.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
fn convert_flux2_klein_diffusers(
    source_file: &Path,
    base_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720 / sc-7459): backend-specific weight converter; not a registry contract.
    runtime_cuda::providers::flux2::convert_and_assemble(source_file, base_dir, out_dir)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(not(native_lane))]
fn convert_flux2_klein_diffusers(
    _source_file: &Path,
    _base_dir: &Path,
    _out_dir: &Path,
) -> Result<(), String> {
    Err(
        "FLUX.2-klein conversion requires the candle backend (Windows/CUDA, --features \
         backend-candle) or macOS (mlx-gen-flux2)."
            .to_owned(),
    )
}

/// The base directory a FLUX.2-klein convert borrows its VAE / text-encoder / tokenizer from (sc-14978).
/// The SceneWorks re-host the base klein card installs (`SceneWorks/flux2-klein-9b-mlx`) keeps each quant
/// tier in its own subdir, so the borrowed components live under `<tier>/`, not the snapshot root. The
/// borrowed components are dense and identical across tiers (only the transformer is packed), so any
/// installed tier serves: prefer `subdir` (the manifest's `convertBaseSubdir`), then fall back to the
/// standard tiers. When `subdir` is `None` the base is a plain root-layout diffusers snapshot and the
/// snapshot root itself is used. `None` when no candidate holds the borrowed subdirs — a torn base whose
/// convert would fail deep with `No such file or directory`.
fn resolve_flux2_klein_base_dir(snapshot: &Path, subdir: Option<&str>) -> Option<PathBuf> {
    let holds_borrowed = |dir: &Path| {
        dir.join("tokenizer").is_dir()
            && dir.join("text_encoder").is_dir()
            && dir.join("vae").is_dir()
    };
    let Some(subdir) = subdir else {
        return holds_borrowed(snapshot).then(|| snapshot.to_path_buf());
    };
    let mut tiers = vec![subdir];
    for fallback in ["bf16", "q8", "q4"] {
        if !tiers.contains(&fallback) {
            tiers.push(fallback);
        }
    }
    tiers
        .into_iter()
        .map(|tier| snapshot.join(tier))
        .find(|dir| holds_borrowed(dir))
}

/// Native Rust/MLX LTX-2.3 weight converter (mlx-gen-ltx, sc-3224 engine + sc-3240 cutover). The LTX
/// path was never actually routed through the Rust job before — only `convert_wan` was ever shelled.
/// Splits the single-file checkpoint (`source_file`, e.g. eros `10Eros_v1_bf16.safetensors`),
/// sanitizes + Q4/Q8-quantizes the transformer, merges the latent upsampler from `upscaler_dir` (the
/// base `Lightricks/LTX-2.3` snapshot — the loader hard-requires `upsampler.safetensors`), and emits
/// the split MLX dir. `bits` is the reference `python -m mlx_video.convert --quantize --q-bits <bits>`
/// recipe (audio-inclusive). Runs MLX, so macOS-only.
#[cfg(target_os = "macos")]
fn convert_ltx_native(
    source_file: &Path,
    upscaler_dir: &Path,
    out_dir: &Path,
    bits: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    let opts = runtime_macos::providers::ltx::LtxConvertOpts::audio_quant(bits);
    runtime_macos::providers::ltx::convert_and_assemble(
        source_file,
        Some(upscaler_dir),
        out_dir,
        &opts,
    )
    .map(|_| ())
    .map_err(|error| error.to_string())
}

#[cfg(not(target_os = "macos"))]
fn convert_ltx_native(
    _source_file: &Path,
    _upscaler_dir: &Path,
    _out_dir: &Path,
    _bits: i32,
) -> Result<(), String> {
    Err("LTX-2.3 MLX conversion requires macOS (mlx-gen-ltx); the candle path serves other platforms."
        .to_owned())
}

/// Native Rust/MLX FLUX.2-dev pre-quantization convert (sc-5921, epic 5914). FLUX.2-dev is too large
/// to quantize in memory at load — the dense bf16 snapshot (60 GB DiT + 45 GB Mistral TE) peaks
/// ~105 GB > the 128 GB ceiling — so the install-time convert pre-quantizes the DiT + text encoder to
/// Q`bits` **on disk** (offline peak ~34.5 GB; MLX mmap-streams the source, sc-5917) and assembles a
/// packed snapshot: real packed `transformer/` + `text_encoder/` dirs (each one safetensors + a
/// `quantization` manifest in config.json) plus the unchanged VAE + tokenizer + `model_index.json`
/// symlinked from the gated source (the VAE is byte-identical to klein's). The worker then loads the
/// packed dir via the `modelPath` seam; the load-time `.quantize()` is a no-op on already-packed
/// weights. Absolute symlinks survive the worker's temp→final atomic rename, and `model_index.json`
/// doubles as the catalog's "converted" marker (`mlx_catalog_status` looks for a top-level
/// config.json/model_index.json). Runs MLX, so macOS-only.
#[cfg(target_os = "macos")]
fn convert_flux2_dev_prequant(
    source_dir: &Path,
    out_dir: &Path,
    bits: i32,
    group_size: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    runtime_macos::providers::flux2::quantize_flux2_dit(
        &source_dir.join("transformer"),
        &out_dir.join("transformer"),
        bits,
        group_size,
    )
    .map_err(|error| format!("quantize FLUX.2-dev transformer: {error}"))?;
    runtime_macos::providers::flux2::quantize_flux2_text_encoder_dir(
        &source_dir.join("text_encoder"),
        &out_dir.join("text_encoder"),
        bits,
        group_size,
    )
    .map_err(|error| format!("quantize FLUX.2-dev text encoder: {error}"))?;
    for sub in ["vae", "tokenizer", "model_index.json"] {
        let src = source_dir.join(sub);
        if !src.exists() {
            return Err(format!(
                "FLUX.2-dev source is missing `{sub}` — expected the gated diffusers snapshot of \
                 black-forest-labs/FLUX.2-dev (transformer/ text_encoder/ vae/ tokenizer/ \
                 model_index.json)."
            ));
        }
        let canonical = std::fs::canonicalize(&src).map_err(|error| error.to_string())?;
        std::os::unix::fs::symlink(&canonical, out_dir.join(sub))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn convert_flux2_dev_prequant(
    _source_dir: &Path,
    _out_dir: &Path,
    _bits: i32,
    _group_size: i32,
) -> Result<(), String> {
    Err(
        "FLUX.2-dev MLX tier conversion requires macOS (mlx-gen-flux2); model inference is also available through Candle/CUDA off-Mac."
            .to_owned(),
    )
}

/// SD3.5 gated diffusers snapshot → packed Q`bits` MLX dir (sc-7871). SD3.5's snapshot is
/// self-contained (its own transformer + triple text encoder + tokenizer + VAE), and the engine
/// loader reads the diffusers layout directly, reusing the SDXL CLIP encoders, the FLUX T5 encoder,
/// and the Z-Image 16-ch VAE — all loaded DENSE. So conversion pre-quantizes ONLY the MMDiT
/// `transformer/` on disk (the bulk: an 8B DiT) and symlinks the unchanged text encoders / tokenizers
/// / VAE / scheduler / model_index.json through. Absolute symlinks survive the worker's temp→final
/// atomic rename, and `model_index.json` doubles as the catalog's "converted" marker. `variant`
/// selects the MMDiT arch (Large/Turbo share one layout; Medium is the MMDiT-X dual-attention layout).
/// Runs MLX, so macOS-only.
#[cfg(target_os = "macos")]
fn convert_sd3_prequant(
    source_dir: &Path,
    out_dir: &Path,
    variant: Sd3Variant,
    bits: i32,
    group_size: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    let arch = match variant {
        Sd3Variant::Large | Sd3Variant::LargeTurbo => {
            runtime_macos::providers::sd3::Sd3Arch::large()
        }
        Sd3Variant::Medium => runtime_macos::providers::sd3::Sd3Arch::medium(),
    };
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let src_transformer = source_dir.join("transformer");
    if !src_transformer.is_dir() {
        return Err(
            "SD3.5 source is missing `transformer/` — expected the gated diffusers snapshot of \
             stabilityai/stable-diffusion-3.5-* (transformer/ text_encoder/ text_encoder_2/ \
             text_encoder_3/ tokenizer/ tokenizer_2/ tokenizer_3/ vae/ model_index.json)."
                .to_owned(),
        );
    }
    // Quantize the MMDiT transformer/ on disk (validates the arch first, writes a packed dir).
    runtime_macos::providers::sd3::quantize_sd3_dir(
        &arch,
        &src_transformer,
        &out_dir.join("transformer"),
        bits,
        group_size,
    )
    .map_err(|error| format!("quantize SD3.5 transformer: {error}"))?;
    // Symlink the dense (reused-as-is) text encoders / tokenizers / VAE / scheduler / model_index.json.
    for sub in [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "vae",
        "scheduler",
        "model_index.json",
    ] {
        let src = source_dir.join(sub);
        if !src.exists() {
            // scheduler/ is optional metadata for the engine (the pipeline uses a static flow-match
            // Euler schedule), so only the load-bearing modules are hard-required.
            if sub == "scheduler" {
                continue;
            }
            return Err(format!(
                "SD3.5 source is missing `{sub}` — expected the full gated diffusers snapshot of \
                 stabilityai/stable-diffusion-3.5-* (the triple text encoder + tokenizers + 16-ch VAE \
                 are reused dense)."
            ));
        }
        let canonical = std::fs::canonicalize(&src).map_err(|error| error.to_string())?;
        std::os::unix::fs::symlink(&canonical, out_dir.join(sub))
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn convert_sd3_prequant(
    _source_dir: &Path,
    _out_dir: &Path,
    _variant: Sd3Variant,
    _bits: i32,
    _group_size: i32,
) -> Result<(), String> {
    Err("SD3.5 MLX tier conversion requires macOS (mlx-gen-sd3); model inference is also available through Candle/CUDA off-Mac.".to_owned())
}

/// The THREE Anima variant DiTs (one per catalog id) under the ungated `circlestone-labs/Anima` repo,
/// as repo-relative paths. A per-variant install/convert pulls exactly ONE of these + the two shared
/// files below — three files, ~5.6 GB. Naming the bare repo would pull ~30 GB (it holds SEVEN full bf16
/// DiTs: these three plus three `preview*` cuts and `aesthetic-v1.0b`, all out of scope). The manifest
/// `downloads[].files` (sc-10523) MUST enumerate these exact paths and never the bare repo (the
/// "30 GB trap"); `resolve_convert_plan` also validates a convert's `sourceFile` is one of these so a
/// stray payload can't drag in an out-of-scope DiT.
const ANIMA_DIT_FILES: &[&str] = &[
    "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "split_files/diffusion_models/anima-aesthetic-v1.0.safetensors",
    "split_files/diffusion_models/anima-turbo-v1.0.safetensors",
];
/// The shared dense Qwen3-0.6B text encoder, reused bf16 in every Anima tier (repo-relative).
const ANIMA_TE_FILE: &str = sceneworks_core::portable_convert_sources::ANIMA_TE_FILE;
/// The shared dense Qwen-Image VAE, reused bf16 in every Anima tier (repo-relative).
const ANIMA_VAE_FILE: &str = sceneworks_core::portable_convert_sources::ANIMA_VAE_FILE;

/// Test-only hardcoded regression pin for the Anima DiT quant-target partition, mirroring
/// `runtime_macos::providers::anima::is_dit_quant_target` (`true` UNLESS the base — a key minus its `.weight` suffix — is
/// inside the bundled `AnimaTextConditioner`, `…llm_adapter.*`, kept dense bf16). The converter itself
/// no longer uses this: `convert_anima_prequant` calls `runtime_macos::providers::anima::quantize_anima_dit` directly
/// (sc-10580 dedup — single source of the quant policy is the crate). This copy survives ONLY so
/// `anima_dit_quant_predicate_matches_mlx_gen_anima_twin` can pin the exact partition against a
/// hand-written table and — on macOS, where the crate is linked — assert the real crate predicate still
/// matches it, so a future mlx-gen rev can't silently diverge. **Prefix-agnostic**: the base cut roots
/// the DiT at `net` (base) and `model.diffusion_model` (turbo/aesthetic), so a hardcoded
/// `net.llm_adapter.` predicate would leave the `model.diffusion_model.llm_adapter.*` conditioner
/// UNPROTECTED and pack it to Q4 — `!contains` catches all roots.
#[cfg(test)]
fn anima_is_dit_quant_target(base: &str) -> bool {
    !base.contains("llm_adapter.")
}

/// Anima `split_files/` source → the local convert-at-install MLX artifact with the full q4/q8/bf16
/// MATRIX (epic 10512, sc-10517). Anima is **convert-at-install**: SceneWorks never redistributes
/// converted weights, so all three tiers are produced ON-DEVICE from the ungated source in ONE job (the
/// 2B DiT is small; packing is quick). Unlike the single-tier converters above, this writes THREE tier
/// subdirs — `bf16/ q8/ q4/` — each a `diffusion_models/ text_encoders/ vae/` tree the Anima loader
/// reads; the bespoke `anima_tier_subdir` resolver (image_jobs/base.rs) picks one at gen time by
/// `mlxQuantize`. Only the Cosmos DiT is packed (via `runtime_macos::providers::anima::quantize_anima_dit`, whose
/// `is_dit_quant_target` predicate keeps the conditioner dense); the bundled
/// `AnimaTextConditioner`, the Qwen3-0.6B TE, and the Qwen-Image VAE stay DENSE bf16 in every tier (the
/// "dense-TE, transformer-only" quant policy the sibling converters share — a Q4 text-conditioning path
/// degrades semantics). A top-level `model_index.json` marks the finished artifact so
/// `mlx_catalog_status` injects `modelPath` (the converted ROOT holding the tier subdirs). The dense
/// bf16 DiT + TE + VAE are ABSOLUTE symlinks (they survive the temp→final atomic rename). Runs MLX, so
/// macOS-only.
#[cfg(target_os = "macos")]
fn convert_anima_prequant(
    dit_source: &Path,
    te_source: &Path,
    vae_source: &Path,
    dit_filename: &str,
    out_dir: &Path,
    group_size: i32,
) -> Result<(), String> {
    // CARVE-OUT(epic 3720): backend-specific weight converter; not a registry contract.
    for source in [dit_source, te_source, vae_source] {
        if !source.is_file() {
            return Err(format!(
                "Anima source is missing `{}` — expected the ungated circlestone-labs/Anima \
                 split_files/ (diffusion_models/<variant>.safetensors + the shared \
                 text_encoders/qwen_3_06b_base.safetensors + vae/qwen_image_vae.safetensors).",
                source.display()
            ));
        }
    }
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    // (subdir, None => dense bf16 passthrough | Some(bits) => pack the Cosmos DiT).
    for (tier, bits) in [("bf16", None), ("q8", Some(8)), ("q4", Some(4))] {
        let tier_dir = out_dir.join(tier);
        for sub in ["diffusion_models", "text_encoders", "vae"] {
            std::fs::create_dir_all(tier_dir.join(sub)).map_err(|error| error.to_string())?;
        }
        let dit_dst = tier_dir.join("diffusion_models").join(dit_filename);
        match bits {
            // Dense bf16: symlink the source DiT unchanged (absolute → survives the atomic rename).
            None => {
                let canonical =
                    std::fs::canonicalize(dit_source).map_err(|error| error.to_string())?;
                std::os::unix::fs::symlink(&canonical, &dit_dst)
                    .map_err(|error| error.to_string())?;
            }
            // Packed: quantize ONLY the Cosmos DiT (the bundled conditioner stays dense) via the
            // crate's install-time quantizer — the single source of the quant policy (sc-10580).
            Some(bits) => runtime_macos::providers::anima::quantize_anima_dit(
                dit_source, &dit_dst, bits, group_size,
            )
            .map_err(|error| format!("quantize Anima DiT: {error}"))?,
        }
        // Symlink the shared dense TE + VAE into every tier (absolute targets).
        for (source, sub, name) in [
            (te_source, "text_encoders", "qwen_3_06b_base.safetensors"),
            (vae_source, "vae", "qwen_image_vae.safetensors"),
        ] {
            let canonical = std::fs::canonicalize(source).map_err(|error| error.to_string())?;
            std::os::unix::fs::symlink(&canonical, tier_dir.join(sub).join(name))
                .map_err(|error| error.to_string())?;
        }
    }
    // Diffusers-style marker so `mlx_catalog_status` treats the tree as a finished MLX artifact (it
    // checks for `model_index.json` OR `config.json` at the converted root) and injects `modelPath`.
    std::fs::write(
        out_dir.join("model_index.json"),
        "{\n  \"_class_name\": \"AnimaPipeline\"\n}\n",
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn convert_anima_prequant(
    _dit_source: &Path,
    _te_source: &Path,
    _vae_source: &Path,
    _dit_filename: &str,
    _out_dir: &Path,
    _group_size: i32,
) -> Result<(), String> {
    Err(
        "Anima MLX tier conversion requires macOS (mlx-gen quantization); model inference is also available through Candle/CUDA off-Mac."
            .to_owned(),
    )
}

/// The one upstream file the Supra2-IMG convert reads (`gen_core_checkpoint::supra2::SUPRA2_SOURCE`); the
/// manifest's `mlx.convertSourceFile` and its primary `downloads[].files` name exactly this file.
const SUPRA2_SOURCE_FILE: &str = "model_final_ema.pt";

/// Diffusers-style marker written beside the Supra2 bundle so `mlx_catalog_status` (which looks for a
/// top-level `config.json` / `model_index.json`) reports the converted artifact as installed and
/// injects its `modelPath`. Same mechanism as the Anima converter's marker; the engine itself reads
/// only `supra2_bundle.json` and the files it lists.
const SUPRA2_MARKER: &str = "{\n  \"_class_name\": \"Supra2ImgBundle\"\n}\n";

/// Supra2-IMG convert-at-install (sc-24213, epic 24212). Everything substantive is inference-owned
/// and backend-neutral (`gen_core_checkpoint::supra2`): the two co-requisite snapshots are re-verified against
/// their pinned sizes + SHA-256 (`FLAN_T5_BASE` / `SD_VAE`), then `convert_pinned_checkpoint`
/// verifies the pinned `model_final_ema.pt` SHA-256, validates the exact state dict and writes the
/// native bundle (`denoiser.safetensors`, optional `conditioning.safetensors`, and
/// `supra2_bundle.json` last), and `read_pinned_bundle` re-reads it under the provider's pin
/// contract (pinned denoiser + conditioning digests). SceneWorks re-implements none of it; it only
/// surfaces the failure.
/// The marker goes in after the bundle, so a failed convert never leaves a dir the catalog reads as
/// installed (the caller also converts into a temp sibling and promotes it atomically).
pub(crate) fn convert_supra2_bundle(
    source_file: &Path,
    text_encoder_dir: &Path,
    vae_dir: &Path,
    out_dir: &Path,
) -> Result<(), String> {
    use gen_core_checkpoint::supra2::{
        convert_pinned_checkpoint, read_pinned_bundle, BundleCheck, FLAN_T5_BASE, SD_VAE,
    };
    FLAN_T5_BASE
        .verify(text_encoder_dir)
        .map_err(|error| format!("Flan-T5-Base text encoder failed verification: {error}"))?;
    SD_VAE
        .verify(vae_dir)
        .map_err(|error| format!("SD VAE ft-MSE failed verification: {error}"))?;
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    convert_pinned_checkpoint(source_file, out_dir)
        .map_err(|error| format!("Supra2-IMG checkpoint conversion failed: {error}"))?;
    // Hold the written bundle to the provider's own pin contract (source identity, geometry,
    // component revisions and the pinned denoiser/conditioning digests) before marking it
    // installed, so a bundle the engine would refuse at load never reads as installed.
    read_pinned_bundle(out_dir, BundleCheck::Digests)
        .map_err(|error| format!("Supra2-IMG bundle failed its pin contract: {error}"))?;
    std::fs::write(out_dir.join("model_index.json"), SUPRA2_MARKER)
        .map_err(|error| error.to_string())
}

/// The pinned co-requisite snapshot a Supra2 convert verifies: the exact `snapshots/<revision>/`
/// the catalog row pins (never `refs/main`), or a typed refusal naming the install to make.
fn supra2_component_snapshot(
    settings: &Settings,
    model_id: &str,
    label: &str,
    repo: &str,
    revision: &str,
) -> Result<PathBuf, ConvertPlanError> {
    huggingface_pinned_snapshot_dir(&settings.data_dir, repo, revision).ok_or_else(|| {
        ConvertPlanError {
            message: "Supra2-IMG components are not installed.",
            detail: format!(
                "The {label} ({repo}@{revision}) is not in the local cache. Install {model_id} from \
                 the Model Manager (it downloads this component) before converting."
            ),
        }
    })
}

/// The base `Lightricks/LTX-2.3` latent upsampler the LTX loader hard-requires (emitted as
/// `upsampler.safetensors` in the converted dir). Neither the eros nor the base single-file
/// checkpoint bundles it, so the converter merges it from the base repo at convert time.
const LTX_SPATIAL_UPSCALER_FILE: &str = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors";

/// Ensure the LTX-2.3 spatial upsampler `file` from `repo` is in the Hugging Face cache — fetching
/// just that file on demand if missing — and return the repo's snapshot dir (the converter's
/// `upscaler_dir`). Mirrors how the torch adapter pulls its spatial upscaler at generation time, so
/// converting eros does not require a full ~157 GB base-LTX install. Returns `None` if the file
/// cannot be obtained (HF CLI unavailable and not already cached) so the caller surfaces a clear
/// failure. The fetch uses a scratch marker dir so its install marker never lands in the model tree.
async fn ensure_ltx_upscaler_cached(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    repo: &str,
    file: &str,
) -> WorkerResult<Option<PathBuf>> {
    if let Some(snapshot) = huggingface_snapshot_dir(&settings.data_dir, repo) {
        if snapshot.join(file).exists() {
            return Ok(Some(snapshot));
        }
    }
    let files = vec![file.to_owned()];
    let snapshot = ensure_hf_files_cached(api, settings, job, repo, "main", &files).await?;
    Ok(snapshot.filter(|dir| dir.join(file).exists()))
}

/// Resolved native conversion plan for [`run_model_convert_job`] (sc-3240). The Python
/// `mlx_video.convert_wan` subprocess (and its mlx-video venv) is gone — every convert-required
/// model maps to exactly one native converter, keyed by the manifest `mlx.converter` discriminator.
enum ConvertPlan {
    /// FLUX.2-klein single-file fine-tune → diffusers dir (sc-3136), borrowing VAE/TE/tokenizer.
    Flux2 {
        source_file: PathBuf,
        base_dir: PathBuf,
    },
    /// Single-file LTX-2.3 → split MLX dir; `upscaler_dir` carries the loader-required upsampler.
    Ltx {
        source_file: PathBuf,
        upscaler_dir: PathBuf,
        bits: i32,
    },
    /// FLUX.2-dev gated diffusers snapshot → packed Q`bits` dir (sc-5921): pre-quantize the DiT +
    /// Mistral text encoder on disk, symlink the unchanged VAE/tokenizer/model_index.json.
    Flux2Dev {
        source_dir: PathBuf,
        bits: i32,
        group_size: i32,
    },
    /// SD3.5 gated diffusers snapshot → packed Q`bits` dir (sc-7871): pre-quantize ONLY the MMDiT
    /// `transformer/` on disk (the triple text encoder + VAE are reused dense and symlinked through
    /// unchanged). `variant` selects the MMDiT arch (Large/Turbo share one layout; Medium is MMDiT-X).
    Sd3 {
        source_dir: PathBuf,
        variant: Sd3Variant,
        bits: i32,
        group_size: i32,
    },
    /// Anima `split_files/` source → the local q4/q8/bf16 MATRIX in ONE job (sc-10517, epic 10512):
    /// pack the variant's Cosmos DiT at q8 + q4 and passthrough the dense bf16 DiT, symlinking the
    /// shared dense Qwen3 TE + Qwen-Image VAE into all three tier subdirs. Convert-at-install (never
    /// redistributed); the bespoke `anima_tier_subdir` resolver picks a tier at gen time. `dit_source`
    /// is the variant DiT (payload `sourceFile`); `dit_filename` is preserved so the loader's
    /// `Variant::dit_filename()` resolves it.
    Anima {
        dit_source: PathBuf,
        te_source: PathBuf,
        vae_source: PathBuf,
        dit_filename: String,
        group_size: i32,
    },
    /// Supra2-IMG `model_final_ema.pt` -> the native bundle (sc-24213), after re-verifying the two
    /// pinned co-requisite snapshots it is loaded beside (`text_encoder`, `vae`).
    Supra2 {
        source_file: PathBuf,
        text_encoder_dir: PathBuf,
        vae_dir: PathBuf,
    },
}

/// A user-facing reason a conversion can't proceed (missing checkpoint, unknown converter, …). The
/// plan resolver returns this instead of calling `fail_job` itself, so the resolution phase names the
/// failure condition once and [`run_model_convert_job`] performs the single `fail_job` + early return
/// — replacing the eight scattered `fail_job(...).await?; return Ok(())` pairs the resolver used to
/// interleave with plan construction (sc-8921, F-119).
struct ConvertPlanError {
    /// The short `fail_job` message (the failure headline).
    message: &'static str,
    /// The `fail_job` detail (the actionable explanation), always present for these conditions.
    detail: String,
}

/// The SD3.5 MMDiT variant a `sd3_5_*_quant` conversion targets — a target-neutral mirror of
/// `runtime_macos::providers::sd3::Sd3Variant` (which is macOS-only), so [`ConvertPlan`] stays buildable on every
/// target. The macOS converter maps it back to the engine variant in [`convert_sd3_prequant`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sd3Variant {
    /// `sd3_5_large_quant` — SD3.5 Large (8B MMDiT, true-CFG).
    Large,
    /// `sd3_5_large_turbo_quant` — SD3.5 Large Turbo (same MMDiT layout, ADD-distilled).
    LargeTurbo,
    /// `sd3_5_medium_quant` — SD3.5 Medium (2.5B MMDiT-X, dual-attention first 13 blocks).
    Medium,
}

fn unknown_converter_error(model_id: &str, converter: &str) -> ConvertPlanError {
    ConvertPlanError {
        message: "Unknown MLX converter.",
        detail: format!("Unrecognized mlx.converter '{converter}' for {model_id}."),
    }
}

/// Resolve the native [`ConvertPlan`] for a convert job from its `converter` discriminator, or a
/// [`ConvertPlanError`] naming why it can't proceed (missing source file, base model not installed,
/// unknown converter, …). Split out of [`run_model_convert_job`] (sc-8921, F-119): the converter-arm
/// match used to interleave eight `fail_job(...).await?; return Ok(())` side-effects with plan
/// construction inside the 350-line job body; here each becomes an `Err(ConvertPlanError)` value and
/// the caller performs the SINGLE `fail_job`. Still async because the LTX arm may fetch the upscaler
/// (`ensure_ltx_upscaler_cached`), and real infra errors (`?`) propagate as the outer `WorkerError`.
async fn resolve_convert_plan(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
    checkpoint_dir: &Path,
    quantize_bits: Option<u64>,
) -> WorkerResult<Result<ConvertPlan, ConvertPlanError>> {
    // Re-read the payload discriminators the caller already validated (cheap, and keeps this resolver's
    // arg list small): `modelId`/`sourceRepo` are required, `converter` optional (empty = the "no
    // converter configured" failure arm).
    let model_id = required_payload_string(&job.payload, "modelId")?;
    let source_repo = required_payload_string(&job.payload, "sourceRepo")?;
    let converter = optional_payload_string(&job.payload, "converter").unwrap_or_default();
    // Reject any non-empty converter that is not in the shared registry up front, so this resolver's
    // recognized set is derived from `NATIVE_CONVERTERS` — the same const the convert-gap gate derives
    // its allow-list from. This guarantees the gate can never be stricter than the set of arms below,
    // the exact drift that shipped as sc-10573 (`ltx_video` handled here but missing from the gate).
    // The empty ("" = no converter configured) case still falls through to its own arm below.
    if !converter.is_empty() && !NATIVE_CONVERTERS.contains(&converter) {
        return Ok(Err(unknown_converter_error(model_id, converter)));
    }
    let plan = match converter {
        "flux2_klein_diffusers" => {
            let source_file_name = required_payload_string(&job.payload, "sourceFile")?.to_owned();
            let base_repo = required_payload_string(&job.payload, "baseRepo")?.to_owned();
            let source_file = checkpoint_dir.join(&source_file_name);
            if !source_file.is_file() {
                return Ok(Err(ConvertPlanError {
                    message: "Converted-model source file is missing.",
                    detail: format!("Expected {source_file_name} in {source_repo}."),
                }));
            }
            let Some(base_snapshot) = huggingface_snapshot_dir(&settings.data_dir, &base_repo)
            else {
                return Ok(Err(ConvertPlanError {
                    message: "Base FLUX.2-klein model is not installed.",
                    detail: format!(
                        "Install {base_repo} before converting {model_id} — its VAE, text encoder, \
                         and tokenizer are reused."
                    ),
                }));
            };
            // The re-host keeps each quant tier in its own subdir (bf16/, q8/, q4/); the borrowed
            // VAE/text-encoder/tokenizer are dense and identical across tiers, so the manifest's
            // `convertBaseSubdir` names the preferred tier and any installed tier serves. A base with a
            // plain root diffusers layout (no `baseSubdir`) borrows from the snapshot root (sc-14978).
            let base_subdir = optional_payload_string(&job.payload, "baseSubdir");
            let Some(base_dir) = resolve_flux2_klein_base_dir(&base_snapshot, base_subdir) else {
                return Ok(Err(ConvertPlanError {
                    message: "Base FLUX.2-klein model is installed but incomplete.",
                    detail: format!(
                        "No base tier with a VAE, text encoder, and tokenizer was found under \
                         {base_repo} for converting {model_id}; re-download the base model."
                    ),
                }));
            };
            ConvertPlan::Flux2 {
                source_file,
                base_dir,
            }
        }
        "ltx_video" => {
            let source_file_name = required_payload_string(&job.payload, "sourceFile")?.to_owned();
            let source_file = checkpoint_dir.join(&source_file_name);
            if !source_file.is_file() {
                return Ok(Err(ConvertPlanError {
                    message: "LTX-2.3 source checkpoint file is missing.",
                    detail: format!("Expected {source_file_name} in {source_repo}."),
                }));
            }
            let upscaler_repo = required_payload_string(&job.payload, "baseRepo")?.to_owned();
            let upscaler_file = optional_payload_string(&job.payload, "upscalerFile")
                .unwrap_or(LTX_SPATIAL_UPSCALER_FILE)
                .to_owned();
            let Some(upscaler_dir) =
                ensure_ltx_upscaler_cached(api, settings, job, &upscaler_repo, &upscaler_file)
                    .await?
            else {
                return Ok(Err(ConvertPlanError {
                    message: "LTX-2.3 spatial upscaler is unavailable.",
                    detail: format!(
                        "Could not obtain {upscaler_file} from {upscaler_repo}; install the base \
                         LTX-2.3 model or check connectivity before converting {model_id}."
                    ),
                }));
            };
            // Default to the reference Q4 recipe when the manifest/request specifies no bits.
            let bits = quantize_bits.map_or(4, |bits| bits as i32);
            ConvertPlan::Ltx {
                source_file,
                upscaler_dir,
                bits,
            }
        }
        "flux2_dev_quant" => {
            // FLUX.2-dev is self-contained (its own VAE/tokenizer/TE in the snapshot), so the whole
            // gated diffusers snapshot dir is the convert source — no single source file, no base
            // repo. Q4 is mandatory (in-app Q4 load of the dense bf16 would peak ~105 GB), so default
            // to Q4 / group-size 64 when the request omits them.
            let bits = quantize_bits.map_or(4, |bits| bits as i32);
            let group_size = job
                .payload
                .get("quantizeGroupSize")
                .and_then(Value::as_u64)
                .map_or(64, |group| group as i32);
            ConvertPlan::Flux2Dev {
                source_dir: checkpoint_dir.to_path_buf(),
                bits,
                group_size,
            }
        }
        // SD3.5 (epic 7841 / sc-7871) — the gated diffusers snapshot is self-contained (its own
        // transformer/ + triple text encoder + tokenizer + VAE), so the whole snapshot dir is the
        // convert source: pre-quantize ONLY the MMDiT transformer/ on disk and symlink the dense TE /
        // VAE / tokenizer / scheduler / model_index.json through. Default Q8 / group-size 64 (the
        // manifest `quantize: 8`, reference group size) when the request omits them. The converter id
        // selects the MMDiT arch variant.
        converter @ ("sd3_5_large_quant" | "sd3_5_large_turbo_quant" | "sd3_5_medium_quant") => {
            let bits = quantize_bits.map_or(8, |bits| bits as i32);
            let group_size = job
                .payload
                .get("quantizeGroupSize")
                .and_then(Value::as_u64)
                .map_or(64, |group| group as i32);
            let variant = match converter {
                "sd3_5_large_quant" => Sd3Variant::Large,
                "sd3_5_large_turbo_quant" => Sd3Variant::LargeTurbo,
                _ => Sd3Variant::Medium,
            };
            ConvertPlan::Sd3 {
                source_dir: checkpoint_dir.to_path_buf(),
                variant,
                bits,
                group_size,
            }
        }
        // Anima (epic 10512, sc-10517) — convert-at-install with a q4/q8/bf16 MATRIX built in ONE job
        // from the ungated circlestone-labs/Anima `split_files/`. `sourceFile` (= manifest
        // `convertSourceFile`) is the variant's DiT repo-relative path; the Qwen3 TE + Qwen-Image VAE
        // are the shared files reused dense in every tier. `quantizeBits` is IGNORED (the converter
        // always emits all three tiers); the default group size is 64. Validate `sourceFile` is one of
        // the three shipped DiTs so a stray payload can't pull an out-of-scope cut (preview*/aesthetic-b).
        "anima_quant" => {
            let source_file_name = required_payload_string(&job.payload, "sourceFile")?.to_owned();
            if !ANIMA_DIT_FILES.contains(&source_file_name.as_str()) {
                return Ok(Err(ConvertPlanError {
                    message: "Unknown Anima source DiT.",
                    detail: format!(
                        "sourceFile '{source_file_name}' is not one of the three shipped Anima \
                         variant DiTs (the preview cuts and aesthetic-v1.0b are out of scope)."
                    ),
                }));
            }
            let group_size = job
                .payload
                .get("quantizeGroupSize")
                .and_then(Value::as_u64)
                .map_or(64, |group| group as i32);
            let dit_source = checkpoint_dir.join(&source_file_name);
            let dit_filename = Path::new(&source_file_name)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&source_file_name)
                .to_owned();
            if !dit_source.is_file() {
                return Ok(Err(ConvertPlanError {
                    message: "Anima source DiT is missing.",
                    detail: format!("Expected {source_file_name} in {source_repo}."),
                }));
            }
            let te_source = checkpoint_dir.join(ANIMA_TE_FILE);
            let vae_source = checkpoint_dir.join(ANIMA_VAE_FILE);
            if !te_source.is_file() || !vae_source.is_file() {
                return Ok(Err(ConvertPlanError {
                    message: "Anima shared components are missing.",
                    detail: format!(
                        "Expected {ANIMA_TE_FILE} and {ANIMA_VAE_FILE} in {source_repo} (the shared \
                         dense Qwen3 text encoder + Qwen-Image VAE reused in every tier)."
                    ),
                }));
            }
            ConvertPlan::Anima {
                dit_source,
                te_source,
                vae_source,
                dit_filename,
                group_size,
            }
        }
        // Supra2-IMG (epic 24212, sc-24213) — the pinned upstream `.pt` is the only source file;
        // the Flan-T5-Base and SD VAE ft-MSE co-requisites are the exact pinned snapshots the
        // engine loads beside the bundle, verified here at install time so generation stays offline.
        "supra2_img" => {
            let source_file_name = required_payload_string(&job.payload, "sourceFile")?.to_owned();
            if source_file_name != SUPRA2_SOURCE_FILE {
                return Ok(Err(ConvertPlanError {
                    message: "Unknown Supra2-IMG source checkpoint.",
                    detail: format!(
                        "sourceFile '{source_file_name}' is not the pinned Supra2-IMG checkpoint \
                         ({SUPRA2_SOURCE_FILE})."
                    ),
                }));
            }
            let source_file = checkpoint_dir.join(SUPRA2_SOURCE_FILE);
            if !source_file.is_file() {
                return Ok(Err(ConvertPlanError {
                    message: "Supra2-IMG source checkpoint is missing.",
                    detail: format!("Expected {SUPRA2_SOURCE_FILE} in {source_repo}."),
                }));
            }
            let text_encoder = gen_core_checkpoint::supra2::FLAN_T5_BASE;
            let vae = gen_core_checkpoint::supra2::SD_VAE;
            let text_encoder_dir = match supra2_component_snapshot(
                settings,
                model_id,
                "Flan-T5-Base text encoder",
                text_encoder.repo,
                text_encoder.revision,
            ) {
                Ok(dir) => dir,
                Err(error) => return Ok(Err(error)),
            };
            let vae_dir = match supra2_component_snapshot(
                settings,
                model_id,
                "SD VAE ft-MSE decoder",
                vae.repo,
                vae.revision,
            ) {
                Ok(dir) => dir,
                Err(error) => return Ok(Err(error)),
            };
            ConvertPlan::Supra2 {
                source_file,
                text_encoder_dir,
                vae_dir,
            }
        }
        "" => {
            return Ok(Err(ConvertPlanError {
                message: "No MLX converter is configured for this model.",
                detail: format!(
                    "{model_id} sets mlx.requiresConversion but no mlx.converter; the legacy \
                     converter was retired (sc-3240)."
                ),
            }));
        }
        other => {
            // The registry precheck above makes this defensive arm unreachable for production
            // inputs. Keep one shared error constructor so the guard and fallback cannot drift.
            return Ok(Err(unknown_converter_error(model_id, other)));
        }
    };
    Ok(Ok(plan))
}

/// Convert a model's native checkpoint into the local MLX format on macOS/Apple Silicon, fully
/// in-process via the linked `mlx-gen-*` converters (epic 2337). The native checkpoint must already
/// be downloaded into the Hugging Face cache (via a model_download job). The converter is selected by
/// the manifest `mlx.converter` discriminator: `flux2_klein_diffusers` (sc-3136), `ltx_video`
/// (mlx-gen-ltx), or `flux2_dev_quant` (FLUX.2-dev pre-quantization, sc-5921) — the models that
/// install via in-app conversion, plus `sd3_5_large_quant` / `sd3_5_large_turbo_quant` /
/// `sd3_5_medium_quant` (SD3.5 transformer pre-quantization, sc-7871). The Python
/// `mlx_video.convert_wan` subprocess + `SCENEWORKS_PYTHON` wiring were retired here (sc-3240); the
/// Wan2.2 converters were decommissioned once those models flipped to pre-converted SceneWorks
/// downloads (sc-5603, epic 5594).
///
/// Real conversion is exercised on Mac hardware via the `#[ignore]` real-weight tests below; this
/// wires the tracked job, progress, cancellation, and failure surfacing.
/// Install-time provisioning of the eros Gemma-3 text encoder after a successful LTX conversion
/// (macOS): surfaces a progress line, then pulls the bundle `gemma/` via the shared provisioner so a
/// fresh eros install is self-contained. A no-op off macOS (the LTX engine + its gemma provisioning
/// are mlx-only; the candle lane resolves its own `google/gemma-3-12b-it`).
#[cfg(target_os = "macos")]
async fn provision_ltx_eros_gemma(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.9,
            "Fetching the Gemma-3 text encoder. This can take several minutes.",
            None,
            None,
            None,
        ),
    )
    .await?;
    crate::video_jobs::ltx::ensure_ltx_bundle_gemma_present(api, settings, job).await
}

#[cfg(not(target_os = "macos"))]
async fn provision_ltx_eros_gemma(
    _api: &ApiClient,
    _settings: &Settings,
    _job: &JobSnapshot,
) -> WorkerResult<()> {
    Ok(())
}

pub(crate) async fn run_model_convert_job(
    api: &ApiClient,
    settings: &Settings,
    job: &JobSnapshot,
) -> WorkerResult<()> {
    let model_id = required_payload_string(&job.payload, "modelId")?.to_owned();
    let source_repo = required_payload_string(&job.payload, "sourceRepo")?.to_owned();
    let output_dir = required_payload_string(&job.payload, "outputDir")?.to_owned();
    // Validate the final destination before heartbeat/status mutation, cancellation checks,
    // checkpoint lookup, plan resolution, or any converter-specific provisioning. A malformed raw
    // job must fail at the worker trust boundary without touching job state or local model storage.
    let final_dir = resolve_model_convert_output(&settings.data_dir, &output_dir)?;
    // NOTE: there is intentionally no `dtype` knob. The native converters each produce a fixed
    // precision (the FLUX.2-klein converter is bf16-only; LTX / FLUX.2-dev honor `quantizeBits`),
    // so a client-supplied `dtype` had no converter to receive it and only ever appeared in the
    // progress message — claiming e.g. a float16 conversion that never happened (F-117). Precision
    // is controlled by `quantizeBits` below.
    // Optional MLX quantization. `quantizeOnly` quantizes an already-converted bf16
    // MLX dir (turnkey models); otherwise quantization rides on the native->MLX
    // conversion. `bits` is validated by the convert tool's choices (LTX honors it; the
    // FLUX.2-klein converter is bf16-only and ignores it).
    let quantize_only = payload_bool(&job.payload, "quantizeOnly");
    let quantize_bits = job.payload.get("quantizeBits").and_then(Value::as_u64);

    // The active backend label (mlx on macOS, candle off-Mac, cpu fallback) so the user-facing
    // conversion progress / failure / completion / cancel strings name the REAL backend instead of a
    // hardcoded "MLX" (sc-9801 fixed the cancel strings; sc-9863 extended it to the remaining
    // progress/failure/completion strings — the F-114 follow-up to sc-8916). Same
    // `backend_label(&settings.gpu_id)` mechanism the caption / training paths use.
    let backend = backend_label(&settings.gpu_id).to_uppercase();

    heartbeat(api, settings, WorkerStatus::Busy, Some(&job.id)).await?;
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Preparing,
            ProgressStage::Preparing,
            0.05,
            &format!("Preparing {backend} conversion for {model_id}."),
            None,
            None,
            None,
        ),
    )
    .await?;
    check_cancel(
        api,
        &job.id,
        &format!("{backend} conversion canceled before it started."),
    )
    .await?;

    let Some(checkpoint_dir) = huggingface_snapshot_dir(&settings.data_dir, &source_repo) else {
        fail_job(
            api,
            &job.id,
            "Native checkpoint is not downloaded.",
            Some(format!(
                "Download {source_repo} before converting it to {backend}."
            )),
        )
        .await?;
        return Ok(());
    };

    // Quantize-only (re-quantize a pre-converted turnkey bf16 MLX dir) was a capability of the
    // Python `convert_wan --quantize-only` with no native equivalent; it is unreachable from the UI
    // and superseded by native-conversion-with-quant. Surface it explicitly rather than silently
    // promoting an unconverted dir.
    if quantize_only {
        fail_job(
            api,
            &job.id,
            &format!("Quantize-only {backend} conversion is no longer supported."),
            Some(
                "In-place re-quantization of a pre-converted MLX model was removed with the legacy \
                 mlx-video converter (sc-3240). Convert the native checkpoint with quantization \
                 instead."
                    .to_owned(),
            ),
        )
        .await?;
        return Ok(());
    }

    // Converter discriminator (sc-2235 / sc-3224 / sc-3240). Every convert-required model declares one
    // in its manifest `mlx.converter` (flux2_klein_diffusers / ltx_video / flux2_dev_quant / the SD3.5
    // quant variants); there is NO Python fallback — the `mlx_video.convert_wan` subprocess was retired
    // at this cutover. `resolve_convert_plan` reads the discriminator + source repo from the payload.
    let plan =
        match resolve_convert_plan(api, settings, job, &checkpoint_dir, quantize_bits).await? {
            Ok(plan) => plan,
            Err(ConvertPlanError { message, detail }) => {
                fail_job(api, &job.id, message, Some(detail)).await?;
                return Ok(());
            }
        };

    // Convert into a unique temp sibling and only promote it on success, so a
    // canceled/failed conversion never leaves a partial directory that the catalog
    // and adapter would treat as a ready model (convert tools write config.json
    // before all weight shards).
    let parent = final_dir
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tokio::fs::create_dir_all(&parent).await?;
    let temp_dir = parent.join(format!(
        ".{}.converting-{}",
        final_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("mlx"),
        job.id
    ));
    let _ = tokio::fs::remove_dir_all(&temp_dir).await;

    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Running,
            ProgressStage::Running,
            0.2,
            &format!("Converting {model_id} to {backend}. This can take several minutes."),
            None,
            None,
            None,
        ),
    )
    .await?;

    // The native MLX converters are blocking and not interruptible mid-run (minutes on real
    // weights), so honor cancel up front and run on a blocking thread. On any failure the partial
    // temp dir is removed so it can never be promoted by the atomic rename below. The Flux2 path's
    // borrowed vae/text-encoder/tokenizer are absolute symlinks that survive the rename.
    let convert_cancel_message = format!("{backend} conversion canceled by user.");
    check_cancel(api, &job.id, &convert_cancel_message).await?;
    let temp = temp_dir.clone();
    // An LTX conversion is always the eros fine-tune (the base model installs as a pre-converted
    // turnkey bundle and never hits this path), and its bare checkpoint ships no bundled Gemma-3 text
    // encoder — provision the bundle `gemma/` after promotion so the install is self-contained.
    let plan_is_ltx = matches!(plan, ConvertPlan::Ltx { .. });
    // Each native converter maps its `Result<(), String>` into a `WorkerResult<()>` so the whole
    // convert can ride `run_blocking_with_heartbeat` below — the failure detail keeps the same
    // "{backend} conversion failed. {detail}" shape the old inline match produced.
    let convert_task = match plan {
        ConvertPlan::Flux2 {
            source_file,
            base_dir,
        } => {
            let backend = backend.clone();
            crate::load_progress::spawn_blocking(move || {
                convert_flux2_klein_diffusers(&source_file, &base_dir, &temp).map_err(|detail| {
                    WorkerError::Engine(format!("{backend} conversion failed. {detail}"))
                })
            })
        }
        ConvertPlan::Ltx {
            source_file,
            upscaler_dir,
            bits,
        } => {
            let backend = backend.clone();
            crate::load_progress::spawn_blocking(move || {
                convert_ltx_native(&source_file, &upscaler_dir, &temp, bits).map_err(|detail| {
                    WorkerError::Engine(format!("{backend} conversion failed. {detail}"))
                })
            })
        }
        ConvertPlan::Flux2Dev {
            source_dir,
            bits,
            group_size,
        } => {
            let backend = backend.clone();
            crate::load_progress::spawn_blocking(move || {
                convert_flux2_dev_prequant(&source_dir, &temp, bits, group_size).map_err(|detail| {
                    WorkerError::Engine(format!("{backend} conversion failed. {detail}"))
                })
            })
        }
        ConvertPlan::Sd3 {
            source_dir,
            variant,
            bits,
            group_size,
        } => {
            let backend = backend.clone();
            crate::load_progress::spawn_blocking(move || {
                convert_sd3_prequant(&source_dir, &temp, variant, bits, group_size).map_err(
                    |detail| WorkerError::Engine(format!("{backend} conversion failed. {detail}")),
                )
            })
        }
        ConvertPlan::Anima {
            dit_source,
            te_source,
            vae_source,
            dit_filename,
            group_size,
        } => {
            let backend = backend.clone();
            crate::load_progress::spawn_blocking(move || {
                convert_anima_prequant(
                    &dit_source,
                    &te_source,
                    &vae_source,
                    &dit_filename,
                    &temp,
                    group_size,
                )
                .map_err(|detail| {
                    WorkerError::Engine(format!("{backend} conversion failed. {detail}"))
                })
            })
        }
        ConvertPlan::Supra2 {
            source_file,
            text_encoder_dir,
            vae_dir,
        } => {
            let backend = backend.clone();
            crate::load_progress::spawn_blocking(move || {
                convert_supra2_bundle(&source_file, &text_encoder_dir, &vae_dir, &temp).map_err(
                    |detail| WorkerError::Engine(format!("{backend} conversion failed. {detail}")),
                )
            })
        }
    };
    // The native converters are multi-minute blocking phases with no internal heartbeat loop, so a
    // large conversion used to run silent past the API's ~90s stale-sweep and get flipped to
    // `interrupted` mid-work — then the terminal `Completed` POST 409'd, surfacing a confusing "failed
    // install" even though the output was promoted (sc-11155). Wrap them in `run_blocking_with_heartbeat`
    // so the worker keeps pinging `Busy` across the convert. `cancel = None` is the explicit per-engine
    // decision: the converters are one bounded native pass with no per-step loop for a flag to poll, so
    // the bounded join already gives everything a cancel flag would (the missing piece was only the
    // keepalive); the up-front `check_cancel` above still honors a cancel requested before it starts.
    if let Err(error) = run_blocking_with_heartbeat(
        api,
        settings,
        &job.id,
        None,
        &convert_cancel_message,
        "model conversion task",
        crate::no_cancel_ack(),
        convert_task,
    )
    .await
    {
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        return Err(error);
    }

    // Promote the completed conversion atomically; on any rename failure the partial
    // temp dir is removed so it can't be picked up later.
    if let Err(error) = finalize_converted_dir(&temp_dir, &final_dir, &settings.data_dir).await {
        let _ = tokio::fs::remove_dir_all(&temp_dir).await;
        return Err(error);
    }

    // The eros checkpoint (the only model that reaches the LTX convert path) ships no bundled Gemma-3
    // text encoder, so fetch the bundle `gemma/` now — before the job reports completion — so the
    // install is self-contained and the first generation needs no on-the-fly ~24 GB download. Runs
    // after promotion; a failure here fails the job loud (the checkpoint stays promoted and the
    // generation path re-attempts the fetch as a backstop).
    if plan_is_ltx {
        provision_ltx_eros_gemma(api, settings, job).await?;
    }
    // A completed conversion is a local derivative. Stamp the producer and exact output bytes
    // before reporting success, so later training/generation admission never trusts a marker or
    // mutable catalog path as its identity.
    write_produced_artifact_receipt(&final_dir, &job.id, &source_repo, &model_id)?;
    let calibration_opt_in = job
        .payload
        .get("memoryCalibrationProvenanceRequired")
        .and_then(Value::as_bool)
        == Some(true);
    if calibration_opt_in {
        let source_revision = checkpoint_dir
            .file_name()
            .and_then(|name| name.to_str())
            .filter(|revision| !revision.is_empty())
            .ok_or_else(|| {
                WorkerError::InvalidPayload(format!(
                    "converted artifact source {} has no immutable snapshot revision",
                    checkpoint_dir.display()
                ))
            })?;
        let tier_dirs = ["q4", "q8", "nvfp4", "bf16", "fp32"]
            .into_iter()
            .filter_map(|tier| {
                let dir = final_dir.join(tier);
                dir.is_dir().then_some((tier, dir))
            })
            .collect::<Vec<_>>();
        if tier_dirs.is_empty() {
            let resolved_tier = match quantize_bits {
                Some(bits) if bits <= 4 => "q4",
                Some(_) => "q8",
                None => "bf16",
            };
            let artifact_variant = format!("converted-{model_id}-{resolved_tier}");
            write_app_managed_artifact_receipt(
                &final_dir,
                &source_repo,
                source_revision,
                &artifact_variant,
                resolved_tier,
            )?;
        } else {
            for (tier, tier_dir) in tier_dirs {
                let artifact_variant = format!("converted-{model_id}-{tier}");
                write_app_managed_artifact_receipt(
                    &tier_dir,
                    &source_repo,
                    source_revision,
                    &artifact_variant,
                    tier,
                )?;
            }
        }
    }

    let mut result = JsonObject::new();
    result.insert("modelId".to_owned(), Value::String(model_id));
    result.insert("sourceRepo".to_owned(), Value::String(source_repo));
    result.insert(
        "path".to_owned(),
        Value::String(final_dir.display().to_string()),
    );
    result.insert("storage".to_owned(), Value::String("mlx_local".to_owned()));
    result.insert("completedAt".to_owned(), Value::String(now_rfc3339()));
    update_job(
        api,
        &job.id,
        progress_payload(
            JobStatus::Completed,
            ProgressStage::Completed,
            1.0,
            &format!("{backend} conversion completed."),
            None,
            Some(result),
            None,
        ),
    )
    .await?;
    Ok(())
}
