use super::lora_weight;
use super::resolve_adapter_file;
use super::resolve_prepared_adapter_file;
use super::AdapterKind;
use super::AdapterSpec;
use super::ImageRequest;
use super::MAX_JOB_LORAS;
use crate::read_safetensors_header;
use crate::BTreeMap;
use crate::Path;
use crate::PathBuf;
use crate::Settings;
use crate::Value;
use crate::WorkerError;
use crate::WorkerResult;
/// Classify a LoRA file into the mlx-gen adapter `kind`. SceneWorks peft-LoKr (stamped
/// `networkType: lokr`) → `Lokr` (the engine's metadata-gated `apply_lokr` peft path). Everything
/// else → `Lora`, INCLUDING third-party LyCORIS (LoHa / kohya non-peft LoKr): since epic 3641
/// (sc-3642/3643/3671) the engine's `apply_adapter_specs_autoprefix` detects `lokr_*` / `hada_*`
/// keys by sniff and routes them to its third-party reconstruction regardless of the declared kind,
/// so `Lora` is the correct hint and the worker no longer rejects them. (A LyCORIS algo the engine
/// doesn't implement — e.g. (IA)³/OFT — has no `lokr_*`/`hada_*` keys, so the engine's LoRA loader
/// finds nothing and surfaces a loud "matched nothing" error rather than mis-applying.)
///
/// Shared by MLX and Candle: every Candle family validates this declared kind against the selected
/// file before application, so the same `networkType: lokr` classification feeds both backends.
#[cfg(native_lane)]
pub(crate) fn classify_adapter(file: &Path) -> WorkerResult<AdapterKind> {
    let header = read_safetensors_header(file)
        .map_err(|error| WorkerError::InvalidPayload(format!("LoRA header: {error}")))?;
    let network_type = header
        .get("__metadata__")
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    if network_type.as_deref() == Some("lokr") {
        return Ok(AdapterKind::Lokr);
    }
    Ok(AdapterKind::Lora)
}

#[cfg(any(native_lane, test))]
pub(crate) fn classify_prepared_adapter(
    pin: &gen_core::PinnedWeightsFile,
) -> WorkerResult<AdapterKind> {
    let header = pin
        .read_unchanged(|file| {
            read_safetensors_header(file)
                .map_err(|error| gen_core::Error::Msg(format!("LoRA header: {error}")))
        })
        .map_err(|error| crate::classify_engine_error("LoRA source validation failed", error))?;
    let network_type = header
        .get("__metadata__")
        .and_then(|meta| meta.get("networkType"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    Ok(if network_type.as_deref() == Some("lokr") {
        AdapterKind::Lokr
    } else {
        AdapterKind::Lora
    })
}

#[derive(Debug)]
#[cfg(any(native_lane, test))]
pub(super) struct PreparedAdapters {
    pub(super) specs: Vec<AdapterSpec>,
    pub(super) pins: Vec<gen_core::PinnedWeightsFile>,
}

#[cfg(any(native_lane, test))]
impl PreparedAdapters {
    // Gated to exactly where it is used. `imported_edit_requires_the_identity_edit_lora` became
    // cross-platform when the Kontext edit surface started being served on both native backends, so
    // a macOS-only helper left the candle test build without it.
    #[cfg(all(native_lane, test))]
    pub(super) fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }
}

/// Prepared counterpart of [`resolve_adapters`]. Classification reads through the exact token that
/// is later installed on the `LoadSpec`; repeated references to the same lexical entry share one
/// token, while preserving every requested adapter spec and its scale/order.
#[cfg(any(native_lane, test))]
pub(super) fn resolve_prepared_adapters(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<PreparedAdapters> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    let mut pins = BTreeMap::<PathBuf, gen_core::PinnedWeightsFile>::new();
    for lora in &request.loras {
        let pin = resolve_prepared_adapter_file(lora, settings, &request.model)?;
        let pin = crate::adapter_resolution::retain_adapter_pin(&mut pins, pin)?;
        let kind = classify_prepared_adapter(pin)?;
        let scale = lora_weight(lora) as f32;
        specs.push(AdapterSpec::new(
            pin.loader_path().to_path_buf(),
            scale,
            kind,
        ));
    }
    Ok(PreparedAdapters {
        specs,
        pins: pins.into_values().collect(),
    })
}

/// Resolve up to [`MAX_JOB_LORAS`] request adapters into engine specs (path + scale + kind).
/// Shared by the MLX and Candle paths so both backends enforce the same stack ceiling.
#[cfg(native_lane)]
pub(super) fn resolve_adapters(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Vec<AdapterSpec>> {
    if request.loras.len() > MAX_JOB_LORAS {
        return Err(WorkerError::InvalidPayload(format!(
            "Generation supports at most {MAX_JOB_LORAS} LoRAs per job."
        )));
    }
    let mut specs = Vec::with_capacity(request.loras.len());
    for lora in &request.loras {
        let file = resolve_adapter_file(lora, settings, &request.model)?;
        let kind = classify_adapter(&file)?;
        let scale = lora_weight(lora) as f32;
        specs.push(AdapterSpec::new(file, scale, kind));
    }
    Ok(specs)
}

/// Whether Krea may use calibrated streamed blocks, plus the load-exact adapter bytes supporting
/// that decision. A non-empty stack with unreadable metadata returns `None`; the caller refuses it
/// rather than inheriting evidence from the adapter-free route.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(super) fn krea_streamed_blocks_adapter_evidence(
    adapters: &[AdapterSpec],
) -> (bool, Option<u64>) {
    if adapters.is_empty() {
        return (true, Some(0));
    }
    let bytes = adapters.iter().try_fold(0_u64, |total, adapter| {
        let bytes = gen_core::safetensors_path_bytes(&adapter.path);
        (bytes > 0).then(|| total.saturating_add(bytes))
    });
    (false, bytes)
}

#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) fn candle_adapter_resident_bytes(
    engine_id: &str,
    tier: &str,
    measured_source_bytes: u64,
) -> u64 {
    // Krea keeps lazily merged adapter overlays on its resident host. Z-Image likewise installs
    // forward-time residuals for dense bf16 as well as packed tiers, and so does MiniMax-H3-Image
    // (SW-7096: every adapter is a residual, never a merged weight), so those families must reserve
    // the exact adapter source bytes at every precision. Other generic Candle providers fold dense
    // factors and retain residuals only on packed tiers.
    if engine_id.starts_with("krea_2_")
        || engine_id.starts_with("z_image")
        || engine_id == "minimax_h3_image"
        || crate::candle_memory_strategy::is_sd35(engine_id)
        || tier != "bf16"
    {
        measured_source_bytes
    } else {
        0
    }
}
