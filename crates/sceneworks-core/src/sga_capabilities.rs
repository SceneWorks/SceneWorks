//! Embedded canonical `sga.capabilities/2` capability exports (`SW-6983`, `REG-3`, `M8`).
//!
//! Every SceneWorks consumer reads these hardware-free, versioned, deterministic exports instead
//! of legacy `config/engine-capabilities/*.json` dumps. The linked worker engine remains
//! authoritative at runtime and verifies itself against these exports via
//! [`gen_core::export::verify_runtime_against_export`].

use std::sync::LazyLock;

pub use gen_core::export::{
    canonical_bytes_equal, semantic_equal, verify_runtime_against_export, CapabilitiesExportV2,
    ExportDisagreement, ExportDisagreementError, ExportDisagreementKind,
    ExportedFeatureApplicabilityV2, ExportedImportedModelEntry, ExportedRouteV2,
    ExportedTrainerEntry, CAPABILITIES_EXPORT_SCHEMA_VERSION, FEATURE_SET_CANDLE_CPU,
    FEATURE_SET_CANDLE_CUDA, FEATURE_SET_IOS_LLM_ONLY, FEATURE_SET_IOS_MEDIA,
    FEATURE_SET_MACOS_MLX,
};

pub const MACOS_MLX_JSON: &str =
    include_str!("../../../config/engine-capabilities/sga/macos-mlx.json");
pub const CANDLE_CUDA_JSON: &str =
    include_str!("../../../config/engine-capabilities/sga/candle-cuda.json");
pub const CANDLE_CPU_JSON: &str =
    include_str!("../../../config/engine-capabilities/sga/candle-cpu.json");
pub const IOS_MEDIA_JSON: &str =
    include_str!("../../../config/engine-capabilities/sga/ios-media.json");
pub const IOS_LLM_ONLY_JSON: &str =
    include_str!("../../../config/engine-capabilities/sga/ios-llm-only.json");

static MACOS_MLX: LazyLock<CapabilitiesExportV2> = LazyLock::new(|| {
    CapabilitiesExportV2::from_json_bytes(MACOS_MLX_JSON.as_bytes())
        .expect("embedded sga.capabilities/2 macos-mlx export must parse")
});

static CANDLE_CUDA: LazyLock<CapabilitiesExportV2> = LazyLock::new(|| {
    CapabilitiesExportV2::from_json_bytes(CANDLE_CUDA_JSON.as_bytes())
        .expect("embedded sga.capabilities/2 candle-cuda export must parse")
});

static CANDLE_CPU: LazyLock<CapabilitiesExportV2> = LazyLock::new(|| {
    CapabilitiesExportV2::from_json_bytes(CANDLE_CPU_JSON.as_bytes())
        .expect("embedded sga.capabilities/2 candle-cpu export must parse")
});

static IOS_MEDIA: LazyLock<CapabilitiesExportV2> = LazyLock::new(|| {
    CapabilitiesExportV2::from_json_bytes(IOS_MEDIA_JSON.as_bytes())
        .expect("embedded sga.capabilities/2 ios-media export must parse")
});

static IOS_LLM_ONLY: LazyLock<CapabilitiesExportV2> = LazyLock::new(|| {
    CapabilitiesExportV2::from_json_bytes(IOS_LLM_ONLY_JSON.as_bytes())
        .expect("embedded sga.capabilities/2 ios-llm-only export must parse")
});

/// Canonical `sga.capabilities/2` export for the shipping macOS MLX desktop/server lane.
pub fn macos_mlx() -> &'static CapabilitiesExportV2 {
    &MACOS_MLX
}

/// Canonical `sga.capabilities/2` export for the shipping Windows/Linux Candle CUDA lane.
pub fn candle_cuda() -> &'static CapabilitiesExportV2 {
    &CANDLE_CUDA
}

/// Canonical `sga.capabilities/2` export for the shipping Candle CPU bundle.
pub fn candle_cpu() -> &'static CapabilitiesExportV2 {
    &CANDLE_CPU
}

/// Canonical `sga.capabilities/2` export for the shipping iOS media+LLM bundle.
pub fn ios_media() -> &'static CapabilitiesExportV2 {
    &IOS_MEDIA
}

/// Canonical `sga.capabilities/2` export for the shipping iOS LLM-only bundle.
pub fn ios_llm_only() -> &'static CapabilitiesExportV2 {
    &IOS_LLM_ONLY
}

/// Primary shipping export for a SceneWorks desktop/server backend (`"mlx"` or `"candle"`).
pub fn for_backend(backend: &str) -> Option<&'static CapabilitiesExportV2> {
    match backend {
        "mlx" => Some(macos_mlx()),
        "candle" => Some(candle_cuda()),
        _ => None,
    }
}

/// Shipping export by canonical `feature_set` identifier.
pub fn for_feature_set(feature_set: &str) -> Option<&'static CapabilitiesExportV2> {
    match feature_set {
        FEATURE_SET_MACOS_MLX => Some(macos_mlx()),
        FEATURE_SET_CANDLE_CUDA => Some(candle_cuda()),
        FEATURE_SET_CANDLE_CPU => Some(candle_cpu()),
        FEATURE_SET_IOS_MEDIA => Some(ios_media()),
        FEATURE_SET_IOS_LLM_ONLY => Some(ios_llm_only()),
        _ => None,
    }
}

/// All five canonical shipping exports in deterministic order.
pub fn all_shipping() -> [(&'static str, &'static CapabilitiesExportV2); 5] {
    [
        (FEATURE_SET_MACOS_MLX, macos_mlx()),
        (FEATURE_SET_CANDLE_CUDA, candle_cuda()),
        (FEATURE_SET_CANDLE_CPU, candle_cpu()),
        (FEATURE_SET_IOS_MEDIA, ios_media()),
        (FEATURE_SET_IOS_LLM_ONLY, ios_llm_only()),
    ]
}

/// Normalize `ExportedImportedModelEntry.source` into SceneWorks' persisted `importSourceShape`
/// vocabulary (`"comfyui_tree"` -> `"comfy_ui_tree"`).
pub fn normalize_import_source_shape(source: &str) -> &str {
    match source {
        "comfyui_tree" => "comfy_ui_tree",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_five_shipping_exports_parse_and_roundtrip_canonically() {
        for (feature_set, export) in all_shipping() {
            assert_eq!(export.schema_version, CAPABILITIES_EXPORT_SCHEMA_VERSION);
            assert_eq!(export.feature_set, feature_set);
            let canonical = export.to_canonical_json_bytes().unwrap();
            let reparsed = CapabilitiesExportV2::from_json_bytes(&canonical).unwrap();
            assert!(semantic_equal(export, &reparsed));
        }
        assert_eq!(macos_mlx().backend, "mlx");
        assert_eq!(macos_mlx().platform, "macos");
        assert_eq!(candle_cuda().backend, "candle");
        assert_eq!(candle_cuda().platform, "cuda");
        assert_eq!(candle_cpu().backend, "candle");
        assert_eq!(candle_cpu().platform, "cpu");
        assert_eq!(ios_media().backend, "mlx");
        assert_eq!(ios_media().platform, "ios");
        assert_eq!(ios_llm_only().backend, "mlx");
        assert_eq!(ios_llm_only().platform, "ios");
        assert!(ios_llm_only().routes.is_empty());
        assert_eq!(ios_media().routes.len(), 7);
    }
}
