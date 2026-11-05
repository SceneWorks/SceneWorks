//! Model HTTP handlers and catalog facade. Concern owners keep the original models scope.
use super::*;

mod license;
pub(crate) use license::*;
mod download;
pub(crate) use download::*;
mod convert;
pub(crate) use convert::*;
mod delete;
pub(crate) use delete::*;
mod import;
pub(crate) use import::*;
mod catalog;
pub(crate) use catalog::*;
mod install_state;
pub(crate) use install_state::*;
mod hf_cache_health;
pub(crate) use hf_cache_health::*;
mod size_estimate;
pub(crate) use size_estimate::*;

use sceneworks_core::base_weights::{
    detect_base_weight_file, import_detection_supported, imported_model_primary_weight_file,
    BaseWeightDetection, ComponentRole,
};
use sceneworks_core::credentials::normalize_host;
use sceneworks_core::lora_family::is_hidden_file;

#[cfg(test)]
#[path = "tests/memory_calibration_job_payload_tests.rs"]
mod memory_calibration_job_payload_tests;

pub(crate) use crate::uploads::cleanup_staged_upload as cleanup_staged_model_upload;

#[cfg(test)]
#[path = "tests/runtime_text_encoder_option_tests.rs"]
mod runtime_text_encoder_option_tests;

#[cfg(test)]
#[path = "tests/import_gate_tests.rs"]
mod import_gate_tests;

#[cfg(test)]
#[path = "tests/download_receipt_tests.rs"]
mod download_receipt_tests;

#[cfg(test)]
#[path = "tests/last_only_projection_tests.rs"]
mod last_only_projection_tests;

#[cfg(test)]
#[path = "tests/decoder_availability_tests.rs"]
mod decoder_availability_tests;

#[cfg(test)]
#[path = "tests/model_size_concurrency_tests.rs"]
mod model_size_concurrency_tests;

/// Restrict a model's `downloads` to the entries applicable to `os` (`std::env::consts::OS`).
/// A download entry with a `platforms` array applies only to the listed OSes; an entry without one
/// is platform-agnostic and always kept. Some video models ship two source repos for the same model
/// — the native MLX-convert checkpoint on macOS vs the diffusers/torch checkpoint on Windows/Linux
/// (sc-3240, Wan2.2) — so filtering here makes the download job, install status, size, and the
/// frontend's `downloads[0]` all resolve to the right per-platform repo from one seam. No-op unless
/// at least one entry is platform-tagged, so single-repo models are untouched.
///
/// sc-19708: the manifest-shape predicates below now live in
/// `sceneworks_core::model_artifacts::artifact_selection` because the availability seam (API
/// preflight AND the worker's pre-loader guard) selects the same closures. These re-exports keep
/// the API's install/display call sites on the identical definitions so they cannot drift.
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os;

pub(crate) use sceneworks_core::model_artifacts::artifact_selection::model_download;

/// True when a download entry is a co-requisite dependency (sc-9696): fetched ALONGSIDE the primary
/// download rather than as a pick-one alternate, and gating the entry's install state. See the
/// manifest schema `downloads[].coRequisite`.
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::is_co_requisite_download;

/// The co-requisite download entries for `model` (sc-9696) — the dependencies that must install
/// alongside the primary (e.g. the PiD decoder's shared gemma-2-2b-it caption encoder). The catalog
/// has already restricted `downloads` to the current OS (`retain_downloads_for_os`), so every entry
/// returned applies to this platform. Only provider-supported entries are returned.
/// Whether a co-requisite row is scoped to ONE quant tier (sc-14980).
///
/// Every co-requisite before sc-14980 is tier-agnostic — the PiD decoder's gemma caption encoder,
/// chatterbox's `ve`/`perth`, MMAudio's five components — and carries no `variant`, so it applies to
/// whatever tier is selected. Mage-Flow's shared Qwen3-VL text encoder and Mage-VAE are the first
/// that are themselves per-tier: they exist as `q4`/`q8`/`bf16` subtrees of one components mirror,
/// and only the one matching the selected tier should be fetched, sized, or gated on. Keying that on
/// the presence of `variant` keeps every existing co-requisite on exactly its current path.
pub(crate) use sceneworks_core::model_artifacts::artifact_selection::{
    co_requisite_rows_gating_install, co_requisite_variant, is_pending_artifact_download,
    model_co_requisite_downloads, model_co_requisite_downloads_for_variant,
    model_co_requisite_downloads_for_variant_all_options, model_download_for_variant,
};

pub(crate) use sceneworks_core::model_artifacts::artifact_selection::is_supported_model_download;

#[cfg(test)]
#[path = "tests/gated_credential_tests.rs"]
mod gated_credential_tests;

#[cfg(test)]
#[path = "tests/variant_install_tests.rs"]
mod variant_install_tests;

#[cfg(test)]
#[path = "tests/mlx_tier_probe_tests.rs"]
mod mlx_tier_probe_tests;

// Per-tier delete (sc-12024, epic 8506). Exercises the blob-aware reclamation on a realistic HF
// hub-cache layout — real `blobs/<etag>` files with snapshot SYMLINKS into them — which is the whole
// reason a tier delete is non-trivial: unlinking the tier's snapshot symlinks alone frees nothing.
// unix-gated because the fixtures use symlinks (the production cache layout on macOS/Linux).
#[cfg(all(test, unix))]
#[path = "tests/variant_delete_tests.rs"]
mod variant_delete_tests;

/// Lane-parameterized coverage for the imported-model LoRA advertisement withdrawal.
///
/// These drive `apply_imported_lora_advertisement_for_lanes` directly rather than through the
/// platform-bound wrapper, so BOTH deployment topologies are exercised on every CI platform.
#[cfg(test)]
#[path = "tests/imported_lora_advertisement_tests.rs"]
mod imported_lora_advertisement_tests;

/// sc-20636 (epic 20398): the import API's ownership mode + discriminated source, and the
/// normalisation that keeps every legacy flat request working and managed.
#[cfg(test)]
#[path = "tests/model_import_source_tests.rs"]
mod model_import_source_tests;

#[cfg(test)]
#[path = "tests/managed_predictor_queue_tests.rs"]
mod managed_predictor_queue_tests;

#[cfg(test)]
#[path = "tests/minimax_h3_import_declaration_tests.rs"]
mod minimax_h3_import_declaration_tests;

/// The imported-checkpoint product surface (sc-26687) — what Model Manager, Video Studio, the API
/// and MCP all read. These assert the VALUES a user acts on: the declared task contract's
/// operations, the backends that will actually claim the import, and the repair a missing
/// component calls for. Not that a field was written.
#[cfg(test)]
#[path = "tests/imported_checkpoint_product_surface_tests.rs"]
mod imported_checkpoint_product_surface_tests;

#[cfg(test)]
pub(crate) use crate::test_hooks::{
    test_catalog_probe_limit, test_catalog_probe_peak, test_delay_catalog_probe,
    test_reset_catalog_probe_concurrency, ModelCatalogBeforePublishTestHook,
    ModelSizeEstimateTestHook,
};

#[cfg(test)]
#[path = "tests/io_offload_tests.rs"]
mod io_offload_tests;

#[cfg(test)]
#[path = "tests/model_contract_metadata_tests.rs"]
mod model_contract_metadata_tests;
