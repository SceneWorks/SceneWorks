// Mechanically checked inventory of production model consumers.
//
// Entries classify source surfaces, not catalog rows. Tests scan every model-path primitive and
// fail when a new production consumer is not covered by exactly one shared resolver or explicit
// unsupported rule. This is the migration map for the two-tier artifact contract.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConsumerCategory {
    Image,
    Video,
    Audio,
    CaptioningUtility,
    Training,
    LoraControl,
    Primary,
    OptionalComponent,
    CoRequisite,
    ImportedConverted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsumerResolution {
    SharedContract { entrypoint: &'static str },
    Unsupported { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelConsumerInventoryEntry {
    /// Repository-relative exact files. Prefixes are deliberately forbidden: adding a new runtime
    /// module must make the source audit fail until its category and entrypoint are reviewed.
    pub source_files: &'static [&'static str],
    pub categories: &'static [ConsumerCategory],
    pub resolution: ConsumerResolution,
}

use ConsumerCategory as Category;
use ConsumerResolution as Resolution;

pub const PRODUCTION_MODEL_CONSUMERS: &[ModelConsumerInventoryEntry] = &[
    ModelConsumerInventoryEntry {
        source_files: &[
            "crates/sceneworks-worker/src/image_jobs.rs",
            "crates/sceneworks-worker/src/image_jobs/base.rs",
            "crates/sceneworks-worker/src/image_jobs/bernini.rs",
            "crates/sceneworks-worker/src/image_jobs/detail.rs",
            "crates/sceneworks-worker/src/image_jobs/flux1_control.rs",
            "crates/sceneworks-worker/src/image_jobs/flux1_control_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/flux2.rs",
            "crates/sceneworks-worker/src/image_jobs/flux2_comfyui_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/flux2_control_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/flux_ipadapter.rs",
            "crates/sceneworks-worker/src/image_jobs/instantid.rs",
            "crates/sceneworks-worker/src/image_jobs/kolors.rs",
            "crates/sceneworks-worker/src/image_jobs/kolors_control.rs",
            "crates/sceneworks-worker/src/image_jobs/kolors_ipadapter.rs",
            "crates/sceneworks-worker/src/image_jobs/krea_control.rs",
            "crates/sceneworks-worker/src/image_jobs/krea_control_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/krea_imported.rs",
            "crates/sceneworks-worker/src/image_jobs/mage_finetuned.rs",
            "crates/sceneworks-worker/src/image_jobs/pid.rs",
            "crates/sceneworks-worker/src/image_jobs/pulid.rs",
            "crates/sceneworks-worker/src/image_jobs/pulid_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/qwen.rs",
            "crates/sceneworks-worker/src/image_jobs/qwen_comfyui_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/qwen_control.rs",
            "crates/sceneworks-worker/src/image_jobs/qwen_edit_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/sdxl.rs",
            "crates/sceneworks-worker/src/image_jobs/sdxl_edit_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/sdxl_imported.rs",
            "crates/sceneworks-worker/src/image_jobs/sdxl_ipadapter.rs",
            "crates/sceneworks-worker/src/image_jobs/strict_control.rs",
            "crates/sceneworks-worker/src/image_jobs/zimage.rs",
            "crates/sceneworks-worker/src/image_jobs/zimage_comfyui_candle.rs",
            "crates/sceneworks-worker/src/image_jobs/zimage_control.rs",
            "crates/sceneworks-worker/src/image_jobs/zimage_edit_candle.rs",
        ],
        categories: &[
            Category::Image,
            Category::Primary,
            Category::OptionalComponent,
            Category::CoRequisite,
            Category::LoraControl,
            Category::ImportedConverted,
        ],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs + paths artifact resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &[
            "crates/sceneworks-worker/src/video_jobs/bernini.rs",
            "crates/sceneworks-worker/src/video_jobs/candle.rs",
            "crates/sceneworks-worker/src/video_jobs/krea_realtime.rs",
            "crates/sceneworks-worker/src/video_jobs/ltx.rs",
            "crates/sceneworks-worker/src/video_jobs/mochi.rs",
            "crates/sceneworks-worker/src/video_jobs/scail2.rs",
            "crates/sceneworks-worker/src/video_jobs/svd.rs",
            "crates/sceneworks-worker/src/video_jobs/vace.rs",
            "crates/sceneworks-worker/src/video_jobs/wan.rs",
        ],
        categories: &[Category::Video, Category::Primary, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs source/receipt resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/audio_jobs.rs"],
        categories: &[Category::Audio, Category::Primary, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs co-requisite + paths runtime resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/caption_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: "paths runtime resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/training_jobs.rs"],
        categories: &[Category::Training, Category::Primary, Category::LoraControl],
        resolution: Resolution::SharedContract {
            entrypoint: "paths runtime resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/prompt_refine_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: "paths runtime resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/catalog_semantic_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: "paths runtime resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/dataset_analysis_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: "paths runtime resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &[
            "crates/sceneworks-worker/src/person_segment.rs",
            "crates/sceneworks-worker/src/person_segment_sam3_common.rs",
        ],
        categories: &[Category::CaptioningUtility, Category::OptionalComponent],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs pinned source resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/person_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::OptionalComponent],
        resolution: Resolution::SharedContract {
            entrypoint: "downloads pinned component resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/pose_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs pinned source resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/voice_register.rs"],
        categories: &[Category::Audio, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs pinned source resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/upscale_jobs.rs"],
        categories: &[Category::Image, Category::OptionalComponent],
        resolution: Resolution::SharedContract {
            entrypoint: "downloads pinned component resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/engines.rs"],
        categories: &[Category::Primary, Category::CaptioningUtility],
        resolution: Resolution::SharedContract {
            entrypoint: "hf_home typed source-library seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/mlx_fit_gate.rs"],
        categories: &[Category::Primary, Category::ImportedConverted],
        resolution: Resolution::SharedContract {
            entrypoint: "model_jobs receipt/provenance artifact resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/models.rs"],
        categories: &[
            Category::Primary,
            Category::OptionalComponent,
            Category::CoRequisite,
            Category::ImportedConverted,
        ],
        resolution: Resolution::SharedContract {
            entrypoint: "hf_home typed source-library + artifact inventory seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/loras.rs"],
        categories: &[Category::LoraControl],
        resolution: Resolution::SharedContract {
            entrypoint: "hf_home typed source-library + receipt resolver seam",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/training.rs"],
        categories: &[Category::Training],
        resolution: Resolution::SharedContract {
            entrypoint: "manifest + worker runtime resolver seam",
        },
    },
];

pub const EXPLICITLY_UNSUPPORTED_ARTIFACTS: &[ModelConsumerInventoryEntry] = &[
    ModelConsumerInventoryEntry {
        source_files: &["mutable Hugging Face revision"],
        categories: &[Category::Primary, Category::CoRequisite],
        resolution: Resolution::Unsupported {
            reason: "runtime identity requires a resolved immutable 40-hex commit",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["empty, glob-only, torn, or traversal-bearing file closure"],
        categories: &[
            Category::Primary,
            Category::OptionalComponent,
            Category::CoRequisite,
            Category::LoraControl,
        ],
        resolution: Resolution::Unsupported {
            reason: "runtime bundles require a complete concrete confined file closure",
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["unproven arbitrary host path"],
        categories: &[Category::ImportedConverted, Category::Training],
        resolution: Resolution::Unsupported {
            reason: "paths require an app-managed or operator-declared root with provenance",
        },
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    const RESOLUTION_MARKERS: &[&str] = &[
        "huggingface_repo_cache_path(",
        "huggingface_snapshot_dir(",
        "huggingface_pinned_snapshot_dir(",
        "huggingface_receipt_weights",
        "resolve_app_managed_model_dir(",
        "normalize_app_managed_model_path(",
        "resolve_hf_component_file(",
    ];

    const INFRASTRUCTURE_SURFACES: &[(&str, &str)] = &[
        (
            "crates/sceneworks-worker/src/downloads.rs",
            "source-library transfer implementation",
        ),
        (
            "crates/sceneworks-worker/src/job_time_download_guard.rs",
            "download admission guard, not a model loader",
        ),
        (
            "crates/sceneworks-worker/src/lib.rs",
            "worker dispatch and module wiring",
        ),
        (
            "crates/sceneworks-worker/src/model_artifact_inventory.rs",
            "this self-auditing inventory",
        ),
        (
            "crates/sceneworks-worker/src/model_jobs.rs",
            "shared resolver implementation",
        ),
        (
            "crates/sceneworks-worker/src/paths.rs",
            "shared path/confinement implementation",
        ),
        (
            "crates/sceneworks-worker/src/snapshot_install.rs",
            "source-library installation implementation",
        ),
        (
            "crates/sceneworks-worker/src/test_env.rs",
            "test-only environment helper",
        ),
        ("apps/rust-api/src/lib.rs", "API state and module wiring"),
    ];

    fn collect_rust_files(root: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(root).expect("read source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                collect_rust_files(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }

    fn is_test_surface(relative: &str) -> bool {
        relative.contains("/tests/") || relative.ends_with("/tests.rs")
    }

    #[test]
    fn every_model_path_consumer_is_classified_by_the_checked_in_inventory() {
        let worker_manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let repository = worker_manifest.join("../..");
        let mut files = Vec::new();
        collect_rust_files(&repository.join("crates/sceneworks-worker/src"), &mut files);
        collect_rust_files(&repository.join("apps/rust-api/src"), &mut files);
        let mut uncovered = Vec::new();
        for file in files {
            let relative = file
                .strip_prefix(&repository)
                .expect("source beneath repository")
                .to_string_lossy()
                .replace('\\', "/");
            if is_test_surface(&relative)
                || INFRASTRUCTURE_SURFACES
                    .iter()
                    .any(|(surface, _)| *surface == relative)
            {
                continue;
            }
            let source = std::fs::read_to_string(&file).expect("read Rust source");
            if !RESOLUTION_MARKERS
                .iter()
                .any(|marker| source.contains(marker))
            {
                continue;
            }
            let matches = PRODUCTION_MODEL_CONSUMERS
                .iter()
                .filter(|entry| entry.source_files.contains(&relative.as_str()))
                .count();
            if matches != 1 {
                uncovered.push(format!("{relative} ({matches} inventory matches)"));
            }
        }
        assert!(
            uncovered.is_empty(),
            "model-path consumers need exactly one inventory classification:\n{}",
            uncovered.join("\n")
        );
    }

    #[test]
    fn inventory_covers_every_required_category_and_has_real_entrypoints() {
        let expected = BTreeSet::from([
            Category::Image,
            Category::Video,
            Category::Audio,
            Category::CaptioningUtility,
            Category::Training,
            Category::LoraControl,
            Category::Primary,
            Category::OptionalComponent,
            Category::CoRequisite,
            Category::ImportedConverted,
        ]);
        let actual = PRODUCTION_MODEL_CONSUMERS
            .iter()
            .flat_map(|entry| entry.categories.iter().copied())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut seen = BTreeSet::new();
        for entry in PRODUCTION_MODEL_CONSUMERS {
            match entry.resolution {
                Resolution::SharedContract { entrypoint } => assert!(!entrypoint.trim().is_empty()),
                Resolution::Unsupported { .. } => {
                    panic!("production consumer unexpectedly marked unsupported")
                }
            }
            for source_file in entry.source_files {
                assert!(
                    seen.insert(*source_file),
                    "consumer file is classified more than once: {source_file}"
                );
                assert!(
                    repository.join(source_file).is_file(),
                    "inventoried consumer file does not exist: {source_file}"
                );
            }
        }
        assert!(EXPLICITLY_UNSUPPORTED_ARTIFACTS
            .iter()
            .all(|entry| matches!(
                    entry.resolution,
                    Resolution::Unsupported { reason } if !reason.is_empty()
            )));
        assert!(INFRASTRUCTURE_SURFACES
            .iter()
            .all(|(surface, reason)| !surface.is_empty() && !reason.is_empty()));
    }
}
