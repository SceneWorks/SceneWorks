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
    SharedContract {
        entrypoint: TypedResolverEntrypoint,
    },
    Unsupported { reason: &'static str },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypedResolverEntrypoint {
    WorkerSnapshot,
    ManagedModelPath,
    PinnedComponent,
    ReceiptProvenance,
    SourceLibrary,
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
            entrypoint: TypedResolverEntrypoint::WorkerSnapshot,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &[
            "crates/sceneworks-worker/src/video_jobs/mod.rs",
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
            entrypoint: TypedResolverEntrypoint::WorkerSnapshot,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/audio_jobs.rs"],
        categories: &[Category::Audio, Category::Primary, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::WorkerSnapshot,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/caption_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ManagedModelPath,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/training_jobs.rs"],
        categories: &[Category::Training, Category::Primary, Category::LoraControl],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ManagedModelPath,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/prompt_refine_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ManagedModelPath,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/catalog_semantic_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ManagedModelPath,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/dataset_analysis_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::Primary],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ManagedModelPath,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &[
            "crates/sceneworks-worker/src/person_segment.rs",
            "crates/sceneworks-worker/src/person_segment_sam3_common.rs",
        ],
        categories: &[Category::CaptioningUtility, Category::OptionalComponent],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::PinnedComponent,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/person_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::OptionalComponent],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::PinnedComponent,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/pose_jobs.rs"],
        categories: &[Category::CaptioningUtility, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::PinnedComponent,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/voice_register.rs"],
        categories: &[Category::Audio, Category::CoRequisite],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::PinnedComponent,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/upscale_jobs.rs"],
        categories: &[Category::Image, Category::OptionalComponent],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::PinnedComponent,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/engines.rs"],
        categories: &[Category::Primary, Category::CaptioningUtility],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/mlx_fit_gate.rs"],
        categories: &[Category::Primary, Category::ImportedConverted],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ReceiptProvenance,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/vram_gate.rs"],
        categories: &[Category::Primary, Category::LoraControl],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::ManagedModelPath,
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
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/loras.rs"],
        categories: &[Category::LoraControl],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/training.rs"],
        categories: &[Category::Training],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
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
    use quote::ToTokens;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    const MODEL_ROOT_MARKERS: &[&str] = &[
        "huggingface_hub_cache_dir(",
        "huggingface_repo_cache_path(",
        "huggingface_snapshot_dir(",
        "huggingface_pinned_snapshot_dir(",
        "huggingface_receipt_weights",
        "model_source_library(",
        "resolve_app_managed_model_dir(",
        "normalize_app_managed_model_path(",
        "normalize_app_managed_lora_path(",
        "resolve_hf_component_file(",
        ".join(\"snapshots\")",
        ".join(\"models--",
    ];

    const TYPED_CONSUMER_MARKERS: &[&str] = &[
        "huggingface_repo_cache_path(",
        "huggingface_snapshot_dir(",
        "huggingface_pinned_snapshot_dir(",
        "huggingface_receipt_weights",
        "model_source_library(",
        "resolve_app_managed_model_dir(",
        "normalize_app_managed_model_path(",
        "normalize_app_managed_lora_path(",
        "resolve_hf_component_file(",
    ];

    const DIRECT_ROOT_MARKERS: &[&str] = &[
        "huggingface_hub_cache_dir(",
        ".join(\"snapshots\")",
        ".join(\"models--",
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

    // These exact modules are compiled only behind `cfg(all(test, target_os = "macos"))` and are
    // ignored real-weight/build harnesses, never production runtime consumers. Keep the list exact:
    // a newly added harness or production file must still be classified by this audit.
    const TEST_ONLY_MODEL_SURFACES: &[&str] = &[
        "crates/sceneworks-worker/src/bernini_tier_build.rs",
        "crates/sceneworks-worker/src/footprint_measure.rs",
        "crates/sceneworks-worker/src/ladder_e2e_sc18101.rs",
        "crates/sceneworks-worker/src/mage_flow_q8_mlx_smoke.rs",
        "crates/sceneworks-worker/src/pid_tier_mlx_smoke.rs",
        "crates/sceneworks-worker/src/resolution_sweep.rs",
        "crates/sceneworks-worker/src/sana_mlx_smoke.rs",
        "crates/sceneworks-worker/src/sd3_5_mlx_smoke.rs",
        "crates/sceneworks-worker/src/voiceclone_smoke.rs",
        "crates/sceneworks-worker/src/wan_i2v_14b_tier_build.rs",
        "crates/sceneworks-worker/src/wan_t2v_14b_tier_build.rs",
        "crates/sceneworks-worker/src/wan_ti2v_5b_tier_build.rs",
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

    fn item_attributes(item: &syn::Item) -> &[syn::Attribute] {
        match item {
            syn::Item::Const(item) => &item.attrs,
            syn::Item::Enum(item) => &item.attrs,
            syn::Item::ExternCrate(item) => &item.attrs,
            syn::Item::Fn(item) => &item.attrs,
            syn::Item::ForeignMod(item) => &item.attrs,
            syn::Item::Impl(item) => &item.attrs,
            syn::Item::Macro(item) => &item.attrs,
            syn::Item::Mod(item) => &item.attrs,
            syn::Item::Static(item) => &item.attrs,
            syn::Item::Struct(item) => &item.attrs,
            syn::Item::Trait(item) => &item.attrs,
            syn::Item::TraitAlias(item) => &item.attrs,
            syn::Item::Type(item) => &item.attrs,
            syn::Item::Union(item) => &item.attrs,
            syn::Item::Use(item) => &item.attrs,
            _ => &[],
        }
    }

    fn item_is_test_only(item: &syn::Item) -> bool {
        item_attributes(item).iter().any(|attribute| {
            attribute.path().is_ident("test")
                || (attribute.path().is_ident("cfg")
                    && attribute
                        .meta
                        .to_token_stream()
                        .to_string()
                        .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                        .any(|token| token == "test"))
        })
    }

    fn retain_production_items(items: &mut Vec<syn::Item>) {
        items.retain_mut(|item| {
            if item_is_test_only(item) {
                return false;
            }
            if let syn::Item::Mod(module) = item {
                if let Some((_, nested)) = &mut module.content {
                    retain_production_items(nested);
                }
            }
            true
        });
    }

    fn production_rust_source(source: &str) -> String {
        let mut syntax = syn::parse_file(source).expect("production source parses as Rust");
        retain_production_items(&mut syntax.items);
        syntax
            .into_token_stream()
            .to_string()
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect()
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
                || TEST_ONLY_MODEL_SURFACES.contains(&relative.as_str())
                || INFRASTRUCTURE_SURFACES
                    .iter()
                    .any(|(surface, _)| *surface == relative)
            {
                continue;
            }
            let source = std::fs::read_to_string(&file).expect("read Rust source");
            let source = production_rust_source(&source);
            if !MODEL_ROOT_MARKERS
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
                continue;
            }
            let direct = DIRECT_ROOT_MARKERS
                .iter()
                .filter(|marker| source.contains(**marker))
                .copied()
                .collect::<Vec<_>>();
            if !direct.is_empty() {
                uncovered.push(format!(
                    "{relative} constructs a model root directly via {direct:?}"
                ));
                continue;
            }
            if !TYPED_CONSUMER_MARKERS
                .iter()
                .any(|marker| source.contains(marker))
            {
                uncovered.push(format!(
                    "{relative} has a model root but no typed resolver entrypoint"
                ));
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
                Resolution::SharedContract { .. } => {}
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
        assert!(TEST_ONLY_MODEL_SURFACES
            .iter()
            .all(|surface| repository.join(surface).is_file()));
        let worker_lib = std::fs::read_to_string(repository.join("crates/sceneworks-worker/src/lib.rs"))
            .expect("read worker module declarations");
        let worker_syntax = syn::parse_file(&worker_lib).expect("worker lib parses as Rust");
        for surface in TEST_ONLY_MODEL_SURFACES {
            let module_name = Path::new(surface)
                .file_stem()
                .and_then(|stem| stem.to_str())
                .expect("test-only surface has a Rust module name");
            let declaration = worker_syntax.items.iter().find(|item| {
                matches!(item, syn::Item::Mod(module) if module.ident == module_name)
            });
            assert!(
                declaration.is_some_and(item_is_test_only),
                "{surface} must remain behind an exact test-only module declaration"
            );
        }
    }

    #[test]
    fn typed_entrypoints_are_compiled_contract_implementations_not_labels() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let implementations = [
            (
                TypedResolverEntrypoint::WorkerSnapshot,
                "crates/sceneworks-worker/src/model_jobs.rs",
                &["ModelArtifactResolver::new(", "discover_source_snapshot("][..],
            ),
            (
                TypedResolverEntrypoint::ManagedModelPath,
                "crates/sceneworks-worker/src/paths.rs",
                &["model_artifacts::confine_artifact_path("][..],
            ),
            (
                TypedResolverEntrypoint::PinnedComponent,
                "crates/sceneworks-worker/src/downloads.rs",
                &["huggingface_pinned_snapshot_dir("][..],
            ),
            (
                TypedResolverEntrypoint::ReceiptProvenance,
                "crates/sceneworks-worker/src/model_jobs.rs",
                &[
                    "model_artifacts::ArtifactIdentity",
                    "model_artifacts::ArtifactProvenance",
                ][..],
            ),
            (
                TypedResolverEntrypoint::SourceLibrary,
                "crates/sceneworks-core/src/hf_home.rs",
                &["ArtifactSourceLibrary", "model_source_library(data_dir)"][..],
            ),
        ];
        let used = PRODUCTION_MODEL_CONSUMERS
            .iter()
            .filter_map(|entry| match entry.resolution {
                Resolution::SharedContract { entrypoint } => Some(entrypoint),
                Resolution::Unsupported { .. } => None,
            })
            .collect::<BTreeSet<_>>();
        let implemented = implementations
            .iter()
            .map(|(entrypoint, _, _)| *entrypoint)
            .collect::<BTreeSet<_>>();
        assert_eq!(used, implemented);
        for (entrypoint, source_file, required_markers) in implementations {
            let source = std::fs::read_to_string(repository.join(source_file))
                .expect("read typed entrypoint source");
            for marker in required_markers {
                assert!(
                    source.contains(marker),
                    "{entrypoint:?} no longer proves typed resolution via {marker:?} in {source_file}"
                );
            }
        }
    }

    #[test]
    fn production_scan_rejects_direct_roots_but_ignores_exact_cfg_test_items() {
        let source = r#"
            fn production_loader(root: &std::path::Path) {
                let _ = root.join("snapshots");
            }

            mod nested {
                #[cfg(
                    all(test, unix)
                )]
                mod tests {
                    fn fixture(root: &std::path::Path) {
                        let _ = root.join("models--Fixture--only").join("snapshots");
                    }
                }
            }
        "#;
        let production = production_rust_source(source);
        assert!(production.contains("join(\"snapshots\")"));
        assert!(!production.contains("Fixture"));

        let only_test = r#"
            #[cfg(test)]
            mod tests {
                fn fixture(root: &std::path::Path) {
                    let _ = root.join("snapshots");
                }
            }
        "#;
        let production = production_rust_source(only_test);
        assert!(DIRECT_ROOT_MARKERS
            .iter()
            .all(|marker| !production.contains(marker)));
    }
}
