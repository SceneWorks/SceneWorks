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
    // sc-19708: the worker's single pre-loader model-source guard. It judges every dispatched
    // job's carriers (all categories) against the typed source-library availability contract and
    // is deliberately the ONLY worker surface holding external-library availability policy.
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/external_library_runtime.rs"],
        categories: &[
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
        ],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    // sc-19706: the idle-time promotion producer. It never resolves a path for a LOAD — it turns a
    // closure a successful load already used into an app-owned bundle — and it constructs no model
    // root of its own: the source root is handed to it by the guard above, which obtained it from
    // the typed `model_source_library` entrypoint. It rebuilds a NON-preferring
    // `ArtifactSourceLibrary` over that same root on purpose, so a bundle can never be promoted
    // from a bundle. Classified rather than exempted because it does resolve closures against the
    // authoritative source.
    ModelConsumerInventoryEntry {
        source_files: &["crates/sceneworks-worker/src/resolved_cache_promotion.rs"],
        categories: &[
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
        ],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    // sc-19708: the API's single model-source seam (carrier attachment + submission preflight);
    // same shared source-library contract, mirrored on the API side of the boundary.
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/model_sources.rs"],
        categories: &[
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
        ],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    // sc-19709: the model source library's status + relocation seam. It never resolves a MODEL
    // path — it probes and re-binds the library ROOT itself — but it reaches that root through the
    // same typed `model_source_library` entrypoint, so it is classified here rather than exempted.
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/model_library.rs"],
        categories: &[
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
        ],
        resolution: Resolution::SharedContract {
            entrypoint: TypedResolverEntrypoint::SourceLibrary,
        },
    },
    // sc-19711: the resolved-model hot cache's read + control surface. It reads the source library
    // ONLY to report its path and to compare its volume against the local tier's — it never
    // resolves weights for a load, so it constructs no model root of its own and goes through the
    // same shared source-library contract. Every category is listed because the resolved cache is
    // artifact-shape agnostic: whatever the shared resolver materializes, this surface reports and
    // can remove.
    ModelConsumerInventoryEntry {
        source_files: &["apps/rust-api/src/model_cache.rs"],
        categories: &[
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
        ],
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
        if item_attributes(item)
            .iter()
            .any(|attribute| attribute.path().is_ident("test"))
        {
            return true;
        }

        let predicates = item_attributes(item)
            .iter()
            .filter(|attribute| attribute.path().is_ident("cfg"))
            .map(|attribute| attribute.parse_args::<syn::Meta>())
            .collect::<Result<Vec<_>, _>>();
        let Ok(predicates) = predicates else {
            // A predicate the audit cannot classify remains production-visible so malformed or
            // newly introduced syntax cannot hide a root constructor.
            return false;
        };
        if predicates.is_empty() {
            return false;
        }
        let expression = CfgExpression::All(
            predicates
                .iter()
                .map(CfgExpression::from_meta)
                .collect(),
        );
        expression.requires_test()
    }

    #[derive(Clone, Debug)]
    enum CfgExpression {
        Test,
        Atom(String),
        All(Vec<Self>),
        Any(Vec<Self>),
        Not(Box<Self>),
    }

    impl CfgExpression {
        fn from_meta(meta: &syn::Meta) -> Self {
            match meta {
                syn::Meta::Path(path) if path.is_ident("test") => Self::Test,
                syn::Meta::List(list)
                    if list.path.is_ident("all") || list.path.is_ident("any") =>
                {
                    let nested = list
                        .parse_args_with(
                            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                        )
                        .map(|items| items.iter().map(Self::from_meta).collect::<Vec<_>>());
                    match nested {
                        Ok(items) if list.path.is_ident("all") => Self::All(items),
                        Ok(items) => Self::Any(items),
                        Err(_) => Self::Atom(normalized_meta(meta)),
                    }
                }
                syn::Meta::List(list) if list.path.is_ident("not") => {
                    let nested = list
                        .parse_args_with(
                            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                        )
                        .ok();
                    match nested.as_ref().and_then(|items| {
                        (items.len() == 1).then(|| Self::from_meta(&items[0]))
                    }) {
                        Some(item) => Self::Not(Box::new(item)),
                        None => Self::Atom(normalized_meta(meta)),
                    }
                }
                _ => Self::Atom(normalized_meta(meta)),
            }
        }

        fn collect_atoms(&self, atoms: &mut BTreeSet<String>) {
            match self {
                Self::Test => {}
                Self::Atom(atom) => {
                    atoms.insert(atom.clone());
                }
                Self::All(items) | Self::Any(items) => {
                    for item in items {
                        item.collect_atoms(atoms);
                    }
                }
                Self::Not(item) => item.collect_atoms(atoms),
            }
        }

        fn evaluate(&self, test: bool, atoms: &[String], assignment: u64) -> bool {
            match self {
                Self::Test => test,
                Self::Atom(atom) => atoms
                    .binary_search(atom)
                    .is_ok_and(|index| assignment & (1_u64 << index) != 0),
                Self::All(items) => items
                    .iter()
                    .all(|item| item.evaluate(test, atoms, assignment)),
                Self::Any(items) => items
                    .iter()
                    .any(|item| item.evaluate(test, atoms, assignment)),
                Self::Not(item) => !item.evaluate(test, atoms, assignment),
            }
        }

        fn requires_test(&self) -> bool {
            let mut atoms = BTreeSet::new();
            self.collect_atoms(&mut atoms);
            let atoms = atoms.into_iter().collect::<Vec<_>>();
            // Real cfg expressions in these sources are small. If that changes, retain the item in
            // the production scan rather than accepting an exponential audit or hiding code.
            if atoms.len() > 20 {
                return false;
            }
            let satisfiable = |test| {
                (0..(1_u64 << atoms.len()))
                    .any(|assignment| self.evaluate(test, &atoms, assignment))
            };
            satisfiable(true) && !satisfiable(false)
        }
    }

    fn normalized_meta(meta: &syn::Meta) -> String {
        meta.to_token_stream()
            .to_string()
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect()
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

            #[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
            fn mixed_any_is_production() { let _ = "mixed_any_is_production"; }

            #[cfg(not(test))]
            fn explicit_non_test_is_production() { let _ = "explicit_non_test_is_production"; }

            #[cfg(all(target_os = "macos", feature = "backend-candle"))]
            fn target_and_feature_are_production() { let _ = "target_and_feature_are_production"; }

            #[cfg(custom_predicate(test))]
            fn unsupported_cfg_stays_visible() { let _ = "unsupported_cfg_stays_visible"; }

            #[cfg(all(test, target_os = "macos"))]
            fn all_requires_test() { let _ = "all_requires_test"; }

            #[cfg(all(
                any(test, target_os = "macos"),
                not(all(test, feature = "never"))
            ))]
            fn nested_has_production_assignment() { let _ = "nested_has_production_assignment"; }

            #[cfg(all(
                any(test, all(feature = "same", not(feature = "same"))),
                not(not(test))
            ))]
            fn nested_requires_test() { let _ = "nested_requires_test"; }

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
        assert!(production.contains("mixed_any_is_production"));
        assert!(production.contains("explicit_non_test_is_production"));
        assert!(production.contains("target_and_feature_are_production"));
        assert!(production.contains("unsupported_cfg_stays_visible"));
        assert!(production.contains("nested_has_production_assignment"));
        assert!(!production.contains("all_requires_test"));
        assert!(!production.contains("nested_requires_test"));

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

        let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let image_jobs = std::fs::read_to_string(
            repository.join("crates/sceneworks-worker/src/image_jobs.rs"),
        )
        .expect("read mixed production/test image resolver");
        let production = production_rust_source(&image_jobs);
        assert!(production.contains("resolve_adapter_file"));
        assert!(production.contains("normalize_app_managed_lora_path"));
    }
}
