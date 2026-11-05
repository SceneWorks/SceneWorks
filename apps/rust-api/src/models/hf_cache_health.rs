use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HuggingFaceCacheHealth {
    pub(crate) installed: bool,
    pub(crate) incomplete: bool,
    pub(crate) missing_files: Vec<String>,
}

impl HuggingFaceCacheHealth {
    pub(super) fn missing(missing_files: Vec<String>) -> Self {
        Self {
            installed: false,
            incomplete: true,
            missing_files,
        }
    }

    pub(super) fn installed() -> Self {
        Self {
            installed: true,
            incomplete: false,
            missing_files: Vec::new(),
        }
    }
}

pub(crate) fn huggingface_cache_health(
    repo_root: &FsPath,
    files: &[String],
) -> HuggingFaceCacheHealth {
    if !huggingface_repo_cache_exists(repo_root) {
        return HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: Vec::new(),
        };
    }
    let snapshots = huggingface_snapshot_dirs(repo_root);
    if snapshots.is_empty() {
        // The repo cache dir exists but holds no snapshot revision at all — an empty skeleton
        // (bare refs/blobs, e.g. a download that resolved zero files against an unpublished tier, or
        // a cache whose weights were pruned). Nothing is partially there, so this is MISSING, not a
        // repairable "incomplete": reporting incomplete surfaced a confusing "Cached files are
        // incomplete: snapshots/<revision>" banner for a tier that simply was never downloaded
        // (sc-9909). incomplete:false keeps it a clean not-installed state.
        return HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: vec!["snapshots/<revision>".to_owned()],
        };
    }
    if !files.is_empty() {
        return huggingface_filtered_cache_health(&snapshots, files);
    }

    let mut best_missing = Vec::new();
    for snapshot in snapshots {
        if path_is_readable_file(&snapshot.join("model_index.json")) {
            let health = diffusers_snapshot_health(&snapshot);
            if health.installed {
                return health;
            }
            if best_missing.is_empty() || health.missing_files.len() < best_missing.len() {
                best_missing = health.missing_files;
            }
            continue;
        }
        // A whole-repo (empty `files`) fetch of a non-diffusers repo declares no per-file claim, so
        // "some payload landed" was the only signal. That accepts an interrupted sharded download as
        // installed. The repo's own `*.safetensors.index.json` files are the declared set for their
        // shards, so validate them here (sc-20526).
        //
        // Cost, stated honestly: ONE snapshot walk per candidate snapshot, whose result answers both
        // the payload question and the index hunt. This lane previously short-circuited on a
        // `config.json` stat and walked nothing, so a `config.json`-bearing snapshot now pays a walk
        // it did not pay before. That walk is not removable by making it lazy: FINDING the indexes is
        // the check, and the check must run before this returns `installed`. What each index then
        // costs is one small JSON read plus a stat per distinct shard — no hashing, no header parse,
        // no tensor read. The whole lane runs on the install/catalog sweep, never per job.
        let files_on_disk = snapshot_files(&snapshot);
        if path_is_readable_file(&snapshot.join("config.json"))
            || snapshot_has_payload_file_in(&files_on_disk)
        {
            let torn = torn_shard_indexes_in(&snapshot, &files_on_disk);
            if torn.is_empty() {
                return HuggingFaceCacheHealth::installed();
            }
            if best_missing.is_empty() || torn.len() < best_missing.len() {
                best_missing = torn;
            }
            continue;
        }
        if best_missing.is_empty() {
            best_missing.push("model_index.json".to_owned());
        }
    }
    HuggingFaceCacheHealth::missing(best_missing)
}

/// If every pattern in a tier's `files` filter is confined to ONE leading directory — the standard
/// quant-tier layout `["q8/*"]` → `q8` (also `["bf16/*"]`, `["q4/*"]`) — return that directory.
/// `None` for a flat single-variant filter (`["*.safetensors"]`, whose leading component is itself a
/// glob) or patterns that span multiple top-level dirs; those are not a tier subdir and keep the
/// coarse glob check.
pub(super) fn tier_subdir_name(files: &[String]) -> Option<String> {
    let mut tier: Option<&str> = None;
    for pattern in files {
        let (head, rest) = pattern.split_once('/')?;
        if head.is_empty() || rest.is_empty() || pattern_contains_glob(head) {
            return None;
        }
        match tier {
            None => tier = Some(head),
            Some(existing) if existing == head => {}
            Some(_) => return None,
        }
    }
    tier.map(str::to_owned)
}

/// The tier a `<dir>/*` whole-subdir glob names (`"q8/*"` → `"q8"`). `None` for any other pattern —
/// a specific file (`"q8/turbo_lora.safetensors"`) or a non-tier glob — so the coarse presence check
/// stays authoritative for explicit files.
pub(super) fn whole_subdir_glob_tier(pattern: &str) -> Option<&str> {
    let (head, rest) = pattern.split_once('/')?;
    (rest == "*" && !head.is_empty() && !pattern_contains_glob(head)).then_some(head)
}

pub(super) fn huggingface_filtered_cache_health(
    snapshots: &[PathBuf],
    files: &[String],
) -> HuggingFaceCacheHealth {
    let mut missing = files
        .iter()
        .filter(|pattern| {
            !snapshots
                .iter()
                .any(|snapshot| snapshot_contains_pattern(snapshot, pattern))
        })
        .cloned()
        .collect::<Vec<_>>();
    // Whether the COARSE check found none of the filter's patterns present — the "cleanly absent
    // tier" signal, captured before the tier-completeness augmentation below can add entries.
    let coarse_all_absent = missing.len() == files.len();

    // A flat sharded Transformers checkpoint commonly declares an index plus `model-*.safetensors`.
    // The glob is satisfied by the FIRST shard, so the coarse pattern check alone can call a torn
    // 15-shard model installed. Evaluate every candidate revision as one coherent snapshot and fold
    // in the shard set declared by its selected index. This reuses the whole-repo/receipt validator;
    // no tensor bytes or headers are read, only the small index and one stat per named shard.
    let has_flat_shard_glob = files.iter().any(|pattern| {
        !pattern.contains('/')
            && pattern_contains_glob(pattern)
            && pattern.to_ascii_lowercase().ends_with(".safetensors")
    });
    let indexed_health = has_flat_shard_glob.then(|| {
        snapshots
            .iter()
            .filter_map(|snapshot| {
                let files_on_disk = snapshot_files(snapshot);
                let selected_indexes = files_on_disk
                    .iter()
                    .filter(|file| {
                        file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX)
                            && files.iter().any(|pattern| pattern_matches(pattern, file))
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                if selected_indexes.is_empty() {
                    return None;
                }
                let mut candidate_missing = files
                    .iter()
                    .filter(|pattern| !snapshot_contains_pattern(snapshot, pattern))
                    .cloned()
                    .collect::<Vec<_>>();
                for shard in torn_shard_indexes_in(snapshot, &selected_indexes) {
                    if !candidate_missing.contains(&shard) {
                        candidate_missing.push(shard);
                    }
                }
                Some(candidate_missing)
            })
            .min_by_key(Vec::len)
    });
    if let Some(Some(indexed_missing)) = indexed_health {
        // Required metadata, index and shards must coexist in ONE revision. Replacing the coarse
        // cross-snapshot result also prevents different revisions from satisfying different files.
        missing = indexed_missing;
    }

    // Flat diffusers snapshots (Mage-Flow's logical q4/q8/bf16 load-time choices) list the
    // root `model_index.json` plus component globs rather than one `<tier>/*` subdir. The coarse
    // glob check only proves that each directory contains *something*; it does not prove the
    // model_index is valid or that every declared component has valid config + weights. Apply the
    // same full snapshot health used by the unfiltered path whenever this filter selects the root
    // model index. Multiple cached revisions are alternatives: one complete revision is enough.
    let flat_diffusers = snapshots
        .iter()
        .filter(|snapshot| {
            path_is_readable_file(&snapshot.join("model_index.json"))
                && files
                    .iter()
                    .any(|pattern| pattern_matches(pattern, "model_index.json"))
        })
        .collect::<Vec<_>>();
    if !flat_diffusers.is_empty() {
        let health = flat_diffusers
            .iter()
            .map(|snapshot| diffusers_snapshot_health(snapshot))
            .min_by_key(|health| health.missing_files.len())
            .expect("non-empty flat diffusers candidates");
        if !health.installed {
            for component in health.missing_files {
                if !missing.contains(&component) {
                    missing.push(component);
                }
            }
        }
    }

    // A `<tier>/*` whole-subdir glob is satisfied as soon as a SINGLE file under `<tier>/` exists, so
    // the coarse check never notices missing weights INSIDE the tier: a torn download (its
    // `model_index.json` + a config or two present, but the transformer/vae weights gone) reported a
    // green "Installed" badge, then failed to load at generation (`No such file or directory`). When a
    // whole-subdir tier is a diffusers pipeline (has `<tier>/model_index.json`), fold its missing
    // weight-bearing components — the SAME per-component check the non-tiered path uses — into
    // `missing`, scoped under the tier. Additional explicit patterns (e.g. a `<tier>/lora.safetensors`
    // co-requisite) are left to the coarse check above, so this never masks an explicitly-listed file.
    // A cleanly-absent tier (no `<tier>/model_index.json` present) adds nothing and stays MISSING, not
    // a repairable "incomplete" — so a valid single-quant install raises no spurious repair prompt
    // (sc-9907/sc-9909).
    for pattern in files {
        let Some(tier) = whole_subdir_glob_tier(pattern) else {
            continue;
        };
        // Evaluate the MOST-COMPLETE cached revision for this tier, not merely the one
        // `huggingface_snapshot_dirs` happens to rank first. A quant-matrix repo keeps every
        // revision's tiers in ONE shared cache, so after a manifest revision bump a freshly
        // re-downloaded complete tier coexists with the older torn snapshot it replaced. Picking
        // the first snapshot (which `refs/main` or a higher file count can front) let that stale
        // torn revision mask the complete one and report a permanent false "incomplete" — e.g. the
        // FLUX.2 MLX re-hosts whose old snapshots predated the `scheduler/` config (GH #1858): even
        // after re-download the card stayed incomplete because the old snapshot was fronted. Take the
        // fewest-missing tier across cached revisions — the SAME `min_by_key` the flat-diffusers and
        // unfiltered paths already use — so "is this tier complete in ANY cached revision?" is the
        // question, matching that a single valid install makes the model usable.
        let Some(health) = snapshots
            .iter()
            .map(|snapshot| snapshot.join(tier))
            .filter(|dir| path_is_readable_file(&dir.join("model_index.json")))
            .map(|dir| diffusers_snapshot_health(&dir))
            .min_by_key(|health| health.missing_files.len())
        else {
            continue;
        };
        for component in health.missing_files {
            let scoped = format!("{tier}/{component}");
            if !missing.contains(&scoped) {
                missing.push(scoped);
            }
        }
    }

    if missing.is_empty() {
        // Every expected file/pattern is present AND (for a diffusers tier) its weights are on disk.
        HuggingFaceCacheHealth::installed()
    } else if coarse_all_absent {
        // NONE of this filter's expected patterns are present: the tier is cleanly absent, not torn.
        // A quant-matrix model keeps every tier in ONE shared repo cache (bf16/, q8/, q4/ subdirs),
        // so downloading one tier populates the repo snapshot the OTHER tiers' filters also probe.
        // Reporting a not-downloaded tier as `incomplete` is what surfaced a false "Cached files are
        // incomplete" warning + Fix button for a perfectly valid single-quant install (sc-9907).
        // "You didn't download this tier" (missing) must stay distinct from "this tier is
        // half-downloaded" (incomplete) so nothing upstream raises a spurious repair prompt.
        HuggingFaceCacheHealth {
            installed: false,
            incomplete: false,
            missing_files: missing,
        }
    } else {
        // Some expected files present but not all — an explicit file is absent, or a diffusers tier is
        // torn (weights missing). A re-fetch repairs it.
        HuggingFaceCacheHealth::missing(missing)
    }
}

pub(super) fn snapshot_contains_pattern(snapshot: &FsPath, pattern: &str) -> bool {
    if pattern_contains_glob(pattern) {
        return snapshot_files(snapshot)
            .into_iter()
            .any(|path| pattern_matches(pattern, &path));
    }
    path_is_readable_file(&snapshot.join(pattern))
}

pub(super) fn pattern_contains_glob(pattern: &str) -> bool {
    pattern
        .chars()
        .any(|character| matches!(character, '*' | '?' | '[' | ']'))
}

pub(super) fn diffusers_snapshot_health(snapshot: &FsPath) -> HuggingFaceCacheHealth {
    let model_index_path = snapshot.join("model_index.json");
    let Ok(contents) = std::fs::read_to_string(&model_index_path) else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };
    let Ok(index) = serde_json::from_str::<Value>(&contents) else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };
    let Some(index) = index.as_object() else {
        return HuggingFaceCacheHealth::missing(vec!["model_index.json".to_owned()]);
    };

    let mut missing = Vec::new();
    let is_mage = index
        .get("_class_name")
        .and_then(Value::as_str)
        .is_some_and(|value| value == "MageFlowPipeline");
    if is_mage
        && sceneworks_core::lora_family::detect_model_family(snapshot)
            .ok()
            .flatten()
            .as_deref()
            != Some("mage-flow")
    {
        missing.push("transformer/config.json (invalid Mage config)".to_owned());
    }
    for (component, spec) in index {
        if component.starts_with('_') || spec.is_null() {
            continue;
        }
        let class_name = spec
            .as_array()
            .and_then(|items| items.get(1))
            .and_then(Value::as_str)
            .unwrap_or_default();
        // diffusers records optional components that the pipeline doesn't use
        // as `[null, null]` (e.g. ChromaPipeline's `feature_extractor` and
        // `image_encoder`). These have no directory or files on disk by design,
        // so an empty class name means "absent" — skip it rather than reporting
        // its config/weights as missing and marking the whole model incomplete.
        if class_name.is_empty() {
            continue;
        }
        if diffusers_component_requires_weights(component, class_name) {
            // Weight-bearing components (unet, transformer, vae, text_encoder,
            // controlnet, …) reliably ship a `config.json` alongside their
            // weight files, so require both.
            if !path_is_valid_json_object(&snapshot.join(format!("{component}/config.json"))) {
                missing.push(format!("{component}/config.json"));
            }
            // ONE listing of the component directory, asked two questions below (sc-20526).
            let component_entries = diffusers_component_entries(snapshot, component);
            if !diffusers_component_has_weight_file_in(&component_entries) {
                missing.push(format!("{component}/<weights>"));
            } else if is_mage && !diffusers_component_safetensors_are_valid(snapshot, component) {
                missing.push(format!("{component}/<weights> (malformed safetensors)"));
            } else {
                // A sharded component holds "a weight file" as soon as ONE shard landed. The
                // component's own index declares the rest, so an interrupted download is only
                // visible by comparing `weight_map` against disk (sc-20526).
                for shard in
                    component_missing_shards_in(&snapshot.join(component), &component_entries)
                {
                    missing.push(format!("{component}/{shard}"));
                }
            }
        } else if is_mage && component == "tokenizer" {
            // Mage's Qwen3-VL AutoProcessor is a logical `tokenizer` component in model_index.json,
            // but the published diffusers snapshots colocate its tokenizer + vision processor
            // configs under `text_encoder/` beside the Qwen3-VL weights.
            for config in ["tokenizer_config.json", "preprocessor_config.json"] {
                if !path_is_valid_json_object(&snapshot.join("text_encoder").join(config)) {
                    missing.push(format!("text_encoder/{config}"));
                }
            }
        } else if !diffusers_component_has_valid_config_file(snapshot, component) {
            // Weightless auxiliary components (scheduler, tokenizer, feature
            // extractors, and image/video/composite processors) ship config
            // files whose names vary by class — scheduler_config.json,
            // tokenizer_config.json, preprocessor_config.json, and more. Hard
            // coding each variant is what produced repeated false "incomplete"
            // reports (Chroma's null optionals, Qwen2VLProcessor), so only
            // require the component directory to exist and hold at least one
            // file. A genuinely missing/partial component still trips this.
            missing.push(format!("{component}/<config>"));
        }
    }
    if missing.is_empty() {
        HuggingFaceCacheHealth::installed()
    } else {
        missing.sort();
        missing.dedup();
        HuggingFaceCacheHealth::missing(missing)
    }
}

/// Classifies a diffusers `model_index.json` component as weight-bearing.
/// Schedulers, tokenizers, feature extractors, and composite `*Processor`
/// wrappers (e.g. Qwen2VLProcessor) carry no model weights — `contains("processor")`
/// subsumes `imageprocessor` and the composite processors.
pub(super) fn diffusers_component_requires_weights(component: &str, class_name: &str) -> bool {
    let class = class_name.to_ascii_lowercase();
    !(component.contains("scheduler")
        || class.contains("scheduler")
        || class.contains("tokenizer")
        || class.contains("featureextractor")
        || class.contains("processor"))
}

pub(super) fn path_is_valid_json_object(path: &FsPath) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|contents| serde_json::from_slice::<Value>(&contents).ok())
        .is_some_and(|value| value.is_object())
}

/// Weightless diffusers components use different JSON config filenames, but every shipped
/// scheduler/tokenizer/processor has at least one JSON object. Requiring a parseable object rejects
/// an empty or corrupted config without hard-coding class-specific filenames.
pub(super) fn diffusers_component_has_valid_config_file(
    snapshot: &FsPath,
    component: &str,
) -> bool {
    std::fs::read_dir(snapshot.join(component))
        .map(|entries| {
            entries.flatten().any(|entry| {
                let path = entry.path();
                !is_hidden_file(&path)
                    && path.extension().and_then(|value| value.to_str()) == Some("json")
                    && path_is_valid_json_object(&path)
            })
        })
        .unwrap_or(false)
}

/// Every non-hidden readable entry directly inside a diffusers component directory.
///
/// The weight-file probe and the shard-index probe ask different questions of the SAME listing, so
/// the health loop reads the directory once and hands the result to both (sc-20526). Threading the
/// list is what makes the "one `read_dir` per component" cost claim true rather than aspirational.
pub(super) fn diffusers_component_entries(snapshot: &FsPath, component: &str) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(snapshot.join(component)) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| !is_hidden_file(path) && path_is_readable_file(path))
        .collect()
}

pub(super) fn diffusers_component_has_weight_file_in(entries: &[PathBuf]) -> bool {
    entries.iter().any(|path| {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        name.ends_with(".safetensors")
            || name.ends_with(".bin")
            || name.ends_with(".msgpack")
            || name.ends_with(".gguf")
    })
}

/// Shards that a component directory's `*.safetensors.index.json` files name but that are not on
/// disk, reported relative to the component (sc-20526).
///
/// A sharded component satisfies [`diffusers_component_has_weight_file_in`] as soon as ONE shard
/// landed, so an interrupted download left `model.safetensors.index.json` + the last shard and read
/// "installed" — then died at the first forward pass with `cannot find tensor
/// model.embed_tokens.weight` (lens_turbo bf16). The index IS the declared file set for a sharded
/// component, so comparing it against disk is the workable ground truth even when the manifest
/// declares only a `<tier>/*` glob or a whole-repo fetch.
///
/// Cost: no directory read at all — `entries` is the listing
/// [`diffusers_component_entries`] already spent for the weight-file probe — plus one small JSON
/// read per index and a stat per distinct shard. No hashing, no header parse, no tensor read.
pub(super) fn component_missing_shards_in(dir: &FsPath, entries: &[PathBuf]) -> Vec<String> {
    let mut missing = Vec::new();
    for path in entries
        .iter()
        .filter(|path| sceneworks_core::safetensors::is_safetensors_index_path(path))
    {
        missing.extend(sceneworks_core::safetensors::missing_indexed_shards(
            dir, path,
        ));
    }
    missing.sort();
    missing.dedup();
    missing
}

/// Whether every `*.safetensors.index.json` NAMED IN `files` (snapshot-relative paths, as a receipt's
/// `resolvedFiles` or a backfill's resolved set records them) has all of its shards on disk.
///
/// This is the receipt-lane half of the shard check and costs nothing beyond a stat per shard: the
/// file list is already in hand, so no directory walk is needed to find the indexes (sc-20526).
pub(super) fn listed_shard_indexes_are_complete(snapshot: &FsPath, files: &[String]) -> bool {
    files
        .iter()
        .filter(|file| file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX))
        .all(|file| {
            let path = snapshot.join(file);
            path.parent().is_some_and(|dir| {
                sceneworks_core::safetensors::indexed_shards_are_present(dir, &path)
            })
        })
}

pub(super) fn diffusers_component_safetensors_are_valid(
    snapshot: &FsPath,
    component: &str,
) -> bool {
    let files = snapshot_files(&snapshot.join(component))
        .into_iter()
        .filter(|path| path.to_ascii_lowercase().ends_with(".safetensors"))
        .collect::<Vec<_>>();
    !files.is_empty()
        && files.iter().all(|path| {
            sceneworks_core::safetensors::file_is_structurally_valid(
                &snapshot.join(component).join(path),
            )
        })
}

/// Whether an already-walked snapshot file list holds anything that is not documentation/artwork.
///
/// Takes the list rather than the directory so the whole-repo health probe walks once and reuses the
/// result for both the payload question and the shard-index validation (sc-20526).
pub(super) fn snapshot_has_payload_file_in(files: &[String]) -> bool {
    files.iter().any(|path| {
        let lower = path.to_ascii_lowercase();
        !lower.ends_with(".md")
            && !lower.ends_with(".png")
            && !lower.ends_with(".jpg")
            && !lower.ends_with(".jpeg")
            && !lower.ends_with(".gitattributes")
    })
}

/// Snapshot-relative shard paths that a `*.safetensors.index.json` in `files` names but that are not
/// on disk. Takes the already-walked file list so no extra directory traversal is spent (sc-20526).
pub(super) fn torn_shard_indexes_in(snapshot: &FsPath, files: &[String]) -> Vec<String> {
    let mut missing = files
        .iter()
        .filter(|file| file.ends_with(sceneworks_core::safetensors::SAFETENSORS_INDEX_SUFFIX))
        .flat_map(|file| {
            let path = snapshot.join(file);
            let dir = path.parent().map(FsPath::to_path_buf).unwrap_or_default();
            let prefix = match file.rsplit_once('/') {
                Some((parent, _)) => format!("{parent}/"),
                None => String::new(),
            };
            sceneworks_core::safetensors::missing_indexed_shards(&dir, &path)
                .into_iter()
                .map(move |shard| format!("{prefix}{shard}"))
        })
        .collect::<Vec<_>>();
    missing.sort();
    missing.dedup();
    missing
}

/// Every readable file under `snapshot`, snapshot-relative, `/`-separated.
///
/// Hidden entries are excluded. They are not payload, and — because this list backs
/// [`snapshot_contains_pattern`]'s glob branch — a `._model.safetensors` sidecar would
/// otherwise satisfy a required `*.safetensors` pattern, reporting a model installed
/// while its real weights file is absent (SceneWorks#1333).
pub(super) fn snapshot_files(snapshot: &FsPath) -> Vec<String> {
    let mut output = Vec::new();
    let mut stack = vec![snapshot.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_hidden_file(&path) {
                continue;
            } else if path_is_readable_file(&path) {
                if let Ok(relative) = path.strip_prefix(snapshot) {
                    output.push(relative.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }
    output
}

pub(super) fn path_is_readable_file(path: &FsPath) -> bool {
    if std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file()) {
        return true;
    }
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return false;
    };
    if !metadata.file_type().is_symlink() {
        return false;
    }
    std::fs::File::open(path).is_ok()
}
