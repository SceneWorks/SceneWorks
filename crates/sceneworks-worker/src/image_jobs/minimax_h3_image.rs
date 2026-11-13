//! MiniMax-H3-Image (SW-7100, feature SW-F363): the worker's half of the `minimax_h3_image`
//! text-to-image route on both native lanes.
//!
//! The engine (`mlx-gen-minimax-h3::h3_image`, `candle-gen-minimax-h3::image_model`) loads from a
//! `MiniMaxAI/MiniMax-H3-Image` snapshot ROOT holding `vae_diffusers/` and `tokenizer/`, and takes
//! two staged components:
//!
//! * `transformer_diffusers` — the tier's DiT directory. bf16 is the root's own
//!   `transformer_diffusers/` (nothing staged); q8/q4 are `{tier}/transformer_diffusers/` of the
//!   SceneWorks re-host (SW-7093).
//! * `text_encoder` — MiniMax-H3's OWN text encoder at the same tier (the catalog's tier-scoped
//!   `text_encoder` co-requisite rows are H3's rows verbatim), never a copy in this repository.
//!
//! So the tier names a directory that is NOT the load root. [`resolve_weights_dir`] returns the
//! TIER directory (bf16: the upstream snapshot root; q8/q4: `<re-host>/{tier}`), which is what the
//! shared down-tier ladder enumerates and [`tier_of_dir`] reads back, and [`attach_components`]
//! (called from the generic `attach_required_components` seam on every lane) re-roots the spec on
//! the upstream snapshot and stages both components. Repos and revisions come from the catalog
//! rows (E29), never from constants here; a `pendingArtifact` tier resolves to nothing.

use super::ImageRequest;
use crate::model_jobs::huggingface_pinned_snapshot_dir;
use crate::{JsonObject, Path, PathBuf, Settings, Value, WorkerError, WorkerResult};
use gen_core::{LoadSpec, Quant, WeightsSource};
use sceneworks_core::model_artifacts::artifact_selection::{
    is_pending_artifact_download, model_download_for_variant,
};

/// The catalog and engine id.
pub(crate) const MODEL_ID: &str = "minimax_h3_image";
/// The staged DiT component (the tier's `transformer_diffusers/`).
pub(crate) const DIT_COMPONENT: &str = "transformer_diffusers";
/// The staged text-encoder component (MiniMax-H3's tier directory).
pub(crate) const TEXT_ENCODER_COMPONENT: &str = "text_encoder";
/// The tier-agnostic co-requisite whose snapshot is the load root (`vae_diffusers/`).
const ROOT_COMPONENT: &str = "image_vae";
/// Tiers, densest first: an unselected request takes the densest installed one.
const TIERS: [&str; 3] = ["bf16", "q8", "q4"];

/// Whether `model` is this route.
pub(crate) fn is_model(model: &str) -> bool {
    model == MODEL_ID
}

/// The engine's load quant for a tier: `None` for the dense upstream checkpoint.
pub(crate) fn tier_quant(tier: &str) -> Option<Quant> {
    match tier {
        "q8" => Some(Quant::Q8),
        "q4" => Some(Quant::Q4),
        _ => None,
    }
}

fn entry_value(entry: &JsonObject) -> Value {
    Value::Object(entry.clone())
}

/// The installable primary row for `tier`, or `None` when the catalog declares none or it is
/// still a `pendingArtifact` placeholder.
fn primary_row(entry: &JsonObject, tier: &str) -> Option<Value> {
    model_download_for_variant(&entry_value(entry), tier)
        .filter(|row| !is_pending_artifact_download(row))
}

fn row_identity(row: &Value) -> Option<(&str, &str)> {
    Some((
        row.get("repo").and_then(Value::as_str)?,
        row.get("revision").and_then(Value::as_str)?,
    ))
}

/// The tier subdirectory a packed row's `files` name (`q8/transformer_diffusers/*` ⇒ `q8`), or
/// `None` for the upstream row whose DiT sits at the snapshot root.
fn row_tier_subdir(row: &Value) -> Option<String> {
    let first = row.get("files")?.as_array()?.first()?.as_str()?;
    let (head, _) = first.split_once('/')?;
    (head != DIT_COMPONENT && sceneworks_core::ids::is_safe_id(head)).then(|| head.to_owned())
}

/// Where `tier`'s directory is in the cache, or `None` when the tier is not declared, pending, or
/// its pinned snapshot is absent. The directory need not hold a complete DiT; see [`installed`].
pub(crate) fn tier_dir(data_dir: &Path, entry: &JsonObject, tier: &str) -> Option<PathBuf> {
    let row = primary_row(entry, tier)?;
    let (repo, revision) = row_identity(&row)?;
    let snapshot = huggingface_pinned_snapshot_dir(data_dir, repo, revision)?;
    Some(match row_tier_subdir(&row) {
        Some(subdir) => snapshot.join(subdir),
        None => snapshot,
    })
}

/// A tier directory holds a loadable DiT when its `transformer_diffusers/config.json` is present.
fn installed(dir: &Path) -> bool {
    dir.join(DIT_COMPONENT).join("config.json").is_file()
}

/// The tier a resolved directory IS, read back from the catalog rows: the upstream snapshot root
/// is bf16, `<re-host snapshot>/{q8,q4}` is that tier. `None` for any other directory.
pub(crate) fn tier_of_dir(entry: &JsonObject, dir: &Path) -> Option<&'static str> {
    if entry.get("id").and_then(Value::as_str) != Some(MODEL_ID) {
        return None;
    }
    TIERS.into_iter().find(|tier| {
        let Some(row) = primary_row(entry, tier) else {
            return false;
        };
        let Some((repo, revision)) = row_identity(&row) else {
            return false;
        };
        let snapshot = match row_tier_subdir(&row) {
            Some(subdir) => {
                if dir.file_name().and_then(|name| name.to_str()) != Some(subdir.as_str()) {
                    return false;
                }
                dir.parent()
            }
            None => Some(dir),
        };
        snapshot.is_some_and(|snapshot| snapshot_names(snapshot, repo, revision))
    })
}

/// Whether `snapshot` is `…/models--<org>--<name>/snapshots/<revision>`.
fn snapshot_names(snapshot: &Path, repo: &str, revision: &str) -> bool {
    let named = |path: Option<&Path>, want: &str| {
        path.and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == want)
    };
    named(Some(snapshot), revision)
        && named(snapshot.parent(), "snapshots")
        && named(
            snapshot.parent().and_then(Path::parent),
            &format!("models--{}", repo.replace('/', "--")),
        )
}

/// The tier an explicit `advanced.mlxQuantize` names (`<= 0` or `>= 9` ⇒ bf16, `1..=4` ⇒ q4,
/// `5..=8` ⇒ q8), or `None` when the request names none.
fn explicit_tier(request: &ImageRequest) -> WorkerResult<Option<&'static str>> {
    let Some(value) = request
        .advanced
        .get("mlxQuantize")
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let bits = crate::payload::quant_int(value).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{MODEL_ID} advanced.mlxQuantize must be an integer bit count, got {value}"
        ))
    })?;
    Ok(Some(match bits {
        bits if bits <= 0 || bits >= 9 => "bf16",
        bits if bits <= 4 => "q4",
        _ => "q8",
    }))
}

/// The tier directory this request renders from (see the module docs): an explicit tier is honoured
/// or refused by name, never substituted; with none, the densest installed tier. `Ok(None)` when no
/// tier is installed and none was asked for, so the caller's ordinary "install the model" path owns
/// the error.
pub(crate) fn resolve_weights_dir(
    request: &ImageRequest,
    settings: &Settings,
) -> WorkerResult<Option<PathBuf>> {
    resolve_tier_dir(
        &request.model_manifest_entry,
        &settings.data_dir,
        explicit_tier(request)?,
    )
}

/// [`resolve_weights_dir`] for a caller that already knows the explicit tier (or `None`): an
/// explicit tier is honoured or refused by name, never substituted; with none, the densest
/// installed tier. Shared with the H3-Image trainer (SW-7103), so training resolves the same tier
/// directory a render of the same selection would.
pub(crate) fn resolve_tier_dir(
    entry: &JsonObject,
    data_dir: &Path,
    explicit: Option<&str>,
) -> WorkerResult<Option<PathBuf>> {
    let installed_dir = |tier: &str| tier_dir(data_dir, entry, tier).filter(|dir| installed(dir));
    if let Some(tier) = explicit {
        if let Some(dir) = installed_dir(tier) {
            return Ok(Some(dir));
        }
        let declared = primary_row(entry, tier).is_some();
        return Err(WorkerError::InvalidPayload(if declared {
            format!(
                "{MODEL_ID} tier {tier} is not installed; install it in Model Manager or pick an \
                 installed tier (an explicit tier selection is never substituted with another)"
            )
        } else {
            format!(
                "{MODEL_ID} tier {tier} is not published yet; pick an installed tier (an explicit \
                 tier selection is never substituted with another)"
            )
        }));
    }
    Ok(TIERS.into_iter().find_map(installed_dir))
}

/// The load root: the pinned snapshot of the tier-agnostic `image_vae` co-requisite, which must
/// hold `vae_diffusers/` and `tokenizer/`.
fn load_root(data_dir: &Path, entry: &JsonObject) -> WorkerResult<PathBuf> {
    let row = co_requisite_rows(entry, ROOT_COMPONENT)
        .next()
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{MODEL_ID}: the catalog entry declares no `{ROOT_COMPONENT}` co-requisite — the \
                 model manifest entry is misconfigured"
            ))
        })?;
    let (repo, revision) = row_identity(row).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{MODEL_ID}: the `{ROOT_COMPONENT}` co-requisite declares no repo/revision"
        ))
    })?;
    let root = huggingface_pinned_snapshot_dir(data_dir, repo, revision)
        .filter(|root| {
            root.join("vae_diffusers").join("config.json").is_file()
                && root.join("tokenizer").join("tokenizer.json").is_file()
        })
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{MODEL_ID}: its image VAE and tokenizer ({repo}) are not installed — install \
                 {MODEL_ID} from the Model Manager"
            ))
        })?;
    Ok(root)
}

fn co_requisite_rows<'a>(
    entry: &'a JsonObject,
    component: &'a str,
) -> impl Iterator<Item = &'a Value> + 'a {
    entry
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(move |row| {
            row.get("coRequisite").and_then(Value::as_bool) == Some(true)
                && row.get("componentId").and_then(Value::as_str) == Some(component)
        })
}

/// MiniMax-H3's text encoder at `tier`: the tier-scoped `text_encoder` co-requisite row (one per
/// tier on this host — the catalog is already narrowed to the running OS), its pinned snapshot
/// narrowed by `subdir`.
fn text_encoder_dir(data_dir: &Path, entry: &JsonObject, tier: &str) -> WorkerResult<PathBuf> {
    let row = co_requisite_rows(entry, TEXT_ENCODER_COMPONENT)
        .find(|row| {
            row.get("variant")
                .and_then(Value::as_str)
                .is_some_and(|variant| variant.eq_ignore_ascii_case(tier))
        })
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{MODEL_ID}: no `text_encoder` co-requisite is declared for the {tier} tier — the \
                 model manifest entry is misconfigured"
            ))
        })?;
    let (repo, revision) = row_identity(row).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{MODEL_ID}: the {tier} `text_encoder` co-requisite declares no repo/revision"
        ))
    })?;
    let subdir = row
        .get("subdir")
        .and_then(Value::as_str)
        .unwrap_or(TEXT_ENCODER_COMPONENT);
    let dir = huggingface_pinned_snapshot_dir(data_dir, repo, revision)
        .map(|snapshot| crate::safe_join(&snapshot, subdir))
        .transpose()?
        .filter(|dir| dir.join("config.json").is_file())
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{MODEL_ID}: its {tier} text encoder ({repo}/{subdir}, shared with MiniMax-H3) is \
                 not installed — install {MODEL_ID}'s {tier} tier from the Model Manager"
            ))
        })?;
    Ok(dir)
}

/// Re-root a spec built on a [`resolve_weights_dir`] tier directory onto the upstream snapshot, and
/// stage the tier's DiT and MiniMax-H3's text encoder. The load quant names the staged tier (the
/// engine's on-disk `quantization` marker stays authoritative).
pub(crate) fn attach_components(
    spec: LoadSpec,
    entry: &JsonObject,
    settings: &Settings,
) -> WorkerResult<LoadSpec> {
    attach_components_in(spec, entry, &settings.data_dir)
}

/// [`attach_components`] against an app data directory.
fn attach_components_in(
    mut spec: LoadSpec,
    entry: &JsonObject,
    data_dir: &Path,
) -> WorkerResult<LoadSpec> {
    let WeightsSource::Dir(dir) = &spec.weights else {
        return Err(WorkerError::InvalidPayload(format!(
            "{MODEL_ID}: expected a resolved tier directory, not a single file"
        )));
    };
    let tier_directory = dir.clone();
    let tier = tier_of_dir(entry, &tier_directory).ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{MODEL_ID}: {} is not one of this model's declared tier directories",
            tier_directory.display()
        ))
    })?;
    let root = load_root(data_dir, entry)?;
    let text_encoder = text_encoder_dir(data_dir, entry, tier)?;
    spec.weights = WeightsSource::Dir(root);
    spec.quantize = tier_quant(tier);
    let mut spec = spec.with_component(TEXT_ENCODER_COMPONENT, WeightsSource::Dir(text_encoder));
    if tier != "bf16" {
        spec = spec.with_component(
            DIT_COMPONENT,
            WeightsSource::Dir(tier_directory.join(DIT_COMPONENT)),
        );
    }
    Ok(spec)
}

/// The trainer's [`LoadSpec`] for a MiniMax-H3-Image LoRA/LoKr run (SW-7103): the shipped catalog
/// entry narrowed to this OS, the tier [`resolve_tier_dir`] picks for `explicit` (the plan's
/// `advanced.baseTier`; `None` = densest installed), re-rooted and staged by [`attach_components`]
/// exactly as a render of that tier is — so the trainer sees the same snapshot root, DiT tier,
/// text encoder and load quant the generator does. Refuses when no tier is installed.
pub(crate) fn training_load_spec(
    data_dir: &Path,
    explicit: Option<&str>,
) -> WorkerResult<LoadSpec> {
    let mut entry =
        crate::training_jobs::builtin_model_manifest_entry(MODEL_ID).ok_or_else(|| {
            WorkerError::InvalidPayload(format!("{MODEL_ID} has no builtin catalog entry"))
        })?;
    sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
        &mut entry,
        std::env::consts::OS,
    );
    let entry = entry.as_object().cloned().ok_or_else(|| {
        WorkerError::InvalidPayload(format!("{MODEL_ID} catalog entry is not an object"))
    })?;
    training_load_spec_in(&entry, data_dir, explicit)
}

/// [`training_load_spec`] against an already-narrowed catalog entry.
fn training_load_spec_in(
    entry: &JsonObject,
    data_dir: &Path,
    explicit: Option<&str>,
) -> WorkerResult<LoadSpec> {
    let tier_directory = resolve_tier_dir(entry, data_dir, explicit)?.ok_or_else(|| {
        WorkerError::InvalidPayload(format!(
            "{MODEL_ID} training needs an installed tier — install {MODEL_ID} from the Model \
             Manager"
        ))
    })?;
    attach_components_in(
        LoadSpec::new(WeightsSource::Dir(tier_directory)),
        entry,
        data_dir,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const IMAGE_REPO: &str = "MiniMaxAI/MiniMax-H3-Image";
    const IMAGE_REV: &str = "492fd8c4eca666f1085f57e30738c1e0d86b3439";

    fn shipped_entry() -> JsonObject {
        let manifest =
            sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc").unwrap();
        let mut entry = manifest["models"]
            .as_array()
            .unwrap()
            .iter()
            .find(|model| model["id"] == MODEL_ID)
            .unwrap()
            .clone();
        sceneworks_core::model_artifacts::artifact_selection::retain_downloads_for_os(
            &mut entry,
            std::env::consts::OS,
        );
        entry.as_object().unwrap().clone()
    }

    /// Registers a private hub cache under the test's temp dir for `data`, so no resolution or
    /// fixture write can reach the real (shared) Hugging Face cache. Asserts that before returning.
    fn isolated_cache(
        temporary: &Path,
    ) -> (
        PathBuf,
        sceneworks_core::hf_home::test_support::CacheRootGuard,
    ) {
        let data = temporary.join("data");
        let hub = temporary.join("private-hub");
        let guard =
            sceneworks_core::hf_home::test_support::CacheRootGuard::new(data.clone(), hub.clone());
        let resolved = crate::huggingface_hub_cache_dir(&data);
        assert_eq!(resolved, hub, "the guard must override HF_HOME");
        assert!(
            resolved.starts_with(temporary),
            "synthetic snapshots must stay inside the guarded temporary root"
        );
        (data, guard)
    }

    fn snapshot(data: &Path, repo: &str, revision: &str) -> PathBuf {
        let dir = crate::huggingface_hub_cache_dir(data)
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(revision);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn touch(path: PathBuf) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"{}").unwrap();
    }

    fn settings(data: &Path) -> Settings {
        Settings::for_test(data.to_path_buf())
    }

    fn request(entry: &JsonObject, advanced: Value) -> ImageRequest {
        ImageRequest::from_payload(
            json!({
                "projectId": "p",
                "model": MODEL_ID,
                "prompt": "a lighthouse",
                "modelManifestEntry": entry,
                "advanced": advanced,
            })
            .as_object()
            .unwrap(),
        )
    }

    /// The bf16 tier stages H3's dense text encoder and no DiT component (the root's own
    /// `transformer_diffusers/` is the tier); the packed tiers are pending, so an explicit q4 pick
    /// is refused by name rather than substituted.
    #[test]
    fn bf16_resolves_the_upstream_root_and_stages_h3s_text_encoder() {
        let temporary = tempfile::tempdir().unwrap();
        let (data, _guard) = isolated_cache(temporary.path());
        let settings = settings(&data);
        let entry = shipped_entry();
        let root = snapshot(&settings.data_dir, IMAGE_REPO, IMAGE_REV);
        assert!(root.starts_with(temporary.path()));
        let resolved = resolve_weights_dir(&request(&entry, json!({})), &settings).unwrap();
        assert_eq!(resolved, None, "nothing installed yet");
        for file in [
            "transformer_diffusers/config.json",
            "vae_diffusers/config.json",
            "tokenizer/tokenizer.json",
        ] {
            touch(root.join(file));
        }
        let te_row = co_requisite_rows(&entry, TEXT_ENCODER_COMPONENT)
            .find(|row| row["variant"] == "bf16")
            .unwrap()
            .clone();
        let te_snapshot = snapshot(
            &settings.data_dir,
            te_row["repo"].as_str().unwrap(),
            te_row["revision"].as_str().unwrap(),
        );
        touch(te_snapshot.join("text_encoder").join("config.json"));
        let resolved = resolve_weights_dir(&request(&entry, json!({})), &settings)
            .unwrap()
            .unwrap();
        assert_eq!(resolved, root);
        assert_eq!(tier_of_dir(&entry, &resolved), Some("bf16"));
        let spec = attach_components(
            LoadSpec::new(WeightsSource::Dir(resolved)),
            &entry,
            &settings,
        )
        .unwrap();
        assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == root));
        assert_eq!(spec.quantize, None);
        assert!(matches!(
            spec.components.get(TEXT_ENCODER_COMPONENT),
            Some(WeightsSource::Dir(dir)) if *dir == te_snapshot.join("text_encoder")
        ));
        assert!(!spec.components.contains_key(DIT_COMPONENT));
        let refused = resolve_weights_dir(&request(&entry, json!({"mlxQuantize": 4})), &settings);
        assert!(
            matches!(&refused, Err(WorkerError::InvalidPayload(message)) if message.contains("not published")),
            "{refused:?}"
        );
    }

    /// A published packed tier resolves `<re-host>/{tier}`, stages that tier's DiT and H3's
    /// packed text encoder of the SAME tier, and loads with the tier's quant.
    #[test]
    fn a_published_packed_tier_stages_its_dit_and_the_same_tier_text_encoder() {
        let temporary = tempfile::tempdir().unwrap();
        let (data, _guard) = isolated_cache(temporary.path());
        let settings = settings(&data);
        let mut entry = shipped_entry();
        let rehost_rev = "1111111111111111111111111111111111111111";
        for row in entry["downloads"].as_array_mut().unwrap() {
            if row.get("pendingArtifact").is_some() {
                row.as_object_mut().unwrap().remove("pendingArtifact");
                row["revision"] = json!(rehost_rev);
            }
        }
        let root = snapshot(&settings.data_dir, IMAGE_REPO, IMAGE_REV);
        assert!(root.starts_with(temporary.path()));
        touch(root.join("vae_diffusers/config.json"));
        touch(root.join("tokenizer/tokenizer.json"));
        let rehost = snapshot(
            &settings.data_dir,
            "SceneWorks/minimax-h3-image-mlx",
            rehost_rev,
        );
        touch(rehost.join("q8/transformer_diffusers/config.json"));
        let te_row = co_requisite_rows(&entry, TEXT_ENCODER_COMPONENT)
            .find(|row| row["variant"] == "q8")
            .unwrap()
            .clone();
        let te_snapshot = snapshot(
            &settings.data_dir,
            te_row["repo"].as_str().unwrap(),
            te_row["revision"].as_str().unwrap(),
        );
        touch(te_snapshot.join("q8/text_encoder/config.json"));
        let resolved = resolve_weights_dir(&request(&entry, json!({})), &settings)
            .unwrap()
            .unwrap();
        assert_eq!(resolved, rehost.join("q8"));
        assert_eq!(tier_of_dir(&entry, &resolved), Some("q8"));
        let spec = attach_components(
            LoadSpec::new(WeightsSource::Dir(resolved)),
            &entry,
            &settings,
        )
        .unwrap();
        assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == root));
        assert_eq!(spec.quantize, Some(Quant::Q8));
        assert!(matches!(
            spec.components.get(DIT_COMPONENT),
            Some(WeightsSource::Dir(dir)) if *dir == rehost.join("q8/transformer_diffusers")
        ));
        assert!(matches!(
            spec.components.get(TEXT_ENCODER_COMPONENT),
            Some(WeightsSource::Dir(dir)) if *dir == te_snapshot.join("q8/text_encoder")
        ));
        // q4 is declared but not on disk: refused by name.
        assert!(
            resolve_weights_dir(&request(&entry, json!({"mlxQuantize": 4})), &settings).is_err()
        );
    }
    /// SW-7103: the trainer's load spec is the render's for the same tier — the upstream root, the
    /// tier's staged DiT and same-tier text encoder, the tier's quant — chosen by the plan's
    /// `baseTier` (or the densest installed tier), and refused by name when nothing (or not the
    /// named tier) is installed, so a run never trains over a substituted base.
    #[test]
    fn the_training_load_spec_resolves_the_render_tier_and_refuses_an_uninstalled_one() {
        let temporary = tempfile::tempdir().unwrap();
        let (data, _guard) = isolated_cache(temporary.path());
        let mut entry = shipped_entry();
        let refused = training_load_spec_in(&entry, &data, None).unwrap_err();
        assert!(
            refused.to_string().contains("needs an installed tier"),
            "{refused}"
        );
        let rehost_rev = "2222222222222222222222222222222222222222";
        for row in entry["downloads"].as_array_mut().unwrap() {
            if row.get("pendingArtifact").is_some() {
                row.as_object_mut().unwrap().remove("pendingArtifact");
                row["revision"] = json!(rehost_rev);
            }
        }
        let root = snapshot(&data, IMAGE_REPO, IMAGE_REV);
        assert!(root.starts_with(temporary.path()));
        for file in [
            "transformer_diffusers/config.json",
            "vae_diffusers/config.json",
            "tokenizer/tokenizer.json",
        ] {
            touch(root.join(file));
        }
        let rehost = snapshot(&data, "SceneWorks/minimax-h3-image-mlx", rehost_rev);
        touch(rehost.join("q4/transformer_diffusers/config.json"));
        let te_dir = |tier: &str| {
            let row = co_requisite_rows(&entry, TEXT_ENCODER_COMPONENT)
                .find(|row| row["variant"] == tier)
                .unwrap()
                .clone();
            let snapshot = snapshot(
                &data,
                row["repo"].as_str().unwrap(),
                row["revision"].as_str().unwrap(),
            );
            snapshot.join(row["subdir"].as_str().unwrap())
        };
        // The bf16 DiT is on disk but its text encoder is not: refused, never another tier.
        let refused = training_load_spec_in(&entry, &data, None).unwrap_err();
        assert!(refused.to_string().contains("text encoder"), "{refused}");
        touch(te_dir("bf16").join("config.json"));
        let spec = training_load_spec_in(&entry, &data, None).unwrap();
        assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == root));
        assert_eq!(spec.quantize, None);
        assert!(!spec.components.contains_key(DIT_COMPONENT));
        // An explicit packed tier trains that tier frozen, with its own text encoder.
        touch(te_dir("q4").join("config.json"));
        let spec = training_load_spec_in(&entry, &data, Some("q4")).unwrap();
        assert!(matches!(&spec.weights, WeightsSource::Dir(dir) if *dir == root));
        assert_eq!(spec.quantize, Some(Quant::Q4));
        assert!(matches!(
            spec.components.get(DIT_COMPONENT),
            Some(WeightsSource::Dir(dir)) if *dir == rehost.join("q4/transformer_diffusers")
        ));
        assert!(matches!(
            spec.components.get(TEXT_ENCODER_COMPONENT),
            Some(WeightsSource::Dir(dir)) if *dir == te_dir("q4")
        ));
        // q8 is declared but not installed: refused by name.
        let refused = training_load_spec_in(&entry, &data, Some("q8")).unwrap_err();
        assert!(refused.to_string().contains("q8"), "{refused}");
    }

    /// SW-7102: a MiniMax-H3-Image job stages a resident prompt expander out of the refine cache
    /// before it resolves (let alone loads or renders) its own weights. The job here has no weights
    /// and an unroutable API, so it fails at route preparation — after the expander is already gone.
    #[cfg(native_lane)]
    #[tokio::test]
    async fn an_image_job_stages_the_resident_expander_out_before_generation() {
        use crate::refine_model_cache::tests::{
            global_refiner_is_resident, seed_global_resident_stub,
        };
        let temporary = tempfile::tempdir().unwrap();
        let (data, _guard) = isolated_cache(temporary.path());
        let mut settings = crate::test_env::offline_settings();
        settings.data_dir = data.clone();
        let project = crate::ProjectStore::new(data, "worker")
            .create_project("sw-7102-expander-stage-out")
            .unwrap();
        let job: crate::JobSnapshot = serde_json::from_value(json!({
            "id": "job_h3_image_stage_out", "type": "image_generate", "status": "queued",
            "payload": {
                "projectId": project.id, "model": MODEL_ID, "modelManifestEntry": shipped_entry(),
                "prompt": "a lighthouse", "width": 2048, "height": 2048, "count": 1,
            },
            "result": {}, "requestedGpu": "auto", "progress": 0, "stage": "queued", "message": "",
            "attempts": 1, "cancelRequested": false,
            "createdAt": "2026-11-12T00:00:00Z", "updatedAt": "2026-11-12T00:00:00Z"
        }))
        .unwrap();

        seed_global_resident_stub().await;
        assert!(
            global_refiner_is_resident().await,
            "the expander is resident"
        );
        let api = crate::ApiClient::new(&settings);
        let result = crate::image_jobs::run_image_generate_job(&api, &settings, &job).await;
        assert!(result.is_err(), "no weights: the job cannot render");
        assert!(
            !global_refiner_is_resident().await,
            "the image job evicted the expander before generation ({result:?})"
        );
    }
}
