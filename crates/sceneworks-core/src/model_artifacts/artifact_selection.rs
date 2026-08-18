//! Data-driven selection of a job's exact artifact requirement closure.
//!
//! Manifests declaratively describe every artifact a model can load (primary weights per tier,
//! hard co-requisites, per-platform rows). This module reduces one manifest entry plus the
//! request's own selection data (worker platform, requested tier) to the exact
//! [`ExternalArtifactRequirement`] closure that one runtime load will open. It is shared by the
//! API's submission preflight/catalog listing and the worker's pre-loader guard, so both judge
//! the same closure through one code path — never a per-route or per-model reimplementation.
//!
//! Completeness rules that reviews pinned down and this module owns:
//! - the closure is computed for the **selected** platform, never the API host's platform;
//! - the closure covers the **selected** variant/tier only — sibling-variant receipts are never
//!   unioned into (or out of) the answer;
//! - receipt-recorded exact files take precedence, with declared exact (non-glob) manifest files
//!   as the fallback so never-downloaded installs still carry a checkable identity.

use super::external_library::ExternalArtifactRequirement;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

/// The `<safe>` in the API/worker's managed `models/<safe>` download directory. Byte-identical to
/// the API (`apps/rust-api/src/lib.rs::safe_download_dir`) and the worker
/// (`sceneworks_worker::paths::safe_download_dir`); receipts written by either are read here.
pub fn safe_download_dir(value: &str) -> String {
    let mut output = String::new();
    let mut in_replacement = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-') {
            output.push(character);
            in_replacement = false;
        } else if !in_replacement {
            output.push_str("__");
            in_replacement = true;
        }
    }
    let output = output.trim_matches('_').to_owned();
    if output.is_empty() {
        "download".to_owned()
    } else {
        output
    }
}

/// True when a download entry names an artifact this contract can resolve (a Hugging Face repo).
pub fn is_supported_model_download(download: &Value) -> bool {
    download.get("provider").and_then(Value::as_str) == Some("huggingface")
        && download
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| !repo.is_empty())
}

/// True when a download entry is a co-requisite dependency (sc-9696): fetched ALONGSIDE the
/// primary download rather than as a pick-one alternate.
pub fn is_co_requisite_download(download: &Value) -> bool {
    download.get("coRequisite").and_then(Value::as_bool) == Some(true)
}

/// Whether a co-requisite row is scoped to ONE quant tier (sc-14980).
pub fn co_requisite_variant(download: &Value) -> Option<String> {
    download
        .get("variant")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
}

/// Drop download rows whose `platforms` list excludes `os`. Rows without a `platforms` key apply
/// everywhere. A manifest with no platform-scoped row at all is left untouched.
pub fn retain_downloads_for_os(model: &mut Value, os: &str) {
    let Some(downloads) = model.get_mut("downloads").and_then(Value::as_array_mut) else {
        return;
    };
    if !downloads
        .iter()
        .any(|entry| entry.get("platforms").is_some())
    {
        return;
    }
    downloads.retain(
        |entry| match entry.get("platforms").and_then(Value::as_array) {
            Some(platforms) => platforms.iter().any(|p| p.as_str() == Some(os)),
            None => true,
        },
    );
}

/// The model's canonical (default-or-first) primary download entry.
pub fn model_download(model: &Value) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let mut fallback = None;
    for download in downloads {
        if !is_supported_model_download(download) || is_co_requisite_download(download) {
            continue;
        }
        fallback.get_or_insert(download);
        if download.get("default").and_then(Value::as_bool) == Some(true) {
            return Some(download.clone());
        }
    }
    fallback.cloned()
}

/// The primary download entry whose `variant` matches `variant` (case-insensitive).
pub fn model_download_for_variant(model: &Value, variant: &str) -> Option<Value> {
    let downloads = model.get("downloads")?.as_array()?;
    let wanted = variant.trim().to_ascii_lowercase();
    downloads
        .iter()
        .find(|download| {
            is_supported_model_download(download)
                && !is_co_requisite_download(download)
                && download
                    .get("variant")
                    .and_then(Value::as_str)
                    .map(|value| value.trim().to_ascii_lowercase())
                    .as_deref()
                    == Some(wanted.as_str())
        })
        .cloned()
}

/// Every provider-supported co-requisite download row of `model`.
pub fn model_co_requisite_downloads(model: &Value) -> Vec<Value> {
    model
        .get("downloads")
        .and_then(Value::as_array)
        .map(|downloads| {
            downloads
                .iter()
                .filter(|download| {
                    is_co_requisite_download(download) && is_supported_model_download(download)
                })
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

/// The co-requisite downloads that apply to `variant` (sc-14980): tier-agnostic rows always
/// apply; a tier-scoped row applies only to its own tier.
pub fn model_co_requisite_downloads_for_variant(
    model: &Value,
    variant: Option<&str>,
) -> Vec<Value> {
    let wanted = variant.map(|value| value.trim().to_ascii_lowercase());
    model_co_requisite_downloads(model)
        .into_iter()
        .filter(
            |download| match (co_requisite_variant(download), wanted.as_deref()) {
                (None, _) => true,
                (Some(_), None) => true,
                (Some(row), Some(wanted)) => row == wanted,
            },
        )
        .collect()
}

fn string_array_field(payload: &Value, field: &str) -> Vec<String> {
    payload
        .get(field)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn pattern_matches(pattern: &str, value: &str) -> bool {
    let (pattern, value) = if cfg!(windows) {
        (pattern.to_ascii_lowercase(), value.to_ascii_lowercase())
    } else {
        (pattern.to_owned(), value.to_owned())
    };
    glob::Pattern::new(&pattern).is_ok_and(|pattern| pattern.matches(&value))
}

fn allow_pattern_matches(path: &str, patterns: &[String]) -> bool {
    if patterns.is_empty() {
        return true;
    }
    patterns
        .iter()
        .any(|pattern| pattern_matches(pattern, path))
}

/// The tier/variant the request actually selected, from the request's own data: an explicit
/// `variant`, an advanced `quantTier`, or the advanced `mlxQuantize` bit width. This is
/// request-shape vocabulary, never a model-name mapping.
pub fn requested_runtime_variant(payload: &Map<String, Value>) -> Option<String> {
    if let Some(variant) = payload
        .get("variant")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(variant.to_ascii_lowercase());
    }
    let advanced = payload.get("advanced").and_then(Value::as_object)?;
    if let Some(tier) = advanced
        .get("quantTier")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(tier.to_ascii_lowercase());
    }
    let bits = advanced.get("mlxQuantize").and_then(|value| {
        value
            .as_i64()
            .or_else(|| value.as_str()?.trim().parse::<i64>().ok())
    })?;
    Some(
        match bits {
            1..=4 => "q4",
            5..=8 => "q8",
            _ => "bf16",
        }
        .to_owned(),
    )
}

/// Reduce a manifest entry to the exact primary tier and hard co-requisites a worker on
/// `platform` will load. The public catalog may retain every installable tier, but source
/// availability is a runtime decision and must never union receipts from sibling variants or
/// filter co-requisites by the API host OS.
pub fn selected_model_artifact_closure(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
) -> Value {
    let mut selected = model.clone();
    retain_downloads_for_os(&mut selected, platform);
    let primary = requested_variant
        .and_then(|variant| model_download_for_variant(&selected, variant))
        .or_else(|| model_download(&selected));
    let selected_variant = primary
        .as_ref()
        .and_then(|download| download.get("variant"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let mut downloads = primary.into_iter().collect::<Vec<_>>();
    downloads.extend(
        model_co_requisite_downloads_for_variant(&selected, selected_variant.as_deref())
            .into_iter()
            .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft")),
    );
    if let Some(object) = selected.as_object_mut() {
        object.insert("downloads".to_owned(), Value::Array(downloads));
    }
    selected
}

struct ReceiptFileSet {
    files: Vec<String>,
    revision: Option<String>,
    variant: Option<String>,
}

fn receipt_entries(managed_path: &Path) -> Vec<Value> {
    let Ok(bytes) = std::fs::read(managed_path.join(".sceneworks-download-complete.json")) else {
        return Vec::new();
    };
    let Ok(receipt) = serde_json::from_slice::<Value>(&bytes) else {
        return Vec::new();
    };
    receipt
        .get("receipts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| vec![receipt])
}

fn receipt_file_sets(
    managed_path: &Path,
    repo: &str,
    model_id: Option<&str>,
) -> Vec<ReceiptFileSet> {
    receipt_entries(managed_path)
        .into_iter()
        .filter_map(|entry| {
            if entry.get("repo").and_then(Value::as_str) != Some(repo) {
                return None;
            }
            // Shared repos back multiple catalog cards. A model-specific receipt protects only
            // the card that produced it; receipts predating modelId remain generic.
            if let (Some(expected), Some(actual)) =
                (model_id, entry.get("modelId").and_then(Value::as_str))
            {
                if actual != expected {
                    return None;
                }
            }
            let files = entry
                .get("resolvedFiles")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>();
            let revision = entry
                .get("snapshotRevision")
                .and_then(Value::as_str)
                .map(str::to_owned);
            let variant = entry
                .get("variant")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned);
            (!files.is_empty()).then_some(ReceiptFileSet {
                files,
                revision,
                variant,
            })
        })
        .collect()
}

/// Requirements recovered from durable install receipts for the (already tier/platform-selected)
/// closure `model`. Receipts record the exact resolved file set of each completed install, so
/// this is the strongest install identity available: it survives a disconnected library without
/// re-reading a single artifact byte.
pub fn receipt_requirements_for_model(
    model: &Value,
    data_dir: &Path,
) -> Vec<ExternalArtifactRequirement> {
    let model_id = model.get("id").and_then(Value::as_str);
    let mut requirements = Vec::new();
    let downloads = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| is_supported_model_download(download));
    for download in downloads {
        if is_co_requisite_download(download)
            && download.get("required").and_then(Value::as_str) == Some("soft")
        {
            continue;
        }
        let Some(repo) = download.get("repo").and_then(Value::as_str) else {
            continue;
        };
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        let receipt_model_id = (!is_co_requisite_download(download))
            .then_some(model_id)
            .flatten();
        let variant = download
            .get("variant")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("default")
            .to_owned();
        let declared_files = string_array_field(download, "files");
        for receipt in receipt_file_sets(&managed, repo, receipt_model_id) {
            // A receipt belongs to this row only when its recorded variant matches, or — for
            // legacy variant-less receipts — when its files fall inside this row's declared
            // patterns. Sibling-tier receipts never leak into the selected closure.
            if receipt
                .variant
                .as_deref()
                .is_some_and(|receipt_variant| receipt_variant != variant)
                || (receipt.variant.is_none()
                    && !declared_files.is_empty()
                    && !receipt
                        .files
                        .iter()
                        .any(|file| allow_pattern_matches(file, &declared_files)))
            {
                continue;
            }
            let requirement = ExternalArtifactRequirement {
                repository: repo.to_owned(),
                revision: receipt.revision,
                variant: variant.clone(),
                files: receipt.files.into_iter().map(PathBuf::from).collect(),
                is_primary: !is_co_requisite_download(download),
            };
            if !requirements.contains(&requirement) {
                requirements.push(requirement);
            }
        }
    }
    requirements.sort_by(|left, right| {
        (&left.repository, &left.revision, &left.variant, &left.files).cmp(&(
            &right.repository,
            &right.revision,
            &right.variant,
            &right.files,
        ))
    });
    requirements
}

/// Requirements the manifest itself declares exactly: an immutable revision and a concrete
/// (non-glob) file list. This is the fallback identity for installs that predate receipts.
pub fn declared_exact_requirements_for_model(model: &Value) -> Vec<ExternalArtifactRequirement> {
    let mut requirements = Vec::new();
    for download in model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|download| is_supported_model_download(download))
        .filter(|download| download.get("required").and_then(Value::as_str) != Some("soft"))
    {
        let Some(repository) = download.get("repo").and_then(Value::as_str) else {
            continue;
        };
        let Some(revision) = download
            .get("revision")
            .and_then(Value::as_str)
            .filter(|revision| {
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
        else {
            continue;
        };
        let files = string_array_field(download, "files");
        if files.is_empty()
            || files.iter().any(|file| {
                file.bytes()
                    .any(|byte| matches!(byte, b'*' | b'?' | b'[' | b']'))
            })
        {
            continue;
        }
        requirements.push(ExternalArtifactRequirement {
            repository: repository.to_owned(),
            revision: Some(revision.to_owned()),
            variant: download
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .to_owned(),
            files: files.into_iter().map(PathBuf::from).collect(),
            is_primary: !is_co_requisite_download(download),
        });
    }
    requirements
}

/// One selected requirement closure plus the strength of its install evidence. Only
/// receipt-backed closures prove a completed installation; a declared-exact closure carries a
/// checkable identity but proves nothing about install state, so it must never produce the
/// typed "installed — external library unavailable" condition on its own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedRequirements {
    pub requirements: Vec<ExternalArtifactRequirement>,
    /// True when at least one requirement came from a durable download receipt.
    pub receipt_backed: bool,
}

/// The exact selected requirement closure for one (already tier/platform-selected) manifest
/// closure: receipt-backed requirements first, declared-exact fallbacks for rows without a
/// receipt, canonically ordered with the primary first. Returns an empty closure when the model
/// carries no checkable install identity (no receipt and no exact declaration) or when the
/// closure would not contain exactly one primary.
pub fn selected_requirements_for_closure(
    selected: &Value,
    data_dir: &Path,
) -> SelectedRequirements {
    let mut requirements = receipt_requirements_for_model(selected, data_dir);
    let receipt_backed = !requirements.is_empty();
    for declared in declared_exact_requirements_for_model(selected) {
        if !requirements.iter().any(|requirement| {
            requirement.repository == declared.repository
                && requirement.variant == declared.variant
                && requirement.is_primary == declared.is_primary
        }) {
            requirements.push(declared);
        }
    }
    if requirements
        .iter()
        .filter(|requirement| requirement.is_primary)
        .count()
        != 1
    {
        return SelectedRequirements {
            requirements: Vec::new(),
            receipt_backed: false,
        };
    }
    requirements.sort_by(|left, right| {
        (
            !left.is_primary,
            &left.repository,
            &left.variant,
            &left.files,
        )
            .cmp(&(
                !right.is_primary,
                &right.repository,
                &right.variant,
                &right.files,
            ))
    });
    SelectedRequirements {
        requirements,
        receipt_backed,
    }
}

/// The schema version of [`LocalCacheEligibility`], so a client can branch on shape rather than
/// on the presence of individual keys.
pub const LOCAL_CACHE_ELIGIBILITY_SCHEMA_VERSION: u32 = 1;

/// How much of a model the resolved model cache can ever serve from a local copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalCacheCoverage {
    /// Everything the selected closure needs can be held locally.
    Full,
    /// The primary can be held locally, but part of the declared closure never enters it, so a
    /// request that needs the excluded part still reads from the source library.
    Partial,
    /// Nothing can be served locally, however much of it is copied.
    None,
}

/// Why a model's local copy cannot cover it. Typed so the UI branches on the reason instead of
/// parsing prose, which is the same discipline `ModelAvailability` established.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalCacheExclusion {
    /// The model declares `required: "soft"` co-requisites. Closure selection drops them
    /// (`selected_model_artifact_closure`, `receipt_requirements_for_model`,
    /// `declared_exact_requirements_for_model`), so they are never promoted and never served
    /// locally, while the primary is — which is exactly how a "local copy" badge over-promises.
    OptionalComponentsExcluded,
    /// A requirement carries no recorded snapshot revision, which makes its WHOLE repository
    /// unserveable from the local tier: with no revision there is no pair to compare coverage
    /// against. Promotion can still build the bundle, so such a model can occupy cache bytes it
    /// will never be served from.
    UnpinnedRevision,
}

/// What a model's local copy can and cannot cover, and why (sc-19712 F-5).
///
/// The epic's acceptance criteria require unsupported artifact classes to be identified in the
/// product rather than silently bypassed. Before this, a model with soft co-requisites or a
/// revision-less requirement showed the same "local copy" affordance as a fully cacheable one and
/// the exclusion was visible only as a `resolved_cache_local_tier_not_selected` log line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalCacheEligibility {
    pub schema_version: u32,
    pub coverage: LocalCacheCoverage,
    /// `None` exactly when `coverage` is [`LocalCacheCoverage::Full`].
    pub reason: Option<LocalCacheExclusion>,
    /// A short sentence naming what is excluded. Supporting copy, never the thing to branch on.
    pub detail: Option<String>,
}

impl LocalCacheEligibility {
    fn full() -> Self {
        Self {
            schema_version: LOCAL_CACHE_ELIGIBILITY_SCHEMA_VERSION,
            coverage: LocalCacheCoverage::Full,
            reason: None,
            detail: None,
        }
    }

    fn excluded(coverage: LocalCacheCoverage, reason: LocalCacheExclusion, detail: String) -> Self {
        Self {
            schema_version: LOCAL_CACHE_ELIGIBILITY_SCHEMA_VERSION,
            coverage,
            reason: Some(reason),
            detail: Some(detail),
        }
    }
}

/// True when the model declares at least one optional (`required: "soft"`) co-requisite download.
/// These are filtered out of every requirement closure this module builds, so they can never be
/// promoted — the check is deliberately made against the RAW entry rather than a selected closure,
/// because by the time selection has run the evidence is already gone.
pub fn declares_optional_co_requisites(model: &Value) -> bool {
    model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|download| {
            is_co_requisite_download(download)
                && download.get("required").and_then(Value::as_str) == Some("soft")
        })
}

/// The one judgement of what a model's local copy can cover, shared by every surface that shows it.
///
/// Ordering matters: an unpinned revision beats an excluded optional component, because it is the
/// stronger statement — a model can be excluded on both counts (a real install of `qwen_image` is),
/// and reporting "some components are excluded" for something that can serve nothing at all would
/// still over-promise.
pub fn local_cache_eligibility_for_model(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    data_dir: &Path,
) -> LocalCacheEligibility {
    let selected = selected_requirements_for_model(model, platform, requested_variant, data_dir);
    if let Some(unpinned) = selected
        .requirements
        .iter()
        .find(|requirement| requirement.revision.is_none())
    {
        return LocalCacheEligibility::excluded(
            LocalCacheCoverage::None,
            LocalCacheExclusion::UnpinnedRevision,
            format!(
                "No snapshot revision was recorded for {}, so no local copy of this model can be \
                 used and it will always load from the model library.",
                unpinned.repository
            ),
        );
    }
    if declares_optional_co_requisites(model) {
        return LocalCacheEligibility::excluded(
            LocalCacheCoverage::Partial,
            LocalCacheExclusion::OptionalComponentsExcluded,
            "Optional components of this model are never copied locally, so a request that needs \
             one still reads from the model library."
                .to_owned(),
        );
    }
    LocalCacheEligibility::full()
}

/// One-call composition: select the exact closure for (`platform`, `requested_variant`), then
/// compute its requirement list. This is the function the API seam and the worker guard share.
pub fn selected_requirements_for_model(
    model: &Value,
    platform: &str,
    requested_variant: Option<&str>,
    data_dir: &Path,
) -> SelectedRequirements {
    let selected = selected_model_artifact_closure(model, platform, requested_variant);
    selected_requirements_for_closure(&selected, data_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const REV_A: &str = "0123456789abcdef0123456789abcdef01234567";
    const REV_B: &str = "89abcdef0123456789abcdef0123456789abcdef";

    fn write_receipts(data_dir: &Path, repo: &str, receipts: Value) {
        let managed = data_dir.join("models").join(safe_download_dir(repo));
        std::fs::create_dir_all(&managed).unwrap();
        std::fs::write(
            managed.join(".sceneworks-download-complete.json"),
            serde_json::to_vec(&json!({ "receipts": receipts })).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn receipt_requirements_preserve_exact_variant_file_closures() {
        let temp = tempfile::tempdir().unwrap();
        write_receipts(
            temp.path(),
            "owner/matrix",
            json!([
                {"repo":"owner/matrix", "modelId":"matrix", "variant":"q4",
                 "resolvedFiles":["q4/model.safetensors"], "snapshotRevision": REV_A},
                {"repo":"owner/matrix", "modelId":"matrix", "variant":"q8",
                 "resolvedFiles":["q8/model.safetensors"], "snapshotRevision": REV_B}
            ]),
        );
        let model = json!({
            "id": "matrix",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/matrix", "variant":"q4", "default":true, "files":["q4/*"]},
                {"provider":"huggingface", "repo":"owner/matrix", "variant":"q8", "files":["q8/*"]}
            ]
        });

        // The un-selected model still yields one requirement per installed variant.
        let all = receipt_requirements_for_model(&model, temp.path());
        assert_eq!(all.len(), 2);

        // Selecting q4 must judge ONLY the q4 closure: exactly one requirement, primary, and the
        // sibling q8 receipt must not appear (a model-wide union would mark q4 incomplete when
        // q8 was removed, or vice versa).
        let q4 =
            selected_requirements_for_model(&model, std::env::consts::OS, Some("q4"), temp.path());
        assert!(q4.receipt_backed);
        let q4 = q4.requirements;
        assert_eq!(q4.len(), 1);
        assert!(q4[0].is_primary);
        assert_eq!(q4[0].variant, "q4");
        assert_eq!(q4[0].files, [PathBuf::from("q4/model.safetensors")]);
        assert_eq!(q4[0].revision.as_deref(), Some(REV_A));

        // Removing the q8 receipt leaves the q4 selection untouched.
        write_receipts(
            temp.path(),
            "owner/matrix",
            json!([
                {"repo":"owner/matrix", "modelId":"matrix", "variant":"q4",
                 "resolvedFiles":["q4/model.safetensors"], "snapshotRevision": REV_A}
            ]),
        );
        let q4_after =
            selected_requirements_for_model(&model, std::env::consts::OS, Some("q4"), temp.path());
        assert_eq!(q4, q4_after.requirements);
        assert!(q4_after.receipt_backed);
    }

    #[test]
    fn selected_closure_uses_worker_platform_and_matching_tier_corequisites() {
        let model = json!({
            "id": "cross-platform",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/mac", "variant":"q4", "default":true,
                 "files":["q4/*"], "platforms":["macos"]},
                {"provider":"huggingface", "repo":"owner/windows", "variant":"q4", "default":true,
                 "files":["q4/*"], "platforms":["windows", "linux"]},
                {"provider":"huggingface", "repo":"owner/candle-component", "variant":"q4",
                 "coRequisite":true, "files":["encoder.safetensors"], "platforms":["windows", "linux"]},
                {"provider":"huggingface", "repo":"owner/wrong-tier", "variant":"q8",
                 "coRequisite":true, "files":["encoder.safetensors"], "platforms":["windows", "linux"]}
            ]
        });
        let repositories = |entry: &Value| {
            entry["downloads"]
                .as_array()
                .unwrap()
                .iter()
                .filter_map(|download| download["repo"].as_str().map(str::to_owned))
                .collect::<Vec<_>>()
        };
        let mac = selected_model_artifact_closure(&model, "macos", Some("q4"));
        assert_eq!(repositories(&mac), vec!["owner/mac".to_owned()]);
        // The same manifest evaluated for a Windows worker must include the Candle-only
        // co-requisite the macOS host does not load — API-host platform filtering is the defect
        // this function exists to prevent.
        let windows = selected_model_artifact_closure(&model, "windows", Some("q4"));
        assert_eq!(
            repositories(&windows),
            vec![
                "owner/windows".to_owned(),
                "owner/candle-component".to_owned()
            ]
        );
    }

    #[test]
    fn declared_exact_requirements_skip_globs_soft_rows_and_mutable_revisions() {
        let model = json!({
            "id": "declared",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/exact", "revision": REV_A,
                 "files":["model.safetensors"]},
                {"provider":"huggingface", "repo":"owner/glob", "revision": REV_A, "files":["q4/*"]},
                {"provider":"huggingface", "repo":"owner/mutable", "revision":"main",
                 "files":["model.safetensors"]},
                {"provider":"huggingface", "repo":"owner/soft", "revision": REV_A,
                 "coRequisite": true, "required":"soft", "files":["extra.safetensors"]}
            ]
        });
        let declared = declared_exact_requirements_for_model(&model);
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].repository, "owner/exact");
        assert_eq!(declared[0].revision.as_deref(), Some(REV_A));
    }

    #[test]
    fn closure_without_exactly_one_primary_is_empty_not_partial() {
        let temp = tempfile::tempdir().unwrap();
        // Only a co-requisite receipt exists: no primary identity means no closure at all —
        // a partial closure would let an unavailable primary read as ready.
        write_receipts(
            temp.path(),
            "owner/encoder",
            json!([
                {"repo":"owner/encoder", "resolvedFiles":["encoder.safetensors"],
                 "snapshotRevision": REV_A}
            ]),
        );
        let model = json!({
            "id": "primaryless",
            "downloads": [
                {"provider":"huggingface", "repo":"owner/primary", "files":["model.safetensors"]},
                {"provider":"huggingface", "repo":"owner/encoder", "coRequisite": true,
                 "files":["encoder.safetensors"]}
            ]
        });
        let selected =
            selected_requirements_for_model(&model, std::env::consts::OS, None, temp.path());
        assert!(selected.requirements.is_empty());
        assert!(!selected.receipt_backed);
    }

    #[test]
    fn requested_runtime_variant_reads_request_data_only() {
        let payload = |value: Value| value.as_object().unwrap().clone();
        assert_eq!(
            requested_runtime_variant(&payload(json!({"variant": "Q8"}))),
            Some("q8".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"quantTier": "bf16"}}))),
            Some("bf16".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"mlxQuantize": 4}}))),
            Some("q4".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"mlxQuantize": "8"}}))),
            Some("q8".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"advanced": {"mlxQuantize": 16}}))),
            Some("bf16".to_owned())
        );
        assert_eq!(
            requested_runtime_variant(&payload(json!({"model": "x"}))),
            None
        );
    }
}
