//! Read a pinned download entry straight out of the embedded builtin catalog (sc-13810).
//!
//! Test harnesses that provision a cataloged model need three things about it: the repo, the pinned
//! commit, and the allow-listed files. Hard-coding them in the test MIRRORS the manifest instead of
//! READING it, so a pin bump or an allow-list edit in `config/manifests/builtin.models.jsonc` leaves
//! the harness silently asserting the OLD contract — it still passes, against weights nobody ships.
//! That was the sc-13797 whisper smoke's gap (sc-13810 item 2).
//!
//! Reading from [`sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS`] — the same `include_str!`d
//! bytes the app seeds — makes the manifest the single source of truth: change the pin and every
//! harness follows it in the same build, or fails loudly if the entry stops existing.

// The macOS-only live install smoke and the cross-platform network-free layout test read different
// subsets of this helper, so not every target compiles a use of every accessor (same situation
// `test_env` documents). Without this, a Windows/Linux `-D warnings` lane fails on dead_code for a
// helper that IS exercised on macOS.
#![allow(dead_code)]

use sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS;
use serde_json::Value;

/// One resolved download entry from the builtin catalog: what to fetch, at which commit, filtered
/// to which files.
#[derive(Debug, Clone)]
pub(crate) struct ManifestPin {
    pub(crate) repo: String,
    /// The pinned 40-hex commit SHA (`revision`). Only entries that carry one are returned — an
    /// unpinned entry cannot anchor an offline/deterministic layout assertion.
    pub(crate) revision: String,
    pub(crate) files: Vec<String>,
}

impl ManifestPin {
    /// The allow-listed files as `&str`, for the `&[&str]` install helpers.
    pub(crate) fn file_refs(&self) -> Vec<&str> {
        self.files.iter().map(String::as_str).collect()
    }

    /// The hub cache repo directory name for this pin (`models--owner--name`).
    pub(crate) fn repo_slug(&self) -> String {
        format!("models--{}", self.repo.replace('/', "--"))
    }
}

/// The embedded `builtin.models.jsonc`, comments stripped, parsed.
fn builtin_models() -> Value {
    let raw = BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .expect("builtin.models.jsonc is embedded in BUILTIN_MANIFESTS");
    let stripped = sceneworks_core::jsonc::strip_jsonc_comments(raw);
    serde_json::from_str(&stripped).expect("builtin.models.jsonc parses as JSON")
}

/// The PRIMARY pinned download for the catalog model `model_id`.
///
/// "Primary" = the first non-`coRequisite` download entry carrying a `revision`. Co-requisites are
/// companion weights, not the model itself, so they are skipped here; a harness that wants one
/// should assert on it explicitly rather than pick it up by accident.
///
/// Panics with a pointed message when the model, or a pinned primary download, is absent — a test
/// that can no longer find its subject must fail loudly, never silently skip.
pub(crate) fn builtin_model_pin(model_id: &str) -> ManifestPin {
    let manifest = builtin_models();
    let model = manifest
        .get("models")
        .and_then(Value::as_array)
        .expect("builtin.models.jsonc has a models array")
        .iter()
        .find(|model| model.get("id").and_then(Value::as_str) == Some(model_id))
        .unwrap_or_else(|| {
            panic!("builtin.models.jsonc has no model with id {model_id:?} — did it get renamed?")
        });

    let download = model
        .get("downloads")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|entry| {
            entry.get("coRequisite").and_then(Value::as_bool) != Some(true)
                && entry.get("revision").and_then(Value::as_str).is_some()
        })
        .unwrap_or_else(|| {
            panic!("model {model_id:?} has no PINNED (revision-carrying) primary download entry")
        });

    let repo = download
        .get("repo")
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("model {model_id:?}'s pinned download has no repo"))
        .to_owned();
    let revision = download
        .get("revision")
        .and_then(Value::as_str)
        .expect("the entry was selected for carrying a revision")
        .to_owned();
    let files = download
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert!(
        !files.is_empty(),
        "model {model_id:?}'s pinned download has an empty files allow-list — a harness filtered by \
         it would fetch the WHOLE repo"
    );

    ManifestPin {
        repo,
        revision,
        files,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_base_pin_is_read_from_the_manifest_not_mirrored() {
        // The whisper_base entry is what the sc-13797 install smoke provisions. Reading it here is
        // the contract: a pin bump or allow-list edit in builtin.models.jsonc reaches the harness in
        // the same build. These assertions are on the SHAPE the harness depends on, not the literal
        // values — pinning the values here would just re-introduce the mirror one layer down.
        let pin = builtin_model_pin("whisper_base");
        assert_eq!(pin.repo, "openai/whisper-base");
        assert_eq!(
            pin.revision.len(),
            40,
            "revision must be a full 40-hex commit SHA, got {:?}",
            pin.revision
        );
        assert!(
            pin.revision
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "revision must be lowercase hex, got {:?}",
            pin.revision
        );
        assert!(
            pin.files.iter().any(|file| file.ends_with(".safetensors")),
            "the allow-list must include weights, got {:?}",
            pin.files
        );
        assert_eq!(pin.repo_slug(), "models--openai--whisper-base");
    }

    #[test]
    fn a_missing_model_id_panics_rather_than_skipping() {
        // A harness whose subject vanished must go RED, not quietly pass.
        let error = std::panic::catch_unwind(|| builtin_model_pin("definitely_not_a_model"))
            .expect_err("a missing model id must panic");
        let message = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(
            message.contains("definitely_not_a_model"),
            "the panic must name the missing id, got {message:?}"
        );
    }
}
