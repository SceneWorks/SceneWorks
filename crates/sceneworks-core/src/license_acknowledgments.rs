//! Server-side licence acknowledgments (sc-22999, epic sc-22988 E2).
//!
//! Before this module, a `requiresLicenseAcknowledgment` model was gated only by the caller's
//! per-request `licenseAcknowledged` assertion (and the web client's origin-keyed `localStorage`,
//! which the desktop shell loses on every relaunch). Nothing the server held could be re-checked
//! later, so a job queued under an acknowledgment could not be refused at execution after the
//! acknowledgment was withdrawn.
//!
//! This is that record: one entry per catalog model id, bound to the SHA-256 of the licence terms
//! the user accepted ([`terms_sha256`]). An acknowledgment is **current** only while those terms are
//! unchanged ([`current_acknowledgment`]) — a catalog update that changes a model's licence notice or
//! URL makes every earlier acknowledgment stale without anyone deleting it.
//!
//! The API is the only writer (`PUT`/`DELETE /api/v1/models/:id/license-acknowledgment`, and the
//! typed download / YuE2 job routes when the caller asserts `licenseAcknowledged: true`). The file is
//! replaced atomically (write to a sibling, then rename), so a reader never sees a torn record.

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// File name inside the config dir.
pub const FILE_NAME: &str = "license-acknowledgments.json";
/// Record schema.
pub const SCHEMA_VERSION: u32 = 1;

/// Where the acknowledgments live.
pub fn acknowledgments_file(config_dir: &Path) -> PathBuf {
    config_dir.join(FILE_NAME)
}

/// One accepted licence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LicenseAcknowledgment {
    pub model_id: String,
    /// [`terms_sha256`] of the entry at the moment of acceptance.
    pub terms_sha256: String,
    pub acknowledged_at: String,
    /// Which door recorded it (`api`, `download`, `yue2_job`, …).
    pub channel: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AcknowledgmentFile {
    #[serde(default)]
    schema_version: u32,
    #[serde(default)]
    acknowledgments: BTreeMap<String, LicenseAcknowledgment>,
}

/// Errors reading or writing the record. A malformed file is an error, never an empty record: an
/// unreadable acknowledgment store must refuse a gated model, not silently grant or forget.
#[derive(Debug)]
pub enum AcknowledgmentError {
    Io(std::io::Error),
    Malformed(String),
}

impl std::fmt::Display for AcknowledgmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "licence acknowledgment store: {error}"),
            Self::Malformed(detail) => {
                write!(f, "licence acknowledgment store is malformed: {detail}")
            }
        }
    }
}

impl std::error::Error for AcknowledgmentError {}

/// SHA-256 of the licence terms an entry declares: its `license`, `licenseUrl` and `licenseNotice`
/// in a fixed order. Anything else in the entry can change without invalidating an acknowledgment.
pub fn terms_sha256(entry: &Value) -> String {
    let terms = json!({
        "license": entry.get("license").cloned().unwrap_or(Value::Null),
        "licenseUrl": entry.get("licenseUrl").cloned().unwrap_or(Value::Null),
        "licenseNotice": entry.get("licenseNotice").cloned().unwrap_or(Value::Null),
    });
    let canonical = serde_json::to_string(&terms).expect("a JSON value serializes");
    hex(&Sha256::digest(canonical.as_bytes()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_file(path: &Path) -> Result<AcknowledgmentFile, AcknowledgmentError> {
    match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|error| AcknowledgmentError::Malformed(error.to_string())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(AcknowledgmentFile::default())
        }
        Err(error) => Err(AcknowledgmentError::Io(error)),
    }
}

fn write_file(path: &Path, file: &AcknowledgmentFile) -> Result<(), AcknowledgmentError> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(AcknowledgmentError::Io)?;
    }
    let mut bytes = serde_json::to_vec_pretty(file).expect("the record serializes");
    bytes.push(b'\n');
    let temp = path.with_extension(format!("json.tmp-{}", std::process::id()));
    {
        let mut handle = std::fs::File::create(&temp).map_err(AcknowledgmentError::Io)?;
        handle.write_all(&bytes).map_err(AcknowledgmentError::Io)?;
        handle.sync_all().map_err(AcknowledgmentError::Io)?;
    }
    std::fs::rename(&temp, path).map_err(AcknowledgmentError::Io)
}

/// The stored acknowledgment of `model_id`, whatever terms it was given for.
pub fn stored_acknowledgment(
    path: &Path,
    model_id: &str,
) -> Result<Option<LicenseAcknowledgment>, AcknowledgmentError> {
    Ok(read_file(path)?.acknowledgments.remove(model_id))
}

/// The acknowledgment of `entry` that is **current**: stored for its id and given for the terms the
/// entry declares now. `None` when there is none, or it was given for other terms.
pub fn current_acknowledgment(
    path: &Path,
    entry: &Value,
) -> Result<Option<LicenseAcknowledgment>, AcknowledgmentError> {
    let Some(model_id) = entry.get("id").and_then(Value::as_str) else {
        return Ok(None);
    };
    let terms = terms_sha256(entry);
    Ok(stored_acknowledgment(path, model_id)?.filter(|ack| ack.terms_sha256 == terms))
}

/// Record that the user accepted `entry`'s current terms (replacing any earlier record).
pub fn record_acknowledgment(
    path: &Path,
    entry: &Value,
    channel: &str,
    acknowledged_at: &str,
) -> Result<LicenseAcknowledgment, AcknowledgmentError> {
    let model_id = entry
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .ok_or_else(|| AcknowledgmentError::Malformed("the catalog entry has no id".into()))?;
    let mut file = read_file(path)?;
    let ack = LicenseAcknowledgment {
        model_id: model_id.to_owned(),
        terms_sha256: terms_sha256(entry),
        acknowledged_at: acknowledged_at.to_owned(),
        channel: channel.to_owned(),
    };
    file.schema_version = SCHEMA_VERSION;
    file.acknowledgments
        .insert(model_id.to_owned(), ack.clone());
    write_file(path, &file)?;
    Ok(ack)
}

/// Withdraw `model_id`'s acknowledgment. Returns whether one was stored.
pub fn revoke_acknowledgment(path: &Path, model_id: &str) -> Result<bool, AcknowledgmentError> {
    let mut file = read_file(path)?;
    let removed = file.acknowledgments.remove(model_id).is_some();
    if removed {
        file.schema_version = SCHEMA_VERSION;
        write_file(path, &file)?;
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(notice: &str) -> Value {
        json!({"id": "yue2", "licenseUrl": "https://example.test/LICENSE", "licenseNotice": notice})
    }

    #[test]
    fn an_acknowledgment_is_current_only_for_the_terms_it_was_given_for() {
        let dir = tempfile::tempdir().unwrap();
        let path = acknowledgments_file(dir.path());
        assert_eq!(current_acknowledgment(&path, &entry("v1")).unwrap(), None);
        let ack =
            record_acknowledgment(&path, &entry("v1"), "api", "2026-09-26T00:00:00Z").unwrap();
        assert_eq!(
            current_acknowledgment(&path, &entry("v1")).unwrap(),
            Some(ack.clone())
        );
        // Mutation that reds this: dropping the terms filter in `current_acknowledgment` keeps a
        // v1 acknowledgment current after the catalog changes the notice.
        assert_eq!(current_acknowledgment(&path, &entry("v2")).unwrap(), None);
        assert_eq!(
            stored_acknowledgment(&path, "yue2").unwrap(),
            Some(ack),
            "the stale record is kept, only not current"
        );
    }

    #[test]
    fn revoking_withdraws_it_and_a_malformed_store_is_an_error_not_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = acknowledgments_file(dir.path());
        record_acknowledgment(&path, &entry("v1"), "api", "2026-09-26T00:00:00Z").unwrap();
        assert!(revoke_acknowledgment(&path, "yue2").unwrap());
        assert!(!revoke_acknowledgment(&path, "yue2").unwrap());
        assert_eq!(current_acknowledgment(&path, &entry("v1")).unwrap(), None);
        std::fs::write(&path, b"{not json").unwrap();
        assert!(matches!(
            current_acknowledgment(&path, &entry("v1")),
            Err(AcknowledgmentError::Malformed(_))
        ));
    }
}
