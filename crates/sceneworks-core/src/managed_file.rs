//! Catalog-owned exact identity for managed single-file components. No network or ambient path.
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path, sync::OnceLock};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManagedFileIdentity {
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub bytes: u64,
    pub sha256: String,
}

impl ManagedFileIdentity {
    pub fn from_download(row: &Value) -> Result<Option<Self>, String> {
        if row.get("expectedSha256").is_none() && row.get("expectedSizeBytes").is_none() {
            return Ok(None);
        }
        let text = |key: &str| {
            row[key]
                .as_str()
                .ok_or_else(|| format!("managed file requires {key}"))
        };
        let repo = text("repo")?;
        let revision = text("revision")?;
        let sha256 = text("expectedSha256")?;
        let hex = |s: &str, n| {
            s.len() == n
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        let files = row["files"]
            .as_array()
            .ok_or("managed file requires files")?;
        let file = files
            .first()
            .and_then(Value::as_str)
            .ok_or("managed file requires one file")?;
        if !hex(revision, 40)
            || !hex(sha256, 64)
            || files.len() != 1
            || file.is_empty()
            || file.contains(['*', '?', '\\'])
            || !Path::new(file)
                .components()
                .all(|p| matches!(p, std::path::Component::Normal(_)))
        {
            return Err(
                "managed file requires an immutable revision, one confined exact file and a SHA256"
                    .into(),
            );
        }
        let bytes = row["expectedSizeBytes"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or("managed file requires positive expectedSizeBytes")?;
        Ok(Some(Self {
            repo: repo.into(),
            revision: revision.into(),
            file: file.into(),
            bytes,
            sha256: sha256.into(),
        }))
    }

    pub fn stamp(&self, payload: &mut Map<String, Value>) {
        payload.insert("expectedSha256".into(), Value::String(self.sha256.clone()));
        payload.insert("expectedSizeBytes".into(), self.bytes.into());
    }

    /// Re-read bytes at each load; same filename, size, or stale install receipt is insufficient.
    pub fn verify(&self, path: &Path) -> Result<(), String> {
        let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        if file.metadata().map_err(|e| e.to_string())?.len() != self.bytes {
            return Err("managed component has the wrong byte count; repair/install it".into());
        }
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        if format!("{:x}", hasher.finalize()) != self.sha256 {
            return Err("managed component SHA256 mismatch; repair/install it".into());
        }
        Ok(())
    }

    pub fn reconcile(&self, payload: &Map<String, Value>) -> Result<(), String> {
        if payload.get("repo").and_then(Value::as_str) != Some(&self.repo)
            || payload.get("revision").and_then(Value::as_str) != Some(&self.revision)
            || payload.get("files") != Some(&serde_json::json!([self.file]))
        {
            return Err(
                "managed component repo/revision/files conflict with the authoritative catalog"
                    .into(),
            );
        }
        for (key, expected) in [
            ("expectedSha256", Value::String(self.sha256.clone())),
            ("expectedSizeBytes", self.bytes.into()),
        ] {
            if payload.get(key).is_some_and(|actual| actual != &expected) {
                return Err(format!("{key} conflicts with the authoritative catalog"));
            }
        }
        Ok(())
    }
}

static BUILTIN_IDENTITIES: OnceLock<Result<Vec<ManagedFileIdentity>, String>> = OnceLock::new();
pub fn builtin_identities() -> Result<&'static [ManagedFileIdentity], String> {
    BUILTIN_IDENTITIES
        .get_or_init(|| {
            let raw = crate::builtin_manifests::BUILTIN_MANIFESTS
                .iter()
                .find(|(name, _)| *name == "builtin.models.jsonc")
                .ok_or("missing builtin catalog")?
                .1;
            let catalog: Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(raw))
                .map_err(|e| e.to_string())?;
            let mut identities = Vec::new();
            for model in catalog["models"].as_array().ok_or("missing models")? {
                for row in model["downloads"].as_array().ok_or("missing downloads")? {
                    if let Some(identity) = ManagedFileIdentity::from_download(row)? {
                        if let Some(previous) = identities
                            .iter()
                            .find(|p: &&ManagedFileIdentity| p.repo == identity.repo)
                        {
                            if previous != &identity {
                                return Err(
                                    "managed file identities disagree between catalog entries"
                                        .into(),
                                );
                            }
                        } else {
                            identities.push(identity);
                        }
                    }
                }
            }
            Ok(identities)
        })
        .as_deref()
        .map_err(Clone::clone)
}

/// Raw/retried jobs cannot replace or omit the immutable selection. Missing digest metadata is
/// recovered from the catalog, never from the client or the HF response.
pub fn authoritative_job_identity(
    payload: &Map<String, Value>,
) -> Result<Option<&'static ManagedFileIdentity>, String> {
    let identity = builtin_identities()?.iter().find(|p| {
        payload
            .get("repo")
            .and_then(Value::as_str)
            .is_some_and(|repo| repo.trim().eq_ignore_ascii_case(&p.repo))
    });
    if let Some(identity) = identity {
        identity.reconcile(payload)?;
    }
    Ok(identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn row(body: &[u8]) -> Value {
        json!({"repo":"test/predictor", "revision":"a".repeat(40), "files":["predictor.safetensors"], "expectedSizeBytes":body.len(), "expectedSha256":format!("{:x}",Sha256::digest(body))})
    }
    #[test]
    fn managed_file_rejects_same_size_corruption_and_invalid_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("predictor.safetensors");
        let pin = ManagedFileIdentity::from_download(&row(b"good"))
            .unwrap()
            .unwrap();
        std::fs::write(&path, b"good").unwrap();
        pin.verify(&path).unwrap();
        std::fs::write(&path, b"evil").unwrap();
        assert!(pin.verify(&path).unwrap_err().contains("SHA256"));
        for (key, value) in [
            ("expectedSha256", json!("bad")),
            ("expectedSizeBytes", json!(0)),
            ("files", json!(["../escape"])),
            ("files", json!(["*.safetensors"])),
            ("revision", json!("main")),
        ] {
            let mut invalid = row(b"good");
            invalid[key] = value;
            assert!(ManagedFileIdentity::from_download(&invalid).is_err());
        }
    }
    #[test]
    fn managed_digest_is_a_typed_round_trip_contract_field() {
        let pin = builtin_identities().unwrap().first().unwrap();
        let row = json!({"provider":"huggingface","repo":pin.repo,"revision":pin.revision,
            "files":[pin.file],"expectedSizeBytes":pin.bytes,"expectedSha256":pin.sha256});
        let download: crate::contracts::ModelDownload =
            serde_json::from_value(row.clone()).unwrap();
        assert_eq!(
            download.expected_sha256.as_deref(),
            Some(pin.sha256.as_str())
        );
        assert_eq!(download.expected_size_bytes, Some(pin.bytes));
        assert!(!download.extra.contains_key("expectedSha256"));
        assert_eq!(serde_json::to_value(download).unwrap(), row);
    }

    #[test]
    fn authoritative_identity_refuses_overrides_and_fills_missing_digests() {
        let pin = builtin_identities().unwrap().first().unwrap();
        let mut payload = json!({"repo":pin.repo,"revision":pin.revision,"files":[pin.file]})
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(authoritative_job_identity(&payload).unwrap(), Some(pin));
        let mut alias = payload.clone();
        alias.insert("repo".into(), json!(pin.repo.to_ascii_lowercase()));
        assert!(
            authoritative_job_identity(&alias).is_err(),
            "HF/case-insensitive filesystem aliases must not bypass the digest authority"
        );
        pin.stamp(&mut payload);
        assert_eq!(authoritative_job_identity(&payload).unwrap(), Some(pin));
        for key in [
            "repo",
            "revision",
            "files",
            "expectedSha256",
            "expectedSizeBytes",
        ] {
            let mut conflict = payload.clone();
            conflict.insert(key.into(), json!("wrong"));
            assert!(pin.reconcile(&conflict).is_err(), "{key}");
        }
    }
}
