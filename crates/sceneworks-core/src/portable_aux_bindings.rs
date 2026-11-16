//! Worker verified auxiliary source paths for one dispatch attempt.
//!
//! The source guard strips all client binding hints before it constructs its private
//! dispatch payload. Only that guard may call `from_trusted_dispatch`; request parsing
//! and the public options resolver must initialize this context to `Default::default()`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::contracts::JsonObject;
use crate::model_artifacts::portable_binding::PortableArtifactFile;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PortableAuxBindings {
    bound: bool,
    paths: BTreeMap<String, PathBuf>,
    frozen_sources: bool,
    gemma_files: Option<Vec<PortableArtifactFile>>,
    bound_tier: Option<String>,
}

impl PortableAuxBindings {
    /// Lower the guard's private, verified dispatch stamp into typed route facts.
    /// A public authored payload is never a valid input to this constructor.
    pub fn from_trusted_dispatch(payload: &JsonObject) -> Result<Self, String> {
        let Some(advanced) = payload.get("advanced").and_then(Value::as_object) else {
            return Ok(Self::default());
        };
        let frozen_sources = advanced.get("__portableFrozenSources") == Some(&Value::Bool(true));
        if !frozen_sources
            && [
                "__portableAuxPaths",
                "__portableGemmaFiles",
                "__portableBoundTier",
            ]
            .iter()
            .any(|key| advanced.contains_key(*key))
        {
            return Err("portable route facts lack a verified source stamp".to_owned());
        }
        let bound = advanced.contains_key("__portableAuxPaths");
        let map = match advanced.get("__portableAuxPaths") {
            Some(paths) => Some(
                paths
                    .as_object()
                    .ok_or_else(|| "portable auxiliary paths are not an object".to_owned())?,
            ),
            None => None,
        };
        let mut result = BTreeMap::new();
        for (source_id, path) in map.into_iter().flatten() {
            if source_id.is_empty() {
                return Err("portable auxiliary source ID is empty".to_owned());
            }
            let path = path
                .as_str()
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .ok_or_else(|| format!("portable auxiliary path for {source_id} is invalid"))?;
            result.insert(source_id.clone(), path);
        }
        let gemma_files = advanced
            .get("__portableGemmaFiles")
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()
            .map_err(|error| format!("portable Gemma ledger is invalid: {error}"))?;
        let bound_tier = advanced
            .get("__portableBoundTier")
            .map(|value| {
                value
                    .as_str()
                    .filter(|tier| !tier.is_empty())
                    .map(str::to_owned)
                    .ok_or_else(|| "portable bound tier is invalid".to_owned())
            })
            .transpose()?;
        Ok(Self {
            bound,
            paths: result,
            frozen_sources,
            gemma_files,
            bound_tier,
        })
    }

    pub fn has_paths(&self) -> bool {
        self.bound
    }

    pub fn path(&self, source_id: &str) -> Option<&Path> {
        self.paths.get(source_id).map(PathBuf::as_path)
    }

    pub fn frozen_sources(&self) -> bool {
        self.frozen_sources
    }

    pub fn gemma_files(&self) -> Option<&[PortableArtifactFile]> {
        self.gemma_files.as_deref()
    }

    pub fn bound_tier(&self) -> Option<&str> {
        self.bound_tier.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn object(value: Value) -> JsonObject {
        value.as_object().unwrap().clone()
    }

    #[test]
    fn verified_aux_paths_are_typed_and_empty_bound_map_stays_distinct() {
        let path = std::env::temp_dir().join("source.safetensors");
        let payload = object(json!({"advanced": {
            "__portableFrozenSources": true,
            "__portableAuxPaths": {"strict_control": path}
        }}));
        let bound = PortableAuxBindings::from_trusted_dispatch(&payload).unwrap();
        assert!(bound.has_paths());
        assert_eq!(bound.path("strict_control"), Some(path.as_path()));
        assert_eq!(bound.path("missing"), None);
        assert!(!PortableAuxBindings::default().has_paths());

        let empty = object(json!({"advanced": {
            "__portableFrozenSources": true, "__portableAuxPaths": {}
        }}));
        assert!(PortableAuxBindings::from_trusted_dispatch(&empty)
            .unwrap()
            .has_paths());
    }

    #[test]
    fn unverified_or_malformed_paths_fail_closed() {
        for payload in [
            json!({"advanced":{"__portableAuxPaths":{"strict_control":"/tmp/source"}}}),
            json!({"advanced":{"__portableFrozenSources":true,"__portableAuxPaths":{"strict_control":"relative/source"}}}),
            json!({"advanced":{"__portableFrozenSources":true,"__portableAuxPaths":[]}}),
        ] {
            assert!(PortableAuxBindings::from_trusted_dispatch(&object(payload)).is_err());
        }
    }
}
