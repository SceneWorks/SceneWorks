//! Attempt-bound execution receipts. The API constructs the trusted fields from the
//! claimed job; a worker supplies only its completed engine report and output map.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const EXECUTION_RECEIPT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OutputExecutionBinding {
    pub output_index: usize,
    /// Ordered asset IDs committed by this output. A document may own its image
    /// assets and its document asset; a text-only output has no asset IDs.
    pub artifact_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerExecutionReceipt {
    pub version: u32,
    /// The provider's completed `ExecutionIdentity`, serialized from the pinned
    /// inference contract. Neither a request nor a preflight report is valid here.
    pub execution: Value,
    pub outputs: Vec<OutputExecutionBinding>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrustedExecutionReceipt {
    pub version: u32,
    pub job_id: String,
    pub job_type: String,
    pub worker_id: String,
    /// SHA-256 of the private per-claim capability. The capability itself never
    /// enters a job snapshot, result, SSE event or share.
    pub attempt_id: String,
    pub accepted_backend: String,
    /// The exact server-approved source route selected by this claim, including
    /// all model and adapter members. None is valid only for model-free work.
    pub accepted_source_route: Option<Value>,
    pub inference_revision: String,
    /// Digest of the server-owned request and logical source binding at claim.
    pub accepted_binding_sha256: String,
    pub completed_result_sha256: String,
    pub execution: Value,
    pub outputs: Vec<OutputExecutionBinding>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayUnavailable {
    pub code: String,
    pub reason: String,
    pub field: String,
}

impl ReplayUnavailable {
    pub fn new(reason: &str, field: &str) -> Self {
        Self {
            code: "replay_unavailable".into(),
            reason: reason.into(),
            field: field.into(),
        }
    }
}

/// Hash exactly the accepted JSON bytes, never a reconstructed browser request.
/// `serde_json::Map` is key-ordered in this workspace, so this is stable across
/// API/worker processes with the same pinned serde implementation.
pub fn json_sha256(value: &Value) -> Result<String, serde_json::Error> {
    let mut digest = Sha256::new();
    digest.update(serde_json::to_vec(value)?);
    Ok(crate::hashing::hex(&digest.finalize()))
}

pub fn capability_id(secret: &str) -> String {
    crate::hashing::hex(&Sha256::digest(secret.as_bytes()))
}

fn output_asset_ids(result: &Map<String, Value>) -> Result<Vec<String>, ReplayUnavailable> {
    let Some(writes) = result.get("assetWrites") else {
        return Ok(Vec::new());
    };
    let writes = writes
        .as_array()
        .ok_or_else(|| ReplayUnavailable::new("invalid_completed_assets", "assetWrites"))?;
    if result
        .get("expectedCount")
        .and_then(Value::as_u64)
        .is_some_and(|count| count as usize != writes.len())
    {
        return Err(ReplayUnavailable::new(
            "incomplete_completed_assets",
            "expectedCount",
        ));
    }
    let mut ids = Vec::with_capacity(writes.len());
    let mut unique = BTreeSet::new();
    for write in writes {
        let id = write
            .get("assetId")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
            .ok_or_else(|| ReplayUnavailable::new("invalid_completed_assets", "assetWrites"))?;
        if !unique.insert(id) {
            return Err(ReplayUnavailable::new(
                "duplicate_completed_asset",
                "assetWrites",
            ));
        }
        ids.push(id.to_owned());
    }
    Ok(ids)
}

/// Validate the worker's output association against the terminal result. This
/// does not make a worker-authored identity trusted by itself: the store must
/// also check the private capability and current claim in the same transaction.
pub fn validate_worker_receipt(
    submitted: &WorkerExecutionReceipt,
    terminal_result: &Map<String, Value>,
) -> Result<(), ReplayUnavailable> {
    if submitted.version != EXECUTION_RECEIPT_VERSION {
        return Err(ReplayUnavailable::new("unknown_receipt_version", "version"));
    }
    if submitted.outputs.is_empty() {
        return Err(ReplayUnavailable::new("missing_outputs", "outputs"));
    }
    let state = submitted
        .execution
        .get("state")
        .and_then(Value::as_str)
        .ok_or_else(|| ReplayUnavailable::new("invalid_execution_identity", "execution"))?;
    match state {
        "resolved" => {
            let identity = submitted.execution.get("identity").ok_or_else(|| {
                ReplayUnavailable::new("invalid_execution_identity", "execution.identity")
            })?;
            if identity.get("version").and_then(Value::as_u64) != Some(1)
                || identity.get("output_units").and_then(Value::as_u64)
                    != Some(submitted.outputs.len() as u64)
                || identity
                    .get("runtime_id")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                || identity.get("deterministic_replay").is_none()
            {
                return Err(ReplayUnavailable::new(
                    "invalid_execution_identity",
                    "execution.identity",
                ));
            }
        }
        "unavailable" => {
            let reason = submitted.execution.get("reason").ok_or_else(|| {
                ReplayUnavailable::new("invalid_execution_identity", "execution.reason")
            })?;
            if reason.get("code").and_then(Value::as_str).is_none()
                || reason.get("detail").and_then(Value::as_str).is_none()
            {
                return Err(ReplayUnavailable::new(
                    "invalid_execution_identity",
                    "execution.reason",
                ));
            }
        }
        _ => {
            return Err(ReplayUnavailable::new(
                "invalid_execution_identity",
                "execution.state",
            ))
        }
    }
    let expected_ids = output_asset_ids(terminal_result)?;
    let mut submitted_ids = Vec::new();
    let mut unique = BTreeSet::new();
    for (index, output) in submitted.outputs.iter().enumerate() {
        if output.output_index != index {
            return Err(ReplayUnavailable::new("invalid_output_order", "outputs"));
        }
        for id in &output.artifact_ids {
            if id.trim().is_empty() || !unique.insert(id.as_str()) {
                return Err(ReplayUnavailable::new("invalid_output_assets", "outputs"));
            }
            submitted_ids.push(id.clone());
        }
    }
    if submitted_ids != expected_ids {
        return Err(ReplayUnavailable::new(
            "output_artifacts_changed",
            "outputs",
        ));
    }
    Ok(())
}

/// Exact replay is a separate action. A resolved identity is insufficient: the
/// provider must attest deterministic replay for the requested pinned scope.
pub fn require_deterministic_scope(
    receipt: &TrustedExecutionReceipt,
    actual_scope: &str,
) -> Result<(), ReplayUnavailable> {
    let Some(identity) = receipt.execution.get("identity") else {
        return Err(ReplayUnavailable::new("execution_unavailable", "execution"));
    };
    let support = identity.get("deterministic_replay").ok_or_else(|| {
        ReplayUnavailable::new("execution_unavailable", "execution.deterministic_replay")
    })?;
    if support.get("kind").and_then(Value::as_str) != Some("deterministic") {
        return Err(ReplayUnavailable::new(
            "deterministic_replay_unavailable",
            "execution.deterministic_replay",
        ));
    }
    let declared_scope = support
        .get("scope")
        .and_then(Value::as_str)
        .filter(|scope| !scope.is_empty())
        .ok_or_else(|| {
            ReplayUnavailable::new("invalid_replay_scope", "execution.deterministic_replay")
        })?;
    if declared_scope != actual_scope {
        return Err(ReplayUnavailable::new(
            "replay_scope_changed",
            "execution.deterministic_replay",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn worker_receipt() -> WorkerExecutionReceipt {
        serde_json::from_value(json!({
            "version": 1,
            "execution": {"state":"resolved","identity":{
                "version":1,"runtime_id":"fixture","output_units":2,
                "deterministic_replay":{"kind":"deterministic","scope":"same-device"}
            }},
            "outputs": [
                {"outputIndex":0,"artifactIds":["a"]},
                {"outputIndex":1,"artifactIds":["b"]}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn output_binding_matches_ordered_terminal_assets() {
        let result = json!({"expectedCount":2,"assetWrites":[{"assetId":"a"},{"assetId":"b"}]});
        assert!(validate_worker_receipt(&worker_receipt(), result.as_object().unwrap()).is_ok());
        let mut changed = worker_receipt();
        changed.outputs[1].artifact_ids = vec!["a".into()];
        assert_eq!(
            validate_worker_receipt(&changed, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_output_assets"
        );
        changed.outputs[1].artifact_ids = vec!["elsewhere".into()];
        assert_eq!(
            validate_worker_receipt(&changed, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_artifacts_changed"
        );
    }

    #[test]
    fn deterministic_scope_is_explicit_and_exact() {
        let input = worker_receipt();
        let receipt = TrustedExecutionReceipt {
            version: 1,
            job_id: "job".into(),
            job_type: "image_generate".into(),
            worker_id: "worker".into(),
            attempt_id: "f".repeat(64),
            accepted_backend: "candle".into(),
            accepted_source_route: None,
            inference_revision: "a".repeat(40),
            accepted_binding_sha256: "b".repeat(64),
            completed_result_sha256: "c".repeat(64),
            execution: input.execution,
            outputs: input.outputs,
        };
        assert!(require_deterministic_scope(&receipt, "same-device").is_ok());
        assert_eq!(
            require_deterministic_scope(&receipt, "other-device")
                .unwrap_err()
                .reason,
            "replay_scope_changed"
        );
        let mut refused = receipt;
        refused.execution["identity"]["deterministic_replay"] =
            json!({"kind":"non_deterministic","reason":"unverified"});
        assert_eq!(
            require_deterministic_scope(&refused, "same-device")
                .unwrap_err()
                .reason,
            "deterministic_replay_unavailable"
        );
    }
}
