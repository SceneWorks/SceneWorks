//! Attempt-bound execution receipts. The API constructs the trusted fields from the
//! claimed job; a worker supplies only its completed engine report and output map.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const EXECUTION_RECEIPT_VERSION: u32 = 1;
pub const SAME_DEVICE_PINNED_RUNTIME_SCOPE: &str = "same_device_pinned_runtime";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OutputExecutionBinding {
    pub output_index: usize,
    /// A completed report for this output when the worker rendered outputs in
    /// separate provider calls. The top-level report covers shared batch calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<Value>,
    /// Ordered asset IDs committed by this output. A document may own its image
    /// assets and its document asset; a text-only output has no asset IDs.
    pub artifact_ids: Vec<String>,
    /// Worker-observed SHA-256 bytes for each artifact in the same order. An
    /// unavailable provider report may leave this empty; exact replay cannot.
    #[serde(default)]
    pub artifact_sha256: Vec<String>,
    /// Ordered provider calls that produced an inline child asset for this output.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<ExecutionStageBinding>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkerExecutionReceipt {
    pub version: u32,
    /// The provider's completed `ExecutionIdentity`, serialized from the pinned
    /// inference contract. Neither a request nor a preflight report is valid here.
    pub execution: Value,
    pub outputs: Vec<OutputExecutionBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<ExecutionStageBinding>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExecutionStageBinding {
    /// Stable call order within one completed worker attempt.
    pub stage: String,
    pub execution: Value,
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
    pub accepted_device: String,
    /// The exact server-approved source route selected by this claim, including
    /// all model and adapter members. None is valid only for model-free work.
    pub accepted_source_route: Option<Value>,
    pub inference_revision: String,
    /// Digest of the server-owned request and logical source binding at claim.
    pub accepted_binding_sha256: String,
    /// Canonical digest of the worker result accepted in the terminal
    /// transaction, before the API adds catalog side-effect metadata.
    pub accepted_result_sha256: String,
    pub execution: Value,
    pub outputs: Vec<OutputExecutionBinding>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<ExecutionStageBinding>,
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

/// Hash the accepted JSON value with recursively ordered object keys, never a
/// reconstructed browser request. This workspace enables serde's insertion
/// order preservation, so hashing raw serialization would change across a DB
/// roundtrip even when the accepted value remains identical.
pub fn json_sha256(value: &Value) -> Result<String, serde_json::Error> {
    let mut canonical = value.clone();
    canonical.sort_all_objects();
    let mut digest = Sha256::new();
    digest.update(serde_json::to_vec(&canonical)?);
    Ok(crate::hashing::hex(&digest.finalize()))
}

pub fn capability_id(secret: &str) -> String {
    crate::hashing::hex(&Sha256::digest(secret.as_bytes()))
}

fn output_asset_ids(
    result: &Map<String, Value>,
    output_count: usize,
) -> Result<Vec<Vec<String>>, ReplayUnavailable> {
    let Some(writes) = result.get("assetWrites") else {
        return Ok(vec![Vec::new(); output_count]);
    };
    let writes = writes
        .as_array()
        .ok_or_else(|| ReplayUnavailable::new("invalid_completed_assets", "assetWrites"))?;
    let mut ids = vec![Vec::new(); output_count];
    let mut unique = BTreeSet::new();
    for (position, write) in writes.iter().enumerate() {
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
        let index = write
            .get("index")
            .map(|value| value.as_u64().and_then(|index| usize::try_from(index).ok()))
            .unwrap_or(Some(position))
            .filter(|index| *index < output_count)
            .ok_or_else(|| ReplayUnavailable::new("invalid_output_order", "assetWrites.index"))?;
        ids[index].push(id.to_owned());
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
    validate_execution_identity(&submitted.execution, submitted.outputs.len())?;
    for output in &submitted.outputs {
        if let Some(execution) = &output.execution {
            validate_execution_identity(execution, 1)?;
        }
        if !output.stages.is_empty()
            && (output.stages.len() != 2
                || output.stages[0].stage != "base_render"
                || output.stages[1].stage != "child_transform")
        {
            return Err(ReplayUnavailable::new(
                "invalid_output_stages",
                "outputs.stages",
            ));
        }
        for stage in &output.stages {
            validate_execution_identity(&stage.execution, 1)?;
        }
    }
    let voice_stages = submitted.stages.len() == 2
        && submitted.stages[0].stage == "base_tts"
        && submitted.stages[1].stage == "voice_conversion";
    let seedvr2_stages = terminal_result
        .get("generationSet")
        .and_then(|set| set.get("mode"))
        .and_then(Value::as_str)
        == Some("video_upscale")
        && submitted
            .stages
            .iter()
            .enumerate()
            .all(|(index, stage)| stage.stage == format!("seedvr2_window_{index}"));
    if !submitted.stages.is_empty()
        && (submitted.outputs.len() != 1 || !(voice_stages || seedvr2_stages))
    {
        return Err(ReplayUnavailable::new("invalid_execution_stages", "stages"));
    }
    let stage_value = (!submitted.stages.is_empty())
        .then(|| serde_json::to_value(&submitted.stages).ok())
        .flatten();
    if terminal_result.get("workerExecutionStages") != stage_value.as_ref() {
        return Err(ReplayUnavailable::new("execution_stages_changed", "stages"));
    }
    let mut seen_stages = BTreeSet::new();
    for stage in &submitted.stages {
        if stage.stage.trim().is_empty() || !seen_stages.insert(stage.stage.as_str()) {
            return Err(ReplayUnavailable::new("invalid_execution_stages", "stages"));
        }
        let units = stage
            .execution
            .pointer("/identity/output_units")
            .and_then(Value::as_u64)
            .and_then(|units| usize::try_from(units).ok())
            .filter(|units| *units > 0 && *units <= 10_000)
            .unwrap_or(1);
        validate_execution_identity(&stage.execution, units)?;
    }
    let expected_ids = output_asset_ids(terminal_result, submitted.outputs.len())?;
    let mut unique = BTreeSet::new();
    for (index, output) in submitted.outputs.iter().enumerate() {
        if output.output_index != index {
            return Err(ReplayUnavailable::new("invalid_output_order", "outputs"));
        }
        for id in &output.artifact_ids {
            if id.trim().is_empty() || !unique.insert(id.as_str()) {
                return Err(ReplayUnavailable::new("invalid_output_assets", "outputs"));
            }
        }
        if !output.artifact_sha256.is_empty()
            && (output.artifact_sha256.len() != output.artifact_ids.len()
                || output.artifact_sha256.iter().any(|hash| {
                    hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                }))
        {
            return Err(ReplayUnavailable::new("invalid_output_hashes", "outputs"));
        }
        let writes = terminal_result.get("assetWrites").and_then(Value::as_array);
        if let Some(writes) = writes {
            let matching = writes
                .iter()
                .enumerate()
                .filter(|(position, fact)| {
                    fact.get("index")
                        .and_then(Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(*position)
                        == index
                })
                .map(|(_, fact)| fact)
                .collect::<Vec<_>>();
            let child_reports = matching
                .iter()
                .filter_map(|fact| fact.get("workerChildExecutionReport"))
                .collect::<Vec<_>>();
            if child_reports.is_empty() != output.stages.is_empty() {
                return Err(ReplayUnavailable::new(
                    "output_stages_changed",
                    "outputs.stages",
                ));
            }
            if !child_reports.is_empty() {
                let base = matching
                    .first()
                    .and_then(|fact| fact.get("workerExecutionReport"));
                let base_matches = base.map_or_else(
                    || {
                        output.stages[0]
                            .execution
                            .get("state")
                            .and_then(Value::as_str)
                            == Some("unavailable")
                    },
                    |report| report == &output.stages[0].execution,
                );
                if matching.len() != 2
                    || matching[1].get("sourceAssetId") != matching[0].get("assetId")
                    || child_reports.len() != 1
                    || !base_matches
                    || child_reports[0] != &output.stages[1].execution
                {
                    return Err(ReplayUnavailable::new(
                        "output_stages_changed",
                        "outputs.stages",
                    ));
                }
            }
        }
    }
    if terminal_result
        .get("expectedCount")
        .and_then(Value::as_u64)
        .is_some_and(|count| count as usize != submitted.outputs.len())
    {
        return Err(ReplayUnavailable::new(
            "incomplete_completed_assets",
            "expectedCount",
        ));
    }
    if submitted
        .outputs
        .iter()
        .zip(&expected_ids)
        .any(|(output, expected)| output.artifact_ids != *expected)
    {
        return Err(ReplayUnavailable::new(
            "output_artifacts_changed",
            "outputs",
        ));
    }
    Ok(())
}

fn parse_execution_identity(
    execution: &Value,
    output_units: usize,
) -> Result<gen_core::generator::ExecutionIdentity, ReplayUnavailable> {
    let report: gen_core::generator::ExecutionIdentity = serde_json::from_value(execution.clone())
        .map_err(|_| ReplayUnavailable::new("invalid_execution_identity", "execution"))?;
    let (provider, backend) = report
        .resolved()
        .map(|identity| (identity.provider_id.as_str(), identity.backend.as_str()))
        .unwrap_or(("", ""));
    if report.resolved().is_some() && (provider.trim().is_empty() || backend.trim().is_empty()) {
        return Err(ReplayUnavailable::new(
            "invalid_execution_identity",
            "execution.identity",
        ));
    }
    report
        .validate(provider, backend, output_units)
        .map_err(|_| ReplayUnavailable::new("invalid_execution_identity", "execution.identity"))?;
    Ok(report)
}

fn validate_execution_identity(
    execution: &Value,
    output_units: usize,
) -> Result<(), ReplayUnavailable> {
    parse_execution_identity(execution, output_units).map(|_| ())
}

/// Exact replay is a separate action. A resolved identity is insufficient: the
/// provider must attest deterministic replay for the requested pinned scope.
pub fn require_deterministic_scope(
    receipt: &TrustedExecutionReceipt,
    actual_scope: &str,
) -> Result<(), ReplayUnavailable> {
    if !receipt.stages.is_empty()
        || receipt
            .outputs
            .iter()
            .any(|output| !output.stages.is_empty())
    {
        return Err(ReplayUnavailable::new(
            "composite_replay_unavailable",
            "stages",
        ));
    }
    for output in &receipt.outputs {
        let execution = output.execution.as_ref().unwrap_or(&receipt.execution);
        let report = parse_execution_identity(
            execution,
            if output.execution.is_some() {
                1
            } else {
                receipt.outputs.len()
            },
        )?;
        let Some(identity) = report.resolved() else {
            return Err(ReplayUnavailable::new("execution_unavailable", "execution"));
        };
        if !identity.deterministic_replay.is_deterministic() {
            return Err(ReplayUnavailable::new(
                "deterministic_replay_unavailable",
                "execution.deterministic_replay",
            ));
        }
        let gen_core::generator::ExecutionReplaySupport::Deterministic {
            scope: declared_scope,
        } = &identity.deterministic_replay
        else {
            unreachable!("is_deterministic guarded this branch")
        };
        if declared_scope != actual_scope {
            return Err(ReplayUnavailable::new(
                "replay_scope_changed",
                "execution.deterministic_replay",
            ));
        }
    }
    if receipt
        .outputs
        .iter()
        .all(|output| output.artifact_ids.is_empty())
        || receipt.outputs.iter().any(|output| {
            output.artifact_sha256.len() != output.artifact_ids.len()
                || output.artifact_sha256.iter().any(|hash| {
                    hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                })
        })
    {
        return Err(ReplayUnavailable::new(
            "output_hashes_unavailable",
            "outputs",
        ));
    }
    Ok(())
}

/// A completed provider identity must name one of the model identities in the
/// server-selected source route before a stored report can authorize replay.
/// Imported wrappers without a stable descriptor-to-source mapping refuse
/// replay until that mapping is made explicit.
pub fn require_provider_route_binding(
    receipt: &TrustedExecutionReceipt,
    route: &crate::model_artifacts::portable_binding::PortableRouteBinding,
) -> Result<(), ReplayUnavailable> {
    if route.backend != receipt.accepted_backend || route.models.is_empty() {
        return Err(ReplayUnavailable::new("provider_route_changed", "source"));
    }
    for output in &receipt.outputs {
        let execution = output.execution.as_ref().unwrap_or(&receipt.execution);
        let report = parse_execution_identity(
            execution,
            if output.execution.is_some() {
                1
            } else {
                receipt.outputs.len()
            },
        )?;
        let identity = report
            .resolved()
            .ok_or_else(|| ReplayUnavailable::new("execution_unavailable", "execution"))?;
        if identity.backend != route.backend
            || !route.models.iter().any(|model| {
                model.model_id == identity.provider_id
                    || (identity.provider_id == "instantid"
                        && model.model_id == "instantid_realvisxl")
            })
        {
            return Err(ReplayUnavailable::new("provider_route_changed", "source"));
        }
    }
    Ok(())
}

/// The current image worker resolves an explicit base seed before per-image
/// seeds. Exact replay uses the unchanged accepted payload, so every observed
/// draw must already equal that payload's effective seed. Unobserved or
/// multi-stream randomness needs a route-specific replay adapter.
pub fn require_image_seed_binding(
    receipt: &TrustedExecutionReceipt,
    payload: &Map<String, Value>,
) -> Result<(), ReplayUnavailable> {
    let count = receipt.outputs.len();
    if payload.get("count").and_then(Value::as_u64).unwrap_or(1) as usize != count {
        return Err(ReplayUnavailable::new(
            "unsupported_draw_mapping",
            "execution.randomness",
        ));
    }
    let base = payload.get("seed").and_then(Value::as_i64);
    let seeds = payload.get("seeds").and_then(Value::as_array);
    for (index, output) in receipt.outputs.iter().enumerate() {
        let randomness = output
            .execution
            .as_ref()
            .unwrap_or(&receipt.execution)
            .pointer("/identity/randomness")
            .ok_or_else(|| ReplayUnavailable::new("missing_randomness", "execution.randomness"))?;
        let draws = randomness
            .get("draws")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ReplayUnavailable::new("unsupported_draw_mapping", "execution.randomness")
            })?;
        if draws.len() != if output.execution.is_some() { 1 } else { count } {
            return Err(ReplayUnavailable::new(
                "unsupported_draw_mapping",
                "execution.randomness",
            ));
        }
        let provider_index = if output.execution.is_some() { 0 } else { index };
        let draw = &draws[provider_index];
        let expected = base
            .map(|seed| seed.wrapping_add(index as i64))
            .or_else(|| seeds?.get(index)?.as_i64())
            .and_then(|seed| u64::try_from(seed).ok());
        if draw.get("output_index").and_then(Value::as_u64) != Some(provider_index as u64)
            || draw
                .get("noise_sha256")
                .is_some_and(|value| !value.is_null())
            || draw.get("seed").and_then(Value::as_u64) != expected
            || expected.is_none()
        {
            return Err(ReplayUnavailable::new(
                "seed_draw_binding_changed",
                "execution.randomness",
            ));
        }
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
                "version":1,"provider_id":"fixture","backend":"candle","runtime_id":"fixture","output_units":2,
                "sampling":{"state":"resolved","value":{"sampler":"ddim"}},
                "schedule":{"state":"resolved","value":{"steps":2}},
                "estimator":{"state":"not_applicable","reason":"none"},
                "randomness":{"kind":"draws","scheme":"cpu_std_rng","draws":[
                    {"output_index":0,"purpose":"initial_noise","seed":41,"noise_sha256":null},
                    {"output_index":1,"purpose":"initial_noise","seed":42,"noise_sha256":null}
                ]},
                "deterministic_replay":{"kind":"deterministic","scope":"same-device"},
                "geometry":{"state":"resolved","value":{"width":64,"height":64}},
                "quality_policy":{"state":"resolved","value":{"rung":"resident"}},
                "output_policy":{"state":"resolved","value":{"mime":"image/png"}},
                "provider_choices":{}
            }},
            "outputs": [
                {"outputIndex":0,"artifactIds":["a"],"artifactSha256":["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"]},
                {"outputIndex":1,"artifactIds":["b"],"artifactSha256":["bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"]}
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
        let mut variants = worker_receipt();
        variants.outputs[0].artifact_ids.push("upscaled-a".into());
        variants.outputs[0].artifact_sha256.push("c".repeat(64));
        let variant_result = json!({"expectedCount":2,"assetWrites":[
            {"assetId":"a","index":0},{"assetId":"b","index":1},{"assetId":"upscaled-a","index":0}
        ]});
        assert!(validate_worker_receipt(&variants, variant_result.as_object().unwrap()).is_ok());
        let mut swapped = variants;
        swapped.outputs.swap(0, 1);
        swapped.outputs[0].output_index = 0;
        swapped.outputs[1].output_index = 1;
        assert_eq!(
            validate_worker_receipt(&swapped, variant_result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_artifacts_changed"
        );
    }

    #[test]
    fn binding_digest_survives_json_object_order_changes() {
        let left: Value = serde_json::from_str(r#"{"a":{"x":1,"y":2},"b":3}"#).unwrap();
        let right: Value = serde_json::from_str(r#"{"b":3,"a":{"y":2,"x":1}}"#).unwrap();
        assert_eq!(json_sha256(&left).unwrap(), json_sha256(&right).unwrap());
    }

    #[test]
    fn incomplete_resolved_provider_report_cannot_become_trusted() {
        let mut report = worker_receipt();
        report.execution["identity"]
            .as_object_mut()
            .unwrap()
            .remove("quality_policy");
        let terminal = json!({"expectedCount":2,"assetWrites":[{"assetId":"a"},{"assetId":"b"}]});
        assert_eq!(
            validate_worker_receipt(&report, terminal.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_identity"
        );
    }

    #[test]
    fn two_stage_voice_receipt_requires_order_and_complete_terminal_binding() {
        let stage_report = json!({"state":"unavailable","reason":{
            "code":"provider_report_unavailable","detail":"fixture","missing_facts":["provider_resolved_execution"]
        }});
        let stages = vec![
            ExecutionStageBinding {
                stage: "base_tts".into(),
                execution: stage_report.clone(),
            },
            ExecutionStageBinding {
                stage: "voice_conversion".into(),
                execution: stage_report,
            },
        ];
        let mut receipt = worker_receipt();
        receipt.execution = json!({"state":"unavailable","reason":{
            "code":"composite_execution_unavailable","detail":"fixture","missing_facts":["composite_determinism"]
        }});
        receipt.outputs.truncate(1);
        receipt.stages = stages.clone();
        let result = json!({"expectedCount":1,"assetWrites":[{"assetId":"a"}],"workerExecutionStages":stages});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        receipt.stages.swap(0, 1);
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_stages"
        );
        receipt.stages.pop();
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_stages"
        );
        let mut omitted = receipt.clone();
        omitted.stages = serde_json::from_value(result["workerExecutionStages"].clone()).unwrap();
        let missing = json!({"expectedCount":1,"assetWrites":[{"assetId":"a"}]});
        assert_eq!(
            validate_worker_receipt(&omitted, missing.as_object().unwrap())
                .unwrap_err()
                .reason,
            "execution_stages_changed"
        );
    }

    #[test]
    fn seedvr2_window_receipts_require_contiguous_order_and_upscale_result() {
        let unavailable = json!({"state":"unavailable","reason":{
            "code":"provider_report_unavailable","detail":"fixture","missing_facts":["provider_resolved_execution"]
        }});
        let stages = (0..3)
            .map(|index| ExecutionStageBinding {
                stage: format!("seedvr2_window_{index}"),
                execution: unavailable.clone(),
            })
            .collect::<Vec<_>>();
        let mut receipt = worker_receipt();
        receipt.execution = unavailable;
        receipt.outputs.truncate(1);
        receipt.stages = stages.clone();
        let result = json!({"expectedCount":1,"generationSet":{"mode":"video_upscale"},
            "assetWrites":[{"assetId":"a"}],"workerExecutionStages":stages});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        receipt.stages.swap(0, 1);
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_stages"
        );
        receipt.stages.swap(0, 1);
        let mut wrong_route = result.clone();
        wrong_route["generationSet"]["mode"] = json!("video_generate");
        assert_eq!(
            validate_worker_receipt(&receipt, wrong_route.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_stages"
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
            accepted_device: "cuda:0".into(),
            accepted_source_route: None,
            inference_revision: "a".repeat(40),
            accepted_binding_sha256: "b".repeat(64),
            accepted_result_sha256: "c".repeat(64),
            execution: input.execution,
            outputs: input.outputs,
            stages: Vec::new(),
        };
        assert!(require_deterministic_scope(&receipt, "same-device").is_ok());
        assert_eq!(
            require_deterministic_scope(&receipt, "other-device")
                .unwrap_err()
                .reason,
            "replay_scope_changed"
        );
        let mut broad_token_claim = receipt.clone();
        broad_token_claim.execution["identity"]["deterministic_replay"] = json!({
            "kind":"deterministic",
            "scope":"Same pinned model/tokenizer bytes, inference revision, device, dtype and effective decode/cache policy; identical conditioning and recorded RNG stream"
        });
        assert_eq!(
            require_deterministic_scope(&broad_token_claim, SAME_DEVICE_PINNED_RUNTIME_SCOPE)
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

    #[test]
    fn instantid_provider_alias_is_narrow_and_backend_bound() {
        use crate::model_artifacts::portable_binding::{
            PortableModelBinding, PortableRouteBinding,
        };
        let mut worker = worker_receipt();
        worker.execution["identity"]["provider_id"] = json!("instantid");
        let receipt = TrustedExecutionReceipt {
            version: 1,
            job_id: "job".into(),
            job_type: "image_generate".into(),
            worker_id: "worker".into(),
            attempt_id: "a".repeat(64),
            accepted_backend: "candle".into(),
            accepted_device: "cuda:0".into(),
            accepted_source_route: None,
            inference_revision: "b".repeat(40),
            accepted_binding_sha256: "c".repeat(64),
            accepted_result_sha256: "d".repeat(64),
            execution: worker.execution,
            outputs: worker.outputs,
            stages: Vec::new(),
        };
        let mut route = PortableRouteBinding {
            backend: "candle".into(),
            platform: "windows".into(),
            models: vec![PortableModelBinding {
                schema_version: 1,
                slot: "modelManifestEntry".into(),
                model_id: "instantid_realvisxl".into(),
                backend: "candle".into(),
                variant: "default".into(),
                tier: None,
                authored_options_sha256: "e".repeat(64),
                members: Vec::new(),
                checkpoint_plan: None,
            }],
        };
        assert!(require_provider_route_binding(&receipt, &route).is_ok());
        route.models[0].model_id = "another_instantid".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
        route.models[0].model_id = "instantid_realvisxl".into();
        route.backend = "mlx".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
    }

    #[test]
    fn exact_image_replay_requires_observed_draws_to_match_frozen_seed() {
        let input = worker_receipt();
        let mut receipt = TrustedExecutionReceipt {
            version: 1,
            job_id: "job".into(),
            job_type: "image_generate".into(),
            worker_id: "worker".into(),
            attempt_id: "f".repeat(64),
            accepted_backend: "candle".into(),
            accepted_device: "cuda:0".into(),
            accepted_source_route: None,
            inference_revision: "a".repeat(40),
            accepted_binding_sha256: "b".repeat(64),
            accepted_result_sha256: "c".repeat(64),
            execution: input.execution,
            outputs: input.outputs,
            stages: Vec::new(),
        };
        receipt.execution["identity"]["randomness"] = json!({
            "kind":"draws", "scheme":"cpu_std_rng", "draws":[
                {"output_index":0,"purpose":"initial_noise","seed":41,"noise_sha256":null},
                {"output_index":1,"purpose":"initial_noise","seed":42,"noise_sha256":null}
            ]
        });
        let fixed = json!({"seed":41,"count":2});
        assert!(require_image_seed_binding(&receipt, fixed.as_object().unwrap()).is_ok());
        let changed = json!({"seed":42,"count":2});
        assert_eq!(
            require_image_seed_binding(&receipt, changed.as_object().unwrap())
                .unwrap_err()
                .reason,
            "seed_draw_binding_changed"
        );
        let implicit = json!({"prompt":"same","count":2});
        assert_eq!(
            require_image_seed_binding(&receipt, implicit.as_object().unwrap())
                .unwrap_err()
                .reason,
            "seed_draw_binding_changed"
        );
    }
}
