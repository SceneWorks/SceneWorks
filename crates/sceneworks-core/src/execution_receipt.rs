//! Attempt-bound execution receipts. The API constructs the trusted fields from the
//! claimed job; a worker supplies only its completed engine report and output map.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const EXECUTION_RECEIPT_VERSION: u32 = 1;
pub const SAME_DEVICE_PINNED_RUNTIME_SCOPE: &str = "same_device_pinned_runtime";
pub const LOW_RES_START_RECIPE_VERSION: u32 = 1;
pub const LOW_RES_START_ESTIMATOR_ENDPOINT_V1: &str = "endpoint_evaluation_v1";
pub const LOW_RES_START_ESTIMATOR_PREVIOUS_V1: &str = "previous_evaluation_v1";
pub const LOW_RES_START_TRANSITION_ARITHMETIC_V1: &str =
    "unpack_f32_bilinear_half_pixel_clamp_renoise_repack_v1";
pub const LOW_RES_START_TRANSITION_SEED_SCHEME_V1: &str = "low_res_start_transition_v1";
pub const LOW_RES_START_SCHEDULE_POLICY_V1: &str = "full_pass_full_resolution_invariant_v1";

/// Resolved `advanced.lowResStart` recipe and estimator provenance persisted in the trusted
/// execution receipt (`OPT-6`, `ADM-3`, `FEAT-1`, `M2`, `E6`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LowResStartReceipt {
    pub version: u32,
    pub k: u32,
    pub executed_steps: u32,
    pub scale: f64,
    pub low_w: u32,
    pub low_h: u32,
    pub full_w: u32,
    pub full_h: u32,
    pub target_pass: usize,
    pub estimator: String,
    pub schedule_policy: String,
    pub transition_arithmetic: String,
    pub transition_seed_scheme: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_rung: Option<String>,
}

impl LowResStartReceipt {
    pub fn validate(&self) -> Result<(), ReplayUnavailable> {
        if self.version != LOW_RES_START_RECIPE_VERSION {
            return Err(ReplayUnavailable::new(
                "unsupported_low_res_recipe_version",
                "lowResStart.version",
            ));
        }
        if self.k == 0 || self.k >= self.executed_steps {
            return Err(ReplayUnavailable::new(
                "invalid_low_res_step_budget",
                "lowResStart.k",
            ));
        }
        if !(self.scale.is_finite() && (0.25..=0.9).contains(&self.scale)) {
            return Err(ReplayUnavailable::new(
                "invalid_low_res_scale",
                "lowResStart.scale",
            ));
        }
        if self.low_w == 0
            || self.low_h == 0
            || self.full_w == 0
            || self.full_h == 0
            || self.low_w > self.full_w
            || self.low_h > self.full_h
            || (self.low_w == self.full_w && self.low_h == self.full_h)
        {
            return Err(ReplayUnavailable::new(
                "invalid_low_res_dimensions",
                "lowResStart.dimensions",
            ));
        }
        if self.target_pass != 0 {
            return Err(ReplayUnavailable::new(
                "invalid_low_res_target_pass",
                "lowResStart.targetPass",
            ));
        }
        if !matches!(
            self.estimator.as_str(),
            LOW_RES_START_ESTIMATOR_ENDPOINT_V1 | LOW_RES_START_ESTIMATOR_PREVIOUS_V1
        ) {
            return Err(ReplayUnavailable::new(
                "unsupported_low_res_estimator",
                "lowResStart.estimator",
            ));
        }
        if self.schedule_policy != LOW_RES_START_SCHEDULE_POLICY_V1 {
            return Err(ReplayUnavailable::new(
                "unsupported_low_res_schedule_policy",
                "lowResStart.schedulePolicy",
            ));
        }
        if self.transition_arithmetic != LOW_RES_START_TRANSITION_ARITHMETIC_V1 {
            return Err(ReplayUnavailable::new(
                "unsupported_low_res_transition_arithmetic",
                "lowResStart.transitionArithmetic",
            ));
        }
        if self.transition_seed_scheme != LOW_RES_START_TRANSITION_SEED_SCHEME_V1 {
            return Err(ReplayUnavailable::new(
                "unsupported_low_res_transition_seed_scheme",
                "lowResStart.transitionSeedScheme",
            ));
        }
        if self
            .selected_rung
            .as_deref()
            .is_some_and(|rung| rung.trim().is_empty())
        {
            return Err(ReplayUnavailable::new(
                "invalid_quality_rung",
                "lowResStart.selectedRung",
            ));
        }
        Ok(())
    }
}

/// Derive the deterministic transition re-noising seed for a single-pass low-res start transition.
/// Matches `gen_core::low_res_start_transition_seed`.
pub fn low_res_start_transition_seed(
    request_seed: u64,
    output_index: u32,
    pass_index: usize,
    phase_index: usize,
) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(LOW_RES_START_TRANSITION_SEED_SCHEME_V1.as_bytes());
    hasher.update([0u8]);
    hasher.update(request_seed.to_le_bytes());
    hasher.update(output_index.to_le_bytes());
    hasher.update((pass_index as u64).to_le_bytes());
    hasher.update((phase_index as u64).to_le_bytes());
    let digest = hasher.finalize();
    let seed = u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]);
    if seed == request_seed {
        seed ^ 0x9E37_79B9_7F4A_7C15
    } else {
        seed
    }
}

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_res_start: Option<LowResStartReceipt>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub low_res_start: Option<LowResStartReceipt>,
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
    if result.contains_key("assetWrites") && result.contains_key("datasetWrites") {
        return Err(ReplayUnavailable::new(
            "invalid_completed_assets",
            "datasetWrites",
        ));
    }
    let Some(writes) = result
        .get("assetWrites")
        .or_else(|| result.get("datasetWrites"))
    else {
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
    if let Some(low_res) = &submitted.low_res_start {
        low_res.validate()?;
    }
    if submitted.outputs.is_empty() {
        return Err(ReplayUnavailable::new("missing_outputs", "outputs"));
    }
    validate_execution_identity(&submitted.execution, submitted.outputs.len())?;
    for output in &submitted.outputs {
        if let Some(execution) = &output.execution {
            validate_execution_identity(execution, 1)?;
        }
        let child_transform_stages = output.stages.len() == 2
            && output.stages[0].stage == "base_render"
            && output.stages[1].stage == "child_transform";
        let instantid_stages = matches!(
            output.stages.first().map(|stage| stage.stage.as_str()),
            Some("identity" | "angle" | "pose")
        ) && matches!(output.stages.len(), 2 | 3)
            && output.stages[1].stage == "restore_face"
            && (output.stages.len() == 2 || output.stages[2].stage == "child_transform");
        let hires_stages = matches!(output.stages.len(), 2 | 3)
            && output.stages[0].stage == "base"
            && output.stages[1].stage == "refine"
            && (output.stages.len() == 2 || output.stages[2].stage == "child_transform");
        if !output.stages.is_empty()
            && !(child_transform_stages || instantid_stages || hires_stages)
        {
            return Err(ReplayUnavailable::new(
                "invalid_output_stages",
                "outputs.stages",
            ));
        }
        for stage in &output.stages {
            let shared_base = stage.stage == "base_render"
                && stage
                    .execution
                    .pointer("/identity/output_units")
                    .and_then(Value::as_u64)
                    == Some(submitted.outputs.len() as u64)
                && submitted.outputs.len() > 1;
            if shared_base
                && !submitted.outputs.iter().all(|other| {
                    other
                        .stages
                        .first()
                        .filter(|first| first.stage == "base_render")
                        .map(|first| &first.execution)
                        .or(other.execution.as_ref())
                        == Some(&stage.execution)
                })
            {
                return Err(ReplayUnavailable::new(
                    "invalid_execution_identity",
                    "outputs.stages",
                ));
            }
            validate_execution_identity(
                &stage.execution,
                if shared_base {
                    submitted.outputs.len()
                } else {
                    1
                },
            )?;
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
    let detail_stages = terminal_result
        .get("generationSet")
        .and_then(|set| set.get("mode"))
        .and_then(Value::as_str)
        == Some("image_detail")
        && terminal_result
            .get("assetWrites")
            .and_then(Value::as_array)
            .and_then(|writes| writes.first())
            .and_then(|fact| fact.pointer("/rawAdapterSettings/detail/tiles"))
            .and_then(Value::as_u64)
            .is_some_and(|tiles| tiles > 0 && tiles as usize == submitted.stages.len())
        && submitted
            .stages
            .iter()
            .enumerate()
            .all(|(index, stage)| stage.stage == format!("detail_tile_{index}"));
    if !submitted.stages.is_empty()
        && (submitted.outputs.len() != 1 || !(voice_stages || seedvr2_stages || detail_stages))
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
        if !output.stages.is_empty() && output.artifact_ids.is_empty() {
            return Err(ReplayUnavailable::new(
                "output_stages_changed",
                "outputs.stages",
            ));
        }
        let writes = terminal_result
            .get("assetWrites")
            .or_else(|| terminal_result.get("datasetWrites"))
            .and_then(Value::as_array);
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
            let inline_stages = matching
                .iter()
                .filter_map(|fact| fact.get("workerExecutionStages"))
                .collect::<Vec<_>>();
            let fact_execution = matching
                .first()
                .and_then(|fact| fact.get("workerExecutionReport"));
            // A shared provider call may be hoisted from every output to the
            // receipt-level report. A fact that names a report must still name
            // that exact effective report for its output.
            let effective_execution = output
                .execution
                .as_ref()
                .or_else(|| fact_execution.map(|_| &submitted.execution));
            if output.stages.is_empty()
                && matching.len() == 1
                && fact_execution != effective_execution
            {
                return Err(ReplayUnavailable::new(
                    "output_execution_changed",
                    "outputs.execution",
                ));
            }
            let inline_output_stages = matches!(
                output.stages.first().map(|stage| stage.stage.as_str()),
                Some("identity" | "angle" | "pose" | "base")
            );
            if inline_output_stages {
                let has_child = output.stages.len() == 3;
                let expected_stages = serde_json::to_value(&output.stages[..2]).map_err(|_| {
                    ReplayUnavailable::new("invalid_output_stages", "outputs.stages")
                })?;
                if matching.len() != if has_child { 2 } else { 1 }
                    || inline_stages != [&expected_stages]
                    || (has_child
                        && (child_reports != [&output.stages[2].execution]
                            || matching[1].get("sourceAssetId") != matching[0].get("assetId")
                            || matching[1].get("workerExecutionReport") != fact_execution))
                    || (!has_child && !child_reports.is_empty())
                    || fact_execution.is_none()
                    || fact_execution != effective_execution
                {
                    return Err(ReplayUnavailable::new(
                        "output_stages_changed",
                        "outputs.stages",
                    ));
                }
            } else if !inline_stages.is_empty()
                || (child_reports.is_empty() != output.stages.is_empty())
            {
                return Err(ReplayUnavailable::new(
                    "output_stages_changed",
                    "outputs.stages",
                ));
            }
            if !inline_output_stages && !child_reports.is_empty() {
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
        if writes.is_none() && !output.stages.is_empty() {
            return Err(ReplayUnavailable::new(
                "output_stages_changed",
                "outputs.stages",
            ));
        }
    }
    if terminal_result
        .get("expectedCount")
        .and_then(Value::as_u64)
        .is_some_and(|count| {
            let assets = submitted
                .outputs
                .iter()
                .map(|output| output.artifact_ids.len())
                .sum();
            count as usize
                != if terminal_result.contains_key("assetWrites") {
                    assets
                } else {
                    submitted.outputs.len()
                }
        })
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

/// A completed identity must match the server-frozen provider for the selected
/// product model, slot, backend, and operation. Legacy routes without this
/// binding cannot authorize replay.
pub fn require_provider_route_binding(
    receipt: &TrustedExecutionReceipt,
    route: &crate::model_artifacts::portable_binding::PortableRouteBinding,
) -> Result<(), ReplayUnavailable> {
    if route.backend != receipt.accepted_backend || route.models.is_empty() {
        return Err(ReplayUnavailable::new("provider_route_changed", "source"));
    }
    let expected = route
        .expected_provider
        .as_ref()
        .ok_or_else(|| ReplayUnavailable::new("provider_route_unavailable", "source"))?;
    route
        .validate()
        .map_err(|_| ReplayUnavailable::new("provider_route_changed", "source"))?;
    if expected.operation != receipt.job_type {
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
        if identity.backend != route.backend || identity.provider_id != expected.provider_id {
            return Err(ReplayUnavailable::new("provider_route_changed", "source"));
        }
    }
    Ok(())
}

/// Verify that a trusted receipt's `low_res_start` provenance and quality-affecting rung match the
/// accepted payload (`OPT-6`, `ADM-3`, `FEAT-1`, `M2`, `E6`).
pub fn require_low_res_start_replay_binding(
    receipt: &TrustedExecutionReceipt,
    payload: &Map<String, Value>,
) -> Result<(), ReplayUnavailable> {
    use crate::resolved_options::{RouteField, RouteOptionFacts};
    let facts = RouteOptionFacts::from_payload_syntax(payload);
    let requested_low_res = match &facts.low_res_start {
        RouteField::Absent | RouteField::Null => None,
        RouteField::Valid(cfg) if cfg.disabled => None,
        RouteField::Valid(cfg) => Some(cfg),
        RouteField::Invalid => {
            return Err(ReplayUnavailable::new(
                "invalid_low_res_recipe",
                "advanced.lowResStart",
            ));
        }
    };
    let Some(requested) = requested_low_res else {
        if receipt.low_res_start.is_some() {
            return Err(ReplayUnavailable::new(
                "low_res_recipe_changed",
                "advanced.lowResStart",
            ));
        }
        return Ok(());
    };
    let version = requested.version.ok_or_else(|| {
        ReplayUnavailable::new(
            "missing_low_res_recipe_version",
            "advanced.lowResStart.version",
        )
    })?;
    if version != LOW_RES_START_RECIPE_VERSION {
        return Err(ReplayUnavailable::new(
            "unsupported_low_res_recipe_version",
            "advanced.lowResStart.version",
        ));
    }
    if payload
        .get("modelManifestEntry")
        .and_then(|entry| entry.get("features"))
        .and_then(|features| features.get("low_res_start"))
        == Some(&Value::Bool(false))
    {
        return Err(ReplayUnavailable::new(
            "low_res_start_unavailable",
            "advanced.lowResStart",
        ));
    }
    let Some(recorded) = &receipt.low_res_start else {
        return Err(ReplayUnavailable::new(
            "low_res_receipt_missing",
            "executionReceipt.lowResStart",
        ));
    };
    recorded.validate()?;
    if Some(recorded.version) != requested.version
        || Some(recorded.k) != requested.k
        || requested
            .scale
            .is_none_or(|s| (recorded.scale - s).abs() > 1e-9)
    {
        return Err(ReplayUnavailable::new(
            "low_res_recipe_changed",
            "executionReceipt.lowResStart",
        ));
    }
    if let (Some(w), Some(h)) = (
        payload
            .get("width")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok()),
        payload
            .get("height")
            .and_then(Value::as_u64)
            .and_then(|v| u32::try_from(v).ok()),
    ) {
        if recorded.full_w != w || recorded.full_h != h {
            return Err(ReplayUnavailable::new(
                "low_res_dimensions_changed",
                "executionReceipt.lowResStart.dimensions",
            ));
        }
    }
    if let RouteField::Valid(strategy) = &facts.memory_strategy {
        if let Some(requested_rung) = strategy.rung.as_deref() {
            if recorded.selected_rung.as_deref() != Some(requested_rung) {
                return Err(ReplayUnavailable::new(
                    "quality_rung_changed",
                    "execution.quality_policy",
                ));
            }
        }
    }
    for output in &receipt.outputs {
        let execution = output.execution.as_ref().unwrap_or(&receipt.execution);
        if let Some(identity_rung) = execution
            .pointer("/identity/quality_policy/value/rung")
            .and_then(Value::as_str)
        {
            if let Some(recorded_rung) = recorded.selected_rung.as_deref() {
                if recorded_rung != identity_rung {
                    return Err(ReplayUnavailable::new(
                        "quality_rung_changed",
                        "execution.quality_policy",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Capabilities of the executing host checked before or during an Exact Replay attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayHostCapabilities {
    pub backend: String,
    pub supports_low_res_start: bool,
    pub available_estimators: BTreeSet<String>,
    pub available_quality_rungs: BTreeSet<String>,
}

/// Verify that the executing host supports every pinned recipe, estimator, backend, and
/// quality-affecting rung required by the trusted execution receipt (`OPT-6`, `ADM-3`, `FEAT-1`, `E6`).
/// Never silently downgrades or substitutes an unavailable quality-affecting rung or estimator.
pub fn require_exact_replay_host_support(
    receipt: &TrustedExecutionReceipt,
    payload: &Map<String, Value>,
    host: &ReplayHostCapabilities,
) -> Result<(), ReplayUnavailable> {
    if receipt.accepted_backend != host.backend {
        return Err(ReplayUnavailable::new("backend_changed", "backend"));
    }
    require_low_res_start_replay_binding(receipt, payload)?;
    for output in &receipt.outputs {
        let execution = output.execution.as_ref().unwrap_or(&receipt.execution);
        if let Some(rung) = execution
            .pointer("/identity/quality_policy/value/rung")
            .and_then(Value::as_str)
        {
            if !host.available_quality_rungs.contains(rung) {
                return Err(ReplayUnavailable::new(
                    "quality_rung_unavailable",
                    "execution.quality_policy",
                ));
            }
        }
    }
    if let Some(low_res) = &receipt.low_res_start {
        if !host.supports_low_res_start {
            return Err(ReplayUnavailable::new(
                "low_res_start_unavailable",
                "advanced.lowResStart",
            ));
        }
        if !host.available_estimators.contains(&low_res.estimator) {
            return Err(ReplayUnavailable::new(
                "estimator_unavailable",
                "lowResStart.estimator",
            ));
        }
        if let Some(rung) = low_res.selected_rung.as_deref() {
            if !host.available_quality_rungs.contains(rung) {
                return Err(ReplayUnavailable::new(
                    "quality_rung_unavailable",
                    "lowResStart.selectedRung",
                ));
            }
        }
    }
    Ok(())
}

/// Compare the expected and actual trusted execution receipts on exact replay completion,
/// returning a specific typed [`ReplayUnavailable`] refusal if the backend, quality-affecting rung,
/// estimator, or low-res recipe changed.
pub fn verify_completed_exact_replay(
    expected: &TrustedExecutionReceipt,
    actual: &TrustedExecutionReceipt,
) -> Result<(), ReplayUnavailable> {
    if expected.accepted_backend != actual.accepted_backend {
        return Err(ReplayUnavailable::new("backend_changed", "backend"));
    }
    if expected.accepted_device != actual.accepted_device {
        return Err(ReplayUnavailable::new("device_changed", "device"));
    }
    if expected.accepted_source_route != actual.accepted_source_route {
        return Err(ReplayUnavailable::new("source_route_changed", "source"));
    }
    if expected.inference_revision != actual.inference_revision {
        return Err(ReplayUnavailable::new(
            "runtime_revision_changed",
            "inferenceRevision",
        ));
    }
    match (&expected.low_res_start, &actual.low_res_start) {
        (Some(exp), Some(act)) => {
            if exp.version != act.version {
                return Err(ReplayUnavailable::new(
                    "unsupported_low_res_recipe_version",
                    "lowResStart.version",
                ));
            }
            if exp.estimator != act.estimator {
                return Err(ReplayUnavailable::new(
                    "estimator_changed",
                    "lowResStart.estimator",
                ));
            }
            if exp.selected_rung != act.selected_rung {
                return Err(ReplayUnavailable::new(
                    "quality_rung_changed",
                    "lowResStart.selectedRung",
                ));
            }
            if exp != act {
                return Err(ReplayUnavailable::new(
                    "low_res_recipe_changed",
                    "lowResStart",
                ));
            }
        }
        (Some(_), None) => {
            return Err(ReplayUnavailable::new(
                "low_res_receipt_missing",
                "executionReceipt.lowResStart",
            ));
        }
        (None, Some(_)) => {
            return Err(ReplayUnavailable::new(
                "low_res_recipe_changed",
                "executionReceipt.lowResStart",
            ));
        }
        (None, None) => {}
    }
    let exp_rung = expected
        .execution
        .pointer("/identity/quality_policy/value/rung");
    let act_rung = actual
        .execution
        .pointer("/identity/quality_policy/value/rung");
    if exp_rung != act_rung {
        return Err(ReplayUnavailable::new(
            "quality_rung_changed",
            "execution.quality_policy",
        ));
    }
    if expected.execution != actual.execution
        || expected.stages != actual.stages
        || expected.outputs.len() != actual.outputs.len()
        || expected
            .outputs
            .iter()
            .zip(&actual.outputs)
            .any(|(before, after)| {
                before.output_index != after.output_index
                    || before.execution != after.execution
                    || before.artifact_sha256 != after.artifact_sha256
            })
    {
        return Err(ReplayUnavailable::new(
            "completed_execution_changed",
            "executionReceipt",
        ));
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
    require_low_res_start_replay_binding(receipt, payload)?;
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
        let execution = output.execution.as_ref().unwrap_or(&receipt.execution);
        let randomness = execution
            .pointer("/identity/randomness")
            .ok_or_else(|| ReplayUnavailable::new("missing_randomness", "execution.randomness"))?;
        let draws = randomness
            .get("draws")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ReplayUnavailable::new("unsupported_draw_mapping", "execution.randomness")
            })?;
        let provider_index = if output.execution.is_some() { 0 } else { index };
        let expected = base
            .map(|seed| seed.wrapping_add(index as i64))
            .or_else(|| seeds?.get(index)?.as_i64())
            .and_then(|seed| u64::try_from(seed).ok());
        // The qualified Candle SDXL route consumes the image seed twice: once
        // for initial noise and once for the pipeline's own ordered stream.
        // Its provider declaration only attests this single-output sequence.
        if execution
            .pointer("/identity/provider_id")
            .and_then(Value::as_str)
            == Some("sdxl")
            && execution
                .pointer("/identity/backend")
                .and_then(Value::as_str)
                == Some("candle")
        {
            if count != 1
                || randomness.get("scheme").and_then(Value::as_str)
                    != Some("candle-ordered-provider-streams-v1:rand-0.9-StdRng-rand_distr-StandardNormal-f32")
                || draws.len() != 2
            {
                return Err(ReplayUnavailable::new("unsupported_draw_mapping", "execution.randomness"));
            }
            if expected.is_none()
                || [
                    ("image_initial_noise", &draws[0]),
                    ("sdxl/pipeline", &draws[1]),
                ]
                .iter()
                .any(|(purpose, draw)| {
                    draw.get("output_index").and_then(Value::as_u64) != Some(0)
                        || draw.get("purpose").and_then(Value::as_str) != Some(*purpose)
                        || draw.get("seed").and_then(Value::as_u64) != expected
                        || !draw.get("noise_sha256").is_some_and(Value::is_null)
                })
            {
                return Err(ReplayUnavailable::new(
                    "seed_draw_binding_changed",
                    "execution.randomness",
                ));
            }
            continue;
        }
        let unit_count = if output.execution.is_some() { 1 } else { count };
        if let Some(low_res) = &receipt.low_res_start {
            if draws.len() == unit_count * 2 {
                let initial_draw = &draws[provider_index * 2];
                let transition_draw = &draws[provider_index * 2 + 1];
                let expected_transition = expected.map(|seed| {
                    low_res_start_transition_seed(
                        seed,
                        provider_index as u32,
                        low_res.target_pass,
                        0,
                    )
                });
                if initial_draw.get("output_index").and_then(Value::as_u64)
                    != Some(provider_index as u64)
                    || initial_draw.get("seed").and_then(Value::as_u64) != expected
                    || transition_draw.get("output_index").and_then(Value::as_u64)
                        != Some(provider_index as u64)
                    || transition_draw.get("seed").and_then(Value::as_u64) != expected_transition
                    || expected.is_none()
                {
                    return Err(ReplayUnavailable::new(
                        "seed_draw_binding_changed",
                        "execution.randomness",
                    ));
                }
                continue;
            }
        }
        if draws.len() != unit_count {
            return Err(ReplayUnavailable::new(
                "unsupported_draw_mapping",
                "execution.randomness",
            ));
        }
        let draw = &draws[provider_index];
        if draw.get("output_index").and_then(Value::as_u64) != Some(provider_index as u64)
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
        let variant_result = json!({"expectedCount":3,"assetWrites":[
            {"assetId":"a","index":0},{"assetId":"b","index":1},{"assetId":"upscaled-a","index":0}
        ]});
        validate_worker_receipt(&variants, variant_result.as_object().unwrap()).unwrap();
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
    fn inline_children_keep_the_shared_batch_base_report() {
        let mut receipt = worker_receipt();
        let base = receipt.execution.clone();
        let child =
            serde_json::to_value(gen_core::generator::ExecutionIdentity::provider_default())
                .unwrap();
        receipt.execution = child.clone();
        let mut writes = Vec::new();
        for (index, output) in receipt.outputs.iter_mut().enumerate() {
            let base_id = format!("base-{index}");
            let child_id = format!("child-{index}");
            output.artifact_ids = vec![base_id.clone(), child_id.clone()];
            output.artifact_sha256 = vec!["a".repeat(64), "b".repeat(64)];
            output.stages = vec![
                ExecutionStageBinding {
                    stage: "base_render".into(),
                    execution: base.clone(),
                },
                ExecutionStageBinding {
                    stage: "child_transform".into(),
                    execution: child.clone(),
                },
            ];
            writes.push(json!({"assetId":base_id,"index":index,"workerExecutionReport":base}));
            writes.push(
                json!({"assetId":child_id,"index":index,"sourceAssetId":base_id,
                "workerExecutionReport":base,"workerChildExecutionReport":child}),
            );
        }
        let result = json!({"expectedCount":4,"assetWrites":writes});
        validate_worker_receipt(&receipt, result.as_object().unwrap()).unwrap();

        receipt.outputs[1].stages[0].execution = child;
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_identity"
        );
    }

    #[test]
    fn hoisted_single_image_report_matches_its_terminal_fact() {
        let mut receipt = worker_receipt();
        receipt.outputs.truncate(1);
        receipt.execution["identity"]["output_units"] = json!(1);
        receipt.execution["identity"]["randomness"]["draws"]
            .as_array_mut()
            .unwrap()
            .truncate(1);
        let result = json!({"expectedCount":1,"assetWrites":[{
            "assetId":"a","index":0,"workerExecutionReport":receipt.execution
        }]});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        let mut mismatched = result.clone();
        mismatched["assetWrites"][0]["workerExecutionReport"]["identity"]["provider_id"] =
            json!("other_provider");
        assert_eq!(
            validate_worker_receipt(&receipt, mismatched.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_execution_changed"
        );
    }

    #[test]
    fn shared_multi_output_report_refuses_one_mismatched_fact() {
        let receipt = worker_receipt();
        let mut result = json!({"expectedCount":2,"assetWrites":[
            {"assetId":"a","index":0,"workerExecutionReport":receipt.execution},
            {"assetId":"b","index":1,"workerExecutionReport":receipt.execution}
        ]});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        result["assetWrites"][1]["workerExecutionReport"]["identity"]["provider_id"] =
            json!("other_provider");
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_execution_changed"
        );
    }

    #[test]
    fn dataset_derivatives_require_each_ordered_output_without_gallery_assets() {
        let mut receipt = worker_receipt();
        receipt.execution = json!({"state":"unavailable","reason":{
            "code":"real_esrgan_onnx_execution_unavailable","detail":"fixture",
            "missing_facts":["onnx_execution_recipe"]
        }});
        receipt.outputs[0].artifact_ids = vec!["dataset:d:a".into()];
        receipt.outputs[1].artifact_ids = vec!["dataset:d:b".into()];
        let result = json!({"expectedCount":2,"datasetWrites":[
            {"assetId":"dataset:d:a","index":0,"mediaPath":"training/a.png"},
            {"assetId":"dataset:d:b","index":1,"mediaPath":"training/b.png"}
        ]});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        let mut swapped = result.clone();
        swapped["datasetWrites"][0]["index"] = json!(1);
        swapped["datasetWrites"][1]["index"] = json!(0);
        assert_eq!(
            validate_worker_receipt(&receipt, swapped.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_artifacts_changed"
        );
        let mut missing = result.clone();
        missing["datasetWrites"].as_array_mut().unwrap().pop();
        assert_eq!(
            validate_worker_receipt(&receipt, missing.as_object().unwrap())
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
    fn instantid_output_stages_bind_each_final_asset_and_order() {
        let unavailable = json!({"state":"unavailable","reason":{
            "code":"composite_execution_unavailable","detail":"fixture",
            "missing_facts":["composite_determinism"]
        }});
        let stage_report = json!({"state":"unavailable","reason":{
            "code":"provider_report_unavailable","detail":"fixture",
            "missing_facts":["provider_resolved_execution"]
        }});
        let mut receipt = worker_receipt();
        receipt.execution = unavailable.clone();
        receipt.outputs[0].execution = Some(unavailable.clone());
        receipt.outputs[1].execution = Some(stage_report.clone());
        receipt.outputs[0].stages = vec![
            ExecutionStageBinding {
                stage: "identity".into(),
                execution: stage_report.clone(),
            },
            ExecutionStageBinding {
                stage: "restore_face".into(),
                execution: stage_report.clone(),
            },
        ];
        let result = json!({"expectedCount":2,"assetWrites":[
            {"assetId":"a","index":0,"workerExecutionReport":unavailable,
             "workerExecutionStages":receipt.outputs[0].stages},
            {"assetId":"b","index":1,"workerExecutionReport":receipt.outputs[1].execution}
        ]});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        let mut wrong_asset = result.clone();
        wrong_asset["assetWrites"][1]["workerExecutionStages"] =
            result["assetWrites"][0]["workerExecutionStages"].clone();
        assert_eq!(
            validate_worker_receipt(&receipt, wrong_asset.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_stages_changed"
        );
        let mut missing = result.clone();
        missing["assetWrites"][0]
            .as_object_mut()
            .unwrap()
            .remove("workerExecutionStages");
        assert_eq!(
            validate_worker_receipt(&receipt, missing.as_object().unwrap())
                .unwrap_err()
                .reason,
            "output_stages_changed"
        );
        receipt.outputs[0].stages.swap(0, 1);
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_output_stages"
        );
    }

    #[test]
    fn hires_base_and_refine_bind_one_final_asset_in_order() {
        let report = json!({"state":"unavailable","reason":{
            "code":"provider_report_unavailable","detail":"fixture",
            "missing_facts":["provider_resolved_execution"]
        }});
        let mut receipt = worker_receipt();
        receipt.execution = report.clone();
        receipt.outputs.truncate(1);
        receipt.outputs[0].execution = Some(report.clone());
        receipt.outputs[0].stages = ["base", "refine"]
            .into_iter()
            .map(|stage| ExecutionStageBinding {
                stage: stage.into(),
                execution: report.clone(),
            })
            .collect();
        let result = json!({"expectedCount":1,"assetWrites":[{
            "assetId":"a","index":0,"workerExecutionReport":report,
            "workerExecutionStages":receipt.outputs[0].stages
        }]});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        receipt.outputs[0].stages.swap(0, 1);
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_output_stages"
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
    fn detail_receipt_requires_every_ordered_tile_before_final_asset() {
        let unavailable = json!({"state":"unavailable","reason":{
            "code":"provider_report_unavailable","detail":"fixture","missing_facts":["provider_resolved_execution"]
        }});
        let stages = (0..4)
            .map(|index| ExecutionStageBinding {
                stage: format!("detail_tile_{index}"),
                execution: unavailable.clone(),
            })
            .collect::<Vec<_>>();
        let mut receipt = worker_receipt();
        receipt.execution = unavailable;
        receipt.outputs.truncate(1);
        receipt.stages = stages.clone();
        let result = json!({"expectedCount":1,"generationSet":{"mode":"image_detail"},
            "assetWrites":[{"assetId":"a","rawAdapterSettings":{"detail":{"tiles":4}}}],
            "workerExecutionStages":stages});
        assert!(validate_worker_receipt(&receipt, result.as_object().unwrap()).is_ok());
        receipt.stages.swap(0, 1);
        assert_eq!(
            validate_worker_receipt(&receipt, result.as_object().unwrap())
                .unwrap_err()
                .reason,
            "invalid_execution_stages"
        );
        receipt.stages.swap(0, 1);
        let mut missing_tile = result.clone();
        missing_tile["assetWrites"][0]["rawAdapterSettings"]["detail"]["tiles"] = json!(3);
        assert_eq!(
            validate_worker_receipt(&receipt, missing_tile.as_object().unwrap())
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
            low_res_start: None,
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
    fn frozen_provider_must_match_product_slot_operation_and_backend() {
        use crate::model_artifacts::portable_binding::{
            portable_closure_sha256, PortableArtifactFile, PortableArtifactMember,
            PortableExpectedProvider, PortableModelBinding, PortableRouteBinding,
            PortableSourceIdentity,
        };
        use crate::model_artifacts::ArtifactMemberRole;
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
            low_res_start: None,
        };
        let files = vec![PortableArtifactFile {
            relative_path: "model.safetensors".into(),
            size_bytes: 1,
            sha256: format!("sha256:{}", "e".repeat(64)),
        }];
        let primary = PortableArtifactMember {
            role: ArtifactMemberRole::Primary,
            component_id: None,
            source: PortableSourceIdentity::Digest {
                sha256: portable_closure_sha256(&files).unwrap(),
            },
            tier: None,
            files,
        };
        let model = PortableModelBinding::new(
            "modelManifestEntry",
            "instantid_realvisxl",
            "candle",
            "default",
            None,
            format!("sha256:{}", "e".repeat(64)),
            vec![primary],
        )
        .unwrap();
        let mut route = PortableRouteBinding::new("candle", "windows", vec![model]).unwrap();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_unavailable"
        );
        route = route
            .with_expected_provider(PortableExpectedProvider {
                model_slot: "modelManifestEntry".into(),
                model_id: "instantid_realvisxl".into(),
                operation: "image_generate".into(),
                provider_id: "instantid".into(),
            })
            .unwrap();
        assert!(require_provider_route_binding(&receipt, &route).is_ok());
        route.expected_provider.as_mut().unwrap().provider_id = "another_instantid".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
        route.expected_provider.as_mut().unwrap().provider_id = "instantid".into();
        route.models[0].model_id = "another_instantid".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
        route.models[0].model_id = "instantid_realvisxl".into();
        route.expected_provider.as_mut().unwrap().model_slot = "modelManifestEntries/1".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
        route.expected_provider.as_mut().unwrap().model_slot = "modelManifestEntry".into();
        route.expected_provider.as_mut().unwrap().operation = "image_edit".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
        route.expected_provider.as_mut().unwrap().operation = "image_generate".into();
        route.backend = "mlx".into();
        assert_eq!(
            require_provider_route_binding(&receipt, &route)
                .unwrap_err()
                .reason,
            "provider_route_changed"
        );
        route.backend = "candle".into();
        let auxiliary = route.models[0].clone();
        let auxiliary = PortableModelBinding {
            slot: "zzAux".into(),
            model_id: "instantid".into(),
            ..auxiliary
        };
        route.models.push(auxiliary);
        route.expected_provider.as_mut().unwrap().provider_id = "spoofed_provider".into();
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
            low_res_start: None,
        };
        receipt.execution["identity"]["randomness"] = json!({
            "kind":"draws", "scheme":"cpu_std_rng", "draws":[
                {"output_index":0,"purpose":"initial_noise","seed":41,"noise_sha256":null},
                {"output_index":1,"purpose":"initial_noise","seed":42,"noise_sha256":null}
            ]
        });
        let fixed = json!({"seed":41,"count":2});
        assert!(require_image_seed_binding(&receipt, fixed.as_object().unwrap()).is_ok());
        receipt.execution["identity"]["randomness"]["draws"][0]["noise_sha256"] =
            json!("a".repeat(64));
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

    #[test]
    fn exact_sdxl_replay_requires_both_ordered_seed_draws() {
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
            low_res_start: None,
        };
        receipt.execution["identity"]["provider_id"] = json!("sdxl");
        receipt.execution["identity"]["randomness"] = json!({
            "kind":"draws",
            "scheme":"candle-ordered-provider-streams-v1:rand-0.9-StdRng-rand_distr-StandardNormal-f32",
            "draws":[
                {"output_index":0,"purpose":"image_initial_noise","seed":16402,"noise_sha256":null},
                {"output_index":0,"purpose":"sdxl/pipeline","seed":16402,"noise_sha256":null}
            ]
        });
        receipt.outputs.truncate(1);
        let payload = json!({"seed":16402,"count":1});
        assert!(require_image_seed_binding(&receipt, payload.as_object().unwrap()).is_ok());
        receipt.execution["identity"]["randomness"]["draws"][1]["seed"] = json!(16403);
        assert_eq!(
            require_image_seed_binding(&receipt, payload.as_object().unwrap())
                .unwrap_err()
                .reason,
            "seed_draw_binding_changed"
        );
        receipt.execution["identity"]["randomness"]["draws"][1]["seed"] = json!(16402);
        receipt.execution["identity"]["randomness"]["draws"][1]["purpose"] = json!("other");
        assert_eq!(
            require_image_seed_binding(&receipt, payload.as_object().unwrap())
                .unwrap_err()
                .reason,
            "seed_draw_binding_changed"
        );
        receipt.execution["identity"]["randomness"]["draws"][1]["purpose"] = json!("sdxl/pipeline");
        receipt.execution["identity"]["randomness"]["draws"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert_eq!(
            require_image_seed_binding(&receipt, payload.as_object().unwrap())
                .unwrap_err()
                .reason,
            "unsupported_draw_mapping"
        );
    }

    #[test]
    fn accepted_binding_digest_and_execution_identity_bind_pag() {
        let base_payload = json!({
            "model": "flux2_dev",
            "prompt": "a lighthouse",
            "seed": 41,
            "count": 1,
            "advanced": { "steps": 28 }
        });
        let pag_payload = json!({
            "model": "flux2_dev",
            "prompt": "a lighthouse",
            "seed": 41,
            "count": 1,
            "advanced": {
                "steps": 28,
                "pag": { "scale": 1.5, "layers": [2, 5], "start": 0.05, "stop": 0.85 }
            }
        });
        let altered_pag_payload = json!({
            "model": "flux2_dev",
            "prompt": "a lighthouse",
            "seed": 41,
            "count": 1,
            "advanced": {
                "steps": 28,
                "pag": { "scale": 2.0, "layers": [2, 5], "start": 0.05, "stop": 0.85 }
            }
        });
        let base_digest = json_sha256(&base_payload).unwrap();
        let pag_digest = json_sha256(&pag_payload).unwrap();
        let altered_digest = json_sha256(&altered_pag_payload).unwrap();
        assert_ne!(base_digest, pag_digest);
        assert_ne!(pag_digest, altered_digest);

        let mut receipt = worker_receipt();
        receipt.outputs.truncate(1);
        receipt.execution["identity"]["output_units"] = json!(1);
        receipt.execution["identity"]["randomness"]["draws"]
            .as_array_mut()
            .unwrap()
            .truncate(1);
        receipt.execution["identity"]["sampling"]["value"]["pag"] = json!({
            "scale": 1.5,
            "adaptive_scale": 0.0,
            "layers": [2, 5],
            "start": 0.05,
            "stop": 0.85,
            "active_steps": 18,
            "total_steps": 28
        });
        let result = json!({
            "expectedCount": 1,
            "assetWrites": [{
                "assetId": "a",
                "index": 0,
                "workerExecutionReport": receipt.execution
            }]
        });
        validate_worker_receipt(&receipt, result.as_object().unwrap())
            .expect("execution identity with PAG sampling receipt must validate");
    }

    fn low_res_receipt_fixture() -> LowResStartReceipt {
        LowResStartReceipt {
            version: LOW_RES_START_RECIPE_VERSION,
            k: 3,
            executed_steps: 9,
            scale: 0.5,
            low_w: 512,
            low_h: 512,
            full_w: 1024,
            full_h: 1024,
            target_pass: 0,
            estimator: LOW_RES_START_ESTIMATOR_ENDPOINT_V1.into(),
            schedule_policy: LOW_RES_START_SCHEDULE_POLICY_V1.into(),
            transition_arithmetic: LOW_RES_START_TRANSITION_ARITHMETIC_V1.into(),
            transition_seed_scheme: LOW_RES_START_TRANSITION_SEED_SCHEME_V1.into(),
            selected_rung: Some("resident".into()),
        }
    }

    #[test]
    fn low_res_start_receipt_validates_and_binds_exact_replay_draws_and_host_support() {
        let mut worker = worker_receipt();
        worker.outputs.truncate(1);
        worker.execution["identity"]["output_units"] = json!(1);
        let transition_seed = low_res_start_transition_seed(41, 0, 0, 0);
        assert_ne!(transition_seed, 41);
        assert_eq!(transition_seed, low_res_start_transition_seed(41, 0, 0, 0));
        worker.execution["identity"]["randomness"]["draws"] = json!([
            {"output_index": 0, "purpose": "initial_noise", "seed": 41, "noise_sha256": null},
            {"output_index": 0, "purpose": "low_res_start_transition_v1", "seed": transition_seed, "noise_sha256": null}
        ]);
        worker.low_res_start = Some(low_res_receipt_fixture());
        let result = json!({
            "expectedCount": 1,
            "assetWrites": [{
                "assetId": "a",
                "index": 0,
                "workerExecutionReport": worker.execution
            }]
        });
        validate_worker_receipt(&worker, result.as_object().unwrap())
            .expect("valid low-res start receipt passes validation");

        let trusted = TrustedExecutionReceipt {
            version: EXECUTION_RECEIPT_VERSION,
            job_id: "job-lowres".into(),
            job_type: "image_generate".into(),
            worker_id: "worker-1".into(),
            attempt_id: "a".repeat(64),
            accepted_backend: "mlx".into(),
            accepted_device: "mlx".into(),
            accepted_source_route: None,
            inference_revision: "b".repeat(40),
            accepted_binding_sha256: "c".repeat(64),
            accepted_result_sha256: "d".repeat(64),
            execution: worker.execution.clone(),
            outputs: worker.outputs.clone(),
            stages: Vec::new(),
            low_res_start: worker.low_res_start.clone(),
        };
        let payload = json!({
            "model": "z_image_turbo",
            "seed": 41,
            "count": 1,
            "width": 1024,
            "height": 1024,
            "modelManifestEntry": {
                "features": { "low_res_start": true }
            },
            "advanced": {
                "steps": 9,
                "lowResStart": { "version": 1, "k": 3, "scale": 0.5 },
                "memoryStrategy": { "rung": "resident" }
            }
        });
        require_image_seed_binding(&trusted, payload.as_object().unwrap())
            .expect("low-res exact replay seed + recipe binding succeeds");

        let host = ReplayHostCapabilities {
            backend: "mlx".into(),
            supports_low_res_start: true,
            available_estimators: BTreeSet::from([LOW_RES_START_ESTIMATOR_ENDPOINT_V1.into()]),
            available_quality_rungs: BTreeSet::from(["resident".into()]),
        };
        require_exact_replay_host_support(&trusted, payload.as_object().unwrap(), &host)
            .expect("matching host capabilities allow exact replay");
        verify_completed_exact_replay(&trusted, &trusted)
            .expect("identical low-res execution receipt passes completion check");

        // Unavailable quality-affecting rung refuses exact replay with typed ReplayUnavailable.
        let mut downgraded_host = host.clone();
        downgraded_host.available_quality_rungs = BTreeSet::from(["tiled_vae".into()]);
        assert_eq!(
            require_exact_replay_host_support(
                &trusted,
                payload.as_object().unwrap(),
                &downgraded_host
            )
            .unwrap_err()
            .reason,
            "quality_rung_unavailable"
        );

        // Unavailable estimator refuses exact replay with typed ReplayUnavailable.
        let mut missing_estimator_host = host.clone();
        missing_estimator_host.available_estimators =
            BTreeSet::from([LOW_RES_START_ESTIMATOR_PREVIOUS_V1.into()]);
        assert_eq!(
            require_exact_replay_host_support(
                &trusted,
                payload.as_object().unwrap(),
                &missing_estimator_host
            )
            .unwrap_err()
            .reason,
            "estimator_unavailable"
        );

        // Unavailable low_res_start feature on host refuses exact replay.
        let mut no_feature_host = host.clone();
        no_feature_host.supports_low_res_start = false;
        assert_eq!(
            require_exact_replay_host_support(
                &trusted,
                payload.as_object().unwrap(),
                &no_feature_host
            )
            .unwrap_err()
            .reason,
            "low_res_start_unavailable"
        );

        // Mismatched recipe version refuses exact replay.
        let mut future_payload = payload.clone();
        future_payload["advanced"]["lowResStart"]["version"] = json!(2);
        assert_eq!(
            require_image_seed_binding(&trusted, future_payload.as_object().unwrap())
                .unwrap_err()
                .reason,
            "unsupported_low_res_recipe_version"
        );

        // Mismatched quality-affecting rung or estimator on completion refuses exact replay.
        let mut changed_rung = trusted.clone();
        changed_rung.low_res_start.as_mut().unwrap().selected_rung = Some("tiled_vae".into());
        assert_eq!(
            verify_completed_exact_replay(&trusted, &changed_rung)
                .unwrap_err()
                .reason,
            "quality_rung_changed"
        );
        let mut changed_estimator = trusted.clone();
        changed_estimator.low_res_start.as_mut().unwrap().estimator =
            LOW_RES_START_ESTIMATOR_PREVIOUS_V1.into();
        assert_eq!(
            verify_completed_exact_replay(&trusted, &changed_estimator)
                .unwrap_err()
                .reason,
            "estimator_changed"
        );
    }
}
