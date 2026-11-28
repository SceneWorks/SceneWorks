//! Per-job opt-in for the two lossy CUDA attention kernels (`advanced.lossyAttention`, SW-6132):
//! the INT8 QK kernel (SageAttention-style) and Sol-Attn (sparse, block-routed).
//!
//! The wire maps field for field onto inference's `runtime_cuda::attention::Int8AttentionConfig`
//! and `SolConfig`, with their defaults (the worker converts it and pins both the defaults and
//! the range rule against the linked types). Neither kernel is ever a default: the option is absent
//! unless a request asks for it, and absence means the exact (FlashAttention / budgeted) path.
//!
//! Two checks live here, shared by the API's request validation and the routing probe:
//!
//! - [`parse_wire_value`] (shape): unknown keys, bad enums and out-of-range values are refused.
//!   The ranges are inference's `SolConfig::validate` rule.
//! - [`requested_lossy_attention`] (model): the requested kernels must be served by the model's
//!   attention. The facts ride on the projected catalog entry as [`CAPABILITY_KEY`], derived by
//!   `sceneworks-core` from the checked-in projection of inference's `SHARED_ATTENTION_ROUTING`.
//!   An entry without them (an MLX-only model, a bespoke-attention model) refuses the option.
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::contracts::JsonObject;
use crate::request_validation::ValidationRefusal;

/// `advanced.lossyAttention`: the request option.
pub const LOSSY_ATTENTION_KEY: &str = "lossyAttention";
/// `rawAdapterSettings.attentionExecution`: the worker's per-job receipt of the attention paths
/// the engine actually took (INT8 / Sol-Attn calls, refusals, adaptations). Worker-owned.
pub const ATTENTION_EXECUTION_KEY: &str = "attentionExecution";
/// `model["lossyAttention"]`: which lossy kernels the model's attention can take.
pub const CAPABILITY_KEY: &str = "lossyAttention";

/// `Int8AttentionConfig::pv`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PvPrecisionWire {
    /// `PvPrecision::ValueDtype` (inference default): only `Q·Kᵀ` is lossy.
    #[default]
    ValueDtype,
    /// `PvPrecision::Int8`: INT8 V and U8 P (comfy-kitchen's `qk_int_sv_i8`).
    Int8,
}

/// `Int8AttentionConfig::k_centering`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KCenteringWire {
    Off,
    Always,
    /// `KCentering::Adaptive` (inference default).
    #[default]
    Adaptive,
}

/// `advanced.lossyAttention.int8`: `Int8AttentionConfig`. Every field defaults to inference's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Int8AttentionWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pv_precision: Option<PvPrecisionWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub k_centering: Option<KCenteringWire>,
}

impl Int8AttentionWire {
    pub fn pv_precision(&self) -> PvPrecisionWire {
        self.pv_precision.unwrap_or_default()
    }
    pub fn k_centering(&self) -> KCenteringWire {
        self.k_centering.unwrap_or_default()
    }
}

/// `advanced.lossyAttention.sol`: `SolConfig`. Exactly one of `topkRatio` (`SolSelection::TopK`)
/// and `tau` (`SolSelection::Tau`); the rest default to `SolConfig::top_k`'s values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SolAttentionWire {
    /// Fraction of the non-sink key blocks each query block routes, in `(0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topk_ratio: Option<f64>,
    /// comfy-kitchen's threshold selection: finite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tau: Option<f64>,
    /// `[start, end)` key blocks every query block attends exactly (default `[0, 0]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sink_blocks: Option<[u32; 2]>,
    /// `[start, end)` query blocks that attend every key block exactly (default `[0, 0]`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sink_q: Option<[u32; 2]>,
    /// Keep the pooled term of the unrouted blocks (default `true`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tail: Option<bool>,
    /// Query blocks per route + exact window; `0` (default) runs one window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_chunk_blocks: Option<u32>,
}

/// The validated block selection of a [`SolAttentionWire`] (`SolSelection`), narrowed to f32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SolSelectionWire {
    TopK(f32),
    Tau(f32),
}

impl SolAttentionWire {
    pub fn sink_blocks(&self) -> (u32, u32) {
        let [start, end] = self.sink_blocks.unwrap_or([0, 0]);
        (start, end)
    }
    pub fn sink_q(&self) -> (u32, u32) {
        let [start, end] = self.sink_q.unwrap_or([0, 0]);
        (start, end)
    }
    pub fn tail(&self) -> bool {
        self.tail.unwrap_or(true)
    }
    pub fn query_chunk_blocks(&self) -> u32 {
        self.query_chunk_blocks.unwrap_or(0)
    }

    /// The block selection, after [`Self::validate`]'s rule.
    pub fn selection(&self) -> Result<SolSelectionWire, ValidationRefusal> {
        let field = |name: &str| format!("advanced.{LOSSY_ATTENTION_KEY}.sol.{name}");
        match (self.topk_ratio, self.tau) {
            (Some(ratio), None) => {
                let narrowed = ratio as f32;
                // `SolConfig::validate`: `0 < r <= 1`, decided on the f32 the kernel receives.
                if ratio.is_finite() && narrowed > 0.0 && narrowed <= 1.0 && ratio > 0.0 {
                    Ok(SolSelectionWire::TopK(narrowed))
                } else {
                    Err(ValidationRefusal::new(
                        field("topkRatio"),
                        format!("{} must be in (0, 1] (got {ratio})", field("topkRatio")),
                    ))
                }
            }
            (None, Some(tau)) => {
                let narrowed = tau as f32;
                if tau.is_finite() && narrowed.is_finite() {
                    Ok(SolSelectionWire::Tau(narrowed))
                } else {
                    Err(ValidationRefusal::new(
                        field("tau"),
                        format!("{} must be a finite number (got {tau})", field("tau")),
                    ))
                }
            }
            (Some(_), Some(_)) => Err(ValidationRefusal::new(
                field("tau"),
                format!(
                    "{} and {} are exclusive: choose one block selection",
                    field("topkRatio"),
                    field("tau")
                ),
            )),
            (None, None) => Err(ValidationRefusal::new(
                field("topkRatio"),
                format!(
                    "advanced.{LOSSY_ATTENTION_KEY}.sol needs a block selection: {} in (0, 1] or {}",
                    field("topkRatio"),
                    field("tau")
                ),
            )),
        }
    }

    /// inference's `SolConfig::validate`, on the wire.
    pub fn validate(&self) -> Result<(), ValidationRefusal> {
        self.selection()?;
        // The kernels take block indices as signed 32-bit ints.
        let i32_max = i32::MAX as u32;
        for (name, range) in [("sinkBlocks", self.sink_blocks), ("sinkQ", self.sink_q)] {
            let Some([start, end]) = range else { continue };
            let field = format!("advanced.{LOSSY_ATTENTION_KEY}.sol.{name}");
            if end < start {
                return Err(ValidationRefusal::new(
                    field.clone(),
                    format!(
                        "{field} must be [start, end) with end >= start (got [{start}, {end}])"
                    ),
                ));
            }
            if end > i32_max {
                return Err(ValidationRefusal::new(
                    field.clone(),
                    format!("{field} endpoints must fit the kernels' i32 block index (got {end})"),
                ));
            }
        }
        Ok(())
    }
}

/// `advanced.lossyAttention`: the job's `LossyAttentionOptIns`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LossyAttentionWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub int8: Option<Int8AttentionWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sol: Option<SolAttentionWire>,
}

impl LossyAttentionWire {
    /// Whether any lossy kernel is requested (`LossyAttentionOptIns::any`).
    pub fn any(&self) -> bool {
        self.int8.is_some() || self.sol.is_some()
    }

    /// The requested kernels' stable names, in `int8`, `sol` order.
    pub fn requested(&self) -> Vec<LossyKernel> {
        let mut kernels = Vec::new();
        if self.int8.is_some() {
            kernels.push(LossyKernel::Int8);
        }
        if self.sol.is_some() {
            kernels.push(LossyKernel::Sol);
        }
        kernels
    }

    pub fn validate(&self) -> Result<(), ValidationRefusal> {
        if let Some(sol) = &self.sol {
            sol.validate()?;
        }
        Ok(())
    }
}

/// One lossy kernel, as the catalog capability and the refusals name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossyKernel {
    Int8,
    Sol,
}

impl LossyKernel {
    pub fn key(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Sol => "sol",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Int8 => "INT8 QK attention",
            Self::Sol => "Sol-Attn sparse attention",
        }
    }
}

/// Parse and range-check a raw `advanced.lossyAttention` value (shape validation).
pub fn parse_wire_value(value: &Value) -> Result<LossyAttentionWire, ValidationRefusal> {
    let field = format!("advanced.{LOSSY_ATTENTION_KEY}");
    if !value.is_object() {
        return Err(ValidationRefusal::new(
            field.clone(),
            format!("{field} must be an object"),
        ));
    }
    let wire: LossyAttentionWire =
        serde_path_to_error::deserialize(value.clone()).map_err(|error| {
            let path = error.path().to_string();
            let at = if path == "." || path.is_empty() {
                field.clone()
            } else {
                format!("{field}.{path}")
            };
            ValidationRefusal::new(at.clone(), format!("{at}: {}", error.inner()))
        })?;
    wire.validate()?;
    Ok(wire)
}

/// One kernel's serve fact on the catalog entry (`model.lossyAttention.int8` / `.sol`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct KernelServeWire {
    pub served: bool,
    /// Why the model's attention can never take the kernel (inference's `LossyServe::Never`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `model.lossyAttention`: which lossy kernels the model's attention can take. Absent on an entry
/// whose attention the shared CUDA helpers never reach (refuse).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LossyAttentionCapability {
    pub int8: KernelServeWire,
    pub sol: KernelServeWire,
}

impl LossyAttentionCapability {
    pub fn serve(&self, kernel: LossyKernel) -> &KernelServeWire {
        match kernel {
            LossyKernel::Int8 => &self.int8,
            LossyKernel::Sol => &self.sol,
        }
    }

    /// Whether any kernel is served (whether the studio offers the control at all).
    pub fn any_served(&self) -> bool {
        self.int8.served || self.sol.served
    }
}

/// One provider's row of inference's `SHARED_ATTENTION_ROUTING` (`SharedAttentionRouting`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "routing", rename_all = "snake_case", deny_unknown_fields)]
pub enum ProviderAttentionRouting {
    /// `SharedHelpers { int8, sol }`: the job switches reach the provider's attention; each
    /// kernel is `LossyServe::Served` or `Never(reason)`.
    SharedHelpers {
        int8: KernelServeWire,
        sol: KernelServeWire,
    },
    /// `Bespoke`: the provider runs its own attention, outside the shared helpers.
    Bespoke,
}

/// The checked-in projection of inference's `SHARED_ATTENTION_ROUTING`
/// (`config/engine-capabilities/attention/shared-attention-routing.candle.json`). The candle
/// worker serializes it from the linked table and its test fails on any drift, so this file is
/// the same facts, not a second table.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SharedAttentionFacts {
    /// Always `candle`: the MLX backend has no lossy attention kernels.
    pub backend: String,
    /// The linked symbol the file is serialized from.
    pub source: String,
    /// Every registered provider id, in id order.
    pub providers: std::collections::BTreeMap<String, ProviderAttentionRouting>,
}

impl SharedAttentionFacts {
    /// The capability of a model whose candle attention runs through `provider_ids` (every engine a
    /// request can select). `None` when there are none, or when any is unknown or bespoke: the
    /// option could not reach that engine's attention. A kernel is served only when every engine
    /// serves it; the first `Never` reason is kept.
    pub fn capability_for<'a>(
        &self,
        provider_ids: impl IntoIterator<Item = &'a str>,
    ) -> Option<LossyAttentionCapability> {
        let mut capability: Option<LossyAttentionCapability> = None;
        for id in provider_ids {
            let ProviderAttentionRouting::SharedHelpers { int8, sol } = self.providers.get(id)?
            else {
                return None;
            };
            let merged = match capability.take() {
                None => LossyAttentionCapability {
                    int8: int8.clone(),
                    sol: sol.clone(),
                },
                Some(previous) => {
                    let pick = |a: KernelServeWire, b: &KernelServeWire| {
                        if a.served {
                            b.clone()
                        } else {
                            a
                        }
                    };
                    LossyAttentionCapability {
                        int8: pick(previous.int8, int8),
                        sol: pick(previous.sol, sol),
                    }
                }
            };
            capability = Some(merged);
        }
        capability
    }
}

/// The capability a projected catalog entry advertises. A malformed block is refused rather than
/// read as "unsupported".
pub fn capability_from_entry(
    entry: &JsonObject,
) -> Result<Option<LossyAttentionCapability>, ValidationRefusal> {
    let Some(raw) = entry.get(CAPABILITY_KEY).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    serde_json::from_value(raw.clone())
        .map(Some)
        .map_err(|error| {
            ValidationRefusal::new(
                format!("advanced.{LOSSY_ATTENTION_KEY}"),
                format!("model lossyAttention capability is malformed: {error}"),
            )
        })
}

/// Why `model` cannot take `wire` with the advertised `capability` (`None`: admitted). Pure: the
/// API validator and the worker's MLX/unsupported-lane guards share the wording.
pub fn admission_refusal(
    model: &str,
    wire: &LossyAttentionWire,
    capability: Option<&LossyAttentionCapability>,
) -> Option<ValidationRefusal> {
    if !wire.any() {
        return None;
    }
    let field = format!("advanced.{LOSSY_ATTENTION_KEY}");
    let Some(capability) = capability else {
        return Some(ValidationRefusal::new(
            field.clone(),
            format!(
                "{model} does not support {field}: its attention does not run through the shared \
                 CUDA (candle) attention kernels"
            ),
        ));
    };
    for kernel in wire.requested() {
        let serve = capability.serve(kernel);
        if !serve.served {
            let why = serve
                .reason
                .as_deref()
                .unwrap_or("its attention cannot take this kernel");
            return Some(ValidationRefusal::new(
                format!("{field}.{}", kernel.key()),
                format!(
                    "{model} does not support {field}.{} ({}): {why}",
                    kernel.key(),
                    kernel.label()
                ),
            ));
        }
    }
    None
}

/// The MLX (Apple) backend has neither kernel: a typed refusal, never a silently exact render.
pub fn mlx_refusal(model: &str) -> ValidationRefusal {
    ValidationRefusal::new(
        format!("advanced.{LOSSY_ATTENTION_KEY}"),
        format!(
            "{model}: advanced.{LOSSY_ATTENTION_KEY} (INT8 / Sol-Attn) runs only on the CUDA \
             (candle) backend; the MLX backend has no lossy attention kernels"
        ),
    )
}

/// Model-aware validation: `Ok(None)` when absent, the parsed wire when the model serves every
/// requested kernel, a typed refusal otherwise.
pub fn requested_lossy_attention(
    model: &str,
    advanced: &JsonObject,
    entry: &JsonObject,
) -> Result<Option<LossyAttentionWire>, ValidationRefusal> {
    let Some(raw) = advanced
        .get(LOSSY_ATTENTION_KEY)
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let wire = parse_wire_value(raw)?;
    let capability = capability_from_entry(entry)?;
    if let Some(refusal) = admission_refusal(model, &wire, capability.as_ref()) {
        return Err(refusal);
    }
    Ok(Some(wire))
}

/// Request-validation kinds whose job names an ENGINE rather than a generation model (SW-6132):
/// their lossy-attention admission is engine-aware (`sceneworks_core::lossy_attention`), not the
/// model catalog capability.
pub fn is_standalone_validation_kind(kind: &str) -> bool {
    kind == "videoUpscale"
}

/// The parsed `advanced.lossyAttention` of a raw payload (`None` when absent, null or requesting no
/// kernel), for the standalone job types (upscale, restoration) that have no typed request.
pub fn payload_lossy_attention(
    payload: &JsonObject,
) -> Result<Option<LossyAttentionWire>, ValidationRefusal> {
    let Some(raw) = payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get(LOSSY_ATTENTION_KEY))
        .filter(|value| !value.is_null())
    else {
        return Ok(None);
    };
    let wire = parse_wire_value(raw)?;
    Ok(wire.any().then_some(wire))
}

/// Whether an authored payload carries a non-null `advanced.lossyAttention` that requests a kernel
/// (routing: such a job is CUDA-only).
pub fn payload_requests_lossy_attention(payload: &JsonObject) -> bool {
    payload
        .get("advanced")
        .and_then(Value::as_object)
        .and_then(|advanced| advanced.get(LOSSY_ATTENTION_KEY))
        .filter(|value| !value.is_null())
        .and_then(|value| serde_json::from_value::<LossyAttentionWire>(value.clone()).ok())
        .is_some_and(|wire| wire.any())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> Result<LossyAttentionWire, ValidationRefusal> {
        parse_wire_value(&value)
    }

    fn entry(capability: Value) -> JsonObject {
        json!({ "id": "m", CAPABILITY_KEY: capability })
            .as_object()
            .unwrap()
            .clone()
    }

    const BOTH: fn() -> Value = || json!({ "int8": { "served": true }, "sol": { "served": true } });

    #[test]
    fn empty_int8_takes_the_inference_defaults() {
        let wire = parse(json!({ "int8": {} })).unwrap();
        let int8 = wire.int8.unwrap();
        assert_eq!(int8.pv_precision(), PvPrecisionWire::ValueDtype);
        assert_eq!(int8.k_centering(), KCenteringWire::Adaptive);
        assert!(wire.sol.is_none());
        assert!(wire.any());
    }

    #[test]
    fn the_full_wire_round_trips_through_json() {
        let raw = json!({
            "int8": { "pvPrecision": "int8", "kCentering": "always" },
            "sol": { "topkRatio": 0.25, "sinkBlocks": [0, 2], "sinkQ": [1, 3], "tail": false,
                     "queryChunkBlocks": 16 }
        });
        let wire = parse(raw.clone()).unwrap();
        assert_eq!(serde_json::to_value(wire).unwrap(), raw);
        let sol = wire.sol.unwrap();
        assert_eq!(sol.selection().unwrap(), SolSelectionWire::TopK(0.25));
        assert_eq!(sol.sink_blocks(), (0, 2));
        assert_eq!(sol.sink_q(), (1, 3));
        assert!(!sol.tail());
        assert_eq!(sol.query_chunk_blocks(), 16);
        let defaults = parse(json!({ "sol": { "tau": -0.5 } }))
            .unwrap()
            .sol
            .unwrap();
        assert_eq!(defaults.selection().unwrap(), SolSelectionWire::Tau(-0.5));
        assert_eq!(
            (defaults.sink_blocks(), defaults.sink_q()),
            ((0, 0), (0, 0))
        );
        assert!(defaults.tail());
        assert_eq!(defaults.query_chunk_blocks(), 0);
    }

    #[test]
    fn ranges_follow_the_sol_config_rule() {
        for ratio in [1.0, 0.5, 1e-6] {
            assert!(
                parse(json!({ "sol": { "topkRatio": ratio } })).is_ok(),
                "{ratio}"
            );
        }
        for ratio in [0.0, -0.1, 1.0001, 2.0, 1e-50] {
            let error = parse(json!({ "sol": { "topkRatio": ratio } })).unwrap_err();
            assert_eq!(
                error.field, "advanced.lossyAttention.sol.topkRatio",
                "{ratio}"
            );
        }
        assert!(
            parse(json!({ "sol": { "tau": 1e300 } })).is_err(),
            "tau must fit f32"
        );
        assert!(parse(json!({ "sol": {} })).is_err(), "needs a selection");
        assert!(parse(json!({ "sol": { "topkRatio": 0.5, "tau": 1.0 } })).is_err());
        let backwards =
            parse(json!({ "sol": { "topkRatio": 0.5, "sinkBlocks": [3, 1] } })).unwrap_err();
        assert_eq!(backwards.field, "advanced.lossyAttention.sol.sinkBlocks");
        let wide = parse(json!({ "sol": { "topkRatio": 0.5, "sinkQ": [0, 3_000_000_000u32] } }))
            .unwrap_err();
        assert_eq!(wide.field, "advanced.lossyAttention.sol.sinkQ");
        assert!(parse(json!({ "sol": { "topkRatio": 0.5, "sinkBlocks": [0, -1] } })).is_err());
        assert!(parse(json!({ "sol": { "topkRatio": 0.5, "queryChunkBlocks": -2 } })).is_err());
    }

    #[test]
    fn shape_refuses_unknown_keys_and_enum_spellings() {
        for bad in [
            json!({ "int9": {} }),
            json!({ "int8": { "pv": "int8" } }),
            json!({ "int8": { "pvPrecision": "fp8" } }),
            json!({ "int8": { "kCentering": "sometimes" } }),
            json!({ "sol": { "topkRatio": "half" } }),
            json!(true),
            json!([]),
        ] {
            assert!(parse(bad.clone()).is_err(), "{bad}");
        }
    }

    #[test]
    fn model_admission_reads_the_catalog_capability() {
        let int8 = json!({ "lossyAttention": { "int8": {} } });
        let sol = json!({ "lossyAttention": { "sol": { "topkRatio": 0.5 } } });
        let int8 = int8.as_object().unwrap();
        let sol = sol.as_object().unwrap();
        assert!(requested_lossy_attention("m", int8, &entry(BOTH()))
            .unwrap()
            .is_some());
        assert!(requested_lossy_attention("m", sol, &entry(BOTH()))
            .unwrap()
            .is_some());

        let int8_only = entry(json!({
            "int8": { "served": true },
            "sol": { "served": false, "reason": "attends block-causally" }
        }));
        assert!(requested_lossy_attention("m", int8, &int8_only).is_ok());
        let refused = requested_lossy_attention("m", sol, &int8_only).unwrap_err();
        assert_eq!(refused.field, "advanced.lossyAttention.sol");
        assert!(
            refused.detail.contains("attends block-causally"),
            "{}",
            refused.detail
        );

        // No capability: an MLX-only or bespoke-attention model.
        let bare = json!({ "id": "m" });
        let refused = requested_lossy_attention("m", int8, bare.as_object().unwrap()).unwrap_err();
        assert_eq!(refused.field, "advanced.lossyAttention");

        // Absent / null option: nothing to check, whatever the model.
        let absent = json!({ "lossyAttention": null });
        assert!(requested_lossy_attention(
            "m",
            absent.as_object().unwrap(),
            bare.as_object().unwrap()
        )
        .unwrap()
        .is_none());
        // An empty object requests no kernel.
        let empty = json!({ "lossyAttention": {} });
        assert!(requested_lossy_attention(
            "m",
            empty.as_object().unwrap(),
            bare.as_object().unwrap()
        )
        .is_ok());
        // A malformed capability is refused, not read as unsupported.
        assert!(requested_lossy_attention("m", int8, &entry(json!({ "int8": true }))).is_err());
    }

    #[test]
    fn a_model_capability_is_the_conjunction_of_its_engines() {
        let facts: SharedAttentionFacts = serde_json::from_value(json!({
            "backend": "candle",
            "source": "test",
            "providers": {
                "both": { "routing": "shared_helpers", "int8": { "served": true },
                          "sol": { "served": true } },
                "int8_only": { "routing": "shared_helpers", "int8": { "served": true },
                               "sol": { "served": false, "reason": "no self-attention" } },
                "own": { "routing": "bespoke" }
            }
        }))
        .unwrap();
        let both = facts.capability_for(["both"]).unwrap();
        assert!(both.int8.served && both.sol.served);
        let mixed = facts.capability_for(["both", "int8_only"]).unwrap();
        assert!(mixed.int8.served);
        assert!(!mixed.sol.served);
        assert_eq!(mixed.sol.reason.as_deref(), Some("no self-attention"));
        assert!(
            facts.capability_for(["both", "own"]).is_none(),
            "bespoke engine"
        );
        assert!(
            facts.capability_for(["missing"]).is_none(),
            "unregistered engine"
        );
        assert!(facts.capability_for([]).is_none(), "no candle engine");
    }

    #[test]
    fn routing_sees_only_a_requesting_payload() {
        let with = json!({ "advanced": { "lossyAttention": { "int8": {} } } });
        let empty = json!({ "advanced": { "lossyAttention": {} } });
        let none = json!({ "advanced": {} });
        assert!(payload_requests_lossy_attention(with.as_object().unwrap()));
        assert!(!payload_requests_lossy_attention(
            empty.as_object().unwrap()
        ));
        assert!(!payload_requests_lossy_attention(none.as_object().unwrap()));
    }
}
