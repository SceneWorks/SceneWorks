//! Compressed KV cache policy and reporting for SceneWorks' local LLM generations (sc-20682).
//!
//! Off by default. An operator opts the worker in with `SCENEWORKS_LLM_KV_COMPRESSION=qualified`,
//! and a prompt-refine job may choose for itself with its payload's `kvCompression`
//! (`"off"` | `"qualified"`). Opting in never forces compression: the engine runs a generation on
//! the compressed cache only for a single sequence on a model matching one of its two measured
//! architectures (exact config geometry, so a same-geometry fine-tune qualifies and another model
//! of the family does not): Llama-3.2-3B-Instruct with a prompt of at least 32 768 tokens and a
//! final context below 130 560, or dense Qwen3-1.7B with a prompt of at least 10 240 tokens and a
//! final context of at most 40 960. Batched, short and out-of-range requests run dense, and any
//! other model reports dense, each with a stable reason.
//! The engine prices its memory admission for the cache it actually runs on and reports what ran.
//! SceneWorks keeps no KV pricing of its own: request admission is the in-process engine's.
//!
//! [`KvCacheRecord`] is the one value SceneWorks records that report in — the prompt-refine
//! result's `generation.kvCache` and the `llm_kv_cache` telemetry event — for every generation,
//! whichever way it ended. A completed one ([`kv_cache_block`]) carries the policy, the format, the
//! fallback reason and the engine's counters; a refused, canceled or failed one
//! ([`kv_cache_failure_block`], sc-20688) carries the policy, the outcome and the engine's reason.
//! Never prompt or output text.

use gen_core::core_llm::{Error as CoreLlmError, KvCacheReport, KvCompressionPolicy};
use serde_json::{json, Value};

use crate::{WorkerError, WorkerResult};

/// Worker-wide opt-in for the compressed KV cache.
pub(crate) const KV_COMPRESSION_ENV: &str = "SCENEWORKS_LLM_KV_COMPRESSION";

/// A prompt-refine job's own choice, overriding the worker-wide default.
pub(crate) const KV_COMPRESSION_PAYLOAD_KEY: &str = "kvCompression";

/// Parse a policy name: `off` or `qualified` (case-insensitive, trimmed).
pub(crate) fn parse_policy(value: &str) -> Option<KvCompressionPolicy> {
    match value.trim().to_ascii_lowercase().as_str() {
        "off" => Some(KvCompressionPolicy::Off),
        "qualified" => Some(KvCompressionPolicy::Qualified),
        _ => None,
    }
}

/// The stable name of `policy`, as the payload and the result spell it.
pub(crate) fn policy_name(policy: KvCompressionPolicy) -> &'static str {
    match policy {
        KvCompressionPolicy::Off => "off",
        KvCompressionPolicy::Qualified => "qualified",
    }
}

/// The worker-wide default from [`KV_COMPRESSION_ENV`]. Unset or empty is off; an unrecognized
/// value is also off (fail closed), with a warning naming it.
pub(crate) fn worker_default_policy() -> KvCompressionPolicy {
    default_policy_from(std::env::var(KV_COMPRESSION_ENV).ok().as_deref())
}

fn default_policy_from(value: Option<&str>) -> KvCompressionPolicy {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None => KvCompressionPolicy::Off,
        Some(value) => parse_policy(value).unwrap_or_else(|| {
            tracing::warn!(
                value,
                "{KV_COMPRESSION_ENV} is neither `off` nor `qualified`; the KV cache stays dense"
            );
            KvCompressionPolicy::Off
        }),
    }
}

/// A job's policy: its payload's [`KV_COMPRESSION_PAYLOAD_KEY`] when present (a value other than
/// `off`/`qualified` is an invalid payload), otherwise the worker-wide default.
pub(crate) fn job_policy(
    payload: &serde_json::Map<String, Value>,
) -> WorkerResult<KvCompressionPolicy> {
    job_policy_with_default(payload, worker_default_policy())
}

fn job_policy_with_default(
    payload: &serde_json::Map<String, Value>,
    default: KvCompressionPolicy,
) -> WorkerResult<KvCompressionPolicy> {
    match payload.get(KV_COMPRESSION_PAYLOAD_KEY) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value.as_str().and_then(parse_policy).ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{KV_COMPRESSION_PAYLOAD_KEY} must be \"off\" or \"qualified\", not {value}"
            ))
        }),
    }
}

/// The KV cache a generation that completed asked for and ran on (`outcome: "completed"`).
/// `reported: false` when the provider does not report one (the engine never guesses).
pub(crate) fn kv_cache_block(policy: KvCompressionPolicy, report: Option<&KvCacheReport>) -> Value {
    let Some(report) = report else {
        return json!({
            "policy": policy_name(policy),
            "outcome": OUTCOME_COMPLETED,
            "reported": false,
        });
    };
    json!({
        "policy": policy_name(policy),
        "outcome": OUTCOME_COMPLETED,
        "reported": true,
        "formatVersion": report.format_version,
        "format": report.format.map(|format| format.id()),
        "ranCompressed": report.ran_compressed(),
        "fallbackReason": report.fallback.map(|reason| reason.id()),
        "detail": report.detail,
        "counters": {
            "fusedAttentionCalls": report.counters.fused_attention_calls,
            "denseFallbackEvents": report.counters.dense_fallback_events,
            "fullCacheDequantizations": report.counters.full_cache_dequantizations,
            "denseGatherFallbacks": report.counters.dense_gather_fallbacks,
            "compressedCacheBytes": report.counters.compressed_cache_bytes,
        },
    })
}

/// The [`KvCacheRecord::outcome`] of a generation that produced an output.
const OUTCOME_COMPLETED: &str = "completed";

/// The block of a generation the engine ended with `error` instead of an output (sc-20688): the
/// requested policy, the outcome — `refused` (request memory admission, while pricing the dense or
/// compressed cache), `canceled` or `failed` — a stable `reason` and the engine's message. A
/// refusal also carries the admission's numbers. The engine reports no cache for a generation
/// without an output, so `reported` is false.
pub(crate) fn kv_cache_failure_block(policy: KvCompressionPolicy, error: &CoreLlmError) -> Value {
    let (outcome, reason) = match error {
        CoreLlmError::RequestResourceExhausted(_) => ("refused", "request_resource_exhausted"),
        CoreLlmError::Canceled => ("canceled", "canceled"),
        CoreLlmError::Unsupported(_) => ("failed", "unsupported"),
        CoreLlmError::InvalidRequest(_) => ("failed", "invalid_request"),
        _ => ("failed", "engine_error"),
    };
    let mut block = json!({
        "policy": policy_name(policy),
        "outcome": outcome,
        "reported": false,
        "reason": reason,
        "error": error.to_string(),
    });
    if let CoreLlmError::RequestResourceExhausted(refusal) = error {
        block["refusal"] = json!({
            "promptTokens": refusal.prompt_tokens,
            "maxNewTokens": refusal.max_new_tokens,
            "maxContextTokens": refusal.max_context_tokens,
            "requiredBytes": refusal.required_bytes,
            "availableBytes": refusal.available_bytes,
        });
    }
    block
}

/// The `llm_kv_cache` telemetry event payload of one generation: the job, the engine lane and the
/// block — identifiers, counters and the engine's message only.
pub(crate) fn kv_cache_event(job_id: &str, engine: &str, block: Value) -> Value {
    json!({ "jobId": job_id, "engine": engine, "kvCache": block })
}

/// The engine lane a native LLM generation runs on: the MLX twin on macOS, Candle elsewhere.
#[cfg(target_os = "macos")]
pub(crate) const NATIVE_LLM_ENGINE: &str = "mlx";
#[cfg(not(target_os = "macos"))]
pub(crate) const NATIVE_LLM_ENGINE: &str = "candle";

/// One generation's KV-cache record, whichever way it ended: the engine lane and the one block
/// both the `llm_kv_cache` event and any result block are built from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct KvCacheRecord {
    engine: &'static str,
    block: Value,
}

impl KvCacheRecord {
    /// The record of `generation` on `engine`, requested with `policy`: [`kv_cache_block`] of the
    /// output's `report` when it completed, [`kv_cache_failure_block`] of the engine's error when
    /// it was refused, canceled or failed.
    pub(crate) fn of<T>(
        engine: &'static str,
        policy: KvCompressionPolicy,
        generation: &Result<T, CoreLlmError>,
        report: impl FnOnce(&T) -> Option<&KvCacheReport>,
    ) -> Self {
        let block = match generation {
            Ok(output) => kv_cache_block(policy, report(output)),
            Err(error) => kv_cache_failure_block(policy, error),
        };
        Self { engine, block }
    }

    /// The block a result carries.
    pub(crate) fn block(&self) -> &Value {
        &self.block
    }

    /// How the generation ended: `completed`, `refused`, `canceled` or `failed`.
    #[cfg(test)]
    pub(crate) fn outcome(&self) -> &str {
        self.block["outcome"].as_str().unwrap_or_default()
    }

    /// The `llm_kv_cache` event of this record for `job_id`.
    pub(crate) fn event(&self, job_id: &str) -> Value {
        kv_cache_event(job_id, self.engine, self.block.clone())
    }

    /// Emit this record as `job_id`'s `llm_kv_cache` telemetry event.
    pub(crate) fn emit(&self, job_id: &str) {
        let event = self.event(job_id);
        #[cfg(test)]
        test_support::capture(&event);
        crate::emit_event("llm_kv_cache", event);
    }
}

/// Test seams shared by the job modules' `llm_kv_cache` tests: the emitted events, and a text
/// provider whose generation the engine refuses.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::{Mutex, OnceLock};

    use gen_core::core_llm::{
        Error as CoreLlmError, RequestResourceExhausted, StreamEvent, TextLlm, TextLlmCapabilities,
        TextLlmDescriptor, TextLlmOutput, TextLlmRequest,
    };
    use serde_json::Value;

    static EVENTS: Mutex<Vec<Value>> = Mutex::new(Vec::new());

    pub(crate) fn capture(event: &Value) {
        EVENTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event.clone());
    }

    /// Every `llm_kv_cache` event emitted for `job_id` (job ids are unique per test).
    pub(crate) fn events_for(job_id: &str) -> Vec<Value> {
        EVENTS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|event| event["jobId"] == job_id)
            .cloned()
            .collect()
    }

    /// The memory-admission refusal the engine raises while pricing a request's KV cache.
    pub(crate) fn refusal() -> CoreLlmError {
        CoreLlmError::RequestResourceExhausted(RequestResourceExhausted {
            prompt_tokens: 40_000,
            max_new_tokens: 512,
            max_context_tokens: 131_072,
            required_bytes: 9_000_000_000,
            available_bytes: 4_000_000_000,
        })
    }

    /// A text provider whose every generation the engine refuses at memory admission.
    pub(crate) struct RefusingLlm;

    impl TextLlm for RefusingLlm {
        fn descriptor(&self) -> &TextLlmDescriptor {
            static DESC: OnceLock<TextLlmDescriptor> = OnceLock::new();
            DESC.get_or_init(|| TextLlmDescriptor {
                id: "refusing".to_owned(),
                family: "stub".to_owned(),
                backend: "stub".to_owned(),
                capabilities: TextLlmCapabilities::default(),
            })
        }

        fn validate(&self, _req: &TextLlmRequest) -> gen_core::core_llm::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            _req: &TextLlmRequest,
            _on_event: &mut dyn FnMut(StreamEvent),
        ) -> gen_core::core_llm::Result<TextLlmOutput> {
            Err(refusal())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::core_llm::{
        KvCacheCounters, KvCacheFallbackReason, KvCompressionFormat, KV_CACHE_FORMAT_VERSION,
    };

    #[test]
    fn policy_names_round_trip_and_anything_else_is_refused() {
        for policy in [KvCompressionPolicy::Off, KvCompressionPolicy::Qualified] {
            assert_eq!(parse_policy(policy_name(policy)), Some(policy));
        }
        assert_eq!(
            parse_policy(" Qualified "),
            Some(KvCompressionPolicy::Qualified)
        );
        for other in ["", "on", "true", "k8v8", "compressed"] {
            assert_eq!(parse_policy(other), None, "{other}");
        }
    }

    #[test]
    fn the_worker_default_is_off_unless_explicitly_qualified() {
        assert_eq!(default_policy_from(None), KvCompressionPolicy::Off);
        assert_eq!(default_policy_from(Some("  ")), KvCompressionPolicy::Off);
        assert_eq!(default_policy_from(Some("yes")), KvCompressionPolicy::Off);
        assert_eq!(default_policy_from(Some("off")), KvCompressionPolicy::Off);
        assert_eq!(
            default_policy_from(Some("qualified")),
            KvCompressionPolicy::Qualified
        );
    }

    #[test]
    fn a_job_payload_overrides_the_worker_default_and_is_validated() {
        let qualified = KvCompressionPolicy::Qualified;
        let off = KvCompressionPolicy::Off;
        assert_eq!(
            job_policy_with_default(json!({}).as_object().unwrap(), qualified).unwrap(),
            qualified
        );
        assert_eq!(
            job_policy_with_default(json!({ "kvCompression": null }).as_object().unwrap(), off)
                .unwrap(),
            off
        );
        assert_eq!(
            job_policy_with_default(
                json!({ "kvCompression": "off" }).as_object().unwrap(),
                qualified
            )
            .unwrap(),
            off
        );
        assert_eq!(
            job_policy_with_default(
                json!({ "kvCompression": "qualified" }).as_object().unwrap(),
                off
            )
            .unwrap(),
            qualified
        );
        for bad in [json!("on"), json!(true), json!(1)] {
            let error =
                job_policy_with_default(json!({ "kvCompression": bad }).as_object().unwrap(), off)
                    .unwrap_err();
            assert!(
                matches!(&error, WorkerError::InvalidPayload(message) if message.contains("kvCompression")),
                "{error:?}"
            );
        }
    }

    #[test]
    fn the_block_records_a_compressed_run_with_its_counters() {
        let report = KvCacheReport {
            format_version: KV_CACHE_FORMAT_VERSION,
            format: Some(KvCompressionFormat::GroupAffineK8V8),
            fallback: None,
            detail: None,
            counters: KvCacheCounters {
                fused_attention_calls: 46,
                dense_fallback_events: 0,
                full_cache_dequantizations: 0,
                dense_gather_fallbacks: 2,
                compressed_cache_bytes: 123_456,
            },
        };
        assert_eq!(
            kv_cache_block(KvCompressionPolicy::Qualified, Some(&report)),
            json!({
                "policy": "qualified",
                "outcome": "completed",
                "reported": true,
                "formatVersion": KV_CACHE_FORMAT_VERSION,
                "format": "group-affine-k8v8",
                "ranCompressed": true,
                "fallbackReason": null,
                "detail": null,
                "counters": {
                    "fusedAttentionCalls": 46,
                    "denseFallbackEvents": 0,
                    "fullCacheDequantizations": 0,
                    "denseGatherFallbacks": 2,
                    "compressedCacheBytes": 123_456,
                },
            })
        );
    }

    #[test]
    fn the_block_records_a_dense_run_with_its_reason_or_an_unreported_one() {
        let dense = KvCacheReport::dense(KvCacheFallbackReason::BelowMinimumContext, None);
        let block = kv_cache_block(KvCompressionPolicy::Qualified, Some(&dense));
        assert_eq!(block["ranCompressed"], false);
        assert_eq!(block["format"], Value::Null);
        assert_eq!(block["fallbackReason"], "below_minimum_context");
        assert_eq!(block["counters"]["fusedAttentionCalls"], 0);
        let interrupted = KvCacheReport {
            format: Some(KvCompressionFormat::GroupAffineK8V8),
            fallback: Some(KvCacheFallbackReason::RuntimeFallback),
            detail: Some("dispatch-fault: reader fault".into()),
            ..dense
        };
        let block = kv_cache_block(KvCompressionPolicy::Qualified, Some(&interrupted));
        assert_eq!(block["ranCompressed"], false);
        assert_eq!(block["format"], "group-affine-k8v8");
        assert_eq!(block["fallbackReason"], "runtime_fallback");
        assert_eq!(block["detail"], "dispatch-fault: reader fault");
        assert_eq!(
            kv_cache_block(KvCompressionPolicy::Off, None),
            json!({ "policy": "off", "outcome": "completed", "reported": false })
        );
    }

    /// The result block and the telemetry event of one generation come from the job's policy and
    /// the engine's report, and the event carries exactly that block under the engine lane.
    #[test]
    fn the_record_pairs_the_result_block_with_its_event() {
        let report = KvCacheReport::dense(KvCacheFallbackReason::UnqualifiedModel, None);
        let record = KvCacheRecord::of(
            NATIVE_LLM_ENGINE,
            KvCompressionPolicy::Qualified,
            &Ok::<_, CoreLlmError>(report.clone()),
            |report| Some(report),
        );
        let block = record.block().clone();
        assert_eq!(
            block,
            kv_cache_block(KvCompressionPolicy::Qualified, Some(&report))
        );
        assert_eq!(record.outcome(), "completed");
        assert_eq!(block["fallbackReason"], "unqualified_model");
        assert_eq!(block["policy"], "qualified");
        assert_eq!(
            record.event("job-7"),
            json!({ "jobId": "job-7", "engine": NATIVE_LLM_ENGINE, "kvCache": block })
        );
        assert!(matches!(NATIVE_LLM_ENGINE, "mlx" | "candle"));
    }

    /// sc-20688: a refused, canceled or failed generation still records its requested policy, the
    /// outcome and the engine's reason, under the same event shape.
    #[test]
    fn a_generation_that_ends_in_an_engine_error_is_recorded_with_its_outcome() {
        let refused = KvCacheRecord::of(
            "mlx",
            KvCompressionPolicy::Qualified,
            &Err::<KvCacheReport, _>(test_support::refusal()),
            |report| Some(report),
        );
        assert_eq!(
            refused.block(),
            &json!({
                "policy": "qualified",
                "outcome": "refused",
                "reported": false,
                "reason": "request_resource_exhausted",
                "error": test_support::refusal().to_string(),
                "refusal": {
                    "promptTokens": 40_000,
                    "maxNewTokens": 512,
                    "maxContextTokens": 131_072,
                    "requiredBytes": 9_000_000_000_u64,
                    "availableBytes": 4_000_000_000_u64,
                },
            })
        );
        assert_eq!(
            refused.event("job-8"),
            json!({ "jobId": "job-8", "engine": "mlx", "kvCache": refused.block() })
        );
        for (error, outcome, reason) in [
            (CoreLlmError::Canceled, "canceled", "canceled"),
            (
                CoreLlmError::Unsupported("x".into()),
                "failed",
                "unsupported",
            ),
            (
                CoreLlmError::InvalidRequest("x".into()),
                "failed",
                "invalid_request",
            ),
            (CoreLlmError::Msg("x".into()), "failed", "engine_error"),
        ] {
            let block = kv_cache_failure_block(KvCompressionPolicy::Off, &error);
            assert_eq!(block["policy"], "off");
            assert_eq!(block["outcome"], outcome);
            assert_eq!(block["reason"], reason);
            assert_eq!(block["error"], error.to_string());
            assert!(block.get("refusal").is_none());
        }
    }

    #[test]
    fn the_telemetry_event_carries_identifiers_and_the_block_only() {
        let block = kv_cache_block(KvCompressionPolicy::Off, None);
        let event = kv_cache_event("job-1", "mlx", block.clone());
        assert_eq!(
            event,
            json!({ "jobId": "job-1", "engine": "mlx", "kvCache": block })
        );
    }
}
