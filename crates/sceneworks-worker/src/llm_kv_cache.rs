//! Compressed KV cache policy and reporting for SceneWorks' local LLM generations (sc-20682).
//!
//! Off by default. An operator opts the worker in with `SCENEWORKS_LLM_KV_COMPRESSION=qualified`,
//! and a prompt-refine job may choose for itself with its payload's `kvCompression`
//! (`"off"` | `"qualified"`). Opting in never forces compression: the engine runs a generation on
//! the compressed cache only where its qualification table admits the model family and context,
//! prices its memory admission for the cache it actually runs on, and reports what ran — dense,
//! with a stable reason, everywhere else. SceneWorks keeps no KV pricing of its own: request
//! admission is the in-process engine's.
//!
//! [`kv_cache_block`] is the one shape SceneWorks records that report in — the prompt-refine
//! result's `generation.kvCache` and the `llm_kv_cache` telemetry event. It carries the policy,
//! the format, the fallback reason and the engine's counters; never prompt or output text.

use gen_core::core_llm::{KvCacheReport, KvCompressionPolicy};
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

/// The KV cache a generation asked for and ran on. `reported: false` when the provider does not
/// report one (the engine never guesses).
pub(crate) fn kv_cache_block(policy: KvCompressionPolicy, report: Option<&KvCacheReport>) -> Value {
    let Some(report) = report else {
        return json!({ "policy": policy_name(policy), "reported": false });
    };
    json!({
        "policy": policy_name(policy),
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

/// The `llm_kv_cache` telemetry event payload of one generation: the job, the engine lane and the
/// [`kv_cache_block`] — identifiers and counters only.
pub(crate) fn kv_cache_event(job_id: &str, engine: &str, block: Value) -> Value {
    json!({ "jobId": job_id, "engine": engine, "kvCache": block })
}

/// The engine lane a native LLM generation runs on: the MLX twin on macOS, Candle elsewhere.
#[cfg(target_os = "macos")]
pub(crate) const NATIVE_LLM_ENGINE: &str = "mlx";
#[cfg(not(target_os = "macos"))]
pub(crate) const NATIVE_LLM_ENGINE: &str = "candle";

/// The KV-cache diagnostics of one generation — its [`kv_cache_block`] and the `llm_kv_cache`
/// telemetry event carrying it — from the job's policy and the engine's report.
pub(crate) fn kv_cache_outcome(
    job_id: &str,
    engine: &str,
    policy: KvCompressionPolicy,
    report: Option<&KvCacheReport>,
) -> (Value, Value) {
    let block = kv_cache_block(policy, report);
    let event = kv_cache_event(job_id, engine, block.clone());
    (block, event)
}

/// Emit one generation's `llm_kv_cache` telemetry event and return its result block.
pub(crate) fn record_kv_cache(
    job_id: &str,
    engine: &str,
    policy: KvCompressionPolicy,
    report: Option<&KvCacheReport>,
) -> Value {
    let (block, event) = kv_cache_outcome(job_id, engine, policy, report);
    crate::emit_event("llm_kv_cache", event);
    block
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
            json!({ "policy": "off", "reported": false })
        );
    }

    /// The result block and the telemetry event of one generation come from the job's policy and
    /// the engine's report, and the event carries exactly that block under the engine lane.
    #[test]
    fn the_outcome_pairs_the_result_block_with_its_event() {
        let report = KvCacheReport::dense(KvCacheFallbackReason::UnqualifiedModel, None);
        let (block, event) = kv_cache_outcome(
            "job-7",
            NATIVE_LLM_ENGINE,
            KvCompressionPolicy::Qualified,
            Some(&report),
        );
        assert_eq!(
            block,
            kv_cache_block(KvCompressionPolicy::Qualified, Some(&report))
        );
        assert_eq!(block["fallbackReason"], "unqualified_model");
        assert_eq!(block["policy"], "qualified");
        assert_eq!(
            event,
            json!({ "jobId": "job-7", "engine": NATIVE_LLM_ENGINE, "kvCache": block })
        );
        assert!(matches!(NATIVE_LLM_ENGINE, "mlx" | "candle"));
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
