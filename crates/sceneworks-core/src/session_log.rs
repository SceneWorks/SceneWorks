//! In-memory session log buffer (epic 3447 / sc-3451, sc-3453).
//!
//! A bounded ring buffer of captured output lines that the app can read back to
//! show "what happened this session" — most importantly the GPU routing
//! decisions (`gpu_route_decision`), claim contention (`claim_lock_contention`)
//! and the worker generation phases — without log-archaeology across the three
//! append-only files in `~/Library/Logs/SceneWorks/`.
//!
//! Two consumers share this type so the entry shape is identical on both surfaces:
//! - the **desktop** holds one process-global buffer, fed by the sidecar stdout
//!   capture in `apps/desktop/src/setup.rs` (api + worker + mlx-worker), read by
//!   the `get_session_logs` Tauri command;
//! - the **API** holds its own buffer of the structured events it emits, served by
//!   `GET /api/v1/logs` for headless/web/Docker runtimes that have no desktop wrapper.

use std::collections::VecDeque;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::time::utc_now;

/// One captured log line, tagged with its origin and severity (the **declared**
/// `level` from the tracing backbone when present, else inferred), plus the parsed
/// structured event when the line was a JSON object (the worker's `emit_worker_event`
/// output or the API's `gpu_route_decision`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogEntry {
    /// Monotonic per-buffer sequence number. Clients pass the highest seq they've
    /// seen back as `afterSeq` to tail only new lines.
    pub seq: u64,
    /// Origin stream: `api` | `worker` | `mlx-worker` (desktop), or `api` (API buffer).
    pub source: String,
    /// Severity: the declared `level` carried by the tracing envelope
    /// (`error`/`warn`/`info`/`debug`), or — for legacy/plain lines — inferred.
    pub level: String,
    /// Best-effort timestamp: the event's `reportedAt` when present, else capture time.
    pub timestamp: String,
    /// Human-facing one-liner: a compact event summary, or the raw line.
    pub message: String,
    /// The parsed structured event, when the line was a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<Value>,
    /// The original captured line, verbatim.
    pub raw: String,
}

/// Filter/window for [`SessionLog::query`].
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LogQuery {
    /// Only return entries with `seq` strictly greater than this (incremental tail).
    pub after_seq: Option<u64>,
    /// Cap on returned entries (newest-N after filtering). Defaults to 500, max 5000.
    pub limit: Option<usize>,
    /// Restrict to a single source.
    pub source: Option<String>,
    /// Restrict to a single level.
    pub level: Option<String>,
    /// Case-insensitive substring match against the raw line.
    pub search: Option<String>,
}

const DEFAULT_CAPACITY: usize = 5000;

/// A thread-safe bounded ring buffer of [`LogEntry`].
pub struct SessionLog {
    inner: Mutex<Inner>,
}

struct Inner {
    entries: VecDeque<LogEntry>,
    next_seq: u64,
    capacity: usize,
}

impl Default for SessionLog {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }
}

impl SessionLog {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: VecDeque::new(),
                next_seq: 0,
                capacity: capacity.max(1),
            }),
        }
    }

    /// Ingest a captured chunk from `source`. The chunk may contain multiple lines
    /// (or a partial line with a trailing newline); each non-blank line becomes one
    /// entry with a parsed event, inferred level, and monotonic seq.
    pub fn push_line(&self, source: &str, chunk: &str) {
        let mut guard = self.inner.lock().expect("session log lock");
        for line in chunk.split('\n') {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.trim().is_empty() {
                continue;
            }
            let seq = guard.next_seq;
            guard.next_seq = guard.next_seq.saturating_add(1);
            let line = redact_secrets(line);
            let entry = classify(source, &line, seq);
            guard.entries.push_back(entry);
            let capacity = guard.capacity;
            while guard.entries.len() > capacity {
                guard.entries.pop_front();
            }
        }
    }

    /// Return the matching entries, oldest→newest, capped to the newest `limit`.
    pub fn query(&self, query: &LogQuery) -> Vec<LogEntry> {
        let guard = self.inner.lock().expect("session log lock");
        let limit = query.limit.unwrap_or(500).clamp(1, 5000);
        let search = query.search.as_deref().map(str::to_ascii_lowercase);
        let matches = |entry: &LogEntry| {
            // MSRV 1.80: `Option::is_none_or` is 1.82, so use `map_or(true, …)`.
            query.after_seq.map_or(true, |seq| entry.seq > seq)
                && query
                    .source
                    .as_deref()
                    .map_or(true, |source| entry.source == source)
                && query
                    .level
                    .as_deref()
                    .map_or(true, |level| entry.level == level)
                && search.as_deref().map_or(true, |needle| {
                    entry.raw.to_ascii_lowercase().contains(needle)
                })
        };
        // F-093: walk newest→oldest and stop after cloning `limit` matches, instead
        // of cloning every matching entry (up to the full 5000-deep buffer) and then
        // `split_off`ing down to `limit`. This clones only what's returned and lets
        // the buffer mutex — contended with `push_line` on the stdout capture threads
        // — drop far sooner on the hot polling path.
        let mut matched: Vec<LogEntry> = guard
            .entries
            .iter()
            .rev()
            .filter(|entry| matches(entry))
            .take(limit)
            .cloned()
            .collect();
        // Collected newest→oldest above; restore the documented oldest→newest order.
        matched.reverse();
        matched
    }

    /// The next seq that would be assigned — i.e. one past the newest entry.
    pub fn next_seq(&self) -> u64 {
        self.inner.lock().expect("session log lock").next_seq
    }
}

// The `"` and `\'` terminators cover the JSON string shape (`"token":"abc"`,
// after the value-opening quote is skipped by `redact_marker_value`); `&` and
// whitespace cover URL query / header / flag shapes.
const VALUE_TERMINATORS: &[char] = &['&', '"', '\'', ' ', '\t', '<', '>', ',', '}'];

fn redact_secrets(line: &str) -> String {
    // Query / flag shapes: `token=…`, `access_token=…`, `api_key=…`.
    let line = redact_marker_value(line.to_owned(), "token=", VALUE_TERMINATORS);
    let line = redact_marker_value(line, "access_token=", VALUE_TERMINATORS);
    let line = redact_marker_value(line, "api_key=", VALUE_TERMINATORS);
    // JSON object shapes: `"token":"…"`, `"access_token":"…"`, `"api_key":"…"`.
    // The marker consumes through the value-opening quote so the redaction span
    // starts at the secret and stops at the closing quote (a terminator). This
    // catches secrets a `key=value` marker never would (F-072).
    let line = redact_json_marker_value(line, "token");
    let line = redact_json_marker_value(line, "access_token");
    let line = redact_json_marker_value(line, "api_key");
    // Bearer credential, in both `Bearer abc` and `"Bearer abc"` forms.
    let line = redact_marker_value(line, "bearer ", VALUE_TERMINATORS);
    redact_authorization_header(line)
}

/// Redact the value of a `"<key>":"` JSON pair. The marker matches through the
/// opening quote of the value so `redact_marker_value` starts replacing at the
/// secret itself and stops at the closing quote (in `VALUE_TERMINATORS`).
fn redact_json_marker_value(output: String, key: &str) -> String {
    // Tolerate the whitespace serializers put around the colon: `"key": "val"`.
    let marker_no_space = format!("\"{key}\":\"");
    let marker_space = format!("\"{key}\": \"");
    let output = redact_marker_value(output, &marker_no_space, VALUE_TERMINATORS);
    redact_marker_value(output, &marker_space, VALUE_TERMINATORS)
}

fn redact_marker_value(mut output: String, marker: &str, terminators: &[char]) -> String {
    let mut search_from = 0;
    loop {
        let lowered = output.to_ascii_lowercase();
        let Some(relative_start) = lowered[search_from..].find(marker) else {
            return output;
        };
        let start = search_from + relative_start;
        let value_start = start + marker.len();
        let value_end = output[value_start..]
            .char_indices()
            .find_map(|(index, character)| {
                terminators
                    .contains(&character)
                    .then_some(value_start + index)
            })
            .unwrap_or(output.len());
        // An immediately-terminated marker (e.g. `token=&…`) has no value here.
        // Previously this returned the whole line, so any *later* real occurrence
        // stayed unredacted (F-072). Advance past this marker and keep scanning.
        if value_end == value_start {
            search_from = value_start;
            continue;
        }
        output.replace_range(value_start..value_end, "[REDACTED]");
        search_from = value_start + "[REDACTED]".len();
    }
}

fn redact_authorization_header(mut output: String) -> String {
    let mut search_from = 0;
    loop {
        let lowered = output.to_ascii_lowercase();
        let Some(relative_start) = lowered[search_from..].find("authorization:") else {
            return output;
        };
        let start = search_from + relative_start;
        let after_colon = start + "authorization:".len();
        // Skip optional whitespace, then an optional value-opening quote (the JSON
        // `"authorization":"Bearer …"` shape). Redact from there.
        let mut value_start = after_colon;
        let bytes = output.as_bytes();
        while value_start < output.len() && matches!(bytes[value_start], b' ' | b'\t') {
            value_start += 1;
        }
        // A quoted value (JSON) terminates at its closing quote; a bare header
        // value (`Authorization: Bearer abc123`) has no quote and runs to
        // end-of-line. Previously this always replaced to end-of-line, clobbering
        // any trailing JSON fields on the same line (F-072).
        let quoted = value_start < output.len() && bytes[value_start] == b'"';
        if quoted {
            value_start += 1; // step over the opening quote
        }
        let value_end = if quoted {
            output[value_start..]
                .find('"')
                .map(|offset| value_start + offset)
                .unwrap_or(output.len())
        } else {
            // Bare header form: redact the whole credential to end-of-line, but
            // stop at a structural JSON delimiter if one follows on the line.
            output[value_start..]
                .find(['"', ',', '}'])
                .map(|offset| value_start + offset)
                .unwrap_or(output.len())
        };
        if value_end == value_start {
            search_from = after_colon;
            continue;
        }
        output.replace_range(value_start..value_end, "[REDACTED]");
        search_from = value_start + "[REDACTED]".len();
    }
}

fn classify(source: &str, line: &str, seq: u64) -> LogEntry {
    let event = serde_json::from_str::<Value>(line)
        .ok()
        .filter(Value::is_object);
    let level = infer_level(source, line, event.as_ref());
    let timestamp = event
        .as_ref()
        .and_then(|value| value.get("reportedAt"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(utc_now);
    let message = event
        .as_ref()
        .and_then(summarize_event)
        .unwrap_or_else(|| line.to_owned());
    LogEntry {
        seq,
        source: source.to_owned(),
        level,
        timestamp,
        message,
        event,
        raw: line.to_owned(),
    }
}

/// Severity for a captured line. A **declared** `level` field (emitted by the
/// `tracing` backbone, [`crate::observability`]) is authoritative and used verbatim
/// — this is what makes the Logs-screen `level` filter trustworthy. Only legacy /
/// plain lines that lack a declared level fall back to the heuristic: a structured
/// `event` name ending in `_failed`/`_error`, or an `error`/`errorType` field, is an
/// error; `claim_lock_contention` is a warn; otherwise sniff the raw text.
fn infer_level(_source: &str, line: &str, event: Option<&Value>) -> String {
    if let Some(event) = event {
        // Declared level wins, verbatim (the sceneworks tracing envelope always
        // carries one). This is what makes the filter trustworthy — e.g. a 4xx
        // `api_error` is emitted at `debug` and must STAY debug, not get re-promoted
        // to error by the `_error`-suffix heuristic below. Only legacy / plain lines
        // with no declared level fall through to the heuristic.
        if let Some(declared) = event.get("level").and_then(Value::as_str) {
            let declared = declared.trim().to_ascii_lowercase();
            if !declared.is_empty() {
                return declared;
            }
        }
        let name = event.get("event").and_then(Value::as_str).unwrap_or("");
        if name.ends_with("_failed")
            || name.ends_with("_error")
            || event.get("error").is_some_and(|v| !v.is_null())
            || event.get("errorType").is_some_and(|v| !v.is_null())
        {
            return "error".to_owned();
        }
        if name == "claim_lock_contention" || name.contains("warn") {
            return "warn".to_owned();
        }
        return "info".to_owned();
    }
    let lowered = line.to_ascii_lowercase();
    if lowered.contains("panic")
        || lowered.contains("traceback")
        || lowered.contains("error")
        || lowered.contains("_failed")
        || lowered.contains("failed:")
    {
        "error".to_owned()
    } else if lowered.contains("warn") {
        "warn".to_owned()
    } else {
        "info".to_owned()
    }
}

/// Compact one-liner for a structured event: the event name plus a curated set of
/// high-signal fields when present, so the routing/claim story reads at a glance.
fn summarize_event(event: &Value) -> Option<String> {
    // A structured event without an `event` name (e.g. a plain `tracing` message
    // routed through the backbone) summarizes to its `message` field, so the Logs
    // screen shows readable text rather than the raw JSON line.
    let Some(name) = event.get("event").and_then(Value::as_str) else {
        return event
            .get("message")
            .and_then(Value::as_str)
            .map(str::to_owned);
    };
    let mut summary = name.to_owned();
    const FIELDS: &[&str] = &[
        "decision",
        "reason",
        "model",
        "jobId",
        "gpuId",
        "adapter",
        "imageIndex",
        "imageCount",
        "consecutiveFailures",
        "status",
        "path",
        "detail",
        "error",
    ];
    for field in FIELDS {
        if let Some(value) = event.get(*field) {
            if value.is_null() {
                continue;
            }
            let rendered = match value {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            summary.push_str(&format!(" {field}={rendered}"));
        }
    }
    Some(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_structured_event_with_summary_and_level() {
        let log = SessionLog::with_capacity(16);
        log.push_line(
            "api",
            &json!({
                "event": "gpu_route_decision",
                "decision": "claimed_by_candle",
                "reason": "candle_worker",
                "model": "qwen_image_edit_2511_lightning",
                "jobId": "job_1",
                "reportedAt": "2026-06-07T00:00:00Z"
            })
            .to_string(),
        );
        let entries = log.query(&LogQuery::default());
        assert_eq!(entries.len(), 1);
        let entry = &entries[0];
        assert_eq!(entry.source, "api");
        assert_eq!(entry.level, "info");
        assert_eq!(entry.timestamp, "2026-06-07T00:00:00Z");
        assert!(entry.message.contains("gpu_route_decision"));
        assert!(entry.message.contains("decision=claimed_by_candle"));
        assert!(entry.message.contains("reason=candle_worker"));
        assert!(entry.event.is_some());
    }

    #[test]
    fn declared_level_overrides_heuristic() {
        let log = SessionLog::with_capacity(16);
        // A 4xx `api_error` is declared at debug — the name ends in `_error`, which
        // the heuristic would promote to error, but the declared level must win so the
        // error filter stays trustworthy.
        log.push_line(
            "api",
            &json!({ "event": "api_error", "level": "debug", "status": 404 }).to_string(),
        );
        // A declared error level is honored even when the text has no error markers.
        log.push_line(
            "api",
            &json!({ "event": "gpu_route_decision", "level": "error" }).to_string(),
        );
        // No declared level -> fall back to the heuristic (legacy / Python worker line).
        log.push_line(
            "worker",
            &json!({ "event": "image_inference_failed", "error": "boom" }).to_string(),
        );
        let entries = log.query(&LogQuery::default());
        assert_eq!(
            entries[0].level, "debug",
            "declared debug wins over _error suffix"
        );
        assert_eq!(entries[1].level, "error", "declared error honored verbatim");
        assert_eq!(
            entries[2].level, "error",
            "heuristic still applies with no level"
        );
    }

    #[test]
    fn failed_event_and_contention_levels() {
        let log = SessionLog::with_capacity(16);
        log.push_line(
            "mlx-worker",
            &json!({ "event": "image_inference_failed", "jobId": "j", "error": "boom" })
                .to_string(),
        );
        log.push_line(
            "worker",
            &json!({ "event": "claim_lock_contention", "consecutiveFailures": 3 }).to_string(),
        );
        log.push_line("worker", "plain info line");
        log.push_line("worker", "Traceback (most recent call last):");
        let entries = log.query(&LogQuery::default());
        assert_eq!(entries[0].level, "error");
        assert_eq!(entries[1].level, "warn");
        assert_eq!(entries[2].level, "info");
        assert_eq!(entries[3].level, "error");
    }

    #[test]
    fn splits_multiline_chunks_and_drops_blank_lines() {
        let log = SessionLog::with_capacity(16);
        log.push_line("worker", "first\n\nsecond\r\n");
        let entries = log.query(&LogQuery::default());
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].raw, "first");
        assert_eq!(entries[1].raw, "second");
        assert_eq!(entries[0].seq, 0);
        assert_eq!(entries[1].seq, 1);
    }

    #[test]
    fn redacts_secret_shapes_on_ingest() {
        let log = SessionLog::with_capacity(16);
        log.push_line(
            "worker",
            "fetch https://example.test/model?token=secret-value&x=1 Authorization: Bearer abc123",
        );
        log.push_line(
            "api",
            &json!({
                "event": "download_started",
                "url": "https://example.test/file?access_token=hidden",
                "authorization": "Bearer nested-secret"
            })
            .to_string(),
        );

        let entries = log.query(&LogQuery::default());
        assert!(entries[0].raw.contains("token=[REDACTED]&x=1"));
        assert!(entries[0].raw.contains("Authorization: [REDACTED]"));
        assert!(!entries[0].raw.contains("secret-value"));
        assert!(entries[1].raw.contains("access_token=[REDACTED]"));
        assert!(!entries[1].raw.contains("nested-secret"));
        assert_eq!(
            entries[1]
                .event
                .as_ref()
                .and_then(|event| event.get("url"))
                .and_then(Value::as_str),
            Some("https://example.test/file?access_token=[REDACTED]")
        );
    }

    #[test]
    fn empty_marker_value_does_not_abandon_later_occurrence() {
        // F-072: a first `token=` with no value (immediately terminated) must not
        // short-circuit redaction — a later real `token=` on the same line has to
        // still be redacted.
        let log = SessionLog::with_capacity(4);
        log.push_line("worker", "token=&next token=real-secret&x=1");
        let entry = &log.query(&LogQuery::default())[0];
        assert!(
            !entry.raw.contains("real-secret"),
            "later token= must still be redacted: {}",
            entry.raw
        );
        assert!(entry.raw.contains("token=[REDACTED]&x=1"));
    }

    #[test]
    fn redacts_json_shaped_secret_values() {
        // F-072: JSON `"token":"abc"` shapes match no `key=value` marker, so they
        // previously survived. The dedicated JSON pass must redact them (both the
        // no-space and spaced serializations), leaving surrounding fields intact.
        let log = SessionLog::with_capacity(4);
        log.push_line(
            "api",
            r#"{"token":"jwt-secret","api_key": "sk-abc","keep":"value"}"#,
        );
        let entry = &log.query(&LogQuery::default())[0];
        assert!(!entry.raw.contains("jwt-secret"), "raw: {}", entry.raw);
        assert!(!entry.raw.contains("sk-abc"), "raw: {}", entry.raw);
        assert!(
            entry.raw.contains("\"keep\":\"value\""),
            "non-secret field preserved: {}",
            entry.raw
        );
        // Still valid JSON after redaction.
        let event = entry.event.as_ref().expect("event parses");
        assert_eq!(event.get("keep").and_then(Value::as_str), Some("value"));
        assert_eq!(
            event.get("token").and_then(Value::as_str),
            Some("[REDACTED]")
        );
    }

    #[test]
    fn authorization_redaction_preserves_trailing_json_fields() {
        // F-072: the old redact-to-end-of-line clobbered every field after an
        // `authorization` key. Terminate at the closing quote so later fields stay.
        let log = SessionLog::with_capacity(4);
        log.push_line(
            "api",
            r#"{"authorization":"Bearer abc123","url":"https://example.test/x","status":200}"#,
        );
        let entry = &log.query(&LogQuery::default())[0];
        assert!(!entry.raw.contains("abc123"), "raw: {}", entry.raw);
        let event = entry.event.as_ref().expect("event still parses as JSON");
        assert_eq!(
            event.get("url").and_then(Value::as_str),
            Some("https://example.test/x"),
            "trailing url field must survive: {}",
            entry.raw
        );
        assert_eq!(event.get("status").and_then(Value::as_u64), Some(200));
    }

    #[test]
    fn authorization_bare_header_still_redacts_to_end_of_line() {
        // The plain header form has no closing quote; the whole credential must
        // still be redacted (regression guard for the delimiter change).
        let log = SessionLog::with_capacity(4);
        log.push_line("worker", "GET /x\nAuthorization: Bearer abc123");
        let entries = log.query(&LogQuery::default());
        let header = entries
            .iter()
            .find(|entry| entry.raw.contains("Authorization"))
            .expect("header line present");
        assert!(!header.raw.contains("abc123"), "raw: {}", header.raw);
        assert!(header.raw.contains("Authorization: [REDACTED]"));
    }

    #[test]
    fn ring_buffer_bounds_and_after_seq_tail() {
        let log = SessionLog::with_capacity(3);
        for i in 0..5 {
            log.push_line("api", &format!("line {i}"));
        }
        let all = log.query(&LogQuery::default());
        assert_eq!(all.len(), 3, "capacity bounds to newest 3");
        assert_eq!(all[0].raw, "line 2");
        assert_eq!(all[2].raw, "line 4");

        // Tail only entries newer than seq 3.
        let tail = log.query(&LogQuery {
            after_seq: Some(3),
            ..Default::default()
        });
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].seq, 4);
    }

    #[test]
    fn query_limit_returns_newest_n_in_order() {
        // F-093: with more matches than `limit`, return exactly the newest `limit`
        // entries, still oldest→newest. Guards the reverse-take-reverse rewrite that
        // avoids cloning the whole buffer.
        let log = SessionLog::with_capacity(100);
        for i in 0..20 {
            log.push_line("api", &format!("line {i}"));
        }
        let limited = log.query(&LogQuery {
            limit: Some(3),
            ..Default::default()
        });
        assert_eq!(limited.len(), 3);
        assert_eq!(limited[0].raw, "line 17");
        assert_eq!(limited[1].raw, "line 18");
        assert_eq!(limited[2].raw, "line 19");
        // Ascending seq order preserved.
        assert!(limited[0].seq < limited[1].seq && limited[1].seq < limited[2].seq);
    }

    #[test]
    fn filters_by_source_level_and_search() {
        let log = SessionLog::with_capacity(64);
        log.push_line("api", "alpha routing");
        log.push_line("worker", "beta error happened");
        log.push_line("mlx-worker", "gamma routing");

        let by_source = log.query(&LogQuery {
            source: Some("mlx-worker".to_owned()),
            ..Default::default()
        });
        assert_eq!(by_source.len(), 1);
        assert_eq!(by_source[0].source, "mlx-worker");

        let by_level = log.query(&LogQuery {
            level: Some("error".to_owned()),
            ..Default::default()
        });
        assert_eq!(by_level.len(), 1);
        assert_eq!(by_level[0].source, "worker");

        let by_search = log.query(&LogQuery {
            search: Some("ROUTING".to_owned()),
            ..Default::default()
        });
        assert_eq!(by_search.len(), 2, "case-insensitive substring");
    }
}
