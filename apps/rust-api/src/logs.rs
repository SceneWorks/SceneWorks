use super::*;

use std::sync::OnceLock;

use sceneworks_core::session_log::{LogEntry, LogQuery, SessionLog};

/// Process-global API-side session log (sc-3453). On the desktop the richer
/// multi-source buffer lives in the Tauri wrapper (sc-3451), fed by every
/// sidecar's stdout; this buffer covers headless/web/Docker runtimes that have no
/// wrapper by retaining the structured events the API process itself emits (MLX
/// routing decisions, etc.), served by `GET /api/v1/logs`. Same `LogEntry` shape
/// as the desktop buffer so the in-app Logs screen is source-agnostic.
static API_SESSION_LOG: OnceLock<SessionLog> = OnceLock::new();

pub(crate) fn api_session_log() -> &'static SessionLog {
    API_SESSION_LOG.get_or_init(SessionLog::default)
}

/// Record a structured JSON event line into the API session buffer. Called from the
/// same sites that `println!` events to stdout, so the HTTP endpoint and the
/// desktop's stdout-capture buffer see the same lines.
pub(crate) fn record_api_event(line: &str) {
    api_session_log().push_line("api", line);
}

/// `GET /api/v1/logs` — the current process's session events, filtered by the
/// `LogQuery` params (`afterSeq`, `limit`, `source`, `level`, `search`).
pub(crate) async fn list_logs(Query(query): Query<LogQuery>) -> Json<Vec<LogEntry>> {
    Json(api_session_log().query(&query))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_queries_api_event() {
        // Unique marker so this is robust against other tests sharing the global buffer.
        let marker = "route-decision-test-marker-9f3a";
        record_api_event(
            &json!({
                "event": "mlx_route_decision",
                "decision": "fell_back_to_torch",
                "reason": "no_idle_mlx_worker",
                "model": marker,
                "reportedAt": "2026-06-07T00:00:00Z"
            })
            .to_string(),
        );
        let hits = api_session_log().query(&LogQuery {
            search: Some(marker.to_owned()),
            ..Default::default()
        });
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, "api");
        assert!(hits[0].message.contains("decision=fell_back_to_torch"));
        assert!(hits[0].event.is_some());
    }
}
