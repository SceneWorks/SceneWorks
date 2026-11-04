//! Shared outbound timeout and retry policies. Streaming transfers deliberately
//! have no total deadline: progress, Range resume and cancellation remain owned
//! by their download controllers. This module has no HTTP/runtime dependency.
use std::time::Duration;

pub const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
pub const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(60);
pub const CONTROL_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
// Planner connections retain their configured per-request total deadline (5–300s).
pub const API_OUTBOUND_TIMEOUT: Duration = Duration::from_secs(300);

/// Capped exponential delay, with a one-based failure count. Callers retain their
/// own healthy-uptime reset and scheduling policy (desktop 20s; worker 300s).
pub fn retry_delay_seconds(base_seconds: u64, attempt: u32) -> u64 {
    let multiplier = 2_u64.saturating_pow(attempt.saturating_sub(1).min(4));
    base_seconds.max(1).saturating_mul(multiplier).clamp(1, 30)
}

/// Advance a delay already returned by `retry_delay_seconds`. The desktop tracks
/// the delay rather than a failure count, preserving its existing restart state.
pub fn next_retry_delay_seconds(current_seconds: u64) -> u64 {
    current_seconds.max(1).saturating_mul(2).min(30)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policies_preserve_control_streaming_and_backoff_bounds() {
        assert!(HTTP_CONNECT_TIMEOUT < CONTROL_REQUEST_TIMEOUT);
        assert!(DOWNLOAD_READ_TIMEOUT < API_OUTBOUND_TIMEOUT);
        assert_eq!(
            (1..=8)
                .map(|n| retry_delay_seconds(1, n))
                .collect::<Vec<_>>(),
            vec![1, 2, 4, 8, 16, 16, 16, 16]
        );
        assert_eq!(retry_delay_seconds(0, 0), 1);
        assert_eq!(retry_delay_seconds(u64::MAX, u32::MAX), 30);
        assert_eq!(next_retry_delay_seconds(u64::MAX), 30);
        assert_eq!(next_retry_delay_seconds(16), 30);
    }
}
