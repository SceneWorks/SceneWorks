//! CANDLE krea control capture owner.
use super::*;

pub(super) fn ensure_krea_quality(maximum: f64, mean: f64, label: &str) -> Result<(), String> {
    if maximum > KREA_CANDLE_MAX_THRESHOLD || mean > KREA_CANDLE_MEAN_THRESHOLD {
        return Err(format!(
            "{label} exceeded the approved Candle Krea parity envelope: max={maximum:.6}, mean={mean:.6}"
        ));
    }
    Ok(())
}
