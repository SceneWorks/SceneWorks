use std::time::{SystemTime, UNIX_EPOCH};

pub fn utc_now() -> String {
    format_unix_seconds(now_unix_seconds())
}

pub fn now_unix_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_secs()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

pub fn format_unix_seconds(timestamp: i64) -> String {
    let days = timestamp.div_euclid(86_400);
    let seconds_of_day = timestamp.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Parse a `YYYY-MM-DDTHH:MM:SSZ` UTC timestamp — the shape [`format_unix_seconds`]
/// emits — back into Unix seconds (truncating any sub-second part). This is the
/// inverse of `format_unix_seconds`; the two now live side by side (sc-8897 /
/// F-095) rather than the parser being buried 8k lines away in `jobs_store`.
///
/// A trailing `.digitsZ` fractional-seconds suffix is tolerated as well, even
/// though this module's `utc_now()` (and the frozen Python predecessor's, which
/// used `microsecond=0`) never emits one: an externally-mutated `started_at` /
/// `completed_at` column carrying RFC-3339 sub-second precision must still yield
/// a sane elapsed time rather than silently reading as `None`. That behavior is
/// pinned by `elapsed_seconds_accepts_fractional_rfc3339_timestamps` in the
/// integration tests, so the branch is kept deliberately here.
pub(crate) fn parse_utc_seconds(value: &str) -> Option<i64> {
    if value.len() < 20 {
        return None;
    }
    let year = value.get(0..4)?.parse::<i32>().ok()?;
    let month = value.get(5..7)?.parse::<u32>().ok()?;
    let day = value.get(8..10)?.parse::<u32>().ok()?;
    let hour = value.get(11..13)?.parse::<i64>().ok()?;
    let minute = value.get(14..16)?.parse::<i64>().ok()?;
    let second = value.get(17..19)?.parse::<i64>().ok()?;
    let suffix = value.get(19..)?;
    if value.get(4..5)? != "-"
        || value.get(7..8)? != "-"
        || value.get(10..11)? != "T"
        || value.get(13..14)? != ":"
        || value.get(16..17)? != ":"
        || month == 0
        || month > 12
        || day == 0
        || day > 31
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    // Canonical `Z`, or a tolerated `.<digits>Z` fractional-seconds suffix. The
    // sub-second digits are truncated (not rounded) — second granularity is all
    // the elapsed-time consumer needs.
    if suffix != "Z" {
        if !suffix.starts_with('.') || !suffix.ends_with('Z') {
            return None;
        }
        if !suffix[1..suffix.len() - 1]
            .chars()
            .all(|character| character.is_ascii_digit())
        {
            return None;
        }
    }
    Some(days_from_civil(year, month, day) * 86_400 + hour * 3_600 + minute * 60 + second)
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let adjusted_days = days + 719_468;
    let era = adjusted_days.div_euclid(146_097);
    let day_of_era = adjusted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);
    (year, month, day)
}

/// Inverse of the `civil_from_days` calendar math: the (year, month, day) → days
/// half of the pair, used by [`parse_utc_seconds`]. Moved here from `jobs_store`
/// so both directions of the date conversion live together (sc-8897 / F-095).
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let adjusted_year = i64::from(year) - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let month = i64::from(month);
    let day = i64::from(day);
    let month_prime = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * month_prime + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

#[cfg(test)]
mod tests {
    use super::{format_unix_seconds, parse_utc_seconds};

    #[test]
    fn format_then_parse_round_trips() {
        // format_unix_seconds and parse_utc_seconds are inverses for the canonical
        // `...ssZ` shape (sc-8897 / F-095). Cover the epoch, a recent timestamp, a
        // leap-day, and a pre-epoch (negative) instant.
        for seconds in [
            0_i64,
            1_600_000_000,
            1_751_500_800,
            // 2024-02-29T12:24:56Z — a leap day.
            1_709_209_496,
            // 1969-12-31T23:59:59Z — one second before the epoch.
            -1,
        ] {
            let formatted = format_unix_seconds(seconds);
            assert_eq!(
                parse_utc_seconds(&formatted),
                Some(seconds),
                "round trip failed for {seconds} ({formatted})"
            );
        }
    }

    #[test]
    fn parse_rejects_malformed_timestamps() {
        // Too short / missing Z / bad separators / out-of-range fields all fail.
        assert_eq!(parse_utc_seconds(""), None);
        assert_eq!(parse_utc_seconds("2026-07-03T00:00:00"), None); // no trailing Z
        assert_eq!(parse_utc_seconds("2026-07-03 00:00:00Z"), None); // space, not T
        assert_eq!(parse_utc_seconds("2026-13-03T00:00:00Z"), None); // month 13
        assert_eq!(parse_utc_seconds("2026-07-03T24:00:00Z"), None); // hour 24
        assert_eq!(parse_utc_seconds("2026-07-03T00:00:00.12x3Z"), None); // non-digit frac
        assert_eq!(parse_utc_seconds("2026-07-03T00:00:00.123"), None); // frac w/o Z
    }

    #[test]
    fn parse_tolerates_fractional_seconds_by_truncating() {
        // A `.digitsZ` fractional-seconds suffix (only ever from an externally
        // mutated DB) parses, truncating the sub-second part to whole seconds —
        // the behavior the elapsed-time consumer relies on (sc-8897 keeps this).
        let base = parse_utc_seconds("2026-05-17T13:00:04Z").expect("canonical parses");
        assert_eq!(
            parse_utc_seconds("2026-05-17T13:00:04.521Z"),
            Some(base),
            "fractional seconds truncate to the same whole-second instant"
        );
    }
}
