use super::{active_statuses_sql, ACTIVE_STATUSES};

/// Anti-drift guard for sc-4207 / F-CORE-3: the five `status in (...)` SQL
/// statements now interpolate [`active_statuses_sql`] instead of a
/// copy-pasted literal, so the generated list must stay exactly in sync with
/// [`ACTIVE_STATUSES`] — every status quoted, comma-separated, none dropped.
#[test]
fn sql_list_matches_active_statuses_const() {
    let expected = ACTIVE_STATUSES
        .iter()
        .map(|status| format!("'{status}'"))
        .collect::<Vec<_>>()
        .join(", ");
    assert_eq!(active_statuses_sql(), expected);

    // Each status appears as a quoted token, guarding against a future const
    // edit that silently fails to reach the SQL filters.
    for status in ACTIVE_STATUSES {
        assert!(
            active_statuses_sql().contains(&format!("'{status}'")),
            "active status {status:?} missing from SQL list"
        );
    }
}
