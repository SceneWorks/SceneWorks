use super::*;

/// The shipped `builtin.models.jsonc` `models` array, parsed from the exact bytes the product
/// embeds. Lifted here (sc-24109) so the MLX and Candle routing suites read ONE copy: both ask the
/// same question of the same file — "can a user on this OS actually obtain weights for a model this
/// lane routes?" — and two private copies of the parse are how the video guard ended up with no
/// image twin at all.
#[cfg(test)]
pub(crate) fn builtin_models() -> Vec<serde_json::Value> {
    let manifest: serde_json::Value = serde_json::from_str(&crate::jsonc::strip_jsonc_comments(
        include_str!("../../../../../config/manifests/builtin.models.jsonc"),
    ))
    .expect("builtin.models.jsonc parses");
    manifest["models"]
        .as_array()
        .expect("builtin.models.jsonc has a models array")
        .clone()
}

/// PRIMARY (non-co-requisite) download rows of `model` that survive `retain_downloads_for_os` for
/// `os`. A row with no `platforms` key is platform-agnostic and always applies.
///
/// Co-requisites install ALONGSIDE a primary and never AS one (`is_co_requisite_download`), so a
/// model whose only off-Mac rows are co-requisites still has no base checkpoint — which is why this
/// counts primaries and not rows.
#[cfg(test)]
pub(crate) fn primary_rows_on(model: &serde_json::Value, os: &str) -> usize {
    model["downloads"]
        .as_array()
        .map(|downloads| {
            downloads
                .iter()
                .filter(|download| download["coRequisite"].as_bool() != Some(true))
                .filter(|download| match download["platforms"].as_array() {
                    Some(platforms) => platforms.iter().any(|value| value.as_str() == Some(os)),
                    None => true,
                })
                .count()
        })
        .unwrap_or(0)
}

mod active_statuses_sql_tests;
mod bulk_reload_tests;
mod candle_routing_tests;
mod claim_projection_tests;
mod connection_pool_tests;
mod derive_job_title_truncate_tests;
mod mlx_routing_tests;
mod pending_workflow_tests;
mod termination_failure_error_tests;
