use super::*;

/// Project a stored/worker-owned job into the public HTTP/SSE shape.
///
/// A selected image text encoder is resolved to an exact filesystem source only after the API has
/// accepted the client's opaque option id. That resolution must remain available to the worker
/// claim, but it is never browser state: exposing it in a job response would turn the otherwise
/// opaque option back into a host path and would let replay clients persist server-private
/// metadata. Keep the stored row intact and remove only that private resolution from public clones.
pub(super) fn public_job_snapshot(mut job: JobSnapshot) -> JobSnapshot {
    let private_resolution = job
        .payload
        .get_mut("modelManifestEntry")
        .and_then(Value::as_object_mut)
        .and_then(|manifest_entry| manifest_entry.remove("resolvedTextEncoder"));
    if let Some(selected_path) = private_resolution
        .as_ref()
        .and_then(|resolution| resolution.get("path"))
        .and_then(Value::as_str)
    {
        let source_kind = private_resolution
            .as_ref()
            .and_then(|resolution| resolution.get("sourceKind"))
            .and_then(Value::as_str);
        let selected_paths = private_text_encoder_path_spellings(selected_path, source_kind);
        redact_private_text_encoder_diagnostic(&mut job.message);
        if let Some(error) = job.error.as_mut() {
            redact_private_text_encoder_diagnostic(error);
        }
        if let Some(title) = job.title.as_mut() {
            redact_selected_text_encoder_paths(title, &selected_paths);
        }
        redact_selected_text_encoder_paths_in_map(&mut job.payload, &selected_paths);
        redact_selected_text_encoder_paths_in_map(&mut job.result, &selected_paths);
        redact_selected_text_encoder_paths_in_btree_map(&mut job.extra, &selected_paths);
    }
    job
}

#[derive(Clone, Debug)]
pub(super) struct PrivatePathSpelling {
    pub(super) value: String,
    pub(super) directory: bool,
    pub(super) case_insensitive: bool,
}

pub(super) fn private_text_encoder_path_spellings(
    path: &str,
    source_kind: Option<&str>,
) -> Vec<PrivatePathSpelling> {
    // Missing/unknown private metadata is not expected, but public projection must fail closed:
    // directory semantics are the safe superset because they also scrub admitted descendants.
    let directory = source_kind != Some("file");
    let path_bytes = path.as_bytes();
    let case_insensitive = (path_bytes.first().is_some_and(u8::is_ascii_alphabetic)
        && path_bytes.get(1) == Some(&b':')
        && path_bytes
            .get(2)
            .is_some_and(|separator| matches!(separator, b'/' | b'\\')))
        || path.starts_with("\\\\");
    let mut spellings = vec![path.to_owned()];
    if case_insensitive {
        spellings.push(path.replace('\\', "/"));
        spellings.push(path.replace('/', "\\"));
    }
    if directory {
        for spelling in &mut spellings {
            while spelling.len() > 1
                && spelling.ends_with(['/', '\\'])
                && !spelling.trim_matches(['/', '\\']).is_empty()
                && !(spelling.len() == 3 && spelling.as_bytes().get(1) == Some(&b':'))
            {
                spelling.pop();
            }
        }
    }
    spellings.retain(|spelling| !spelling.is_empty());
    spellings.sort_by_key(|spelling| std::cmp::Reverse(spelling.len()));
    spellings.dedup();
    spellings
        .into_iter()
        .map(|value| PrivatePathSpelling {
            value,
            directory,
            case_insensitive,
        })
        .collect()
}

pub(super) fn redact_private_text_encoder_diagnostic(value: &mut String) {
    redact_absolute_paths(value);
}

/// Redact only the admitted selected source spelling from public contract data.
///
/// Payload/result/extra can legitimately contain unrelated filesystem fields (for example an
/// installed LoRA or generated output). They must remain stable when a text encoder is selected.
/// HTTP(S) and scheme-relative URL spans are skipped so a URL whose path happens to contain the
/// same spelling is not rewritten.
pub(super) fn redact_selected_text_encoder_paths(
    value: &mut String,
    spellings: &[PrivatePathSpelling],
) {
    const REDACTED: &str = "[selected text encoder]";
    let original = value.as_bytes();
    let is_url_end = |byte: u8| {
        byte.is_ascii_whitespace()
            || matches!(
                byte,
                b'"' | b'\'' | b'`' | b'<' | b'>' | b'|' | b')' | b']' | b'}' | b',' | b';'
            )
    };
    let starts_with_ignore_ascii_case = |index: usize, needle: &[u8]| {
        original
            .get(index..index.saturating_add(needle.len()))
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
    };
    let url_boundary = |index: usize| {
        index == 0
            || original[index - 1].is_ascii_whitespace()
            || matches!(
                original[index - 1],
                b'(' | b'[' | b'{' | b'=' | b',' | b';' | b'\'' | b'"' | b'`' | b'<'
            )
    };
    let is_web_url = |index: usize| {
        (url_boundary(index)
            && (starts_with_ignore_ascii_case(index, b"http://")
                || starts_with_ignore_ascii_case(index, b"https://")))
            || ((index == 0 || url_boundary(index))
                && original.get(index..index + 2) == Some(b"//")
                && original
                    .get(index + 2)
                    .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'['))
    };
    let is_file_url = |index: usize| {
        url_boundary(index)
            && starts_with_ignore_ascii_case(index, b"file:")
            && original
                .get(index + 5)
                .is_some_and(|separator| matches!(separator, b'/' | b'\\'))
    };

    let mut output = String::with_capacity(value.len());
    let mut copied_through = 0;
    let mut index = 0;
    while index < original.len() {
        if is_file_url(index) {
            output.push_str(&value[copied_through..index]);
            output.push_str(REDACTED);
            index += 5;
            while index < original.len() && !is_url_end(original[index]) {
                index += 1;
            }
            copied_through = index;
            continue;
        }
        let after_scheme_slashes = || {
            let mut cursor = index;
            while cursor > 0 && matches!(original[cursor - 1], b'/' | b'\\') {
                cursor -= 1;
            }
            cursor > 0 && original[cursor - 1] == b':'
        };
        let Some(spelling) = spellings.iter().find(|spelling| {
            (index == 0
                || after_scheme_slashes()
                || !matches!(
                    original[index - 1],
                    b'.' | b'-' | b'_' | b'/' | b'\\' | b'0'..=b'9' | b'A'..=b'Z'
                        | b'a'..=b'z'
                ))
                && original
                    .get(index..index.saturating_add(spelling.value.len()))
                    .is_some_and(|candidate| {
                        if spelling.case_insensitive {
                            candidate.eq_ignore_ascii_case(spelling.value.as_bytes())
                        } else {
                            candidate == spelling.value.as_bytes()
                        }
                    })
                && original
                    .get(index + spelling.value.len())
                    .is_none_or(|next| {
                        if spelling.directory {
                            matches!(next, b'/' | b'\\')
                                || next.is_ascii_whitespace()
                                || matches!(
                                    next,
                                    b'"' | b'\''
                                        | b'`'
                                        | b'<'
                                        | b'>'
                                        | b'|'
                                        | b')'
                                        | b']'
                                        | b'}'
                                        | b','
                                        | b';'
                                        | b':'
                                        | b'='
                                        | b'?'
                                        | b'#'
                                        | b'&'
                                )
                        } else if *next == b'.' {
                            original
                                .get(index + spelling.value.len() + 1)
                                .is_none_or(|after| {
                                    after.is_ascii_whitespace()
                                        || matches!(
                                            after,
                                            b'"' | b'\''
                                                | b'`'
                                                | b'<'
                                                | b'>'
                                                | b'|'
                                                | b')'
                                                | b']'
                                                | b'}'
                                                | b','
                                                | b';'
                                        )
                                })
                        } else {
                            next.is_ascii_whitespace()
                                || matches!(
                                    next,
                                    b'"' | b'\''
                                        | b'`'
                                        | b'<'
                                        | b'>'
                                        | b'|'
                                        | b')'
                                        | b']'
                                        | b'}'
                                        | b','
                                        | b';'
                                        | b':'
                                        | b'='
                                        | b'?'
                                        | b'#'
                                        | b'&'
                                        | b'!'
                                )
                        }
                    })
        }) else {
            if is_web_url(index) {
                index += 1;
                while index < original.len() && !is_url_end(original[index]) {
                    index += 1;
                }
                continue;
            }
            index += 1;
            continue;
        };
        output.push_str(&value[copied_through..index]);
        output.push_str(REDACTED);
        index += spelling.value.len();
        copied_through = index;
    }
    if copied_through != 0 {
        output.push_str(&value[copied_through..]);
        *value = output;
    }
}

/// Remove any remaining absolute filesystem token from a selected-encoder job's public clone.
///
/// A confinement error can enumerate unrelated allowed roots and a later symlink error can name a
/// target that no longer matches the selected source. This pure lexical pass is deliberately
/// scoped to jobs carrying the server-private encoder resolution; it performs no filesystem I/O
/// while projecting an HTTP/SSE response. Once a filesystem location is found, the remainder is
/// dropped rather than guessing where an unquoted `Path::display()` value containing spaces ends.
/// Web URLs and relative paths remain intact, while `file://` URLs are filesystem locations and
/// are therefore private.
pub(super) fn redact_absolute_paths(value: &mut String) {
    const REDACTED: &str = "[selected text encoder]";
    let bytes = value.as_bytes();
    let boundary = |index: usize| match index.checked_sub(1).and_then(|i| bytes.get(i)) {
        None => true,
        Some(previous) => {
            !previous.is_ascii_alphanumeric()
                && !matches!(previous, b'_' | b'.' | b'-' | b'/' | b'\\')
        }
    };
    let starts_with_ignore_ascii_case = |index: usize, needle: &[u8]| {
        bytes
            .get(index..index.saturating_add(needle.len()))
            .is_some_and(|candidate| candidate.eq_ignore_ascii_case(needle))
    };
    let is_web_url = |index: usize| {
        boundary(index)
            && (starts_with_ignore_ascii_case(index, b"http://")
                || starts_with_ignore_ascii_case(index, b"https://")
                || (bytes.get(index..index + 2) == Some(b"//")
                    && bytes
                        .get(index + 2)
                        .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'[')))
    };
    let is_file_url =
        |index: usize| boundary(index) && starts_with_ignore_ascii_case(index, b"file://");
    let is_home_relative = |index: usize| {
        if bytes.get(index.wrapping_sub(1)) == Some(&b'~') {
            return true;
        }
        if bytes.get(index.wrapping_sub(1)) != Some(&b'}') {
            return false;
        }
        bytes[..index]
            .iter()
            .rposition(|byte| *byte == b'{')
            .is_some_and(|open| {
                open > 0
                    && bytes[open - 1] == b'$'
                    && bytes[open + 1..index - 1]
                        .iter()
                        .all(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
            })
    };
    let is_absolute_path = |index: usize, require_boundary: bool| {
        if require_boundary && !boundary(index) {
            return false;
        }
        let starts_component = |byte: &u8| {
            !byte.is_ascii_whitespace()
                && !matches!(
                    byte,
                    b'/' | b'\\'
                        | b'"'
                        | b'\''
                        | b'`'
                        | b'<'
                        | b'>'
                        | b'|'
                        | b')'
                        | b']'
                        | b'}'
                        | b','
                        | b';'
                )
        };
        let unix = !is_home_relative(index)
            && bytes.get(index) == Some(&b'/')
            && bytes.get(index + 1).is_some_and(starts_component);
        let drive = bytes.get(index).is_some_and(u8::is_ascii_alphabetic)
            && bytes.get(index + 1) == Some(&b':')
            && bytes
                .get(index + 2)
                .is_some_and(|separator| matches!(separator, b'/' | b'\\'))
            && bytes.get(index + 3).is_some_and(starts_component);
        let unc = bytes.get(index..index + 2) == Some(b"\\\\")
            && bytes.get(index + 2).is_some_and(starts_component);
        unix || drive || unc
    };
    let is_url_end = |byte: u8| {
        byte.is_ascii_whitespace()
            || matches!(
                byte,
                b'"' | b'\'' | b'`' | b'<' | b'>' | b'|' | b')' | b']' | b'}' | b',' | b';'
            )
    };
    let web_token_end = |start: usize| {
        let mut end = start;
        while end < bytes.len() && !is_url_end(bytes[end]) {
            end += 1;
        }
        end
    };
    let mut index = 0;
    while index < bytes.len() {
        if is_web_url(index) {
            index = web_token_end(index);
            continue;
        }
        let file_url = is_file_url(index);
        if !file_url && !is_absolute_path(index, true) {
            index += 1;
            continue;
        }

        let prefix_end = index
            .checked_sub(1)
            .filter(|previous| matches!(bytes[*previous], b'"' | b'\'' | b'`' | b'<'));
        value.truncate(prefix_end.unwrap_or(index));
        value.push_str(REDACTED);
        return;
    }
}

pub(super) fn redact_selected_text_encoder_paths_in_map(
    object: &mut serde_json::Map<String, Value>,
    spellings: &[PrivatePathSpelling],
) {
    let original = std::mem::take(object);
    let mut redacted = serde_json::Map::with_capacity(original.len());
    let mut collisions = std::collections::HashSet::new();
    for (mut key, mut value) in original {
        redact_selected_text_encoder_paths(&mut key, spellings);
        redact_selected_text_encoder_paths_in_value(&mut value, spellings);
        if collisions.contains(&key) {
            continue;
        }
        if redacted.contains_key(&key) {
            redacted.insert(
                key.clone(),
                Value::String("[redacted collision]".to_owned()),
            );
            collisions.insert(key);
        } else {
            redacted.insert(key, value);
        }
    }
    *object = redacted;
}

pub(super) fn redact_selected_text_encoder_paths_in_btree_map(
    object: &mut std::collections::BTreeMap<String, Value>,
    spellings: &[PrivatePathSpelling],
) {
    let original = std::mem::take(object);
    let mut redacted = std::collections::BTreeMap::new();
    let mut collisions = std::collections::HashSet::new();
    for (mut key, mut value) in original {
        redact_selected_text_encoder_paths(&mut key, spellings);
        redact_selected_text_encoder_paths_in_value(&mut value, spellings);
        if collisions.contains(&key) {
            continue;
        }
        if redacted.contains_key(&key) {
            redacted.insert(
                key.clone(),
                Value::String("[redacted collision]".to_owned()),
            );
            collisions.insert(key);
        } else {
            redacted.insert(key, value);
        }
    }
    *object = redacted;
}

pub(super) fn redact_selected_text_encoder_paths_in_value(
    value: &mut Value,
    spellings: &[PrivatePathSpelling],
) {
    match value {
        Value::String(value) => redact_selected_text_encoder_paths(value, spellings),
        Value::Array(values) => {
            for value in values {
                redact_selected_text_encoder_paths_in_value(value, spellings);
            }
        }
        Value::Object(object) => redact_selected_text_encoder_paths_in_map(object, spellings),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

pub(super) fn public_job_snapshots(jobs: Vec<JobSnapshot>) -> Vec<JobSnapshot> {
    jobs.into_iter().map(public_job_snapshot).collect()
}

pub(super) fn public_queue_summary(mut summary: QueueSummary) -> QueueSummary {
    summary.active_jobs = public_job_snapshots(summary.active_jobs);
    summary
}

/// Defense in depth for live publications. Most producers already pass typed public projections,
/// but `job.updated` also originates at worker/stale-sweep seams that intentionally retain the raw
/// job for lifecycle work. Redact at the serialization boundary so no new producer can accidentally
/// publish the worker-private selection receipt. `jobs.snapshot` is currently built directly in the
/// SSE bootstrap; supporting it here keeps this boundary safe if that snapshot becomes live later.
///
/// sc-8953 follow-through (2026-10-27): `job.updated` ALSO slims the live payload here — for a
/// NON-terminal job, `result.assets` (the fully-built normalized asset records, KBs each, one per
/// finished image) is dropped from the EVENT while `assetIds` stays. Every SSE tick used to carry
/// the whole accumulated array (a 260-image batch ≈ 650 KB+ per tick, thousands of ticks), and the
/// WebView2 renderer `JSON.parse`d every one — cumulative multi-GB allocation churn that hit V8's
/// ~4 GB old-space ceiling and OOM-killed the renderer mid-batch (Crashpad `0xE0000008`, daily
/// since 2026-10-24), leaving the desktop app a dead white window. The frontend resolves job
/// outputs id-first (`jobResultAssets.js`: `assetIds` is its FIRST preference, `assets` only a
/// pre-catalog fallback) and re-fetches the catalog on every asset-count change, so a live event
/// without the embedded records renders identically; terminal events keep the full `assets` (the
/// durable row's enrichment contract, `persist_reported_assets`). Snapshots stay full — bootstrap
/// is once per connect, not per tick.
pub(super) fn redact_private_job_metadata_from_event(event: &str, value: &mut Value) {
    fn redact_job(value: &mut Value) {
        let Ok(job) = serde_json::from_value::<JobSnapshot>(value.clone()) else {
            // A job event that cannot be projected through the public typed contract is not safe to
            // publish. Clear it so the caller's serialization remains path-free and fail-closed.
            *value = Value::Null;
            return;
        };
        let mut job = public_job_snapshot(job);
        if !sceneworks_core::jobs_store::TERMINAL_STATUSES.contains(&job.status.as_str()) {
            job.result.remove("assets");
        }
        *value = serde_json::to_value(job).unwrap_or(Value::Null);
    }

    match event {
        "job.updated" => redact_job(value),
        "queue.updated" => {
            if let Some(jobs) = value.get_mut("activeJobs").and_then(Value::as_array_mut) {
                for job in jobs {
                    redact_job(job);
                }
            }
        }
        "jobs.snapshot" => {
            if let Some(jobs) = value.get_mut("jobs").and_then(Value::as_array_mut) {
                for job in jobs {
                    redact_job(job);
                }
            }
        }
        _ => {}
    }
}
