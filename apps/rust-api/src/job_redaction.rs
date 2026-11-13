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
    redact_manifest_locators_from_job_payload(&mut job.payload);
    redact_manifest_locators_from_job_payload(&mut job.result);
    for map in [&mut job.payload, &mut job.result] {
        map.remove("portableModelBinding");
        map.remove("portableJobBinding");
        map.remove(sceneworks_core::jobs_store::IMPORT_CLAIM_CAPABILITY_KEY);
        for value in map.values_mut() {
            redact_private_binding_markers(value);
        }
    }
    job.extra.remove("portableModelBinding");
    job.extra.remove("portableJobBinding");
    job.extra
        .remove(sceneworks_core::jobs_store::IMPORT_CLAIM_CAPABILITY_KEY);
    for value in job.extra.values_mut() {
        redact_private_binding_markers(value);
    }
    // Worker source guards may include an OS path in a refusal. Public diagnostics keep the
    // actionable prefix while removing the path and any following private detail.
    redact_absolute_paths_with(&mut job.message, "[redacted path]");
    if let Some(error) = job.error.as_mut() {
        redact_absolute_paths_with(error, "[redacted path]");
    }
    if let Some(title) = job.title.as_mut() {
        redact_absolute_paths_with(title, "[redacted path]");
    }
    job
}

pub(super) fn redact_private_binding_markers(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            fields.remove("portableModelBinding");
            fields.remove("portableJobBinding");
            fields.remove(sceneworks_core::jobs_store::IMPORT_CLAIM_CAPABILITY_KEY);
            fields.remove("__portableAuxPaths");
            for child in fields.values_mut() {
                redact_private_binding_markers(child);
            }
        }
        Value::Array(entries) => {
            for entry in entries {
                redact_private_binding_markers(entry);
            }
        }
        _ => {}
    }
}

/// Worker-local model paths are never part of the public job contract. Keep logical model ids,
/// component roles, revisions and relative file names while dropping physical install hints from
/// every resolved model carrier, including companions and trained checkpoint entries.
fn redact_manifest_locators_from_job_payload(payload: &mut JsonObject) {
    payload.remove("modelPath");
    payload.remove("installedPath");
    payload.remove("sourcePath");
    payload.remove("secondarySourcePath");
    payload.remove("targetDir");
    payload.remove("manifestPath");
    for (key, value) in payload {
        redact_nested_manifest_carriers(key, value);
    }
}

fn redact_nested_manifest_carriers(key: &str, value: &mut Value) {
    if matches!(
        key,
        "modelManifestEntry"
            | "baseModelManifestEntry"
            | "modelManifestEntries"
            | "loras"
            | "adapters"
            | "controls"
            | "advanced"
            | "plan"
    ) {
        redact_manifest_locators(value);
        return;
    }
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                redact_nested_manifest_carriers(key, child);
            }
        }
        Value::Array(entries) => {
            for entry in entries {
                redact_nested_manifest_carriers("", entry);
            }
        }
        _ => {}
    }
}

fn redact_manifest_locators(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            fields.retain(|key, value| {
                if matches!(
                    key.as_str(),
                    "paths"
                        | "modelPath"
                        | "installedPath"
                        | "sourcePath"
                        | "resolvedTextEncoder"
                        | "installationRoot"
                        | "cacheRoot"
                ) {
                    return false;
                }
                if key.eq_ignore_ascii_case("path") || key.eq_ignore_ascii_case("directory") {
                    return !value.as_str().is_some_and(is_physical_path);
                }
                if (key.ends_with("Path") || key.ends_with("Dir") || key.ends_with("Root"))
                    && value.as_str().is_some_and(is_physical_path)
                {
                    return false;
                }
                true
            });
            for (key, child) in fields.iter_mut() {
                // A prompt guide's `path` is the web app's own asset URL (`/prompt-guides/…`,
                // checked against the web assets at build time), not a host locator.
                let web_guide = (key == "promptGuide")
                    .then(|| child.as_object_mut())
                    .flatten()
                    .and_then(|guide| guide.remove("path"))
                    .and_then(|path| {
                        if path.as_str().is_some_and(is_web_prompt_guide_path) {
                            Some(path)
                        } else {
                            None
                        }
                    });
                redact_manifest_locators(child);
                if let (Some(path), Some(guide)) = (web_guide, child.as_object_mut()) {
                    guide.insert("path".to_owned(), path);
                }
            }
        }
        Value::Array(entries) => {
            for entry in entries {
                redact_manifest_locators(entry);
            }
        }
        _ => {}
    }
}

/// A web-app prompt-guide asset URL: `/prompt-guides/<name>.md`, one plain segment.
fn is_web_prompt_guide_path(value: &str) -> bool {
    value.strip_prefix("/prompt-guides/").is_some_and(|name| {
        !name.is_empty()
            && name.ends_with(".md")
            && !name.contains(['/', '\\'])
            && name != ".."
            && !name.starts_with('.')
    })
}

fn is_physical_path(value: &str) -> bool {
    value.starts_with('/')
        || value.starts_with("\\\\")
        || value.starts_with("file://")
        || (value.as_bytes().get(1) == Some(&b':')
            && value
                .as_bytes()
                .get(2)
                .is_some_and(|separator| matches!(separator, b'/' | b'\\')))
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
    redact_absolute_paths_with(value, "[selected text encoder]");
}

fn redact_absolute_paths_with(value: &mut String, replacement: &str) {
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
        value.push_str(replacement);
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

#[cfg(test)]
mod portable_redaction_tests {
    use super::*;
    use serde_json::json;

    /// SW-7033 Tier 2: the public job kept dropping `ui.promptGuide.path` because the web asset
    /// URL `/prompt-guides/flux-dev.md` starts with `/`. It is served data, not a host locator; a
    /// host path placed in the same field is still removed, as is every real locator.
    #[test]
    fn public_job_keeps_the_web_prompt_guide_path_and_still_removes_host_paths() {
        let mut payload = json!({
            "modelManifestEntry":{
                "id":"flux_dev", "paths":{"model":"${HF_CACHE}/SceneWorks/flux1-dev-mlx"},
                "ui":{"promptGuide":{"path":"/prompt-guides/flux-dev.md",
                    "sources":[{"label":"card","url":"https://example.invalid/card"}]}}
            },
            "modelManifestEntries":[
                {"id":"user_model","ui":{"promptGuide":{"path":"/home/user/private/guide.md"}}},
                {"id":"escape","ui":{"promptGuide":{"path":"/prompt-guides/../etc/passwd"}}}
            ]
        })
        .as_object()
        .unwrap()
        .clone();
        redact_manifest_locators_from_job_payload(&mut payload);
        let entry = &payload["modelManifestEntry"];
        assert_eq!(
            entry["ui"]["promptGuide"]["path"],
            "/prompt-guides/flux-dev.md"
        );
        assert_eq!(
            entry["ui"]["promptGuide"]["sources"][0]["url"],
            "https://example.invalid/card"
        );
        assert!(entry.get("paths").is_none());
        for index in [0, 1] {
            assert!(
                payload["modelManifestEntries"][index]["ui"]["promptGuide"]
                    .get("path")
                    .is_none(),
                "{}",
                payload["modelManifestEntries"][index]
            );
        }
    }

    #[test]
    fn public_job_keeps_logical_identity_but_removes_all_manifest_and_adapter_locators() {
        let temp = tempfile::tempdir().unwrap();
        let store = sceneworks_core::jobs_store::JobsStore::new(temp.path().join("jobs.db"));
        store.initialize().unwrap();
        let payload = json!({
            "model":"trained_iris",
            "modelManifestEntry":{
                "id":"trained_iris", "modelPath":"/private/trained.safetensors",
                "installedPath":"C:\\Models\\trained", "paths":{"model":"/private/trained"},
                "source":{"provider":"local","path":"/private/source"},
                "components":[{"componentId":"text_encoder","role":"required","path":"/private/encoder"}],
                "downloads":[{"repo":"owner/model","revision":"a".repeat(40),"files":["q4/weights.safetensors"]}]
            },
            "baseModelManifestEntry":{"id":"builtin_base","paths":{"model":"${HF_CACHE}/base"}},
            "modelManifestEntries":[{"id":"comfy_base","components":[{"componentId":"vae","path":"D:\\Comfy\\vae.safetensors","files":["vae.safetensors"]}]}],
            "upscale":{"modelManifestEntry":{"id":"iris_restore","modelPath":"/private/restorer.safetensors"}},
            "loras":[{"id":"adapter","installedPath":"/private/adapter.safetensors"}]
            ,"plan":{"target":{"baseModel":"builtin_base","baseModelPath":"/private/training-base"}}
        });
        let raw = store
            .create_job(sceneworks_core::jobs_store::CreateJob {
                job_type: JobType::ImageGenerate,
                project_id: None,
                project_name: None,
                payload: payload.as_object().unwrap().clone(),
                requested_gpu: "auto".into(),
                source_job_id: None,
                duplicate_of_job_id: None,
                attempts: 1,
                initial_status: None,
            })
            .unwrap();
        let mut raw_with_echo = raw.clone();
        raw_with_echo.message =
            "Portable source refused: /private/worker/weights.safetensors".into();
        raw_with_echo.error = Some("Cannot confine C:\\private\\worker\\model.safetensors".into());
        raw_with_echo.title = Some("Source \\\\worker\\share\\model refused".into());
        raw_with_echo.payload.insert(
            sceneworks_core::jobs_store::IMPORT_CLAIM_CAPABILITY_KEY.into(),
            json!("private-import-proof"),
        );
        raw_with_echo.result.insert(
            "nested".into(),
            json!({"portableModelBinding":{"secret":"must-not-leak"},"importClaimCapability":"private-import-proof"}),
        );
        let public = public_job_snapshot(raw_with_echo);
        let encoded = serde_json::to_string(&public.payload).unwrap();
        for private in ["/private", "C:\\\\Models", "D:\\\\Comfy", "${HF_CACHE}"] {
            assert!(!encoded.contains(private), "leaked {private}: {encoded}");
        }
        assert!(!serde_json::to_string(&public)
            .unwrap()
            .contains("must-not-leak"));
        assert!(!serde_json::to_string(&public)
            .unwrap()
            .contains("private-import-proof"));
        let diagnostics = format!("{} {:?} {:?}", public.message, public.error, public.title);
        assert!(!diagnostics.contains("/private/worker"));
        assert!(!diagnostics.contains("C:\\private"));
        assert!(!diagnostics.contains("\\\\worker"));
        assert!(diagnostics.contains("[redacted path]"));
        assert_eq!(public.payload["modelManifestEntry"]["id"], "trained_iris");
        assert_eq!(
            public.payload["modelManifestEntry"]["components"][0]["componentId"],
            "text_encoder"
        );
        assert_eq!(
            public.payload["modelManifestEntry"]["downloads"][0]["files"][0],
            "q4/weights.safetensors"
        );
        assert_eq!(
            public.payload["baseModelManifestEntry"]["id"],
            "builtin_base"
        );
        assert_eq!(
            public.payload["modelManifestEntries"][0]["components"][0]["files"][0],
            "vae.safetensors"
        );
        assert_eq!(public.payload["loras"][0]["id"], "adapter");
        assert_eq!(
            public.payload["plan"]["target"]["baseModel"],
            "builtin_base"
        );
        assert!(public.payload["plan"]["target"]
            .get("baseModelPath")
            .is_none());
        assert_eq!(
            public.payload["upscale"]["modelManifestEntry"]["id"],
            "iris_restore"
        );
        assert_eq!(
            raw.payload["modelManifestEntry"]["modelPath"],
            "/private/trained.safetensors"
        );
        let share = sceneworks_core::workflow_share::build_workflow_share_from(
            &sceneworks_core::workflow_share::WorkflowAssetFacts {
                mode: "text_to_image".into(),
                model: "trained_iris".into(),
                prompt: "a lighthouse".into(),
                negative_prompt: String::new(),
                seed: 7,
                width: Some(512),
                height: Some(512),
            },
            &raw.payload,
        );
        let shared = serde_json::to_string(&share).unwrap();
        assert_eq!(share.model, "trained_iris");
        assert!(!shared.contains("/private") && !shared.contains("Comfy"));
        assert!(!shared.contains("portableModelBinding"));
        let mut receipt = json!({
            "model":"trained_iris",
            "rawAdapterSettings":{
                "portableModelBinding":{"secret":"must-not-leak"},
                "advanced":{"__portableAuxPaths":{"reference":"/private/worker/reference.safetensors"}}
            },
            "modelProvenance":{"modelId":"trained_iris"}
        });
        redact_private_binding_markers(&mut receipt);
        assert_eq!(receipt["model"], "trained_iris");
        assert_eq!(receipt["modelProvenance"]["modelId"], "trained_iris");
        assert!(!serde_json::to_string(&receipt)
            .unwrap()
            .contains("must-not-leak"));
        assert!(!serde_json::to_string(&receipt)
            .unwrap()
            .contains("/private/worker/reference.safetensors"));
        let mut event = serde_json::to_value(raw).unwrap();
        event["payload"][sceneworks_core::jobs_store::IMPORT_CLAIM_CAPABILITY_KEY] =
            json!("private-import-proof");
        event["message"] = json!("Portable source refused: /private/worker/weights.safetensors");
        event["error"] = json!("Cannot confine C:\\private\\worker\\model.safetensors");
        redact_private_job_metadata_from_event("job.updated", &mut event);
        assert!(!event.to_string().contains("private-import-proof"));
        assert!(!serde_json::to_string(&event).unwrap().contains("/private"));
        assert!(!serde_json::to_string(&event)
            .unwrap()
            .contains("C:\\\\private"));
    }
}
