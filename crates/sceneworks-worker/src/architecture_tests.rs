//! Executable drift guards for `ARCHITECTURE.md`.
//!
//! The capability matrix is checked against source text rather than a second hand-maintained Rust
//! list: adding a `JobType` or omitting an explicit worker dispatch must make this test fail until
//! the documentation and dispatcher are reconciled.

use std::collections::{BTreeMap, BTreeSet};

const CONTRACTS: &str = include_str!("../../sceneworks-core/src/contracts.rs");
const WORKER: &str = include_str!("lib.rs");
const ARCHITECTURE: &str = include_str!("../ARCHITECTURE.md");
const FACE_ANALYSIS_JOBS: &str = include_str!("face_analysis_jobs.rs");

fn known_job_types() -> BTreeMap<String, String> {
    let body = CONTRACTS
        .split_once("pub enum JobType {")
        .expect("contracts.rs must declare JobType")
        .1
        .split_once("\n    }")
        .expect("JobType declaration must have a closing brace")
        .0;

    body.lines()
        .filter_map(|line| {
            let (variant, wire) = line.split_once("=>")?;
            let wire = wire.trim().strip_prefix('"')?.split('"').next()?;
            Some((variant.trim().to_owned(), wire.to_owned()))
        })
        .collect()
}

fn documented_job_types() -> Vec<String> {
    let matrix = ARCHITECTURE
        .split_once("<!-- job-matrix:start -->")
        .expect("architecture must mark the job matrix start")
        .1
        .split_once("<!-- job-matrix:end -->")
        .expect("architecture must mark the job matrix end")
        .0;

    matrix
        .lines()
        .filter_map(|line| {
            let value = line.strip_prefix("| `")?;
            Some(value.split('`').next()?.to_owned())
        })
        .collect()
}

/// The end of a `'x'` / `'\n'` / `'\''` / `'\u{1F600}'` char literal starting at `start`, or `None`
/// when the quote opens a lifetime (`'a`, `'_`, `'static`).
///
/// Without this, a `'"'` char literal — `downloads.rs`, `session_log.rs`, `jsonc.rs` and four other
/// scanned files contain one — flips the string-literal parity of everything that follows it in the
/// file. The consequence is not a false positive but a silent BLIND SPOT: the tail of the file gets
/// absorbed into a phantom string literal and stops being scanned at all.
///
/// # The two ways to get this wrong, both of which shipped
///
/// * **Escapes.** The closing quote of `'\''` is the byte at `start + 3`; the byte at `start + 2` is
///   the *escaped* quote. Searching from `start + 2` ends the literal one byte early and orphans a
///   `'`, which then eats the opening `"` of the next string literal and inverts parity for the rest
///   of the file. `media_jobs.rs:3087` (`.replace('\'', "'\\''")`) put its whole tail out of reach of
///   both gates that way — strictly worse than the stripper this replaced.
/// * **Lifetimes.** "a quote somewhere in the next five bytes" reads `Foo<'a, 'b>` as a char literal
///   spanning `'a, '`, which can again leave the cursor on a `"`. A plain char literal is exactly one
///   code point wide, so the closing quote must sit *immediately* after it — required here, never
///   searched for.
pub(crate) fn char_literal_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start) != Some(&b'\'') {
        return None;
    }
    if bytes.get(start + 1) == Some(&b'\\') {
        // The escaped byte is at `start + 2` and is never the terminator. Escapes run from `'\n'`
        // (4 bytes) to `'\u{1F600}'` (11).
        return (start + 3..=start + 11)
            .find(|index| bytes.get(*index) == Some(&b'\''))
            .map(|index| index + 1);
    }
    let width = match *bytes.get(start + 1)? {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => return None,
    };
    (bytes.get(start + 1 + width) == Some(&b'\'')).then_some(start + 2 + width)
}

/// A raw string literal (`r"…"`, `r#"…"#`, `br#"…"#`) whose `r`/`b` prefix starts at `start`, as
/// `(number of hashes, index just past the whole literal)`.
///
/// A raw string can contain unescaped `"` — `ideogram_caption.rs` has `r#"a "quoted" fox"#` — which
/// the plain-string walk closes early on, flipping parity exactly as a `'"'` char literal does.
pub(crate) fn raw_string_parts(bytes: &[u8], start: usize) -> Option<(usize, usize)> {
    let mut index = start;
    if bytes.get(index) == Some(&b'b') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'r') {
        return None;
    }
    index += 1;
    let hash_start = index;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    let hashes = index - hash_start;
    if bytes.get(index) != Some(&b'"') {
        return None;
    }
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'"'
            && bytes[index + 1..]
                .iter()
                .take(hashes)
                .filter(|byte| **byte == b'#')
                .count()
                == hashes
        {
            return Some((hashes, index + 1 + hashes));
        }
        index += 1;
    }
    Some((hashes, bytes.len()))
}

/// Whether the byte at `index` could START an `r"…"` / `b"…"` prefix rather than continue an
/// identifier (`for r in …` must not read `r` as a raw-string prefix, but `let x = r"…"` must).
fn starts_a_token(bytes: &[u8], index: usize) -> bool {
    index
        .checked_sub(1)
        .map(|previous| !(bytes[previous] == b'_' || bytes[previous].is_ascii_alphanumeric()))
        .unwrap_or(true)
}

/// Remove comments — and nothing else — before inspecting Rust syntax. Understands nested block
/// comments and steps over quoted strings, raw strings and char literals, so a `//` inside a URL
/// literal is not mistaken for the start of a comment and a `'"'` does not open a phantom string.
///
/// String literal CONTENTS survive, which is what separates this from
/// [`code_without_comments_or_literals`]. [`crate::job_time_download_guard`] (sc-17637) needs both:
/// its download-reachability sweep must not let a doc-comment mention of `ensure_hf_cached_file`
/// count as a call, while its `<data_dir>/cache` sweep is looking for `.join("cache")` — a
/// destination that only exists *as* a string literal, and which the blanking pass below erases.
pub(crate) fn code_without_comments(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        // Char literals and raw strings first: both can carry a `"` that must not open a string.
        if bytes[index] == b'\'' {
            if let Some(end) = char_literal_end(bytes, index) {
                output.extend_from_slice(&bytes[index..end.min(bytes.len())]);
                index = end.min(bytes.len());
                continue;
            }
        }
        if (bytes[index] == b'r' || bytes[index] == b'b') && starts_a_token(bytes, index) {
            if let Some((_, end)) = raw_string_parts(bytes, index) {
                let end = end.min(bytes.len());
                output.extend_from_slice(&bytes[index..end]);
                index = end;
                continue;
            }
        }
        match (bytes[index], bytes.get(index + 1).copied()) {
            (b'/', Some(b'/')) => {
                index += 2;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            (b'/', Some(b'*')) => {
                index += 2;
                let mut depth = 1_u32;
                while index < bytes.len() && depth > 0 {
                    match (bytes[index], bytes.get(index + 1).copied()) {
                        (b'/', Some(b'*')) => {
                            depth += 1;
                            index += 2;
                        }
                        (b'*', Some(b'/')) => {
                            depth -= 1;
                            index += 2;
                        }
                        _ => index += 1,
                    }
                }
            }
            (b'"', _) => {
                output.push(b'"');
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => {
                            output.push(b'\\');
                            if let Some(escaped) = bytes.get(index + 1).copied() {
                                output.push(escaped);
                            }
                            index += 2;
                        }
                        b'"' => {
                            output.push(b'"');
                            index += 1;
                            break;
                        }
                        byte => {
                            output.push(byte);
                            index += 1;
                        }
                    }
                }
            }
            (byte, _) => {
                output.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8(output).expect("Rust source outside comments remains UTF-8")
}

/// Remove comments and literal contents before inspecting Rust syntax. This is intentionally small
/// (the production source remains compiled by rustc), but it understands nested block comments and
/// escaped quoted strings so a `JobType::Variant` mention outside code cannot satisfy the guard.
///
/// Shared with [`crate::candle_preview_wiring_tests`] (sc-16962), whose guard reads the candle image
/// lanes' own source: without comment stripping, a doc comment that NAMES the forbidden
/// `preview: Default::default()` would trip the guard it documents. [`crate::job_time_download_guard`]
/// (sc-17637) uses it for the same reason — ~25 doc comments name `ensure_hf_cached_file` in files
/// that never call it.
pub(crate) fn code_without_comments_or_literals(source: &str) -> String {
    let comment_free = code_without_comments(source);
    let bytes = comment_free.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        // A char literal is kept verbatim — callers look for `'{'` — but it must be *stepped over*,
        // or the `"` in a `'"'` literal opens a phantom string that blanks the rest of the file.
        if bytes[index] == b'\'' {
            if let Some(end) = char_literal_end(bytes, index) {
                let end = end.min(bytes.len());
                output.extend_from_slice(&bytes[index..end]);
                index = end;
                continue;
            }
        }
        // A raw string keeps its delimiter shape and loses its contents, like a plain one.
        if (bytes[index] == b'r' || bytes[index] == b'b') && starts_a_token(bytes, index) {
            if let Some((hashes, end)) = raw_string_parts(bytes, index) {
                if bytes[index] == b'b' {
                    output.push(b'b');
                }
                output.push(b'r');
                output.extend(std::iter::repeat(b'#').take(hashes));
                output.extend_from_slice(b"\"\"");
                output.extend(std::iter::repeat(b'#').take(hashes));
                index = end.min(bytes.len());
                continue;
            }
        }
        match bytes[index] {
            b'"' => {
                output.push(b'"');
                index += 1;
                while index < bytes.len() {
                    match bytes[index] {
                        b'\\' => index += 2,
                        b'"' => {
                            output.push(b'"');
                            index += 1;
                            break;
                        }
                        _ => index += 1,
                    }
                }
            }
            byte => {
                output.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8(output).expect("Rust source outside comments and literals remains UTF-8")
}

#[test]
fn catalog_face_adapter_uses_each_backends_actual_score_field() {
    let code = code_without_comments_or_literals(FACE_ANALYSIS_JOBS);
    let mlx_arm = code
        .split_once("CatalogFaceBackend::Mlx(analysis) => {")
        .expect("catalog face adapter must keep an MLX conversion arm")
        .1
        .split_once("CatalogFaceBackend::Candle(analysis) =>")
        .expect("MLX conversion must precede the candle conversion")
        .0;
    assert!(
        mlx_arm.contains("confidence: face.score"),
        "mlx-gen-face::Detection exposes `score`, not gen-core's `det_score`"
    );
    assert!(
        !mlx_arm.contains("face.det_score"),
        "the MLX conversion must not use the candle result field"
    );

    let candle_arm = code
        .split_once("CatalogFaceBackend::Candle(analysis) =>")
        .expect("catalog face adapter must keep a candle conversion arm")
        .1
        .split_once("Ok((image.width, image.height, detections))")
        .expect("candle conversion must remain inside CatalogFaceDetector::detect")
        .0;
    assert!(
        candle_arm.contains("confidence: face.det_score"),
        "gen-core's candle face-analysis result exposes `det_score`"
    );
    assert!(
        candle_arm.contains(".detect(&image)"),
        "catalog face counting must use the candle detection-only path"
    );
    assert!(
        !candle_arm.contains(".analyze(&image)"),
        "catalog face counting must not compute unused ArcFace embeddings"
    );
}

/// Extract only `JobType` variants occurring in top-level patterns of
/// `run_utility_job`'s `match job.job_type`. Mentions in comments, strings, helper calls, or arm
/// bodies are deliberately ignored.
fn explicit_dispatch_variants(source: &str) -> BTreeSet<String> {
    let code = code_without_comments_or_literals(source);
    let function = code
        .split_once("async fn run_utility_job(")
        .expect("worker must declare run_utility_job")
        .1;
    let match_body = function
        .split_once("match job.job_type {")
        .expect("run_utility_job must match job.job_type")
        .1;

    let bytes = match_body.as_bytes();
    let mut variants = BTreeSet::new();
    let mut arm_start = 0;
    let mut braces = 0_u32;
    let mut parentheses = 0_u32;
    let mut brackets = 0_u32;
    let mut index = 0;

    while index < bytes.len() {
        match bytes[index] {
            b'{' => braces += 1,
            b'}' if braces == 0 => break,
            b'}' => braces -= 1,
            b'(' => parentheses += 1,
            b')' => parentheses -= 1,
            b'[' => brackets += 1,
            b']' => brackets -= 1,
            b'=' if bytes.get(index + 1) == Some(&b'>')
                && braces == 0
                && parentheses == 0
                && brackets == 0 =>
            {
                let pattern = &match_body[arm_start..index];
                let mut rest = pattern;
                while let Some((_, after_prefix)) = rest.split_once("JobType::") {
                    let variant: String = after_prefix
                        .chars()
                        .take_while(|character| {
                            character.is_ascii_alphanumeric() || *character == '_'
                        })
                        .collect();
                    if !variant.is_empty() {
                        variants.insert(variant);
                    }
                    rest = after_prefix;
                }
                index += 1;
            }
            b',' if braces == 0 && parentheses == 0 && brackets == 0 => arm_start = index + 1,
            _ => {}
        }
        index += 1;
    }

    variants
}

#[test]
fn architecture_matrix_covers_every_known_job_type_and_dispatch_arm() {
    let known = known_job_types();
    let rows = documented_job_types();
    let documented: BTreeSet<_> = rows.iter().cloned().collect();
    let expected: BTreeSet<_> = known.values().cloned().collect();

    assert_eq!(
        rows.len(),
        documented.len(),
        "ARCHITECTURE.md has duplicate JobType rows"
    );
    assert_eq!(
        documented, expected,
        "ARCHITECTURE.md must have exactly one row for every known JobType"
    );

    let dispatched = explicit_dispatch_variants(WORKER);
    let expected_variants: BTreeSet<_> = known.keys().cloned().collect();
    assert_eq!(
        dispatched, expected_variants,
        "run_utility_job must have an explicit match arm for every known JobType"
    );

    assert!(
        !ARCHITECTURE.contains("Proof (file:line)"),
        "architecture anchors must name durable modules/functions, not line numbers"
    );
}

#[test]
fn dispatch_guard_rejects_removed_arm_even_when_a_comment_keeps_the_variant_name() {
    let mutated = WORKER.replacen(
        "JobType::PromptRefine =>",
        "// JobType::PromptRefine => removed arm\n            _removed_prompt_refine =>",
        1,
    );
    assert_ne!(mutated, WORKER, "mutation target must exist");
    assert!(
        !explicit_dispatch_variants(&mutated).contains("PromptRefine"),
        "a comment containing JobType::PromptRefine must not masquerade as a dispatch arm"
    );
}

#[test]
fn real_image_job_modules_import_the_json_macro_they_invoke() {
    let image_jobs = include_str!("image_jobs.rs");
    let module_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/image_jobs");

    for module in image_jobs.lines().filter_map(|line| {
        line.trim()
            .strip_prefix("mod ")
            .and_then(|line| line.strip_suffix(';'))
    }) {
        let path = module_dir.join(format!("{module}.rs"));
        if !path.is_file() {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let code = code_without_comments_or_literals(&source);
        if !code.contains("json!") {
            continue;
        }
        let imports_json = code.split(';').any(|statement| {
            statement.trim_start().starts_with("use serde_json")
                && statement
                    .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
                    .any(|word| word == "json")
        });
        assert!(
            imports_json,
            "{} invokes json! but does not explicitly import serde_json::json",
            path.display()
        );
    }
}

/// sc-21534 — the pre-loader source guard (sc-19708) can re-verify a multi-GB resolved bundle,
/// which outlasts the API's 90s stale-worker timeout. `run_utility_job` must therefore await the
/// guard's blocking task through the `heartbeat_while_blocking` keepalive (whose behavior is
/// pinned by `heartbeat_while_blocking_keeps_worker_live_through_a_long_pass`), never bare —
/// a bare `.await` is exactly how the Krea bf16 lost-heartbeat incident swept a healthy worker.
/// Source-structure assertion in the same style as the dispatch-matrix test above: behavior lives
/// in the wrapper's own test; this pins the WIRING at the one admission call site.
#[test]
fn the_source_guard_await_is_wrapped_in_the_heartbeat_keepalive() {
    let code = code_without_comments_or_literals(WORKER);
    let function = code
        .split_once("async fn run_utility_job(")
        .expect("worker must declare run_utility_job")
        .1;
    let after_guard_spawn = function
        .split_once("RuntimeSourceGuard::begin(")
        .expect("run_utility_job must admit every job through the source guard")
        .1;
    let between_guard_and_dispatch = after_guard_spawn
        .split_once("match job.job_type {")
        .expect("the guard must run before the job-type dispatch")
        .0;
    let after_wrapper = between_guard_and_dispatch
        .split_once("heartbeat_while_blocking(")
        .expect(
            "the source-guard blocking task must be awaited via heartbeat_while_blocking, not \
             bare — a guard pass longer than the stale-worker timeout would get the worker swept \
             mid-verify",
        )
        .1;
    // The wrapper must be wrapping THE GUARD's handle, not some other blocking task that later
    // lands in the same window while the guard goes back to a bare await.
    assert!(
        after_wrapper
            .split_once(')')
            .expect("the keepalive call must close its argument list")
            .0
            .contains("guard_task"),
        "heartbeat_while_blocking must receive the source guard's own JoinHandle (guard_task)"
    );
}
