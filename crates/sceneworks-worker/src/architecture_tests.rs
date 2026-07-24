//! Executable drift guards for `ARCHITECTURE.md`.
//!
//! The capability matrix is checked against source text rather than a second hand-maintained Rust
//! list: adding a `JobType` or omitting an explicit worker dispatch must make this test fail until
//! the documentation and dispatcher are reconciled.

use std::collections::{BTreeMap, BTreeSet};

const CONTRACTS: &str = include_str!("../../sceneworks-core/src/contracts.rs");
const WORKER: &str = include_str!("lib.rs");
const ARCHITECTURE: &str = include_str!("../ARCHITECTURE.md");

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

/// Remove comments and literal contents before inspecting Rust syntax. This is intentionally small
/// (the production source remains compiled by rustc), but it understands nested block comments and
/// escaped quoted strings so a `JobType::Variant` mention outside code cannot satisfy the guard.
fn code_without_comments_or_literals(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
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
            (byte, _) => {
                output.push(byte);
                index += 1;
            }
        }
    }

    String::from_utf8(output).expect("Rust source outside comments and literals remains UTF-8")
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
