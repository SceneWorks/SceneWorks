//! sc-11161 (F-004): `truncate_prompt` used to slice `prompt[..max]` at a
//! raw byte index. `max` (80/60) is a byte budget, so a prompt of multi-byte
//! UTF-8 (CJK/emoji/accents) whose byte at `max` fell mid-char panicked with
//! "byte index is not a char boundary". Because this runs inside
//! `derive_job_title`, which `row_to_job` calls on EVERY row read, one such
//! prompt bricked list/get/queue/claim/sweep for the whole project. These
//! guard that a long non-ASCII prompt truncates safely, never panicking.
use super::{derive_job_title, JobType};
use serde_json::{Map, Value};

fn payload_with_prompt(prompt: &str) -> Map<String, Value> {
    let mut payload = Map::new();
    payload.insert("prompt".to_owned(), Value::String(prompt.to_owned()));
    payload
}

#[test]
fn long_cjk_prompt_truncates_without_panicking() {
    // 40 three-byte CJK chars = 120 bytes; byte index 80 lands mid-char.
    let prompt = "一二三四五六七八九十".repeat(4);
    assert!(prompt.len() > 80);
    let payload = payload_with_prompt(&prompt);

    let title =
        derive_job_title(&JobType::ImageGenerate, &payload).expect("image generate yields a title");

    assert!(title.starts_with("Generate Image — "));
    assert!(
        title.ends_with('…'),
        "truncated title keeps the ellipsis: {title}"
    );
    // The rendered subject stays within the byte budget (+ ellipsis), and is
    // valid UTF-8 by construction (a String was returned, no panic).
    let subject = title.trim_start_matches("Generate Image — ");
    assert!(subject.trim_end_matches('…').len() <= 80);
}

#[test]
fn long_emoji_prompt_truncates_without_panicking() {
    // Four-byte emoji; byte index 80 (=20 emoji) falls on a boundary here,
    // but a mixed run exercises the char-boundary walk under the budget.
    let prompt = format!("{}sunset over the bay", "🌅".repeat(30));
    assert!(prompt.len() > 80);
    let payload = payload_with_prompt(&prompt);

    let title =
        derive_job_title(&JobType::VideoGenerate, &payload).expect("video generate yields a title");

    assert!(title.starts_with("Generate Video — "));
    assert!(
        title.ends_with('…'),
        "truncated title keeps the ellipsis: {title}"
    );
}

#[test]
fn short_non_ascii_prompt_is_untouched() {
    let prompt = "日本語のプロンプト"; // < 80 bytes, no truncation
    assert!(prompt.len() <= 80);
    let payload = payload_with_prompt(prompt);

    let title =
        derive_job_title(&JobType::ImageGenerate, &payload).expect("image generate yields a title");

    assert_eq!(title, format!("Generate Image — {prompt}"));
}

#[test]
fn prompt_refine_60_byte_budget_truncates_safely() {
    let prompt = "가나다라마바사아자차카타파하".repeat(4); // 3-byte Hangul, > 60 bytes
    assert!(prompt.len() > 60);
    let payload = payload_with_prompt(&prompt);

    let title =
        derive_job_title(&JobType::PromptRefine, &payload).expect("prompt refine yields a title");

    assert!(title.starts_with("Prompt Refine — "));
    assert!(title.ends_with('…'));
}
