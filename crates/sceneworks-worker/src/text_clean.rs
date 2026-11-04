//! Model-output text cleanup. Policies preserve each provider's established output contract.

#[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
#[derive(Clone, Copy)]
pub(crate) enum ThinkPolicy {
    /// Prompt refinement mirrors its case-insensitive non-greedy regex: keep unfinished blocks.
    PreserveUnclosed,
    /// SenseNova drops a truncated reasoning suffix, matching exact lowercase tags.
    DropUnclosed,
}

#[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
pub(crate) fn strip_think_blocks(input: &str, policy: ThinkPolicy) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let find = |text: &str, tag: &str| match policy {
        ThinkPolicy::PreserveUnclosed => first_ci(text, tag),
        ThinkPolicy::DropUnclosed => text.find(tag),
    };
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        match find(rest, OPEN) {
            Some(open) => {
                out.push_str(&rest[..open]);
                let after_open = &rest[open + OPEN.len()..];
                match find(after_open, CLOSE) {
                    Some(close) => rest = &after_open[close + CLOSE.len()..],
                    None => {
                        if matches!(policy, ThinkPolicy::PreserveUnclosed) {
                            out.push_str(&rest[open..]);
                        }
                        return out;
                    }
                }
            }
            None => {
                out.push_str(rest);
                return out;
            }
        }
    }
}

#[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
pub(crate) fn first_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

#[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
fn last_ci(haystack: &str, needle: &str) -> Option<usize> {
    let (h, n) = (haystack.as_bytes(), needle.as_bytes());
    if n.is_empty() || n.len() > h.len() {
        return None;
    }
    (0..=h.len() - n.len())
        .rev()
        .find(|&i| h[i..i + n.len()].eq_ignore_ascii_case(n))
}

#[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
pub(crate) fn strip_reasoning_and_fence(text: &str) -> String {
    let mut text = strip_think_blocks(text.trim(), ThinkPolicy::PreserveUnclosed)
        .trim()
        .to_owned();
    // An orphan closing tag (no matching open): keep only what follows the last one.
    if let Some(pos) = last_ci(&text, "</think>") {
        text = text[pos + "</think>".len()..].trim().to_owned();
    }
    // A wrapping ```…``` code fence: drop the fence lines.
    if text.starts_with("```") && text.ends_with("```") {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() >= 2 {
            text = lines[1..lines.len() - 1].join("\n").trim().to_owned();
        }
    }
    text
}

/// Prompt cleanup: reasoning/orphan tags, a wrapping fence, then the outermost JSON object.
#[cfg(any(test, target_os = "macos", feature = "backend-candle"))]
pub(crate) fn clean_json_output(text: &str) -> String {
    let text = strip_reasoning_and_fence(text);
    isolate_json_object(&text)
}

fn isolate_json_object(text: &str) -> String {
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if end >= start => text[start..=end].to_owned(),
        _ => text.to_owned(),
    }
}

/// Catalog cleanup deliberately preserves reasoning and only removes a wrapping fence/prose.
pub(crate) fn clean_fenced_json_output(text: &str) -> String {
    let trimmed = text.trim();
    let unfenced = if trimmed.starts_with("```") && trimmed.ends_with("```") {
        let after_header = trimmed
            .find('\n')
            .map(|index| &trimmed[index + 1..])
            .unwrap_or(trimmed);
        after_header
            .strip_suffix("```")
            .unwrap_or(after_header)
            .trim()
    } else {
        trimmed
    };
    isolate_json_object(unfenced)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_policies_preserve_provider_case_and_truncation_contracts() {
        let raw = "猫<ThInK>secret</tHiNk> answer<think>unfinished";
        assert_eq!(
            strip_think_blocks(raw, ThinkPolicy::PreserveUnclosed),
            "猫 answer<think>unfinished"
        );
        assert_eq!(
            strip_think_blocks(raw, ThinkPolicy::DropUnclosed),
            "猫<ThInK>secret</tHiNk> answer"
        );
        assert_eq!(
            strip_think_blocks(
                "a<think>x</think>b<think>y</think>c",
                ThinkPolicy::DropUnclosed
            ),
            "abc"
        );
        assert_eq!(
            strip_reasoning_and_fence("discard</THINK> ```json\n{\"猫\":1}\n```"),
            "{\"猫\":1}"
        );
    }

    #[test]
    fn json_policies_keep_catalog_reasoning_and_prompt_cleanup_distinct() {
        let raw = " <think>{\"private\":1}</think>```json\n{\"answer\":2}\n``` ";
        assert_eq!(clean_json_output(raw), "{\"answer\":2}");
        assert_eq!(
            clean_fenced_json_output(raw),
            "{\"private\":1}</think>```json\n{\"answer\":2}"
        );
        for clean in [clean_json_output, clean_fenced_json_output] {
            assert_eq!(clean("prefix {\"a\":[1,2]} suffix"), "{\"a\":[1,2]}");
            assert_eq!(clean("```json\n{}\n```"), "{}");
            assert_eq!(clean(" refusal "), "refusal");
        }
        // The catalog's original fence parser also accepts a missing header newline.
        assert_eq!(clean_fenced_json_output("```plain```"), "```plain");
    }
}
