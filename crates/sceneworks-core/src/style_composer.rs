//! Rust port of the web Style composer (`apps/web/src/styleComposer.js`, sc-13129),
//! so a headless/MCP job that carries a `styleId` + a raw prompt is folded into the
//! SAME `Style:`/`Description:` composition the web app sends (sc-13134).
//!
//! This is a byte-for-byte port of `composeStyledPrompt`: the directive-collision splice
//! (line-anchored directive detection over a fixed recognized key set; the caller's own
//! `Style:` line MERGES — catalog text first, the user's words appended after `", "`; every
//! other recognized directive is kept as a top-level sibling; free text folds into a single
//! `Description:` block; CRLF/lone-CR safe; identity when no style is selected).
//!
//! Cross-language parity is pinned by the shared golden fixtures in
//! `documents/style-composer.fixtures.json`, exercised by both this module's test and
//! `apps/web/src/styleComposer.test.js`, so the two implementations can never drift.

/// The recognized directive keys that seed the parser. A line only counts as a directive
/// when its line-anchored key is one of these — identical to the JS `STYLE_DIRECTIVE_KEYS`.
pub const STYLE_DIRECTIVE_KEYS: [&str; 10] = [
    "Style",
    "Description",
    "Setting",
    "Environment",
    "Angle",
    "Lighting",
    "Camera",
    "Mood",
    "Composition",
    "Negative",
];

const STYLE_KEY: &str = "Style";
const DESCRIPTION_KEY: &str = "Description";

/// Longest recognized key we treat as a directive (mirrors the JS guard). Redundant with the
/// set membership below, but keeps the structural rule explicit.
const MAX_DIRECTIVE_KEY_LENGTH: usize = 20;

/// A classified prompt line: a recognized `Directive { key, content, raw }` or free `Prose`.
enum Line {
    Directive {
        key: String,
        content: String,
        raw: String,
    },
    Prose {
        content: String,
    },
}

/// Split the line-anchored directive shape the JS regex encodes:
/// `^\s*([A-Z][A-Za-z]*(?:\s+[A-Za-z]+){0,2}):(?:[ \t]+(.*\S))?[ \t]*$`.
///
/// Returns `Some((key, content))` when `raw` is a directive line (whatever its key), else
/// `None`. Set-membership + length are checked by the caller, exactly as the JS does after the
/// regex matches. Ported by hand (no regex dependency in this crate); see the module test's
/// golden fixtures for the equivalence proof.
fn match_directive_line(raw: &str) -> Option<(String, String)> {
    // `^\s*`: greedy leading-whitespace skip. `char::is_whitespace` matches the JS `\s` class
    // closely enough for prompt text (the only leading whitespace in practice is spaces/tabs).
    let s = raw.trim_start();
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return None;
    }
    // Word 1: `[A-Z][A-Za-z]*` — must start uppercase ASCII, then ASCII letters.
    if !chars[0].is_ascii_uppercase() {
        return None;
    }
    let mut i = 1;
    while i < chars.len() && chars[i].is_ascii_alphabetic() {
        i += 1;
    }
    // Up to 2 more words: each `\s+[A-Za-z]+`. A colon can never sit inside a word (words are
    // ASCII letters only), so greedily consuming maximal words never overshoots the terminating
    // colon — no backtracking is needed to place it (see the port notes / fixtures).
    let mut words = 1;
    while words < 3 {
        // `\s+`
        let mut j = i;
        let mut saw_ws = false;
        while j < chars.len() && chars[j].is_whitespace() {
            j += 1;
            saw_ws = true;
        }
        // `[A-Za-z]+`
        if !saw_ws || j >= chars.len() || !chars[j].is_ascii_alphabetic() {
            break;
        }
        while j < chars.len() && chars[j].is_ascii_alphabetic() {
            j += 1;
        }
        i = j;
        words += 1;
    }
    // The key must be immediately followed by the terminating colon.
    if i >= chars.len() || chars[i] != ':' {
        return None;
    }
    let key: String = chars[..i].iter().collect();
    // Everything after the colon: `(?:[ \t]+(.*\S))?[ \t]*$`.
    let rest: Vec<char> = chars[i + 1..].to_vec();
    // Count the leading `[ \t]+` separator (space/tab only, matching the JS character class).
    let sep = rest
        .iter()
        .take_while(|c| **c == ' ' || **c == '\t')
        .count();
    let after_sep = &rest[sep..];
    let all_space_tab = after_sep.iter().all(|c| *c == ' ' || *c == '\t');
    let content = if after_sep.is_empty() || all_space_tab {
        // Optional content group is skipped; the remainder is all `[ \t]*` (or empty).
        String::new()
    } else {
        // A non-space char is present, so the optional group must match — which requires at
        // least one `[ \t]` separator. Without it the whole line is prose (e.g. `Style:x`).
        if sep == 0 {
            return None;
        }
        // `(.*\S)` = the remainder right-trimmed of trailing spaces/tabs.
        let end = after_sep
            .iter()
            .rposition(|c| *c != ' ' && *c != '\t')
            .map(|p| p + 1)
            .unwrap_or(0);
        after_sep[..end].iter().collect()
    };
    Some((key, content))
}

/// Split `text` into classified lines (directive vs prose), matching `parseDirectiveLines`.
/// CRLF (`\r\n`) and lone-CR (`\r`) line breaks split the same as LF so a trailing `\r` never
/// leaks into content or breaks the anchored match.
fn parse_directive_lines(text: &str) -> Vec<Line> {
    // Normalize `\r\n` and lone `\r` to `\n`, then split on `\n` — equivalent to the JS
    // `split(/\r\n?|\n/)`.
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    normalized
        .split('\n')
        .map(|raw| match match_directive_line(raw) {
            Some((key, content))
                if key.chars().count() <= MAX_DIRECTIVE_KEY_LENGTH
                    && STYLE_DIRECTIVE_KEYS.contains(&key.as_str()) =>
            {
                Line::Directive {
                    key,
                    content,
                    raw: raw.trim().to_owned(),
                }
            }
            _ => Line::Prose {
                content: raw.to_owned(),
            },
        })
        .collect()
}

/// Compose the outgoing prompt from a catalog-selected style text and the user's
/// (already preset-folded) prompt — the Rust port of `composeStyledPrompt`.
///
/// Returns the untouched `user_prompt` when `style_text` is empty/whitespace (identity, "no
/// style selected"), else the spliced `Style:`/siblings/`Description:` composition.
pub fn compose_styled_prompt(style_text: &str, user_prompt: &str) -> String {
    let style = style_text.trim();
    if style.is_empty() {
        return user_prompt.to_owned();
    }

    let parsed = parse_directive_lines(user_prompt);

    // The user's own `Style:` content merges into our block; every other directive is a sibling.
    let mut user_style_parts: Vec<String> = Vec::new();
    let mut sibling_directives: Vec<String> = Vec::new();
    let mut prose_lines: Vec<String> = Vec::new();
    for line in parsed {
        match line {
            Line::Directive { key, content, raw } => {
                if key == STYLE_KEY {
                    if !content.trim().is_empty() {
                        user_style_parts.push(content.trim().to_owned());
                    }
                } else {
                    sibling_directives.push(raw);
                }
            }
            Line::Prose { content } => prose_lines.push(content),
        }
    }

    let mut style_content = String::from(style);
    for part in &user_style_parts {
        style_content.push_str(", ");
        style_content.push_str(part);
    }

    let mut blocks: Vec<String> = vec![format!("{STYLE_KEY}: {style_content}")];
    blocks.extend(sibling_directives);

    let description = prose_lines.join("\n");
    let description = description.trim();
    if !description.is_empty() {
        blocks.push(format!("{DESCRIPTION_KEY}: {description}"));
    }

    blocks.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// The shared cross-language golden fixtures — the SAME file `apps/web/src/styleComposer.test.js`
    /// reads. Embedding it here (repo-root-relative, like `builtin_manifests`) proves this Rust port
    /// and the JS composer emit byte-identical output for every case, so the server-side fold matches
    /// the web path exactly.
    const FIXTURES: &str = include_str!("../../../documents/style-composer.fixtures.json");

    #[test]
    fn golden_fixtures_match_the_js_composer() {
        let root: Value = serde_json::from_str(FIXTURES).expect("fixtures parse as JSON");
        let cases = root
            .get("cases")
            .and_then(Value::as_array)
            .expect("fixtures carry a `cases` array");
        assert!(
            cases.len() >= 20,
            "expected a non-trivial fixture set, got {}",
            cases.len()
        );
        for case in cases {
            let name = case
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("<unnamed>");
            // A null styleText models "no style selected" (JS undefined) — the composer treats an
            // empty style as identity, so mapping it to "" is faithful.
            let style_text = case.get("styleText").and_then(Value::as_str).unwrap_or("");
            let user_prompt = case
                .get("userPrompt")
                .and_then(Value::as_str)
                .expect("fixture userPrompt is a string");
            let expected = case
                .get("expected")
                .and_then(Value::as_str)
                .expect("fixture expected is a string");
            assert_eq!(
                compose_styled_prompt(style_text, user_prompt),
                expected,
                "golden fixture mismatch: {name}"
            );
        }
    }

    #[test]
    fn empty_style_is_identity() {
        assert_eq!(compose_styled_prompt("", "a fox"), "a fox");
        assert_eq!(compose_styled_prompt("   \n\t", "a fox"), "a fox");
    }

    #[test]
    fn style_only_when_prompt_is_blank() {
        assert_eq!(compose_styled_prompt("cinematic", ""), "Style: cinematic");
        assert_eq!(
            compose_styled_prompt("cinematic", "  \n "),
            "Style: cinematic"
        );
    }

    #[test]
    fn no_space_after_colon_is_prose_not_a_directive() {
        // `Setting:x` fails the JS `[ \t]+` separator requirement, so it stays prose and folds
        // into Description — discriminates the separator rule the port must preserve.
        assert_eq!(
            compose_styled_prompt("noir", "Setting:x"),
            "Style: noir\nDescription: Setting:x"
        );
    }
}
