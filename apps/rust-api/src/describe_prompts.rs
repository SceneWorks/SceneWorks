// The describing prompt for the dataset prompt-batch describe job (SW-7085, feature SW-F362).
//
// Training Studio's "Create Prompt Batch from Dataset" dialog prefills its editable prompt with the
// worker's default instruction before the first run, so the API serves that text. ONE source of
// truth: the text is the `[SYSTEM]` section of the worker's own prompt asset, embedded here with
// `include_str!` from the worker crate's source tree, so the default the dialog shows and the
// default the worker describes with are the same bytes. The asset stays where the worker owns it;
// the API only reads it. The `[OUTPUT]` JSON contract is never served: the worker always appends
// it, and the user never edits it.

use super::*;
use sceneworks_core::dataset_batch_describe::MAX_DESCRIBE_PROMPT_CHARS;

const IMAGE_BATCH_DESCRIBE_ASSET: &str =
    include_str!("../../../crates/sceneworks-worker/src/image_batch_describe_v1.txt");

/// One `[NAME]` section of a prompt asset, trimmed. Same marker rule as the worker's
/// `magic_section_from`: a marker is a whole line `[WORD]` with no spaces, and a section runs to the
/// next marker.
fn asset_section(source: &str, name: &str) -> String {
    let mut capturing = false;
    let mut body: Vec<&str> = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        let is_marker = trimmed.len() >= 2
            && trimmed.starts_with('[')
            && trimmed.ends_with(']')
            && !trimmed.contains(' ');
        if is_marker {
            if capturing {
                break;
            }
            capturing = trimmed[1..trimmed.len() - 1].eq_ignore_ascii_case(name);
            continue;
        }
        if capturing {
            body.push(line);
        }
    }
    body.join("\n").trim().to_owned()
}

/// The default editable describing instruction (the asset's `[SYSTEM]` section).
pub(crate) fn default_describe_instruction() -> String {
    asset_section(IMAGE_BATCH_DESCRIBE_ASSET, "SYSTEM")
}

/// `GET /api/v1/training/describe-prompt/default`: the default describing instruction and the
/// longest instruction a describe job accepts.
pub(crate) async fn get_default_describe_prompt() -> Json<Value> {
    Json(json!({
        "prompt": default_describe_instruction(),
        "maxChars": MAX_DESCRIBE_PROMPT_CHARS,
    }))
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn default_instruction_is_the_system_section_without_the_output_contract() {
        let prompt = default_describe_instruction();
        assert!(prompt.starts_with("You are a meticulous scene describer."));
        assert!(prompt.contains("## LIGHTING"));
        assert!(
            !prompt.contains("## OUTPUT"),
            "the JSON contract is never editable"
        );
        assert!(!prompt.contains("[OUTPUT]") && !prompt.contains("[META]"));
        assert!(prompt.chars().count() <= MAX_DESCRIBE_PROMPT_CHARS);
    }

    #[test]
    fn asset_section_stops_at_the_next_marker() {
        let source = "[A]\none\n[B]\ntwo\n  three  \n[C]\nfour";
        assert_eq!(asset_section(source, "B"), "two\n  three");
        assert_eq!(asset_section(source, "missing"), "");
    }
}
