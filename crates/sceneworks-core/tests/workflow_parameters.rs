//! The A1111 `parameters` chunk, measured against a reader rather than against itself (sc-15957,
//! epic 15945).
//!
//! # Why the parser below exists
//!
//! The point of this chunk is that software we do not control can display it. A test that asserts
//! our renderer produces the string our renderer produces proves nothing about that. So the corpus
//! here goes through [`parse_a1111`], a faithful reimplementation of AUTOMATIC1111's
//! `modules/generation_parameters_copypaste.py::parse_generation_parameters` — the function Civitai
//! and the rest of the ecosystem's parsers are modelled on:
//!
//! * the text is split on newlines and the LAST line is the settings line, **but only when it holds
//!   at least three `Key: value` pairs**; otherwise it is prompt text. That threshold is A1111's,
//!   not ours, and it is the reason `the_settings_line_clears_the_pair_floor_a_gallery_parses_with`
//!   exists;
//! * a line beginning `Negative prompt:` ends the prompt and starts the negative;
//! * the settings line is scanned for `key: value` pairs where the value is either a JSON string in
//!   double quotes or everything up to the next comma — the semantics of A1111's `re_param_code`.
//!
//! # And why PIL is also here
//!
//! The parser proves the TEXT is legible. It says nothing about whether a reader finds the chunk at
//! all, which is a question about `tEXt` versus `iTXt` and about compression flags.
//! [`pil_agrees_with_our_encoding_choice`] answers that against the real
//! `PIL.PngImagePlugin`, which is what A1111 itself writes through and what third-party parsers are
//! tested against. It skips loudly when Python or Pillow is not installed rather than failing the
//! build on a machine that has neither.
//!
//! # What is NOT proven here
//!
//! A live upload to Civitai. That acceptance criterion is a publication to a third-party service and
//! is deliberately left for a human; see the story and the PR. Everything below is the offline half.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

use image::{Rgb, RgbImage};
use sceneworks_core::contracts::{Asset, JsonObject};
use sceneworks_core::workflow_parameters::{
    parameters_text, NEGATIVE_PROMPT_LABEL, PARAMETERS_CHUNK_KEYWORD, SETTINGS_FIELDS,
    SETTINGS_PAIR_FLOOR,
};
use sceneworks_core::workflow_png::{
    read_workflow_chunk_file, strip_workflow_chunk, write_workflow_chunk, WORKFLOW_CHUNK_KEYWORD,
};
use sceneworks_core::workflow_share::{
    build_workflow_share, build_workflow_share_from, WorkflowAssetFacts, WorkflowShare,
    PRODUCER_VERSION,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// A faithful A1111 / gallery parser
// ---------------------------------------------------------------------------

/// What a gallery gets out of a `parameters` block.
#[derive(Debug, Default, PartialEq, Eq)]
struct Parsed {
    prompt: String,
    negative_prompt: String,
    /// The trailing settings line, key to value, unquoted.
    params: BTreeMap<String, String>,
}

/// A1111's `parse_generation_parameters`, reimplemented.
///
/// Deliberately not a loose "find the words I expect" scan: the failure this is here to catch is a
/// renderer that produces something *we* can read back and a gallery cannot, and only a faithful
/// parser can see that. The three rules it enforces — the three-pair floor on the last line, the
/// `Negative prompt:` boundary, and comma-splitting with JSON-quoted escapes — are the three places
/// our renderer could break a reader without breaking itself.
fn parse_a1111(text: &str) -> Parsed {
    let trimmed = text.trim();
    let mut lines: Vec<&str> = trimmed.split('\n').collect();
    // A1111 pops the last line and puts it back when it does not look like a settings line.
    let last = lines.pop().unwrap_or_default();
    let settings = if parse_params(last).len() >= 3 {
        parse_params(last)
    } else {
        lines.push(last);
        BTreeMap::new()
    };

    let mut prompt = String::new();
    let mut negative = String::new();
    let mut in_negative = false;
    for line in lines {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix(NEGATIVE_PROMPT_LABEL.trim_end()) {
            in_negative = true;
            push_line(&mut negative, rest.trim_start());
            continue;
        }
        if in_negative {
            push_line(&mut negative, line);
        } else {
            push_line(&mut prompt, line);
        }
    }

    Parsed {
        prompt,
        negative_prompt: negative,
        params: settings,
    }
}

fn push_line(buffer: &mut String, line: &str) {
    if !buffer.is_empty() {
        buffer.push('\n');
    }
    buffer.push_str(line);
}

/// The `key: value` pairs on one line, with A1111's quoting rule applied in reverse.
///
/// Mirrors `re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'`: a key of word
/// characters, spaces, hyphens and slashes; a value that is either a double-quoted JSON string
/// (backslash escapes honoured) or a run of anything up to the next comma.
fn parse_params(line: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let bytes: Vec<char> = line.chars().collect();
    let mut index = 0;
    while index < bytes.len() {
        while index < bytes.len() && (bytes[index] == ' ' || bytes[index] == ',') {
            index += 1;
        }
        let key_start = index;
        while index < bytes.len()
            && (bytes[index].is_alphanumeric() || matches!(bytes[index], '_' | ' ' | '-' | '/'))
        {
            index += 1;
        }
        // A key must be followed by a colon and must not be empty or start with a space.
        if index >= bytes.len() || bytes[index] != ':' || index == key_start {
            // Not a parameter line at this position — give up on the rest, as the regex would by
            // simply not matching here.
            return out;
        }
        let key: String = bytes[key_start..index].iter().collect();
        if key.starts_with(' ') {
            return out;
        }
        index += 1; // the colon
        while index < bytes.len() && bytes[index] == ' ' {
            index += 1;
        }
        let value = if index < bytes.len() && bytes[index] == '"' {
            let start = index;
            index += 1;
            while index < bytes.len() {
                if bytes[index] == '\\' {
                    index += 2;
                    continue;
                }
                if bytes[index] == '"' {
                    index += 1;
                    break;
                }
                index += 1;
            }
            let raw: String = bytes[start..index.min(bytes.len())].iter().collect();
            serde_json::from_str::<String>(&raw).unwrap_or(raw)
        } else {
            let start = index;
            while index < bytes.len() && bytes[index] != ',' {
                index += 1;
            }
            bytes[start..index]
                .iter()
                .collect::<String>()
                .trim_end()
                .to_owned()
        };
        out.insert(key, value);
    }
    out
}

#[test]
fn the_local_parser_reads_a_real_a1111_block() {
    // The parser is the measuring instrument, so it is calibrated against a block A1111 itself
    // emits before anything is measured with it. Copied in the shape the WebUI writes: multi-line
    // prompt, a negative, a settings line with quoted and unquoted values.
    let block = "a lighthouse in heavy fog\nsecond line of the prompt\nNegative prompt: text, \
                 watermark\nSteps: 28, Sampler: \"DPM++ 2M, Karras\", CFG scale: 7, Seed: 12345, \
                 Size: 512x768, Model hash: abc123, Model: sd_xl_base, Version: v1.9.4";
    let parsed = parse_a1111(block);
    assert_eq!(
        parsed.prompt,
        "a lighthouse in heavy fog\nsecond line of the prompt"
    );
    assert_eq!(parsed.negative_prompt, "text, watermark");
    assert_eq!(parsed.params["Steps"], "28");
    assert_eq!(parsed.params["Sampler"], "DPM++ 2M, Karras");
    assert_eq!(parsed.params["CFG scale"], "7");
    assert_eq!(parsed.params["Size"], "512x768");
    assert_eq!(parsed.params["Version"], "v1.9.4");

    // And the three-pair floor, which is a real behaviour and not an implementation detail: a
    // trailing line with two pairs is PROMPT to A1111, not settings.
    let thin = parse_a1111("just a prompt\nModel: z_image_turbo, Version: 0.8.1");
    assert!(thin.params.is_empty(), "{thin:?}");
    assert!(thin.prompt.ends_with("Version: 0.8.1"));

    // The renderer's `SETTINGS_PAIR_FLOOR` is that same threshold, and this is where the two are
    // joined. The parser above keeps its own literal on purpose — one that read the renderer's
    // constant would agree with it whatever it was set to, which is the opposite of a measurement.
    assert_eq!(
        SETTINGS_PAIR_FLOOR, 3,
        "the renderer's floor and the reader's threshold have diverged; one of them is now wrong \
         about what a gallery does with a thin line"
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn rgb_fixture(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    })
}

fn asset_fixture(prompt: &str) -> Asset {
    serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "asset_9f2c",
        "projectId": "project_7a10",
        "generationSetId": "genset_31bb",
        "type": "image",
        "displayName": "Parameters #1",
        "createdAt": "2026-07-30T13:04:11Z",
        "file": {
            "path": "assets/images/2026-07-30_z_image_turbo_parameters_0001.png",
            "mimeType": "image/png",
            "width": 64,
            "height": 48,
            "duration": null,
            "fps": null
        },
        "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
        "recipe": {
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": prompt,
            "negativePrompt": "",
            "seed": 880_412,
            "loras": [],
            "stylePreset": null,
            "normalizedSettings": {},
            "rawAdapterSettings": {}
        },
        "lineage": { "parents": [], "sourceAssetId": null, "sourceTimestamp": null, "jobId": "job_1" }
    }))
    .expect("asset fixture parses")
}

/// A realistic Image Studio payload, with `extra` merged over it.
fn payload_fixture(prompt: &str, negative: &str, extra: Value) -> JsonObject {
    let mut payload = json!({
        "projectId": "project_7a10",
        "mode": "text_to_image",
        "prompt": prompt,
        "negativePrompt": negative,
        "model": "z_image_turbo",
        "count": 1,
        "width": 64,
        "height": 48,
        "advanced": { "steps": 28, "sampler": "euler", "guidanceScale": 3.5 }
    });
    let map = payload.as_object_mut().expect("an object");
    for (key, value) in extra.as_object().expect("extra is an object") {
        map.insert(key.clone(), value.clone());
    }
    map.clone()
}

fn built(prompt: &str, negative: &str, extra: Value) -> WorkflowShare {
    build_workflow_share(
        &asset_fixture(prompt),
        &payload_fixture(prompt, negative, extra),
    )
}

// ---------------------------------------------------------------------------
// Reading text chunks back out of a PNG, by hand
// ---------------------------------------------------------------------------

/// One text chunk as it sits in a file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TextChunk {
    /// `iTXt`, `tEXt` or `zTXt`.
    kind: String,
    keyword: String,
    /// `true` only for an `iTXt` whose compression flag is set.
    compressed: bool,
    /// The text, for the uncompressed cases. Empty for a compressed chunk, which this walker does
    /// not inflate — the envelope is read back through the real reader instead.
    text: String,
}

/// Every text chunk in `png`, in file order, walked without the `png` crate.
///
/// By hand on purpose: what is under test includes which CHUNK TYPE the text landed in, and a
/// decoder that normalizes `tEXt` and `iTXt` into one text map would hide exactly that.
fn text_chunks(png: &[u8]) -> Vec<TextChunk> {
    let mut out = Vec::new();
    let mut cursor = 8;
    while cursor + 8 <= png.len() {
        let length = usize::try_from(u32::from_be_bytes(
            png[cursor..cursor + 4].try_into().expect("a length field"),
        ))
        .expect("fits usize");
        let kind = &png[cursor + 4..cursor + 8];
        let end = cursor + 12 + length;
        assert!(end <= png.len(), "chunk at {cursor} runs past the end");
        let data = &png[cursor + 8..cursor + 8 + length];
        if kind == b"iTXt" || kind == b"tEXt" || kind == b"zTXt" {
            let mut parts = data.splitn(2, |byte| *byte == 0);
            let keyword = String::from_utf8_lossy(parts.next().unwrap_or_default()).into_owned();
            let rest = parts.next().unwrap_or_default();
            let (compressed, text) = if kind == b"iTXt" {
                // flag, method, language NUL, translated keyword NUL, text.
                let compressed = rest.first().copied().unwrap_or(0) == 1;
                let after_method = rest.get(2..).unwrap_or_default();
                let mut fields = after_method.splitn(3, |byte| *byte == 0);
                fields.next();
                fields.next();
                let body = fields.next().unwrap_or_default();
                (
                    compressed,
                    if compressed {
                        String::new()
                    } else {
                        String::from_utf8_lossy(body).into_owned()
                    },
                )
            } else if kind == b"tEXt" {
                // Latin-1 by spec; every byte is one character.
                (false, rest.iter().map(|byte| *byte as char).collect())
            } else {
                (true, String::new())
            };
            out.push(TextChunk {
                kind: String::from_utf8_lossy(kind).into_owned(),
                keyword,
                compressed,
                text,
            });
        }
        cursor = end;
    }
    out
}

fn parameters_chunk_of(png: &[u8]) -> TextChunk {
    text_chunks(png)
        .into_iter()
        .find(|chunk| chunk.keyword == PARAMETERS_CHUNK_KEYWORD)
        .expect("the file carries a parameters chunk")
}

/// Write one embedded PNG and hand back its bytes.
fn embedded_png(share: &WorkflowShare, size: (u32, u32)) -> (PathBuf, Vec<u8>, tempfile::TempDir) {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("embedded.png");
    write_workflow_chunk(&rgb_fixture(size.0, size.1), &path, Some(share)).expect("writes");
    let bytes = fs::read(&path).expect("reads");
    (path, bytes, directory)
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len() && haystack.windows(needle.len()).any(|w| w == needle)
}

// ---------------------------------------------------------------------------
// The round trip a gallery makes
// ---------------------------------------------------------------------------

#[test]
fn a_generated_png_round_trips_through_a_gallery_parser() {
    // The AC, stated the way the story states it: a real generated PNG, read the way Civitai reads
    // one — walk the chunks, take the `parameters` text, split on the `Negative prompt:` line and
    // the trailing comma-separated key/value line.
    let share = built("a lighthouse in heavy fog", "text, watermark", json!({}));
    let (_path, bytes, _dir) = embedded_png(&share, (64, 48));

    let chunk = parameters_chunk_of(&bytes);
    let parsed = parse_a1111(&chunk.text);

    assert_eq!(parsed.prompt, "a lighthouse in heavy fog");
    assert_eq!(parsed.negative_prompt, "text, watermark");
    assert_eq!(parsed.params["Steps"], "28");
    assert_eq!(parsed.params["Sampler"], "euler");
    assert_eq!(parsed.params["CFG scale"], "3.5");
    assert_eq!(parsed.params["Seed"], "880412");
    assert_eq!(parsed.params["Size"], "64x48");
    assert_eq!(parsed.params["Model"], "z_image_turbo");
    assert_eq!(parsed.params["Version"], PRODUCER_VERSION);

    // The recipe chunk is still there and still the authoritative one — the second chunk is a
    // display convenience and does not replace the thing a SceneWorks install reads back.
    assert!(
        contains(&bytes, WORKFLOW_CHUNK_KEYWORD.as_bytes()),
        "the envelope chunk must ride alongside"
    );
}

#[test]
fn the_version_is_the_producer_version_exactly() {
    // The AC names this one directly: the two chunks cannot disagree about which build wrote the
    // file, so `Version:` is fed from the envelope's own `producer.version`.
    let share = built("a lighthouse", "", json!({}));
    let (path, bytes, _dir) = embedded_png(&share, (64, 48));
    let parsed = parse_a1111(&parameters_chunk_of(&bytes).text);

    let envelope = read_workflow_chunk_file(&path)
        .expect("reads")
        .expect("carries a workflow");
    assert_eq!(parsed.params["Version"], envelope.producer.version);
    assert_eq!(parsed.params["Version"], PRODUCER_VERSION);
}

/// The envelope shape one of the worker's derived-pass write seams produces.
///
/// Built through the public `build_workflow_share_from` rather than through the worker's own
/// helpers, because those are `pub(crate)` in a crate this one does not depend on — and because what
/// is under test is the SHAPE (which facts are present, which are empty, whether the geometry is the
/// encoded file's), which is exactly what the facts struct carries.
fn lane_share(facts: WorkflowAssetFacts, payload: Value) -> WorkflowShare {
    build_workflow_share_from(&facts, payload.as_object().expect("a payload object"))
}

#[test]
fn the_settings_line_clears_the_pair_floor_a_gallery_parses_with() {
    // A1111 treats a trailing line of fewer than three `Key: value` pairs as PROMPT text, and
    // `SETTINGS_PAIR_FLOOR` withholds the line rather than emitting one below it. So there are two
    // things to pin, and the version of this test that shipped first pinned neither well: it
    // measured two `text_to_image` fixtures that both emit SIX pairs, which is not where the cliff
    // is.
    //
    // The lanes below are the ones that actually sit at or near the floor — the derived passes,
    // whose envelopes are deliberately thin. `standalone_upscale_workflow_share` is the extreme: no
    // prompt, no `advanced`, source geometry that is not the encoded file's, and the engine id
    // standing in for the model. It clears the floor by exactly nothing.
    //
    // Each lane is checked through `parse_a1111` — the gallery's own reader — rather than by
    // counting pairs on `lines().last()`, because when the floor guard withholds a line there may be
    // no last line at all, and "the settings fell into the prompt" is precisely the failure worth
    // catching.
    let lanes: [(&str, WorkflowShare, (u32, u32), &str); 4] = [
        (
            "an ordinary text_to_image generation",
            built("a lighthouse", "", json!({})),
            (64, 48),
            "a lighthouse",
        ),
        (
            "a generation with no advanced settings at all",
            built(
                "a lighthouse",
                "",
                json!({ "advanced": Value::Object(serde_json::Map::new()) }),
            ),
            (64, 48),
            "a lighthouse",
        ),
        (
            // `standalone_upscale_workflow_share` in the worker: mode `image_upscale`, the engine
            // as the model, no prompt at all, and the SOURCE geometry — so `Size` is correctly
            // withheld from a file twice that size. Seed + Model + Version and nothing else.
            "the standalone upscale lane",
            lane_share(
                WorkflowAssetFacts {
                    mode: "image_upscale".to_owned(),
                    model: "seedvr2".to_owned(),
                    prompt: String::new(),
                    negative_prompt: String::new(),
                    seed: 880_412,
                    width: Some(1024),
                    height: Some(1024),
                },
                json!({ "projectId": "project_7a10", "sourceAssetId": "asset_9f2c",
                        "upscale": { "enabled": true, "engine": "seedvr2", "factor": 2 } }),
            ),
            (2048, 2048),
            "",
        ),
        (
            // `detail_workflow_share`: its own job, its own payload, and the Detail UI exposes only
            // `strength` and `cnScale` — neither of which maps to an A1111 field. So `advanced`
            // travels full and contributes nothing to the line.
            "the detail lane",
            lane_share(
                WorkflowAssetFacts {
                    mode: "image_detail".to_owned(),
                    model: "sdxl_base".to_owned(),
                    prompt: "sharper rigging".to_owned(),
                    negative_prompt: String::new(),
                    seed: 4,
                    width: Some(1536),
                    height: Some(1536),
                },
                json!({ "projectId": "project_7a10", "sourceAssetId": "asset_9f2c",
                        "advanced": { "strength": 0.4, "cnScale": 0.8 } }),
            ),
            (1536, 1536),
            "sharper rigging",
        ),
    ];

    // The two derived lanes have to be thin or this pins the wrong thing: the upscale lane carries
    // no prompt and no `advanced` at all, and the detail lane's two knobs both travel and both map
    // to nothing on the line. Checked, because a fixture that quietly grew a mapped field would
    // leave the floor untested again while still passing.
    assert!(
        lanes[2].1.prompt.is_empty() && lanes[2].1.advanced.is_empty(),
        "the upscale fixture is no longer the thin shape the seam produces: {:?}",
        lanes[2].1.advanced
    );
    assert_eq!(
        lanes[3].1.advanced.len(),
        2,
        "the detail fixture must carry both allow-listed knobs and nothing that maps: {:?}",
        lanes[3].1.advanced
    );
    // And the sharpest statement of where the cliff is: the upscale lane clears the floor by
    // NOTHING. Stated as an equality so a change that removes a mapped field fails here rather than
    // silently reducing the margin from zero to negative.
    assert_eq!(
        parse_a1111(&parameters_text(&lanes[2].1, lanes[2].2))
            .params
            .len(),
        SETTINGS_PAIR_FLOOR,
        "the standalone upscale lane is the one that sits ON the floor; if it no longer does, the \
         lane that does is the one this test should be pinning"
    );

    for (label, share, encoded, prompt) in lanes {
        let text = parameters_text(&share, encoded);
        let parsed = parse_a1111(&text);
        assert!(
            parsed.params.len() >= SETTINGS_PAIR_FLOOR,
            "on {label} a gallery reads {} pairs off {text:?} — fewer than the \
             {SETTINGS_PAIR_FLOOR} A1111 needs, so the whole line is displayed as part of the prompt",
            parsed.params.len()
        );
        assert_eq!(
            parsed.prompt, prompt,
            "on {label} the settings line fell into the prompt: {text:?}"
        );
    }
}

#[test]
fn a_lane_that_cannot_reach_the_floor_withholds_the_line_instead_of_leaking_it_into_the_prompt() {
    // The other half, and the reason the pin above is worth extending rather than just widening.
    // The upscale lane clears the floor by ZERO margin, so any envelope that loses one of its three
    // fields falls off the cliff — and the renderer used to emit the two-pair remainder, which every
    // gallery displays as prompt text.
    //
    // Reachable? Not through today's catalog: `model` here is an upscaler engine id, and no id in it
    // is empty, over 200 characters or path-shaped, which are the three ways the sanitizer drops a
    // label. So this is latent rather than live — which is exactly the kind of thing that becomes
    // live when someone adds an engine whose id happens to look like a path.
    let stripped = lane_share(
        WorkflowAssetFacts {
            mode: "image_upscale".to_owned(),
            // Three segments, so `is_path_shaped` catches it and `shareable_label` withholds it —
            // two would be a Hugging Face repo id and would survive.
            model: "engines/upscalers/seedvr2".to_owned(),
            prompt: "a lighthouse".to_owned(),
            negative_prompt: String::new(),
            seed: 880_412,
            width: Some(1024),
            height: Some(1024),
        },
        json!({ "projectId": "project_7a10" }),
    );
    assert!(
        stripped.model.is_empty(),
        "the fixture must actually lose its model or it proves nothing: {:?}",
        stripped.model
    );

    let text = parameters_text(&stripped, (2048, 2048));
    assert_eq!(
        text, "a lighthouse",
        "two mapped fields is a line a gallery reads as prompt, so the line must be withheld whole \
         — the prompt stays correct and the settings are simply absent"
    );
    let parsed = parse_a1111(&text);
    assert_eq!(parsed.prompt, "a lighthouse");
    assert!(parsed.params.is_empty());
    // And nothing was lost that mattered: the authoritative recipe is the other chunk, which still
    // carries the seed and the producer version this line could not state legibly.
    assert_eq!(stripped.seed, Some(880_412));
    assert_eq!(stripped.producer.version, PRODUCER_VERSION);
}

// ---------------------------------------------------------------------------
// Sanitization applies identically
// ---------------------------------------------------------------------------

#[test]
fn the_parameters_chunk_cannot_carry_a_withheld_field() {
    // The AC: content derives from the sanitized envelope, so a field the sc-15946 allow-list
    // withholds is absent from BOTH chunks. Seeded into the RAW payload, which is the only place a
    // second, laxer channel could have picked it up from.
    //
    // The search is over the whole file's BYTES, which is what makes it a real test of the second
    // chunk: `parameters` is written uncompressed, so a leaked value would sit in the file in
    // plain text and a byte search finds it. The envelope chunk is deflated, so its half is
    // asserted against the parsed envelope instead.
    let secrets = [
        ("advanced.quantTier", "q4-secret-tier"),
        ("advanced.controlImage", "asset_secret_control"),
        ("advanced.recipePresetId", "preset_secret_id"),
    ];
    let share = built(
        "a lighthouse",
        "",
        json!({
            "projectId": "project_secret_id",
            "jobId": "job_secret_id",
            "seeds": [1, 2, 3],
            "advanced": {
                "steps": 28,
                "sampler": "euler",
                "guidanceScale": 3.5,
                "quantTier": "q4-secret-tier",
                "controlImage": "asset_secret_control",
                "recipePresetId": "preset_secret_id",
                "flashAttn": true
            }
        }),
    );
    let (path, bytes, _dir) = embedded_png(&share, (64, 48));
    let rendered = parameters_chunk_of(&bytes).text;
    let envelope = serde_json::to_string(
        &read_workflow_chunk_file(&path)
            .expect("reads")
            .expect("carries a workflow"),
    )
    .expect("serializes");

    for (key, secret) in secrets {
        assert!(
            !rendered.contains(secret),
            "{key} reached the `parameters` chunk: {rendered:?}"
        );
        assert!(
            !contains(&bytes, secret.as_bytes()),
            "{key} is in the file's raw bytes — the withheld value leaked into a chunk"
        );
        assert!(
            !envelope.contains(secret),
            "{key} reached the SceneWorks envelope, which is the sc-15946 sanitizer's job"
        );
    }
    // Project and job identity, which the envelope has no slot for at all.
    for secret in ["project_secret_id", "job_secret_id"] {
        assert!(
            !contains(&bytes, secret.as_bytes()),
            "{secret} is in the file"
        );
    }
    // And the fixture is not vacuous: the things that SHOULD travel did.
    assert!(rendered.contains("Sampler: euler") && rendered.contains("Steps: 28"));
}

#[test]
fn the_setting_governs_both_chunks() {
    // sc-15953's switch is a `None` at this seam, and both chunks hang off the same `Some`. Off
    // means neither is written — there is no arrangement in which the prompt rides out in the
    // A1111 block after the user turned embedding off.
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("opted-out.png");
    write_workflow_chunk(&rgb_fixture(64, 48), &path, None).expect("writes");
    let bytes = fs::read(&path).expect("reads");

    assert!(text_chunks(&bytes).is_empty(), "{:?}", text_chunks(&bytes));
    for keyword in [WORKFLOW_CHUNK_KEYWORD, PARAMETERS_CHUNK_KEYWORD] {
        assert!(
            !contains(&bytes, keyword.as_bytes()),
            "the opted-out file carries `{keyword}`"
        );
    }
    assert!(
        !contains(&bytes, b"a lighthouse"),
        "the prompt is in a file written with embedding off"
    );
}

// ---------------------------------------------------------------------------
// Encoding: tEXt when ASCII, iTXt when not
// ---------------------------------------------------------------------------

/// The three scripts that break Latin-1, plus a Latin-1-representable case that is the one band
/// where our rule and PIL's differ.
const NON_ASCII_PROMPT: &str = "霧の中の灯台、シネマティック 🌊🗼 café naïve";

#[test]
fn an_ascii_block_is_a_text_chunk_and_a_non_ascii_one_is_an_itxt() {
    let ascii = built("a lighthouse in heavy fog", "text, watermark", json!({}));
    let (_p, bytes, _d) = embedded_png(&ascii, (64, 48));
    let chunk = parameters_chunk_of(&bytes);
    assert_eq!(
        chunk.kind, "tEXt",
        "an ASCII block is what A1111 itself writes"
    );
    assert!(!chunk.compressed);
    assert!(chunk.text.contains("a lighthouse in heavy fog"));

    let wide = built(NON_ASCII_PROMPT, "文字、透かし", json!({}));
    let (_p, bytes, _d) = embedded_png(&wide, (64, 48));
    let chunk = parameters_chunk_of(&bytes);
    assert_eq!(
        chunk.kind, "iTXt",
        "a prompt outside Latin-1 cannot be a tEXt chunk at all — `png` refuses to encode one"
    );
    assert!(
        !chunk.compressed,
        "uncompressed, matching PIL's zip=False default: a compressed chunk is the form \
         third-party readers are least reliably tested against"
    );
    // And it survives character for character, which a tEXt chunk could not have promised.
    let parsed = parse_a1111(&chunk.text);
    assert_eq!(parsed.prompt, NON_ASCII_PROMPT);
    assert_eq!(parsed.negative_prompt, "文字、透かし");

    // The band where we are deliberately narrower than PIL: Latin-1-representable but not ASCII.
    // PIL would write tEXt; we write iTXt, because a 0xE9 byte in a tEXt chunk is `é` to a Latin-1
    // reader and a replacement character to the very common reader that assumes UTF-8.
    let latin1 = built("café naïve", "", json!({}));
    let (_p, bytes, _d) = embedded_png(&latin1, (64, 48));
    let chunk = parameters_chunk_of(&bytes);
    assert_eq!(chunk.kind, "iTXt");
    assert!(chunk.text.starts_with("café naïve"));
}

#[test]
fn pil_agrees_with_our_encoding_choice() {
    // The offline stand-in for the live-upload AC. PIL is what A1111 writes through and what the
    // ecosystem's parsers are tested against, so "PIL finds our chunk under `parameters` and hands
    // back the exact string we wrote" is the strongest offline statement available about whether a
    // gallery will render it.
    //
    // Two things are checked, and the second is the one the story asked to be verified rather than
    // assumed:
    //
    //   1. `Image.open(...).text["parameters"]` equals our text, for the ASCII and the non-ASCII
    //      case — so both encodings are FOUND and DECODED correctly by the reference reader.
    //   2. PIL's own boundary, reported back. `PngInfo.add_text` tries `encode("latin-1")` and
    //      writes `tEXt` on success, falling back to an uncompressed `iTXt` — so its rule is
    //      LATIN-1, not ASCII as the story's description says. Ours is narrower on purpose; see
    //      `parameters_chunk` in workflow_png.rs.
    //
    // Skips loudly when Python or Pillow is absent. A test that fails on a machine without an
    // optional interpreter is worse than one that says why it did nothing.
    let ascii = built("a lighthouse in heavy fog", "text, watermark", json!({}));
    let (ascii_path, ascii_bytes, _d1) = embedded_png(&ascii, (64, 48));
    let wide = built(NON_ASCII_PROMPT, "文字、透かし", json!({}));
    let (wide_path, wide_bytes, _d2) = embedded_png(&wide, (64, 48));

    let script = r#"
import json, sys
try:
    from PIL import Image, PngImagePlugin
except Exception as error:
    print(json.dumps({"skip": str(error)}))
    sys.exit(0)

result = {"pil": getattr(__import__("PIL"), "__version__", "?"), "files": {}}
for label, path in (("ascii", sys.argv[1]), ("wide", sys.argv[2])):
    with Image.open(path) as image:
        image.load()
        result["files"][label] = image.text.get("parameters")

# PIL's own boundary, reported rather than assumed.
def kind(value):
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", value)
    return [chunk[0].decode("ascii") for chunk in info.chunks]

result["pil_kind"] = {
    "ascii": kind("plain ascii"),
    "latin1": kind("café"),
    "wide": kind("灯台"),
}
print(json.dumps(result))
"#;

    let output = Command::new("python")
        .arg("-c")
        .arg(script)
        .arg(&ascii_path)
        .arg(&wide_path)
        .output();
    let Ok(output) = output else {
        println!("SKIPPED pil_agrees_with_our_encoding_choice: no `python` on PATH");
        return;
    };
    assert!(
        output.status.success(),
        "the PIL probe failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&output.stdout).expect("the probe prints one JSON line");
    if let Some(reason) = report.get("skip") {
        println!("SKIPPED pil_agrees_with_our_encoding_choice: Pillow is not installed ({reason})");
        return;
    }

    // (1) Both encodings are found under the keyword and decode to exactly what we wrote.
    let expected = [
        ("ascii", parameters_chunk_of(&ascii_bytes).text),
        ("wide", parameters_chunk_of(&wide_bytes).text),
    ];
    for (label, text) in expected {
        assert_eq!(
            report["files"][label].as_str(),
            Some(text.as_str()),
            "PIL read the `{PARAMETERS_CHUNK_KEYWORD}` chunk of the {label} fixture back as \
             something other than what we wrote"
        );
    }

    // (2) PIL's boundary, recorded. If a future Pillow changes it, this fails and the comment in
    // `parameters_chunk` is re-derived rather than left describing a library that moved.
    assert_eq!(
        report["pil_kind"]["ascii"],
        json!(["tEXt"]),
        "PIL no longer writes ASCII as tEXt"
    );
    assert_eq!(
        report["pil_kind"]["latin1"],
        json!(["tEXt"]),
        "PIL's boundary is LATIN-1, not ASCII — that is the band our rule is deliberately \
         narrower in, and this is the assertion that says so"
    );
    assert_eq!(
        report["pil_kind"]["wide"],
        json!(["iTXt"]),
        "PIL no longer falls back to iTXt outside Latin-1"
    );
    println!(
        "PIL {} read both chunks back verbatim; its own boundary is latin-1 (tEXt) / iTXt",
        report["pil"]
    );
}

// ---------------------------------------------------------------------------
// The strip interaction (sc-15953)
// ---------------------------------------------------------------------------

#[test]
fn save_a_copy_without_the_workflow_removes_the_parameters_chunk_too() {
    // The sharpest edge in the story. sc-15953 ships a control promising the recipe is removed; a
    // `parameters` chunk that survived it would leave the prompt in a file the user deliberately
    // stripped, and they would have no way to know.
    let share = built("a lighthouse in heavy fog", "text, watermark", json!({}));
    let (_p, bytes, _d) = embedded_png(&share, (64, 48));
    assert_eq!(text_chunks(&bytes).len(), 2, "both chunks were written");

    let stripped = strip_workflow_chunk(&bytes)
        .expect("strips")
        .expect("had something to take out");

    assert!(
        text_chunks(&stripped).is_empty(),
        "{:?}",
        text_chunks(&stripped)
    );
    // At the byte level, which is what a stranger's tooling sees: neither keyword and neither
    // prompt survives anywhere in the file.
    for needle in [
        WORKFLOW_CHUNK_KEYWORD.as_bytes(),
        PARAMETERS_CHUNK_KEYWORD.as_bytes(),
        b"a lighthouse in heavy fog".as_slice(),
        b"text, watermark".as_slice(),
    ] {
        assert!(
            !contains(&stripped, needle),
            "the stripped copy still carries {:?}",
            String::from_utf8_lossy(needle)
        );
    }
    // And the pixels are untouched — a strip is an excision, never a re-encode.
    let plain = {
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("plain.png");
        write_workflow_chunk(&rgb_fixture(64, 48), &path, None).expect("writes");
        fs::read(&path).expect("reads")
    };
    assert!(
        stripped == plain,
        "the stripped file is not the None-path file"
    );
}

#[test]
fn an_imported_a1111_image_keeps_its_own_parameters_block() {
    // The other half of the co-presence rule, and the half that stops "remove the workflow"
    // becoming "scrub every gallery's metadata out of files the user imported". A foreign PNG with
    // a `parameters` chunk and no `sceneworks:workflow` beside it has nothing of ours in it, so the
    // strip is a clean no-op and the caller uses the source bytes unchanged.
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("foreign.png");
    write_workflow_chunk(&rgb_fixture(32, 32), &path, None).expect("writes a plain PNG");
    let plain = fs::read(&path).expect("reads");
    let foreign = splice_text_chunk(
        &plain,
        b"tEXt",
        PARAMETERS_CHUNK_KEYWORD,
        "somebody else's prompt\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, \
         Model: sd_xl_base, Version: v1.9.4",
    );

    assert_eq!(
        strip_workflow_chunk(&foreign),
        Ok(None),
        "a foreign A1111 export has nothing of OURS in it, so the strip must be a no-op rather \
         than a scrub of someone else's metadata"
    );

    // But the moment our envelope is beside it, the pair is ours and both come out.
    let ours = built("a lighthouse", "", json!({}));
    let (_p, embedded, _d) = embedded_png(&ours, (32, 32));
    let stripped = strip_workflow_chunk(&embedded)
        .expect("strips")
        .expect("had something to take out");
    assert!(text_chunks(&stripped).is_empty());
}

#[test]
fn a_stripped_copy_strips_again_to_itself() {
    // Idempotence falls out of the co-presence rule and is worth pinning: the second strip finds no
    // envelope, so it correctly declines to touch anything, and the download route's ETag-revalidated
    // representation cannot drift between two requests for the same file.
    let share = built("a lighthouse", "", json!({}));
    let (_p, bytes, _d) = embedded_png(&share, (64, 48));
    let once = strip_workflow_chunk(&bytes)
        .expect("strips")
        .expect("had something to take out");
    assert_eq!(
        strip_workflow_chunk(&once),
        Ok(None),
        "a file already stripped has nothing left of ours"
    );
}

/// Splice a hand-framed text chunk in before the first IDAT.
fn splice_text_chunk(png: &[u8], kind: &[u8; 4], keyword: &str, text: &str) -> Vec<u8> {
    let mut data = keyword.as_bytes().to_vec();
    data.push(0);
    data.extend_from_slice(text.as_bytes());

    let mut checked = kind.to_vec();
    checked.extend_from_slice(&data);
    let mut chunk = u32::try_from(data.len())
        .expect("fits")
        .to_be_bytes()
        .to_vec();
    chunk.extend_from_slice(&checked);
    chunk.extend_from_slice(&crc32(&checked).to_be_bytes());

    let mut cursor = 8;
    let at = loop {
        assert!(cursor + 8 <= png.len(), "no IDAT");
        if &png[cursor + 4..cursor + 8] == b"IDAT" {
            break cursor;
        }
        let length = usize::try_from(u32::from_be_bytes(
            png[cursor..cursor + 4].try_into().expect("a length"),
        ))
        .expect("fits");
        cursor += 12 + length;
    };
    let mut out = png[..at].to_vec();
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[at..]);
    out
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

// ---------------------------------------------------------------------------
// Omit rather than approximate
// ---------------------------------------------------------------------------

#[test]
fn the_settings_line_emits_exactly_the_declared_keys() {
    // `SETTINGS_FIELDS` is the decision record for "omit rather than approximate", and it is only
    // worth anything if it is the truth. Pinned in BOTH directions: an envelope carrying everything
    // must emit every declared key, and it must emit nothing else.
    let share = built(
        "a lighthouse",
        "text",
        json!({
            "advanced": {
                "steps": 28,
                "sampler": "euler",
                "guidanceScale": 3.5,
                "scheduler": "karras",
                "schedulerShift": 3.0,
                "strength": 0.6,
                "textStyleGain": 1.4,
                "controlMode": "canny",
                "controlScale": 0.8,
                "faceRestore": true,
                "enhancePrompt": true,
                "usePid": true,
                "pidTarget": "4k",
                "resolution": "1024x1024",
                "styleId": "ghibli"
            },
            "stylePreset": "Ghibli",
            "fitMode": "crop",
            "upscale": { "enabled": true, "factor": 2, "engine": "real-esrgan" },
            "loras": [{ "name": "Mira", "weight": 0.8, "repo": "acme/mira" }]
        }),
    );
    let text = parameters_text(&share, (64, 48));
    let emitted: BTreeSet<String> = parse_params(text.lines().last().expect("a settings line"))
        .into_keys()
        .collect();
    let declared: BTreeSet<String> = SETTINGS_FIELDS
        .iter()
        .map(|(key, _)| (*key).to_owned())
        .collect();
    assert_eq!(
        emitted, declared,
        "the settings line and SETTINGS_FIELDS disagree; the declaration is what a reviewer reads \
         to check that nothing was approximated"
    );
    for (key, reason) in SETTINGS_FIELDS {
        assert!(reason.len() > 20, "`{key}` needs a real reason");
    }

    // And the things with no clean A1111 equivalent are absent from the rendered block by VALUE,
    // not merely un-keyed: a scheduler, a strength, an upscale factor, a LoRA and a style all
    // travel in the envelope and none of them may appear here.
    for absent in [
        "karras",
        "textStyleGain",
        "canny",
        "Ghibli",
        "crop",
        "real-esrgan",
        "Mira",
        "acme/mira",
        "<lora:",
        "Denoising strength",
        "Hires",
        "Schedule type",
        "Batch size",
        "Face restoration",
    ] {
        assert!(
            !text.contains(absent),
            "the parameters block approximates {absent:?}: {text:?}"
        );
    }
}

#[test]
fn a_multi_phase_krea_recipe_omits_the_fields_it_cannot_state() {
    // The case the AC names. A Krea multi-phase run has a SCHEDULE of steps and guidance, not one
    // of each; emitting the top-level numbers beside it would describe a run that did not happen,
    // and a gallery reader has no way to tell.
    let share = built(
        "a lighthouse",
        "",
        json!({
            "model": "krea_2_turbo",
            "advanced": {
                "steps": 28,
                "guidanceScale": 3.5,
                "sampler": "euler",
                "textStyleGain": 1.4,
                "phases": [
                    { "steps": 12, "guidance": 4.0, "loras": [{ "index": 0, "weight": 0.7 }] },
                    { "steps": 16, "guidance": 2.0 }
                ]
            }
        }),
    );
    assert!(
        share.advanced.contains_key("phases"),
        "the fixture must actually carry a multi-phase schedule"
    );
    let text = parameters_text(&share, (64, 48));
    let params = parse_params(text.lines().last().expect("a settings line"));
    assert!(!params.contains_key("Steps"), "{text:?}");
    assert!(!params.contains_key("CFG scale"), "{text:?}");
    // Nothing was invented for the schedule or the tap-reweight gain either.
    for absent in ["Phases", "phases", "1.4", "textStyleGain"] {
        assert!(
            !text.contains(absent),
            "{absent:?} reached the block: {text:?}"
        );
    }
    // What can be stated exactly still is.
    assert_eq!(params["Sampler"], "euler");
    assert_eq!(params["Model"], "krea_2_turbo");
    assert_eq!(params["Version"], PRODUCER_VERSION);
}

#[test]
fn a_prompt_carrying_a_comma_or_a_newline_still_parses() {
    // A prompt is user prose and routinely has both. The prompt is positional so a comma in it is
    // harmless, but the settings line is comma-separated — this is the case where a renderer that
    // forgot A1111's quoting rule would split one field into two for every reader.
    let prompt = "a lighthouse, in heavy fog\nshot on 35mm, cinematic";
    let share = built(prompt, "text, watermark, blurry", json!({}));
    let (_p, bytes, _d) = embedded_png(&share, (64, 48));
    let parsed = parse_a1111(&parameters_chunk_of(&bytes).text);
    assert_eq!(parsed.prompt, prompt);
    assert_eq!(parsed.negative_prompt, "text, watermark, blurry");
    assert_eq!(parsed.params["Model"], "z_image_turbo");
}
