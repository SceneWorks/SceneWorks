//! `docs/workflow-share-envelope.md` against the shipped contract (sc-15955, epic 15945).
//!
//! A document is the artifact most able to claim a guarantee the code does not deliver, and the
//! least able to be caught doing it. So every field list, vocabulary, registry and numeric ceiling
//! in that file is fenced in a `<!-- PINNED: name -->` block and asserted here **in both
//! directions**: a row the code does not have fails, and a thing the code has that is not a row
//! fails too.
//!
//! Two of the pins are behavioural rather than a constant read back:
//!
//! * [`the_doc_lists_exactly_the_path_exempt_prose_fields`] seeds a filesystem path into every
//!   string-bearing field of a real request and asserts that the fields which carry it through are
//!   exactly the ones the document's privacy callout names. That is the claim a user is most likely
//!   to be surprised by, so it is checked against what the sanitizer does rather than against a
//!   `PROSE_KEYS` constant that could itself be wrong.
//! * [`the_documented_ceilings_are_the_real_limits`] finds each collection and string bound by
//!   experiment — the largest input that survives and the smallest that does not — so the numbers in
//!   the document are the numbers a user actually hits. Its last column is read too, not assumed:
//!   whether overflowing a ceiling truncates, drops or refuses is parsed out of the table and
//!   turned into the assertion.
//!
//! Three claims live in the narrative prose rather than in a block, and each had a way to drift
//! silently, so each is pinned on its own: the prose ceiling the privacy callout quotes
//! ([`the_privacy_callout_quotes_the_real_prose_ceiling`]), the counts it spells out in words
//! ([`the_privacy_callout_counts_its_own_prose_fields`]) and the pose coordinate arrays it names
//! twice ([`the_doc_names_exactly_the_pose_fields_that_travel`]).

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use sceneworks_core::contracts::{Asset, JsonObject};
use sceneworks_core::workflow_png::{
    MAX_METADATA_BYTES, MAX_WORKFLOW_TEXT_BYTES, WORKFLOW_CHUNK_KEYWORD,
};
use sceneworks_core::workflow_share::{
    build_workflow_share, is_path_shaped, parse_workflow_share, AdvancedDisposition, AdvancedShape,
    SeamDisposition, WorkflowInput, WorkflowLora, WorkflowProducer, WorkflowShare, WorkflowUpscale,
    ADVANCED_BUILDERS, ADVANCED_KEY_RULES, DEFERRED_ADVANCED_BUILDERS, INPUT_KINDS,
    INPUT_KIND_SOURCE, OMITTED_FIELDS, OMITTED_LORAS, POSE_FIELDS, PRODUCER_NAME, PRODUCER_URL,
    WORKFLOW_KINDS, WORKFLOW_KIND_IMAGE, WORKFLOW_SHARE_MARKER_KEY, WORKFLOW_SHARE_MAX_BYTES,
    WORKFLOW_SHARE_SCHEMA_VERSION, WORKFLOW_WRITE_SEAMS,
};
use serde_json::{json, Value};

/// Repo-relative path of the document under test. Named once so a move is one edit.
const DOC_PATH: &str = "docs/workflow-share-envelope.md";

/// The integration-test file whose lint names the document quotes.
const LINT_TEST_PATH: &str = "crates/sceneworks-core/tests/workflow_share.rs";

/// The module the contributor section sends people to edit.
const SHARE_SOURCE_PATH: &str = "crates/sceneworks-core/src/workflow_share.rs";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn read_repo_file(relative_path: &str) -> String {
    let path = repo_root().join(relative_path);
    fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}

fn doc() -> String {
    read_repo_file(DOC_PATH)
}

// ---------------------------------------------------------------------------
// Reading a pinned table out of the markdown
// ---------------------------------------------------------------------------

/// The DATA rows of the one markdown table inside `<!-- PINNED: name -->`, as trimmed cells.
///
/// Panics rather than returning an empty set when the block or its table is missing: a parser that
/// quietly finds nothing is the failure mode this whole lint exists to prevent — it would pass
/// against a document that had been emptied.
fn pinned_rows(doc: &str, name: &str) -> Vec<Vec<String>> {
    let open = format!("<!-- PINNED: {name} -->");
    let close = format!("<!-- END PINNED: {name} -->");
    let start = doc.find(&open).unwrap_or_else(|| {
        panic!("{DOC_PATH} has no `{open}` block — the document and this lint have drifted apart")
    });
    let end = doc.find(&close).unwrap_or_else(|| {
        panic!("{DOC_PATH} has no `{close}` marker to close the `{name}` block")
    });
    assert!(
        end > start,
        "{DOC_PATH}: the `{name}` block closes before it opens"
    );

    let mut rows = Vec::new();
    let mut past_header = false;
    for line in doc[start + open.len()..end].lines() {
        // The privacy callout is a blockquote, so its table rows carry a `>` prefix.
        let line = line.trim_start().trim_start_matches('>').trim();
        if !line.starts_with('|') {
            continue;
        }
        let cells: Vec<String> = line
            .trim_matches('|')
            .split('|')
            .map(|cell| cell.trim().to_owned())
            .collect();
        // The `| --- | --- |` rule separates the header from the data; everything before it is the
        // header, everything after it is a row.
        if cells
            .iter()
            .all(|cell| !cell.is_empty() && cell.chars().all(|c| c == '-' || c == ':'))
        {
            past_header = true;
            continue;
        }
        if past_header {
            rows.push(cells);
        }
    }
    assert!(
        !rows.is_empty(),
        "{DOC_PATH}: the `{name}` block has no table rows"
    );
    rows
}

/// One cell, with its surrounding backticks removed.
fn code(cell: &str) -> String {
    cell.trim().trim_matches('`').trim().to_owned()
}

/// The document with every run of whitespace collapsed to a single space.
///
/// A claim in the narrative prose is a sentence, not a line: the markdown is hard-wrapped at 100
/// columns, so any phrase this file searches for would otherwise break the moment someone reflowed
/// a paragraph around it.
fn flowed(doc: &str) -> String {
    doc.split_whitespace().collect::<Vec<&str>>().join(" ")
}

/// The phrase between `anchor` and the next `terminator` after it, in a [`flowed`] document.
///
/// Panics when either marker is gone. A prose claim that was reworded out of the shape this lint
/// reads is a claim nothing is checking any more, which is the failure mode the whole file exists
/// to refuse — so it fails loudly rather than matching nothing and passing.
fn phrase(flowed: &str, anchor: &str, terminator: &str) -> String {
    let start = flowed.find(anchor).unwrap_or_else(|| {
        panic!(
            "{DOC_PATH} no longer contains {anchor:?}. That sentence is pinned to the code; if it \
             was reworded, teach this lint the new shape rather than leaving the claim unchecked."
        )
    }) + anchor.len();
    let rest = &flowed[start..];
    let end = rest.find(terminator).unwrap_or_else(|| {
        panic!("{DOC_PATH}: the phrase after {anchor:?} is never closed by {terminator:?}")
    });
    rest[..end].to_owned()
}

/// The backticked identifiers of `text`, in the order they are written.
fn backticked(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = text;
    while let Some(open) = rest.find('`') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('`') else { break };
        out.insert(after[..close].trim().to_owned());
        rest = &after[close + 1..];
    }
    out
}

/// The English number words this file can spell, indexed by the number they name.
///
/// The document counts things in prose ("These six fields"), and a count written by hand is a
/// count that goes stale the first time the list under it changes.
const NUMBER_WORDS: &[&str] = &[
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
];

fn number_word(count: usize) -> &'static str {
    NUMBER_WORDS.get(count).copied().unwrap_or_else(|| {
        panic!(
            "{DOC_PATH} now has {count} of something this lint spells out in prose, which is past \
             the {} this file can name — extend NUMBER_WORDS",
            NUMBER_WORDS.len() - 1
        )
    })
}

/// Column `index` of every row in a pinned block, un-backticked.
fn pinned_column(doc: &str, name: &str, index: usize) -> Vec<String> {
    pinned_rows(doc, name)
        .into_iter()
        .map(|row| {
            assert!(
                row.len() > index,
                "{DOC_PATH}: a row in the `{name}` block has fewer than {} columns: {row:?}",
                index + 1
            );
            code(&row[index])
        })
        .collect()
}

/// Column `index` as a SET, refusing duplicates — a table that lists a key twice is a table two
/// people edited without seeing each other.
fn pinned_set(doc: &str, name: &str, index: usize) -> BTreeSet<String> {
    let values = pinned_column(doc, name, index);
    let unique: BTreeSet<String> = values.iter().cloned().collect();
    assert_eq!(
        unique.len(),
        values.len(),
        "{DOC_PATH}: the `{name}` block lists a value twice"
    );
    unique
}

/// The both-directions assertion every pin below is built from.
fn assert_same_set(
    documented: &BTreeSet<String>,
    actual: &BTreeSet<String>,
    block: &str,
    code_source: &str,
) {
    let undocumented: Vec<&String> = actual.difference(documented).collect();
    assert!(
        undocumented.is_empty(),
        "{code_source} has {undocumented:?}, which the `{block}` table in {DOC_PATH} does not \
         list. The document is what a user reads before deciding to share an image and what a \
         contributor reads before adding a knob; a thing the code does that the document omits is \
         the document lying by silence. Add the row."
    );
    let invented: Vec<&String> = documented.difference(actual).collect();
    assert!(
        invented.is_empty(),
        "the `{block}` table in {DOC_PATH} lists {invented:?}, which is not in {code_source}. \
         Either the code dropped it and the document went stale, or the document claims something \
         that was never true. Remove the row or restore the code."
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn asset_fixture() -> Asset {
    serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "asset_doc",
        "projectId": "project_doc",
        "generationSetId": "genset_doc",
        "type": "image",
        "displayName": "Doc fixture",
        "createdAt": "2026-07-30T09:00:00Z",
        "file": {
            "path": "assets/images/doc.png",
            "mimeType": "image/png",
            "width": 1024,
            "height": 1024,
            "duration": null,
            "fps": null
        },
        "status": { "favorite": false, "rating": 0, "rejected": false, "trashed": false },
        "recipe": {
            "mode": "text_to_image",
            "model": "z_image_turbo",
            "adapter": "z_image_diffusers",
            "prompt": "a lighthouse",
            "negativePrompt": "blurry",
            "seed": 7,
            "loras": [],
            "stylePreset": "cinematic",
            "normalizedSettings": {},
            "rawAdapterSettings": {}
        },
        "lineage": {
            "parents": [],
            "sourceAssetId": null,
            "sourceTimestamp": null,
            "jobId": "job_doc"
        }
    }))
    .expect("asset fixture parses")
}

fn payload_fixture() -> JsonObject {
    json!({
        "mode": "text_to_image",
        "model": "z_image_turbo",
        "prompt": "a lighthouse",
        "negativePrompt": "blurry",
        "count": 1,
        "width": 1024,
        "height": 1024,
        "advanced": {}
    })
    .as_object()
    .cloned()
    .expect("payload fixture is an object")
}

fn build(payload: &JsonObject) -> WorkflowShare {
    build_workflow_share(&asset_fixture(), payload)
}

fn advanced_of(payload: &mut JsonObject) -> &mut JsonObject {
    payload
        .get_mut("advanced")
        .and_then(Value::as_object_mut)
        .expect("advanced object")
}

/// Every field of the envelope populated, so the serialized shape is the WIDEST one the contract
/// can produce.
///
/// A struct literal on purpose: a new field on `WorkflowShare` fails to compile here rather than
/// slipping past a `..Default::default()`, which makes "the document lists every field" a
/// build-time property and not a hope.
fn fully_populated_envelope() -> WorkflowShare {
    WorkflowShare {
        kind: WORKFLOW_KIND_IMAGE.to_owned(),
        schema_version: WORKFLOW_SHARE_SCHEMA_VERSION,
        producer: WorkflowProducer::default(),
        mode: "text_to_image".to_owned(),
        model: "z_image_turbo".to_owned(),
        prompt: "a lighthouse".to_owned(),
        negative_prompt: "blurry".to_owned(),
        seed: Some(7),
        width: Some(1024),
        height: Some(1024),
        count: Some(2),
        // The video arm (sc-15956). Populated even though `kind` above is the image one: this
        // fixture's job is the WIDEST serialized shape the contract can produce, so that the
        // document is pinned against every field rather than against one lane's.
        duration_seconds: Some(5.0),
        fps: Some(24),
        quality: Some("balanced".to_owned()),
        style_preset: Some("cinematic".to_owned()),
        style_id: Some("noir".to_owned()),
        fit_mode: Some("crop".to_owned()),
        upscale: Some(WorkflowUpscale {
            enabled: true,
            factor: Some(2),
            engine: Some("seedvr2".to_owned()),
            softness: Some(0.25),
        }),
        loras: vec![WorkflowLora {
            name: Some("coast".to_owned()),
            weight: Some(0.6),
            repo: Some("acme/coast".to_owned()),
        }],
        inputs: vec![WorkflowInput {
            kind: INPUT_KIND_SOURCE.to_owned(),
            count: 1,
            control_mode: Some("canny".to_owned()),
        }],
        advanced: json!({ "steps": 8 })
            .as_object()
            .cloned()
            .expect("advanced object"),
        omitted: vec![OMITTED_LORAS.to_owned()],
    }
}

/// Every leaf path of a serialized envelope, with `advanced` and `omitted` kept whole because the
/// document gives each its own table.
fn field_paths(value: &Value, prefix: &str, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                if path == "advanced" || path == "omitted" {
                    out.insert(path);
                    continue;
                }
                field_paths(child, &path, out);
            }
        }
        Value::Array(items) => {
            let path = format!("{prefix}[]");
            for item in items {
                field_paths(item, &path, out);
            }
        }
        _ => {
            out.insert(prefix.to_owned());
        }
    }
}

/// Every string leaf of a serialized envelope, as (path, value). Descends into `advanced`, unlike
/// [`field_paths`] — the prose exemption lives in there.
fn string_paths(value: &Value, prefix: &str, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                string_paths(child, &path, out);
            }
        }
        Value::Array(items) => {
            let path = format!("{prefix}[]");
            for item in items {
                string_paths(item, &path, out);
            }
        }
        Value::String(text) => out.push((prefix.to_owned(), text.clone())),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// The envelope's own shape
// ---------------------------------------------------------------------------

#[test]
fn the_doc_lists_exactly_the_envelope_fields() {
    let mut actual = BTreeSet::new();
    field_paths(
        &serde_json::to_value(fully_populated_envelope()).expect("serializes"),
        "",
        &mut actual,
    );
    assert_same_set(
        &pinned_set(&doc(), "envelope-fields", 0),
        &actual,
        "envelope-fields",
        "a fully-populated `WorkflowShare`",
    );
}

#[test]
fn the_doc_quotes_the_contract_constants() {
    let doc = doc();
    let documented: BTreeMap<String, String> = pinned_rows(&doc, "identity")
        .into_iter()
        .map(|row| (code(&row[0]), code(&row[1])))
        .collect();
    let actual: BTreeMap<String, String> = [
        ("WORKFLOW_CHUNK_KEYWORD", WORKFLOW_CHUNK_KEYWORD.to_owned()),
        (
            "WORKFLOW_SHARE_MARKER_KEY",
            WORKFLOW_SHARE_MARKER_KEY.to_owned(),
        ),
        ("WORKFLOW_KIND_IMAGE", WORKFLOW_KIND_IMAGE.to_owned()),
        (
            "WORKFLOW_SHARE_SCHEMA_VERSION",
            WORKFLOW_SHARE_SCHEMA_VERSION.to_string(),
        ),
        ("PRODUCER_NAME", PRODUCER_NAME.to_owned()),
        ("PRODUCER_URL", PRODUCER_URL.to_owned()),
    ]
    .into_iter()
    .map(|(name, value)| (name.to_owned(), value))
    .collect();

    assert_same_set(
        &documented.keys().cloned().collect(),
        &actual.keys().cloned().collect(),
        "identity",
        "the workflow-share contract constants",
    );
    for (name, value) in &actual {
        assert_eq!(
            documented.get(name),
            Some(value),
            "{DOC_PATH} quotes `{name}` as {:?}, but it is {value:?}",
            documented.get(name)
        );
    }
}

// ---------------------------------------------------------------------------
// The allow-list
// ---------------------------------------------------------------------------

fn rule_keys(disposition: AdvancedDisposition) -> BTreeSet<String> {
    ADVANCED_KEY_RULES
        .iter()
        .filter(|rule| rule.disposition == disposition)
        .map(|rule| rule.key.to_owned())
        .collect()
}

#[test]
fn the_doc_lists_exactly_the_shared_advanced_keys() {
    let doc = doc();
    assert_same_set(
        &pinned_set(&doc, "advanced-shared", 0),
        &rule_keys(AdvancedDisposition::Allow),
        "advanced-shared",
        "`ADVANCED_KEY_RULES` (allowed)",
    );

    // The `Shape` column is load-bearing prose in the table next to it ("reduced to its scalar
    // fields", "reduced to the coordinate arrays"), so it is pinned too.
    for row in pinned_rows(&doc, "advanced-shared") {
        let key = code(&row[0]);
        let rule = ADVANCED_KEY_RULES
            .iter()
            .find(|rule| rule.key == key)
            .expect("key set already asserted equal");
        assert_eq!(
            code(&row[1]),
            format!("{:?}", rule.shape),
            "{DOC_PATH} documents `{key}` with the wrong shape"
        );
    }
}

#[test]
fn the_doc_lists_exactly_the_withheld_advanced_keys() {
    let doc = doc();
    let withheld = pinned_set(&doc, "advanced-withheld", 0);
    assert_same_set(
        &withheld,
        &rule_keys(AdvancedDisposition::Deny),
        "advanced-withheld",
        "`ADVANCED_KEY_RULES` (denied)",
    );
    let shared = pinned_set(&doc, "advanced-shared", 0);
    let both: Vec<&String> = shared.intersection(&withheld).collect();
    assert!(
        both.is_empty(),
        "{DOC_PATH} lists {both:?} as both shared and withheld"
    );
}

#[test]
fn the_doc_lists_exactly_the_input_kinds() {
    assert_same_set(
        &pinned_set(&doc(), "input-kinds", 0),
        &INPUT_KINDS.iter().map(|kind| (*kind).to_owned()).collect(),
        "input-kinds",
        "`INPUT_KINDS`",
    );
}

#[test]
fn the_doc_lists_exactly_the_omitted_vocabulary() {
    assert_same_set(
        &pinned_set(&doc(), "omitted-fields", 0),
        &OMITTED_FIELDS
            .iter()
            .map(|field| (*field).to_owned())
            .collect(),
        "omitted-fields",
        "`OMITTED_FIELDS`",
    );
}

/// Every reader named in the `container-kinds` table really asserts the kind the table claims
/// (sc-15956 review).
///
/// The rule this pins is the one the marker's widening broke: the shared parser accepts every kind
/// in `WORKFLOW_KINDS`, so a reader that hands its result to one lane owes a container assert of its
/// own. The PNG reader had one, the MP4 reader had one, and the asset route — the third reader,
/// in a crate the widening did not touch — silently lost the refusal it used to inherit from the
/// parser.
///
/// Not a prose check. Each row names a file and a function, and this reads that file and fails if
/// the assert it claims is not in it, so deleting a gate breaks the document as well as the route's
/// own test.
#[test]
fn the_doc_lists_exactly_the_container_kind_asserts() {
    let rows = pinned_rows(&doc(), "container-kinds");
    assert!(
        rows.len() >= WORKFLOW_KINDS.len(),
        "{DOC_PATH}: the `container-kinds` block lists {} readers for {} kinds — every kind needs \
         at least one container that accepts it",
        rows.len(),
        WORKFLOW_KINDS.len()
    );
    let mut accepted: BTreeSet<String> = BTreeSet::new();
    for row in &rows {
        assert!(
            row.len() >= 4,
            "{DOC_PATH}: a `container-kinds` row has fewer than four columns: {row:?}"
        );
        let path = code(&row[1]);
        let function = code(&row[2]);
        let kind = code(&row[3]);
        assert!(
            WORKFLOW_KINDS.contains(&kind.as_str()),
            "{DOC_PATH}: `container-kinds` says {function} accepts `{kind}`, which is not a kind \
             this contract has. Known kinds: {WORKFLOW_KINDS:?}"
        );
        accepted.insert(kind.clone());

        let source = read_repo_file(&path);
        assert!(
            source.contains(&format!("fn {function}")),
            "{DOC_PATH}: `container-kinds` names `{function}` in {path}, which has no such \
             function. A reader that moved needs the table moved with it."
        );
        let constant = if kind == WORKFLOW_KIND_IMAGE {
            "WORKFLOW_KIND_IMAGE"
        } else {
            "WORKFLOW_KIND_VIDEO"
        };
        assert!(
            source.contains(&format!("share.kind != {constant}")),
            "{path} does not assert `share.kind != {constant}` anywhere, and {DOC_PATH}'s \
             `container-kinds` table says `{function}` accepts only `{kind}`.\n\
             The shared parser accepts EVERY kind, so a reader that feeds one lane is the only \
             thing standing between a `{kind}`-only surface and an envelope of another kind being \
             presented as one it is not."
        );
    }
    assert_eq!(
        accepted,
        WORKFLOW_KINDS
            .iter()
            .map(|kind| (*kind).to_owned())
            .collect::<BTreeSet<String>>(),
        "{DOC_PATH}: every kind the contract has needs a container that accepts it, or the kind is \
         one nothing can read"
    );
}

/// Every in-document `](#anchor)` link resolves to a heading in this document.
///
/// The sc-15956 review found `[Person replacement](#person-replacement-is-withheld-by-default)`
/// pointing at a heading that did not exist — and that dead link was the single pointer a future
/// author had to the person-replacement rationale, from the withheld table's own row. A reader
/// following it landed at the top of the page and could reasonably conclude the reasoning had never
/// been written down.
///
/// GitHub's slug rule, near enough for this document: lowercase, drop everything that is not a
/// letter, digit, space or hyphen, then spaces to hyphens.
#[test]
fn every_internal_link_in_the_doc_points_at_a_heading_that_exists() {
    let doc = doc();
    let mut headings: BTreeSet<String> = BTreeSet::new();
    for line in doc.lines() {
        // The privacy callout is a blockquote, so its headings carry a `>` prefix.
        let line = line.trim_start().trim_start_matches('>').trim_start();
        let Some(rest) = line.strip_prefix('#') else {
            continue;
        };
        let title = rest.trim_start_matches('#').trim();
        if title.is_empty() {
            continue;
        }
        let slug: String = title
            .to_lowercase()
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-')
            .map(|c| if c == ' ' { '-' } else { c })
            .collect();
        headings.insert(slug);
    }
    assert!(
        !headings.is_empty(),
        "{DOC_PATH}: no headings parsed at all"
    );

    let mut dead: Vec<String> = Vec::new();
    let mut seen = 0_usize;
    // Links are hard-wrapped across lines, so search the FLOWED document.
    let flowed = flowed(&doc);
    for chunk in flowed.split("](#").skip(1) {
        let Some(end) = chunk.find(')') else {
            continue;
        };
        seen += 1;
        // A wrapped link puts a space inside the anchor; the rendered href has none.
        let anchor = chunk[..end].replace(' ', "");
        if !headings.contains(&anchor) {
            dead.push(anchor);
        }
    }
    assert!(seen > 0, "{DOC_PATH}: no internal links parsed at all");
    assert!(
        dead.is_empty(),
        "{DOC_PATH} links to {dead:?}, and no heading in it produces those anchors. \
         The headings that exist are {headings:?}."
    );
}

// ---------------------------------------------------------------------------
// The registries a contributor is sent to
// ---------------------------------------------------------------------------

/// `path::function`, so a builder that moved file fails as loudly as one that was renamed.
fn documented_builders(doc: &str, block: &str) -> BTreeSet<String> {
    pinned_rows(doc, block)
        .into_iter()
        .map(|row| format!("{}::{}", code(&row[0]), code(&row[1])))
        .collect()
}

#[test]
fn the_doc_lists_exactly_the_registered_builders() {
    assert_same_set(
        &documented_builders(&doc(), "builders"),
        &ADVANCED_BUILDERS
            .iter()
            .map(|builder| format!("{}::{}", builder.path, builder.function))
            .collect(),
        "builders",
        "`ADVANCED_BUILDERS`",
    );
}

#[test]
fn the_doc_lists_exactly_the_deferred_builders() {
    assert_same_set(
        &documented_builders(&doc(), "deferred-builders"),
        &DEFERRED_ADVANCED_BUILDERS
            .iter()
            .map(|builder| format!("{}::{}", builder.path, builder.function))
            .collect(),
        "deferred-builders",
        "`DEFERRED_ADVANCED_BUILDERS`",
    );
}

/// The write-seam table, and the disposition word each row claims (sc-16113).
///
/// The disposition is read out of the prose column rather than trusted, because the whole point of
/// this story is that a lane can flip from declining to embedding: a row that goes on saying
/// "Declines" about a seam that now embeds would be the document repeating the exact class of
/// untrue claim the enforcement was built to end.
#[test]
fn the_doc_lists_exactly_the_write_seams() {
    let rows = pinned_rows(&doc(), "write-seams");
    let documented: BTreeSet<String> = rows
        .iter()
        .map(|row| {
            let word = ["Embeds", "Conduit", "Declines", "Inert"]
                .into_iter()
                .find(|word| row[2].contains(word))
                .unwrap_or_else(|| {
                    panic!(
                        "{DOC_PATH}: the `write-seams` row for {} says nothing about what it does \
                         with the chunk — it must open with `Embeds`, `Conduit`, `Declines` or \
                         `Inert`",
                        row[1]
                    )
                });
            format!("{}::{} {word}", code(&row[0]), code(&row[1]))
        })
        .collect();
    assert_same_set(
        &documented,
        &WORKFLOW_WRITE_SEAMS
            .iter()
            .map(|seam| {
                let word = match seam.disposition {
                    SeamDisposition::Embeds(_) => "Embeds",
                    SeamDisposition::Conduit(_) => "Conduit",
                    SeamDisposition::Declines(_) => "Declines",
                    SeamDisposition::Inert(_) => "Inert",
                };
                format!("{}::{} {word}", seam.path, seam.function)
            })
            .collect(),
        "write-seams",
        "`WORKFLOW_WRITE_SEAMS`",
    );
}

/// Every builder a documented seam embeds for is a documented REGISTERED builder, never a deferred
/// one — the enforcement, restated in the two tables a reader actually sees.
#[test]
fn no_documented_write_seam_embeds_for_a_documented_deferred_builder() {
    let deferred = documented_builders(&doc(), "deferred-builders");
    for seam in WORKFLOW_WRITE_SEAMS {
        let SeamDisposition::Embeds(builders) = seam.disposition else {
            continue;
        };
        for reference in builders {
            let key = format!("{}::{}", reference.path, reference.function);
            assert!(
                !deferred.contains(&key),
                "{DOC_PATH} lists {key} in the `deferred-builders` table while \
                 `WORKFLOW_WRITE_SEAMS` has {}::{} embedding for it. The document would be telling \
                 a reader that lane does not embed while the code says it does.",
                seam.path,
                seam.function
            );
            assert!(
                documented_builders(&doc(), "builders").contains(&key),
                "{DOC_PATH}: {}::{} embeds for {key}, which the `builders` table does not list",
                seam.path,
                seam.function
            );
        }
    }
}

/// The document tells a contributor which test failed. A renamed test would send them looking for
/// something that no longer exists, which is worse than saying nothing.
///
/// One direction only, and deliberately: `workflow_share.rs` has dozens of tests and the document
/// names the four that hold the coverage class. It does not claim to be an index of that file.
#[test]
fn the_doc_names_lints_that_exist() {
    let lints = read_repo_file(LINT_TEST_PATH);
    let documented = pinned_set(&doc(), "lints", 0);
    assert!(
        documented.len() >= 4,
        "{DOC_PATH}: the `lints` table shrank to {} rows — the contributor section is the whole \
         point of documenting the lint",
        documented.len()
    );
    for name in &documented {
        assert!(
            lints.contains(&format!("fn {name}(")),
            "{DOC_PATH} sends a contributor to `{name}`, which no longer exists in \
             {LINT_TEST_PATH}"
        );
    }
}

// ---------------------------------------------------------------------------
// The privacy callout, checked against what the sanitizer actually does
// ---------------------------------------------------------------------------

/// Record a distinctive filesystem path against `path` and hand it back to be seeded there.
fn seed(seeded: &mut BTreeMap<String, String>, slot: &mut usize, path: &str) -> String {
    *slot += 1;
    let value = format!("C:\\Clients\\Acme\\brief-{}.md", *slot);
    seeded.insert(path.to_owned(), value.clone());
    value
}

/// Seed a filesystem path into EVERY string-bearing field of a request, and report what came out.
///
/// The seeded set has to be exhaustive for the reverse direction to mean anything: a field nobody
/// seeded could be prose-exempt and this would never notice. So it covers every top-level string
/// the envelope copies, both string fields of a LoRA entry, the upscale engine label, every
/// allow-listed `Scalar` key, and every field of the structured-prompt recipe.
fn envelope_with_paths_everywhere() -> (WorkflowShare, BTreeMap<String, String>) {
    let mut seeded = BTreeMap::new();
    let mut slot = 0usize;

    let mut payload = payload_fixture();
    for key in [
        "mode",
        "model",
        "prompt",
        "negativePrompt",
        "stylePreset",
        "styleId",
        "fitMode",
    ] {
        let value = seed(&mut seeded, &mut slot, key);
        payload.insert(key.to_owned(), json!(value));
    }
    let engine = seed(&mut seeded, &mut slot, "upscale.engine");
    payload.insert(
        "upscale".to_owned(),
        json!({ "enabled": true, "factor": 2, "engine": engine, "softness": 0.2 }),
    );
    let lora_name = seed(&mut seeded, &mut slot, "loras[].name");
    let lora_repo = seed(&mut seeded, &mut slot, "loras[].repo");
    payload.insert(
        "loras".to_owned(),
        json!([{
            "name": lora_name,
            "weight": 0.5,
            "source": { "provider": "huggingface", "repo": lora_repo }
        }]),
    );
    // A source image, so `inputs[]` exists at all; the control input below exercises
    // `inputs[].controlMode`.
    payload.insert("sourceAssetId".to_owned(), json!("asset_source"));

    let mut structured = JsonObject::new();
    for field in [
        "version",
        "intent",
        "magicPromptBackend",
        "edited",
        "runtimePrompt",
    ] {
        let path = format!("advanced.structuredPrompt.{field}");
        let value = seed(&mut seeded, &mut slot, &path);
        structured.insert(field.to_owned(), json!(value));
    }

    let scalar_keys: Vec<&'static str> = ADVANCED_KEY_RULES
        .iter()
        .filter(|rule| {
            rule.disposition == AdvancedDisposition::Allow && rule.shape == AdvancedShape::Scalar
        })
        .map(|rule| rule.key)
        .collect();
    assert!(
        scalar_keys.len() >= 20,
        "only {} allow-listed scalar keys were found — this sweep is not exercising the \
         allow-list",
        scalar_keys.len()
    );
    let mut advanced = JsonObject::new();
    for key in &scalar_keys {
        let path = format!("advanced.{key}");
        let value = seed(&mut seeded, &mut slot, &path);
        advanced.insert((*key).to_owned(), json!(value));
    }
    advanced.insert("structuredPrompt".to_owned(), Value::Object(structured));
    // A control input, so `inputs[].controlMode` is a real slot rather than an absent one.
    advanced.insert("controlImage".to_owned(), json!("asset_control"));
    payload.insert("advanced".to_owned(), Value::Object(advanced));

    (build(&payload), seeded)
}

#[test]
fn the_doc_lists_exactly_the_path_exempt_prose_fields() {
    let (share, seeded) = envelope_with_paths_everywhere();
    let encoded = serde_json::to_value(&share).expect("serializes");

    let mut strings = Vec::new();
    string_paths(&encoded, "", &mut strings);
    let carrying_a_path: BTreeSet<String> = strings
        .iter()
        .filter(|(_, value)| is_path_shaped(value))
        .map(|(path, _)| path.clone())
        .collect();

    assert_same_set(
        &pinned_set(&doc(), "prose-fields", 0),
        &carrying_a_path,
        "prose-fields",
        "the fields that carried a seeded filesystem path all the way into the envelope",
    );

    // The survivors are verbatim, not merely path-shaped: the callout says "exactly as you typed
    // it", and a mangled-but-still-path-shaped value would satisfy the set assertion above.
    for path in &carrying_a_path {
        let expected = seeded
            .get(path)
            .unwrap_or_else(|| panic!("{path} carried a path this sweep never seeded"));
        let actual = strings
            .iter()
            .find(|(candidate, _)| candidate == path)
            .map(|(_, value)| value.as_str())
            .expect("collected above");
        assert_eq!(
            actual, expected,
            "{path} is documented as travelling verbatim but was altered"
        );
    }

    // And the other half of the claim: every seeded field the callout does NOT name lost its path.
    let documented = pinned_set(&doc(), "prose-fields", 0);
    for (path, value) in &seeded {
        if documented.contains(path) {
            continue;
        }
        assert!(
            !strings
                .iter()
                .any(|(candidate, actual)| candidate == path && actual == value),
            "{path} is not in the {DOC_PATH} prose callout, but it carried {value:?} into the \
             envelope — either the guard regressed or the callout is missing a field"
        );
    }
}

/// The web module that holds the Settings copy's machine-checkable claims (sc-15953).
const SETTINGS_COPY_PATH: &str = "apps/web/src/workflowEmbed.js";

/// The first quoted string of every `["key", LABEL],` entry of a `Object.freeze([…])` list in
/// [`SETTINGS_COPY_PATH`], as a set.
///
/// Only the KEY is read. The labels beside them are shared constants rather than literals — one
/// phrase legitimately covers several keys — and the prose is not what these lints check: the
/// question is whether every field the contract carries has been *classified* by someone writing
/// the copy, which is the step that would have caught `advanced.poses` going unnamed.
///
/// Panics when the list is gone, for the reason every parser in this file does: a lint that
/// quietly finds nothing passes against a file that was emptied.
fn declared_keys(name: &str) -> BTreeSet<String> {
    let source = read_repo_file(SETTINGS_COPY_PATH);
    let anchor = format!("{name} = Object.freeze([");
    let start = source.find(&anchor).unwrap_or_else(|| {
        panic!(
            "{SETTINGS_COPY_PATH} no longer declares {anchor:?}. The settings copy's field lists \
             are pinned to the tables in {DOC_PATH}; if one moved, teach this lint where."
        )
    }) + anchor.len();
    let rest = &source[start..];
    let end = rest
        .find("]);")
        .unwrap_or_else(|| panic!("{SETTINGS_COPY_PATH}: {name} is never closed"));

    let mut listed: BTreeSet<String> = BTreeSet::new();
    for line in rest[..end].lines() {
        let Some(open) = line.trim().strip_prefix("[\"") else {
            continue;
        };
        let Some(close) = open.find('"') else {
            continue;
        };
        let key = open[..close].to_owned();
        assert!(
            listed.insert(key.clone()),
            "{SETTINGS_COPY_PATH}: {name} lists {key:?} twice"
        );
    }
    assert!(
        !listed.is_empty(),
        "{SETTINGS_COPY_PATH}: {name} parsed to no keys at all"
    );
    listed
}

/// The `advanced-shared` / `advanced-withheld` keys as the settings copy spells them: prefixed,
/// because `styleId` is both a top-level envelope field and an advanced setting and the copy has to
/// be able to say something about each.
fn prefixed_advanced(block: &str) -> BTreeSet<String> {
    pinned_set(&doc(), block, 0)
        .into_iter()
        .map(|key| format!("advanced.{key}"))
        .collect()
}

#[test]
fn the_settings_copy_accounts_for_every_shared_and_withheld_field() {
    // The finding this test exists for: "Also in the file" and "Not in the file" are the two
    // paragraphs a user acts on before sending an image to a client, and they were pinned to
    // NOTHING. `advanced.poses` — the `keypoints` / `hands` / `face` coordinate arrays, facial
    // landmarks included — reached the file while the copy named none of it, and the next paragraph
    // said "the input images themselves" are not in the file, which invites exactly the wrong
    // reading: that nothing derived from a reference travels. Claiming more privacy than the
    // allow-list delivers is the forbidden direction, and this is the surface where it matters.
    //
    // So both lists are declared per key and pinned here in both directions. What this buys is not
    // that the prose is good — no lint can say that — but that a field cannot reach a shared image
    // without a human deciding, in the copy, what to say about it.
    let mut shared = pinned_set(&doc(), "envelope-fields", 0);
    shared.extend(prefixed_advanced("advanced-shared"));
    assert_same_set(
        &shared,
        &declared_keys("WORKFLOW_FIELDS_IN_FILE"),
        "envelope-fields + advanced-shared",
        &format!("{SETTINGS_COPY_PATH}'s WORKFLOW_FIELDS_IN_FILE"),
    );

    // The withheld half. Every key the contract classified as withheld must be spoken for by the
    // "Not in the file" list, so a decision someone wrote down against a key is a decision the user
    // is actually told about.
    let withheld = prefixed_advanced("advanced-withheld");
    let claimed_absent = declared_keys("WORKFLOW_FIELDS_NOT_IN_FILE");
    let unspoken: Vec<&String> = withheld.difference(&claimed_absent).collect();
    assert!(
        unspoken.is_empty(),
        "{SETTINGS_COPY_PATH}'s WORKFLOW_FIELDS_NOT_IN_FILE does not account for {unspoken:?}, \
         which the `advanced-withheld` table in {DOC_PATH} says is deliberately dropped. A \
         withheld key nobody mentions is a decision the user never hears about."
    );

    // And the direction that is a leak rather than an omission: nothing the copy claims is ABSENT
    // may appear in either table of things that travel. This is what would fire the day someone
    // moved `quantTier` from withheld to shared and left the paragraph promising it stays home.
    let contradicted: Vec<&String> = claimed_absent.intersection(&shared).collect();
    assert!(
        contradicted.is_empty(),
        "{SETTINGS_COPY_PATH} tells the user {contradicted:?} are NOT in the file, but {DOC_PATH} \
         lists them among the fields that travel. The copy is claiming a privacy the allow-list \
         does not deliver, which is the one direction this contract must never drift in."
    );
}

#[test]
fn the_settings_copy_names_the_save_without_workflow_control_once() {
    // Three surfaces name this control — the context menu, the button beside Save As, and the copy
    // that tells you to use it — and they said three different things ("Save without workflow",
    // "Save a copy without the workflow…", and a third in the document). A user told to look for a
    // control that is not spelled that way anywhere is a user who concludes it does not exist.
    let source = read_repo_file(SETTINGS_COPY_PATH);
    let anchor = "SAVE_WITHOUT_WORKFLOW_LABEL = \"";
    let start = source
        .find(anchor)
        .unwrap_or_else(|| panic!("{SETTINGS_COPY_PATH} no longer declares {anchor:?}"))
        + anchor.len();
    let label = &source[start
        ..start
            + source[start..]
                .find('"')
                .unwrap_or_else(|| panic!("{SETTINGS_COPY_PATH}: the label is never closed"))];
    assert!(
        !label.is_empty(),
        "{SETTINGS_COPY_PATH}: the label parsed as empty"
    );
    assert!(
        doc().contains(label),
        "{DOC_PATH} must name the control exactly as the UI does ({label:?}); the document is what \
         a user is sent to when the copy says to use it"
    );
}

#[test]
fn the_settings_copy_derives_its_doc_link_from_the_producer_url() {
    // The link in the app and the link a shared file carries in `producer.url` have to be the same
    // repository, or "what travels, exactly" sends someone somewhere the envelope does not vouch
    // for. It used to be a hardcoded URL under a comment CLAIMING it was rooted at PRODUCER_URL,
    // with nothing deriving or checking that — so the full prefix is pinned here, not just the
    // path substring.
    let source = read_repo_file(SETTINGS_COPY_PATH);
    let declared = format!("PRODUCER_URL = \"{PRODUCER_URL}\";");
    assert!(
        source.contains(&declared),
        "{SETTINGS_COPY_PATH} must declare {declared:?} — the web copy's idea of the project URL \
         and the one the envelope carries are the same fact"
    );
    let derived = format!(
        "WORKFLOW_SHARE_DOC_URL = `${{PRODUCER_URL}}/blob/main/{}`;",
        DOC_PATH.replace('\\', "/")
    );
    assert!(
        source.contains(&derived),
        "{SETTINGS_COPY_PATH} must derive the doc link as {derived:?} rather than writing the URL \
         out beside PRODUCER_URL, where the two can drift apart silently"
    );
}

#[test]
fn the_settings_copy_names_exactly_the_path_exempt_prose_fields() {
    // The settings toggle's copy is the most user-visible place a claim about this contract can be
    // wrong, and "your prompt travels as you wrote it" is the claim a user acts on before sharing.
    // So the six fields are not written out as a sentence in the JSX — they are rendered from a
    // declared list, and this pins that list against the same `prose-fields` table the sanitizer
    // itself is pinned to. A seventh prose field added to `PROSE_KEYS` therefore fails three
    // things: the doc, the sanitizer's own lint, and the copy that would otherwise have gone on
    // naming six.
    let listed = declared_keys("EMBEDDED_PROSE_FIELDS");
    let documented = pinned_set(&doc(), "prose-fields", 0);

    let missing: Vec<&String> = documented.difference(&listed).collect();
    assert!(
        missing.is_empty(),
        "{SETTINGS_COPY_PATH} does not list {missing:?}, which the `prose-fields` table in \
         {DOC_PATH} says travels exactly as the user typed it. The settings copy would be telling \
         someone their prompt is safe while a field that carries a filesystem path is not named."
    );
    let invented: Vec<&String> = listed.difference(&documented).collect();
    assert!(
        invented.is_empty(),
        "{SETTINGS_COPY_PATH} lists {invented:?} as travelling verbatim, which the `prose-fields` \
         table in {DOC_PATH} does not. The copy is claiming something about the file that is not \
         true of it."
    );

    // And the copy points at the document this test reads, so the "what travels, exactly" link is
    // the contract rather than a page that happens to exist. The FULL URL, prefix included, is
    // pinned by `the_settings_copy_derives_its_doc_link_from_the_producer_url`.
    let link_target = DOC_PATH.replace('\\', "/");
    assert!(
        read_repo_file(SETTINGS_COPY_PATH).contains(&link_target),
        "{SETTINGS_COPY_PATH} must link to {link_target} — the copy is a summary, and the summary \
         has to be able to send the user to the exact list"
    );
}

/// The value of a `const NAME = "…";` string in the settings copy, unescaped enough to compare.
///
/// The person-replacement bullet is the one claim in that file no key-level lint can reach, so it
/// has to be read as PROSE. Panics when the constant is gone, for the reason every parser here
/// does: a lint that finds nothing passes against a file that was emptied.
fn settings_copy_constant(name: &str) -> String {
    let source = read_repo_file(SETTINGS_COPY_PATH);
    let anchor = format!("const {name} =");
    let start = source.find(&anchor).unwrap_or_else(|| {
        panic!("{SETTINGS_COPY_PATH} no longer declares `{anchor}`; teach this lint where it went")
    }) + anchor.len();
    let rest = &source[start..];
    let open = rest
        .find('"')
        .unwrap_or_else(|| panic!("{SETTINGS_COPY_PATH}: `{name}` has no string literal"));
    let close = rest[open + 1..]
        .find('"')
        .unwrap_or_else(|| panic!("{SETTINGS_COPY_PATH}: `{name}`'s string is never closed"))
        + open
        + 1;
    let value = rest[open + 1..close].to_owned();
    assert!(
        !value.trim().is_empty(),
        "{SETTINGS_COPY_PATH}: `{name}` parsed to an empty string"
    );
    value
}

/// The person-replacement bullet says only what a real `replace_person` envelope actually withholds
/// (sc-15956 review).
///
/// # The failure this exists for
///
/// The bullet read *"anything about a person replacement — the selected track, the person's name,
/// the original file name, the mask images, or which replacement mode ran"*, rendered into the
/// **"Not in the file"** list. Every item after the dash was true. The leading clause was not:
/// `mode` is in the **"In the file"** list, and a person-replacement run writes it verbatim as
/// `replace_person`. `model` says it a second time — `wan_2_2_vace_fun_14b` is a replacement
/// engine. So the file announces the TECHNIQUE while the copy said it announced nothing, which is
/// the one direction `workflowEmbed.js`'s own header forbids: claiming more privacy than the
/// allow-list delivers.
///
/// # Why the key-level pins could not catch it
///
/// `the_settings_copy_accounts_for_every_shared_and_withheld_field` joins the copy to the doc on
/// KEYS, and `mode` is not a key in either of the two lists it compares — it is a shared field the
/// copy already declares correctly. The error was in a sentence about a different key. So this
/// asserts the sentence against the bytes of a real envelope instead.
#[test]
fn the_settings_copy_does_not_overclaim_about_person_replacement() {
    // A real Video Studio person-replacement payload, with every person-shaped value the builder
    // can be handed.
    let payload = json!({
        "mode": "replace_person",
        "model": "wan_2_2_vace_fun_14b",
        "prompt": "the same shot, continuous motion",
        "duration": 5.0,
        "fps": 24,
        "quality": "balanced",
        "personTrackId": "track_9f3c1b2a",
        "replacementMode": "full_person_keep_outfit",
        "advanced": {
            "motion": "static",
            "replacementModeLabel": "Full Person, Keep Outfit",
            "selectedPersonTrack": {
                "name": "Sarah Whitfield",
                "sourceDisplayName": "sarah-audition-take3.mov",
                "frames": [{ "mask": "D:/Clients/Acme/masks/0001.png" }]
            }
        }
    })
    .as_object()
    .expect("an object")
    .clone();
    let share = sceneworks_core::workflow_share::build_video_workflow_share_from(
        &sceneworks_core::workflow_share::WorkflowAssetFacts {
            mode: "replace_person".to_owned(),
            model: "wan_2_2_vace_fun_14b".to_owned(),
            prompt: "the same shot, continuous motion".to_owned(),
            negative_prompt: String::new(),
            seed: 7,
            width: Some(832),
            height: Some(480),
        },
        &payload,
    );
    let bytes = serde_json::to_string(&share).expect("the envelope serializes");

    // Half one: the withholding is real. These are the things the bullet promises are absent.
    for withheld in [
        "track_9f3c1b2a",
        "Sarah Whitfield",
        "sarah-audition-take3.mov",
        "0001.png",
        "full_person_keep_outfit",
        "Full Person, Keep Outfit",
        "selectedPersonTrack",
    ] {
        assert!(
            !bytes.contains(withheld),
            "`{withheld}` reached a real person-replacement envelope: {bytes}"
        );
    }

    // Half two: and the technique is NOT withheld. This is what makes the old leading clause false,
    // asserted rather than argued — if this ever stops holding, the copy may go back to the
    // stronger sentence, and this assert is what will say so.
    assert_eq!(share.mode, "replace_person");
    assert!(
        bytes.contains("\"mode\":\"replace_person\""),
        "the envelope must still carry the generation mode verbatim: {bytes}"
    );
    assert!(
        bytes.contains("wan_2_2_vace_fun_14b"),
        "and the model, which names a replacement engine: {bytes}"
    );

    // Half three: the sentence a user reads matches both halves.
    let copy = settings_copy_constant("N_PERSON");
    assert!(
        !copy.contains("anything about a person replacement"),
        "{SETTINGS_COPY_PATH}: `N_PERSON` claims the file says nothing about a replacement, and \
         `mode` says `replace_person` in every one: {copy}"
    );
    for named in [
        "who was replaced",
        "the selected track",
        "the person's name",
        "the original file name",
        "the mask images",
    ] {
        assert!(
            copy.contains(named),
            "{SETTINGS_COPY_PATH}: `N_PERSON` no longer names {named:?}, which the envelope really \
             does withhold: {copy}"
        );
    }
    assert!(
        copy.contains("generation mode") && copy.contains("do travel"),
        "{SETTINGS_COPY_PATH}: `N_PERSON` must say that the mode and the model travel, because \
         they do — an absence list that omits this is the overclaim this test exists for: {copy}"
    );
}

// ---------------------------------------------------------------------------
// The ceilings, found by experiment
// ---------------------------------------------------------------------------

/// A repeated non-path, control-free label of `chars` characters.
fn label(chars: usize) -> String {
    "m".repeat(chars)
}

fn lora_entries(count: usize) -> Value {
    Value::Array(
        (0..count)
            .map(|index| json!({ "name": format!("lora {index}"), "weight": 0.5 }))
            .collect(),
    )
}

fn envelope_with_advanced(key: &str, value: Value) -> WorkflowShare {
    let mut payload = payload_fixture();
    advanced_of(&mut payload).insert(key.to_owned(), value);
    build(&payload)
}

fn pose_entries(count: usize, points: usize) -> Value {
    Value::Array(
        (0..count)
            .map(|_| json!({ "keypoints": vec![1; points] }))
            .collect(),
    )
}

fn phase_entries(count: usize) -> Value {
    Value::Array(
        (0..count)
            .map(|index| json!({ "steps": index + 1, "guidance": 3.5 }))
            .collect(),
    )
}

/// Parse an envelope carrying `inputs`, which is the only direction that can exceed the input cap
/// (the builder emits at most one entry per kind).
fn parsed_with_inputs(count: usize) -> WorkflowShare {
    let inputs: Vec<Value> = (0..count)
        .map(|index| json!({ "kind": INPUT_KINDS[index % INPUT_KINDS.len()], "count": 1 }))
        .collect();
    let value = json!({
        "sceneworksWorkflow": WORKFLOW_KIND_IMAGE,
        "schemaVersion": WORKFLOW_SHARE_SCHEMA_VERSION,
        "producer": { "name": PRODUCER_NAME, "url": PRODUCER_URL, "version": "1.0.0" },
        "mode": "edit_image",
        "model": "z_image_turbo",
        "prompt": "a lighthouse",
        "inputs": inputs
    });
    parse_workflow_share(&value).expect("a bounded envelope parses")
}

/// What the document's last column says happens to a value that is over a ceiling.
///
/// Read out of the table rather than assumed, which is the difference between a lint and a
/// decoration: while this was hard-coded, flipping `LABEL_MAX_CHARS`'s cell from "**Dropped**, not
/// truncated" to "Truncated at the ceiling, exactly like prose" left every test in this file green
/// — the document would have promised a slug survives cut in half, and nothing would have argued.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Overflow {
    /// The value survives, cut to the ceiling.
    Truncated,
    /// The value — or the whole collection — goes.
    Dropped,
    /// Nothing partial happens: the read or the write fails outright.
    Refused,
}

/// The one verb a "what happens at the limit" cell claims.
///
/// Markdown emphasis is stripped and a NEGATED verb ("not truncated", "never dropped") is removed
/// before the search, so a cell may usefully contrast its behaviour with the one it is not — which
/// the `LABEL_MAX_CHARS` row does, and which is exactly the phrasing a naive `contains` would
/// misread. Exactly one verb must survive: a cell naming none has said nothing checkable, and a
/// cell naming two has said two things.
fn overflow_behaviour(name: &str, cell: &str) -> Overflow {
    const VERBS: &[(&str, Overflow)] = &[
        ("truncat", Overflow::Truncated),
        ("drop", Overflow::Dropped),
        ("refus", Overflow::Refused),
    ];
    let mut text = cell.to_ascii_lowercase().replace(['*', '`'], " ");
    for negation in ["not ", "never ", "rather than ", "instead of "] {
        for (stem, _) in VERBS {
            text = text.replace(&format!("{negation}{stem}"), " ");
        }
    }
    let claimed: BTreeSet<Overflow> = VERBS
        .iter()
        .filter(|(stem, _)| text.contains(stem))
        .map(|(_, behaviour)| *behaviour)
        .collect();
    let mut claimed = claimed.into_iter();
    let (Some(behaviour), None) = (claimed.next(), claimed.next()) else {
        panic!(
            "{DOC_PATH}: the `{name}` row's \"What happens at the limit\" cell must say exactly \
             one of truncated / dropped / refused, and {cell:?} says {} of them. That column is \
             asserted, not decorative.",
            VERBS.iter().filter(|(stem, _)| text.contains(stem)).count()
        )
    };
    behaviour
}

/// Assert an over-ceiling STRING did what the document's last column says it did.
///
/// `surviving` is measured in the ceiling's own unit — bytes for prose, characters for a label.
fn assert_string_overflow(name: &str, behaviour: Overflow, surviving: usize, documented: usize) {
    match behaviour {
        Overflow::Truncated => assert_eq!(
            surviving, documented,
            "{DOC_PATH} documents `{name}` as TRUNCATED at the ceiling, but an over-long value \
             came back {surviving} long instead of {documented}"
        ),
        Overflow::Dropped => assert_eq!(
            surviving, 0,
            "{DOC_PATH} documents `{name}` as DROPPED rather than truncated, but {surviving} of \
             an over-long value survived"
        ),
        Overflow::Refused => panic!(
            "{DOC_PATH} documents `{name}` as a refusal, but it is a per-value bound that reduces \
             the value in place — nothing about the envelope is refused"
        ),
    }
}

/// Assert an over-cap COLLECTION did what the document's last column says it did.
fn assert_collection_overflow(
    name: &str,
    behaviour: Overflow,
    still_present: bool,
    omitted_gained: bool,
    omitted_field: &str,
) {
    match behaviour {
        Overflow::Dropped => {
            assert!(
                !still_present,
                "{DOC_PATH} documents `{name}` as dropping the collection whole, but it survived"
            );
            assert!(
                omitted_gained,
                "{DOC_PATH} documents `{name}` as adding `{omitted_field}` to `omitted`, and it \
                 did not — a silent drop is the failure the marker exists to prevent"
            );
        }
        Overflow::Truncated => {
            assert!(
                still_present,
                "{DOC_PATH} documents `{name}` as TRUNCATING at the cap, but the over-cap \
                 collection was dropped whole"
            );
            assert!(
                !omitted_gained,
                "{DOC_PATH} documents `{name}` as truncating, but `omitted` gained \
                 `{omitted_field}` — which is what a DROP records"
            );
        }
        Overflow::Refused => panic!(
            "{DOC_PATH} documents `{name}` as a refusal, but an over-cap collection is reduced, \
             not refused: the rest of the envelope is still written"
        ),
    }
}

/// Assert a ceiling the document describes as a refusal is documented as one.
fn assert_refusal(name: &str, behaviour: Overflow) {
    assert_eq!(
        behaviour,
        Overflow::Refused,
        "{DOC_PATH} documents `{name}` as {behaviour:?}, but it refuses the whole read or write — \
         nothing partial is kept, so the last column may not promise a survivor"
    );
}

/// Every ceiling the document quotes, verified against the limit a user actually hits — both the
/// number and, from the last column, what overflowing it does.
///
/// Named exhaustively rather than looked up, so a ceiling added to the document that nobody taught
/// this test about fails instead of being waved through.
fn verify_ceiling(name: &str, documented: usize, behaviour: Overflow) {
    match name {
        "WORKFLOW_SHARE_MAX_BYTES" => {
            assert_eq!(documented, WORKFLOW_SHARE_MAX_BYTES);
            assert_refusal(name, behaviour);
        }
        "MAX_WORKFLOW_TEXT_BYTES" => {
            assert_eq!(documented, MAX_WORKFLOW_TEXT_BYTES);
            assert_refusal(name, behaviour);
        }
        "MAX_METADATA_BYTES" => {
            assert_eq!(documented, MAX_METADATA_BYTES);
            assert_refusal(name, behaviour);
        }
        "PROSE_MAX_BYTES" => {
            let mut payload = payload_fixture();
            payload.insert("prompt".to_owned(), json!("x".repeat(documented)));
            assert_eq!(
                build(&payload).prompt.len(),
                documented,
                "a prompt of exactly the documented prose ceiling must survive whole"
            );
            payload.insert("prompt".to_owned(), json!("x".repeat(documented * 2)));
            assert_string_overflow(name, behaviour, build(&payload).prompt.len(), documented);
        }
        "LABEL_MAX_CHARS" => {
            let mut payload = payload_fixture();
            payload.insert("model".to_owned(), json!(label(documented)));
            assert_eq!(
                build(&payload).model.chars().count(),
                documented,
                "a label of exactly the documented ceiling must survive whole"
            );
            payload.insert("model".to_owned(), json!(label(documented + 1)));
            assert_string_overflow(
                name,
                behaviour,
                build(&payload).model.chars().count(),
                documented,
            );
        }
        "MAX_SHARE_LORAS" => {
            let mut payload = payload_fixture();
            payload.insert("loras".to_owned(), lora_entries(documented));
            let at_cap = build(&payload);
            assert_eq!(at_cap.loras.len(), documented);
            assert!(at_cap.omitted.is_empty());
            payload.insert("loras".to_owned(), lora_entries(documented + 1));
            let over = build(&payload);
            assert_collection_overflow(
                name,
                behaviour,
                !over.loras.is_empty(),
                over.omitted.iter().any(|field| field == "loras"),
                "loras",
            );
        }
        "MAX_SHARE_INPUTS" => {
            let at_cap = parsed_with_inputs(documented);
            assert_eq!(at_cap.inputs.len(), documented);
            assert!(at_cap.omitted.is_empty());
            let over = parsed_with_inputs(documented + 1);
            assert_collection_overflow(
                name,
                behaviour,
                !over.inputs.is_empty(),
                over.omitted.iter().any(|field| field == "inputs"),
                "inputs",
            );
        }
        "MAX_SHARE_PHASES" => {
            let at_cap = envelope_with_advanced("phases", phase_entries(documented));
            assert!(at_cap.advanced.contains_key("phases"));
            assert!(at_cap.omitted.is_empty());
            let over = envelope_with_advanced("phases", phase_entries(documented + 1));
            assert_collection_overflow(
                name,
                behaviour,
                over.advanced.contains_key("phases"),
                over.omitted.iter().any(|field| field == "advanced.phases"),
                "advanced.phases",
            );
        }
        "MAX_SHARE_POSES" => {
            let at_cap = envelope_with_advanced("poses", pose_entries(documented, 2));
            assert!(at_cap.advanced.contains_key("poses"));
            assert!(at_cap.omitted.is_empty());
            let over = envelope_with_advanced("poses", pose_entries(documented + 1, 2));
            assert_collection_overflow(
                name,
                behaviour,
                over.advanced.contains_key("poses"),
                over.omitted.iter().any(|field| field == "advanced.poses"),
                "advanced.poses",
            );
        }
        "MAX_SHARE_POSE_SLOTS" => {
            let at_cap = envelope_with_advanced("poses", pose_entries(1, documented));
            assert!(at_cap.advanced.contains_key("poses"));
            assert!(at_cap.omitted.is_empty());
            let over = envelope_with_advanced("poses", pose_entries(1, documented + 1));
            assert_collection_overflow(
                name,
                behaviour,
                over.advanced.contains_key("poses"),
                over.omitted.iter().any(|field| field == "advanced.poses"),
                "advanced.poses",
            );
        }
        other => panic!(
            "{DOC_PATH} quotes a ceiling `{other}` that \
             crates/sceneworks-core/tests/workflow_share_doc.rs does not verify. A number in that \
             document with nothing checking it is exactly what this lint exists to refuse — teach \
             `verify_ceiling` how to measure it."
        ),
    }
}

/// The ceilings this test knows how to measure. Asserted to be a subset of what the document
/// lists, so a row deleted from the document fails as loudly as one added to it.
const VERIFIED_CEILINGS: &[&str] = &[
    "WORKFLOW_SHARE_MAX_BYTES",
    "PROSE_MAX_BYTES",
    "LABEL_MAX_CHARS",
    "MAX_SHARE_LORAS",
    "MAX_SHARE_INPUTS",
    "MAX_SHARE_PHASES",
    "MAX_SHARE_POSES",
    "MAX_SHARE_POSE_SLOTS",
    "MAX_WORKFLOW_TEXT_BYTES",
    "MAX_METADATA_BYTES",
];

/// The `ceilings` table as `name -> value`, with the quoted numbers parsed.
fn documented_ceilings(doc: &str) -> BTreeMap<String, usize> {
    pinned_rows(doc, "ceilings")
        .into_iter()
        .map(|row| {
            let name = code(&row[0]);
            let raw = code(&row[1]).replace(',', "");
            let value = raw.parse::<usize>().unwrap_or_else(|error| {
                panic!("{DOC_PATH}: the `{name}` ceiling is not a number ({raw:?}): {error}")
            });
            (name, value)
        })
        .collect()
}

#[test]
fn the_documented_ceilings_are_the_real_limits() {
    let doc = doc();
    let documented = documented_ceilings(&doc);
    // The last column, which says what overflowing the ceiling DOES. Read here and passed down, so
    // the assertion each row gets is the one the document promised rather than one this file
    // remembers.
    let behaviours: BTreeMap<String, Overflow> = pinned_rows(&doc, "ceilings")
        .into_iter()
        .map(|row| {
            let name = code(&row[0]);
            assert_eq!(
                row.len(),
                4,
                "{DOC_PATH}: the `{name}` ceiling row must have a \"What happens at the limit\" \
                 column"
            );
            let behaviour = overflow_behaviour(&name, &row[3]);
            (name, behaviour)
        })
        .collect();

    assert_same_set(
        &documented.keys().cloned().collect(),
        &VERIFIED_CEILINGS
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        "ceilings",
        "the bounds this lint measures",
    );
    for (name, value) in &documented {
        let behaviour = behaviours
            .get(name)
            .copied()
            .expect("every ceiling row was read for its behaviour");
        verify_ceiling(name, *value, behaviour);
    }
}

// ---------------------------------------------------------------------------
// Claims made in prose, outside any pinned block
// ---------------------------------------------------------------------------

/// The privacy callout restates the prose ceiling in words, above the table that measures it.
///
/// A number restated is a number that drifts: changing the callout to "64 KiB" left every other
/// test in this file green, under a heading promising the user their prompt travels as written. So
/// the sentence is read back and pinned to the `ceilings` row, which
/// [`the_documented_ceilings_are_the_real_limits`] measures against the sanitizer.
#[test]
fn the_privacy_callout_quotes_the_real_prose_ceiling() {
    let doc = doc();
    let bytes = documented_ceilings(&doc)
        .get("PROSE_MAX_BYTES")
        .copied()
        .expect("the `ceilings` table names PROSE_MAX_BYTES");
    assert_eq!(
        bytes % 1024,
        0,
        "{DOC_PATH}: the prose ceiling is no longer a whole number of KiB, so the callout cannot \
         quote it as one — reword both together"
    );
    let quoted = format!("{} KiB prose ceiling", bytes / 1024);
    assert!(
        flowed(&doc).contains(&quoted),
        "{DOC_PATH}: the privacy callout must say {quoted:?}. It is what a user reads before \
         deciding to share an image, and the `ceilings` table says the bound is {bytes} bytes."
    );
}

/// The callout counts its own fields in prose ("These six fields"), and so does the path bullet
/// further down ("the six prose fields"). Both are pinned to the row count of the table itself.
#[test]
fn the_privacy_callout_counts_its_own_prose_fields() {
    let doc = doc();
    let count = pinned_rows(&doc, "prose-fields").len();
    let word = number_word(count);
    let flowed = flowed(&doc);
    for sentence in [
        format!("These {word} fields are recorded as authored"),
        format!("the {word} prose fields"),
    ] {
        assert!(
            flowed.contains(&sentence),
            "{DOC_PATH} must say {sentence:?}: the `prose-fields` table has {count} rows, and a \
             count written out in prose beside a table is the first thing to go stale"
        );
    }
}

/// The pose coordinate arrays, named in prose twice and pinned to `POSE_FIELDS` both times.
///
/// Neither mention is inside a pinned block, and both are a promise about what leaves the machine.
/// Removing `"face"` from `POSE_FIELDS` while the document kept saying "keypoints / hands / face"
/// left this file green.
#[test]
fn the_doc_names_exactly_the_pose_fields_that_travel() {
    let doc = doc();
    let actual: BTreeSet<String> = POSE_FIELDS
        .iter()
        .map(|field| (*field).to_owned())
        .collect();

    let bullet = phrase(
        &flowed(&doc),
        "A pose selection travels as coordinate arrays (",
        ")",
    );
    assert_same_set(
        &backticked(&bullet),
        &actual,
        "pose bullet in \"What does not travel\"",
        "`POSE_FIELDS`",
    );

    let row = pinned_rows(&doc, "advanced-shared")
        .into_iter()
        .find(|row| code(&row[0]) == "poses")
        .expect("the `advanced-shared` table has a `poses` row");
    let described = phrase(&row[2], "reduced to the ", " coordinate arrays");
    assert_same_set(
        &backticked(&described),
        &actual,
        "`poses` row of the `advanced-shared` table",
        "`POSE_FIELDS`",
    );
}

// ---------------------------------------------------------------------------
// The two registries a contributor has to fill in correctly
// ---------------------------------------------------------------------------

/// Every `const fn … -> AdvancedKeyRule` constructor in the module, by name.
///
/// Parsed rather than listed, so a sixth helper nobody documented fails. The return type is what
/// identifies one: `workflow_share.rs` has other `const fn`s in principle, and a helper that does
/// not build a rule is not a helper this section is about.
fn rule_constructors(source: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for chunk in source.split("const fn ").skip(1) {
        let Some(open) = chunk.find('(') else {
            continue;
        };
        let name = chunk[..open].trim();
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            continue;
        }
        let Some(body) = chunk.find('{') else {
            continue;
        };
        if chunk[..body].contains("-> AdvancedKeyRule") {
            out.insert(name.to_owned());
        }
    }
    out
}

/// The contributor section tells someone which constructor to reach for. While it named only
/// `allow` and `deny`, following it literally for a key on any non-studio builder produced a rule
/// tagged `StudioBuilder` and failed three lints with no hint in the document that a tag existed.
#[test]
fn the_doc_lists_exactly_the_rule_helpers() {
    let documented: BTreeSet<String> = pinned_column(&doc(), "rule-helpers", 0)
        .into_iter()
        .map(|cell| {
            cell.split('(')
                .next()
                .expect("split yields at least one part")
                .trim()
                .to_owned()
        })
        .collect();
    assert_same_set(
        &documented,
        &rule_constructors(&read_repo_file(SHARE_SOURCE_PATH)),
        "rule-helpers",
        "the rule constructors in `workflow_share.rs`",
    );
}

/// The field names of a `struct` declared as `declaration`, up to the line that closes it.
fn struct_fields(source: &str, declaration: &str) -> BTreeSet<String> {
    let start = source
        .find(declaration)
        .unwrap_or_else(|| panic!("{SHARE_SOURCE_PATH} no longer declares `{declaration}`"))
        + declaration.len();
    let mut out = BTreeSet::new();
    for line in source[start..].lines() {
        let line = line.trim();
        if line == "}" {
            return out;
        }
        if line.starts_with("//") || line.starts_with('#') || line.is_empty() {
            continue;
        }
        let line = line.strip_prefix("pub ").unwrap_or(line);
        let Some(colon) = line.find(':') else {
            continue;
        };
        let name = line[..colon].trim();
        if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            out.insert(name.to_owned());
        }
    }
    panic!("{SHARE_SOURCE_PATH}: `{declaration}` is never closed by a line holding only `}}`")
}

/// A registry entry a contributor is told to add has eight fields, and getting `anchors` or
/// `minimum_keys` wrong is how a lint reads zero keys and passes. The document listed none of
/// them, so this pins the list it now gives against the struct itself.
#[test]
fn the_doc_lists_exactly_the_builder_registry_fields() {
    assert_same_set(
        &pinned_set(&doc(), "builder-fields", 0),
        &struct_fields(
            &read_repo_file(SHARE_SOURCE_PATH),
            "pub struct AdvancedBuilder {",
        ),
        "builder-fields",
        "the fields of `AdvancedBuilder`",
    );
}
