//! Contract tests for the PNG `iTXt` workflow codec (sc-15947, epic 15945).
//!
//! The adversarial corpus lives in the module's own `#[cfg(test)]` tests, because framing a
//! deliberately broken chunk needs the private encoder. What is here is the outside-facing
//! contract: the round trip, the two halves of the byte-identity guarantee sc-15948 rests on (the
//! opt-out is `save_with_format` exactly, and opting *in* is that same file plus the chunk and
//! nothing else), the non-ASCII prompt that justifies `iTXt` over `tEXt`, and the measured per-image
//! cost.
//!
//! Value comparisons throughout, never serialized text. `serde_json::Map` is a `BTreeMap` (sorted)
//! or an `IndexMap` (insertion-ordered) depending on the `preserve_order` feature, which Cargo
//! unifies across the workspace — so the same envelope serializes its keys in a different ORDER
//! under `cargo test -p sceneworks-core` than under the workspace build the `parity` job runs. The
//! CHUNK contents therefore differ in byte order between the two configurations while being the
//! same envelope, which is why nothing here asserts on chunk bytes (sc-15946).

use std::fs;
use std::path::{Path, PathBuf};

use image::{ImageFormat, Rgb, RgbImage};
use sceneworks_core::contracts::{Asset, JsonObject};
use sceneworks_core::workflow_png::{
    read_workflow_chunk, read_workflow_chunk_file, workflow_chunk_size, write_workflow_chunk,
    MAX_WORKFLOW_TEXT_BYTES, WORKFLOW_CHUNK_KEYWORD,
};
use sceneworks_core::workflow_share::{build_workflow_share, WorkflowShare};
use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// The sc-15946 golden envelope — a full Krea edit with LoRAs, inputs, an upscale pass and twelve
/// allow-listed `advanced` keys. Reused rather than re-invented so "representative recipe" means
/// the same thing in both stories.
fn golden_envelope() -> WorkflowShare {
    let path = repo_root()
        .join("tests")
        .join("fixtures")
        .join("workflow_share")
        .join("image-workflow-share.json");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()))
}

/// A generated asset's sidecar, in the shape `project_store::build_image_sidecar_parts` writes.
fn asset_fixture(prompt: &str) -> Asset {
    serde_json::from_value(json!({
        "schemaVersion": 1,
        "id": "asset_9f2c",
        "projectId": "project_7a10",
        "generationSetId": "genset_31bb",
        "type": "image",
        "displayName": "Round trip #1",
        "createdAt": "2026-07-29T13:04:11Z",
        "file": {
            "path": "assets/images/2026-07-29_z_image_turbo_round_trip_0001.png",
            "mimeType": "image/png",
            "width": 9,
            "height": 7,
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
            "seed": 880412,
            "loras": [],
            "stylePreset": null,
            "normalizedSettings": {},
            "rawAdapterSettings": {}
        },
        "lineage": { "parents": [], "sourceAssetId": null, "sourceTimestamp": null, "jobId": "job_1" }
    }))
    .expect("asset fixture parses")
}

fn payload_fixture(prompt: &str, negative_prompt: &str) -> JsonObject {
    json!({
        "projectId": "project_7a10",
        "mode": "text_to_image",
        "prompt": prompt,
        "negativePrompt": negative_prompt,
        "model": "z_image_turbo",
        "count": 1,
        "width": 9,
        "height": 7,
        "advanced": { "steps": 8, "sampler": "euler", "guidanceScale": 3.5 }
    })
    .as_object()
    .cloned()
    .expect("payload fixture is an object")
}

/// A small deterministic image. Nine by seven so the row stride is not a power of two.
fn rgb_fixture() -> RgbImage {
    RgbImage::from_fn(9, 7, |x, y| {
        Rgb([
            u8::try_from(x * 20).expect("fits"),
            u8::try_from(y * 30).expect("fits"),
            u8::try_from((x + y) * 10).expect("fits"),
        ])
    })
}

fn as_value(share: &WorkflowShare) -> Value {
    serde_json::to_value(share).expect("the envelope serializes")
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

#[test]
fn the_representative_envelope_round_trips_through_a_png() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("with-workflow.png");
    let original = golden_envelope();

    write_workflow_chunk(&rgb_fixture(), &path, Some(&original)).expect("writes the chunk");
    let read = read_workflow_chunk_file(&path)
        .expect("reads back")
        .expect("carries a workflow");

    assert_eq!(
        as_value(&read),
        as_value(&original),
        "the envelope changed on its way through the PNG"
    );
}

#[test]
fn a_built_envelope_round_trips_through_a_png() {
    // The builder's output, not a hand-written envelope: this is the shape sc-15948 will actually
    // hand the writer.
    let built = build_workflow_share(
        &asset_fixture("a lighthouse in heavy fog"),
        &payload_fixture("a lighthouse in heavy fog", "text, watermark"),
    );
    let mut bytes = Vec::new();
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("built.png");
    write_workflow_chunk(&rgb_fixture(), &path, Some(&built)).expect("writes");
    bytes.extend_from_slice(&fs::read(&path).expect("reads the file back"));

    let read = read_workflow_chunk(&bytes)
        .expect("reads back")
        .expect("carries a workflow");
    assert_eq!(as_value(&read), as_value(&built));

    // Both public entry points must agree — one streams the file, the other takes a slice.
    let from_file = read_workflow_chunk_file(&path)
        .expect("reads back from the path")
        .expect("carries a workflow");
    assert_eq!(as_value(&from_file), as_value(&read));
}

#[test]
fn the_chunk_does_not_disturb_the_pixels() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("pixels.png");
    let original = rgb_fixture();
    write_workflow_chunk(&original, &path, Some(&golden_envelope())).expect("writes");

    // A shared image must still be a perfectly ordinary PNG to every other reader.
    let decoded = image::open(&path)
        .expect("`image` decodes the chunked PNG")
        .to_rgb8();
    assert_eq!(decoded.dimensions(), original.dimensions());
    assert_eq!(decoded.as_raw(), original.as_raw());
}

// ---------------------------------------------------------------------------
// The opt-out is byte-identical
// ---------------------------------------------------------------------------

#[test]
fn the_none_path_is_byte_identical_to_save_with_format() {
    // The proof sc-15948 rests on: a user who turns embedding off gets the file they get today,
    // to the byte. Asserted against a real `save_with_format` write of the same pixel buffer
    // rather than by reading the implementation, so "we call the same function" stays true instead
    // of being true only on the day it was written.
    let directory = tempfile::tempdir().expect("temp dir");
    let via_codec = directory.path().join("via-codec.png");
    let via_image = directory.path().join("via-image.png");
    let rgb = rgb_fixture();

    write_workflow_chunk(&rgb, &via_codec, None).expect("writes without a chunk");
    rgb.save_with_format(&via_image, ImageFormat::Png)
        .expect("the current encoder writes");

    let codec_bytes = fs::read(&via_codec).expect("reads");
    let image_bytes = fs::read(&via_image).expect("reads");
    assert_eq!(
        codec_bytes,
        image_bytes,
        "write_workflow_chunk(.., None) diverged from save_with_format: {} vs {} bytes",
        codec_bytes.len(),
        image_bytes.len()
    );

    // And the same pixels WITH an envelope must differ, or the test above would pass for the
    // trivial reason that nothing is ever embedded.
    let with_chunk = directory.path().join("with-chunk.png");
    write_workflow_chunk(&rgb, &with_chunk, Some(&golden_envelope())).expect("writes");
    let chunked_bytes = fs::read(&with_chunk).expect("reads");
    assert_ne!(chunked_bytes, image_bytes);
    assert!(
        chunked_bytes.len() > image_bytes.len(),
        "the embedded file must be the larger of the two"
    );
}

/// A deterministic image with enough entropy that the DEFLATE level shows up in the output size.
///
/// The fixture matters as much as the assertion here. `rgb_fixture` is 9x7 and a smooth gradient —
/// it compresses to nearly the same size at any level, which is exactly how an encoder-settings
/// divergence stayed invisible: the `Some` and `None` outputs landed a few hundred bytes apart, near
/// enough to the chunk size to look correct. Gradient plus cheap xorshift noise keeps a level-6
/// deflate roughly 2:1 away from the fast one at 1024x1024, so mixing the two up is a loud failure
/// rather than a plausible one.
fn noisy_rgb(width: u32, height: u32) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        // An integer hash, so the noise is reproducible on every platform and needs no dependency.
        let mut hash = x
            .wrapping_mul(0x9E37_79B9)
            .wrapping_add(y.wrapping_mul(0x85EB_CA6B));
        hash ^= hash >> 15;
        hash = hash.wrapping_mul(0xC2B2_AE35);
        hash ^= hash >> 13;
        let byte = |shift: u32, gradient: u32| {
            let noise = (hash >> shift) & 0xFF;
            u8::try_from((gradient.wrapping_add(noise)) & 0xFF).expect("masked to a byte")
        };
        // A gradient the filters can predict, plus noise they cannot: compressible enough to be a
        // realistic render, random enough that the level matters.
        Rgb([byte(0, x / 4), byte(8, y / 4), byte(16, (x + y) / 8)])
    })
}

/// Rebuild `png` with every `iTXt` chunk under [`WORKFLOW_CHUNK_KEYWORD`] removed, walking the
/// chunk framing by hand. Returns the remaining bytes and how many were taken out.
///
/// Deliberately does not use the `png` crate: the point is to remove the chunk without re-encoding
/// anything, so what is left is the original file's own bytes minus a slice.
fn strip_workflow_chunks(png: &[u8]) -> (Vec<u8>, usize) {
    assert!(png.len() > 8, "not a PNG");
    let mut out = png[..8].to_vec();
    let mut removed = 0;
    let mut cursor = 8;
    while cursor < png.len() {
        let length = usize::try_from(u32::from_be_bytes(
            png[cursor..cursor + 4].try_into().expect("a length field"),
        ))
        .expect("a chunk length fits usize");
        let kind = &png[cursor + 4..cursor + 8];
        let data = &png[cursor + 8..cursor + 8 + length];
        // length + type + data + CRC.
        let end = cursor + 12 + length;
        assert!(end <= png.len(), "chunk at {cursor} runs past the end");

        // The keyword is the `iTXt` payload up to its first NUL.
        let keyword = data
            .split(|byte| *byte == 0)
            .next()
            .expect("split yields one");
        if kind == b"iTXt" && keyword == WORKFLOW_CHUNK_KEYWORD.as_bytes() {
            removed += end - cursor;
        } else {
            out.extend_from_slice(&png[cursor..end]);
        }
        cursor = end;
    }
    (out, removed)
}

#[test]
fn the_some_path_is_the_none_path_plus_the_chunk() {
    // The other half of the guarantee, and the half that is easy to assert too weakly. `None` going
    // through `save_with_format` makes the opt-out structurally identical, but `Some` CANNOT — no
    // `image` encoder writes a text chunk — so it drives `png` directly and could quietly encode
    // with different settings. It did: `png::Encoder::new` defaults to a level-6 deflate while
    // `image` defaults to the fdeflate fast path, so embedding a chunk also halved the file and
    // cost ~20x the encode time.
    //
    // Asserting "the embedded file is bigger" cannot catch that, and neither can a small fixture.
    // So this strips our chunk back out of the `Some` output at the byte level and requires what
    // remains to be the `None` output exactly. Any change to compression, filtering or chunking
    // shows up as a diff no matter which direction it moves the size.
    let directory = tempfile::tempdir().expect("temp dir");
    let envelope = golden_envelope();
    let chunk_size = workflow_chunk_size(&envelope).expect("the chunk encodes");

    // Several sizes, and at least one big enough for a compression difference to dominate the
    // chunk. The row stride varies too, since that is what the adaptive filter keys off.
    for (width, height) in [(9, 7), (67, 41), (512, 512), (1024, 1024)] {
        let rgb = noisy_rgb(width, height);
        let embedded = directory.path().join(format!("some-{width}x{height}.png"));
        let plain = directory.path().join(format!("none-{width}x{height}.png"));

        write_workflow_chunk(&rgb, &embedded, Some(&envelope)).expect("writes with a chunk");
        write_workflow_chunk(&rgb, &plain, None).expect("writes without one");

        let embedded_bytes = fs::read(&embedded).expect("reads");
        let plain_bytes = fs::read(&plain).expect("reads");
        let (stripped, removed) = strip_workflow_chunks(&embedded_bytes);

        assert_eq!(
            removed, chunk_size,
            "at {width}x{height} the stripper removed {removed} bytes but the chunk measures \
             {chunk_size}"
        );
        assert_eq!(
            stripped.len(),
            plain_bytes.len(),
            "at {width}x{height} the two arms disagree on encoded size by {} bytes — the chunk is \
             {chunk_size} and was already removed, so this is an ENCODER difference, not the chunk",
            stripped.len().abs_diff(plain_bytes.len())
        );
        assert!(
            stripped == plain_bytes,
            "at {width}x{height} the Some output is not the None output plus the chunk: the bytes \
             differ after stripping the {chunk_size}-byte chunk"
        );
        // Stated the other way round as well, because this is the number sc-15948 quotes: the file
        // grows by the chunk and by nothing else.
        assert_eq!(
            embedded_bytes.len(),
            plain_bytes.len() + chunk_size,
            "at {width}x{height} embedding cost {} bytes, not the {chunk_size}-byte chunk",
            embedded_bytes.len() - plain_bytes.len()
        );

        // And the fixture must actually be compression-sensitive at the larger sizes, or the
        // assertions above would hold for an image where every level produces the same bytes.
        if width >= 512 {
            let raw = rgb.as_raw().len();
            assert!(
                plain_bytes.len() * 5 > raw * 2,
                "at {width}x{height} the fixture compressed to {} of {raw} raw bytes — too \
                 compressible to distinguish deflate levels",
                plain_bytes.len()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Why iTXt and not tEXt
// ---------------------------------------------------------------------------

/// A prompt in the three scripts that break Latin-1: CJK, an emoji with a variation selector, and
/// combining accents.
const NON_ASCII_PROMPT: &str = "霧の中の灯台、シネマティック 🌊🗼✨ café naïve — ünicode";

#[test]
fn a_non_ascii_prompt_survives_the_round_trip() {
    let directory = tempfile::tempdir().expect("temp dir");
    let path = directory.path().join("non-ascii.png");
    let built = build_workflow_share(
        &asset_fixture(NON_ASCII_PROMPT),
        &payload_fixture(NON_ASCII_PROMPT, "文字、透かし 🚫"),
    );
    assert_eq!(
        built.prompt, NON_ASCII_PROMPT,
        "the builder kept the prompt"
    );

    write_workflow_chunk(&rgb_fixture(), &path, Some(&built)).expect("writes");
    let read = read_workflow_chunk_file(&path)
        .expect("reads back")
        .expect("carries a workflow");

    // Character for character, not just "roughly the same length".
    assert_eq!(read.prompt, NON_ASCII_PROMPT);
    assert_eq!(read.negative_prompt, "文字、透かし 🚫");
    assert_eq!(as_value(&read), as_value(&built));
}

#[test]
fn the_same_prompt_cannot_be_written_as_a_text_chunk_at_all() {
    // The concrete reason `iTXt` is not a preference. `tEXt` is Latin-1, and `png` refuses to
    // encode a keyword or text it cannot represent rather than transliterating — so a `tEXt`
    // workflow chunk would not mangle this prompt, it would fail to exist. Either way the format
    // is unusable for the one field that is guaranteed to be user prose.
    use png::text_metadata::{EncodableTextChunk, TEXtChunk};

    let mut encoded = Vec::new();
    let result = TEXtChunk::new(WORKFLOW_CHUNK_KEYWORD, NON_ASCII_PROMPT).encode(&mut encoded);
    assert!(
        result.is_err(),
        "a Latin-1 tEXt chunk must not silently accept {NON_ASCII_PROMPT:?}"
    );
}

// ---------------------------------------------------------------------------
// Per-image cost
// ---------------------------------------------------------------------------

#[test]
fn the_chunk_stays_a_text_chunk_and_not_a_payload() {
    // Recorded on sc-15947 so sc-15948 can sanity-check what it is about to add to every generated
    // image. Numbers as measured on the sc-15946 golden envelope (a full Krea edit: LoRAs, four
    // input shapes, an upscale pass, twelve allow-listed `advanced` keys):
    //
    //   uncompressed envelope JSON  960 bytes
    //   framed compressed iTXt      565 bytes  (length + type + keyword + separators + CRC)
    //
    // Both figures move by a few bytes between the two `serde_json` map backends, since key ORDER
    // changes what deflate finds to share — which is exactly why the assertions below are bounds.
    //
    // Against a 1024x1024 PNG of a few megabytes that is a rounding error, which is the point of
    // compressing it. The bounds below are deliberately loose — this test exists to catch the
    // chunk becoming a payload (a base64 image, an un-reduced `advanced` map), not to pin a size.
    let envelope = golden_envelope();
    let uncompressed = serde_json::to_string(&envelope).expect("serializes").len();
    let framed = workflow_chunk_size(&envelope).expect("the chunk encodes");

    println!("workflow iTXt chunk: {framed} bytes framed and compressed, {uncompressed} bytes of uncompressed JSON");

    assert!(
        framed < uncompressed,
        "compression must earn its keep: {framed} framed vs {uncompressed} raw"
    );
    assert!(
        framed < 8 * 1024,
        "the chunk grew to {framed} bytes — that is a payload, not metadata"
    );
    assert!(
        uncompressed < MAX_WORKFLOW_TEXT_BYTES / 8,
        "a representative envelope at {uncompressed} bytes is uncomfortably close to the \
         {MAX_WORKFLOW_TEXT_BYTES}-byte read cap"
    );
}

// ---------------------------------------------------------------------------
// One parse path
// ---------------------------------------------------------------------------

#[test]
fn the_reader_has_exactly_one_parse_path() {
    // The sanitizer is the reason this module exists as a thin layer over sc-15946: an envelope
    // that arrived in a stranger's PNG must come in through `parse_workflow_share_json`, which
    // runs the allow-list and the value-level guards. A local `serde_json::from_str::<
    // WorkflowShare>` would read the same JSON with none of them, and nothing about the resulting
    // struct would look wrong. Pinned as source structure because that is where it is visible.
    let path = repo_root()
        .join("crates")
        .join("sceneworks-core")
        .join("src")
        .join("workflow_png.rs");
    let source = fs::read_to_string(&path).expect("reads the module");
    // Only the shipped half, and only the code: the prose above and in the module deliberately
    // NAMES the calls it forbids, so a scan that read comments would fail on its own explanation.
    let shipped = source
        .split("#[cfg(test)]")
        .next()
        .expect("the module has a non-test half");
    let code: String = shipped
        .lines()
        .map(str::trim_start)
        .filter(|line| !line.starts_with("//"))
        .collect::<Vec<&str>>()
        .join("\n");

    assert!(
        code.contains("parse_workflow_share_json("),
        "the reader stopped parsing through sc-15946's entry point"
    );
    for forbidden in [
        "from_str::<WorkflowShare>",
        "from_value::<WorkflowShare>",
        "from_slice::<WorkflowShare>",
        "serde_json::from_str",
        "serde_json::from_value",
        "serde_json::from_slice",
    ] {
        assert!(
            !code.contains(forbidden),
            "workflow_png.rs deserializes an envelope with `{forbidden}`, bypassing the sc-15946 \
             sanitizer. Parse through `parse_workflow_share_json` instead."
        );
    }
}

// ---------------------------------------------------------------------------
// Ordinary third-party PNGs
// ---------------------------------------------------------------------------

#[test]
fn an_image_the_user_did_not_generate_here_is_a_clean_absence() {
    // The common case, and the one that must not be an error: sc-15952 will run this over whatever
    // the user drags in.
    let directory = tempfile::tempdir().expect("temp dir");
    let foreign = directory.path().join("foreign.png");
    rgb_fixture()
        .save_with_format(&foreign, ImageFormat::Png)
        .expect("writes a plain PNG");
    assert_eq!(read_workflow_chunk_file(&foreign), Ok(None));

    // Including one that is a PNG only after a re-encode, so the bytes are not ours at all.
    let jpeg = directory.path().join("foreign.jpg");
    rgb_fixture()
        .save_with_format(&jpeg, ImageFormat::Jpeg)
        .expect("writes a JPEG");
    assert!(
        read_workflow_chunk_file(&jpeg).is_err(),
        "a JPEG is not a PNG and must say so"
    );
}

#[test]
fn a_missing_file_is_an_io_error_not_a_panic() {
    let missing = Path::new("does-not-exist-sc-15947.png");
    assert!(read_workflow_chunk_file(missing).is_err());
}
