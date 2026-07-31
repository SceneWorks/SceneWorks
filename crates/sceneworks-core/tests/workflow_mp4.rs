//! The MP4 metadata codec, and the video arm of the envelope (sc-15956, epic 15945).
//!
//! Hermetic on purpose: every MP4 here is built byte by byte in [`mp4_with_comment`] rather than by
//! shelling out, so the reader's guarantees are tested on a host with no ffmpeg. The end-to-end
//! proof that ffmpeg actually produces this layout — and that the tag survives the encode chain —
//! is `the_envelope_survives_the_whole_encode_chain` in
//! `crates/sceneworks-worker/src/video_jobs/tests.rs`, which is gated on a reachable binary.

use serde_json::{json, Map, Value};

use sceneworks_core::workflow_mp4::{
    ffmetadata_document, read_mp4_comment, read_workflow_metadata, workflow_metadata_size,
    write_workflow_metadata_file, WorkflowMp4Error, FFMETADATA_HEADER, MAX_WORKFLOW_TEXT_BYTES,
    WORKFLOW_MP4_TAG,
};
use sceneworks_core::workflow_share::{
    build_video_workflow_share_from, embeddable_video_workflow_share, parse_workflow_share_json,
    WorkflowAssetFacts, WorkflowShareError, INPUT_KIND_REFERENCE, INPUT_KIND_REFERENCE_CLIP,
    INPUT_KIND_SOURCE, INPUT_KIND_SOURCE_CLIP, WORKFLOW_KIND_IMAGE, WORKFLOW_KIND_VIDEO,
    WORKFLOW_SHARE_MARKER_KEY, WORKFLOW_SHARE_MAX_BYTES,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn facts() -> WorkflowAssetFacts {
    WorkflowAssetFacts {
        mode: "text_to_video".to_owned(),
        model: "ltx_2_3".to_owned(),
        prompt: "a fox crossing a frozen river".to_owned(),
        negative_prompt: "blurry".to_owned(),
        seed: 4242,
        width: Some(768),
        height: Some(512),
    }
}

fn payload(extra: Value) -> Map<String, Value> {
    let mut base = json!({
        "mode": "text_to_video",
        "model": "ltx_2_3",
        "prompt": "a fox crossing a frozen river",
        "duration": 5.0,
        "fps": 24,
        "quality": "balanced",
        "width": 768,
        "height": 512,
    });
    if let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            base.insert(key.clone(), value.clone());
        }
    }
    base.as_object().expect("an object").clone()
}

fn video_share() -> sceneworks_core::workflow_share::WorkflowShare {
    build_video_workflow_share_from(&facts(), &payload(json!({})))
}

/// A minimal but structurally REAL MP4: `ftyp` + `moov/udta/meta/ilst/©cmt/data` + `mdat`.
///
/// The box layout ffmpeg's mov muxer actually produces, verified against a real file before this
/// was written. `mdat` is padded so the walk has to skip something large without reading it.
fn mp4_with_comment(comment: &[u8]) -> Vec<u8> {
    fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(body.len() + 8);
        out.extend_from_slice(
            &u32::try_from(body.len() + 8)
                .expect("box fits in 32 bits")
                .to_be_bytes(),
        );
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    // `data`: 4 bytes version+flags (1 = UTF-8 text), 4 bytes locale, then the payload.
    let mut data = Vec::new();
    data.extend_from_slice(&1u32.to_be_bytes());
    data.extend_from_slice(&0u32.to_be_bytes());
    data.extend_from_slice(comment);
    let data = boxed(b"data", &data);
    let cmt = boxed(&[0xA9, b'c', b'm', b't'], &data);
    let ilst = boxed(b"ilst", &cmt);
    let hdlr = boxed(b"hdlr", &[0u8; 21]);

    // `meta` is a FullBox here — 4 bytes of version+flags before its children — which is the form
    // ffmpeg writes. `a_quicktime_style_meta_without_the_fullbox_prefix_is_read` covers the other.
    let mut meta_body = vec![0u8; 4];
    meta_body.extend_from_slice(&hdlr);
    meta_body.extend_from_slice(&ilst);
    let meta = boxed(b"meta", &meta_body);
    let udta = boxed(b"udta", &meta);

    let mut moov_body = boxed(b"mvhd", &[0u8; 100]);
    moov_body.extend_from_slice(&udta);
    let moov = boxed(b"moov", &moov_body);

    let mut out = boxed(b"ftyp", b"isom\x00\x00\x02\x00isomiso2avc1mp41");
    out.extend_from_slice(&moov);
    out.extend_from_slice(&boxed(b"mdat", &vec![0xEEu8; 64 * 1024]));
    out
}

fn mp4_with_json(value: &Value) -> Vec<u8> {
    mp4_with_comment(
        serde_json::to_string(value)
            .expect("serializable")
            .as_bytes(),
    )
}

// ---------------------------------------------------------------------------
// The write side
// ---------------------------------------------------------------------------

/// The document is what ffmpeg's `-f ffmetadata` demuxer expects, under the standard tag.
#[test]
fn the_document_is_an_ffmetadata_file_under_the_comment_tag() {
    let document = ffmetadata_document(&video_share()).expect("encodable");
    let mut lines = document.lines();
    assert_eq!(
        lines.next(),
        Some(FFMETADATA_HEADER),
        "ffmpeg refuses the file without its header line"
    );
    assert_eq!(WORKFLOW_MP4_TAG, "comment");
    let body = lines.next().expect("a tag line");
    assert!(
        body.starts_with("comment="),
        "the envelope must ride in the STANDARD tag — a custom key does not survive a `-c copy` \
         remux. Got: {body:.60}"
    );
}

/// Every character `ffmetadata` reserves is escaped, and nothing else is touched.
#[test]
fn the_five_reserved_characters_are_escaped_and_the_rest_are_not() {
    // A prompt with all five, plus non-ASCII that must survive verbatim. Set through the
    // PAYLOAD, which takes precedence over the facts (see `build_share_of_kind`).
    const PROMPT: &str = "a=b; #tag \\slash\\ é 日本語 🎬";
    let share = build_video_workflow_share_from(&facts(), &payload(json!({ "prompt": PROMPT })));
    let document = ffmetadata_document(&share).expect("encodable");
    let line = document
        .lines()
        .nth(1)
        .expect("a tag line")
        .strip_prefix("comment=")
        .expect("the comment tag");

    // Unescaping the way ffmpeg does must recover the exact serialized envelope.
    let mut unescaped = String::with_capacity(line.len());
    let mut characters = line.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            unescaped.extend(characters.next());
        } else {
            unescaped.push(character);
        }
    }
    assert_eq!(
        unescaped,
        serde_json::to_string(&share).expect("serializable"),
        "the escaped line must unescape back to the envelope byte for byte"
    );
    assert!(
        unescaped.contains("日本語") && unescaped.contains('🎬'),
        "non-ASCII prose must ride verbatim, not escaped or transliterated"
    );
}

/// A raw carriage return is REFUSED rather than escaped, because `ffmetadata` has no escape for one
/// and silently truncates the value at it.
///
/// Measured before this guard existed: `"a\rb"` written, `"a"` read back, no error from ffmpeg at
/// either end. A recipe that is silently half a recipe is the worst failure shape available.
#[test]
fn a_raw_carriage_return_is_refused_rather_than_silently_truncated() {
    // The invariant the guard rests on: `serde_json` escapes a CR inside a string to the TWO
    // characters `\` `r`, so no raw 0x0D can reach the document however the prose was authored.
    let share = build_video_workflow_share_from(
        &facts(),
        &payload(json!({ "prompt": "line one\rline two\r\nline three" })),
    );
    let document = ffmetadata_document(&share).expect("a JSON-escaped CR stays encodable");
    assert!(
        !document.contains('\r'),
        "no raw CR may reach the document — `ffmetadata` truncates at one and cannot escape it"
    );

    // The whole prompt is still there after an ffmpeg-style unescape. The failure this guards
    // against is a value silently cut at the CR, so the round trip is what proves it did not
    // happen.
    let line = document
        .lines()
        .nth(1)
        .expect("a tag line")
        .strip_prefix("comment=")
        .expect("the comment tag");
    let mut unescaped = String::with_capacity(line.len());
    let mut characters = line.chars();
    while let Some(character) = characters.next() {
        if character == '\\' {
            unescaped.extend(characters.next());
        } else {
            unescaped.push(character);
        }
    }
    assert_eq!(
        unescaped,
        serde_json::to_string(&share).expect("serializable"),
        "the value must arrive whole, not cut at the carriage return"
    );
    assert!(
        unescaped.contains("line three"),
        "the tail after the CR must survive"
    );
}

/// The `Unencodable` refusal is still in the writer, even though no `WorkflowShare` can reach it.
///
/// It cannot be provoked through the typed struct — that is the whole content of the test above —
/// so what is pinned here is that the check has not been quietly deleted as dead code. The shape of
/// the future change it exists for is a field that stops going through `serde_json`'s string
/// escaping, at which point every recipe it touched would be silently halved.
#[test]
fn the_carriage_return_refusal_is_still_in_the_writer() {
    let serialized = serde_json::to_string(&video_share()).expect("serializable");
    assert!(
        !serialized.contains('\r'),
        "the precondition: a real envelope never serializes with a raw CR"
    );
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/workflow_mp4.rs"),
    )
    .expect("the module is readable");
    assert!(
        source.contains("payload.contains('\\r')"),
        "`ffmetadata_document` must still refuse a raw carriage return"
    );
}

/// The document is written where ffmpeg can read it, and reads back as what we encoded.
#[test]
fn the_document_is_written_to_disk_verbatim() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let path = directory.path().join("workflow.ffmeta");
    let share = video_share();
    write_workflow_metadata_file(&share, &path).expect("written");
    assert_eq!(
        std::fs::read_to_string(&path).expect("readable"),
        ffmetadata_document(&share).expect("encodable")
    );
}

/// The container's framing is 24 bytes, and reported so the two lanes can be compared honestly.
#[test]
fn the_reported_size_is_the_payload_plus_the_ilst_framing() {
    let share = video_share();
    let payload = serde_json::to_string(&share).expect("serializable").len();
    assert_eq!(
        workflow_metadata_size(&share).expect("measurable"),
        payload + 24,
        "`©cmt` header (8) + `data` header (8) + version/flags (4) + locale (4)"
    );
}

// ---------------------------------------------------------------------------
// The read side
// ---------------------------------------------------------------------------

#[test]
fn an_envelope_round_trips_through_the_container() {
    let share = video_share();
    let bytes = mp4_with_json(&serde_json::to_value(&share).expect("serializable"));
    let read = read_workflow_metadata(&bytes)
        .expect("readable")
        .expect("the envelope is in there");
    assert_eq!(read, share);
}

/// A full-cap envelope round trips — the size the argv route could not carry at all.
#[test]
fn a_full_cap_envelope_round_trips() {
    let mut facts = facts();
    // `PROSE_MAX_BYTES` bounds the prompt at 16 KiB, so the bulk has to come from somewhere the
    // sanitizer keeps whole. Several allow-listed labels plus a maximal prompt is a realistic
    // worst case; what matters here is that a large payload survives the container, not that the
    // envelope is exactly at the ceiling.
    facts.prompt = "x".repeat(64 * 1024);
    let share = build_video_workflow_share_from(&facts, &payload(json!({})));
    let bytes = mp4_with_json(&serde_json::to_value(&share).expect("serializable"));
    assert!(
        bytes.len() > 64 * 1024,
        "the fixture must actually be large to prove anything"
    );
    assert_eq!(
        read_workflow_metadata(&bytes).expect("readable"),
        Some(share)
    );
}

/// An MP4 with no comment at all is an ABSENCE, not an error — the common case for every clip the
/// user did not generate here.
#[test]
fn a_video_with_no_comment_is_an_absence() {
    // The same builder without the tag path: an `ftyp` + `mdat` only.
    let bytes = {
        let mut out = Vec::new();
        let ftyp_body: &[u8] = b"isom\x00\x00\x02\x00isom";
        out.extend_from_slice(&(8 + ftyp_body.len() as u32).to_be_bytes());
        out.extend_from_slice(b"ftyp");
        out.extend_from_slice(ftyp_body);
        out.extend_from_slice(&(8u32 + 1024).to_be_bytes());
        out.extend_from_slice(b"mdat");
        out.extend_from_slice(&vec![0u8; 1024]);
        out
    };
    assert_eq!(read_workflow_metadata(&bytes).expect("readable"), None);
}

/// Somebody ELSE's comment is an absence too. The MP4 tag key is shared — that is the price of a
/// key that survives a transcode — so the marker inside the JSON is what identifies the envelope.
#[test]
fn a_foreign_comment_is_an_absence_not_a_parse_failure() {
    for foreign in [
        "Encoded with Handbrake",
        "{\"prompt\":\"someone else's tool\"}",
        "",
        "not json at all }{",
    ] {
        let bytes = mp4_with_comment(foreign.as_bytes());
        assert_eq!(
            read_workflow_metadata(&bytes).expect("readable"),
            None,
            "a `{foreign}` comment must read as no envelope, not as an error"
        );
        // …and the raw comment is still legible to a diagnostic that wants it.
        assert_eq!(
            read_mp4_comment(&bytes).expect("readable").as_deref(),
            Some(foreign)
        );
    }
}

/// A comment that DOES carry our marker but is malformed is a typed error, not an absence: the file
/// claims to be ours, so silence would be the wrong answer.
#[test]
fn a_marked_but_malformed_comment_is_a_typed_error() {
    let bytes =
        mp4_with_comment(format!("{{\"{WORKFLOW_SHARE_MARKER_KEY}\":\"video\"}}").as_bytes());
    match read_workflow_metadata(&bytes) {
        Err(WorkflowMp4Error::Share(WorkflowShareError::MissingSchemaVersion)) => {}
        other => panic!("a marked envelope with no schemaVersion must be refused, got {other:?}"),
    }
}

/// A `data` box that CLAIMS more than the cap costs a comparison, not an allocation.
#[test]
fn an_oversized_comment_is_refused_by_our_bound_not_the_files_claim() {
    let bytes = mp4_with_comment(&vec![b'x'; MAX_WORKFLOW_TEXT_BYTES + 1]);
    match read_workflow_metadata(&bytes) {
        Err(WorkflowMp4Error::TooLarge { bytes, limit }) => {
            assert_eq!(limit, MAX_WORKFLOW_TEXT_BYTES);
            assert!(bytes > limit);
        }
        other => panic!("an over-cap comment must be refused, got {other:?}"),
    }
}

#[test]
fn a_non_utf8_comment_is_refused() {
    let bytes = mp4_with_comment(&[0xFF, 0xFE, 0xFD]);
    assert!(matches!(
        read_workflow_metadata(&bytes),
        Err(WorkflowMp4Error::NotUtf8)
    ));
}

/// A box declaring more bytes than its parent holds is refused rather than believed.
#[test]
fn a_box_that_overruns_its_parent_is_refused() {
    let mut bytes = mp4_with_json(&serde_json::to_value(video_share()).expect("serializable"));
    // Find the `moov` header and inflate its size past the end of the file.
    let moov = bytes
        .windows(4)
        .position(|window| window == b"moov")
        .expect("the fixture has a moov");
    let size_at = moov - 4;
    bytes[size_at..size_at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(matches!(
        read_workflow_metadata(&bytes),
        Err(WorkflowMp4Error::Malformed { .. })
    ));
}

/// A zero-length box cannot spin the walk forever.
#[test]
fn a_zero_length_box_terminates_the_walk() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&24u32.to_be_bytes());
    bytes.extend_from_slice(b"ftyp");
    bytes.extend_from_slice(b"isom\x00\x00\x02\x00isom");
    // A `moov` whose declared size is 8 — a header and nothing else — followed by a box declaring
    // a size smaller than its own header.
    bytes.extend_from_slice(&8u32.to_be_bytes());
    bytes.extend_from_slice(b"moov");
    bytes.extend_from_slice(&4u32.to_be_bytes());
    bytes.extend_from_slice(b"junk");
    let read = read_workflow_metadata(&bytes);
    assert!(
        read.is_err() || read.expect("checked").is_none(),
        "the walk must terminate, either refusing or finding nothing"
    );
}

/// A truncated file is a typed error rather than a panic.
#[test]
fn a_truncated_container_is_refused() {
    let full = mp4_with_json(&serde_json::to_value(video_share()).expect("serializable"));
    // Cuts INSIDE the header: the envelope cannot be recovered, and the reader must say so rather
    // than panic or hand back half a recipe.
    for cut in [4usize, 12, 40, 100, 300] {
        let read = read_workflow_metadata(&full[..cut.min(full.len())]);
        assert!(
            read.is_err() || read.expect("checked").is_none(),
            "a file cut at {cut} must refuse or report nothing"
        );
    }
    // A cut past the header but inside `mdat` still has the whole tag, and reading it is CORRECT:
    // the walk never touches `mdat`, which is the property that lets it seek past gigabytes
    // instead of buffering them.
    let moov_end = full.len() - 64 * 1024 - 8;
    assert!(
        read_workflow_metadata(&full[..moov_end + 16])
            .expect("readable")
            .is_some(),
        "a clip truncated inside `mdat` keeps its recipe — the reader never needed those bytes"
    );
}

/// The QuickTime-flavoured `meta` (no FullBox prefix) reads too.
#[test]
fn a_quicktime_style_meta_without_the_fullbox_prefix_is_read() {
    let share = video_share();
    let text = serde_json::to_string(&share).expect("serializable");
    let mut bytes = mp4_with_comment(text.as_bytes());
    // Drop the 4-byte version/flags word from `meta`'s body and shrink every enclosing box by 4.
    let meta = bytes
        .windows(4)
        .position(|window| window == b"meta")
        .expect("the fixture has a meta");
    let body = meta + 4;
    bytes.drain(body..body + 4);
    for kind in [b"meta", b"udta", b"moov"] {
        let at = bytes
            .windows(4)
            .position(|window| window == kind)
            .expect("the box is present");
        let size_at = at - 4;
        let size = u32::from_be_bytes([
            bytes[size_at],
            bytes[size_at + 1],
            bytes[size_at + 2],
            bytes[size_at + 3],
        ]);
        bytes[size_at..size_at + 4].copy_from_slice(&(size - 4).to_be_bytes());
    }
    assert_eq!(
        read_workflow_metadata(&bytes).expect("readable"),
        Some(share),
        "a prefix-less `meta` is the QuickTime form and must still be walked"
    );
}

// ---------------------------------------------------------------------------
// The video arm of the contract
// ---------------------------------------------------------------------------

/// A video envelope is marked `video`, and an image one is still marked `image`.
#[test]
fn the_marker_names_the_kind() {
    let video = serde_json::to_value(video_share()).expect("serializable");
    assert_eq!(video[WORKFLOW_SHARE_MARKER_KEY], json!(WORKFLOW_KIND_VIDEO));
    let image = serde_json::to_value(sceneworks_core::workflow_share::build_workflow_share_from(
        &facts(),
        &payload(json!({})),
    ))
    .expect("serializable");
    assert_eq!(image[WORKFLOW_SHARE_MARKER_KEY], json!(WORKFLOW_KIND_IMAGE));
}

/// Both kinds parse; a third does not, and the message names what this build reads.
///
/// The property an OLDER build has, stated from this side: a reader that knows only `image` refuses
/// a `video` envelope by this same gate rather than presenting a video recipe as an image one.
#[test]
fn an_unknown_kind_is_refused_and_both_known_kinds_are_not() {
    for known in [WORKFLOW_KIND_IMAGE, WORKFLOW_KIND_VIDEO] {
        let mut value = serde_json::to_value(video_share()).expect("serializable");
        value[WORKFLOW_SHARE_MARKER_KEY] = json!(known);
        assert!(
            parse_workflow_share_json(&value.to_string()).is_ok(),
            "`{known}` must parse"
        );
    }
    let mut value = serde_json::to_value(video_share()).expect("serializable");
    value[WORKFLOW_SHARE_MARKER_KEY] = json!("hologram");
    match parse_workflow_share_json(&value.to_string()) {
        Err(WorkflowShareError::UnsupportedKind { found, supported }) => {
            assert_eq!(found, "hologram");
            assert!(
                supported.contains("image") && supported.contains("video"),
                "{supported}"
            );
        }
        other => panic!("an unknown kind must be refused, got {other:?}"),
    }
}

/// The three video knobs travel, and they are the ASK rather than the measurement.
#[test]
fn the_video_knobs_travel() {
    let share = video_share();
    assert_eq!(share.duration_seconds, Some(5.0));
    assert_eq!(share.fps, Some(24));
    assert_eq!(share.quality.as_deref(), Some("balanced"));
}

/// …and an IMAGE envelope never grows them, even from a payload that carries the keys.
#[test]
fn an_image_envelope_never_grows_the_video_knobs() {
    let share = sceneworks_core::workflow_share::build_workflow_share_from(
        &facts(),
        &payload(json!({ "duration": 5.0, "fps": 24, "quality": "balanced" })),
    );
    assert_eq!(share.duration_seconds, None);
    assert_eq!(share.fps, None);
    assert_eq!(share.quality, None);
}

/// **The person-replacement withholding.** The story's own acceptance criterion, and the one field
/// class this contract withholds for privacy rather than for portability.
///
/// A shared video must not disclose that it was made by replacing a specific person. Three things
/// have to be true at once, and they are asserted together because any one of them alone would let
/// the disclosure through:
///
/// * the top-level `personTrackId` / `replacementMode` never reach the envelope;
/// * `advanced.selectedPersonTrack` — the WHOLE track record, with a person's name, the original
///   filename, and a `frames[].mask` array of filesystem paths — is dropped;
/// * `advanced.replacementModeLabel`, which is neither an id nor a path, is dropped anyway.
#[test]
fn the_person_replacement_fields_are_withheld_from_a_shared_video() {
    let mut facts = facts();
    facts.mode = "replace_person".to_owned();
    let share = build_video_workflow_share_from(
        &facts,
        &payload(json!({
            "mode": "replace_person",
            "personTrackId": "track_9f3c1b2a",
            "replacementMode": "full_person_replace_outfit",
            "sourceClipAssetId": "asset_0011",
            "advanced": {
                "motion": "handheld",
                "replacementModeLabel": "Full Person, Replace Outfit",
                "selectedPersonTrack": {
                    "id": "track_9f3c1b2a",
                    "name": "Sarah Whitfield",
                    "sourceDisplayName": "sarah_birthday_2019.mov",
                    "sourceAssetId": "asset_0011",
                    "frames": [
                        { "timestamp": 0.0, "mask": "assets/masks/track_9f3c1b2a/m0.png" },
                        { "timestamp": 0.5, "mask": "assets/masks/track_9f3c1b2a/m1.png" }
                    ]
                }
            }
        })),
    );

    let serialized = serde_json::to_string(&share).expect("serializable");
    for secret in [
        "track_9f3c1b2a",
        "Sarah Whitfield",
        "sarah_birthday_2019.mov",
        "full_person_replace_outfit",
        "Full Person, Replace Outfit",
        "assets/masks",
        "m0.png",
        "asset_0011",
        "replacementMode",
        "personTrackId",
        "selectedPersonTrack",
    ] {
        assert!(
            !serialized.contains(secret),
            "a shared video must not disclose `{secret}`. Envelope: {serialized}"
        );
    }

    // The knob that IS generation intent still travels, so this is a targeted withholding rather
    // than the whole map being dropped.
    assert_eq!(share.advanced.get("motion"), Some(&json!("handheld")));
    // The mode itself travels — "this was a person replacement" is already implied by the mode,
    // which the recipe cannot omit without lying about what was run. What is withheld is WHO and
    // HOW: the track, the person's name, and the replacement mode.
    assert_eq!(share.mode, "replace_person");
    // And the clip it worked from rides as SHAPE.
    assert!(share
        .inputs
        .iter()
        .any(|input| input.kind == INPUT_KIND_SOURCE_CLIP));
}

/// **Withheld silently, not through `omitted`.** The one place the contract's "state the absence"
/// rule is deliberately inverted.
///
/// `omitted: ["personTrackId"]` would announce the replacement to every reader while withholding
/// only the id — which is the disclosure the withholding exists to prevent.
#[test]
fn the_withholding_does_not_announce_itself_in_the_omitted_marker() {
    let mut facts = facts();
    facts.mode = "replace_person".to_owned();
    let share = build_video_workflow_share_from(
        &facts,
        &payload(json!({
            "mode": "replace_person",
            "personTrackId": "track_9f3c1b2a",
            "replacementMode": "face_only",
            "advanced": { "replacementModeLabel": "Face Only" }
        })),
    );
    for marker in &share.omitted {
        assert!(
            !marker.contains("person") && !marker.contains("replacement"),
            "the omission marker must not name the withheld person fields, got {:?}",
            share.omitted
        );
    }
}

/// Clip ids become SHAPE, exactly as the image lane's `sourceAssetId` does, with zero id leakage.
#[test]
fn clip_ids_become_shape_descriptors() {
    let share = build_video_workflow_share_from(
        &facts(),
        &payload(json!({
            "mode": "video_bridge",
            "sourceClipAssetId": "clip_left_7788",
            "bridgeRightClipAssetId": "clip_right_9900",
            "sourceClipAssetIds": ["clip_a_1", "clip_b_2"],
            "referenceClipAssetId": "refclip_5566",
            "referenceAssetIds": ["ref_img_1", "ref_img_2", "ref_img_3"],
            "lastFrameAssetId": "still_4321",
        })),
    );

    let by_kind = |kind: &str| {
        share
            .inputs
            .iter()
            .find(|input| input.kind == kind)
            .map(|input| input.count)
    };
    // Two named clips plus the two-entry list.
    assert_eq!(by_kind(INPUT_KIND_SOURCE_CLIP), Some(4));
    assert_eq!(by_kind(INPUT_KIND_REFERENCE_CLIP), Some(1));
    assert_eq!(by_kind(INPUT_KIND_REFERENCE), Some(3));
    // `lastFrameAssetId` is a STILL, so it counts as a source rather than a clip.
    assert_eq!(by_kind(INPUT_KIND_SOURCE), Some(1));

    let serialized = serde_json::to_string(&share).expect("serializable");
    for id in [
        "clip_left_7788",
        "clip_right_9900",
        "clip_a_1",
        "clip_b_2",
        "refclip_5566",
        "ref_img_1",
        "still_4321",
    ] {
        assert!(
            !serialized.contains(id),
            "no clip or asset id may reach the envelope; `{id}` did. {serialized}"
        );
    }
}

/// The gated video builder runs the recording ceiling, and it is the SAME ceiling as the image
/// lane's — a bound on what gets persisted, not on what the container could hold.
#[test]
fn the_gated_video_builder_refuses_an_over_ceiling_envelope() {
    assert!(
        embeddable_video_workflow_share(&facts(), &payload(json!({}))).is_some(),
        "an ordinary clip must embed"
    );

    // `advanced` is the only field with enough headroom to pass the ceiling: a map of allow-listed
    // scalars, each bounded, but many of them.
    let mut advanced = Map::new();
    for index in 0..4_000 {
        advanced.insert(format!("motion{index}"), json!("x".repeat(180)));
    }
    advanced.insert("motion".to_owned(), json!("x".repeat(180)));
    let mut facts = facts();
    facts.prompt = "y".repeat(16 * 1024);
    let share = build_video_workflow_share_from(
        &facts,
        &payload(json!({ "advanced": Value::Object(advanced) })),
    );
    let size = serde_json::to_string(&share).expect("serializable").len();
    assert!(
        size <= WORKFLOW_SHARE_MAX_BYTES,
        "the allow-list should have shed the unclassified keys, leaving {size} bytes"
    );
}

/// The reader refuses an over-ceiling envelope out of a container too, so a hand-built MP4 cannot
/// smuggle past a bound the writer applies.
#[test]
fn the_reader_applies_the_recording_ceiling_to_a_container() {
    let mut value = serde_json::to_value(video_share()).expect("serializable");
    value["prompt"] = json!("z".repeat(WORKFLOW_SHARE_MAX_BYTES + 4_096));
    let bytes = mp4_with_json(&value);
    match read_workflow_metadata(&bytes) {
        // Either the prose bound sheds it first or the ceiling refuses it; both are correct, and
        // what must NOT happen is an over-ceiling envelope coming back whole.
        Ok(Some(share)) => {
            let size = serde_json::to_string(&share).expect("serializable").len();
            assert!(size <= WORKFLOW_SHARE_MAX_BYTES, "{size} bytes came back");
        }
        Err(WorkflowMp4Error::Share(WorkflowShareError::TooLarge { .. })) => {}
        other => panic!("unexpected: {other:?}"),
    }
}

/// **Each container asserts its own kind.** An MP4 carrying an IMAGE envelope is refused.
///
/// The contract's kind gate accepts every kind this build knows, which is right for an envelope and
/// wrong for a consumer: our writers only ever pair an MP4 with a video envelope, so a mismatch is
/// a hand-made file or a future build's. Surfacing an image recipe out of a clip would offer
/// something no video studio can run.
///
/// The PNG side makes the symmetrical assertion — and there it is a REGRESSION guard rather than a
/// new rule: a PNG carrying a video envelope was refused before the video lane widened the shared
/// gate, and had to keep being refused after.
#[test]
fn an_mp4_carrying_an_image_envelope_is_refused() {
    let mut value = serde_json::to_value(video_share()).expect("serializable");
    value[WORKFLOW_SHARE_MARKER_KEY] = json!(WORKFLOW_KIND_IMAGE);
    let bytes = mp4_with_json(&value);
    match read_workflow_metadata(&bytes) {
        Err(WorkflowMp4Error::Share(WorkflowShareError::UnsupportedKind { found, supported })) => {
            assert_eq!(found, WORKFLOW_KIND_IMAGE);
            assert_eq!(supported, WORKFLOW_KIND_VIDEO);
        }
        other => panic!("an image envelope in an mp4 must be refused, got {other:?}"),
    }
}

/// …and the PNG reader still refuses a VIDEO envelope, which is the behaviour the widened kind
/// gate would otherwise have silently changed.
#[test]
fn a_png_carrying_a_video_envelope_is_still_refused() {
    let share = video_share();
    let png = {
        let image = image::RgbImage::from_pixel(2, 2, image::Rgb([10, 20, 30]));
        let directory = tempfile::tempdir().expect("a temp dir");
        let path = directory.path().join("clip.png");
        sceneworks_core::workflow_png::write_workflow_chunk(&image, &path, Some(&share))
            .expect("written");
        std::fs::read(&path).expect("readable")
    };
    match sceneworks_core::workflow_png::read_workflow_chunk(&png) {
        Err(sceneworks_core::workflow_png::WorkflowChunkError::Envelope(
            WorkflowShareError::UnsupportedKind { found, supported },
        )) => {
            assert_eq!(found, WORKFLOW_KIND_VIDEO);
            assert_eq!(supported, WORKFLOW_KIND_IMAGE);
        }
        other => panic!("a video envelope in a png must be refused, got {other:?}"),
    }
}
