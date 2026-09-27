use super::*;

use crate::yue2_score::store::{Actor, Channel, EditInput, SourceKind, SourceReference};
use crate::yue2_score::ScoreEditOperation;
use serde_json::json;
use sha2::Digest;

const SCORE: &str = include_str!("fixtures/score.abc");
const MELODY: &str = include_str!("fixtures/melody.abc");
const RUN: &str = "yue2run_abc123";
const RECORDING_SHA: &str = "1111111111111111111111111111111111111111111111111111111111111111";

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", sha2::Sha256::digest(bytes))
}

fn provenance() -> Provenance {
    Provenance {
        actor: Actor::User,
        agent_name: None,
        channel: Channel::Worker,
        source: Some(SourceReference {
            kind: SourceKind::Transcription,
            id: Some("yue2t_abc123".into()),
        }),
    }
}

fn policy() -> Value {
    json!({"schema": "sceneworks.usagePolicy.v1", "modelId": "yue2", "nonCommercial": true,
           "commercialUse": {"verdict": "refused", "modelId": "yue2", "reason": "CC BY-NC 4.0"}})
}

/// Write a review artifact shaped like `candle_audio_sheetsage2::review::Transcription::save`'s:
/// every file first, then the manifest naming each with its SHA-256. Returns the manifest digest.
fn write_artifact(project: &Path, melody_ready: bool, full_score: &str) -> String {
    let dir = project.join(crate::yue2_score::jobs::transcription_dir(RUN));
    fs::create_dir_all(&dir).unwrap();
    let mut files: Vec<(&str, Vec<u8>)> = vec![
        (FULL_SCORE, full_score.as_bytes().to_vec()),
        ("chord.lab", b"0.000\t2.000\tC:maj\n".to_vec()),
        ("melody.mid", vec![0x4d, 0x54, 0x68, 0x64]),
    ];
    if melody_ready {
        files.push((MELODY_SCORE, MELODY.as_bytes().to_vec()));
    }
    let mut artifacts = serde_json::Map::new();
    for (rel, bytes) in &files {
        fs::write(dir.join(rel), bytes).unwrap();
        artifacts.insert((*rel).to_owned(), json!(sha(bytes)));
    }
    let refused = json!({"ready": false, "reason": "the transcription has no melody notes"});
    let manifest = json!({
        "schema": ARTIFACT_SCHEMA,
        "source": {"sha256": "22", "samples": 24000, "sample_rate": 24000, "duration_seconds": 1.0,
                   "name": "take.wav", "original_sha256": RECORDING_SHA, "conversion": "none"},
        "settings": {"overlap_seconds": 200.0, "lookahead_seconds": 100.0, "max_seconds": null},
        "closure": {"device": "cpu", "ported_code_revision": "4f89269db831bdc1880124164a00d4f9385cd129"},
        "abc_error": {"full": null, "melody": null},
        "decode_warnings": [],
        "octave_evidence": {"notes_checked": 12, "fraction_f0_half_dominant": 0.75},
        "review": {
            "voices": {"vocal": {"notes": 30}, "instrumental": {"notes": 0}},
            "distinct_chords": ["C:maj", "G:maj"], "keys": ["C:major"], "sections": ["verse"],
            "bars": 8,
            "warnings": [{"code": "octave_f0_half_evidence", "message": "9 of 12 notes ..."}],
            "cover": {
                "melody": if melody_ready { json!({"ready": true}) } else { refused.clone() },
                "full": if melody_ready { json!({"ready": true}) } else { refused.clone() },
            }
        },
        "artifacts": artifacts,
    });
    let bytes = serde_json::to_vec_pretty(&manifest).unwrap();
    fs::write(dir.join(ARTIFACT_MANIFEST), &bytes).unwrap();
    sha(&bytes)
}

fn import(manifest_sha256: String) -> TranscriptionImport {
    TranscriptionImport {
        transcription_id: "yue2t_abc123".into(),
        job_id: "job-transcribe".into(),
        source_audio_asset_id: "asset_take".into(),
        recording_sha256: RECORDING_SHA.into(),
        artifact_dir: crate::yue2_score::jobs::transcription_dir(RUN),
        manifest_sha256,
        replay: json!({"artifactsMatched": 4, "windows": 1}),
        unload: json!({"released": true, "parameterBytes": 10, "liveModelsAfter": 0}),
        usage_policy: policy(),
        provenance: provenance(),
    }
}

/// AT2: both transcribed scores become versions linked to the transcription AND the recording,
/// carrying the job's usage policy; an edit keeps that lineage. Mutations that red this: dropping
/// the `VersionLineage` from `import_score` (no link), or not inheriting it in `apply_edit`.
#[test]
fn a_transcription_imports_both_scores_linked_to_the_recording() {
    let temp = tempfile::tempdir().unwrap();
    let store = Yue2ScoreStore::new(temp.path(), "project-1");
    let digest = write_artifact(temp.path(), true, SCORE);
    let record = store.import_transcription(import(digest.clone())).unwrap();
    assert_eq!(record.device, "cpu");
    assert_eq!(record.manifest_sha256, digest);
    assert_eq!(record.warnings[0]["code"], "octave_f0_half_evidence");
    assert_eq!(record.octave_evidence["notes_checked"], 12);
    assert!(
        record.version_errors.is_empty(),
        "{:?}",
        record.version_errors
    );
    let melody = store
        .get_version_record(record.versions.melody.as_deref().unwrap())
        .unwrap();
    let full = store
        .get_version_record(record.versions.full.as_deref().unwrap())
        .unwrap();
    for (version, cot, text) in [(&melody, Cot::Melody, MELODY), (&full, Cot::Full, SCORE)] {
        assert_eq!(version.origin, VersionOrigin::Transcription);
        assert_eq!(version.request.cot, cot);
        assert_eq!(version.score.abc, text);
        let link = version.transcription.as_ref().expect("linked");
        assert_eq!(link.source_audio_asset_id, "asset_take");
        assert_eq!(link.transcription_id, "yue2t_abc123");
        assert_eq!(link.manifest_sha256, digest);
        assert_eq!(link.transcribed_score_sha256, sha(text.as_bytes()));
        assert_eq!(version.usage_policy.as_ref(), Some(&policy()));
    }
    // Every artifact file is listed as an export with its kind.
    let kinds: Vec<(&str, &str)> = record
        .exports
        .iter()
        .map(|e| (e.path.as_str(), e.kind.as_str()))
        .collect();
    assert!(kinds.contains(&("melody.mid", "midi")), "{kinds:?}");
    assert!(kinds.contains(&("chord.lab", "lab")), "{kinds:?}");
    // The review edit keeps the lineage and the policy.
    let edited = store
        .apply_edit(
            &melody.id,
            EditInput {
                operation: serde_json::from_value::<ScoreEditOperation>(
                    json!({"op": "set_tempo", "bpm": 96}),
                )
                .unwrap(),
                brief: "slower".into(),
                provenance: provenance(),
            },
            false,
        )
        .unwrap();
    assert_eq!(edited.transcription, melody.transcription);
    assert_eq!(edited.usage_policy, melody.usage_policy);
    let summary = store
        .list_versions()
        .unwrap()
        .items
        .into_iter()
        .find(|v| v.id == edited.id)
        .unwrap();
    assert!(summary.non_commercial);
    assert_eq!(summary.transcription, melody.transcription);
    // Idempotent: a re-run returns the same record and creates no new versions.
    let again = store.import_transcription(import(digest)).unwrap();
    assert_eq!(again, record);
    assert_eq!(store.list_versions().unwrap().items.len(), 3);
    // Read back: both scores against their digests, and an export file.
    let detail = store.get_transcription_detail("yue2t_abc123").unwrap();
    assert_eq!(detail.scores["melody"].as_deref(), Some(MELODY));
    let (export, bytes) = store
        .transcription_file("yue2t_abc123", "melody.mid")
        .unwrap();
    assert_eq!((export.kind.as_str(), bytes.len()), ("midi", 4));
    assert!(matches!(
        store.transcription_file("yue2t_abc123", "../../secrets"),
        Err(Yue2ScoreError::NotFound(_))
    ));
}

/// An empty melody is a refusal the reviewer sees: no version is imported for a refused mode, and
/// the reason is on the record. Mutation that reds this: importing regardless of readiness.
#[test]
fn a_refused_mode_is_recorded_and_not_imported() {
    let temp = tempfile::tempdir().unwrap();
    let store = Yue2ScoreStore::new(temp.path(), "project-1");
    let digest = write_artifact(temp.path(), false, SCORE);
    let record = store.import_transcription(import(digest)).unwrap();
    assert_eq!(record.versions, ImportedVersions::default());
    assert!(!record.readiness.melody.ready);
    assert_eq!(
        record.readiness.melody.reason.as_deref(),
        Some("the transcription has no melody notes")
    );
    assert!(store.list_versions().unwrap().items.is_empty());
}

/// A cover-ready score SceneWorks' ABC dialect cannot hold is an import ERROR on the record, never
/// dropped; the other mode still imports. Mutation that reds this: `?` on the import (the whole
/// record is lost) or skipping the error.
#[test]
fn an_unimportable_score_is_a_visible_error() {
    let temp = tempfile::tempdir().unwrap();
    let store = Yue2ScoreStore::new(temp.path(), "project-1");
    let digest = write_artifact(temp.path(), true, "X:1\nK:C\nabc");
    let record = store.import_transcription(import(digest)).unwrap();
    assert!(record.versions.full.is_none());
    assert!(record.versions.melody.is_some());
    assert!(
        record.version_errors["full"].contains("Unsupported YuE2 ABC notation"),
        "{:?}",
        record.version_errors
    );
}

/// The import trusts nothing it cannot check: a manifest that is not the one the worker reported,
/// a transcription of a different recording, or a score file altered after the transcription.
/// Mutations that red this: dropping any of the three checks.
#[test]
fn a_foreign_or_altered_artifact_is_refused() {
    let temp = tempfile::tempdir().unwrap();
    let store = Yue2ScoreStore::new(temp.path(), "project-1");
    let digest = write_artifact(temp.path(), true, SCORE);
    let mut wrong_manifest = import(digest.clone());
    wrong_manifest.manifest_sha256 = "0".repeat(64);
    assert!(matches!(
        store.import_transcription(wrong_manifest),
        Err(Yue2ScoreError::Conflict(_))
    ));
    let mut other_recording = import(digest.clone());
    other_recording.recording_sha256 = "2".repeat(64);
    assert!(matches!(
        store.import_transcription(other_recording),
        Err(Yue2ScoreError::Conflict(_))
    ));
    fs::write(
        temp.path()
            .join(crate::yue2_score::jobs::transcription_dir(RUN))
            .join(MELODY_SCORE),
        "X:1\n",
    )
    .unwrap();
    let record = store.import_transcription(import(digest)).unwrap();
    assert!(record.versions.melody.is_none());
    assert!(
        record.version_errors["melody"].contains("altered"),
        "{:?}",
        record.version_errors
    );
}

/// The public version-create input can never forge a transcription link on a non-transcription
/// version. Mutation that reds this: dropping the origin check in `create_version_with_lineage`.
#[test]
fn only_a_transcription_version_links_a_transcription() {
    let temp = tempfile::tempdir().unwrap();
    let store = Yue2ScoreStore::new(temp.path(), "project-1");
    let link = TranscriptionLink {
        transcription_id: "yue2t_x".into(),
        job_id: "j".into(),
        source_audio_asset_id: "a".into(),
        manifest_sha256: "0".repeat(64),
        mode: Cot::Full,
        transcribed_score_sha256: "0".repeat(64),
    };
    let err = store
        .create_version_with_lineage(
            CreateVersionInput {
                abc: SCORE.into(),
                request: SongRequest {
                    style: String::new(),
                    lyrics: String::new(),
                    cot: Cot::Full,
                    seed: DEFAULT_SEED,
                    cfg_scale: None,
                },
                origin: VersionOrigin::Import,
                provenance: provenance(),
            },
            VersionLineage {
                transcription: Some(link),
                usage_policy: None,
            },
        )
        .unwrap_err();
    assert!(matches!(err, Yue2ScoreError::BadRequest(_)), "{err}");
}
