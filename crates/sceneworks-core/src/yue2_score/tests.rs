//! sc-22997: the released YuE2 harmony example plus negative invariant fixtures.

use serde_json::json;

use super::abc::{parse_score, Frac};
use super::compare::{check_edit, ChangeContract, CheckStatus, FormChange, MelodyScope, VoiceName};
use super::ops::{apply_operation, ChordChange, KeepVoice, ReplaceAllowance, ScoreEditOperation};
use super::store::{
    Actor, Channel, ComparisonInput, ComponentIdentity, CreateVersionInput, EditInput, Lineage,
    Provenance, RenderInput, RenderStatus, Truncation, VersionOrigin,
};
use super::{Cot, SongRequest, Yue2ScoreError, REGENERATION_NOTICE};
use crate::project_store::ProjectStore;

const SCORE: &str = include_str!("fixtures/score.abc");
const SCORE_JAZZ: &str = include_str!("fixtures/score-jazz.abc");
const MELODY: &str = include_str!("fixtures/melody.abc");

fn request() -> SongRequest {
    let song: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/song.json")).expect("song.json");
    SongRequest {
        style: song["style"].as_str().unwrap().to_owned(),
        lyrics: song["lyrics"].as_str().unwrap().to_owned(),
        cot: Cot::Full,
        seed: song["seed"].as_u64().unwrap(),
        cfg_scale: None,
    }
}

fn replace(abc: &str, allow: ReplaceAllowance) -> ScoreEditOperation {
    ScoreEditOperation::ReplaceScore {
        abc: abc.to_owned(),
        allow,
        lyrics: None,
        style: None,
        cot: None,
    }
}

fn harmony_only() -> ReplaceAllowance {
    ReplaceAllowance {
        harmony: true,
        ..ReplaceAllowance::default()
    }
}

fn violations(error: Yue2ScoreError) -> Vec<String> {
    match error {
        Yue2ScoreError::Invariant(report) => {
            assert!(!report.matches);
            report.violations.clone()
        }
        other => panic!("expected an invariant violation, got {other:?}"),
    }
}

fn apply(
    source: &str,
    operation: &ScoreEditOperation,
) -> Result<super::ops::EditOutcome, Yue2ScoreError> {
    apply_operation(&parse_score(source).unwrap(), &request(), operation)
}

// ---------------------------------------------------------------------------------------------
// Dialect port
// ---------------------------------------------------------------------------------------------

#[test]
fn released_examples_parse_to_exact_events() {
    let score = parse_score(SCORE).expect("released score parses");
    assert_eq!(score.bpm, 88);
    assert_eq!(score.unit, Frac::new(1, 16));
    assert_eq!(score.bar_count(), 8);
    assert_eq!(score.duration(), Frac::int(32));
    // Seven onsets per bar (examples/README.md), no instrumental notes.
    assert_eq!(score.voices[0].notes.len(), 56);
    assert!(score.voices[1].notes.is_empty());
    assert_eq!(score.voices[0].chords.len(), 8);
    let first = score.voices[0].notes[0];
    assert_eq!(
        (first.onset, first.pitch, first.duration),
        (Frac::ZERO, 64, Frac::new(1, 2))
    );
    let labels: Vec<_> = score.sections.iter().map(|s| s.label.clone()).collect();
    assert_eq!(
        labels,
        [Some("verse".to_owned()), Some("chorus".to_owned())]
    );
    assert_eq!(
        (score.sections[1].first_bar, score.sections[1].bar_count),
        (4, 4)
    );

    let inspection = serde_json::to_value(super::inspect(&score)).unwrap();
    assert_eq!(inspection["voices"]["Vocal"]["soundingNotes"], 56);
    assert_eq!(
        inspection["voices"]["Vocal"]["notes"][0]["durationQuarters"],
        "1/2"
    );
    assert_eq!(
        inspection["nominalDurationSeconds"].as_f64().unwrap(),
        32.0 * 60.0 / 88.0
    );
    parse_score(SCORE_JAZZ).expect("released jazz score parses");
    parse_score(MELODY).expect("released melody parses");
}

#[test]
fn accidentals_propagate_by_letter_and_ties_keep_pitch_across_the_barline() {
    // abc-editing.md: in C major `^F32-|F8F24|` is a five-quarter F# then a three-quarter F.
    let abc = native("4/4", "32", "C", &[("% verse", "^F32-|F8F24|", "Z2|")]);
    let score = parse_score(&abc).unwrap();
    let notes = &score.voices[0].notes;
    assert_eq!(notes.len(), 2);
    assert_eq!((notes[0].pitch, notes[0].duration), (66, Frac::int(5)));
    assert_eq!((notes[1].pitch, notes[1].duration), (65, Frac::int(3)));
    // Across octaves: after ^F, a lower-case f in the same bar is F#5.
    let score = parse_score(&native("4/4", "32", "C", &[("% verse", "^F16f16|", "Z|")])).unwrap();
    assert_eq!(score.voices[0].notes[1].pitch, 78);
}

/// Build a native two-voice score. Each group is `(comment lines, Vocal line, Ins line)`.
fn native(meter: &str, unit: &str, key: &str, groups: &[(&str, &str, &str)]) -> String {
    let mut lines = vec![
        "X:1".to_owned(),
        "T:".to_owned(),
        format!("M:{meter}"),
        format!("L:1/{unit}"),
        "Q:1/4=88".to_owned(),
        r#"V: Vocal clef=treble name="Vocal Melody" snm="Vocal""#.to_owned(),
        r#"V: Ins clef=treble name="Ins Melody" snm="Inst.""#.to_owned(),
        format!("K:{key}"),
    ];
    for (comments, vocal, ins) in groups {
        if !comments.is_empty() {
            lines.extend(comments.lines().map(str::to_owned));
        }
        lines.push("V: Vocal".to_owned());
        lines.extend(vocal.lines().map(str::to_owned));
        lines.push("V: Ins".to_owned());
        lines.extend(ins.lines().map(str::to_owned));
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

#[test]
fn unsupported_notation_is_reported_as_outside_the_dialect() {
    let cases = [
        ("(3E2G2A2", "unsupported token"),
        ("{g}E2G2A2G2E2D2C4F4", "unsupported token"),
        ("[CEG]8z8z16", "unsupported token"),
        ("\"Cmaj9\"E2G2A2G2E2D2C4z16", "unsupported chord \"Cmaj9\""),
        ("E5z27", "unsupported duration 5"),
        ("E2G2A2G2", "duration 2 quarter notes != meter duration 4"),
        ("E8-z8", "tie enters a rest"),
        ("E8-^F8", "tie changes pitch from 64 to 66"),
    ];
    for (bar, expected) in cases {
        let abc = native("4/4", "16", "C", &[("% verse", &format!("{bar}|"), "Z|")]);
        let error = parse_score(&abc).expect_err(bar);
        assert!(
            error.message.contains(expected),
            "{bar}: {} does not mention {expected}",
            error.message
        );
    }
    let repeat = SCORE.replacen("C4|\"G\"", "C4|:\"G\"", 1);
    assert!(parse_score(&repeat)
        .unwrap_err()
        .message
        .contains("unsupported token"));
    let modal = SCORE.replacen("K:C", "K:Dmix", 1);
    assert!(parse_score(&modal)
        .unwrap_err()
        .message
        .contains("Unsupported key \"Dmix\""));
    let lyrics_field = SCORE.replacen("V: Ins\nZ4|", "V: Ins\nZ4|\nw: la la", 1);
    assert!(parse_score(&lyrics_field).is_err());

    // Through the edit surface the same rejection is a typed Notation error, not a violation.
    let error = apply(SCORE, &replace(&modal, harmony_only())).unwrap_err();
    assert!(matches!(error, Yue2ScoreError::Notation(_)), "{error:?}");
    assert!(error
        .to_string()
        .contains("outside the supported native dialect"));
}

// ---------------------------------------------------------------------------------------------
// The released harmony example and negative invariant fixtures
// ---------------------------------------------------------------------------------------------

#[test]
fn released_harmony_edit_passes_every_fixed_invariant() {
    let outcome = apply(SCORE, &replace(SCORE_JAZZ, harmony_only())).expect("released edit passes");
    let report = &outcome.report;
    assert!(report.matches, "{:?}", report.violations);
    let status = |name: &str| {
        report
            .checks
            .iter()
            .find(|check| check.name == name)
            .unwrap_or_else(|| panic!("missing check {name}"))
            .status
    };
    for fixed in [
        "tempo",
        "sections",
        "barGrid:Vocal",
        "barGrid:Ins",
        "notes:Vocal",
        "notes:Ins",
        "lyrics",
        "style",
        "seed",
        "cfgScale",
    ] {
        assert_eq!(status(fixed), CheckStatus::Unchanged, "{fixed}");
    }
    // cot may change together with harmony (strip_chords -> melody); here it did not.
    assert_eq!(status("cot"), CheckStatus::DeclaredUnchanged);
    assert_eq!(status("harmony"), CheckStatus::ChangedAsDeclared);
    assert_eq!(report.harmony_changes.len(), 8);
    assert_eq!(report.harmony_changes[0].before.as_deref(), Some("C"));
    assert_eq!(report.harmony_changes[0].after.as_deref(), Some("Cmaj7"));
    assert_eq!(report.harmony_changes[0].bar, Some(1));
}

#[test]
fn harmony_only_edit_rejects_a_changed_melody_note() {
    // Mutation: the jazz edit also moves the first melody note E -> F.
    let edited = SCORE_JAZZ.replacen("\"Cmaj7\"E2G2", "\"Cmaj7\"F2G2", 1);
    let found = violations(apply(SCORE, &replace(&edited, harmony_only())).unwrap_err());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("notes:Vocal"), "{found:?}");
    assert!(found[0].contains("note 1 (bar 1)"), "{found:?}");
}

#[test]
fn harmony_only_edit_rejects_a_changed_duration() {
    // Same pitches, same bar total, different rhythm: E2G2 -> E3G1.
    let edited = SCORE_JAZZ.replacen("\"Cmaj7\"E2G2", "\"Cmaj7\"E3G", 1);
    let found = violations(apply(SCORE, &replace(&edited, harmony_only())).unwrap_err());
    assert!(
        found.iter().any(|v| v.starts_with("notes:Vocal")),
        "{found:?}"
    );
}

#[test]
fn harmony_only_edit_rejects_a_rearticulated_tie() {
    // C4 -> C2C2: one sounding note becomes two attacks with the same pitches.
    let edited = SCORE_JAZZ.replacen("D2C4|\"G7\"", "D2C2C2|\"G7\"", 1);
    let found = violations(apply(SCORE, &replace(&edited, harmony_only())).unwrap_err());
    assert!(
        found.iter().any(|v| v.starts_with("notes:Vocal")),
        "{found:?}"
    );
    // …while the tied spelling of the same sounding note is NOT a change.
    let tied = SCORE_JAZZ.replacen("D2C4|\"G7\"", "D2C2-C2|\"G7\"", 1);
    assert!(apply(SCORE, &replace(&tied, harmony_only())).is_ok());
}

#[test]
fn harmony_only_edit_rejects_bar_section_and_tempo_changes() {
    // Bar grid: same bar lengths, different meter (4/4 -> 2/2).
    let bars = SCORE_JAZZ.replacen("M:4/4", "M:2/2", 1);
    let found = violations(apply(SCORE, &replace(&bars, harmony_only())).unwrap_err());
    assert!(
        found.iter().any(|v| v.starts_with("barGrid:Vocal")),
        "{found:?}"
    );

    // Section: relabel the chorus.
    let section = SCORE_JAZZ.replacen("% chorus", "% bridge", 1);
    let found = violations(apply(SCORE, &replace(&section, harmony_only())).unwrap_err());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("sections"), "{found:?}");

    // Tempo.
    let tempo = SCORE_JAZZ.replacen("Q:1/4=88", "Q:1/4=96", 1);
    let found = violations(apply(SCORE, &replace(&tempo, harmony_only())).unwrap_err());
    assert_eq!(found, ["tempo: 88 -> 96 BPM"]);
}

#[test]
fn undeclared_harmony_change_is_a_violation() {
    let found =
        violations(apply(SCORE, &replace(SCORE_JAZZ, ReplaceAllowance::default())).unwrap_err());
    assert_eq!(found, ["harmony: chord symbols differ"]);
}

#[test]
fn check_operates_on_events_not_text() {
    // Re-express the whole score at L:1/32: every duration doubles (the fixture's music lines
    // use only 2 and 4, and its chord names no digit but 7). Text differs on every music line;
    // the music is identical, so a pure unit change is not a violation.
    let doubled: Vec<String> = SCORE_JAZZ
        .lines()
        .map(|line| match line {
            "L:1/16" => "L:1/32".to_owned(),
            music if music.starts_with('"') => music.replace('4', "8").replace('2', "4"),
            other => other.to_owned(),
        })
        .collect();
    let doubled = doubled.join("\n") + "\n";
    assert!(doubled.contains("\"Cmaj7\"E4G4A4G4E4D4C8|"), "{doubled}");
    let source = parse_score(SCORE_JAZZ).unwrap();
    let edited = parse_score(&doubled).unwrap();
    let report = check_edit(
        &ChangeContract::default(),
        &source,
        &request(),
        &edited,
        &request(),
    );
    assert!(report.matches, "{:?}", report.violations);
    assert_ne!(report.source_score_sha256, report.edited_score_sha256);
}

// ---------------------------------------------------------------------------------------------
// Structured operations
// ---------------------------------------------------------------------------------------------

#[test]
fn reharmonize_splits_held_notes_with_ties_and_preserves_every_note() {
    let operation = ScoreEditOperation::Reharmonize {
        changes: vec![
            // On a token boundary (A2 starts at one quarter).
            ChordChange {
                bar: 1,
                onset_quarters: "1".into(),
                chord: Some("Am7".into()),
            },
            // Inside the held C4 of bar 1 (C4 spans 3..4 quarters).
            ChordChange {
                bar: 1,
                onset_quarters: "7/2".into(),
                chord: Some("G7/B".into()),
            },
            // Replace an existing chord.
            ChordChange {
                bar: 2,
                onset_quarters: "0".into(),
                chord: Some("G7".into()),
            },
            // Remove one.
            ChordChange {
                bar: 3,
                onset_quarters: "0".into(),
                chord: None,
            },
        ],
    };
    let outcome = apply(SCORE, &operation).expect("reharmonization preserves the melody");
    assert!(outcome.report.matches);
    assert!(
        outcome
            .score
            .text
            .contains("\"C\"E2G2\"Am7\"A2G2E2D2C2-\"G7/B\"C2|\"G7\"D2"),
        "{}",
        outcome.score.text
    );
    assert!(outcome.score.text.contains("|E2G2A2c2B2A2G4|"));
    let chords: Vec<_> = outcome.score.voices[0]
        .chords
        .iter()
        .map(|c| (c.onset.to_string(), c.chord.clone()))
        .collect();
    assert_eq!(
        &chords[..4],
        [
            ("0".to_owned(), "C".to_owned()),
            ("1".to_owned(), "Am7".to_owned()),
            ("7/2".to_owned(), "G7/B".to_owned()),
            ("4".to_owned(), "G7".to_owned()),
        ]
    );
    assert_eq!(
        outcome.contract,
        ChangeContract {
            harmony: true,
            ..ChangeContract::default()
        }
    );
}

#[test]
fn reharmonize_keeps_accidentals_and_cross_bar_tie_pitches() {
    // Chord inside a held ^F, and inside the tied continuation of ^F32 across the barline.
    let abc = native(
        "4/4",
        "32",
        "C",
        &[("% verse", "^F8z24|^F32-|F8F24|", "Z3|")],
    );
    let operation = ScoreEditOperation::Reharmonize {
        changes: vec![
            ChordChange {
                bar: 1,
                onset_quarters: "1/2".into(),
                chord: Some("D7".into()),
            },
            ChordChange {
                bar: 3,
                onset_quarters: "1/8".into(),
                chord: Some("Bm7b5".into()),
            },
        ],
    };
    let outcome = apply(&abc, &operation).expect("pitches preserved");
    assert!(
        outcome.score.text.contains("^F4-\"D7\"^F4z24|"),
        "{}",
        outcome.score.text
    );
    assert!(
        outcome.score.text.contains("|F-\"Bm7b5\"F6-FF24|"),
        "{}",
        outcome.score.text
    );
    let pitches: Vec<i32> = outcome.score.voices[0]
        .notes
        .iter()
        .map(|n| n.pitch)
        .collect();
    assert_eq!(pitches, [66, 66, 65]);
}

#[test]
fn reharmonize_places_a_chord_in_a_compressed_full_measure_rest() {
    let abc = native("4/4", "16", "C", &[("% verse", "Z2|", "C16|D16|")]);
    let operation = ScoreEditOperation::Reharmonize {
        changes: vec![ChordChange {
            bar: 2,
            onset_quarters: "2".into(),
            chord: Some("F".into()),
        }],
    };
    let outcome = apply(&abc, &operation).unwrap();
    assert!(
        outcome.score.text.contains("V: Vocal\nZ|z8\"F\"z8|\n"),
        "{}",
        outcome.score.text
    );
}

#[test]
fn reharmonize_rejects_off_grid_onsets_unknown_chords_and_missing_removals() {
    let cases = [
        (
            ChordChange {
                bar: 1,
                onset_quarters: "1/8".into(),
                chord: Some("C".into()),
            },
            "rhythmic grid",
        ),
        (
            ChordChange {
                bar: 1,
                onset_quarters: "8".into(),
                chord: Some("C".into()),
            },
            "before the measure end",
        ),
        (
            ChordChange {
                bar: 9,
                onset_quarters: "0".into(),
                chord: Some("C".into()),
            },
            "within 1..=8",
        ),
        (
            ChordChange {
                bar: 1,
                onset_quarters: "0".into(),
                chord: Some("C13".into()),
            },
            "unsupported chord",
        ),
        (
            ChordChange {
                bar: 1,
                onset_quarters: "1".into(),
                chord: None,
            },
            "no chord starts here",
        ),
    ];
    for (change, expected) in cases {
        let error = apply(
            SCORE,
            &ScoreEditOperation::Reharmonize {
                changes: vec![change],
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
fn strip_chords_reproduces_the_released_melody_fixture() {
    let outcome = apply(
        SCORE,
        &ScoreEditOperation::StripChords {
            keep_voice: KeepVoice::Both,
        },
    )
    .unwrap();
    assert_eq!(outcome.score.text, MELODY);
    assert_eq!(outcome.request.cot, Cot::Melody);

    // Keeping only the instrumental voice silences the vocal melody on the same grid.
    let abc = native("4/4", "16", "C", &[("% verse", "\"C\"E8G8|", "c16|")]);
    let outcome = apply(
        &abc,
        &ScoreEditOperation::StripChords {
            keep_voice: KeepVoice::Ins,
        },
    )
    .unwrap();
    assert!(outcome.score.voices[0].notes.is_empty());
    assert_eq!(outcome.score.voices[1].notes.len(), 1);
    assert!(
        outcome.score.text.contains("V: Vocal\nz8z8|"),
        "{}",
        outcome.score.text
    );
}

#[test]
fn melody_cot_requires_a_chord_free_score() {
    let operation = ScoreEditOperation::ReplaceScore {
        abc: SCORE_JAZZ.to_owned(),
        allow: harmony_only(),
        lyrics: None,
        style: None,
        cot: Some(Cot::Melody),
    };
    let error = apply(SCORE, &operation).unwrap_err();
    assert!(error.to_string().contains("strip_chords"), "{error}");
}

#[test]
fn tempo_edit_changes_only_the_declared_tempo() {
    let outcome = apply(
        SCORE,
        &ScoreEditOperation::SetTempo {
            bpm: 100,
            style: None,
        },
    )
    .unwrap();
    assert_eq!(outcome.score.bpm, 100);
    let changed: Vec<_> = outcome
        .report
        .checks
        .iter()
        .filter(|check| check.status != CheckStatus::Unchanged)
        .map(|check| (check.name.as_str(), check.status))
        .collect();
    assert_eq!(changed, [("tempo", CheckStatus::ChangedAsDeclared)]);
    // Beat-relative rhythm is fixed: a tempo edit that also shifts a note fails.
    let shifted = SCORE
        .replacen("Q:1/4=88", "Q:1/4=100", 1)
        .replacen("\"C\"E2G2", "\"C\"E2A2", 1);
    let allow = ReplaceAllowance {
        tempo: true,
        ..ReplaceAllowance::default()
    };
    let found = violations(apply(SCORE, &replace(&shifted, allow)).unwrap_err());
    assert!(
        found.iter().any(|v| v.starts_with("notes:Vocal")),
        "{found:?}"
    );
}

#[test]
fn melody_edit_is_confined_to_its_declared_window() {
    let edited = SCORE.replacen("\"Am\"E2G2A2c2", "\"Am\"E2G2A2d2", 1); // bar 3
    let scope = |from, to| ReplaceAllowance {
        melody: Some(MelodyScope {
            voices: vec![VoiceName::Vocal],
            from_bar: from,
            to_bar: to,
        }),
        ..ReplaceAllowance::default()
    };
    apply(SCORE, &replace(&edited, scope(3, 3))).expect("inside the window");
    let found = violations(apply(SCORE, &replace(&edited, scope(4, 8))).unwrap_err());
    assert!(
        found[0].contains("outside the declared melody window"),
        "{found:?}"
    );
}

#[test]
fn melody_window_fixes_notes_tied_across_its_edges() {
    let ins_bars = |from, to| ReplaceAllowance {
        melody: Some(MelodyScope {
            voices: vec![VoiceName::Ins],
            from_bar: from,
            to_bar: to,
        }),
        ..ReplaceAllowance::default()
    };
    // Review repro: a note that STARTS in bar 1 but is tied through bar 4.
    let tied_out = SCORE.replacen("V: Ins\nZ4|", "V: Ins\nc16-|c16-|c16-|c16|", 1);
    let found = violations(apply(SCORE, &replace(&tied_out, ins_bars(1, 1))).unwrap_err());
    assert!(
        found.iter().any(|v| v.starts_with("notes:Ins")),
        "{found:?}"
    );
    // The same note kept inside bar 1 is a legitimate melody edit.
    let inside = SCORE.replacen("V: Ins\nZ4|", "V: Ins\nc16|Z3|", 1);
    apply(SCORE, &replace(&inside, ins_bars(1, 1))).expect("confined to the window");
    // A SOURCE note tied out of the window cannot be re-pitched from inside it.
    let source = native("4/4", "16", "C", &[("% verse", "Z2|", "c16-|c16|")]);
    let repitched = source.replacen("c16-|c16|", "d16-|d16|", 1);
    let found = violations(apply(&source, &replace(&repitched, ins_bars(1, 1))).unwrap_err());
    assert!(
        found.iter().any(|v| v.starts_with("notes:Ins")),
        "{found:?}"
    );
}

#[test]
fn key_signature_is_fixed_unless_harmony_or_melody_is_declared() {
    // K:C -> K:Am keeps every sounding pitch but changes the conditioning text.
    let lyrics_only = ScoreEditOperation::ReplaceScore {
        abc: SCORE.replacen("K:C", "K:Am", 1),
        allow: ReplaceAllowance::default(),
        lyrics: Some("[Verse]\nnew words".into()),
        style: None,
        cot: None,
    };
    let found = violations(apply(SCORE, &lyrics_only).unwrap_err());
    assert_eq!(found, ["keySignatures: the key-signature timeline differs"]);
    let outcome = apply(
        SCORE,
        &replace(&SCORE_JAZZ.replacen("K:C", "K:Am", 1), harmony_only()),
    )
    .expect("a declared harmony edit may respell the key");
    let key_check = outcome
        .report
        .checks
        .iter()
        .find(|check| check.name == "keySignatures")
        .unwrap();
    assert_eq!(key_check.status, CheckStatus::ChangedAsDeclared);
}

#[test]
fn every_operation_output_is_size_bounded() {
    let bars = "\"C\"E2G2A2G2E2D2C4|".repeat(4);
    let groups: Vec<(&str, &str, &str)> = (0..30)
        .map(|index| {
            (
                if index == 0 { "% verse" } else { "" },
                bars.as_str(),
                "Z4|",
            )
        })
        .collect();
    let abc = native("4/4", "16", "C", &groups);
    assert!(abc.len() * 128 > super::MAX_ABC_BYTES && abc.len() < super::MAX_ABC_BYTES);
    let operation = ScoreEditOperation::ArrangeSections {
        section_order: vec![0; 128],
        lyrics: "[Verse]\nx".into(),
        style: None,
    };
    let error = apply(&abc, &operation).unwrap_err();
    assert!(
        matches!(&error, Yue2ScoreError::BadRequest(detail) if detail.contains("the limit is")),
        "{error:?}"
    );
}

#[test]
fn oversized_measures_are_refused_before_any_expansion() {
    // Review repro: an unbounded meter let one measure hold millions of units.
    let huge = native("20000/4", "1024", "C", &[("% verse", "Z|", "Z|")]);
    let error = parse_score(&huge).unwrap_err();
    assert!(error.message.contains("meter numerator 20000"), "{error}");
    // Within the meter bound, a chord in a very long full-measure rest is refused rather than
    // expanded (64/4 at L:1/1024 is 16384 units).
    let long = native("64/4", "1024", "C", &[("% verse", "Z|", "Z|")]);
    let operation = ScoreEditOperation::Reharmonize {
        changes: vec![ChordChange {
            bar: 1,
            onset_quarters: "0".into(),
            chord: Some("C".into()),
        }],
    };
    let error = apply(&long, &operation).unwrap_err();
    assert!(error.to_string().contains("at most 4096 units"), "{error}");
}

#[test]
fn form_edit_moves_whole_sections_and_nothing_else() {
    let operation = ScoreEditOperation::ArrangeSections {
        section_order: vec![1, 0, 1],
        lyrics: "[Chorus]\na\n\n[Verse]\nb\n\n[Chorus]\na".into(),
        style: None,
    };
    let outcome = apply(SCORE, &operation).expect("reordering preserves section content");
    let labels: Vec<_> = outcome
        .score
        .sections
        .iter()
        .map(|s| s.label.clone().unwrap())
        .collect();
    assert_eq!(labels, ["chorus", "verse", "chorus"]);
    assert_eq!(outcome.score.voices[0].notes.len(), 84);
    assert_eq!(outcome.score.bpm, 88);

    // Mutation: the same arrangement with one note changed in the repeated chorus fails the
    // form contract even though the section order matches.
    let tampered =
        outcome
            .score
            .text
            .replacen("\"C\"E2G2A2G2E2D2C4|\n", "\"C\"E2G2A2G2E2D2D4|\n", 1);
    let contract = ChangeContract {
        form: Some(FormChange {
            section_order: vec![1, 0, 1],
        }),
        lyrics: true,
        ..ChangeContract::default()
    };
    let report = check_edit(
        &contract,
        &parse_score(SCORE).unwrap(),
        &request(),
        &parse_score(&tampered).unwrap(),
        &outcome.request,
    );
    assert!(!report.matches);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.contains("section 1 <- source section 1")),
        "{:?}",
        report.violations
    );
}

#[test]
fn form_edit_restates_the_key_a_moved_section_inherited() {
    let abc = native(
        "4/4",
        "16",
        "C",
        &[
            ("% verse", "\"C\"E16|", "Z|"),
            ("% chorus", "K:G\n\"G\"F16|", "K:G\nZ|"),
        ],
    );
    let source = parse_score(&abc).unwrap();
    assert_eq!(source.voices[0].notes[1].pitch, 66, "F is sharp in G major");
    let operation = ScoreEditOperation::ArrangeSections {
        section_order: vec![1, 0],
        lyrics: "[Chorus]\nb\n[Verse]\na".into(),
        style: None,
    };
    let outcome = apply(&abc, &operation).expect("keys restated");
    assert!(
        outcome.score.text.contains("% verse\nV: Vocal\nK:C\n"),
        "{}",
        outcome.score.text
    );
    let pitches: Vec<i32> = outcome.score.voices[0]
        .notes
        .iter()
        .map(|n| n.pitch)
        .collect();
    assert_eq!(pitches, [66, 64]);

    // Respelling the moved verse's restated key (C -> Am keeps E natural) is a form violation.
    let respelled = outcome
        .score
        .text
        .replace("V: Vocal\nK:C\n", "V: Vocal\nK:Am\n")
        .replace("V: Ins\nK:C\n", "V: Ins\nK:Am\n");
    let contract = ChangeContract {
        form: Some(FormChange {
            section_order: vec![1, 0],
        }),
        lyrics: true,
        ..ChangeContract::default()
    };
    let edited = parse_score(&respelled).unwrap();
    assert_eq!(edited.voices[0].notes, outcome.score.voices[0].notes);
    let report = check_edit(&contract, &source, &request(), &edited, &outcome.request);
    assert!(
        report
            .violations
            .iter()
            .any(|v| v.contains("key signatures differ")),
        "{:?}",
        report.violations
    );
}

#[test]
fn form_edit_refuses_a_tie_across_the_section_boundary() {
    let abc = native(
        "4/4",
        "16",
        "C",
        &[("% verse", "E16-|", "Z|"), ("% chorus", "E16|", "Z|")],
    );
    let operation = ScoreEditOperation::ArrangeSections {
        section_order: vec![1, 0],
        lyrics: "x".into(),
        style: None,
    };
    assert!(apply(&abc, &operation)
        .unwrap_err()
        .to_string()
        .contains("tie"));
}

#[test]
fn lyric_and_style_edits_change_only_their_field_and_no_op_edits_are_refused() {
    let outcome = apply(
        SCORE,
        &ScoreEditOperation::SetLyrics {
            lyrics: "[Verse]\nnew".into(),
        },
    )
    .unwrap();
    assert_eq!(outcome.score.text, SCORE);
    assert_eq!(outcome.report.violations, Vec::<String>::new());
    let style = request().style;
    let error = apply(SCORE, &ScoreEditOperation::SetStyle { style }).unwrap_err();
    assert!(error.to_string().contains("changes nothing"), "{error}");
    // An undeclared lyric change through replace_score's request is impossible: lyrics are only
    // changed when given, and that declares them. A replaced score cannot smuggle a seed change.
    let operation = ScoreEditOperation::ReplaceScore {
        abc: SCORE.to_owned(),
        allow: ReplaceAllowance::default(),
        lyrics: Some("changed".into()),
        style: None,
        cot: None,
    };
    let outcome = apply(SCORE, &operation).unwrap();
    assert!(outcome.contract.lyrics);
}

#[test]
fn edit_operations_are_a_closed_bounded_set() {
    let unknown =
        serde_json::from_value::<ScoreEditOperation>(json!({"op": "free_text", "abc": "x"}));
    assert!(unknown.is_err());
    let extra = serde_json::from_value::<ScoreEditOperation>(
        json!({"op": "set_tempo", "bpm": 90, "skipChecks": true}),
    );
    assert!(extra.is_err(), "unknown fields must not be ignored");
    let parsed: ScoreEditOperation =
        serde_json::from_value(json!({"op": "strip_chords", "keepVoice": "Vocal"})).unwrap();
    assert_eq!(
        parsed,
        ScoreEditOperation::StripChords {
            keep_voice: KeepVoice::Vocal
        }
    );
    let too_many = ScoreEditOperation::Reharmonize {
        changes: vec![
            ChordChange {
                bar: 1,
                onset_quarters: "0".into(),
                chord: Some("C".into())
            };
            1025
        ],
    };
    assert!(apply(SCORE, &too_many)
        .unwrap_err()
        .to_string()
        .contains("1..=1024"));
    let huge = replace(&"x".repeat(super::MAX_ABC_BYTES + 1), harmony_only());
    assert!(apply(SCORE, &huge)
        .unwrap_err()
        .to_string()
        .contains("limit"));
}

// ---------------------------------------------------------------------------------------------
// Versions, renders and comparisons
// ---------------------------------------------------------------------------------------------

fn agent() -> Provenance {
    Provenance {
        actor: Actor::Agent,
        agent_name: Some("claude".into()),
        channel: Channel::Mcp,
        source: None,
    }
}

fn audio_asset(store: &ProjectStore, project_id: &str, asset_id: &str) {
    let project_path = std::path::PathBuf::from(store.get_project(project_id).unwrap().path);
    let relative = format!("assets/audios/set_{asset_id}/{asset_id}.wav");
    let media = project_path.join(&relative);
    std::fs::create_dir_all(media.parent().unwrap()).unwrap();
    std::fs::write(&media, b"RIFF").unwrap();
    store
        .persist_generated_asset(
            project_id,
            "job",
            &format!("set_{asset_id}"),
            &json!({
                "type": "audio",
                "assetId": asset_id,
                "mediaPath": relative,
                "mimeType": "audio/wav",
                "displayName": asset_id,
                "createdAt": "2026-09-26T00:00:00Z",
                "mode": "text_to_music",
                "model": "yue2",
                "adapter": "yue2",
                "prompt": "x",
            }),
        )
        .unwrap();
}

fn render(sha: &str, request_sha: &str, audio: &str, truncated: Truncation) -> RenderInput {
    RenderInput {
        status: RenderStatus::Completed,
        score_sha256: sha.to_owned(),
        request_sha256: request_sha.to_owned(),
        truncated,
        model: ComponentIdentity {
            id: "m-a-p/YuE2-3B".into(),
            revision: Some("1a96eca".into()),
        },
        decoder: Some(ComponentIdentity {
            id: "m-a-p/YuE2-Vae".into(),
            revision: None,
        }),
        job_id: Some("job_1".into()),
        audio_asset_id: Some(audio.to_owned()),
        effective_settings: Some(json!({"seed": 831001})),
        error: None,
        provenance: Provenance {
            actor: Actor::User,
            agent_name: None,
            channel: Channel::Worker,
            source: None,
        },
    }
}

#[test]
fn edits_create_linked_versions_and_never_touch_the_source() {
    let temp = tempfile::tempdir().unwrap();
    let projects = ProjectStore::new(temp.path().join("data"), "test");
    let project = projects.create_project("YuE2").unwrap();
    let store = projects.yue2_score_store(&project.id).unwrap();
    let root = store
        .create_version(CreateVersionInput {
            abc: SCORE.to_owned(),
            request: request(),
            origin: VersionOrigin::Plan,
            provenance: agent(),
        })
        .unwrap();
    assert_eq!(root.root_version_id, root.id);
    assert_eq!(root.render_notice, REGENERATION_NOTICE);
    let root_file = store
        .project_path()
        .join(format!("yue2/versions/{}.json", root.id));
    let root_bytes = std::fs::read(&root_file).unwrap();

    let edit = |dry_run| {
        store.apply_edit(
            &root.id,
            EditInput {
                operation: replace(SCORE_JAZZ, harmony_only()),
                brief: "Reharmonize with seventh chords; keep every note, bar, section and tempo."
                    .into(),
                provenance: agent(),
            },
            dry_run,
        )
    };
    let preview = edit(true).unwrap();
    assert_eq!(
        store.list_versions().unwrap().items.len(),
        1,
        "dry run stores nothing"
    );
    let child = edit(false).unwrap();
    assert_ne!(child.id, preview.id);
    assert_eq!(child.parent_version_id.as_deref(), Some(root.id.as_str()));
    assert_eq!(child.root_version_id, root.id);
    let recorded = child.edit.as_ref().unwrap();
    assert_eq!(recorded.source_score_sha256, root.score.sha256);
    assert_eq!(recorded.invariants["match"], true);
    assert!(recorded.operation.get("abc").is_none());
    assert_eq!(recorded.operation["op"], "replace_score");
    assert_eq!(
        std::fs::read(&root_file).unwrap(),
        root_bytes,
        "source untouched"
    );

    // A refused edit stores nothing.
    let bad = SCORE_JAZZ.replacen("\"Cmaj7\"E2", "\"Cmaj7\"F2", 1);
    let error = store
        .apply_edit(
            &root.id,
            EditInput {
                operation: replace(&bad, harmony_only()),
                brief: "x".into(),
                provenance: agent(),
            },
            false,
        )
        .unwrap_err();
    assert!(matches!(error, Yue2ScoreError::Invariant(_)));
    let listing = store.list_versions().unwrap();
    assert_eq!(listing.items.len(), 2);
    // Parent before child even when both were created in the same second.
    assert_eq!(
        (listing.items[0].id.as_str(), listing.items[1].id.as_str()),
        (root.id.as_str(), child.id.as_str())
    );
    assert!(listing.unreadable.is_empty());

    // A blank brief is refused: every edit states what it changes.
    let error = store
        .apply_edit(
            &root.id,
            EditInput {
                operation: replace(SCORE_JAZZ, harmony_only()),
                brief: "  ".into(),
                provenance: agent(),
            },
            false,
        )
        .unwrap_err();
    assert!(error.to_string().contains("brief"));

    // A tampered record is reported, not trusted.
    let mut tampered: serde_json::Value = serde_json::from_slice(&root_bytes).unwrap();
    tampered["score"]["abc"] = json!(SCORE_JAZZ);
    std::fs::write(&root_file, serde_json::to_vec(&tampered).unwrap()).unwrap();
    assert!(store
        .get_version(&root.id)
        .unwrap_err()
        .to_string()
        .contains("corrupt"));
    assert_eq!(store.list_versions().unwrap().unreadable.len(), 1);
}

#[test]
fn renders_and_listening_comparisons_retain_request_score_brief_and_truncation() {
    let temp = tempfile::tempdir().unwrap();
    let projects = ProjectStore::new(temp.path().join("data"), "test");
    let project = projects.create_project("YuE2").unwrap();
    audio_asset(&projects, &project.id, "audio_before");
    audio_asset(&projects, &project.id, "audio_after");
    let store = projects.yue2_score_store(&project.id).unwrap();
    let root = store
        .create_version(CreateVersionInput {
            abc: SCORE.to_owned(),
            request: request(),
            origin: VersionOrigin::Import,
            provenance: agent(),
        })
        .unwrap();
    let brief = "Seventh-chord reharmonization; melody, bars, sections and tempo fixed.";
    let child = store
        .apply_edit(
            &root.id,
            EditInput {
                operation: replace(SCORE_JAZZ, harmony_only()),
                brief: brief.into(),
                provenance: agent(),
            },
            false,
        )
        .unwrap();

    // The render must come from the version's exact score.
    let mismatch = projects
        .record_yue2_render(
            &project.id,
            &child.id,
            render(
                &root.score.sha256,
                &root.request_sha256,
                "audio_after",
                Truncation {
                    abc: false,
                    semantic: false,
                },
            ),
        )
        .unwrap_err();
    assert!(
        matches!(mismatch, Yue2ScoreError::Conflict(_)),
        "{mismatch:?}"
    );
    // …and name a real audio asset.
    let missing = projects
        .record_yue2_render(
            &project.id,
            &child.id,
            render(
                &child.score.sha256,
                &child.request_sha256,
                "nope",
                Truncation {
                    abc: false,
                    semantic: false,
                },
            ),
        )
        .unwrap_err();
    assert!(missing.to_string().contains("does not exist"), "{missing}");
    // Truncation has no default.
    let mut no_truncation = serde_json::to_value(render(
        &child.score.sha256,
        &child.request_sha256,
        "audio_after",
        Truncation {
            abc: false,
            semantic: false,
        },
    ))
    .unwrap();
    no_truncation.as_object_mut().unwrap().remove("truncated");
    assert!(serde_json::from_value::<RenderInput>(no_truncation).is_err());

    let before = projects
        .record_yue2_render(
            &project.id,
            &root.id,
            render(
                &root.score.sha256,
                &root.request_sha256,
                "audio_before",
                Truncation {
                    abc: false,
                    semantic: false,
                },
            ),
        )
        .unwrap();
    let after = projects
        .record_yue2_render(
            &project.id,
            &child.id,
            render(
                &child.score.sha256,
                &child.request_sha256,
                "audio_after",
                Truncation {
                    abc: false,
                    semantic: true,
                },
            ),
        )
        .unwrap();
    assert_eq!(after.request, child.request);
    assert_eq!(after.score_sha256, child.score.sha256);
    assert_eq!(after.score_abc, SCORE_JAZZ);
    assert_eq!(after.edit_brief.as_deref(), Some(brief));
    assert_eq!(after.parent_version_id.as_deref(), Some(root.id.as_str()));
    assert!(after.whole_recording_regenerated);
    assert!(after.truncated.semantic);
    assert_eq!(
        store.get_version(&child.id).unwrap().renders,
        std::slice::from_ref(&after)
    );

    let comparison = store
        .create_comparison(ComparisonInput {
            version_a: root.id.clone(),
            version_b: child.id.clone(),
            render_a: Some(before.id.clone()),
            render_b: Some(after.id.clone()),
            notes: Some("B's sevenths land; verse entry smoother.".into()),
            provenance: agent(),
        })
        .unwrap();
    assert_eq!(comparison.lineage, Lineage::BDerivesFromA);
    assert!(!comparison.identical_music_and_request);
    assert_eq!(
        comparison.symbolic_differences,
        ["harmony: chord symbols differ"]
    );
    assert_eq!(comparison.edit_invariants.as_ref().unwrap()["match"], true);
    assert_eq!(comparison.b.edit_brief.as_deref(), Some(brief));
    assert_eq!(
        (
            comparison.a.score_abc.as_str(),
            comparison.b.score_abc.as_str()
        ),
        (SCORE, SCORE_JAZZ)
    );
    assert_eq!(
        comparison.b.render.as_ref().unwrap().audio_asset_id,
        "audio_after"
    );
    assert!(
        comparison.warnings.iter().any(|w| w.contains("TRUNCATED")),
        "{:?}",
        comparison.warnings
    );
    assert_eq!(comparison.render_notice, REGENERATION_NOTICE);
    assert_eq!(
        store.get_comparison(&comparison.id).unwrap(),
        comparison,
        "persisted"
    );

    // A render of another version cannot stand in for this side.
    let error = store
        .create_comparison(ComparisonInput {
            version_a: root.id.clone(),
            version_b: child.id.clone(),
            render_a: Some(after.id.clone()),
            render_b: None,
            notes: None,
            provenance: agent(),
        })
        .unwrap_err();
    assert!(error.to_string().contains("belongs to version"), "{error}");

    // The render must also come from the version's exact request (409 like the score hash).
    let other_request = SongRequest {
        seed: 7,
        ..child.request.clone()
    };
    let error = projects
        .record_yue2_render(
            &project.id,
            &child.id,
            render(
                &child.score.sha256,
                &super::request_sha256(&other_request),
                "audio_after",
                Truncation {
                    abc: false,
                    semantic: false,
                },
            ),
        )
        .unwrap_err();
    assert!(
        matches!(&error, Yue2ScoreError::Conflict(detail) if detail.contains("request")),
        "{error:?}"
    );

    // A failed render has no recording, so it cannot name one.
    let mut failed = render(
        &child.score.sha256,
        &child.request_sha256,
        "audio_after",
        Truncation {
            abc: false,
            semantic: false,
        },
    );
    failed.status = RenderStatus::Failed;
    failed.error = Some("decoder ran out of memory".into());
    let error = projects
        .record_yue2_render(&project.id, &child.id, failed)
        .unwrap_err();
    assert!(
        error.to_string().contains("cannot name an audio asset"),
        "{error}"
    );

    // Stored render and comparison snapshots are integrity-checked on read.
    let render_file = store
        .project_path()
        .join(format!("yue2/renders/{}.json", after.id));
    let original = std::fs::read(&render_file).unwrap();
    let mut tampered: serde_json::Value = serde_json::from_slice(&original).unwrap();
    tampered["scoreAbc"] = json!(SCORE);
    std::fs::write(&render_file, serde_json::to_vec(&tampered).unwrap()).unwrap();
    assert!(store
        .get_render(&after.id)
        .unwrap_err()
        .to_string()
        .contains("corrupt"));
    let mut tampered: serde_json::Value = serde_json::from_slice(&original).unwrap();
    tampered["request"]["seed"] = json!(7);
    std::fs::write(&render_file, serde_json::to_vec(&tampered).unwrap()).unwrap();
    assert!(store
        .get_render(&after.id)
        .unwrap_err()
        .to_string()
        .contains("corrupt"));
    std::fs::write(&render_file, &original).unwrap();

    let comparison_file = store
        .project_path()
        .join(format!("yue2/comparisons/{}.json", comparison.id));
    let original = std::fs::read(&comparison_file).unwrap();
    let mut tampered: serde_json::Value = serde_json::from_slice(&original).unwrap();
    tampered["b"]["scoreAbc"] = json!(SCORE);
    std::fs::write(&comparison_file, serde_json::to_vec(&tampered).unwrap()).unwrap();
    let error = store
        .get_comparison(&comparison.id)
        .unwrap_err()
        .to_string();
    assert!(error.contains("side B is corrupt"), "{error}");
    let mut tampered: serde_json::Value = serde_json::from_slice(&original).unwrap();
    tampered["a"]["request"]["lyrics"] = json!("rewritten");
    std::fs::write(&comparison_file, serde_json::to_vec(&tampered).unwrap()).unwrap();
    let error = store
        .get_comparison(&comparison.id)
        .unwrap_err()
        .to_string();
    assert!(error.contains("side A is corrupt"), "{error}");
}
