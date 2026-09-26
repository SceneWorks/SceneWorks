//! Change contracts and the musical-invariant check between a source and an edited version.
//!
//! Every check here operates on the PARSED musical content of both scores — sounding notes
//! (onset, MIDI pitch, merged duration), per-voice bar grids (start, length, meter), chord onsets,
//! section boundaries and the quarter-note tempo — plus the generation-request fields. Two scores
//! whose text differs (a tie split, a different `L:` unit, whitespace) but whose events are equal
//! pass; two scores whose text differs by one note fail. Nothing compares strings or line counts.
//!
//! The upstream `compare` (abc_tools.py) fixes tempo, bar grids and sounding notes. SceneWorks adds
//! the section map, chord symbols (fixed unless harmony is declared), the request fields, and
//! declared-scope relaxations: a [`ChangeContract`] names exactly what an edit may change and every
//! aspect it does not name must be unchanged.

use serde::{Deserialize, Serialize};

use super::abc::{Frac, Note, Score, VoiceTrack, VOICE_NAMES};
use super::SongRequest;

/// What the invariant report establishes.
pub const INVARIANT_SCOPE: &str = "exact symbolic events (notes, durations, bars, sections, \
chords, tempo) and request fields; no claim about the generated audio";

/// A native voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoiceName {
    Vocal,
    Ins,
}

impl VoiceName {
    pub fn index(self) -> usize {
        match self {
            Self::Vocal => 0,
            Self::Ins => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        VOICE_NAMES[self.index()]
    }
}

/// A declared melody-change window: in these voices, notes whose onset falls in measures
/// `fromBar..=toBar` (1-based, source numbering) may change. Everything else stays fixed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MelodyScope {
    pub voices: Vec<VoiceName>,
    pub from_bar: usize,
    pub to_bar: usize,
}

/// A declared form change: the edited score's sections are, in order, the source sections at
/// these 0-based indices (reordered, repeated or omitted). Each mapped section's content is fixed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FormChange {
    pub section_order: Vec<usize>,
}

/// The aspects an edit declares it changes. Anything not declared must be unchanged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChangeContract {
    /// Chord symbols may change (and the request's `cot` with them).
    #[serde(default)]
    pub harmony: bool,
    /// The quarter-note tempo may change (beat-relative rhythm stays fixed).
    #[serde(default)]
    pub tempo: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub melody: Option<MelodyScope>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<FormChange>,
    #[serde(default)]
    pub lyrics: bool,
    #[serde(default)]
    pub style: bool,
}

impl ChangeContract {
    /// Validate the contract against the source score it is applied to.
    pub fn validate(&self, source: &Score) -> Result<(), String> {
        if let Some(melody) = &self.melody {
            if melody.voices.is_empty() {
                return Err("melody scope must name at least one voice".to_owned());
            }
            let bars = source.bar_count();
            if melody.from_bar < 1 || melody.from_bar > melody.to_bar || melody.to_bar > bars {
                return Err(format!(
                    "melody scope bars {}..={} must lie within 1..={bars}",
                    melody.from_bar, melody.to_bar
                ));
            }
        }
        if let Some(form) = &self.form {
            if self.harmony || self.tempo || self.melody.is_some() {
                return Err(
                    "a form change combines only with lyric and style changes; make harmony, \
                     tempo or melody changes as separate versions"
                        .to_owned(),
                );
            }
            if form.section_order.is_empty() {
                return Err("sectionOrder must list at least one section".to_owned());
            }
            if let Some(bad) = form
                .section_order
                .iter()
                .find(|index| **index >= source.sections.len())
            {
                return Err(format!(
                    "sectionOrder index {bad} is out of range; the source has {} sections",
                    source.sections.len()
                ));
            }
        }
        Ok(())
    }
}

/// Outcome of one invariant check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CheckStatus {
    /// Fixed by the contract and unchanged.
    Unchanged,
    /// Declared by the contract and changed.
    ChangedAsDeclared,
    /// Declared by the contract but left unchanged.
    DeclaredUnchanged,
    /// Changed although the contract fixes it.
    Violated,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvariantCheck {
    pub name: String,
    pub status: CheckStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// One chord-symbol difference, for the edit manifest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HarmonyChange {
    pub onset_quarters: Frac,
    /// 1-based measure of the source score (edited numbering for form changes is not reported).
    pub bar: Option<usize>,
    pub before: Option<String>,
    pub after: Option<String>,
}

/// The musical-invariant check result carried by every edited version.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InvariantReport {
    #[serde(rename = "match")]
    pub matches: bool,
    pub contract: ChangeContract,
    pub checks: Vec<InvariantCheck>,
    pub violations: Vec<String>,
    pub harmony_changes: Vec<HarmonyChange>,
    pub source_score_sha256: String,
    pub edited_score_sha256: String,
    pub scope: &'static str,
}

struct Checker {
    checks: Vec<InvariantCheck>,
}

impl Checker {
    fn push(&mut self, name: impl Into<String>, status: CheckStatus, detail: Option<String>) {
        self.checks.push(InvariantCheck {
            name: name.into(),
            status,
            detail,
        });
    }

    /// Record a fixed-or-declared aspect.
    fn aspect(&mut self, name: &str, declared: bool, changed: bool, detail: Option<String>) {
        let status = match (declared, changed) {
            (_, false) if declared => CheckStatus::DeclaredUnchanged,
            (_, false) => CheckStatus::Unchanged,
            (true, true) => CheckStatus::ChangedAsDeclared,
            (false, true) => CheckStatus::Violated,
        };
        self.push(name, status, if changed { detail } else { None });
    }
}

fn describe_note(note: Option<&Note>) -> String {
    match note {
        Some(note) => format!(
            "pitch {} at {} for {}",
            note.pitch, note.onset, note.duration
        ),
        None => "no note".to_owned(),
    }
}

/// Index of the first differing note, upstream-style (`None` = equal).
fn first_note_difference(before: &[Note], after: &[Note]) -> Option<usize> {
    if before == after {
        return None;
    }
    let common = before.len().min(after.len());
    Some(
        (0..common)
            .find(|index| before[*index] != after[*index])
            .unwrap_or(common),
    )
}

fn note_detail(score: &Score, before: &[Note], after: &[Note], index: usize) -> String {
    let at = before.get(index).or_else(|| after.get(index));
    let bar = at
        .and_then(|note| score.bar_number_at(note.onset))
        .map(|bar| format!(" (bar {bar})"))
        .unwrap_or_default();
    format!(
        "sounding notes differ starting at note {}{bar}: before {}, after {}",
        index + 1,
        describe_note(before.get(index)),
        describe_note(after.get(index)),
    )
}

fn harmony_changes(source: &Score, before: &VoiceTrack, after: &VoiceTrack) -> Vec<HarmonyChange> {
    let mut onsets: Vec<Frac> = before
        .chords
        .iter()
        .chain(after.chords.iter())
        .map(|chord| chord.onset)
        .collect();
    onsets.sort();
    onsets.dedup();
    let symbols_at = |track: &VoiceTrack, onset: Frac| -> Option<String> {
        let symbols: Vec<&str> = track
            .chords
            .iter()
            .filter(|chord| chord.onset == onset)
            .map(|chord| chord.chord.as_str())
            .collect();
        (!symbols.is_empty()).then(|| symbols.join(" "))
    };
    onsets
        .into_iter()
        .filter_map(|onset| {
            let before_symbol = symbols_at(before, onset);
            let after_symbol = symbols_at(after, onset);
            (before_symbol != after_symbol).then(|| HarmonyChange {
                onset_quarters: onset,
                bar: source.bar_number_at(onset),
                before: before_symbol,
                after: after_symbol,
            })
        })
        .collect()
}

fn in_window(onset: Frac, window: (Frac, Frac)) -> bool {
    window.0 <= onset && onset < window.1
}

/// Whether the whole sounding note (onset through its tied end) lies inside `window`.
fn lies_within(note: &Note, window: (Frac, Frac)) -> bool {
    window.0 <= note.onset && note.onset + note.duration <= window.1
}

/// Key events of `track` over `[start, start+length)`, relative to `start`: the key in effect at
/// `start`, then every change inside the span.
fn relative_keys(track: &VoiceTrack, start: Frac, length: Frac) -> Vec<(Frac, String)> {
    let mut keys: Vec<(Frac, String)> = track
        .keys
        .iter()
        .rfind(|event| event.onset <= start)
        .map(|event| (Frac::ZERO, event.key.clone()))
        .into_iter()
        .collect();
    keys.extend(
        track
            .keys
            .iter()
            .filter(|event| start < event.onset && event.onset < start + length)
            .map(|event| (event.onset - start, event.key.clone())),
    );
    keys
}

/// Events of `track` inside `[start, start+length)`, shifted to be relative to `start`.
fn relative_notes(track: &VoiceTrack, start: Frac, length: Frac) -> Vec<Note> {
    track
        .notes
        .iter()
        .filter(|note| in_window(note.onset, (start, start + length)))
        .map(|note| Note {
            onset: note.onset - start,
            ..*note
        })
        .collect()
}

fn relative_bars(
    track: &VoiceTrack,
    first_bar: usize,
    count: usize,
) -> Vec<(Frac, super::abc::Meter)> {
    track.bars[first_bar..first_bar + count]
        .iter()
        .map(|bar| (bar.length, bar.meter))
        .collect()
}

fn relative_chords(track: &VoiceTrack, start: Frac, length: Frac) -> Vec<(Frac, String)> {
    track
        .chords
        .iter()
        .filter(|chord| in_window(chord.onset, (start, start + length)))
        .map(|chord| (chord.onset - start, chord.chord.clone()))
        .collect()
}

fn check_requests(
    checker: &mut Checker,
    contract: &ChangeContract,
    before: &SongRequest,
    after: &SongRequest,
) {
    checker.aspect(
        "lyrics",
        contract.lyrics,
        before.lyrics != after.lyrics,
        Some("lyrics text differs".to_owned()),
    );
    checker.aspect(
        "style",
        contract.style,
        before.style != after.style,
        Some("style prompt differs".to_owned()),
    );
    checker.aspect(
        "cot",
        contract.harmony,
        before.cot != after.cot,
        Some(format!(
            "cot {} -> {}",
            before.cot.as_str(),
            after.cot.as_str()
        )),
    );
    checker.aspect(
        "seed",
        false,
        before.seed != after.seed,
        Some(format!("seed {} -> {}", before.seed, after.seed)),
    );
    checker.aspect(
        "cfgScale",
        false,
        before.cfg_scale != after.cfg_scale,
        Some(format!(
            "cfgScale {:?} -> {:?}",
            before.cfg_scale, after.cfg_scale
        )),
    );
}

fn check_score_in_place(
    checker: &mut Checker,
    contract: &ChangeContract,
    before: &Score,
    after: &Score,
) {
    checker.aspect(
        "tempo",
        contract.tempo,
        before.bpm != after.bpm,
        Some(format!("{} -> {} BPM", before.bpm, after.bpm)),
    );
    let section_map = |score: &Score| -> Vec<(Option<String>, usize, usize)> {
        score
            .sections
            .iter()
            .map(|section| (section.label.clone(), section.first_bar, section.bar_count))
            .collect()
    };
    let (before_sections, after_sections) = (section_map(before), section_map(after));
    checker.aspect(
        "sections",
        false,
        before_sections != after_sections,
        Some(format!(
            "section labels/bar ranges differ: {} -> {} sections",
            before_sections.len(),
            after_sections.len()
        )),
    );
    let window = contract.melody.as_ref().map(|scope| {
        let bars = &before.voices[0].bars;
        let start = bars[scope.from_bar - 1].start;
        let last = bars[scope.to_bar - 1];
        (start, last.start + last.length)
    });
    for (index, name) in VOICE_NAMES.iter().enumerate() {
        let (a, b) = (&before.voices[index], &after.voices[index]);
        checker.aspect(
            &format!("barGrid:{name}"),
            false,
            a.bars != b.bars,
            Some(format!(
                "{name}: bar meter/time grid differs ({} -> {} measures)",
                a.bars.len(),
                b.bars.len()
            )),
        );
        let scoped = contract
            .melody
            .as_ref()
            .is_some_and(|scope| scope.voices.iter().any(|voice| voice.index() == index));
        if scoped {
            let window = window.expect("melody scope has a window");
            // A note is free only when it lies ENTIRELY inside the window, on either side: a note
            // that starts inside but is tied past the window end (or starts before it) is fixed.
            let outside = |track: &VoiceTrack| -> Vec<Note> {
                track
                    .notes
                    .iter()
                    .filter(|note| !lies_within(note, window))
                    .copied()
                    .collect()
            };
            let (fixed_before, fixed_after) = (outside(a), outside(b));
            if let Some(first) = first_note_difference(&fixed_before, &fixed_after) {
                checker.push(
                    format!("notes:{name}"),
                    CheckStatus::Violated,
                    Some(format!(
                        "{name} outside the declared melody window: {}",
                        note_detail(before, &fixed_before, &fixed_after, first)
                    )),
                );
            } else {
                let inside_changed = a.notes != b.notes;
                checker.aspect(
                    &format!("notes:{name}"),
                    true,
                    inside_changed,
                    Some(format!(
                        "{name}: notes changed only inside bars {}..={}",
                        contract.melody.as_ref().expect("scope").from_bar,
                        contract.melody.as_ref().expect("scope").to_bar
                    )),
                );
            }
        } else {
            let first = first_note_difference(&a.notes, &b.notes);
            checker.aspect(
                &format!("notes:{name}"),
                false,
                first.is_some(),
                first.map(|first| {
                    format!("{name}: {}", note_detail(before, &a.notes, &b.notes, first))
                }),
            );
        }
    }
    // The key signature is part of the model's conditioning text, so it is fixed unless the
    // edit declares a harmony or melody change.
    checker.aspect(
        "keySignatures",
        contract.harmony || contract.melody.is_some(),
        before.voices[0].keys != after.voices[0].keys,
        Some("the key-signature timeline differs".to_owned()),
    );
    let chords_changed = before.voices[0].chords != after.voices[0].chords;
    checker.aspect(
        "harmony",
        contract.harmony,
        chords_changed,
        Some("chord symbols differ".to_owned()),
    );
}

fn check_form(checker: &mut Checker, form: &FormChange, before: &Score, after: &Score) {
    checker.aspect(
        "tempo",
        false,
        before.bpm != after.bpm,
        Some(format!("{} -> {} BPM", before.bpm, after.bpm)),
    );
    if after.sections.len() != form.section_order.len() {
        checker.push(
            "form",
            CheckStatus::Violated,
            Some(format!(
                "declared {} sections but the edited score has {}",
                form.section_order.len(),
                after.sections.len()
            )),
        );
        return;
    }
    let identity: Vec<usize> = (0..before.sections.len()).collect();
    for (position, (&source_index, edited)) in
        form.section_order.iter().zip(&after.sections).enumerate()
    {
        let source = &before.sections[source_index];
        let name = format!(
            "form:section {} <- source section {source_index}",
            position + 1
        );
        let mut problems = Vec::new();
        if source.label != edited.label {
            problems.push(format!("label {:?} != {:?}", source.label, edited.label));
        }
        if source.bar_count != edited.bar_count || source.length != edited.length {
            problems.push(format!(
                "{} bars ({} quarters) != {} bars ({} quarters)",
                source.bar_count, source.length, edited.bar_count, edited.length
            ));
        } else {
            for (index, voice) in VOICE_NAMES.iter().enumerate() {
                let (a, b) = (&before.voices[index], &after.voices[index]);
                if relative_bars(a, source.first_bar, source.bar_count)
                    != relative_bars(b, edited.first_bar, edited.bar_count)
                {
                    problems.push(format!("{voice} bar meters differ"));
                }
                let (notes_a, notes_b) = (
                    relative_notes(a, source.start, source.length),
                    relative_notes(b, edited.start, edited.length),
                );
                if let Some(first) = first_note_difference(&notes_a, &notes_b) {
                    problems.push(format!(
                        "{voice} sounding notes differ at section note {}: before {}, after {}",
                        first + 1,
                        describe_note(notes_a.get(first)),
                        describe_note(notes_b.get(first))
                    ));
                }
            }
            if relative_chords(&before.voices[0], source.start, source.length)
                != relative_chords(&after.voices[0], edited.start, edited.length)
            {
                problems.push("chord symbols differ".to_owned());
            }
            if relative_keys(&before.voices[0], source.start, source.length)
                != relative_keys(&after.voices[0], edited.start, edited.length)
            {
                problems.push("key signatures differ".to_owned());
            }
        }
        if problems.is_empty() {
            checker.push(name, CheckStatus::Unchanged, None);
        } else {
            checker.push(name, CheckStatus::Violated, Some(problems.join("; ")));
        }
    }
    checker.aspect(
        "form",
        true,
        form.section_order != identity,
        Some(format!("section order {:?}", form.section_order)),
    );
}

/// Check an edited (score, request) against its source under `contract`.
///
/// The contract must already be [`ChangeContract::validate`]d against `source_score`.
pub fn check_edit(
    contract: &ChangeContract,
    source_score: &Score,
    source_request: &SongRequest,
    edited_score: &Score,
    edited_request: &SongRequest,
) -> InvariantReport {
    let mut checker = Checker { checks: Vec::new() };
    match &contract.form {
        Some(form) => check_form(&mut checker, form, source_score, edited_score),
        None => check_score_in_place(&mut checker, contract, source_score, edited_score),
    }
    check_requests(&mut checker, contract, source_request, edited_request);
    let violations: Vec<String> = checker
        .checks
        .iter()
        .filter(|check| check.status == CheckStatus::Violated)
        .map(|check| match &check.detail {
            Some(detail) => format!("{}: {detail}", check.name),
            None => check.name.clone(),
        })
        .collect();
    let harmony = if contract.form.is_some() {
        Vec::new()
    } else {
        harmony_changes(
            source_score,
            &source_score.voices[0],
            &edited_score.voices[0],
        )
    };
    InvariantReport {
        matches: violations.is_empty(),
        contract: contract.clone(),
        checks: checker.checks,
        violations,
        harmony_changes: harmony,
        source_score_sha256: source_score.sha256(),
        edited_score_sha256: edited_score.sha256(),
        scope: INVARIANT_SCOPE,
    }
}

/// A descriptive, contract-free comparison of two arbitrary versions (for A/B listening records):
/// every aspect is checked as fixed, so each difference is listed. `matches == true` means the two
/// versions carry identical musical content and request fields.
pub fn describe_differences(
    a_score: &Score,
    a_request: &SongRequest,
    b_score: &Score,
    b_request: &SongRequest,
) -> InvariantReport {
    check_edit(
        &ChangeContract::default(),
        a_score,
        a_request,
        b_score,
        b_request,
    )
}
