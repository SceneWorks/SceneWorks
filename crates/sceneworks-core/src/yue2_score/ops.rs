//! The bounded set of YuE2 score edit operations a user or agent may apply.
//!
//! Every operation produces a complete candidate (ABC text + generation request) and a declared
//! [`ChangeContract`]; [`apply_operation`] then re-parses the candidate in the native dialect and
//! runs the invariant check against the source. A candidate that is outside the dialect or that
//! changes anything its contract does not declare is refused — there is no path that stores an
//! edited score without both checks, including `replace_score`, which is the only operation that
//! takes free ABC text (bounded by [`super::MAX_ABC_BYTES`]).

use serde::{Deserialize, Serialize};

use super::abc::{
    is_native_chord, lex_token, parse_score, split_music_line, Frac, Score, Token, CHORD_QUALITIES,
    DURATIONS,
};
use super::compare::{
    check_edit, describe_differences, ChangeContract, FormChange, InvariantReport, MelodyScope,
    VoiceName,
};
use super::{Cot, SongRequest, Yue2ScoreError};

/// Upper bound on chord changes in one `reharmonize` operation.
pub const MAX_CHORD_CHANGES: usize = 1024;
/// Upper bound on entries in an `arrange_sections` order.
pub const MAX_SECTION_ORDER: usize = 128;
/// Largest full-measure rest (in `L:` units) `reharmonize` will expand into rest tokens to place
/// a chord in it — 4096 units is a 4/4 bar at `L:1/1024`, the finest supported grid.
pub const MAX_REST_EXPANSION_UNITS: i128 = 4096;

/// One chord-symbol change: set (or, with `chord: null`, remove) the chord starting at
/// `onsetQuarters` (exact quarter-note offset inside the 1-based measure `bar`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ChordChange {
    pub bar: usize,
    pub onset_quarters: String,
    pub chord: Option<String>,
}

/// Which melody voices `strip_chords` keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeepVoice {
    #[serde(rename = "both")]
    Both,
    Vocal,
    Ins,
}

/// Declared relaxations for `replace_score`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplaceAllowance {
    #[serde(default)]
    pub harmony: bool,
    #[serde(default)]
    pub tempo: bool,
    #[serde(default)]
    pub melody: Option<MelodyScope>,
}

/// The bounded edit-operation set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "op",
    rename_all = "snake_case",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum ScoreEditOperation {
    /// Harmony-only: set or remove chord symbols at exact onsets; notes are split with ties where
    /// a chord lands inside a held note, so every sounding note is preserved.
    Reharmonize { changes: Vec<ChordChange> },
    /// Remove every chord symbol (cover preparation, `cot=melody`); optionally silence one voice.
    StripChords { keep_voice: KeepVoice },
    /// Change only the quarter-note tempo (and, if given, the style prompt).
    SetTempo {
        bpm: u32,
        #[serde(default)]
        style: Option<String>,
    },
    /// Reorder, repeat or drop whole sections; lyrics must be restated to match the new form.
    ArrangeSections {
        section_order: Vec<usize>,
        lyrics: String,
        #[serde(default)]
        style: Option<String>,
    },
    /// Change only the lyrics.
    SetLyrics { lyrics: String },
    /// Change only the style prompt.
    SetStyle { style: String },
    /// Submit a complete edited score (and optional request changes) under declared relaxations.
    ReplaceScore {
        abc: String,
        #[serde(default)]
        allow: ReplaceAllowance,
        #[serde(default)]
        lyrics: Option<String>,
        #[serde(default)]
        style: Option<String>,
        #[serde(default)]
        cot: Option<Cot>,
    },
}

impl ScoreEditOperation {
    /// Stable name of the operation.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Reharmonize { .. } => "reharmonize",
            Self::StripChords { .. } => "strip_chords",
            Self::SetTempo { .. } => "set_tempo",
            Self::ArrangeSections { .. } => "arrange_sections",
            Self::SetLyrics { .. } => "set_lyrics",
            Self::SetStyle { .. } => "set_style",
            Self::ReplaceScore { .. } => "replace_score",
        }
    }

    /// The operation as recorded on the version: `replace_score`'s ABC is the version's own score,
    /// so the record keeps its hash instead of a second copy of the text.
    pub fn record(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self).expect("operation serializes");
        if let (Self::ReplaceScore { abc, .. }, Some(object)) = (self, value.as_object_mut()) {
            object.remove("abc");
            object.insert(
                "abcSha256".to_owned(),
                serde_json::Value::String(super::abc::sha256_hex(abc)),
            );
        }
        value
    }
}

/// A checked edit, ready to be stored as a new version.
#[derive(Clone, Debug)]
pub struct EditOutcome {
    pub score: Score,
    pub request: SongRequest,
    pub contract: ChangeContract,
    pub report: InvariantReport,
}

fn bad(message: impl Into<String>) -> Yue2ScoreError {
    Yue2ScoreError::BadRequest(message.into())
}

/// Apply `operation` to a source version and run the invariant check.
pub fn apply_operation(
    source_score: &Score,
    source_request: &SongRequest,
    operation: &ScoreEditOperation,
) -> Result<EditOutcome, Yue2ScoreError> {
    let mut request = source_request.clone();
    let mut contract = ChangeContract::default();
    let abc = match operation {
        ScoreEditOperation::Reharmonize { changes } => {
            contract.harmony = true;
            reharmonize(source_score, changes)?
        }
        ScoreEditOperation::StripChords { keep_voice } => {
            contract.harmony = true;
            if let Some(silenced) = match keep_voice {
                KeepVoice::Both => None,
                KeepVoice::Vocal => Some(VoiceName::Ins),
                KeepVoice::Ins => Some(VoiceName::Vocal),
            } {
                contract.melody = Some(MelodyScope {
                    voices: vec![silenced],
                    from_bar: 1,
                    to_bar: source_score.bar_count(),
                });
            }
            request.cot = Cot::Melody;
            strip_chords(source_score, *keep_voice)
        }
        ScoreEditOperation::SetTempo { bpm, style } => {
            if *bpm == 0 {
                return Err(bad("bpm must be a positive integer"));
            }
            contract.tempo = true;
            if let Some(style) = style {
                contract.style = true;
                request.style = style.clone();
            }
            replace_line(&source_score.text, 4, &format!("Q:1/4={bpm}"))
        }
        ScoreEditOperation::ArrangeSections {
            section_order,
            lyrics,
            style,
        } => {
            if section_order.is_empty() || section_order.len() > MAX_SECTION_ORDER {
                return Err(bad(format!(
                    "sectionOrder must list 1..={MAX_SECTION_ORDER} source section indices"
                )));
            }
            contract.form = Some(FormChange {
                section_order: section_order.clone(),
            });
            contract.lyrics = true;
            request.lyrics = lyrics.clone();
            if let Some(style) = style {
                contract.style = true;
                request.style = style.clone();
            }
            contract
                .validate(source_score)
                .map_err(Yue2ScoreError::BadRequest)?;
            arrange_sections(source_score, section_order)?
        }
        ScoreEditOperation::SetLyrics { lyrics } => {
            contract.lyrics = true;
            request.lyrics = lyrics.clone();
            source_score.text.clone()
        }
        ScoreEditOperation::SetStyle { style } => {
            contract.style = true;
            request.style = style.clone();
            source_score.text.clone()
        }
        ScoreEditOperation::ReplaceScore {
            abc,
            allow,
            lyrics,
            style,
            cot,
        } => {
            contract.harmony = allow.harmony;
            contract.tempo = allow.tempo;
            contract.melody = allow.melody.clone();
            if let Some(lyrics) = lyrics {
                contract.lyrics = true;
                request.lyrics = lyrics.clone();
            }
            if let Some(style) = style {
                contract.style = true;
                request.style = style.clone();
            }
            if let Some(cot) = cot {
                request.cot = *cot;
            }
            abc.clone()
        }
    };
    contract
        .validate(source_score)
        .map_err(Yue2ScoreError::BadRequest)?;
    // Every operation's OUTPUT is bounded, not only submitted text: a structured edit (repeated
    // sections, chords splitting long rests) can grow a score past what the store will read back.
    super::check_abc_size(&abc)?;
    let score = parse_score(&abc).map_err(Yue2ScoreError::Notation)?;
    super::validate_request(&request, &score)?;
    if let ScoreEditOperation::StripChords { keep_voice } = operation {
        if score.voices.iter().any(|voice| !voice.chords.is_empty()) {
            return Err(bad("chord removal left a chord symbol"));
        }
        let silenced = match keep_voice {
            KeepVoice::Both => None,
            KeepVoice::Vocal => Some(1),
            KeepVoice::Ins => Some(0),
        };
        if silenced.is_some_and(|index| !score.voices[index].notes.is_empty()) {
            return Err(bad("the unselected voice was not silenced"));
        }
    }
    let report = check_edit(&contract, source_score, source_request, &score, &request);
    if !report.matches {
        return Err(Yue2ScoreError::Invariant(Box::new(report)));
    }
    if describe_differences(source_score, source_request, &score, &request).matches {
        return Err(bad(
            "the edit changes nothing: the musical content and the request equal the source",
        ));
    }
    Ok(EditOutcome {
        score,
        request,
        contract,
        report,
    })
}

// ---------------------------------------------------------------------------------------------
// Text surgery helpers
// ---------------------------------------------------------------------------------------------

fn line_ending(text: &str) -> &'static str {
    if text.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    }
}

/// Rebuild `text` from its lines with `replacements` (line index → new text) applied.
fn rebuild_lines(text: &str, replacements: &[(usize, String)]) -> String {
    let ending = line_ending(text);
    let mut lines: Vec<String> = text.lines().map(str::to_owned).collect();
    for (index, line) in replacements {
        lines[*index] = line.clone();
    }
    let mut output = lines.join(ending);
    if text.ends_with('\n') {
        output.push_str(ending);
    }
    output
}

fn replace_line(text: &str, index: usize, line: &str) -> String {
    rebuild_lines(text, &[(index, line.to_owned())])
}

/// Decompose `units` into supported token lengths, largest first.
fn duration_pieces(mut units: u64) -> Vec<u64> {
    let mut pieces = Vec::new();
    while units > 0 {
        let piece = DURATIONS
            .iter()
            .rev()
            .map(|value| *value as u64)
            .find(|value| *value <= units)
            .expect("1 is a supported duration");
        pieces.push(piece);
        units -= piece;
    }
    pieces
}

fn duration_text(units: u64) -> String {
    if units == 1 {
        String::new()
    } else {
        units.to_string()
    }
}

fn token_unit_count(token: &Token) -> u64 {
    match token {
        Token::Note { duration, .. } => {
            super::abc::token_units(duration).expect("parsed score has supported durations") as u64
        }
        _ => 0,
    }
}

/// Replace one duration token by tied pieces (`units` each). A rest splits into plain rests; a
/// note repeats its accidental on every piece and ties all but the last, which keeps the original
/// tie — so the sounding note is unchanged.
fn split_token(token: &Token, parts: &[u64]) -> Vec<Token> {
    let Token::Note {
        accidental,
        note,
        octave,
        tie,
        ..
    } = token
    else {
        unreachable!("only duration tokens split");
    };
    let mut pieces = Vec::new();
    for (part_index, part) in parts.iter().enumerate() {
        let part_pieces = duration_pieces(*part);
        for (piece_index, piece) in part_pieces.iter().enumerate() {
            let last = part_index + 1 == parts.len() && piece_index + 1 == part_pieces.len();
            pieces.push(Token::Note {
                accidental: *accidental,
                note: *note,
                octave: octave.clone(),
                duration: duration_text(*piece),
                tie: if *note == 'z' {
                    false
                } else if last {
                    *tie
                } else {
                    true
                },
            });
        }
    }
    pieces
}

fn lex_body(body: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut cursor = 0;
    while cursor < body.len() {
        let character = body[cursor..].chars().next().expect("in bounds");
        if character.is_whitespace() {
            cursor += character.len_utf8();
            continue;
        }
        let (token, next) = lex_token(body, cursor).expect("parsed score lexes");
        tokens.push(token);
        cursor = next;
    }
    tokens
}

/// Set/remove one chord at `onset_units` inside a measure's token list.
fn set_chord_in_bar(
    tokens: &mut Vec<Token>,
    onset_units: u64,
    chord: Option<&str>,
    label: &str,
) -> Result<(), Yue2ScoreError> {
    // Ensure a duration token starts exactly at the onset.
    let mut offset = 0u64;
    let mut found = None;
    for (index, token) in tokens.iter().enumerate() {
        let units = token_unit_count(token);
        if units == 0 {
            continue;
        }
        if offset == onset_units || (offset < onset_units && onset_units < offset + units) {
            found = Some((index, offset, units));
            break;
        }
        offset += units;
    }
    let (index, offset, units) =
        found.ok_or_else(|| bad(format!("{label}: onset is outside the measure")))?;
    let position = if offset == onset_units {
        index
    } else {
        if chord.is_none() {
            return Err(bad(format!("{label}: no chord starts here to remove")));
        }
        let pieces = split_token(
            &tokens[index],
            &[onset_units - offset, offset + units - onset_units],
        );
        tokens.splice(index..=index, pieces);
        index + duration_pieces(onset_units - offset).len()
    };
    // Zero-width chord tokens at this onset sit between the previous duration token and
    // `position`.
    let mut start = position;
    while start > 0 && token_unit_count(&tokens[start - 1]) == 0 {
        start -= 1;
    }
    let mut index = start;
    let mut removed = false;
    let mut position = position;
    while index < position {
        if matches!(tokens[index], Token::Chord(_)) {
            tokens.remove(index);
            position -= 1;
            removed = true;
        } else {
            index += 1;
        }
    }
    match chord {
        Some(symbol) => tokens.insert(position, Token::Chord(symbol.to_owned())),
        None if !removed => {
            return Err(bad(format!("{label}: no chord starts here to remove")));
        }
        None => {}
    }
    Ok(())
}

fn reharmonize(score: &Score, changes: &[ChordChange]) -> Result<String, Yue2ScoreError> {
    if changes.is_empty() || changes.len() > MAX_CHORD_CHANGES {
        return Err(bad(format!(
            "reharmonize takes 1..={MAX_CHORD_CHANGES} chord changes"
        )));
    }
    let bars = &score.voices[0].bars;
    // Validate and resolve every change first; group them by measure.
    let mut by_bar: std::collections::BTreeMap<usize, Vec<(u64, Option<String>, String)>> =
        std::collections::BTreeMap::new();
    let unit_quarters = score.unit * Frac::int(4);
    for change in changes {
        let label = format!("bar {} onset {}", change.bar, change.onset_quarters);
        if change.bar < 1 || change.bar > bars.len() {
            return Err(bad(format!(
                "{label}: bar must lie within 1..={}",
                bars.len()
            )));
        }
        let onset = Frac::parse(&change.onset_quarters).ok_or_else(|| {
            bad(format!(
                "{label}: onsetQuarters must be a non-negative number or fraction like \"3/2\""
            ))
        })?;
        let bar = bars[change.bar - 1];
        if onset >= bar.length {
            return Err(bad(format!(
                "{label}: onset must be before the measure end ({} quarters)",
                bar.length
            )));
        }
        let units = Frac::new(
            onset.numer() * unit_quarters.denom(),
            onset.denom() * unit_quarters.numer(),
        );
        if !units.is_integer() {
            return Err(bad(format!(
                "{label}: onset is not on the score's L:{} rhythmic grid",
                score.unit
            )));
        }
        if let Some(symbol) = &change.chord {
            if !is_native_chord(symbol) {
                return Err(bad(format!(
                    "{label}: unsupported chord {symbol:?}; use a root/bass note name with one of \
                     the native qualities {:?}",
                    CHORD_QUALITIES
                )));
            }
        }
        let entries = by_bar.entry(change.bar - 1).or_default();
        if entries
            .iter()
            .any(|(existing, _, _)| *existing == units.numer() as u64)
        {
            return Err(bad(format!("{label}: more than one change at this onset")));
        }
        entries.push((units.numer() as u64, change.chord.clone(), label));
    }

    // Rewrite each affected Vocal line.
    let mut replacements: Vec<(usize, String)> = Vec::new();
    for (group_index, group) in score.groups.iter().enumerate() {
        let range = group.first_bar..group.first_bar + group.bar_count;
        if !by_bar.keys().any(|bar| range.contains(bar)) {
            continue;
        }
        let block = &group.blocks[0];
        let context = format!("group {}, Vocal", group_index + 1);
        let bodies = split_music_line(&block.line, &context).map_err(Yue2ScoreError::Notation)?;
        let segments: Vec<&str> = block
            .line
            .strip_suffix('|')
            .expect("validated line")
            .split('|')
            .collect();
        let mut rewritten: Vec<Option<String>> = vec![None; bodies.len()];
        for (local, (body, _)) in bodies.iter().enumerate() {
            let global = group.first_bar + local;
            let Some(entries) = by_bar.get(&global) else {
                continue;
            };
            let bar = bars[global];
            let bar_units = Frac::new(
                bar.length.numer() * unit_quarters.denom(),
                bar.length.denom() * unit_quarters.numer(),
            );
            let mut tokens = if body == "Z" {
                if !bar_units.is_integer() {
                    return Err(bad(format!(
                        "bar {}: this full-measure rest is not a whole number of L: units, so a \
                         chord cannot be placed in it",
                        global + 1
                    )));
                }
                if bar_units.numer() > MAX_REST_EXPANSION_UNITS {
                    return Err(bad(format!(
                        "bar {}: this full-measure rest spans {} L: units; a chord can be placed \
                         in a rest of at most {MAX_REST_EXPANSION_UNITS} units",
                        global + 1,
                        bar_units.numer()
                    )));
                }
                split_token(
                    &Token::Note {
                        accidental: None,
                        note: 'z',
                        octave: String::new(),
                        duration: String::new(),
                        tie: false,
                    },
                    &[bar_units.numer() as u64],
                )
            } else {
                lex_body(body)
            };
            for (onset_units, chord, label) in entries {
                set_chord_in_bar(&mut tokens, *onset_units, chord.as_deref(), label)?;
            }
            rewritten[local] = Some(tokens.iter().map(Token::render).collect());
        }
        let mut out_segments: Vec<String> = Vec::new();
        for (segment_index, segment) in segments.iter().enumerate() {
            let members: Vec<usize> = bodies
                .iter()
                .enumerate()
                .filter(|(_, (_, owner))| *owner == segment_index)
                .map(|(local, _)| local)
                .collect();
            if members.iter().all(|local| rewritten[*local].is_none()) {
                out_segments.push((*segment).to_owned());
            } else {
                for local in members {
                    out_segments.push(
                        rewritten[local]
                            .clone()
                            .unwrap_or_else(|| bodies[local].0.clone()),
                    );
                }
            }
        }
        replacements.push((block.line_index, format!("{}|", out_segments.join("|"))));
    }
    Ok(rebuild_lines(&score.text, &replacements))
}

/// Port of upstream `strip_chords`: drop quoted chord symbols from music lines; with a selected
/// voice, replace the other voice's notes by rests on the same grid.
fn strip_chords(score: &Score, keep_voice: KeepVoice) -> String {
    let mut replacements = Vec::new();
    for group in &score.groups {
        for (voice_index, block) in group.blocks.iter().enumerate() {
            let silence = match keep_voice {
                KeepVoice::Both => false,
                KeepVoice::Vocal => voice_index == 1,
                KeepVoice::Ins => voice_index == 0,
            };
            let line = &block.line;
            let mut output = String::with_capacity(line.len());
            let mut cursor = 0;
            while cursor < line.len() {
                let character = line[cursor..].chars().next().expect("in bounds");
                match lex_token(line, cursor) {
                    Some((Token::Chord(_), next)) => cursor = next,
                    Some((Token::Note { note, duration, .. }, next)) if silence && note != 'z' => {
                        output.push('z');
                        output.push_str(&duration);
                        cursor = next;
                    }
                    Some((_, next)) => {
                        output.push_str(&line[cursor..next]);
                        cursor = next;
                    }
                    None => {
                        output.push(character);
                        cursor += character.len_utf8();
                    }
                }
            }
            replacements.push((block.line_index, output));
        }
    }
    rebuild_lines(&score.text, &replacements)
}

/// Rebuild the score with its sections in `order` (source indices), inserting `M:`/`K:` fields
/// where a moved section would otherwise inherit a different meter or key.
fn arrange_sections(score: &Score, order: &[usize]) -> Result<String, Yue2ScoreError> {
    if let Some(index) = order
        .iter()
        .find(|index| score.sections[**index].ends_with_open_tie)
    {
        return Err(bad(format!(
            "section {index} ends inside a tie that continues into the next section; the \
             native dialect cannot move it independently — close the tie first"
        )));
    }
    let lines: Vec<&str> = score.text.lines().collect();
    let ending = line_ending(&score.text);
    let first = &score.sections[order[0]];
    let mut output: Vec<String> = lines[..8].iter().map(|line| (*line).to_owned()).collect();
    output[2] = format!("M:{}", first.inherited_meter);
    output[7] = format!("K:{}", first.inherited_key);
    let mut running_meter = first.inherited_meter;
    let mut running_key = first.inherited_key.clone();
    for &section_index in order {
        let section = &score.sections[section_index];
        for (offset, group_index) in section.groups.clone().enumerate() {
            let group = &score.groups[group_index];
            output.extend(group.comments.iter().cloned());
            for (voice_index, block) in group.blocks.iter().enumerate() {
                output.push(format!("V: {}", super::abc::VOICE_NAMES[voice_index]));
                if offset == 0 {
                    let has = |field: char| block.fields.iter().any(|(name, _)| *name == field);
                    if section.inherited_meter != running_meter && !has('M') {
                        output.push(format!("M:{}", section.inherited_meter));
                    }
                    if section.inherited_key != running_key && !has('K') {
                        output.push(format!("K:{}", section.inherited_key));
                    }
                }
                for (name, value) in &block.fields {
                    output.push(format!("{name}:{value}"));
                }
                output.push(block.line.clone());
            }
        }
        // The state this section leaves behind, as it did in the source.
        let (end_meter, end_key) = section_end_state(score, section_index);
        running_meter = end_meter;
        running_key = end_key;
    }
    let mut text = output.join(ending);
    if score.text.ends_with('\n') {
        text.push_str(ending);
    }
    Ok(text)
}

/// Meter and key in effect at the end of a source section.
fn section_end_state(score: &Score, section_index: usize) -> (super::abc::Meter, String) {
    let section = &score.sections[section_index];
    let end = section.start + section.length;
    let vocal = &score.voices[0];
    let last_bar = vocal.bars[section.first_bar + section.bar_count - 1];
    let key = vocal
        .keys
        .iter()
        .rfind(|event| event.onset < end)
        .map(|event| event.key.clone())
        .unwrap_or_else(|| score.header_key.clone());
    (last_bar.meter, key)
}
