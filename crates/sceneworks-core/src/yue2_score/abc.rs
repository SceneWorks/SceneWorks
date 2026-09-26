//! Parser for the bounded two-voice ABC dialect YuE2 plans, SheetSage2 transcriptions and the
//! released editing workflow use.
//!
//! A native port of `skills/yue2-music/scripts/abc_tools.py` (`parse`/`parse_bar`/`report`) at
//! YuE commit `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`, extended with the section/group structure
//! the SceneWorks edit operations need. Like the upstream helper it is deliberately NOT a general
//! ABC parser: it fails closed on anything outside the native dialect (tuplets, grace notes, chord
//! stacks, repeats, slurs, broken rhythm, decorations, `w:` lyrics, custom voices, unsupported key
//! modes or chord qualities) instead of guessing a timing for it. A rejection means "outside this
//! dialect", not "invalid under the full ABC standard" — [`AbcError`] says so.
//!
//! All times are exact fractions of a quarter note ([`Frac`]). Sounding notes are counted AFTER
//! merging ties, and accidentals follow the native exporter's convention (propagated by letter
//! across octaves within a bar; a tied unmarked continuation keeps its pitch across a barline).

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Mul, Sub};

use serde::{Serialize, Serializer};
use sha2::{Digest, Sha256};

/// The two native voices, in the order every group lists them.
pub const VOICE_NAMES: [&str; 2] = ["Vocal", "Ins"];

/// Supported per-token duration multipliers of the `L:` unit.
pub const DURATIONS: [u32; 11] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48];

/// Largest accepted meter numerator (`M:n/d`).
pub const MAX_METER_NUMERATOR: u32 = 64;

/// The native quoted chord-quality vocabulary (`""` = major).
pub const CHORD_QUALITIES: [&str; 15] = [
    "", "m", "dim", "aug", "7", "maj7", "m7", "dim7", "m7b5", "sus4", "sus2", "6", "m6", "7sus4",
    "m(maj7)",
];

const VOICE_DEFINITIONS: [&str; 2] = [
    r#"V: Vocal clef=treble name="Vocal Melody" snm="Vocal""#,
    r#"V: Ins clef=treble name="Ins Melody" snm="Inst.""#,
];

/// What the dialect check establishes — carried on every inspection so no caller mistakes it for
/// a serializer reconstruction or an audio check.
pub const INSPECTION_SCOPE: &str = "native-dialect structural check; not native serializer \
reconstruction or audio verification";

// ---------------------------------------------------------------------------------------------
// Exact rational time
// ---------------------------------------------------------------------------------------------

/// An exact, reduced rational number of quarter notes (den > 0). Serialized like Python's
/// `str(Fraction)`: `"3"` or `"3/2"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Frac {
    num: i128,
    den: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

impl Frac {
    pub const ZERO: Frac = Frac { num: 0, den: 1 };

    /// `num/den` reduced. Panics on a zero denominator (every call site passes a validated one).
    pub fn new(num: i128, den: i128) -> Self {
        assert!(den != 0, "Frac denominator must be non-zero");
        let sign = if den < 0 { -1 } else { 1 };
        let divisor = gcd(num, den).max(1);
        Self {
            num: sign * num / divisor,
            den: sign * den / divisor,
        }
    }

    pub fn int(value: i128) -> Self {
        Self { num: value, den: 1 }
    }

    pub fn numer(self) -> i128 {
        self.num
    }

    pub fn denom(self) -> i128 {
        self.den
    }

    pub fn is_integer(self) -> bool {
        self.den == 1
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    /// Parse a non-negative `"a"` or `"a/b"` (b > 0).
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        let (num, den) = match text.split_once('/') {
            Some((num, den)) => (num.trim(), den.trim()),
            None => (text, "1"),
        };
        if num.is_empty()
            || den.is_empty()
            || !num.bytes().all(|b| b.is_ascii_digit())
            || !den.bytes().all(|b| b.is_ascii_digit())
            || num.len() > 18
            || den.len() > 18
        {
            return None;
        }
        let num: i128 = num.parse().ok()?;
        let den: i128 = den.parse().ok()?;
        (den != 0).then(|| Self::new(num, den))
    }
}

impl Add for Frac {
    type Output = Frac;
    fn add(self, other: Frac) -> Frac {
        Frac::new(
            self.num * other.den + other.num * self.den,
            self.den * other.den,
        )
    }
}

impl Sub for Frac {
    type Output = Frac;
    fn sub(self, other: Frac) -> Frac {
        Frac::new(
            self.num * other.den - other.num * self.den,
            self.den * other.den,
        )
    }
}

impl Mul for Frac {
    type Output = Frac;
    fn mul(self, other: Frac) -> Frac {
        Frac::new(self.num * other.num, self.den * other.den)
    }
}

impl Ord for Frac {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.num * other.den).cmp(&(other.num * self.den))
    }
}

impl PartialOrd for Frac {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Frac {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(formatter, "{}", self.num)
        } else {
            write!(formatter, "{}/{}", self.num, self.den)
        }
    }
}

impl Serialize for Frac {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Notation outside the supported native dialect, or a failed structural invariant of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AbcError {
    pub message: String,
}

impl AbcError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for AbcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AbcError {}

fn fail(condition: bool, message: impl FnOnce() -> String) -> Result<(), AbcError> {
    if condition {
        Err(AbcError::new(message()))
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Keys, meters, chords
// ---------------------------------------------------------------------------------------------

const MAJOR_KEYS: [&str; 15] = [
    "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#",
];
const MINOR_KEYS: [&str; 15] = [
    "Abm", "Ebm", "Bbm", "Fm", "Cm", "Gm", "Dm", "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m",
    "A#m",
];

fn natural_semitone(letter: char) -> i32 {
    match letter {
        'C' => 0,
        'D' => 2,
        'E' => 4,
        'F' => 5,
        'G' => 7,
        'A' => 9,
        'B' => 11,
        other => unreachable!("not a note letter: {other}"),
    }
}

fn letter_index(letter: char) -> usize {
    "CDEFGAB".find(letter).expect("note letter")
}

/// Per-letter alteration (index = C..B) for a supported major/minor key.
pub(crate) fn key_accidentals(key: &str) -> Result<[i32; 7], AbcError> {
    let count = MAJOR_KEYS
        .iter()
        .position(|candidate| *candidate == key)
        .or_else(|| MINOR_KEYS.iter().position(|candidate| *candidate == key))
        .map(|index| index as i32 - 7)
        .ok_or_else(|| {
            AbcError::new(format!(
                "Unsupported key {key:?}; use a standard major or minor K: field"
            ))
        })?;
    let mut result = [0; 7];
    let order = if count > 0 { "FCGDAEB" } else { "BEADGCF" };
    for letter in order.chars().take(count.unsigned_abs() as usize) {
        result[letter_index(letter)] = if count > 0 { 1 } else { -1 };
    }
    Ok(result)
}

/// A time signature `n/d`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Meter {
    pub numerator: u32,
    pub denominator: u32,
}

impl Meter {
    /// Bar length in quarter notes.
    pub fn length(self) -> Frac {
        Frac::new(4 * self.numerator as i128, self.denominator as i128)
    }
}

impl fmt::Display for Meter {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.numerator, self.denominator)
    }
}

impl Serialize for Meter {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

fn positive_decimal(text: &str) -> Option<u32> {
    let bytes = text.as_bytes();
    if bytes.is_empty() || bytes[0] == b'0' || !bytes.iter().all(u8::is_ascii_digit) {
        return None;
    }
    text.parse().ok()
}

fn is_power_of_two_upto_1024(value: u32) -> bool {
    value <= 1024 && value.is_power_of_two()
}

pub(crate) fn meter_value(text: &str) -> Result<Meter, AbcError> {
    let parsed = text.split_once('/').and_then(|(n, d)| {
        let numerator = positive_decimal(n)?;
        let denominator = positive_decimal(d)?;
        Some((numerator, denominator))
    });
    let (numerator, denominator) = parsed.ok_or_else(|| {
        AbcError::new(format!(
            "Unsupported meter {text:?}; write an explicit fraction"
        ))
    })?;
    fail(!is_power_of_two_upto_1024(denominator), || {
        format!("Unsupported meter denominator {denominator}")
    })?;
    // SceneWorks bound (upstream accepts any numerator): real native meters are single or low
    // double digits, and an unbounded one lets a single measure hold millions of units.
    fail(numerator > MAX_METER_NUMERATOR, || {
        format!(
            "Unsupported meter numerator {numerator}; at most {MAX_METER_NUMERATOR} beats per bar"
        )
    })?;
    Ok(Meter {
        numerator,
        denominator,
    })
}

fn pitch_name_len(text: &str) -> Option<usize> {
    let mut chars = text.chars();
    let letter = chars.next()?;
    if !('A'..='G').contains(&letter) {
        return None;
    }
    let rest = &text[1..];
    for accidental in ["bb", "##", "b", "#"] {
        if rest.starts_with(accidental) {
            return Some(1 + accidental.len());
        }
    }
    Some(1)
}

/// Whether `symbol` is a native quoted chord: `<root><quality>[/<bass>]` with a root/bass pitch
/// name (`[A-G](bb|##|b|#)?`) and a quality from [`CHORD_QUALITIES`].
pub fn is_native_chord(symbol: &str) -> bool {
    let Some(root_len) = pitch_name_len(symbol) else {
        return false;
    };
    let rest = &symbol[root_len..];
    let (quality, bass) = match rest.split_once('/') {
        Some((quality, bass)) => (quality, Some(bass)),
        None => (rest, None),
    };
    if !CHORD_QUALITIES.contains(&quality) {
        return false;
    }
    match bass {
        None => true,
        Some(bass) => pitch_name_len(bass) == Some(bass.len()),
    }
}

// ---------------------------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------------------------

/// An explicit accidental mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Accidental {
    DoubleSharp,
    DoubleFlat,
    Sharp,
    Flat,
    Natural,
}

impl Accidental {
    fn alteration(self) -> i32 {
        match self {
            Self::DoubleSharp => 2,
            Self::DoubleFlat => -2,
            Self::Sharp => 1,
            Self::Flat => -1,
            Self::Natural => 0,
        }
    }

    pub(crate) fn text(self) -> &'static str {
        match self {
            Self::DoubleSharp => "^^",
            Self::DoubleFlat => "__",
            Self::Sharp => "^",
            Self::Flat => "_",
            Self::Natural => "=",
        }
    }
}

/// One lexed token of a measure body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Token {
    Chord(String),
    Key(String),
    /// A note (`note` is `A-G`/`a-g`) or a rest (`note == 'z'`).
    Note {
        accidental: Option<Accidental>,
        note: char,
        octave: String,
        /// Raw duration digits (empty = 1 unit).
        duration: String,
        tie: bool,
    },
}

impl Token {
    /// Serialize back to native ABC text.
    pub(crate) fn render(&self) -> String {
        match self {
            Token::Chord(symbol) => format!("\"{symbol}\""),
            Token::Key(key) => format!("[K:{key}]"),
            Token::Note {
                accidental,
                note,
                octave,
                duration,
                tie,
            } => format!(
                "{}{note}{octave}{duration}{}",
                accidental.map(Accidental::text).unwrap_or(""),
                if *tie { "-" } else { "" }
            ),
        }
    }
}

/// Lex one token at `cursor` (which must not be whitespace). `None` = unsupported token.
pub(crate) fn lex_token(body: &str, cursor: usize) -> Option<(Token, usize)> {
    let rest = &body[cursor..];
    if let Some(inner) = rest.strip_prefix('"') {
        let end = inner.find(['"', '\n'])?;
        if inner.as_bytes()[end] != b'"' {
            return None;
        }
        return Some((Token::Chord(inner[..end].to_owned()), cursor + end + 2));
    }
    if let Some(inner) = rest.strip_prefix("[K:") {
        let end = inner.find([']', '\n'])?;
        if end == 0 || inner.as_bytes()[end] != b']' {
            return None;
        }
        return Some((Token::Key(inner[..end].to_owned()), cursor + 3 + end + 1));
    }
    let mut position = 0;
    let mut accidental = None;
    for (text, mark) in [
        ("^^", Accidental::DoubleSharp),
        ("__", Accidental::DoubleFlat),
        ("^", Accidental::Sharp),
        ("_", Accidental::Flat),
        ("=", Accidental::Natural),
    ] {
        if rest.starts_with(text) {
            accidental = Some(mark);
            position = text.len();
            break;
        }
    }
    let note = rest[position..].chars().next()?;
    if !(matches!(note, 'A'..='G' | 'a'..='g' | 'z')) {
        return None;
    }
    position += 1;
    let octave_len = rest[position..]
        .bytes()
        .take_while(|b| matches!(b, b',' | b'\''))
        .count();
    let octave = rest[position..position + octave_len].to_owned();
    position += octave_len;
    let duration_len = rest[position..]
        .bytes()
        .take_while(u8::is_ascii_digit)
        .count();
    let duration = rest[position..position + duration_len].to_owned();
    position += duration_len;
    let tie = rest[position..].starts_with('-');
    if tie {
        position += 1;
    }
    Some((
        Token::Note {
            accidental,
            note,
            octave,
            duration,
            tie,
        },
        cursor + position,
    ))
}

/// Duration digits → supported unit count, or `None` for an unsupported multiplier.
pub(crate) fn token_units(duration: &str) -> Option<u32> {
    if duration.is_empty() {
        return Some(1);
    }
    if duration.len() > 9 {
        return None;
    }
    let units: u32 = duration.parse().ok()?;
    DURATIONS.contains(&units).then_some(units)
}

fn preview(body: &str, cursor: usize) -> String {
    body[cursor..].chars().take(24).collect()
}

// ---------------------------------------------------------------------------------------------
// Parsed score
// ---------------------------------------------------------------------------------------------

/// A sounding note after merging ties.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Note {
    #[serde(rename = "onsetQuarters")]
    pub onset: Frac,
    #[serde(rename = "midiPitch")]
    pub pitch: i32,
    #[serde(rename = "durationQuarters")]
    pub duration: Frac,
}

/// One measure of a voice's time grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Bar {
    #[serde(rename = "startQuarters")]
    pub start: Frac,
    #[serde(rename = "lengthQuarters")]
    pub length: Frac,
    pub meter: Meter,
}

/// A quoted chord symbol and where it starts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChordEvent {
    #[serde(rename = "onsetQuarters")]
    pub onset: Frac,
    pub chord: String,
}

/// A key (header, group field or inline `[K:]`) and where it takes effect.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyEvent {
    #[serde(rename = "onsetQuarters")]
    pub onset: Frac,
    pub key: String,
}

/// The interpreted events of one voice.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VoiceTrack {
    pub notes: Vec<Note>,
    pub bars: Vec<Bar>,
    pub chords: Vec<ChordEvent>,
    pub keys: Vec<KeyEvent>,
}

impl VoiceTrack {
    pub fn duration(&self) -> Frac {
        self.bars
            .last()
            .map(|bar| bar.start + bar.length)
            .unwrap_or(Frac::ZERO)
    }
}

/// The text of one voice block inside a group: its `M:`/`K:` field lines and its music line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupVoiceBlock {
    /// `("M", value)` / `("K", value)` in source order.
    pub fields: Vec<(char, String)>,
    /// 0-based index of the music line in the source text.
    pub line_index: usize,
    pub line: String,
}

/// One native group: a `V: Vocal` block followed by a `V: Ins` block over 1–4 measures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// `% ...` comment lines immediately preceding this group (non-empty ⇒ a new section).
    pub comments: Vec<String>,
    pub blocks: [GroupVoiceBlock; 2],
    /// 0-based index of this group's first measure.
    pub first_bar: usize,
    pub bar_count: usize,
}

/// A structural section: a run of groups introduced by `% label` comment lines (the groups before
/// the first comment, if any, form an unlabeled leading section).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// The comment text with the `% ` prefix removed (several lines joined by `" / "`).
    pub label: Option<String>,
    pub groups: std::ops::Range<usize>,
    /// 0-based first measure.
    pub first_bar: usize,
    pub bar_count: usize,
    pub start: Frac,
    pub length: Frac,
    /// Meter/key in effect when the section begins, BEFORE its first group's own fields.
    pub inherited_meter: Meter,
    pub inherited_key: String,
    /// Whether a tie is still open (in either voice) when the section ends.
    pub ends_with_open_tie: bool,
}

/// A parsed native score.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Score {
    pub text: String,
    pub unit: Frac,
    pub bpm: u32,
    pub header_meter: Meter,
    pub header_key: String,
    /// `[Vocal, Ins]`.
    pub voices: [VoiceTrack; 2],
    pub groups: Vec<Group>,
    pub sections: Vec<Section>,
}

impl Score {
    pub fn vocal(&self) -> &VoiceTrack {
        &self.voices[0]
    }

    pub fn duration(&self) -> Frac {
        self.voices[0].duration()
    }

    pub fn bar_count(&self) -> usize {
        self.voices[0].bars.len()
    }

    /// 1-based measure number containing `onset`, if any.
    pub fn bar_number_at(&self, onset: Frac) -> Option<usize> {
        self.voices[0]
            .bars
            .iter()
            .position(|bar| bar.start <= onset && onset < bar.start + bar.length)
            .map(|index| index + 1)
    }

    pub fn sha256(&self) -> String {
        sha256_hex(&self.text)
    }
}

pub fn sha256_hex(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Mutable per-voice interpretation state (the upstream `Voice` dataclass).
struct VoiceState {
    meter: Meter,
    key: String,
    time: Frac,
    /// `(sounding pitch, written pitch)` of an open tie.
    pending: Option<(i32, i32)>,
    track: VoiceTrack,
}

fn parse_bar(
    body: &str,
    voice: &mut VoiceState,
    unit: Frac,
    context: &str,
) -> Result<(), AbcError> {
    let length = voice.meter.length();
    let start = voice.time;
    let mut offset = Frac::ZERO;
    // Native exporters propagate accidentals by letter, across octaves.
    let mut local: [Option<i32>; 7] = [None; 7];
    if body == "Z" {
        fail(voice.pending.is_some(), || {
            format!("{context}: tie enters a full-measure rest")
        })?;
        offset = length;
    } else {
        let mut cursor = 0;
        while cursor < body.len() {
            let character = body[cursor..].chars().next().expect("in bounds");
            if character.is_whitespace() {
                cursor += character.len_utf8();
                continue;
            }
            let (token, next) = lex_token(body, cursor).ok_or_else(|| {
                AbcError::new(format!(
                    "{context}: unsupported token at {:?}",
                    preview(body, cursor)
                ))
            })?;
            cursor = next;
            fail(offset >= length, || {
                format!("{context}: event after the measure end")
            })?;
            match token {
                Token::Chord(symbol) => {
                    fail(!is_native_chord(&symbol), || {
                        format!("{context}: unsupported chord {symbol:?}")
                    })?;
                    voice.track.chords.push(ChordEvent {
                        onset: start + offset,
                        chord: symbol,
                    });
                }
                Token::Key(key) => {
                    key_accidentals(&key)?;
                    voice.key = key.clone();
                    voice.track.keys.push(KeyEvent {
                        onset: start + offset,
                        key,
                    });
                    local = [None; 7];
                }
                Token::Note {
                    accidental,
                    note,
                    octave,
                    duration,
                    tie,
                } => {
                    let units = token_units(&duration).ok_or_else(|| {
                        AbcError::new(format!(
                            "{context}: unsupported duration {}; split it into tied supported lengths",
                            if duration.is_empty() { "1" } else { &duration }
                        ))
                    })?;
                    let length_of = Frac::int(units as i128) * unit * Frac::int(4);
                    fail(offset + length_of > length, || {
                        format!("{context}: note/rest exceeds meter duration")
                    })?;
                    fail(octave.contains(',') && octave.contains('\''), || {
                        format!("{context}: mixed octave marks")
                    })?;
                    if note == 'z' {
                        fail(accidental.is_some() || !octave.is_empty() || tie, || {
                            format!(
                                "{context}: a rest cannot have accidentals, octave marks or ties"
                            )
                        })?;
                        fail(voice.pending.is_some(), || {
                            format!("{context}: tie enters a rest")
                        })?;
                    } else {
                        let letter = note.to_ascii_uppercase();
                        let index = letter_index(letter);
                        let mut written = 60 + natural_semitone(letter);
                        if note.is_ascii_lowercase() {
                            written += 12;
                        }
                        written += 12
                            * (octave.matches('\'').count() as i32
                                - octave.matches(',').count() as i32);
                        let mut alteration = match local[index] {
                            Some(value) => value,
                            None => key_accidentals(&voice.key)?[index],
                        };
                        if let Some(mark) = accidental {
                            alteration = mark.alteration();
                            local[index] = Some(alteration);
                        }
                        let mut pitch = written + alteration;
                        if let Some((old_pitch, old_written)) = voice.pending {
                            // An unmarked continuation retains its tied accidental across a
                            // barline. It does not alter later untied notes in that bar.
                            if accidental.is_none() && written == old_written {
                                pitch = old_pitch;
                            }
                            fail(pitch != old_pitch, || {
                                format!("{context}: tie changes pitch from {old_pitch} to {pitch}")
                            })?;
                            let last = voice
                                .track
                                .notes
                                .last_mut()
                                .expect("an open tie always follows a note");
                            last.duration = last.duration + length_of;
                        } else {
                            fail(!(0..=127).contains(&pitch), || {
                                format!("{context}: pitch {pitch} is outside MIDI range")
                            })?;
                            voice.track.notes.push(Note {
                                onset: start + offset,
                                pitch,
                                duration: length_of,
                            });
                        }
                        voice.pending = tie.then_some((pitch, written));
                    }
                    offset = offset + length_of;
                }
            }
        }
    }
    fail(offset != length, || {
        format!("{context}: duration {offset} quarter notes != meter duration {length}")
    })?;
    voice.track.bars.push(Bar {
        start,
        length,
        meter: voice.meter,
    });
    voice.time = voice.time + length;
    Ok(())
}

/// Split one music line (without validating bar bodies) into its measures, expanding compressed
/// `Z2`–`Z4` full-measure rests. Returns `(bar body, index of the source segment)` pairs.
pub(crate) fn split_music_line(
    line: &str,
    context: &str,
) -> Result<Vec<(String, usize)>, AbcError> {
    let body = line.strip_suffix('|').ok_or_else(|| {
        AbcError::new(format!(
            "{context}: music line must end with a plain barline"
        ))
    })?;
    let mut bars = Vec::new();
    for (segment, bar) in body.split('|').enumerate() {
        let bar = bar.trim();
        fail(bar.is_empty(), || {
            format!("{context}: empty measure or unsupported double/repeat barline")
        })?;
        match bar.strip_prefix('Z') {
            Some("") => bars.push(("Z".to_owned(), segment)),
            Some(count @ ("2" | "3" | "4")) => {
                let count: usize = count.parse().expect("digit");
                bars.extend(std::iter::repeat_n(("Z".to_owned(), segment), count));
            }
            _ => bars.push((bar.to_owned(), segment)),
        }
    }
    Ok(bars)
}

/// Parse and validate a native YuE2 score. Fails closed on unsupported notation; resolves sounding
/// notes (ties merged), not token counts.
pub fn parse_score(text: &str) -> Result<Score, AbcError> {
    let lines: Vec<&str> = text.lines().collect();
    fail(lines.len() < 12, || {
        "Incomplete native two-voice ABC".to_owned()
    })?;
    fail(lines[0] != "X:1" || lines[1] != "T:", || {
        "Expected native X:1 and blank T: header".to_owned()
    })?;
    let meter_text = lines[2]
        .strip_prefix("M:")
        .ok_or_else(|| AbcError::new("Missing header M:"))?;
    let meter = meter_value(meter_text)?;
    let denominator = lines[3]
        .strip_prefix("L:1/")
        .and_then(positive_decimal)
        .ok_or_else(|| AbcError::new("Expected L:1/<power of two>, usually L:1/32"))?;
    fail(!is_power_of_two_upto_1024(denominator), || {
        "Unsupported L: denominator".to_owned()
    })?;
    let unit = Frac::new(1, denominator as i128);
    let bpm = lines[4]
        .strip_prefix("Q:1/4=")
        .and_then(positive_decimal)
        .ok_or_else(|| AbcError::new("Expected integer quarter-note tempo Q:1/4=<BPM>"))?;
    fail(lines[5..7] != VOICE_DEFINITIONS, || {
        "Preserve native Vocal and Ins voice definitions".to_owned()
    })?;
    let key = lines[7]
        .strip_prefix("K:")
        .ok_or_else(|| AbcError::new("Missing header K:"))?
        .to_owned();
    key_accidentals(&key)?;

    let new_voice = || VoiceState {
        meter,
        key: key.clone(),
        time: Frac::ZERO,
        pending: None,
        track: VoiceTrack {
            keys: vec![KeyEvent {
                onset: Frac::ZERO,
                key: key.clone(),
            }],
            ..VoiceTrack::default()
        },
    };
    let mut voices = [new_voice(), new_voice()];
    let mut groups: Vec<Group> = Vec::new();
    let mut sections: Vec<Section> = Vec::new();
    let mut cursor = 8;
    let mut group_number = 0;
    while cursor < lines.len() {
        let mut comments = Vec::new();
        while cursor < lines.len() && lines[cursor].starts_with("% ") {
            comments.push(lines[cursor].to_owned());
            cursor += 1;
        }
        fail(cursor == lines.len(), || {
            "Dangling section comment without music".to_owned()
        })?;
        group_number += 1;
        let inherited_meter = voices[0].meter;
        let inherited_key = voices[0].key.clone();
        let first_bar = voices[0].track.bars.len();
        let group_start = voices[0].time;
        let mut counts = [0usize; 2];
        let mut blocks: Vec<GroupVoiceBlock> = Vec::with_capacity(2);
        for (voice_index, name) in VOICE_NAMES.iter().enumerate() {
            let context = format!("group {group_number}, {name}");
            fail(
                cursor >= lines.len() || lines[cursor] != format!("V: {name}"),
                || format!("{context}: expected V: {name}"),
            )?;
            cursor += 1;
            let voice = &mut voices[voice_index];
            let mut fields: Vec<(char, String)> = Vec::new();
            while cursor < lines.len()
                && (lines[cursor].starts_with("M:") || lines[cursor].starts_with("K:"))
            {
                let field = lines[cursor].chars().next().expect("non-empty");
                let value = &lines[cursor][2..];
                fail(
                    fields.iter().any(|(existing, _)| *existing == field),
                    || format!("{context}: duplicate {field}: field"),
                )?;
                if field == 'M' {
                    voice.meter = meter_value(value)?;
                } else {
                    key_accidentals(value)?;
                    voice.key = value.to_owned();
                    voice.track.keys.push(KeyEvent {
                        onset: voice.time,
                        key: value.to_owned(),
                    });
                }
                fields.push((field, value.to_owned()));
                cursor += 1;
            }
            fail(cursor >= lines.len(), || {
                format!("{context}: missing music line")
            })?;
            let line = lines[cursor];
            let line_index = cursor;
            cursor += 1;
            let bars = split_music_line(line, &context)?;
            fail(!(1..=4).contains(&bars.len()), || {
                format!("{context}: expected 1–4 measures after expanding Z rests")
            })?;
            counts[voice_index] = bars.len();
            for (bar, _) in &bars {
                let bar_context = format!("{context}, bar {}", voice.track.bars.len() + 1);
                parse_bar(bar, voice, unit, &bar_context)?;
            }
            blocks.push(GroupVoiceBlock {
                fields,
                line_index,
                line: line.to_owned(),
            });
        }
        fail(counts[0] != counts[1], || {
            format!("group {group_number}: voices have different measure counts")
        })?;
        let group_index = groups.len();
        let starts_section = !comments.is_empty() || sections.is_empty();
        if starts_section {
            let label = (!comments.is_empty()).then(|| {
                comments
                    .iter()
                    .map(|line| line[2..].trim().to_owned())
                    .collect::<Vec<_>>()
                    .join(" / ")
            });
            sections.push(Section {
                label,
                groups: group_index..group_index,
                first_bar,
                bar_count: 0,
                start: group_start,
                length: Frac::ZERO,
                inherited_meter,
                inherited_key,
                ends_with_open_tie: false,
            });
        }
        let section = sections.last_mut().expect("section exists");
        section.groups.end = group_index + 1;
        section.bar_count += counts[0];
        section.length = voices[0].time - section.start;
        section.ends_with_open_tie = voices.iter().any(|voice| voice.pending.is_some());
        let blocks: [GroupVoiceBlock; 2] = blocks.try_into().expect("two voice blocks");
        groups.push(Group {
            comments,
            blocks,
            first_bar,
            bar_count: counts[0],
        });
    }
    for (name, voice) in VOICE_NAMES.iter().zip(&voices) {
        fail(voice.pending.is_some(), || {
            format!("{name}: unresolved tie at end of score")
        })?;
    }
    fail(!voices[1].track.chords.is_empty(), || {
        "Native chord symbols belong in Vocal, not Ins".to_owned()
    })?;
    fail(voices[0].track.bars != voices[1].track.bars, || {
        "Voice meter/time grids differ".to_owned()
    })?;
    fail(voices[0].track.keys != voices[1].track.keys, || {
        "Voice key-change timelines differ".to_owned()
    })?;
    let [vocal, ins] = voices;
    Ok(Score {
        text: text.to_owned(),
        unit,
        bpm,
        header_meter: meter,
        header_key: key,
        voices: [vocal.track, ins.track],
        groups,
        sections,
    })
}

// ---------------------------------------------------------------------------------------------
// Inspection (the upstream `report`)
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceInspection {
    pub sounding_notes: usize,
    pub measures: usize,
    pub notes: Vec<Note>,
    pub chords: Vec<ChordEvent>,
    pub keys: Vec<KeyEvent>,
    pub bars: Vec<Bar>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VoicesInspection {
    #[serde(rename = "Vocal")]
    pub vocal: VoiceInspection,
    #[serde(rename = "Ins")]
    pub ins: VoiceInspection,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SectionInspection {
    pub index: usize,
    pub label: Option<String>,
    /// 1-based.
    pub first_bar: usize,
    pub bar_count: usize,
    pub start_quarters: Frac,
    pub length_quarters: Frac,
}

/// Exact events of a score: MIDI pitches, merged note durations, per-voice bar grids, chord
/// onsets, key changes, sections and nominal duration.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScoreInspection {
    pub scope: &'static str,
    pub sha256: String,
    pub bpm: u32,
    pub unit_length: Frac,
    pub duration_quarters: Frac,
    pub nominal_duration_seconds: f64,
    pub sections: Vec<SectionInspection>,
    pub voices: VoicesInspection,
}

pub(crate) fn section_inspections(score: &Score) -> Vec<SectionInspection> {
    score
        .sections
        .iter()
        .enumerate()
        .map(|(index, section)| SectionInspection {
            index,
            label: section.label.clone(),
            first_bar: section.first_bar + 1,
            bar_count: section.bar_count,
            start_quarters: section.start,
            length_quarters: section.length,
        })
        .collect()
}

pub(crate) fn nominal_seconds(score: &Score) -> f64 {
    (score.duration() * Frac::int(60)).to_f64() / score.bpm as f64
}

pub fn inspect(score: &Score) -> ScoreInspection {
    let voice = |track: &VoiceTrack| VoiceInspection {
        sounding_notes: track.notes.len(),
        measures: track.bars.len(),
        notes: track.notes.clone(),
        chords: track.chords.clone(),
        keys: track.keys.clone(),
        bars: track.bars.clone(),
    };
    ScoreInspection {
        scope: INSPECTION_SCOPE,
        sha256: score.sha256(),
        bpm: score.bpm,
        unit_length: score.unit,
        duration_quarters: score.duration(),
        nominal_duration_seconds: nominal_seconds(score),
        sections: section_inspections(score),
        voices: VoicesInspection {
            vocal: voice(&score.voices[0]),
            ins: voice(&score.voices[1]),
        },
    }
}
