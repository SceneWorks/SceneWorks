// sc-23002 (epic 22988): the redistributable source recording the YuE2 terminal acceptance run
// transcribes (AT2: recording → reviewed transcription → cover). It is generated, never committed:
// the theme of Beethoven's "Ode to Joy" (public domain), played by a harmonic-rich lead over a
// bass line and a soft chord bed, as 44.1 kHz mono PCM16 WAV.
//
// DETERMINISM. Every sample comes from IEEE-754 +, -, *, / and Math.floor/round only — no Math.sin,
// Math.pow or Math.exp, whose results the ECMAScript spec leaves implementation-approximated — so
// the bytes (and their SHA-256) are the same on every host and Node release.

import { createHash } from "node:crypto";

export const RECORDING_GENERATOR_ID = "sceneworks-yue2-ode-to-joy-v1";
export const SILENCE_GENERATOR_ID = "sceneworks-digital-silence-v1";
export const RECORDING_SAMPLE_RATE = 44100;
export const RECORDING_TEMPO_BPM = 96;

const TWO_PI = 2 * Math.PI;
const LN2 = 0.6931471805599453;

/** sin(x) from a Taylor series after reduction to [-π, π]: basic arithmetic only. */
export function detSin(x) {
  let y = x - TWO_PI * Math.floor(x / TWO_PI + 0.5);
  const y2 = y * y;
  let term = y;
  let sum = y;
  for (let n = 1; n < 14; n += 1) {
    term = (-term * y2) / ((2 * n) * (2 * n + 1));
    sum += term;
  }
  return sum;
}

/** 2^x from exact doubling of the integer part and a Taylor series of e^(f·ln2) for the rest. */
export function detPow2(x) {
  const whole = Math.floor(x);
  const t = (x - whole) * LN2;
  let term = 1;
  let sum = 1;
  for (let n = 1; n < 24; n += 1) {
    term = (term * t) / n;
    sum += term;
  }
  let scale = 1;
  for (let i = 0; i < Math.abs(whole); i += 1) scale *= 2;
  return whole >= 0 ? sum * scale : sum / scale;
}

const midiHz = (note) => 440 * detPow2((note - 69) / 12);

// The theme, in C major, as [midi note, beats]: phrase A, then A' with its cadence on the tonic.
const PHRASE = [[64, 1], [64, 1], [65, 1], [67, 1], [67, 1], [65, 1], [64, 1], [62, 1], [60, 1], [60, 1], [62, 1], [64, 1]];
export const MELODY = Object.freeze([
  ...PHRASE, [64, 1.5], [62, 0.5], [62, 2],
  ...PHRASE, [62, 1.5], [60, 0.5], [60, 2],
]);
// One chord per half bar (2 beats): I, V, I, V | I, V, I, V–I.
const C = { root: 36, pad: [52, 55, 60] };
const G = { root: 43, pad: [50, 55, 59] };
const CHORDS = [C, C, G, G, C, C, G, G, C, C, G, G, C, C, G, C];

/** One cycle of a waveform from its harmonic amplitudes, as a lookup table. */
function wavetable(harmonics, size = 4096) {
  const table = new Float64Array(size + 1);
  for (let i = 0; i <= size; i += 1) {
    let value = 0;
    harmonics.forEach((amp, index) => { value += amp * detSin((TWO_PI * (index + 1) * i) / size); });
    table[i] = value;
  }
  return table;
}

const LEAD = wavetable([1, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.12]);
const BASS = wavetable([1, 0.5, 0.25]);
const PAD = wavetable([1, 0.3]);

/** Add one note (linear-attack ADSR) into `out`. */
function addNote(out, { table, hz, start, seconds, gain, rate }) {
  const size = table.length - 1;
  const first = Math.round(start * rate);
  const gate = seconds * 0.92;
  const total = Math.round((gate + 0.06) * rate);
  const step = (hz * size) / rate;
  let phase = 0;
  for (let i = 0; i < total && first + i < out.length; i += 1) {
    const t = i / rate;
    let env;
    if (t < 0.01) env = t / 0.01;
    else if (t < 0.09) env = 1 - (0.3 * (t - 0.01)) / 0.08;
    else if (t < gate) env = 0.7;
    else env = 0.7 * (1 - (t - gate) / 0.06);
    const at = Math.floor(phase);
    const frac = phase - at;
    const value = table[at] + (table[at + 1] - table[at]) * frac;
    out[first + i] += gain * env * value;
    phase += step;
    if (phase >= size) phase -= size;
  }
}

/** A mono PCM16 WAV of the given samples (clamped to [-1, 1]). */
export function pcm16Wav(samples, rate) {
  const data = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    data.writeInt16LE(Math.round(clamped * 32767), i * 2);
  }
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii");
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8, "ascii");
  header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36, "ascii");
  header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]);
}

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

/** The Ode to Joy test recording: its WAV bytes, SHA-256 and what made it. */
export function generateTestRecording() {
  const rate = RECORDING_SAMPLE_RATE;
  const beat = 60 / RECORDING_TEMPO_BPM;
  const beats = MELODY.reduce((sum, [, length]) => sum + length, 0);
  const lead = 0.5;
  const out = new Float64Array(Math.round((lead + beats * beat + 1) * rate));
  let at = lead;
  for (const [note, length] of MELODY) {
    addNote(out, { table: LEAD, hz: midiHz(note), start: at, seconds: length * beat, gain: 0.16, rate });
    at += length * beat;
  }
  CHORDS.forEach((chord, index) => {
    const start = lead + index * 2 * beat;
    addNote(out, { table: BASS, hz: midiHz(chord.root), start, seconds: beat, gain: 0.14, rate });
    addNote(out, { table: BASS, hz: midiHz(chord.root), start: start + beat, seconds: beat, gain: 0.12, rate });
    for (const note of chord.pad) addNote(out, { table: PAD, hz: midiHz(note), start, seconds: 2 * beat, gain: 0.035, rate });
  });
  const bytes = pcm16Wav(out, rate);
  return {
    bytes,
    sha256: sha256(bytes),
    filename: "ode-to-joy-acceptance.wav",
    generator: {
      id: RECORDING_GENERATOR_ID,
      source: "scripts/lib/yue2-test-recording.mjs",
      piece: "Ode to Joy theme (Beethoven, Symphony No. 9) — public-domain composition, synthesized by the generator",
      tempoBpm: RECORDING_TEMPO_BPM,
      melodyNotes: MELODY.length,
      sampleRate: rate,
      channels: 1,
      encoding: "pcm16",
    },
  };
}

/** Digital silence (all-zero PCM16): the recording with no melody at all. */
export function generateSilence(seconds = 10) {
  const bytes = pcm16Wav(new Float64Array(Math.round(seconds * RECORDING_SAMPLE_RATE)), RECORDING_SAMPLE_RATE);
  return {
    bytes,
    sha256: sha256(bytes),
    filename: "digital-silence-acceptance.wav",
    generator: { id: SILENCE_GENERATOR_ID, source: "scripts/lib/yue2-test-recording.mjs", seconds, sampleRate: RECORDING_SAMPLE_RATE, channels: 1, encoding: "pcm16" },
  };
}
