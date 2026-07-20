# MOSS-TTSD Multi-Speaker Guide

MOSS-TTSD v0.5 is a **multi-speaker / long-form dialogue** text-to-speech model. Instead of one voice reading a single prompt, you give it a **segmented script** — an ordered list of turns, each with a speaker — and it renders the whole conversation in one clip, each turn in its own voice. It is a Speech-tab model, like Kokoro, revealed with a segmented-script editor when it is the selected model.

## Installation

MOSS-TTSD runs natively (Candle) on every platform, on CPU / Accelerate. Install it once from the **Models** screen. It is two matched downloads from the shared Hugging Face cache (Apache-2.0): the ~4.1 GB autoregressive backbone (`OpenMOSS-Team/MOSS-TTSD-v0.5`, a Qwen3 dialogue brain) and its required ~2.1 GB codec co-requisite (`OpenMOSS-Team/XY_Tokenizer_TTSD_V0`, which turns the model's speech tokens into a 24 kHz waveform). Both install together.

## Writing the script

- Add one row per turn. Assign each row a speaker (Speaker 1 / Speaker 2) and type what that speaker says.
- The model honors up to **two distinct speakers** (`[S1]` / `[S2]`), so the editor offers at most that many labels.
- Alternate speakers for a back-and-forth dialogue, or keep the same speaker across several rows for a longer monologue with natural turn breaks.
- Use normal punctuation — periods and commas shape the pacing. 20 in-band languages are supported (Chinese, English, and 18 more); write each turn in the language you want spoken.
- There is no fixed voice bank: the model assigns a distinct natural voice to each speaker label. For a specific cloned voice, use the **Voice Clone** tab instead.

## Single voice vs. multi-speaker

A plain single-voice Speech model (Kokoro, MOSS-TTS-Realtime) reads one prompt in one voice. MOSS-TTSD is the model to reach for when you want a *dialogue* — an interview, a two-person scene, a narrated exchange — rendered as one continuous clip with the turns already voiced apart.

## Duration

Output is 24 kHz mono, up to ~5 minutes. Longer scripts simply take longer to render; the finished, fully-assembled clip lands in your library and plays back.
