# MOSS-TTS-Realtime Guide

MOSS-TTS-Realtime is a **streaming text-to-speech** model. You type a script and it speaks it — it is a Speech-tab model, like Kokoro, but it renders the clip **incrementally**: the first audio arrives well before the full clip finishes, and the Audio Studio shows the stream progressing as it renders.

## Installation

MOSS-TTS-Realtime runs natively (Candle) on every platform, on CPU / Accelerate. Install it once from the **Models** screen. It is two matched downloads from the shared Hugging Face cache (Apache-2.0): the ~4.66 GB autoregressive backbone (`OpenMOSS-Team/MOSS-TTS-Realtime`, a Qwen3-1.7B brain plus a local/depth transformer) and its required ~7.1 GB codec co-requisite (`OpenMOSS-Team/MOSS-Audio-Tokenizer`, which turns the model's speech tokens into a waveform). Both install together.

## Writing the script

- Type the words you want spoken, with normal punctuation — periods and commas shape the pacing.
- English and Chinese are supported; pick the language that matches your script.
- There is no fixed voice bank: the model speaks in its own natural voice. For a specific cloned voice, use the **Voice Clone** tab instead.

## Streaming

As the clip renders, the results card advances chunk by chunk. The current browser player becomes audible after the fully assembled clip is saved; it does not yet play partial chunks while synthesis is still running.

## Duration

Output is 24 kHz mono. Ask for the length you need; longer scripts simply take longer to finish streaming.

## Practical notes

Because synthesis is autoregressive (one block of speech tokens at a time), worker progress can begin well before a long script is complete.

**Refine my prompt** treats the input as a script. It preserves the words and meaning and makes only minimal punctuation or readability adjustments.
