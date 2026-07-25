# MOSS-TTSD Multi-Speaker Guide

MOSS-TTSD v0.5 is a **multi-speaker / long-form dialogue** text-to-speech model. Instead of one voice reading a single prompt, you give it a **segmented script** — an ordered list of turns, each with a speaker — and it renders the whole conversation in one clip, each turn in its own voice. It is a Speech-tab model, like Kokoro, revealed with a segmented-script editor when it is the selected model.

## Expectations

Be aware before you invest in a long script: **v0.5 is an early dialogue model and its output quality is uneven.** It tends to rush delivery, run turns together, clip words at speaker changes, and stop before the end of the script. This is the model's own behaviour, not a SceneWorks limitation — the same script through the upstream reference implementation truncates at least as hard.

For a single clean voice, **Kokoro** or **MOSS-TTS-Realtime** are markedly more reliable. Reach for MOSS-TTSD when you specifically need two voices in one continuous clip and can accept the rough edges.

## Installation

MOSS-TTSD runs natively (Candle) on every platform, on CPU / Accelerate. Install it once from the **Models** screen. It is two matched downloads from the shared Hugging Face cache (Apache-2.0): the ~4.1 GB autoregressive backbone (`OpenMOSS-Team/MOSS-TTSD-v0.5`, a Qwen3 dialogue brain) and its required ~2.1 GB codec co-requisite (`OpenMOSS-Team/XY_Tokenizer_TTSD_V0`, which turns the model's speech tokens into a 24 kHz waveform). Both install together.

## Writing the script

- Add one row per turn. Assign each row a speaker (Speaker 1 / Speaker 2) and type what that speaker says.
- The model honors up to **two distinct speakers** (`[S1]` / `[S2]`), so the editor offers at most that many labels.
- **Keep it short.** Two to four brief turns render far more reliably than a long exchange: the model frequently stops before the end of a longer script, and everything after that point is simply missing from the clip.
- Alternate speakers for a back-and-forth dialogue, or keep the same speaker across several rows for a longer monologue with natural turn breaks.
- Use normal punctuation — periods and commas shape the pacing. **Go easy on exclamation marks**: the model reads them as delivery cues and tends to render whole turns shouted. 20 in-band languages are supported (Chinese, English, and 18 more); write each turn in the language you want spoken.
- There is no fixed voice bank: the model assigns a distinct natural voice to each speaker label. For a specific cloned voice, use the **Voice Clone** tab instead.

Each turn has its own **Refine my prompt** action. Refinement treats that turn as spoken script, preserves its words and meaning, and leaves every other turn and speaker selection unchanged until you choose **Apply**.

## Single voice vs. multi-speaker

A plain single-voice Speech model (Kokoro, MOSS-TTS-Realtime) reads one prompt in one voice. MOSS-TTSD is the model to reach for when you want a _dialogue_ — an interview, a two-person scene, a narrated exchange — rendered as one continuous clip with the turns already voiced apart.

## Duration

Output is 24 kHz mono. The duration control sets an **upper bound** on the render, not a target: the model stops when it decides the dialogue is finished, which is frequently well short of the length you asked for. Expect the clip to be shorter than the script suggests.

Output quality also varies noticeably between renders of the same script — some takes are markedly clearer than others. Leaving the seed blank picks a new one each time, so **regenerating is a legitimate way to shop for a usable take**; pin the seed once you have one you like.
