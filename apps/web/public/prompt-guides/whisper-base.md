# Whisper base (ASR)

OpenAI Whisper base is a speech-to-text transcriber. SceneWorks uses it as a small utility dependency — **it does not generate audio, and there is nothing to prompt.** It backs audio-validation and ASR round-trip checks (transcribing generated speech to confirm it matches the requested text).

## Installation

Install once from the **Models** screen; it downloads the pinned `openai/whisper-base` snapshot into the shared Hugging Face cache, so other tools reuse it. The native worker loads it from that cached snapshot — it is never fetched mid-job.

## Practical Notes

There are no settings — the transcriber runs automatically as part of audio validation. Apache-2.0, commercial-use OK.
