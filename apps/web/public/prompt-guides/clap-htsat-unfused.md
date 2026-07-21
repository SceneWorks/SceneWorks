# CLAP HTSAT (audio embedder)

LAION CLAP (HTSAT-unfused) is a joint audio-text embedding model. SceneWorks uses it as a small utility dependency — **it does not generate audio, and there is nothing to prompt.** It backs semantic audio validation: embedding an audio clip and a text description into one space to measure how well they match.

## Installation

Install once from the **Models** screen; it downloads the pinned `laion/clap-htsat-unfused` snapshot into the shared Hugging Face cache. The native worker loads it from that cached snapshot — it is never fetched mid-job.

## Practical Notes

There are no settings — the embedder runs automatically as part of audio validation. Apache-2.0, commercial-use OK.
