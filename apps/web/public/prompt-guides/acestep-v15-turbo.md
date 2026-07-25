# ACE-Step v1.5 XL Turbo Music Guide

ACE-Step v1.5 XL Turbo is a **text-to-music** model. Audio Studio gives it two different text inputs: a **Music Description** (the prompt) for the recording’s overall sound, and optional **Lyrics** for the words and song timeline.

## Installation

ACE-Step runs natively (Candle) on every platform. Install it once from the **Models** screen — it is about 11 GB and downloads into the shared Hugging Face cache from `ACE-Step/acestep-v15-xl-turbo-diffusers` (MIT). It ships its own Oobleck VAE, so there is no separately-licensed audio component.

## Music Description

Describe the whole recording in a compact paragraph. Useful dimensions are:

- genre and style;
- lead and supporting instruments;
- mood and atmosphere;
- vocal gender, character, and delivery when vocals are wanted;
- tempo feel (or use the separate BPM control);
- how verses, choruses, bridges, or instrumental passages change in energy;
- production character, such as intimate live-room, raw demo, spacious cinematic, or polished modern mix.

Example:

> Bittersweet indie pop with intimate female lead vocals, shimmering clean electric guitars, warm melodic bass, restrained acoustic drums, and subtle analog synth pads. Reflective close-miked verses expand into wide, cathartic choruses, ending with layered harmonies and a fading guitar figure.

**Refine my prompt** rewrites only this Music Description. It must return one description—not lyrics, field labels, BPM/key metadata, or commentary. Lyrics and all other controls remain unchanged until you edit them yourself.

## Lyrics and song structure

Lyrics are separate from the Music Description. Use section tags to describe the song’s timeline:

```text
[Intro]
[Instrumental]

[Verse 1]
Cardboard boxes by the door
Dust outlines across the floor

[Chorus]
I'm driving past the county line
Leaving half my heart behind

[Bridge]
Maybe leaving doesn't mean
Losing everything we've been

[Outro]
[Instrumental]
```

Useful tags include `[Intro]`, `[Verse 1]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Instrumental]`, and `[Outro]`. Keep each section’s lyric length realistic for the selected duration. Leave Lyrics empty for an instrumental track.

## Audio Studio controls

- **BPM**: optional whole-number tempo. Leave blank for automatic tempo.
- **Key**: optional musical key such as `C minor` or `A Major`.
- **Language**: select the language matching the lyrics. SceneWorks currently exposes English, Chinese, Japanese, Korean, French, German, Spanish, Italian, Portuguese, and Russian.
- **Length**: target duration, up to 600 seconds. Musical coherence declines as the clip gets longer — a 30-second render holds its beat less consistently than a 12-second one, and it continues from there. The 600-second cap is a capability, not a recommendation; for longer pieces prefer generating sections and arranging them, or use **Extend**.
- **Steps**: advanced solver-step override; blank keeps the Turbo model’s default.

**Seed**: leaving it blank picks a new one each render, so the same inputs give a different take every time. If you get a result you like, pin its seed to keep it and to make small prompt edits comparable — re-rolling is a legitimate way to shop for a better take.

Time signature is not a separate SceneWorks control. If meter matters, describe it in the Music Description (for example, “slow 6/8 ballad”).

The Turbo checkpoint is guidance-distilled, so Audio Studio does not show unsupported CFG or negative-prompt controls.

## Editing existing audio

Choose a Source track to enable the edit modes advertised by the installed model:

- **Inpaint** — regenerate a bounded interior span fresh from the prompt.
- **Repaint** — regenerate a span while conditioning on the surrounding audio for continuity.
- **Extend** — continue the clip past its end; Length is the new total duration.
- **Cover** — restyle the whole source using the Music Description.

For Inpaint/Repaint, set Region start/end in seconds. Edit strength is optional. Output is 48 kHz stereo.
