# MMAudio (video→audio / Foley)

MMAudio generates a **synchronized soundtrack for a silent video clip** (Foley). It is a pure five-component assembly — a DFN5B-CLIP semantic conditioner and a Synchformer frame-sync conditioner drive an MM-DiT flow-matching generator, decoded to a waveform by a mel-VAE + BigVGAN vocoder. Two tiers ship: **16 kHz (small)** and the **44.1 kHz (large)** quality ceiling.

> **Research / non-commercial use only.** The assembled model's effective license is the intersection of its parts — CC-BY-NC-4.0 on the MMAudio checkpoints and the Apple ML Research Model License on the DFN5B-CLIP conditioner (the 44.1 kHz NVIDIA BigVGAN v2 vocoder is MIT). See **About → Licenses**. SceneWorks is non-commercial, so the weights are usable here, but the restriction is recorded and surfaced.

## Installation

Install from the **Models** screen. Each tier downloads its five components — the DFN5B-CLIP conditioner (`apple/DFN5B-CLIP-ViT-H-14-384`), the Synchformer / MM-DiT / mel-VAE weights (`hkchengrex/MMAudio`), and the vocoder (the in-repo BigVGAN for 16 kHz, or the external `nvidia/bigvgan_v2_44khz_128band_512x` for 44.1 kHz). Everything is pinned to exact revisions for a reproducible, offline-resolvable install.

## Practical Notes

There is nothing to prompt directly — MMAudio conditions on a video's frames (an optional text hint is supported by the model). The user-facing video→audio render lands in a later update; today the model is catalogued so it is installable and its components are provisioned for the rendering path.
