# Krea 2 Raw Prompt Guide

## Best For

Highest-fidelity, most-controllable **still images** from natural-language prompts. Krea 2 Raw is Krea
AI's **undistilled** foundation image model — a 12B single-stream rectified-flow DiT paired with a
Qwen3-VL-4B text encoder, so it reads your prompt as plain language. Describe the image the way you'd
describe it to a person; it is especially strong on **photographic realism, lighting, and coherent
composition**.

> **Krea 2 Raw** is the full-fidelity base — it runs **TRUE classifier-free guidance**: a real guidance
> scale, an optional **negative prompt**, and ~52 steps. That extra control is exactly what makes it the
> sharpest, most faithful Krea variant. (The distilled **Krea 2 Turbo** is the opposite regime —
> few-step and CFG-free, with no guidance slider and no negative prompt.)

## Prompt Shape

`subject + scene + lighting + composition + camera/framing + aesthetic/style`

Krea reads the whole prompt as intent, so natural descriptive language works better than a pile of
disconnected tags. **Always name the lighting** — it is the strongest single lever on Raw's realism.

## How Krea Reads A Prompt (keep it faithful and concise)

The goal is a clear prompt, not a long one:

- **Already clear → barely change it.** A short, well-defined prompt ("a cup of coffee on a windowsill")
  needs little more than one lighting or style word. Don't invent scenes, props, or mood the prompt
  didn't mention.
- **Be concise, but keep the concrete specifications.** Drop only *empty praise* — subjective filler
  like `masterpiece`, `stunning`, `premium`, `gorgeous`, `award-winning`, `8k`, `ultra-detailed` — which
  adds no visual information. **Keep the concrete photographic specifications:** camera body
  (`Canon EOS R5`), lens and focal length (`50mm`, `85mm f/1.4`), film stock (`Kodak Portra 400`), and
  aperture. Krea understands and honors these — dropping them makes a photographic prompt *worse*, not
  cleaner.
- **Specify the light.** "soft overcast light", "golden-hour backlight", "hard noon sun", "warm
  tungsten interior" — Krea is highly lighting-responsive, and its own showcase prompts all name the
  lighting. This is the single highest-value detail you can add.
- **Use the negative prompt.** Because Raw runs true CFG, a concise **quality negative** measurably
  sharpens the result. The Studio seeds a mild default (`blurry, soft focus, low detail, low quality`)
  into the negative box; you can edit or clear it. Keep it *mild* — an aggressive skin/texture negative
  (e.g. `plastic skin, waxy skin`) over-corrects into an artificial plastic look. State what you want in
  the prompt; use the negative only to push away genuine quality faults.

## Build The Prompt

### Subject

Name the main subject and its visible traits (color, material, clothing, expression). One or two clear
subjects render more coherently than a crowded frame.

### Scene & Lighting

Describe background, foreground, weather, time of day, and — above all — the **lighting**. Naming the
light source, direction, and quality ("soft window light from the left", "golden-hour backlight", "hard
noon sun") drives much of Raw's photorealism.

### Composition & Framing

Direct framing language works: `low angle`, `wide shot`, `medium close-up`, `centered subject`,
`rule of thirds`, `shallow depth of field`.

### Camera & Style

Concrete camera specs are meaningful to Krea — name the body, lens/focal length, film stock, and
aperture when you want a specific photographic look (`shot on a 50mm lens`, `Kodak Portra 400`,
`f/1.8, shallow depth of field`). Concise style labels also work well: `photorealistic`, `cinematic`,
`editorial photography`, `studio portrait`, `film noir`. Name a known style by its name only (e.g.
`Kodak Portra`, `ukiyo-e`, `cyberpunk`) — you don't need to describe what it looks like. For everyday
realistic subjects you don't need to say "photorealistic"; that's already the default.

## Quality & Speed Notes

- **Resolution:** use a native bucket (1024² / 768×1024 / 1024×768 / 1280×720 / 720×1280 / 1536²); width
  and height must be multiples of 16. 1024² is the default.
- **Steps:** ~52 (Raw is undistilled — it needs the full step budget, unlike few-step Turbo).
- **Guidance:** a real classifier-free-guidance scale; **~3.5** is the reference default. Raise it a
  little for stronger prompt adherence, lower it for a softer, more natural look.
- **Negative prompt:** supported and recommended (see above). A concise quality negative is the biggest
  lever on sharpness.
- **Sampler / scheduler:** `default` is the native rectified-flow loop and the best starting point; the
  curated samplers/schedulers are exposed for experimentation.
- **Quantization:** Q8 is the default (near-lossless, ~20.5 GB download — needs a 48 GB-class Mac or a
  ~32 GB-class CUDA GPU); Q4 is a lighter option; the dense bf16 tier doubles as the LoRA training base.

## Using LoRAs

Krea LoRAs train on the full **Krea 2 Raw** base and apply at inference here on that same base, so their
effect reads at full strength (more directly than on the distilled Turbo). Krea LoRAs start at a higher
default apply weight (1.5) than other families:

- If a LoRA's style or subject isn't coming through on a strongly-described scene, raise the weight
  toward **2.0** and leave the prompt a little room rather than over-specifying every detail.
- If a LoRA over-dominates, lower it. The weight slider is the main lever; the default is just a
  starting point.

## Turbo Speed On Raw: the accelerator LoRA & multi-phase denoise

Raw's full-fidelity regime is ~52 steps with true CFG. Two features let you keep that Raw base while
paying far less of the step cost — both run **on the Raw base** (not on distilled Turbo), so you keep
Raw's fidelity and control where it matters.

### The accelerator (turbo) LoRA — one-click ~8-step Raw

Select an **accelerator-role LoRA** — the builtin **Krea 2 Turbo (accelerator)**, or any LoRA of type
`acceleration` — on a plain Raw **text-to-image** job. The render switches to the distilled **Turbo
sampling regime**: a fixed few-step schedule (**~8 steps**, **CFG-off**), while still loading the **Raw
base weights** with the LoRA folded in additively. Net effect: *Raw base + LoRA, sampled as Turbo* —
roughly Turbo speed (~8 steps instead of 52) at close to Raw fidelity (the community `raw+lora` recipe).

- The accelerated pass is **CFG-off**, so the **guidance slider and negative prompt are inert** for it
  (exactly like Turbo). State everything you want in the positive prompt.
- **Text-to-image only.** A Raw job that also carries an **img2img reference** keeps the normal full-CFG
  Raw regime — the reference is honored and the accelerator LoRA still applies additively — it does not
  switch to the few-step turbo pass.
- Stack a **style/subject LoRA alongside** the accelerator to accelerate a LoRA render.

### Multi-phase denoise — Raw structure, Turbo finish

The Image Studio's **multi-phase editor** splits **one** Raw denoise trajectory (one global schedule)
into ordered **phases**. Each phase has its own step count, its own guidance (**CFG on or off, per
phase**), and its own active subset of the loaded LoRAs (**toggled per phase**). This lets you spend the
expensive true-CFG Raw steps only where they matter — early, on structure and prompt adherence — then
finish fast.

The **canonical workflow** (the Studio's "Turbo finish (4+4)" preset) is two phases:

1. **4 steps — Raw, true-CFG on (guidance ~3.5), no accelerator LoRA.** Builds composition, structure,
   and prompt adherence with the full-fidelity base.
2. **4 steps — Raw + the turbo (accelerator) LoRA, CFG off (guidance 0).** Fast distilled finishing on
   the trajectory the first phase set up.

Tune from there:

- Want stronger adherence or cleaner structure? Add steps to **phase 1** (e.g. `6 + 4`, `8 + 4`).
- Keep the **turbo/accelerator LoRA in the CFG-off phase**, and leave the CFG-on phase base-only (or with
  just your style LoRA) so its guidance is honored.
- Give the CFG-on phase a real guidance (**~3.5**, the Raw default); set each CFG-off phase's guidance to
  **0**.
- A style/subject LoRA can be toggled into either phase; the accelerator belongs only in a CFG-off phase.
- Keep the total step budget modest — a 2–3 phase split well under the 52-step single-phase Raw cost is
  the whole point.

Multi-phase renders from **pure noise**, so it is text-to-image only: edit, strict-pose, img2img-
reference, and PiD-decode jobs are not supported (remove the phase plan to run those).

## Example Prompts

`A weathered lighthouse on a rocky cliff at golden hour, waves breaking below, gulls in the distance.
Wide shot, low angle, warm directional light. Shot on a 35mm lens. Photorealistic, cinematic.`

`Portrait of an elderly fisherman, deep wrinkles, soft overcast window light from the left, shallow
depth of field. 85mm f/1.4, Kodak Portra 400. Editorial photography.`

`A quiet Kyoto side street after rain at dusk, wet stone reflecting paper-lantern light, a lone cyclist.
Medium shot, centered, cool tungsten glow. Cinematic, photorealistic.`

## Sources

- [Krea 2 (Hugging Face)](https://huggingface.co/krea)
- [Krea 2 Raw model card](https://huggingface.co/krea/Krea-2-Raw)
- [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report)
- [Krea 2 Community License](https://www.krea.ai/krea-2-licensing)
