# Boogu Image Prompt Guide

## Best For

General-purpose **still images** from natural-language prompts — Boogu-Image-0.1 is a flow-matching
model (a ~10.3B DiT paired with a Qwen3-VL-8B condition encoder) that reads your prompt as plain
language, so describe the image the way you'd describe it to a person. It is especially strong at
**rendering legible text in the image** (signs, labels, posters, diagrams) in both **English and
Chinese**.

> Three variants ship: **Boogu Image** (full quality, true-CFG, ~50 steps), **Boogu Image Turbo**
> (few-step, ~4 steps, much faster — best for iteration), and **Boogu Image Edit** (instruction-driven
> editing of a source image). They share prompting style; pick the variant by speed vs. task.

## Task Modes

| Mode | Variant | What it does | Inputs you provide |
|---|---|---|---|
| **Text → Image** | Boogu Image / Turbo | Generates an image from the prompt alone. | Prompt |
| **Image → Image (Edit)** | Boogu Image Edit | Follows a text instruction to edit a source image, keeping its overall content. | Source image + instruction |

Tips per mode:

- **Text → Image:** start with **Turbo** to explore compositions quickly, then switch to the full
  **Boogu Image** for the final render when you want maximum quality.
- **Edit:** write the prompt as an *instruction* about what to change ("make it night", "add a red
  scarf", "turn the sign into a neon sign reading OPEN"), not a full re-description of the image. The
  source provides the content; the instruction provides the change.

## Prompt Shape

`subject + scene + composition + camera/framing + aesthetic/style`

Boogu reads the whole prompt as intent, so natural descriptive language works better than a pile of
disconnected tags.

## How Boogu Reads A Prompt (keep it faithful and concise)

The goal is a clear prompt, not a long one:

- **Already clear → barely change it.** A short, well-defined prompt ("a cup of coffee", "a kingfisher
  on a branch") needs little more than maybe one style word. Don't invent scenes, props, actions, or
  mood the prompt didn't mention. Only genuinely abstract prompts ("fruit destined with Newton") need
  real expansion.
- **Be concise.** Short sentences; don't repeat ideas, pile up synonyms ("realistic, photographic,
  ultra-real"), or add empty praise ("stunning", "premium", "high-end", "tech feel"). Quality words
  like `cinematic` or `refined texture` are fine.
- **Don't add camera/photography parameters** the prompt didn't ask for (`35mm`, `f/1.8`, `bokeh`,
  `shallow depth of field`, `cinematic lighting`) — keep them only if you wrote them.
- **Describe what you want, not what you don't.** Boogu doesn't use a negative prompt (it manages
  classifier-free guidance internally), so avoid negations ("no chopsticks", "without text") — state
  the positive scene instead; a thing you don't name simply won't appear.

## Build The Prompt

### Subject

Name the main subject and its visible traits (color, material, clothing, expression). One or two clear
subjects render more coherently than a crowded frame.

### Scene Details

Describe background, foreground, lighting, weather, and time of day. A focused scene reads cleaner.

### Counts, layout & relationships

- **Exact counts:** if you ask for a specific number or arrangement ("seven", "three rows of four"),
  honor it exactly and describe each subject in a fixed order (left-to-right, top-to-bottom).
- **Relationships:** keep logical structure explicit — "a food chain on the grassland" should describe
  the arrows and each icon, not just the words.
- **Diagrams, infographics, posters, menus, UI:** the conciseness advice flips — be *exhaustive*. Spell
  out every node's text, arrow direction, connections, module hierarchy, colors, and layout positions.

### Text In The Image

Boogu renders typography well. Put the **exact** words in quotes, say where they go, and don't add
text the prompt didn't ask for:

- `a storefront with a neon sign that reads "OPEN 24 HOURS"` (give position/color/font/size when it
  matters: top-left, brush script, medium-brown)
- `a book cover titled "山海经" in elegant brush calligraphy`

Make vague text concrete: "the invitation has a name and date" → `the lower invitation reads "Name:
Zhang San — Date: July 2025"`. Keep each quoted string short and explicit; long paragraphs of in-image
text degrade. (For a real existing logo, name it — don't transcribe its text.)

### Composition & Framing

Direct framing language works: `low angle`, `wide shot`, `medium close-up`, `centered subject`,
`rule of thirds`.

### Style

Concise style labels work well: `cinematic`, `photorealistic`, `watercolor`, `flat illustration`,
`studio portrait`, `film noir`. Name a known style by its name only (e.g. `Ghibli`, `ukiyo-e`,
`cyberpunk`) — you don't need to describe what the style looks like. For everyday realistic subjects
you don't need to say "photorealistic"; that's already the default.

## Quality & Speed Notes

- **Resolution:** use a native bucket (1024² / 768×1024 / 1024×768 / 1280×720 / 720×1280); width and
  height must be multiples of 16. 1024² is the default.
- **Steps:** Boogu Image ~25–50 (more = cleaner, slower); Boogu Image Turbo ~4 (few-step distilled —
  more steps rarely help).
- **Guidance:** Boogu Image ~2–5 (default 4); Turbo ignores guidance (CFG-free).
- **Quantization:** Q8 is the default (~23 GB, fits a 64 GB-class Mac). The full-precision bf16 build
  is also hosted for maximum quality if you have the memory.
- **Count:** keep batches modest while iterating.

## Example Prompts

`A weathered lighthouse on a rocky cliff at golden hour, waves breaking below, gulls in the distance.
Wide shot, low angle, warm directional light. Photorealistic, cinematic.`

`A cozy bakery storefront at dusk, warm window glow, a chalkboard sign that reads "FRESH BREAD".
Medium shot, centered. Photorealistic.`

`Hand-drawn water-cycle diagram on light paper: green mountains and a river flowing into a blue ocean,
a sun top-left and clouds top-right; a blue upward arrow labeled "Evaporation", an arrow to the clouds
labeled "Condensation", a downward arrow labeled "Precipitation". Clear labels, bright colors.`

`Edit: change the season to winter — snow on the rooftops, bare trees, cold blue light — keep the
buildings and street layout.`

## Sources

- [Boogu-Image-0.1 (Hugging Face)](https://huggingface.co/Boogu)
- [SceneWorks/boogu-image-mlx (MLX weights)](https://huggingface.co/SceneWorks/boogu-image-mlx)
- [Boogu-Image GitHub (prompt rewriter)](https://github.com/boogu-project/Boogu-Image)
