# SenseNova-U1 8B Infographic V2 Prompt Guide

## Best For

Information-dense visual content: infographics, posters, presentations, comics, resumes, and knowledge illustrations where dense small-text, precise layout, and visual hierarchy all matter. V2 refines the base model with sharper small-text edges, more stable complex layouts, better overall aesthetics, and a fix for the black-background issue. It also handles the same unified surface as the base — image editing, wardrobe-preserving Character Studio, VQA, and Document Studio (interleaved text+image).

## Prompt Shape

Use a structured design brief:

`purpose + subject/content + layout + visual hierarchy + exact text + style + camera/composition + details`

Short prompts under-constrain the model, especially for infographics. Expand simple ideas into a clear layout and content plan.

## Build The Prompt

### Subject

State the topic and main visual subject. For informational images, define the message the viewer should understand.

Good: `an infographic explaining how urban rain gardens reduce street flooding`

### Details

For dense layouts, specify sections, labels, icons, charts, captions, and reading order. Keep every required text string exact and in quotes. V2 renders small labels and dense tables more crisply — you can push more text per panel than with the base model, but still bound each section clearly.

### Style

Name the design language:

- `clean educational infographic`
- `flat vector poster`
- `arXiv-style technical page`
- `presentation slide`
- `comic explainer`

### Layout And Composition

For design layouts, use layout terms:

- `two-column layout`
- `four-card grid`
- `large title at the top`
- `central diagram with callout labels`
- `clear margins and no overlapping text`

### Text And Typography

V2 is tuned for dense text, but it still needs exact instructions. Include font feel, relative size, alignment, and hierarchy. Quote every string you want rendered verbatim.

### Backgrounds

V2 fixes the base model's black-background failure mode, but it is still worth naming the background you want (`white background`, `soft gradient`, `light paper texture`) so the palette stays intentional.

### Editing

For edits, say what to preserve first, then what to change. Keep the edit instruction concrete and ordered.

### Avoid

- Vague one-line prompts for infographics or posters — they under-constrain the layout.
- Approximate or unquoted text; always quote the exact strings you want rendered.
- Conflicting or overlapping layout instructions (e.g. "centered" and "left-aligned" for the same element).
- Cramming too many competing sections into one image; fewer, clearly bounded sections render more reliably.
- Generic quality tags like `masterpiece` or `best quality`; describe concrete content and structure instead.

## Example Prompts

`Create a vertical educational infographic titled "RAIN GARDENS AT WORK". Use a clean flat vector style with a blue and green palette on a white background. Top section: a city street with rain falling. Middle section: a cutaway soil diagram with arrows labeled "runoff", "plant roots", and "filtered water". Bottom section: three benefit cards reading "Less flooding", "Cleaner rivers", and "More habitat". Large readable sans-serif text, crisp small labels, clear margins, no overlapping elements.`

`A one-page resume layout for "Jordan Lee, Product Designer" on a light background. Left sidebar with "Contact", "Skills", and "Education" headers in small bold caps; right column with "Experience" entries in a clean two-line-per-role format. Muted navy accent color, generous whitespace, sharp legible body text.`

## Sources

- [SenseNova-U1 Infographic V2 model card](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT-Infographic-V2)
- [SenseNova-U1 prompt enhancement doc](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT/blob/main/docs/prompt_enhancement.md)
