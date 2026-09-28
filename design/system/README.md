# SceneWorks Design System

The shared design language of **SceneWorks** — a desktop-native AI image and
video generation studio that runs generation directly on your machine's GPU
(MLX on Apple Silicon, candle/CUDA on Windows), with no cloud, no Python, and no
Docker required. This design system is the real `@sceneworks/ui` foundation:
OKLCH design tokens, light/dark theming, seven user-selectable accent palettes,
the app-shell layout, and the core React primitives — the same system that
dresses SceneWorks and its siblings **ChatWorks** and **SoundWorks**.

## Sources

Everything here is lifted from the real SceneWorks code (not reconstructed from
memory). If you have access, read the sources directly to go deeper:

- **Design system package** — https://github.com/SceneWorks/ui
  (`src/theme/theme.css`, `src/theme/accents.js`, `src/shell/shell.css`, and the
  `src/components/*` primitives). This is the ground-truth token + component
  library and the primary source for this project.
- **Product app** — https://github.com/SceneWorks/SceneWorks
  (`apps/web/src/styles.css` is the single source of truth for tokens + classes;
  `apps/web/src/screens/*` and `apps/web/src/components/*` are the real screens
  the UI kit recreates; `apps/web/public/fonts/*` are the vendored webfonts,
  copied into `assets/fonts/`; `apps/web/public/sceneworks-logo.svg` is the mark).
- Related repos worth exploring for product context:
  `SceneWorks/inference` (native runtime, private), `SceneWorks/ChatWorks`,
  `SceneWorks/SoundWorks`.

Explore those repositories to build higher-fidelity designs against the real
product.

## Product context

SceneWorks is organized into **studios** (creative surfaces) and **libraries**
(management surfaces). Capabilities are advertised per device, so the UI only
offers what the current machine can run.

- **Studios** — Image Studio (text-to-image, image-to-image, reference/edit/
  inpaint), Video Studio (text/image-to-video, extend, bridge, person replace),
  Character Studio (consistent identity via InstantID / PuLID-FLUX / LoRAs),
  Document Studio (interleaved text-and-image docs), Training Studio (captioned
  datasets + local LoRA / ControlNet training), Image Editor, Video Editor.
- **Libraries & management** — Assets, Data Sets, Pose Library, Key Point
  Library, Presets, Model Manager (Image / Video / Utility / LoRAs), Queue,
  Logs, Settings.

The product ships a built-in catalog it downloads and runs natively (45 image,
9 video, 10 utility models at time of writing); weights are pulled on first use,
never bundled, and each declares its own memory floor.

---

## CONTENT FUNDAMENTALS

The voice is that of a **precise, local-first power tool for creators** — plain,
technical, and confident. It respects the user's control over their machine and
their media.

- **Person & address.** Second person, describing the user's machine and work:
  "runs directly on **your** machine's GPU", "**your** prompts, models, and
  media never leave the workstation." First person plural is rare and only for
  product intent.
- **Casing.** Sentence case everywhere — buttons ("Generate", "Import", "Add
  token"), headings ("Recent renders"), nav ("Image Studio", "Model Manager").
  The one exception is the **eyebrow / label** treatment: `UPPERCASE`, 0.06–0.08em
  tracking, 700 weight, used for section kickers, stat-chip labels, and field
  labels.
- **Tone.** Matter-of-fact and specific. Prefer concrete numbers and named
  things over adjectives: "12.1 GB · 18 GB min memory", "seed 84213 · cfg 4.5 ·
  28 steps", "Z-Image-Turbo". Marketing superlatives are avoided.
- **Failure copy is honest and actionable.** The product's rule is "fail loudly
  with an actionable error instead of queueing forever." Error states name the
  reason and offer the fix (e.g. a gated model shows "Add token" that jumps to
  Settings). Mirror this: never a vague "Something went wrong" without a next
  step (the ErrorBoundary pairs the message with **Try again** / **Reload**).
- **Punctuation.** Em-dashes for asides, mid-sentence qualifiers set off with
  commas. Units are spelled compactly ("GB", "ms", "px"). Model/technical names
  keep their exact casing and brackets ("FLUX.2 [klein] 9B", "Wan 2.2 (TI2V-5B)").
- **No emoji.** The product uses none in its UI; do not introduce them. Status
  is shown with dots and pills, not emoji.
- **Vibe.** Studio software for someone who knows what they want: dense but
  calm, information-rich without clutter, everything labeled.

**Examples (from the product):**
> "Generation runs directly on your machine's GPU — no cloud, no Python."
> "Weights download on first use, not bundled."
> "A job that needs more memory than the device has fails with a precise reason
> rather than hanging."

---

## VISUAL FOUNDATIONS

**Overall aesthetic.** A calm, professional creative tool. Cool near-neutral
grays carry the surfaces; a single saturated accent (teal by default) does all
the pointing. Flat, bordered surfaces with soft shadows — not glassy, not
gradient-heavy, not playful. The generated media is the color in the room; the
chrome stays quiet.

**Color.** Authored entirely in **OKLCH**. Neutrals are hue 240 at very low
chroma (a hint of cool blue), giving grays that read neutral but never dead. The
accent is **hue-driven**: `--accent-h` / `--warm-h` are the only things a
`[data-accent]` palette changes, and every accent ramp (`--accent`,
`--accent-strong`, `--accent-soft`, `--accent-fg`) derives from that hue — so one
attribute swap recolors the entire app. Seven palettes ship: **teal** (brand
default), indigo, cobalt, violet, coral, amber, emerald. A separate fixed
**violet AI accent** (`--ai*`, hue-locked at 300) marks *generative* affordances
so they read distinctly from the swappable primary. Semantic colors: `--success`
(green 155), `--warn` (amber 70), `--danger` (red 25), each with a soft tint.
Brand constants (theme-independent): `--ink #0f1622`, `--mist #b9babb`,
`--paper #fbfaf6`, `--teal #2fa193`.

**Light & dark.** Both are first-class. Light is `:root`; dark overrides under
`[data-theme="dark"]` (surfaces ~oklch 0.16–0.25, text ~0.96). Accent ramps
lighten and `--accent-fg` flips to ink on dark. Set `data-theme` +
`data-accent` on `<html>`; SceneWorks persists them to `localStorage`
(`sceneworks-theme`, `sceneworks-accent`) and applies before first paint to
avoid a flash.

**Type.** Two families, both self-hosted variable OFL fonts:
**Plus Jakarta Sans** (`--font-ui`) for all UI at 400/500/600/700, with the
stylistic sets `ss01` + `cv11` enabled on `body`; **JetBrains Mono**
(`--font-mono`) for numbers, seeds, cfg/steps, paths, and code. The scale is
compact and functional: topbar title 17/600, section heading 22/600, body 14/1.6,
labels 12/600, mono values ~13. Headings use slightly negative tracking
(-0.01em); eyebrows use positive tracking + uppercase.

**Spacing & density.** A small gap scale (`--gap-1..5` = 6/10/14/20/28) and
fixed control heights (`--control-h 38`, `--row-h 40`, buttons 36, primary CTA
44). The app is information-dense but breathes via consistent 14–20px panel
padding. Sidebar is a fixed `--side-w` 232px column.

**Radii.** `--r-xs 6 · --r-sm 8 · --r-md 12 · --r-lg 16 · --r-xl 22 · --r-pill
999`. Inputs/buttons use sm–md; cards/panels use lg; chips and status pills use
pill.

**Cards.** The signature surface is the **work-panel**: a `--surface` card with a
1px `--border`, `--r-lg` corners, `--shadow-md`, and a 3px **accent gradient
top-rule** (`.work-panel-rule`) hugging the top edge. The rule is one of the
strongest brand signals — one elevated work-panel per page (the "Purpose" zone),
with bare results below on the plain canvas. No colored-left-border cards, no
rounded blobs.

**Borders & shadows.** Everything is defined by hairline 1px borders first,
shadows second. Three elevation steps (`--shadow-sm/md/lg`) plus a focus
`--shadow-glow` (a 4px accent-tinted ring). Dark mode uses deeper, softer
shadows. Inputs on focus: accent border + glow ring.

**Hover / press.** Hover is a quiet surface shift — neutral controls go to
`--surface-hover`; primary buttons darken to `--accent-strong`; icon buttons
tint to `--accent-soft`. Press is a 1px downward nudge (`transform:
translateY(1px)`) on primary/action buttons. Nav items get a left accent bar +
elevated surface when active. No scale-up bounces; motion is understated.

**Motion.** One easing — `--ease-out: cubic-bezier(0.22, 1, 0.36, 1)` — and two
durations: `--t-fast 120ms` (hovers, carets, popovers) and `--t-med 220ms`
(theme/color transitions on `body`). Transitions are fades and small position
shifts; nothing springs or overshoots.

**Transparency & blur.** Used sparingly and purposefully: the topbar is a
near-opaque `color-mix` of `--bg` (a deliberate near-solid, not a heavy blur, to
avoid compositor glitches in the desktop WebView); the modal backdrop is a
translucent ink wash with a light `blur(6px)`; label chips over media use a
`surface`-mix with a small backdrop blur. `color-mix(in oklch, …)` is the
standard way soft tints and translucent borders are built.

**Imagery.** The real color comes from user-generated stills and clips. Where a
thumbnail is absent, the app draws a **diagonal checkerboard placeholder**
(a 45° `repeating-linear-gradient` of accent over a warm-mixed surface) — reused
in this kit's `AssetTile`. Media tiles are `--r-md`, bordered, `object-fit:
cover`. Don't invent photographic imagery; use the checkerboard placeholder or
real assets.

**Layout rules.** Fixed viewport shell (`.app` is `100vh`, `overflow: hidden`);
the sidebar and workspace scroll independently. The topbar is sticky. Content
surfaces get 22px horizontal margins inside the workspace.

---

## ICONOGRAPHY

SceneWorks ships its **own hand-built icon set** — there is no icon-font or
third-party icon dependency. Every glyph lives in one small React module
(`Icon`, ported verbatim from `@sceneworks/ui`): a **24×24 grid, 1.7px strokes,
round caps and joins, `fill="none"`, `stroke="currentColor"`**. Because they use
`currentColor`, icons inherit text color and theme automatically; size via the
`size` prop.

- **Access as members:** `<Icon.Video />`, `<Icon.Sparkle size={16} />`.
- A few glyphs are filled rather than stroked (`Play`, `Pause`, `Stars`);
  `Star` takes a `filled` prop.
- **Studio glyphs** map to the nav: `Library, Image, ImageEditor, Video, Editor,
  Train, Character, Preset, Model, Queue, Logs`. **Action glyphs**: `Search,
  Sparkle, Wand, Plus, Sliders, Bell, Folder, Book, Info, ChevDown, ArrowLeft,
  ArrowRight`. **Theme**: `Sun`, `Moon`. `Sparkle` / `Wand` conventionally mark
  generative actions.
- **Emoji / Unicode as icons:** never. Status is dots and pills.
- Don't hand-roll new SVG icons in a mismatched style — reuse this set; if a
  glyph is genuinely missing, note it and request it rather than inventing one.

The **brand mark** is the "scene cut": a rounded square split by a diagonal seam
— a teal triangle over a theme-flipping ground. It is drawn from CSS variables
(`--logo-ground`, `--logo-seam`, `--teal`) so it tracks theme + accent. The real
`assets/logo.svg` is included; the `Logo` component reproduces it. The wordmark
is "Scene" + "Works" (Works in `--mist`).

---

## Components

Reusable primitives, faithful to `@sceneworks/ui` (import via
`const { X } = window.SceneWorksDesignSystem_b6febf`):

- **Logo** / **Wordmark** (`components/brand/`) — the scene-cut mark and the full
  lockup, theme/accent-aware.
- **Icon** (`components/core/`) — the full stroked icon set (`<Icon.Video />`).
- **Button** (`components/core/`) — action button over the real
  `.primary-action` / `.secondary-action` / `.danger-action` / `.icon-btn`
  classes. *(Intentional addition — see below.)*
- **CompactSelector** (`components/controls/`) — thumbnail + name switcher pill
  with dropdown (active project / character / dataset).
- **AccentPicker** (`components/controls/`) — the topbar accent-color switcher
  across all seven palettes.
- **StatusDot** (`components/feedback/`) — tiny green/red health dot.
- **Modal** (`components/feedback/`) — accessible overlay dialog (Escape +
  backdrop close, focus-in-on-mount).
- **Markdown** (`components/content/`) — safe, dependency-free Markdown renderer
  for prompt guides and help copy.

Accent metadata (`ACCENTS`, `DEFAULT_ACCENT`, `isAccentId`) is exported from
`theme/accents.js`.

### Intentional additions

The SceneWorks source styles buttons as **pure CSS classes** (no exported React
`Button`). Two thin, faithful wrappers are added here for consumer ergonomics
and are the only components without a 1:1 source counterpart:

- **Button** — wraps the real action classes in a typed `variant` API.
- **Wordmark** — the mark + "SceneWorks" lockup used in the sidebar `.brand`
  block, packaged as one component.

Neither introduces new visuals; both compose existing classes/values.

---

## UI kits

- **`ui_kits/studio/`** — an interactive, cosmetic recreation of the SceneWorks
  desktop studio. `index.html` boots the full app shell (sidebar nav + project
  switcher, topbar with live theme + accent switching) and clickable screens:
  **Image Studio** (mode tabs, prompt composer, settings bar, results grid),
  **Library** (toolbar filters, stat strip, asset grid + detail rail), and
  **Model Manager** (Image/Video/Utility/LoRA tabs with size + memory + download
  states). Other studios render branded empty states. Screens are split into
  small files: `shell.jsx`, `imageStudio.jsx`, `library.jsx`, `modelManager.jsx`,
  `app.jsx`.

---

## Project index / manifest

```
styles.css              Global entry — @import list only (link THIS one file)
shell.css               App-shell layout + reusable component classes
tokens/
  fonts.css             @font-face (Plus Jakarta Sans, JetBrains Mono)
  colors.css            OKLCH light/dark + 7 accent palettes + semantic + shadows
  typography.css        --font-ui / --font-mono
  primitives.css        radius, density, control heights, motion, side width
theme/
  accents.js            ACCENTS, DEFAULT_ACCENT, isAccentId
components/
  brand/                Logo, Wordmark
  core/                 Icon, Button
  controls/             CompactSelector, AccentPicker
  feedback/             StatusDot, Modal
  content/              Markdown
guidelines/             Foundation specimen cards (Colors / Type / Spacing / Brand)
ui_kits/studio/         Interactive SceneWorks studio recreation
assets/
  logo.svg              Scene-cut brand mark
  fonts/                Vendored woff2 (+ LICENSE.md)
thumbnail.html          Homepage tile
SKILL.md                Agent Skills manifest
```

The **Design System** tab renders every `@dsCard`-tagged file (foundations +
component cards + the studio kit). `_ds_bundle.js`, `_ds_manifest.json`, and
`_adherence.oxlintrc.json` are generated automatically — do not edit them.

## Fonts

Both webfonts are the **real vendored files** from the SceneWorks repo
(`assets/fonts/`, OFL-1.1). **No substitutions were made** — Plus Jakarta Sans
and JetBrains Mono are the genuine article, variable fonts covering all weights
the UI uses.

## License notes

SceneWorks' own source is AGPL-3.0-or-later (© 2026 Michael Trefry and the
SceneWorks contributors); `@sceneworks/ui` is Apache-2.0. The vendored fonts are
SIL OFL-1.1 (see `assets/fonts/LICENSE.md`). Model weights the product runs keep
their own separate licenses.
