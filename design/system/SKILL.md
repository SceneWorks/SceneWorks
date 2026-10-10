---
name: sceneworks-design
description: Use this skill to generate well-branded interfaces and assets for SceneWorks, either for production or throwaway prototypes/mocks/etc. Contains essential design guidelines, colors, type, fonts, assets, and UI kit components for prototyping.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.

SceneWorks is a desktop-native, local-first AI image & video generation studio.
The design system is OKLCH-token-based with full light/dark theming and seven
user-selectable accent palettes (teal is the brand default). Drive appearance
with `data-theme` (light|dark) and `data-accent` (teal|indigo|cobalt|violet|
coral|amber|emerald) on `<html>`.

If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy
assets out and create static HTML files for the user to view. If working on
production code, copy assets and read the rules here to become an expert in
designing with this brand.

Key files:
- `styles.css` — link this one file; it reaches all tokens + the app-shell /
  component classes. (Split under `tokens/` + `shell.css`.)
- `theme/accents.js` — the accent palette metadata.
- `components/` — React primitives (Logo, Icon, Button, CompactSelector,
  AccentPicker, StatusDot, Modal, Markdown). Each has a `.prompt.md` with usage.
- `ui_kits/studio/` — a full interactive recreation of the SceneWorks studio to
  copy patterns from.
- `assets/` — the real scene-cut logo and vendored webfonts.

Guardrails: sentence case; no emoji; one saturated accent doing the pointing on
cool-gray neutrals; the work-panel (accent top-rule card) is the signature
surface; icons come from the built-in `Icon` set (never hand-roll SVG icons or
use emoji as icons); prefer concrete numbers over adjectives.

If the user invokes this skill without any other guidance, ask them what they
want to build or design, ask some questions, and act as an expert designer who
outputs HTML artifacts _or_ production code, depending on the need.
