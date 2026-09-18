**Icon** — the SceneWorks icon set. One stroked 24×24 grid (1.7px strokes, round caps, `currentColor`), so icons inherit text color and sit inline with labels. Access glyphs as members.

```jsx
<Icon.Video />
<Icon.Sun size={16} />
<button className="icon-btn"><Icon.Bell /></button>
<Icon.Star filled />
```

- Sizing via `size` (px); color via CSS `color` on a parent (they use `currentColor`).
- `Play` / `Pause` / `Stars` are filled; everything else is stroked. `Star` takes `filled`.
- Studio glyphs (`Library, Image, Video, Editor, Train, Character, Model, Queue, Logs, Preset`) match the sidebar nav; `Sun`/`Moon` drive the theme toggle, `Sparkle`/`Wand` mark generative actions.
- Don't hand-roll new SVGs — reuse these; if a glyph is missing, note it rather than inventing an off-style one.
