**AccentPicker** — the topbar accent switcher. Shows the current accent as a swatch; clicking opens the other six. SceneWorks ships seven accent palettes (teal is the brand default) and the whole app recolors from one attribute.

```jsx
const [accent, setAccent] = useState("teal");
useEffect(() => document.documentElement.setAttribute("data-accent", accent), [accent]);

<AccentPicker accent={accent} onChange={setAccent} />
```

- The picker only reports the choice via `onChange`; YOU set `data-accent` on `<html>` (persist to `localStorage` as `sceneworks-accent` to match the app).
- Pair with a theme toggle (`<Icon.Sun />` / `<Icon.Moon />` on an `.icon-btn`) that flips `data-theme` between `light`/`dark`.
- Accent ids + swatches come from `theme/accents.js` (`ACCENTS`, `DEFAULT_ACCENT`, `isAccentId`).
