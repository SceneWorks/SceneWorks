**Logo** — the SceneWorks "scene-cut" brand mark (a rounded square split by a diagonal seam, teal triangle over a theme-flipping ground). Use it in sidebars, headers, splash/empty states. Colors follow `data-theme` and `data-accent` automatically.

```jsx
<Logo size={32} />
<Wordmark size={28} />          {/* mark + "SceneWorks" wordmark lockup */}
```

- `Logo` — just the mark. `size` (px), `title` (a11y label).
- `Wordmark` — the full lockup used in the app sidebar (`.brand`).
- The mark is drawn from CSS variables, so never hardcode its colors — let the theme drive them.
