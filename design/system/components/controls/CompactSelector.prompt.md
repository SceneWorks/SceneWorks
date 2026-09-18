**CompactSelector** — the SceneWorks switcher pill: a thumbnail + name button that opens a dropdown to change the active item (character, dataset, project). Closes on outside-click or Escape.

```jsx
<CompactSelector
  items={characters}
  selectedId={activeId}
  onSelect={(item) => setActiveId(item.id)}
  onCreate={() => openCreate()}
  createLabel="New character"
  getSubtitle={(c) => `${c.shots} shots`}
  label="Character"
/>
```

- `items` need `{ id, name }`; everything else is optional.
- `onCreate` adds a "＋ create" row at the top of the menu.
- Thumbnails are injected: `getThumbAsset(item)` → `renderThumbnail(asset)`, so the DS stays app-agnostic (pass your own media resolver).
- Use it wherever the sidebar/topbar needs to switch the active object — not for generic form selects (use a native `<select>` for those).
