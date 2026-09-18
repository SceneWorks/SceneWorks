**Modal** — SceneWorks' overlay dialog. Renders a blurred backdrop and a centered `.modal-card`; closes on backdrop click or Escape, and moves focus into the dialog on mount.

```jsx
{open ? (
  <Modal onClose={() => setOpen(false)} label="Batch operations">
    <h2 style={{ marginTop: 0 }}>Rename 12 assets</h2>
    <Markdown content={"Applies to the current selection."} />
    <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
      <Button variant="secondary" onClick={() => setOpen(false)}>Cancel</Button>
      <Button variant="primary" onClick={apply}>Apply</Button>
    </div>
  </Modal>
) : null}
```

- Render it conditionally — mounting IS opening; there's no `open` prop.
- Always give it a title: a heading + `labelledBy`, or an `aria-label` via `label`.
- Actions go at the bottom, right-aligned; primary action last.
