**Button** — SceneWorks' action button. Wraps the real CSS action classes so you get a consistent, accent-aware button in one tag. (In the SceneWorks source these are pure CSS classes; this typed wrapper is an intentional design-system addition.)

```jsx
<Button variant="primary" icon={<Icon.Sparkle />}>Generate</Button>
<Button variant="secondary">Cancel</Button>
<Button variant="danger">Delete</Button>
<Button variant="icon" aria-label="Notifications"><Icon.Bell /></Button>
```

- `variant`: `primary` (accent CTA, 44px), `secondary` (neutral, default), `danger` (destructive), `icon` (34×34 square).
- `icon` / `iconRight` take icon elements; pair with `<Icon.* />`.
- All native `<button>` props pass through (`onClick`, `disabled`, `type`, `aria-*`).
- Prefer `primary` for the single main action per panel; everything else is `secondary`.
