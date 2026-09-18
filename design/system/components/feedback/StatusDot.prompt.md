**StatusDot** — a tiny state dot for inline health/status readouts (worker online, API reachable). Green when `ok`, red otherwise.

```jsx
<span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
  <StatusDot ok />
  <span style={{ color: "var(--text-muted)", fontSize: 12 }}>worker online</span>
</span>
```

- Pairs with a muted label. For richer states (busy/idle), the `.dot`/`.dot.busy`/`.dot.idle` CSS classes exist too.
