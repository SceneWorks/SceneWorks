**Markdown** — a small, safe Markdown-to-React renderer for first-party copy (prompt guides, model notes, help text). No `dangerouslySetInnerHTML`; link hrefs are restricted to safe schemes. Output is styled by `.markdown-body`.

```jsx
<Markdown content={`# Prompt guide
Renders **bold**, *italic*, \`code\`, and [links](https://example.com).

- dependency-free
- safe

> Block quotes too.`} />
```

- Supports ATX headings, ordered/unordered lists, blockquotes, fenced code, and inline bold/italic/code/links.
- For trusted, first-party content only — it's a compact subset, not a full CommonMark parser.
