# Embedded production web delivery

SceneWorks serves the production Vite bundle from the Rust API when the
`embed-web` feature is enabled. The bundle has one explicit response contract
across desktop, Docker/server, and RunPod.

## Compression

`apps/web/vite-plugin-precompress.js` writes Brotli (`.br`) and gzip (`.gz`)
siblings for compressible files of at least 256 bytes. The source, Brotli, and
gzip representations are embedded into the API binary. The handler selects
`br`, `gzip`, or identity from `Accept-Encoding`, adds
`Vary: Accept-Encoding`, and uses a representation-specific ETag.
Quality values, wildcard exclusions, and identity are honored; if a client
rejects every available representation, the handler returns
`406 Not Acceptable`.

This is deliberately build-time precompression rather than response middleware:

- compression CPU is paid once during the production web build, never on a
  request;
- already-compressed images, audio, video, archives, and web fonts are not
  recompressed;
- response latency and CPU use are bounded to an embedded lookup and header
  selection; and
- the production artifact is identical for desktop, server, and RunPod.

The `.br` and `.gz` siblings are internal representations. They are not public
routes and retain the original asset's content type when selected.

## Caching

- Content-hashed files under `/assets/` use
  `Cache-Control: public, max-age=31536000, immutable`.
- `index.html`, `/theme-init.js`, every other mutable root asset, and SPA
  fallback responses use `Cache-Control: no-cache`.
- Every representation has a strong SHA-256 ETag. Conditional reloads can
  answer `304 Not Modified`, while a changed deployment cannot pin an old
  entrypoint to new hashed chunks.

Vite writes an internal `.sceneworks-immutable-assets` allowlist from the
current Rollup output. Only generated `/assets/` names matching Vite's exact
eight-character base64url hash shape are listed; digits-only date/version
suffixes are deliberately treated as mutable. Rust consumes that allowlist
once and never grants immutable caching by guessing from the request filename.
The allowlist and `.br`/`.gz` siblings are internal metadata, not public routes.

The existing content type, content-security-policy, API access-control boundary,
SPA fallback, and project media/range routes remain owned by their original
handlers.

## Extension point for SC-14798

SC-14798 can add response compression for dynamic API JSON at the router
middleware boundary. It must predicate that middleware to dynamic,
compressible responses and skip any response that already has
`Content-Encoding`. Embedded web responses already carry their final
representation and must not pass through a second compression stage. Static
asset policy and dynamic API policy therefore remain one layered design without
double compression.
