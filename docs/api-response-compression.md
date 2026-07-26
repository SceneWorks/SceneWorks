# API response compression

SceneWorks dynamically compresses JSON responses under `/api` when the client
advertises `br` or `gzip`. This is intentionally the dynamic half of the same
delivery policy established by SC-14785: embedded production web assets keep
their build-time `.br`/`.gz`, representation-specific ETag, and cache policy,
while API JSON is compressed at request time. The API layer is explicitly
marked by request path so an embedded `.json` asset cannot be compressed a
second time.

## Policy

- Compression begins at **1,024 bytes**. Tiny health, access, validation, and
  mutation responses stay as identity bytes, avoiding encoder allocation and
  CPU for transfers where the header/framing overhead would erase most gains.
- Supported dynamic codings are **Brotli and gzip**. Equal client quality values
  prefer Brotli; explicit quality values are honored. An unsupported coding
  safely falls back to identity and still carries `Vary: Accept-Encoding` for
  a response large enough to be a compression candidate.
- `tower-http`'s default quality is used: dynamic Brotli level **4** (the
  middleware's documented NGINX-aligned default) and the async-compression gzip
  default. These favor bounded interactive-request CPU over release-asset
  density. Static web assets remain compressed more aggressively at build time
  because they do not spend request CPU.
- Only `application/json` API responses are candidates. Media
  (`/api/v1/projects/:id/files/*` and preview routes), `206 Partial Content`,
  and `text/event-stream` SSE are never compressed. Range headers and streaming
  framing therefore remain unchanged.

Every compressed or negotiated identity candidate includes
`Vary: Accept-Encoding`. Responses that are below the threshold are invariant
and do not add that cache key.

## Bounded large-response evidence

The ignored exact-route test
`large_assets_response_compression_timing` seeds a populated project whose real
`GET /api/v1/projects/:id/assets` response is approximately 2 MB, warms the
catalog path, then records identity, gzip, and Brotli elapsed delivery time and
transferred bytes. Both encoders have a ten-second test timeout as a hard
regression bound.

Run it with:

```powershell
cargo test -p sceneworks-rust-api large_assets_response_compression_timing -- --ignored --nocapture
```

On the SC-14798 Windows development build, the warmed real `/assets` route
measured:

| Representation | Transferred bytes | End-to-end server/body time | Added time vs identity |
| --- | ---: | ---: | ---: |
| identity | 2,364,225 | 112.7 ms | — |
| gzip | 122,268 | 164.1 ms | 51.5 ms |
| Brotli | 104,145 | 273.4 ms | 160.8 ms |

The fixture uses 80 varied generation sets with no more than eight assets in
each set, plus distinct asset names, prompts, timestamps, recipes, settings,
and catalog metadata. Its 19.3× gzip reduction approximates the measured live
catalog instead of relying on one repeated prompt across the whole response.
Both dynamic encoders remained far below the ten-second hard bound; gzip is the
lower-CPU compatibility option and Brotli trades more server time for the
smallest remote transfer.

SC-14789 records browser-visible transferred bytes for `/assets` and `/models`
on RunPod, because local one-shot tests are not remote-network evidence.
