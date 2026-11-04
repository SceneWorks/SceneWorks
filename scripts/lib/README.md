# Shared script contracts

Use `file-sha256.mjs` for exact byte/text SHA-256 (`sha256`) and streaming whole-file
SHA-256 (`fileSha256`). `fileSha256` accepts an AbortSignal and a synchronous
`onChunk` callback for existing deadline, liveness and byte-count policies. The
caller retains path confinement, metadata checks and protocol domain separation.
Synchronous/range/tensor/multipart digests are different contracts and retain their
specific owners. Do not replace a content hash with a cache filename.

## JSON bytes are protocol bytes

The canonical script format is `canonicalJson` in `canonical-json.mjs`: compact
JSON, object keys sorted lexicographically by their UTF-8 bytes, array order
preserved, no newline. Integer-looking keys follow that same byte order. It is
not RFC 8785. Inputs are JSON values; callers own validation.

Existing evidence formats keep their established bytes through named compatibility
helpers. Do not change old receipts, closure hashes, history, currency checks or
deliberately disabled gates to make them match the canonical format.

| Existing protocol | Shared format | Preserved details |
| --- | --- | --- |
| StarVector production closure | `canonicalJson` | UTF-8 byte key order, including integer keys; compact; no newline |
| StarVector terminal records and LTX watchdog chains | `legacyUtf16Json` | JavaScript UTF-16 key sort, including integer keys; compact; no newline |
| Memory calibration | `legacySortedJson` / `legacySortedValue` | Recursive UTF-16 sort, ordinary objects then `JSON.stringify`; integer index keys enumerate numerically; pretty output has 2 spaces and a final newline; compact digest has neither |
| KV baseline receipts | `legacySortedJson` | Same ordinary-object ordering; 2 spaces; no final newline |
| Flux3 resource/accounting identity | `legacySortedValue` | Same ordinary-object ordering; compact `JSON.stringify`, existing domain prefixes preserved |

KV numeric semantic seals additionally encode finite numbers as exact f64 bit
strings. That protocol transformation remains in the KV owner. Flux3 approval
serialization retains its existing strict plain-JSON validation. Neither is an
interchangeable generic canonicalizer.

Use the existing `stripJsoncComments` in `jsonc.mjs` before parsing JSONC. JSONC
comment removal is parsing support, never a replacement for hashing original bytes.

## Process execution

`execFileBounded` in `exec.mjs` shares buffered command execution. It defaults to a
five-minute timeout and a 1 MiB limit **per output stream**; callers can supply
larger finite bounds. It forwards argv, cwd, env, encoding, AbortSignal, killSignal
and other native options without shell interpolation or an implicit env merge.
Ordinary calls return Node's native promise, including `.child` for streaming or
cancellation; native exit/spawn/abort errors retain stdout/stderr. Extra inherited
file descriptors use the existing `exec-file-with-fds.mjs` adapter.

YuE2 command probes and StarVector diagnostics use this helper. Their explicit
probe bounds remain authoritative; formerly unbounded YuE2 git/taskkill commands
now have the five-minute default. Long-running services, builds, streamed campaign
logs and resource supervisors retain their spawn owners and lifecycle policies.
Do not replace those with a buffered exec wrapper.

`script-helpers.test.mjs` fixes byte ordering/digests and exercises bounded commands
with metadata-only Node children. Its duplicate guard rejects new trivial byte
SHA-256 wrappers; use the shared owner instead. No helper test runs learned code.
