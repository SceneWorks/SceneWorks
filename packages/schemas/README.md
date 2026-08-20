# SceneWorks Schemas

Manifest JSON schemas live here for the built-in model, LoRA, and recipe-preset catalogs under `config/manifests/`.

Generated admission artifacts with a persisted cross-component wire contract also live here;
`video-memory-curves.schema.json` documents the strict producer/consumer shape for the fitted video
curve bundle under `docs/generated/`. Schema v2 carries a sorted immutable source catalog and, on
each independently fitted complete selector, the exact record-id subset contributed by each source;
consumers validate those subsets against the compiled source bytes before admitting the bundle.

Sidecar and job payload contracts are enforced by the Rust domain types plus the fixtures in `tests/fixtures/rust_migration_contracts/`; keep those fixtures and their tests as the source of truth instead of adding unused schemas here. The explicit exception is `checkpoint-import.schema.json`: its contracts are cross-cutting discovery/import/cache contracts, so its standalone JSON Schema is published alongside the Rust serde types in `sceneworks-core::checkpoint_import`.

The schema is the portable structural gate (version, variants, unknown fields, paths, hashes, representable layer limits, and exact duplicate array entries). Every producer and consumer must then deserialize with the matching `*V1` Rust type before admitting a document; that required semantic step checks keyed uniqueness (checkpoint, plan, and layer ids), sorted layers, summary counts (including role-count equality), and reference/summary digest agreement that standard JSON Schema cannot express portably. `ImportLayerV1` and `ManagedProvenanceV1` are validation-gated fragments of those published envelopes rather than separate versioned documents.

The Rust reader captures one complete raw JSON value before selecting an envelope, with serde_json's 128-level recursion guard as the supported resource boundary. Within that bound, decoded-key duplicate detection and version precedence are independent of field order, and future-version bodies are never numerically materialized before the required recompile/rescan decision.

The nonblank text/path patterns explicitly list Unicode's `White_Space` code points instead of using ECMAScript `\s`/`\S`. This keeps AJV's regex dialect aligned with Rust `str::trim()` for values such as U+0085 and U+FEFF.
