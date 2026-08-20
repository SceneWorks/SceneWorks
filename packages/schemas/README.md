# SceneWorks Schemas

Manifest JSON schemas live here for the built-in model, LoRA, and recipe-preset catalogs under `config/manifests/`.

Generated admission artifacts with a persisted cross-component wire contract also live here;
`video-memory-curves.schema.json` documents the strict producer/consumer shape for the fitted video
curve bundle under `docs/generated/`. Schema v2 carries a sorted immutable source catalog and, on
each independently fitted complete selector, the exact record-id subset contributed by each source;
consumers validate those subsets against the compiled source bytes before admitting the bundle.

Sidecar and job payload contracts are enforced by the Rust domain types plus the fixtures in `tests/fixtures/rust_migration_contracts/`; keep those fixtures and their tests as the source of truth instead of adding unused schemas here. The explicit exception is `checkpoint-import.schema.json`: its contracts are cross-cutting discovery/import/cache contracts, so its standalone JSON Schema is published alongside the Rust serde types in `sceneworks-core::checkpoint_import`.

The schema is the portable structural gate (version, variants, unknown fields, paths, hashes, and exact duplicate array entries). Every producer and consumer must then deserialize with the matching `*V1` Rust type before admitting a document; that required semantic step checks keyed uniqueness, sorted layers, summary counts, and reference/summary digest agreement that standard JSON Schema cannot express portably.
