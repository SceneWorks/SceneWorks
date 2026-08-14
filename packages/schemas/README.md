# SceneWorks Schemas

Manifest JSON schemas live here for the built-in model, LoRA, and recipe-preset catalogs under `config/manifests/`.

Generated admission artifacts with a persisted cross-component wire contract also live here;
`video-memory-curves.schema.json` documents the strict producer/consumer shape for the fitted video
curve bundle under `docs/generated/`. Schema v2 carries a sorted immutable source catalog and, on
each independently fitted complete selector, the exact record-id subset contributed by each source;
consumers validate those subsets against the compiled source bytes before admitting the bundle.

Sidecar and job payload contracts are enforced by the Rust domain types plus the fixtures in `tests/fixtures/rust_migration_contracts/`; keep those fixtures and their tests as the source of truth instead of adding unused schemas here.
