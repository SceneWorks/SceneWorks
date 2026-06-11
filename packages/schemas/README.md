# SceneWorks Schemas

Manifest JSON schemas live here for the built-in model, LoRA, and recipe-preset catalogs under `config/manifests/`.

Sidecar and job payload contracts are enforced by the Rust domain types plus the fixtures in `tests/fixtures/rust_migration_contracts/`; keep those fixtures and their tests as the source of truth instead of adding unused schemas here.
