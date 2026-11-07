# SceneWorks Core

Shared Rust domain contracts and services used by the Axum API, native workers,
MCP server and desktop application. Core owns request validation, persistent stores,
model/checkpoint facts and media/workflow metadata. It depends on SQLite and media
libraries and on the pinned inference gen-core contracts.

## Static model registry

For migrated builtins, core joins the checked-in backend descriptor dumps with the
shipping product fields and memory evidence under `config/manifests/registry` at
build time. The first row, `flux_schnell`, uses both backends' `flux1_schnell`
descriptors. The shared sampler and scheduler menus are the product choices that
both native routes support. The product supplies distribution, restrictions and
presentation. Evidence retains the original provider, artifact-tier, load-shape
and calibration payloads. Host installation and credential observations are
supplied by the API or worker at runtime.

Regenerate the checked-in static snapshot after a semantic input change:

```sh
cargo run --locked -q -p sceneworks-core --bin dump-model-registry > config/manifests/registry/specs.json
node scripts/rearchitecture/update-catalog-candidate.mjs
```

Until the v1 readers migrate, the converted row in `builtin.models.jsonc` is
their compatibility projection. The core registry test compares it with the
complete projected row, including evidence. The frozen old UI fixtures stay at
their original revision.
