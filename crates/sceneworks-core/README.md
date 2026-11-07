# SceneWorks Core

Shared Rust domain contracts and services used by the Axum API, native workers,
MCP server and desktop application. Core owns request validation, persistent stores,
model/checkpoint facts and media/workflow metadata. It depends on SQLite and media
libraries and on the pinned inference gen-core contracts.

## Static model registry

For all 143 shipped builtins, core joins the v2 product manifest, checked-in
backend descriptor dumps, and memory evidence at build time. The manifest schema
owns distribution, restrictions, menus and presentation. Descriptor dumps own
native image, video, audio, vector and utility provider routes. The typed
`route-policy.json` owns SceneWorks router admission, which is narrower than engine
capability; it replaces the former image/video hand tables while retaining three
legacy job aliases. The generated preview feature export is joined into each spec
at build time, with legacy-only preview rows compiled separately. The evidence store
retains the original provider, artifact-tier, load-shape and calibration payloads.
Host installation and credential observations are supplied by the API or worker
at runtime. The generated snapshot is a typed dump, not another authored table.

The API merges `user.models.jsonc` through the same projection for catalog,
license and job reads. A user can narrow product menus, choose compatible defaults,
edit display text or hide a builtin from the public model list. Explicit
job lookup and license checks retain its registered source.

Regenerate the checked-in static snapshot after a semantic input change:

```sh
cargo run --locked -q -p sceneworks-core --bin dump-model-registry > config/manifests/registry/specs.json
node scripts/rearchitecture/update-catalog-candidate.mjs
```

Until the v1 readers migrate, the product rows in `builtin.models.jsonc` remain
their compatibility input. The core registry test compares every compiled row
with the complete effective manifest, including evidence. The frozen old UI
fixtures stay at their original revision.
