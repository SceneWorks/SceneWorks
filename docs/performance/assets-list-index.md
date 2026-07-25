# Indexed asset-list performance and reconciliation

`ProjectStore::list_assets` serves the normalized `asset_json` envelope stored
in each project's `project.db`. A clean list does not read or hash asset
sidecars. SceneWorks-owned asset mutations write the sidecar and update the
indexed envelope before returning, so the next list sees them immediately.

## Out-of-band sidecar edits

Direct edits to `*.sceneworks.json` files are supported through explicit
reconciliation. After editing sidecars, call:

```text
POST /api/v1/projects/<project-id>/reindex
```

or use `ProjectStore::reindex_project`. Until reindex completes, list responses
continue to use the last indexed envelope. Reindex reads the sidecar contents,
so this contract also covers same-size rewrites and edits made by tools that
preserve file timestamps. A corrupt sidecar is skipped during reindex; a corrupt
indexed envelope is skipped during listing without preventing healthy assets
from loading.

## Benchmark harness

The harness records asset count, elapsed time, and the single per-request
filesystem-operation inventory (`sidecar_reads`, `generation_set_reads`,
`poster_stats`, `directory_create_calls`, and `db_opens`):

```shell
cargo run --release -p sceneworks-core --example assets_list_benchmark -- synthetic 1000 10
```

To measure an existing project, including one on a mounted network volume:

```shell
cargo run --release -p sceneworks-core --example assets_list_benchmark -- \
  existing <data-dir> <project-id> 10
```

Run the existing-project form once with a cold filesystem cache and again warm;
record the storage type and mount/provider with the output. The generation-set,
poster, and DB counts intentionally remain visible for SC-14799 rather than
being treated as part of this sidecar-hashing fix.

## SC-14787 validation record

Windows local SSD, release build, 500 synthetic indexed image assets sharing
one generation-set id, 10 warm iterations:

```text
cold: assets=500 elapsed_ms=5.884 fs_total=7 sidecar_reads=0 generation_set_reads=1 poster_stats=0 directory_create_calls=3 db_opens=3
warm-average(10): assets=500 elapsed_ms=4.918 fs_total=7 sidecar_reads=0 generation_set_reads=1 poster_stats=0 directory_create_calls=3 db_opens=3
```

A populated RunPod network volume was not available in the implementation
environment. Run the existing-project command on RunPod before final rollout;
do not treat the local SSD result as network-volume evidence.
