# Indexed asset-list performance and reconciliation

`ProjectStore::list_assets` serves the normalized `asset_json` envelope stored
in each project's `project.db`. A clean list does not read or hash asset
sidecars. SceneWorks-owned asset mutations write the sidecar and update the
indexed envelope before returning, so the next list sees them immediately. A
durable `.asset-index-dirty` marker spans the filesystem write and separate
SQLite transaction. If either fails or the process exits between them, the next
indexed read rebuilds from disk before serving data. Marker inspection, repair,
orphan pruning, and marker clear share the same reentrant per-project lock as
the owning mutation, so a concurrent list cannot clear an in-flight writer's
marker. Native frame/render workers persist sidecar, recipe, and index through
the same guarded core operation.

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
from loading. Here "corrupt sidecar" means unreadable or invalid JSON; a
parseable sidecar that violates the asset schema can make reindex fail so the
dirty marker remains and the prior DB transaction stays intact for retry.

## Benchmark harness

The harness records asset count, elapsed time, and one per-request logical
filesystem-call inventory. It covers registry opens/metadata/content reads,
path stats, directory scans, asset-sidecar, generation-set, timeline, and
character JSON reads, poster stats, dirty-marker reads/writes/removes,
directory-create calls, and DB opens. `fs_total` is the sum of those categories.
Counts are logical calls made by the SceneWorks asset-list/reconciliation code;
they do not claim to expose hidden syscalls inside SQLite, the standard library,
or the operating system.

```shell
cargo run --release -p sceneworks-core --example assets_list_benchmark -- synthetic 1000 10
```

To measure an existing project, including one on a mounted network volume:

```shell
cargo run --release -p sceneworks-core --example assets_list_benchmark -- \
  existing <data-dir> <project-id> 10
```

Synthetic setup uses one store to seed data, then creates a fresh `ProjectStore`
for the reported `first-call`, so that measurement includes registry-cache
population. Later measurements are labeled `steady-state-average`. The harness
does not evict the OS filesystem cache and therefore does not call the first
measurement "cold storage." For a real cold-cache comparison, evict or remount
using the storage provider's supported procedure before starting the process,
and record that procedure with the output. The generation-set, timeline,
character, poster, registry, path, marker, directory, and DB counts remain
visible for SC-14799 rather than being treated as part of this sidecar fix.

## SC-14787 validation record

Windows local SSD, release build, 500 synthetic indexed image assets sharing
one generation-set id, 10 steady-state iterations. This is first-call versus
steady-state evidence, not an evicted cold-cache measurement:

```text
storage=synthetic-local path=<temporary local directory>
first-call: assets=500 elapsed_ms=5.681 fs_total=15 registry_opens=2 registry_metadata_reads=2 registry_content_reads=2 path_stats=1 directory_scans=0 sidecar_reads=0 generation_set_reads=1 timeline_reads=0 character_reads=0 poster_stats=0 index_marker_reads=1 index_marker_writes=0 index_marker_removes=0 directory_create_calls=3 db_opens=3
steady-state-average(10): assets=500 elapsed_ms=4.721 fs_total=12 registry_opens=1 registry_metadata_reads=1 registry_content_reads=1 path_stats=1 directory_scans=0 sidecar_reads=0 generation_set_reads=1 timeline_reads=0 character_reads=0 poster_stats=0 index_marker_reads=1 index_marker_writes=0 index_marker_removes=0 directory_create_calls=3 db_opens=3
```

A populated RunPod network volume was not available in the implementation
environment. Run the existing-project command on RunPod before final rollout;
do not treat the local SSD result as network-volume evidence.
