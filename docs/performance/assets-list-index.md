# Indexed asset-list performance and reconciliation

`ProjectStore::list_assets` serves the normalized `asset_json` envelope stored
in each project's `project.db`. That envelope includes filesystem-derived
generation-set hydration. Indexed video posters are copied to the flat managed
`assets/posters` directory, which a clean list snapshots once; no per-row file
is read or statted. SceneWorks-owned asset mutations write the sidecar and
update the indexed envelope before returning, so the next list sees them
immediately. A durable `.asset-index-dirty` marker spans the filesystem write
and separate SQLite transaction. If either fails or the process exits between
them, the next indexed read rebuilds from disk before serving data. Marker
inspection, repair, orphan pruning, and marker clear share the same reentrant
per-project lock as the owning mutation, so a concurrent list cannot clear an
in-flight writer's marker. Native frame/render workers persist sidecar, recipe,
and index through the same guarded core operation.

## Out-of-band filesystem edits

Direct edits to `*.sceneworks.json`, `generation-sets/*.json`, media, or source
sibling `*.poster.jpg` files are supported through explicit reconciliation.
After editing those files, call:

```text
POST /api/v1/projects/<project-id>/reindex
```

or use `ProjectStore::reindex_project`. Until reindex completes, list responses
continue to use the last indexed envelope, including its embedded generation
set. Reindex reads the current source files, so this contract also covers
deletions, same-size rewrites, and edits made by tools that preserve file
timestamps.

Poster presence has a narrower freshness contract. Indexing copies an existing
source sibling poster to `assets/posters/<asset-id>.poster.jpg` and advertises
that managed path. Deleting the advertised managed file is visible on the next
ordinary list: its URL is suppressed without a reindex. Adding, replacing, or
deleting only the source sibling still requires reindexing to refresh the
managed copy.

A corrupt sidecar is skipped during reindex; a corrupt indexed envelope is
skipped during listing without preventing healthy assets from loading. Here
"corrupt sidecar" means unreadable or invalid JSON; a parseable sidecar that
violates the asset schema can make reindex fail so the dirty marker remains and
the prior DB transaction stays intact for retry.

## SQLite connection and journal policy

A clean asset-list request opens `project.db` once and reuses that connection
for migration/index-readiness checks, the legacy empty-index check, and the
asset query. Repair paths may open again after a transactional rebuild.

`project.db` deliberately remains in SQLite's rollback-journal mode rather than
enabling WAL. SceneWorks supports projects on mounted network volumes, while
SQLite's [WAL design](https://www.sqlite.org/wal.html) requires same-host shared
memory and explicitly does not support network filesystems. A per-project pool
would also retain handles to removable or remounted project paths and complicate
the existing project-file lock/repair lifecycle. One connection per request
removes the redundant opens without narrowing the supported storage contract.

## Benchmark harness

The harness records asset count, elapsed time, and one per-request logical
filesystem-call inventory. It covers registry opens/metadata/content reads,
path stats, directory scans, asset-sidecar, generation-set, timeline, and
character JSON reads, poster stats, dirty-marker reads/writes/removes,
directory-create calls, and DB opens. `fs_total` is the sum of those categories.
Counts are logical calls made by the SceneWorks asset-list/reconciliation code;
they do not claim to expose hidden syscalls inside SQLite, the standard library,
or the operating system. In particular, the managed-poster snapshot is one
logical directory scan per list, but its enumeration bytes and CPU grow with
the number of managed posters. Benchmark wall time therefore remains necessary
alongside the constant logical-operation count, especially on network volumes.

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

## SC-14799 validation record

Windows local SSD, release build, 500 synthetic indexed video assets with 500
distinct generation sets and 500 existing posters, 10 steady-state iterations.
This deliberately reproduces the formerly row-scaled generation-set-read and
poster-stat workload. It is first-call versus steady-state evidence, not an
evicted cold-cache measurement:

```text
storage=synthetic-local-distinct-video-sets path=<temporary local directory>
first-call: assets=500 elapsed_ms=9.384 fs_total=11 registry_opens=2 registry_metadata_reads=2 registry_content_reads=2 path_stats=1 directory_scans=1 sidecar_reads=0 generation_set_reads=0 timeline_reads=0 character_reads=0 poster_stats=0 index_marker_reads=1 index_marker_writes=0 index_marker_removes=0 directory_create_calls=1 db_opens=1
steady-state-average(10): assets=500 elapsed_ms=8.418 fs_total=8 registry_opens=1 registry_metadata_reads=1 registry_content_reads=1 path_stats=1 directory_scans=1 sidecar_reads=0 generation_set_reads=0 timeline_reads=0 character_reads=0 poster_stats=0 index_marker_reads=1 index_marker_writes=0 index_marker_removes=0 directory_create_calls=1 db_opens=1
```

The implementation environment had no mounted network volume or RunPod access,
so no network-volume wall-time claim is recorded here. SC-14789 owns the
populated-volume run; use the `existing` command above there and attach its
first-call/steady-state output before final rollout.
