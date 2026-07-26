# Indexed asset-list performance and reconciliation

`ProjectStore::list_assets` serves the normalized `asset_json` envelope stored
in each project's `project.db`. That envelope includes filesystem-derived
generation-set hydration plus a content-addressed `posterUrl`. Poster bytes and
their SHA-256 digest live in the same indexed asset row, and the dedicated
poster endpoint serves that blob. A clean list therefore performs no poster
filesystem read, stat, or directory enumeration. SceneWorks-owned asset
mutations write the sidecar and update the indexed envelope before returning,
so the next list sees them immediately. A durable `.asset-index-dirty` marker
spans the filesystem write and separate SQLite transaction. If either fails or
the process exits between them, the next indexed read rebuilds from disk before
serving data. Marker
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

At indexing time, a source sibling poster is accepted only when its relative
path is safe, every component is non-symlink, canonical resolution remains
inside the project, the final open uses the platform no-follow flag, and the
file is at most 16 MiB. The bounded payload must fully decode as JPEG under
dimension and allocation limits; its filename alone never determines MIME.
Verified bytes and their digest are committed atomically with the envelope.
Missing or rejected sources commit a null blob and no URL, clearing any previous
value for the same asset id. Adding, replacing, or deleting only the source
sibling requires reindexing; until then the advertised DB-backed URL continues
to resolve independently of that source file. The serving path recomputes the
blob digest and constant-time compares it with both the stored and requested
digests, failing closed if a DB row is inconsistent.

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

Schema v8 stores one poster blob per video asset. This deliberately trades
project DB size (and rollback-journal write volume during poster updates) for
zero poster filesystem work on ordinary lists and an authoritative serving
endpoint. Each blob is capped at 16 MiB. The list query selects `asset_json` but
not `poster_bytes`, so SQLite does not materialize blob contents while listing;
the poster endpoint reads one requested blob. The version-bump rebuild reads
existing sibling posters once. Purge/orphan lifecycle cleanup removes safe
source siblings so later reuse of an id/path cannot inherit old poster content.

## Benchmark harness

The harness records asset count, elapsed time, and one per-request logical
filesystem-call inventory. It covers registry opens/metadata/content reads,
path stats, directory scans, asset-sidecar, generation-set, timeline, and
character JSON reads, poster stats/reads, dirty-marker reads/writes/removes,
directory-create calls, and DB opens. `fs_total` is the sum of those categories.
Counts are logical calls made by the SceneWorks asset-list/reconciliation code;
they do not claim to expose hidden syscalls inside SQLite, the standard library,
or the operating system. The SC-14799 regression additionally pins zero clean
list directory scans and statically rejects `read_dir`/`file_type` calls in the
list implementation, so entry enumeration cannot be collapsed into one
false-green logical operation.

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
first-call: assets=500 elapsed_ms=11.716 fs_total=10 registry_opens=2 registry_metadata_reads=2 registry_content_reads=2 path_stats=1 directory_scans=0 sidecar_reads=0 generation_set_reads=0 timeline_reads=0 character_reads=0 poster_stats=0 poster_reads=0 index_marker_reads=1 index_marker_writes=0 index_marker_removes=0 directory_create_calls=1 db_opens=1
steady-state-average(10): assets=500 elapsed_ms=11.533 fs_total=7 registry_opens=1 registry_metadata_reads=1 registry_content_reads=1 path_stats=1 directory_scans=0 sidecar_reads=0 generation_set_reads=0 timeline_reads=0 character_reads=0 poster_stats=0 poster_reads=0 index_marker_reads=1 index_marker_writes=0 index_marker_removes=0 directory_create_calls=1 db_opens=1
```

The implementation environment had no mounted network volume or RunPod access,
so no network-volume wall-time claim is recorded here. SC-14789 owns the
populated-volume run; use the `existing` command above there and attach its
first-call/steady-state output before final rollout.
