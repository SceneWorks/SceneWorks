# Historical artifact archive (F-182, F-218, F-267)

Historical calibration renders, design handoffs and completed spike payloads live in the
[immutable source archive at `a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f`](https://github.com/SceneWorks/SceneWorks/archive/a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f.tar.gz).
That revision is preserved in protected `main` ancestry. Repository owners maintain this archive
by preserving that history; never force-push it away or rewrite its objects. It uses the existing
repository and its access controls, with no expiring Actions artifacts, new service or credentials.
Removing these paths from the current tree reduces checkout size; Git history still contains them.

[The maintained manifest](historical-artifacts.json) pins the source commit/tree and each original
path, Git blob, file mode, byte length and SHA-256. It is the inventory of the migration, not new
measurement evidence. Existing evidence JSON, stdout/request receipts, logs, closure digests and
memory matrix data retain their original bytes and identities.

## Retrieve and verify

Use Node and the authenticated GitHub CLI (`gh auth status` must grant repository read access).
Restore to an empty directory outside the checkout:

```sh
node scripts/historical-artifacts.mjs restore --output /absolute/path/historical-artifacts
node scripts/historical-artifacts.mjs verify --root /absolute/path/historical-artifacts
```

The command first reads the exact remote commit and recursive tree. It refuses a missing,
truncated or mismatched tree, or a wrong path/blob/mode/size binding. It retrieves exact blobs via
the GitHub Git API, then checks both SHA-256 and the Git object digest before writing with exclusive
creation. It refuses differing existing files and symlink destinations. Retrieval failure is an
error, with no receipt-only substitute, automatic regeneration or model execution. A partial
retrieval can be retried; already restored correct files are verified again.

`--group calibration|design|spikes|experiments|poses|nonruntime` selects one group. For historical
spike reproduction, restore **all groups**: old scripts reference experiment images and root pose
paths. Run historical commands from the restored directory, preserving the original layout.
Read each retained spike report first: its dependencies and local paths describe its original
environment, and some scripts depend on retired workers or external weights. Retrieving the
archive does not execute those scripts or claim they run against today's product.

The source tarball above provides a second retrieval route. Unpack it, then run `verify --root`
against its top-level directory. Git users who already hold the immutable commit can also recover
an individual file with `git show a1d95e2c43eef4a34a3fe5bf420bb2cc2abe181f:path/to/file` and verify
it against the manifest. Do not use a mutable branch or tag as the identity of restored bytes.

## Calibration reproduction

All 61 formerly tracked RGB files (191,889,408 bytes) are in group `calibration`. Their existing
source-session receipts and the five CUDA frame digests remain unchanged. Restore and check the
actual bytes without rendering new images:

```sh
node scripts/historical-artifacts.mjs restore --group calibration --output /absolute/path/historical-artifacts
node scripts/historical-artifacts.mjs verify --group calibration --root /absolute/path/historical-artifacts
node scripts/memory-calibration-harness.mjs check \
  --input docs/generated/memory-calibration-evidence.json \
  --source-root /absolute/path/historical-artifacts
```

The harness continues to read session/request receipts from the checkout and physical MLX renders
from the explicit source root. `--source-root` requires rendered outputs and checks their existing
digest/size receipts. Ordinary receipt-only checks retain the pre-existing policy: missing rendered
outputs are optional, present outputs are hashed, and stdout/request receipts are required. The
archive verifier additionally hashes every restored RGB, including the older CUDA outputs.
No validator, evidence digest, runtime admission rule or dismantled pin-keyed gate changed.
New raw renders belong in their campaign's recoverable artifact storage, with retained digest
receipts; `.gitignore` prevents bulk RGB payloads from returning to normal checkouts.

## Disposition and consumers

| Group | Files | Disposition |
| --- | ---: | --- |
| Calibration RGB | 61 | Archived; evidence JSON and logs stay in place; restore before byte-level reproduction. |
| Design ZIPs | 12 | Archived; the live `design/system/` source, fonts and components stay in place. |
| `scripts/spikes/` payloads | 14 | Archived completed Python probes and detached Rust projects; [index](../../scripts/spikes/README.md) and historical reports explain retrieval. |
| Experiment payloads | 16 | Archived `.py`/`.png` under sc-15360, sc-16962, sc-3494, sc-3734 and sc-4422; reports and landmark JSON remain in place. |
| Root pose PNGs | 46 | Removed identical duplicates; canonical previews remain in `apps/web/public/poses/`, and both Krea smoke consumers use that path. Archive preserves old script inputs. Root pose JSON is retained. |
| `logo-large.png` | 1 | Archived unused logo; README images and runtime branding remain in place. |

Runtime source, compiled-in calibration corpora, `docs/generated/memory-matrix.json`, historical
logs and README screenshots remain tracked. Vendored `Cargo.toml.orig` files remain as upstream
package provenance, not experiment payloads. Historical review reports remain unchanged records
of their original findings; their old artifact locations resolve through this manifest/archive.

No ordinary CI job retrieves this archive or executes learned inference. Offline retrieval
regressions and exact canonical pose checks run in the normal Node test suite. The migration's
full remote round-trip was completed before source removal; provenance is recorded in
[the verification receipt](historical-artifacts-verification.json).
