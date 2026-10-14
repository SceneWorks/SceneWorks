# H3 owned CPU cache preparation (sc-25777)

This route prepares bytes for S9. It runs no generation, imports no carrier, does not admit any GPU phase, and supplies no terminal media acceptance. The existing terminal workflow's six modes and ten inputs are unchanged. The separate `h3-veda-cache-prepare.yml` dispatch has CPU `build` and `prepare` modes on the unique Mac2 runner (`nax-macos-2`, 5281, `rw-mage`). Its same-repository live run/job proof uses the built-in token; it cannot substitute for global resource queue authority.

## Immutable source and managed download authority

The download product is the accepted ordinary SceneWorks source `16567882f0e07b5d6d95941adc42210d41dff7ab`, with **every original inference pin and Cargo.lock source** remaining `75e03b5abc504d06688b75c9f7bc7a23f0873f5b`. The only source overlay is `epic-25140-cpu-worker-startup.patch`: exact selector `cpu` avoids the worker's MLX memory-limit setter. All other selectors retain the original call and bytes. This is a CPU bootstrap build, separate from the provisional terminal native pin and Binding bridge.

The original ordinary catalog SHA is `07e3079ecd9af63221743fbc6ec7125ffd2ea7f50303f83cd257e49265ab4989`. The source-owned cache catalog carries its two exact H3 model entries, including the managed Veda predictor co-requisite. Publication-base main does not yet contain that co-requisite. The binary build and preparation recheck this capsule against the original catalog; neither rewrites main's product catalog. Before submitting jobs, the running normal API `/api/v1/models` must return the exact predictor repository, immutable revision, single file, digest and size.

Missing bytes are fetched only by the normal `/api/v1/jobs` `model_download` or `lora_download` executor, with exact repository/revision/file scope. No independent HF fetch implementation exists here. Predictor download authority is the ordinary model catalog's managed-file identity and license gate, followed by actual job readback and full payload SHA verification. This preserves E7's managed co-requisite path. The other three files use the same product download transport. Style download does not by itself register a style for generation or prove style quality.

## Source-only recipe and later host execution

After root publishes and verifies this CPU workflow source, choose an epic-owned directory **outside the checkout and shared cache**. The original clone route uses the same APFS volume, for example `/Volumes/Models/epic-25140-owned-cache`. The explicit internal copy route uses exactly `/Users/MTrefry/.h3-veda-epic-25140/private-cache-<published-source-first9>`, separately from the owned `cpu-build-*` root. Do not reuse an unrelated directory. Supply the exact published workflow source SHA in `sceneworks_sha`. Root handles dispatch and readback.

1. Dispatch `build` with `run_dir`. It bootstraps byte-attested owned Python 3.12, installs Rust 1.97.1 into owned toolchain homes, fetches the immutable ordinary source, applies only the CPU guard overlay, and builds the normal API executable and its exact MLX metallib into durable external storage. The source-owned Python supervisor permits only Rustup installation (600-second child bound) and Cargo build (3,600-second child bound), reserves drain time within the original deadline, watches controller PID/birth, and records actual child PID/birth/process-group reaping, descendant drainage, log hashes and tool byte attestations. Failed or uncertain closure cannot yield a reusable build. `cpu-download-build.json` binds source, overlay, catalogs, worker source, original lock, actual binary/metallib, tools and host. An incomplete build is refused, never treated as an attestation. Build source reuse requires its completed immutable attestation; a failed checkout must be inspected before choosing a fresh owned build directory.
2. Dispatch `prepare` using the same root, the actual allowlisted shared root `/Volumes/Models/huggingface`, and the actual host-local `cpu-download-build.json` path. The controller verifies clean exact workflow/helper source and the complete attested build. It uses the byte-bound selected interpreter and a hash-bound immutable plan. No pre-existing terminal manifest or future permanent pin is needed.
3. Preparation verifies actual APFS clone support and same-volume identity. The 190 existing source files are read through declared snapshot paths only; independent `clonefile` blobs are created under the owned HF cache. No foreign writable blob symlinks or hard links are permitted. API and utility-worker launch explicitly override and verify every HF cache alias, including higher-priority `HF_HUB_CACHE`, under the private root; inherited shared-cache aliases cannot redirect writes. Clone failure has no implicit copy fallback. Only the source-owned internal path selects cross-volume streaming. Every payload requires full SHA-256 and size; observed size alone never clears readiness. Verified blobs become read-only. The four explicitly allowed absent tuples are fetched into private writable blobs through owned CPU product services, then verified and made read-only.
4. Each phase has the original 480-second preparation deadline, with a nonzero exit and durable partial/refusal receipt on incompleteness, timeout, APFS refusal, transport ambiguity or hash mismatch. Repeat controlled `prepare` runs over the same immutable closure to resume completed verified files. A complete private blob can recover after a crash before snapshot symlink publication even if the shared source is absent. No uncertain POST is blindly repeated: readback must resolve one exact owned job; absence or duplicates remain refusal. Retry uses the normal job retry route only after known failed/canceled/interrupted readback.
5. Cleanup always attempts birth-verified owned worker and API process-group drain, independently of GET/cancel failures. Closure uncertainty remains in the receipt. The cache flock is inherited by owned children, so controller death does not make a surviving child release its cache lease. No unrelated PID is killed. Completed managed-job readbacks, no unresolved intent, exact full file membership, unchanged read-only verified stat receipts and owned-service closure are all required for `contentVerified:true`.

The exact closure is **194 files, 377,576,304,518 bytes**. The four download tuples total **5,381,086,712 bytes**: predictor 275,415,648; text encoder shard 4,875,990,584; Rough 74,570,640; Clay 155,109,840. Full repository/revision/file/size/digest tuples are source constants and are independently tested. The complete host template retains both backend descriptors, all baseline 11,118 identities and both styles; this CPU route projects the 194 Mac payloads without deleting counterpart provenance or adding stress coverage.

## Truthful evidence boundary

CPU tests cover tiny files, mock product responses, injected failures and real owned CPU fake child processes. They do not run the product or download weights. Source tracing and the tested CPU setter policy do not prove runtime startup avoids GPU context creation. The first actual Mac2 CPU service run records before/startup/drain process and IOGPU/AGX observations; `actualStartupGpuContextVerified` stays false until those actual observations are independently qualified. `physicalAdmission`, `resourceQueueVerified` and `campaignReady` always remain false here, including a fully hashed cache.

The uploaded artifact contains metadata receipts and bounded process/device observations only. No checkpoint, binary, metallib, media, API token or credential is uploaded. Later source-freeze assembly must consume actual complete host payload receipts, immutable binary attestations and the required full resource guard authority. The S5 ordinary lifecycle/style catalog readback, E7 product assertions, native proof gates and actual terminal renders remain required; this route changes none of their acceptance conditions.

## Internal private staging and bounded resume

The internal route qualifies the actual Mac2 hardware UUID, runner5281/name,
MTrefry account/HOME and UID/effective UID, non-symlink owned path and HOME device.
The shared root remains exactly `/Volumes/Models/huggingface`; its device must
be different. Destination APFS support is still proved by an owned clone/private
mutation probe. The full source head and closed storage profile join the existing
immutable cache-owner marker. Shared blobs never become writable aliases.

Before staging, at each file/job boundary and during copy, fresh `statvfs`
available bytes must cover remaining unverified payload plus the declared
5,381,086,712-byte managed-download scratch and a 64MiB nonpayload workspace/log
growth budget. Unverified partial files receive no initial capacity discount.
Existing attested build/tools allocated bytes are recorded as already occupied;
PREPARE never compiles. Nonpayload workspace growth is checked during the phase.
The old disk snapshot is not admission evidence. The full private closure still
requires 377,576,304,518 logical bytes; cross-volume copy is not an APFS reflink.

Each copy uses at most 1MiB per read, with streaming digests and fsynced prefix
checkpoints under the same private cache lease. On timeout, the unpublished
`.partial` and its source/stat/offset/prefix-SHA journal remain. Resume requires
matching source identity and a fresh hash of the exact committed prefix before
appending. Unknown tails, changed source, tampered state or prefix, and foreign
aliases are retained and refused. Complete data is independently read back for
full destination SHA, made read-only, and published with atomic no-replace rename
before the snapshot link is created. Completed per-file journals are sealed.
Copies change neither shared file bytes nor metadata in the checked signature.
The copy buffers occupy at most two 1MiB blocks; no whole-file allocation occurs.

The original 480-second deadline includes internal host qualification and staging.
Incomplete phases exit nonzero; completed verified files retain the existing
stat/read-only reuse rule. No throughput is assumed and no phase is extended.
The four normal managed model/LoRA jobs, API/worker CPU startup observations,
owned service closure and later actual GPU-context qualification stay required.
CPU tests use tiny fixture bytes and mocked Mac2/volume metadata; they do not
copy actual checkpoints, qualify available space on Mac2, or launch the product.
