# Exact failed Mac2 CPU compiler preparation diagnostic

The complete diagnostic described below ran at immutable source698. The matched control ran at source351 as described in
[h3-cache-controlled-repro.md](h3-cache-controlled-repro.md). The current normal
API recovery build is described in [h3-cache-api-build-recovery.md](h3-cache-api-build-recovery.md); original external
299/69fa/698 evidence roots remain read-only.

This source changes only `build` execution of the registered owned-cache workflow
into a bounded diagnostic of run 38534168074, job 115752914601, source
299c99e8dbeb5ad37a9fa8ad3e7ba51e79f67328. The five dispatch inputs remain literal
and unchanged. Root publishes this isolated diagnostic source and dispatches
`mode=build`, its exact immutable source SHA, and a **fresh** external run directory
such as `/Volumes/Models/epic-25140-cache-diagnostic-<source-prefix>`.
`shared_cache_root` and `binary_attestation_path` are unused by this build diagnostic.
No dispatch has been executed as part of source validation.

The controller uses the same-repository builtin token to verify the exact live
current Mac2 job and the completed failed prior job. It rehashes both prior child
receipts, all four logs, the successful Rustup source/runtime/plan bindings, and
reads the exact proc-macro dylib emitted in the failed Cargo JSON. The failed
root `/Volumes/Models/epic-25140-owned-cache-299c99e8d` is never a writable Cargo
home, target, or output. Missing, changed, symlinked or still-active prior evidence
refuses the diagnostic. The original receipts/logs are source-owned fixture bytes.

Every probe runs under the existing CPU PID/birth/process-group supervisor,
with durable context written before launch, bounded log receipts and reaped,
drained closure. Metadata probes are bounded to 60 seconds; each tiny build to
600 seconds; the diagnostic has an original 1800-second deadline. Timeout,
controller loss, launch error or uncertain closure produces a refusal and a
nonzero controller exit. Darwin xattrs are enumerated and read losslessly through supervised `/usr/bin/xattr`
CLI calls, with exact binary SHA256 values and immutable stdout receipts; missing
CLI capability/read failure refuses rather than omitting hashes. Python `os` xattr
APIs are not required. The failed diagnostic69fa root is also excluded from fresh
output roots. A compiler's ordinary nonzero status is retained as
an observed diagnostic result, never a successful preparation.

The actual old owned rustc/cargo paths, byte hashes, `-Vv`, sysroot, command
resolution, current allowlisted compiler settings, configuration-file hashes,
rustc-driver libraries, dylib stat/xattr hashes, `file`, `otool` linkage/load
commands, codesign metadata and volume space are captured. No credentials or
whole configuration contents are exported. An inherited unbound compiler wrapper
is recorded and refused before execution. The old source head must still be
ordinary165. Tool bytes and the prior macro bytes are rechecked after probes.

Two controlled experiments follow:

1. The same owned rustc attempts to load the actual old emitted derive dylib into
   a tiny crate. This is **not** the unavailable original whole-product rustc argv.
2. An offline tiny crate exercises bytemuck1.25.0/derive1.10.2 and their exact four
   transitive lock packages (including syn2.0.119). Original checksum-verified
   `.crate` archives are read from the old private cache and safely expanded to
   a new private vendor directory. Cargo uses locked/offline release defaults,
   matching the original source's absence of a custom release profile, and
   records verbose compiler commands plus every Cargo JSON artifact. It builds
   once into an external Models target and once into an internal RUNNER_TEMP
   target. These are explicitly separate targets, not modifications to the
   failed target or changes to product dependencies/optimization.

Historical compiler tool byte hashes, inherited flags/config state and original
verbose rustc argv were not recorded by the failed Cargo receipt. Current hashes
cannot retroactively fill those gaps. The new supervisor fixes that failure-path
context omission for future preparations. The diagnostic receipt therefore keeps
`prepared:false`, `campaignReady:false`, `physicalAdmission:false`, and
`cause:unknown`. Actual host execution and any causal conclusion remain pending.
