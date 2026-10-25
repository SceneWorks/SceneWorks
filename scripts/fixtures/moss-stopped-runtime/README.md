These gzip files preserve the exact, unmodified runtime.json bytes retained from
sc-26752 stopped CUDA build runs R4 (40145054879), R5 (40152001538), and R6
(40155160639), attempt 1. They contain package metadata and original source/cache
identities, no model bytes or credentials. The production maintenance allowlist
pins their decompressed SHA256. Tests use these historical bytes without
relabelling them as the next candidate's runtime or execution evidence.

R10 `r10.json.gz` preserves the actual 71,286-byte CUDA runtime from artifact 12587034916, run 40163530825 attempt 1, source c17fd74650f2464427b30cea7669c7a284ea4a73. Raw SHA256: `546808654e2a0d44d26a5d594b9e7c20961b2b9d947e900effdefdf9f8c2d625`. Only its exact durable private HF leaf is eligible; target/binaries/logs/captures and parent directories are retained. Actual file metadata is observed before cleanup, not inferred from this runtime.

R12 `r12.json.gz` preserves the actual 71,286-byte CUDA runtime from run 40167292776 attempt 1, source e6475176bf1cf07a46530adbe659aa0d6492c71a. Raw SHA256: `47a25951ea0e1d6fd076802c2ca68421e36d90a296fa1813fc344e6d0a461355`. Only its fixed private HF leaf is eligible; metadata and all 43 package hashes must be observed before cleanup. Optional original-preserving target copy requires the independently retained binary SHA `07cb4ef31caba2f875d6750c461715f880c79288201af88af11997e31809d69d`, absence and existing capacity proof.

R13 `r13.json.gz` preserves the actual 71,262-byte CUDA runtime from run 40169818205 attempt 1, source ba6bdd824823f01f8c4dc435f9fad50f45f97893, on the primary `actions-runner` workspace (not runner-2). Raw SHA256: `76554eb462c580cd6afa7b5102aaeaa434de85049fe1b6701fe7ba2347f6d61f`. Only its exact private HF leaf is eligible; all 43 hash/completeness and runtime/stopped/source/path guards remain required. Independently hashed retained binary SHA: `09088a12b26968177adf5a6ead76b088be2f791e5756224497db4fb7d20cf8fd` (624,182,784 bytes).

R14 `r14.json.gz` retains actual run 40174802721 attempt 1 runtime 71,286B SHA256 `4082f9c28ae31aee7dda8e850d04b197d7741a01edbc3856aa53ddac6ba8a532`, source 12e1b364360b43b28e864879be4c68bb28a696bc on runner-2. Optional cache binary 624,182,784B SHA256 `ab36062090356912f9131654fa5c3ec87ed808f5e0fc0cbd32e388773edb75c2`; exact HF-only scope, all 43 metadata/hash completeness, stopped/source/path guards and original-preserving capacity-safe copy remain required.

R15 `r15.json.gz` binds run 40181253824 attempt 1, source a0692191b09813fab859659bf382bd7c91a60140, raw runtime 71,286B SHA256 `a4c17012e7143f8775809decbd72216a7c3114209bf91c3f7f0d9608f0585cc4`. Only its fixed stopped HF leaf is added; repeated metadata/hash, latest-attempt/all-jobs-stopped and exact ownership/confinement guards remain. Its original target is neither removed nor selected as a new cache source.
