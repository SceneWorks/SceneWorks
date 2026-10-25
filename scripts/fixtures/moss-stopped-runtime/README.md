These gzip files preserve the exact, unmodified runtime.json bytes retained from
sc-26752 stopped CUDA build runs R4 (40145054879), R5 (40152001538), and R6
(40155160639), attempt 1. They contain package metadata and original source/cache
identities, no model bytes or credentials. The production maintenance allowlist
pins their decompressed SHA256. Tests use these historical bytes without
relabelling them as the next candidate's runtime or execution evidence.

R10 `r10.json.gz` preserves the actual 71,286-byte CUDA runtime from artifact 12587034916, run 40163530825 attempt 1, source c17fd74650f2464427b30cea7669c7a284ea4a73. Raw SHA256: `546808654e2a0d44d26a5d594b9e7c20961b2b9d947e900effdefdf9f8c2d625`. Only its exact durable private HF leaf is eligible; target/binaries/logs/captures and parent directories are retained. Actual file metadata is observed before cleanup, not inferred from this runtime.
