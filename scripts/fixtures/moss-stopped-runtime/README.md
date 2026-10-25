These gzip files preserve the exact, unmodified runtime.json bytes retained from
sc-26752 stopped CUDA build runs R4 (40145054879), R5 (40152001538), and R6
(40155160639), attempt 1. They contain package metadata and original source/cache
identities, no model bytes or credentials. The production maintenance allowlist
pins their decompressed SHA256. Tests use these historical bytes without
relabelling them as the next candidate's runtime or execution evidence.
