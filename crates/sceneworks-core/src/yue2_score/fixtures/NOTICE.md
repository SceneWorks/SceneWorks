# YuE2 released editing examples (test fixtures)

`score.abc`, `score-jazz.abc`, `melody.abc` and `song.json` are copied byte-for-byte from the
YuE repository's `examples/` directory at commit
`92a73cc7652fcc1f937855e4b765e0a0edd7ff2e`
(https://github.com/multimodal-art-projection/YuE/tree/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e/examples).

That source tree is licensed under the Apache License, Version 2.0
(https://github.com/multimodal-art-projection/YuE/blob/92a73cc7652fcc1f937855e4b765e0a0edd7ff2e/LICENSE).
Copyright (c) 2026 the YuE2 authors. The files are unmodified; they are used only as test inputs
for the native score-editing port in `crates/sceneworks-core/src/yue2_score/`.

`score-jazz.abc` is upstream's released harmony-only edit of `score.abc` (seventh-chord
reharmonization with every melody note, duration, bar, section and tempo unchanged).
