# sc-26691 H3 terminal campaign — dispatch kit

The request payloads and dispatch script the reduced terminal campaign (epic sc-26679, story
sc-26691) was run with. The full procedure and evidence rules are in
[`../sc-26691-h3-terminal.md`](../sc-26691-h3-terminal.md); the current state and open findings are
in the closeout handoff on the epic in Shortcut.

## Dispatch

```bash
# Both lanes, paired (each lane waits up to 6 h for the peer's anchor artifact).
docs/testing/sc-26691-campaign/dispatch.sh --ack "$(git rev-parse origin/feature/sc-26679-h3-finetune-import)"

# One lane (the peer must be dispatched at the SAME SceneWorks SHA to pair).
BACKENDS=cuda docs/testing/sc-26691-campaign/dispatch.sh --ack <sceneworks-sha>
```

- `--ack` records the operator's scoped MiniMax-H3 Community License acknowledgment (sc-26691
  activity 27075) and must name the branch's current head; the script refuses a moved head.
- `inference_sha` must be reachable from inference `origin/main` — the bootstrap refuses an interim
  feature pin. It is `232f3f24` (inference main, SceneWorks/inference#1367).
- Requires `bash`, `jq` and an authenticated `gh` (on Windows: Git Bash with `jq` on `PATH`).

## Knobs

- `bootstrap.operator.width` / `height` (1088x544) set the canvas of the turbo/ref cases;
  `timeoutSeconds` (7200) is the per-case bound. See the epic handoff for why these are under review.
- `expected_runner`: `cuda-windows` / `cuda-windows-2` (the job is pinned by the runner-name label),
  `nax-macos` for MLX.
- CUDA uses `gpuId`/`requestedGpu` `0`; MLX uses `mlx`.
