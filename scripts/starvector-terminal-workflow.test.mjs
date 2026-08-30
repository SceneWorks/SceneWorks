import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const workflow = await readFile(".github/workflows/starvector-terminal.yml", "utf8");

test("terminal workflow is dispatch-only, serial, and seals raw evidence", () => {
  assert.match(workflow, /^\s+workflow_dispatch:/m);
  assert.doesNotMatch(workflow, /^\s+(push|pull_request|schedule):/m);
  for (const edge of ["needs: mlx-1b", "needs: mlx-8b", "needs: cuda-1b"]) assert.match(workflow, new RegExp(edge));
  assert.match(workflow, /needs: \[mlx-1b, mlx-8b, cuda-1b, cuda-8b\]/);
  assert.match(workflow, /starvector-terminal-producer\.mjs run/g);
  assert.match(workflow, /starvector-terminal-producer\.mjs seal/);
  assert.match(workflow, /STARVECTOR_TERMINAL_LEASE_ROOT: \/var\/lib\/sceneworks-terminal\/terminal-leases/);
  assert.match(workflow, /STARVECTOR_TERMINAL_LEASE_ROOT: C:\\\\ProgramData\\\\SceneWorks\\\\terminal-leases/);
  assert.match(workflow, /scripts\/starvector-terminal-route\.mjs/);
  assert.doesNotMatch(workflow, /RUNNER_TEMP[^\n]*\.lease/);
  assert.match(workflow, /Upload combined evidence even on failure/);
});

test("terminal workflow has no install or model download step", () => {
  assert.doesNotMatch(workflow, /(?:pip|npm|cargo)\s+install|huggingface-cli|curl .*models|wget .*models/i);
  assert.match(workflow, /STARVECTOR_TERMINAL_WEIGHTS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_METRICS_ROOT/);
  assert.match(workflow, /STARVECTOR_TERMINAL_NO_JOB_DOWNLOADS: "1"/);
  assert.match(workflow, /cross-repository lease/);
  assert.match(workflow, /starvector-terminal-metrics-environment-v1\.json/);
});
