import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";

// Inspect shell bodies, including both block and inline run scalars. Expressions
// in env/with/if are evaluated by Actions without becoming shell program text.
test("workflow run bodies never interpolate dispatch inputs", () => {
  const violations = [];
  for (const name of readdirSync(".github/workflows").filter((file) => /\.ya?ml$/.test(file))) {
    const lines = readFileSync(`.github/workflows/${name}`, "utf8").split("\n");
    let runIndent = null;
    for (const [index, line] of lines.entries()) {
      const indent = line.match(/^ */)[0].length;
      if (runIndent !== null && line.trim() && indent <= runIndent) runIndent = null;
      const run = line.match(/^(\s*)(?:-\s+)?run:\s*(.*)$/);
      if (run) runIndent = run[1].length;
      if ((run || runIndent !== null) && /\$\{\{\s*(?:inputs\.|github\.event\.inputs\.)/.test(line)) {
        violations.push(`${name}:${index + 1}`);
      }
    }
  }
  assert.deepEqual(violations, []);
});

test("runner input is data even when it contains shell syntax", () => {
  const workflow = readFileSync(".github/workflows/flux3-quant-matrix.yml", "utf8");
  const run = workflow.match(/^\s*run: '(\[\[.*)'$/m)?.[1];
  assert.ok(run);
  for (const input of ["expected-runner", 'bad"; touch /tmp/dispatch-injected; #']) {
    const result = spawnSync("bash", ["-c", run], { env: { ...process.env, RUNNER_NAME: input, DISPATCH_RUNNER: input }, encoding: "utf8" });
    assert.equal(result.status, 0, result.stderr);
  }
  assert.ok(workflow.includes("wanted=\"$DISPATCH_CELLS\""));
  assert.ok(workflow.includes("$env:RUNNER_NAME -ne $env:DISPATCH_RUNNER"));
});
