import assert from "node:assert/strict";
import test from "node:test";
import { validateUpstreamInputs } from "./starvector-terminal-upstream.mjs";
test("both upstream models validate entirely offline before caller claims an attempt", async () => {
  const calls = [];
  const reports = await validateUpstreamInputs({ sceneWorksRoot: "/repo", python: "/oracle/python", upstreamRoot: "/source", weightsRoot: "/weights", assetsRoot: "/assets", componentsRoot: "/components", sanitizer: "/sanitize" }, "/output", async (command, args, options) => {
    calls.push({ command, args, options }); return { stdout: JSON.stringify({ tier: args.at(-1), status: "validated" }) };
  });
  assert.deepEqual(reports.map(x => x.tier), ["1b", "8b"]);
  for (const {command,args,options} of calls) { assert.equal(command, "/oracle/python"); assert.equal(args[1], "validate"); assert.equal(options.env.HF_HUB_OFFLINE, "1"); assert.equal(options.env.TRANSFORMERS_OFFLINE, "1"); assert.ok(args.includes("--components-root")); }
});

test("one upstream job precedes all four native tuples and every tuple consumes its artifact", async () => {
  const { readFile } = await import("node:fs/promises");
  const workflow = await readFile(".github/workflows/starvector-terminal.yml", "utf8");
  assert.equal((workflow.match(/starvector-terminal-upstream\.mjs run/g) ?? []).length, 1);
  assert.match(workflow, /mlx-1b:\n    needs: upstream-reference/);
  assert.equal((workflow.match(/Download the shared upstream reference from this workflow attempt/g) ?? []).length, 4);
});
