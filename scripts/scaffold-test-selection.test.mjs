import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { HISTORICAL_TESTS, scaffoldTestFiles, scaffoldTestGroups } from "./scaffold-test-selection.mjs";
const command = JSON.parse(readFileSync("package.json", "utf8")).scripts.check;
test("portable and historical suites partition the entire existing npm check inventory", () => {
  const portable = scaffoldTestFiles(command, "portable");
  const history = scaffoldTestFiles(command, "history");
  assert.deepEqual(history.sort(), [...HISTORICAL_TESTS].sort());
  assert.equal(portable.filter((file) => history.includes(file)).length, 0);
  assert.equal(portable.length + history.length, command.match(/scripts\/[\w/.-]+\.test\.mjs/g).length);
});
test("inventory changes cannot silently omit external-source tests or shell commands", () => {
  for (const bad of [command.replace(HISTORICAL_TESTS[0], ""), `${command} && echo unreviewed`, command.replace("node --test", "node --test --test-name-pattern=fast"), `${command} ${HISTORICAL_TESTS[0]}`]) {
    assert.throws(() => scaffoldTestFiles(bad, "portable"));
  }
  assert.throws(() => scaffoldTestFiles(command, "unknown"));
});

test("the existing serial matrix-source mutation barrier survives selection", () => {
  const groups = scaffoldTestGroups(command, "portable");
  assert.equal(groups.length, 2);
  assert.deepEqual(groups[1], ["scripts/matrix-removed-sources.test.mjs"]);
  assert.equal(scaffoldTestGroups(command, "history").length, 1);
});
