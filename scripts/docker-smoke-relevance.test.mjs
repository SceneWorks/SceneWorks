import assert from "node:assert/strict";
import test from "node:test";

import { dockerSmokeDecision } from "./docker-smoke-relevance.mjs";

test("root Cargo.toml is Docker relevant", () => {
  assert.equal(dockerSmokeDecision(1, ["Cargo.toml"]).run, true);
});

test("an exhaustively listed unrelated pull request skips the Docker smoke", () => {
  assert.equal(
    dockerSmokeDecision(2, ["README.md", "apps/web/src/App.jsx"]).run,
    false,
  );
});

test("a paginated result mismatch runs conservatively instead of skipping", () => {
  const firstThreeThousand = Array.from(
    { length: 3000 },
    (_, index) => `docs/change-${index}.md`,
  );
  const decision = dockerSmokeDecision(3001, firstThreeThousand);
  assert.equal(decision.run, true);
  assert.match(decision.reason, /pagination returned 3000/);
});
