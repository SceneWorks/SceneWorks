import assert from "node:assert/strict";
import test from "node:test";

import { stripJsoncComments } from "./jsonc.mjs";

test("stripJsoncComments removes comments without changing string contents", () => {
  const source = `{
    // line comment
    "url": "https://example.test/a//b",
    "escaped": "quote: \\" // still a string",
    /* block
       comment */
    "value": 7
  }`;

  assert.deepEqual(JSON.parse(stripJsoncComments(source)), {
    url: "https://example.test/a//b",
    escaped: 'quote: " // still a string',
    value: 7,
  });
});
