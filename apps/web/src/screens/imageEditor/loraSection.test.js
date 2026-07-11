import { describe, expect, it } from "vitest";

import { loraAddHint } from "./loraSection.js";

// The + Add note in the edit LoRA section (epic 10644 / sc-10653). The regression: at the
// cap, + Add used to disable in silence. loraAddHint is the single source for that note.
describe("loraAddHint", () => {
  const max = 5;

  it("is silent when a LoRA can still be added", () => {
    expect(loraAddHint({ selectedCount: 2, hasNext: true, max })).toBeNull();
  });

  it("is silent on an empty section — its own empty-state note speaks", () => {
    expect(loraAddHint({ selectedCount: 0, hasNext: false, max })).toBeNull();
    expect(loraAddHint({ selectedCount: 0, hasNext: true, max })).toBeNull();
  });

  // The bug this fixes: at the cap with LoRAs applied, say the limit rather than dim in silence.
  it("names the cap once it is reached", () => {
    expect(loraAddHint({ selectedCount: 5, hasNext: true, max })).toBe("Up to 5 LoRAs per edit.");
    expect(loraAddHint({ selectedCount: 6, hasNext: false, max })).toBe("Up to 5 LoRAs per edit.");
  });

  it("explains an exhausted compatible list under the cap", () => {
    expect(loraAddHint({ selectedCount: 2, hasNext: false, max })).toBe("No more compatible LoRAs to add.");
  });
});
