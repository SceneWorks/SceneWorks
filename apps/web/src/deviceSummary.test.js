import { describe, expect, it } from "vitest";

import { describeDevice } from "./deviceSummary.js";

describe("describeDevice", () => {
  it("labels binary worker memory as GiB", () => {
    expect(
      describeDevice([
        {
          gpuId: "gpu-0",
          gpuName: "Fixture GPU",
          utilization: { memoryTotalMb: 24_576 },
        },
      ]),
    ).toEqual({ engine: "Candle · discrete GPU", gpu: "Fixture GPU · 24 GiB" });
  });
});
