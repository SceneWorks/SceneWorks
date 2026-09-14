import { describe, expect, it } from "vitest";

import { appendMaskStrokePoint } from "./maskStroke.js";

describe("appendMaskStrokePoint", () => {
  it("keeps the active point buffer and only copies the outer stroke list", () => {
    const points = [1, 2];
    const lines = [{ points, size: 64, erase: false }];

    const next = appendMaskStrokePoint(lines, { x: 3, y: 4 });

    expect(next).not.toBe(lines);
    expect(next[0]).toBe(lines[0]);
    expect(next[0].points).toBe(points);
    expect(points).toEqual([1, 2, 3, 4]);
  });
});
