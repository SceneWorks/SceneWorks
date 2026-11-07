import { describe, expect, it, vi } from "vitest";
const { apiFetchMock } = vi.hoisted(() => ({ apiFetchMock: vi.fn() }));
vi.mock("../api.js", () => ({ apiFetch: apiFetchMock }));
import { timelineRequest } from "./timelines.js";

describe("timeline API", () => {
  it("encodes project and timeline identifiers and forwards options", () => {
    const signal = new AbortController().signal;
    timelineRequest("p/1", "t/1", "token", { signal }, "/exports");
    expect(apiFetchMock).toHaveBeenCalledWith("/api/v1/projects/p%2F1/timelines/t%2F1/exports", "token", { signal });
  });
});
