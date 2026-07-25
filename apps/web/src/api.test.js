import { afterEach, describe, expect, it, vi } from "vitest";

import { apiFetch } from "./api.js";

describe("apiFetch response handling", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("stringifies structured API details instead of coercing them to [object Object]", async () => {
    global.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ detail: { field: "prompt", reason: "required" } }), {
        status: 422,
        headers: { "Content-Type": "application/json" },
      }),
    );

    await expect(apiFetch("/structured-error", "")).rejects.toThrow(
      '{"field":"prompt","reason":"required"}',
    );
  });

  it.each([
    ["204 response", new Response(null, { status: 204 })],
    ["empty successful response", new Response("", { status: 200 })],
  ])("returns null for an empty %s", async (_label, response) => {
    global.fetch = vi.fn(async () => response);

    await expect(apiFetch("/empty", "")).resolves.toBeNull();
  });
});
