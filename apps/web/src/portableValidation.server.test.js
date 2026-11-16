import { afterAll, beforeAll, afterEach, expect, it, vi } from "vitest";
import { apiFetch } from "./api.js";
import {
  MAX_VALIDATION_BATCH, setServerValidatorVersion, validatePortableBatch,
  validatePortableRequest, validatePortableRequestNow,
} from "./portableValidation.js";

vi.mock("./api.js", () => ({ apiFetch: vi.fn(), isAbortError: (error) => error?.name === "AbortError" }));

let testServer;
beforeAll(() => { testServer = globalThis.__sceneworksOptionsTestServer; globalThis.__sceneworksOptionsTestServer = null; });
afterAll(() => { globalThis.__sceneworksOptionsTestServer = testServer; });

afterEach(() => {
  vi.mocked(apiFetch).mockReset();
  setServerValidatorVersion(null);
});

it("sends exact authored requests and token to the authoritative server", async () => {
  const request = { kind: "image", payload: { model: "flux_dev", prompt: "a lighthouse", width: 1024 }, token: "secret" };
  vi.mocked(apiFetch).mockResolvedValue({ validatorSchemaVersion: 1, results: [{ ok: true, resolved: request.payload }] });
  expect(await validatePortableRequest(request)).toMatchObject({ state: "valid", resolved: request.payload, metadataSource: "server" });
  expect(apiFetch).toHaveBeenCalledWith("/api/v1/options/validate", "secret", expect.objectContaining({
    method: "POST", body: JSON.stringify({ validatorSchemaVersion: 1, kind: "image", requests: [request.payload] }),
  }));
  expect(validatePortableRequestNow(request)).toMatchObject({ state: "server-required" });
  expect(validatePortableRequestNow({ ...request, payload: { ...request.payload, width: 1001 } }))
    .toMatchObject({ state: "server-required" });
});

it("preserves typed refusals and refuses incomplete or mismatched responses", async () => {
  const payload = { model: "flux_dev", prompt: "x", width: 1001 };
  const refusal = { code: "invalid_generation_option", field: "width", detail: "width must align" };
  vi.mocked(apiFetch).mockResolvedValueOnce({ validatorSchemaVersion: 1, results: [{ ok: false, refusal }] });
  expect(await validatePortableRequest({ kind: "image", payload })).toEqual({
    state: "invalid", refusal, metadataSource: "server", liveChecks: "server",
  });
  expect(validatePortableRequestNow({ kind: "image", payload })).toMatchObject({ state: "server-required" });
  vi.mocked(apiFetch).mockResolvedValueOnce({ validatorSchemaVersion: 2, results: [{ ok: true, resolved: payload }] });
  expect(await validatePortableRequest({ kind: "image", payload })).toMatchObject({
    state: "server-required", reason: "server-validator-version-mismatch",
  });
  setServerValidatorVersion(2);
  expect(validatePortableRequestNow({ kind: "image", payload })).toMatchObject({ state: "server-required" });
});

it("bounds batches to 32 and keeps offline feedback server-owned", async () => {
  vi.mocked(apiFetch).mockRejectedValue(new TypeError("offline"));
  const requests = Array.from({ length: MAX_VALIDATION_BATCH }, (_, index) => ({ prompt: `p${index}` }));
  expect(await validatePortableBatch({ kind: "image", requests })).toHaveLength(32);
  expect((await validatePortableBatch({ kind: "image", requests }))[0]).toMatchObject({
    state: "server-required", reason: "server-unavailable",
  });
  expect((await validatePortableBatch({ kind: "image", requests: [...requests, {}] }))[0])
    .toMatchObject({ state: "server-required", reason: "invalid-batch-size" });
  expect(apiFetch).toHaveBeenCalledTimes(2);
});
