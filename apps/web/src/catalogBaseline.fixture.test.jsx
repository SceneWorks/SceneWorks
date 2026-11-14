import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
vi.mock("./api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import old from "./testUtils/catalogBaseline/old-catalog.json";
import { currentCandidate } from "./testUtils/catalogBaseline/candidate.js";
import { controls, observe } from "./testUtils/catalogBaseline/harness.jsx";

const row = { platform: "linux", studio: "vector", model: "starvector_1b", mode: "create_from_prompt" };
let directory;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.stubEnv("CATALOG_CANDIDATE", "");
  vi.spyOn(Date, "now").mockReturnValue(1794000000000);
  vi.spyOn(Math, "random").mockReturnValue(0.25);
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Catalog fixture must not fetch or execute a model"); }));
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  if (directory) rmSync(directory, { recursive: true });
  directory = undefined;
});

it("the ordinary CI manifest path submits the same vector workflow as the captured native inputs", async () => {
  const ordinary = await observe(currentCandidate(), row);
  const native = await observe(old.catalog, row);
  expect(ordinary.submission.calls).not.toHaveLength(0);
  expect(ordinary).toEqual(native);
});

it("explicit native producer order still changes the actual submitted raster choice", async () => {
  const native = await observe(old.catalog, row);
  const reordered = { ...old.catalog, models: [...old.catalog.models].reverse() };
  directory = mkdtempSync(join(tmpdir(), "catalog-native-order-"));
  const input = join(directory, "native.json");
  writeFileSync(input, JSON.stringify(reordered));
  vi.stubEnv("CATALOG_CANDIDATE", input);
  const candidate = currentCandidate();
  expect(candidate.models.map(({ id }) => id)).toEqual(reordered.models.map(({ id }) => id));
  const actual = await observe(candidate, row);
  expect(actual.submission.calls[0].payload.rasterModel).not.toBe(native.submission.calls[0].payload.rasterModel);
});

it("control observations round-trip through JSON without dropping option order or state", () => {
  const container = document.createElement("div");
  container.innerHTML = '<button>Generate</button><label>Sampler<select><option value="b">Second</option><option value="a" disabled>First</option></select></label>';
  const observed = controls(container, "fixture-settings");
  expect(Object.hasOwn(observed[0], "options")).toBe(false);
  expect(observed).toStrictEqual(JSON.parse(JSON.stringify(observed)));
  expect(observed[1].options).toEqual([
    { id: "b", label: "Second", disabled: false },
    { id: "a", label: "First", disabled: true },
  ]);
  expect(observed[1].value).toBe("b");
});
