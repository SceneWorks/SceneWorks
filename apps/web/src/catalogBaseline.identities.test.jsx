import { readFileSync } from "node:fs";
import { beforeAll, expect, it, vi } from "vitest";
vi.mock("./api.js", async (original) => ({ ...await original(), apiFetch: vi.fn(async () => ({})) }));
import identities from "./testUtils/catalogBaseline/controlIdentities.json";
import { frozenControlInventory } from "./testUtils/catalogBaseline/frozenControlInventory.js";
import old from "./testUtils/catalogBaseline/old-catalog.json";
import { currentCandidate } from "./testUtils/catalogBaseline/candidate.js";
import { controlInventory, assertDeclaredEffects } from "./testUtils/catalogBaseline/controlInventory.js";
import { mount, click, change } from "./testUtils/catalogBaseline/harness.jsx";
const baseline = JSON.parse(readFileSync("src/testUtils/catalogBaseline/old-observations.json", "utf8"));
const allIds = (inventory) => [inventory.controls, ...inventory.menus.map((menu) => menu.controls)].map((controls) => controls.map(({ id, controlId }) => [id, controlId]));
beforeAll(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("Identity/effect tests must not execute a model"); }));
});
it("all 1,197 frozen surfaces have scoped identities independent of defaults and selected state", () => {
  expect(identities.sourceRevision).toBe(old.source.revision);
  expect(Object.keys(baseline.rows)).toHaveLength(1197);
  for (const [key, packed] of Object.entries(baseline.rows)) {
    const [platform, studio] = key.split("/");
    const observed = { controls: packed.controls.map((id) => baseline.controls[id]), menus: packed.menus };
    const inventory = controlInventory(observed, old.catalog.platforms[platform], { studio });
    expect(inventory, key).toEqual(frozenControlInventory(key));
    for (const ids of allIds(inventory)) {
      expect(ids.every(([id, controlId]) => id.length > 0 && controlId.length > 0), key).toBe(true);
      expect(new Set(ids.map(([, controlId]) => controlId)).size, key).toBe(ids.length);
    }
    const changed = structuredClone(observed);
    for (const control of [...changed.controls, ...changed.menus.flatMap((menu) => menu.controls)]) {
      if (!["button", "tab", "radio"].includes(control.role)) {
        if (["textbox", "text"].includes(control.role) && typeof control.value === "string" && control.value) control.label = control.label.replace(control.value, "Different default");
        for (const option of control.options ?? []) {
          control.label = control.label.replace(option.label, "Different choice");
          option.label = "Different choice";
        }
      }
      control.value = typeof control.value === "boolean" ? !control.value : "Different default";
      control.pressed = control.pressed === "true" ? "false" : "true";
      control.selected = control.selected === "true" ? "false" : "true";
    }
    expect(allIds(controlInventory(changed, old.catalog.platforms[platform], { studio })), key).toEqual(allIds(inventory));
  }
});
it("renaming a semantic ID cannot change the frozen expected identity", () => {
  const key = "linux/audio/yue_en_cot/music";
  const packed = baseline.rows[key];
  const observed = { controls: packed.controls.map((id) => baseline.controls[id]), menus: packed.menus };
  const expected = frozenControlInventory(key);
  const seed = identities.fields.Seed;
  try {
    identities.fields.Seed = "silentlyRenamedSeed";
    const mutated = controlInventory(observed, old.catalog.platforms.linux, { studio: "audio" });
    expect(mutated.controls.some((control) => control.id === "silentlyRenamedSeed")).toBe(true);
    expect(frozenControlInventory(key)).toEqual(expected);
    expect(() => expect(mutated).toEqual(expected)).toThrow();
  } finally {
    identities.fields.Seed = seed;
  }
});
it("simple picker identities survive changed model, resolution and voice defaults", () => {
  const controls = [
    { role: "button", label: "FLUX.1 [dev]" },
    { role: "button", label: "1:1 · 1024×1024" },
    { role: "button", label: "Alloy · american female" },
  ];
  const before = controlInventory({ controls, menus: [] }, {}, { studio: "simpleImage" });
  const after = controlInventory({ controls: [
    { ...controls[0], label: "FLUX.1 [schnell]" },
    { ...controls[1], label: "16:9 · 1280×720" },
    { ...controls[2], label: "Aoede · american female" },
  ], menus: [] }, {}, { studio: "simpleImage" });
  expect(before.controls.map(({ id }) => id)).toEqual(["model", "resolution", "voice"]);
  expect(allIds(after)).toEqual(allIds(before));
});
it("Chinese choices and similarly spelled genre actions retain distinct explicit identities", () => {
  const row = baseline.rows["linux/audio/yue_en_cot/music"];
  const controls = row.controls.map((id) => baseline.controls[id]);
  const inventory = controlInventory({ controls, menus: [] }, {}, { studio: "audio" });
  const idFor = (label) => inventory.controls[controls.findIndex((control) => control.label === label)].id;
  expect(idFor("拉丁")).not.toBe(idFor("民谣"));
  expect(idFor("hip-hop")).not.toBe(idFor("Hip Hop"));
});
it("unknown fields fail instead of inventing option identities", () => {
  expect(() => controlInventory({ controls: [{ role: "number", label: "Unregistered field" }], menus: [] }, {})).toThrow(/No declared control identity/);
});
const state = (mounted) => {
  const saved = mounted.getPersisted();
  return { usePid: saved.usePid, upscale: saved.upscaleEnabled, hiresFix: saved.hiresFixEnabled };
};
const checkbox = (container, label) => [...container.querySelectorAll('input[type="checkbox"]')].find((node) => node.closest("label")?.textContent.trim() === label);
for (const [id, other, label] of [["upscale", "hiresFix", "Upscale"], ["hiresFix", "upscale", "Hires.fix"]]) {
  it(`${id} clears a conflicting option through the actual handler`, async () => {
    const mounted = await mount(currentCandidate(), { platform: "linux", studio: "image", model: "sdxl", mode: "text_to_image" }, { upscaleEnabled: other === "upscale", hiresFixEnabled: other === "hiresFix" });
    try {
      const before = state(mounted);
      expect(before[other]).toBe(true);
      const node = checkbox(mounted.container, label);
      expect(node?.disabled).toBe(false);
      await click(node);
      const after = state(mounted);
      expect(after[id]).toBe(true);
      assertDeclaredEffects(id, "enabled", before, after);
      expect(() => assertDeclaredEffects(id, "enabled", before, { ...after, [other]: before[other] })).toThrow(`must clear ${other}`);
    } finally { await mounted.close(); }
  });
}
it("restored PiD plus upscale or Hires.fix clears PiD through the actual effect", async () => {
  for (const id of ["upscale", "hiresFix"]) {
    const before = { usePid: true, upscale: id === "upscale", hiresFix: id === "hiresFix" };
    const mounted = await mount(currentCandidate(), { platform: "linux", studio: "image", model: "sdxl", mode: "text_to_image" }, { usePid: true, upscaleEnabled: before.upscale, hiresFixEnabled: before.hiresFix });
    try {
      const after = state(mounted);
      expect(after[id]).toBe(true);
      assertDeclaredEffects(id, "enabled", before, after);
      expect(() => assertDeclaredEffects(id, "enabled", before, { ...after, usePid: true })).toThrow("must clear usePid");
    } finally { await mounted.close(); }
  }
});
it("non-native decoder clears restored PiD through the actual effect", async () => {
  // Installed alternate-decoder host state: the public picker uses this API
  // availability projection. No decoder or model code executes in jsdom.
  const catalog = currentCandidate();
  const model = catalog.models.find((entry) => entry.id === "krea_2_raw");
  model.decoders.byBackend.mlx = model.decoders.byBackend.mlx.map((option) => ({ ...option, available: true }));
  const mounted = await mount(catalog, { platform: "macos", studio: "image", model: "krea_2_raw", mode: "text_to_image" }, { usePid: true });
  try {
    const before = state(mounted);
    expect(before.usePid).toBe(true);
    const decoder = [...mounted.container.querySelectorAll("select")].find((node) => node.closest("label")?.textContent.startsWith("Decoder"));
    const alternative = [...decoder.options].find((option) => option.value !== "native" && !option.disabled);
    expect(alternative).toBeTruthy();
    await change(decoder, alternative.value);
    const after = state(mounted);
    assertDeclaredEffects("decoder", "nonNative", before, after);
    expect(() => assertDeclaredEffects("decoder", "nonNative", before, { ...after, usePid: true })).toThrow("must clear usePid");
  } finally { await mounted.close(); }
});
