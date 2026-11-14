import { describe, it, expect, beforeEach, vi } from "vitest";
import corpus from "../../../crates/sceneworks-options/tests/fixtures/historical-builders.json";
import { upgradeCarrier } from "./requestUpgrade.js";
import { loadStudioSettings, seedStudioSettingsFromServer } from "./hooks/useStudioSettings.js";
import { readSimpleStudioSnapshot, writeSimpleStudioSnapshot } from "./simpleStudioStore.js";
import { imageRecipeRequest } from "./regenerateImage.js";
vi.mock("./uiPreferences.js", () => ({persistNavigationPreferences:vi.fn(async()=>{})}));
beforeEach(()=>localStorage.clear());
describe("historical option carriers",()=>{
  for (const row of corpus.rows) it(`preserves ${row.name}`,()=>{
    const original=JSON.stringify(row.value);
    expect(upgradeCarrier(row.value,row.carrier)).toEqual(row.value);
    expect(JSON.stringify(row.value)).toBe(original);
  });
  it("refuses unknown versions without overwriting advanced or Simple originals",()=>{
    const advanced=JSON.stringify({optionsSchemaVersion:9,prompt:"keep me"});
    localStorage.setItem("sceneworks-studio-image-p",advanced);
    expect(()=>loadStudioSettings("image","p")).toThrow(/version 9/);
    expect(()=>seedStudioSettingsFromServer({p:{image:{prompt:"replacement"}}})).toThrow(/version 9/);
    expect(localStorage.getItem("sceneworks-studio-image-p")).toBe(advanced);
    const simple=JSON.stringify({p:{image:{optionsSchemaVersion:9,prompt:"keep me"}}});
    localStorage.setItem("sceneworks-simple-studio",simple);
    expect(()=>readSimpleStudioSnapshot("p")).toThrow(/version 9/);
    expect(()=>writeSimpleStudioSnapshot("p",{image:{prompt:"replacement"}})).toThrow(/version 9/);
    expect(localStorage.getItem("sceneworks-simple-studio")).toBe(simple);
  });
  it("normalizes each settings source before authoritative server hydration",()=>{
    localStorage.setItem("sceneworks-studio-image-p",JSON.stringify({timestepShift:3}));
    seedStudioSettingsFromServer({p:{image:{timestepShift:4}}});
    expect(loadStudioSettings("image","p")).toEqual({timestepShift:4,schedulerShift:4});
  });
  it("does not overwrite a conflicting alias and preserves omitted defaults",()=>{
    const input={advanced:{schedulerShift:3,timestepShift:4},seed:null};
    expect(()=>upgradeCarrier(input)).toThrow(/different authored values/);
    expect(input.advanced.timestepShift).toBe(4);
    expect(upgradeCarrier({advanced:{timestepShift:"3"}})).toEqual({advanced:{timestepShift:"3",schedulerShift:"3"}});
    expect(upgradeCarrier({})).toEqual({});
  });
  it("accepts equivalent numeric alias spellings and retains each authored value",()=>{
    for (const input of [
      {advanced:{schedulerShift:3,timestepShift:"3"}},
      {advanced:{schedulerShift:"3.0",timestepShift:3}},
    ]) expect(upgradeCarrier(input)).toEqual(input);
    expect(()=>upgradeCarrier({advanced:{schedulerShift:3,timestepShift:"4"}}))
      .toThrow(/different authored values/);
  });
  it("regenerates with fresh seed policy and removes prior machine decisions",()=>{
    const recipe={model:"z_image_turbo",prompt:"fox",seed:42,normalizedSettings:{count:1},rawAdapterSettings:{memoryStrategy:{rung:2},memoryEvidenceRevision:3,moss_image_execution_report:{provider:"old"},steps:9}};
    const request=imageRecipeRequest({recipe});
    expect(request.seed).toBeNull();
    expect(request.advanced).toEqual({steps:9});
    expect(recipe.rawAdapterSettings.memoryStrategy).toEqual({rung:2});
  });
});
