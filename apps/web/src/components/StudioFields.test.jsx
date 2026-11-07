import React, { act } from "react";
import { afterEach, beforeEach, expect, it } from "vitest";
import { mountRoot, setSelect, unmountRoot } from "../testUtils/dom.js";
import { useStudioAdvancedSettings } from "../hooks/useStudioAdvancedSettings.js";
import { CharacterLookFields, GpuSelect } from "./StudioFields.jsx";
let root, container, settings;
beforeEach(() => { globalThis.IS_REACT_ACT_ENVIRONMENT = true; ({ root, container } = mountRoot()); });
afterEach(() => unmountRoot(root, container));

it("keeps lane defaults explicit, preserves sticky zero values and leaves mounted authoring state intact", async () => {
  function Harness({ saved = {}, defaults }) {
    settings = useStudioAdvancedSettings(saved, defaults);
    return <button onClick={settings.toggleAdvanced}>Advanced</button>;
  }
  await act(async () => root.render(<Harness key="ordinary" />));
  expect(settings).toMatchObject({ schedulerShift: null, stepsOverride: "", guidanceOverride: "", advancedOpen: false });
  await act(async () => root.render(<Harness key="moss" defaults={{ schedulerShift: 5, steps: 30, guidanceScale: 4 }} />));
  expect(settings).toMatchObject({ schedulerShift: 5, stepsOverride: 30, guidanceOverride: 4 });
  await act(async () => root.render(<Harness key="video" defaults={{ schedulerShift: 3 }}
    saved={{ seed: 0, steps: 0, guidanceScale: 0, schedulerShift: 0 }} />));
  expect(settings).toMatchObject({ seed: 0, stepsOverride: 0, guidanceOverride: 0, schedulerShift: 0 });
  await act(async () => { settings.setSeed("12"); container.querySelector("button").click(); });
  await act(async () => root.render(<Harness key="video" defaults={{ schedulerShift: 3 }} saved={{ seed: 99 }} />));
  expect(settings.seed).toBe("12");
  expect(settings.advancedOpen).toBe(true);
});

it("reads current character looks and event owners while preserving each caller's placeholder and GPU choices", async () => {
  const calls = [];
  const characters = [{ id: "a", name: "A", looks: [{ id: "a-look", name: "A look" }] },
    { id: "b", name: "B", looks: [{ id: "b-look", name: "B look" }] }];
  function Harness({ owner, placeholder }) {
    const [characterId, setCharacterId] = React.useState("a");
    const [lookId, setLookId] = React.useState("");
    return <>
      <CharacterLookFields characters={characters} characterId={characterId} lookId={lookId}
        characterPlaceholder={placeholder}
        onCharacterChange={(id) => { calls.push([owner, "character", id]); setCharacterId(id); }}
        onLookChange={(id) => { calls.push([owner, "look", id]); setLookId(id); }} />
      <GpuSelect options={["auto", "worker-gpu"]} value="auto" onChange={(id) => calls.push([owner, "gpu", id])} />
    </>;
  }
  await act(async () => root.render(<Harness owner="old" placeholder="Select character" />));
  expect(container.querySelectorAll("select")[0].options[0].text).toBe("Select character");
  await act(async () => root.render(<Harness owner="current" placeholder="No character" />));
  await act(async () => setSelect(container.querySelectorAll("select")[0], "b"));
  const selects = container.querySelectorAll("select");
  expect([...selects[1].options].map((option) => option.value)).toEqual(["", "b-look"]);
  expect(selects[2].options[0].text).toBe("Auto");
  await act(async () => { setSelect(selects[1], "b-look"); setSelect(selects[2], "worker-gpu"); });
  expect(calls).toEqual([["current", "character", "b"], ["current", "look", "b-look"], ["current", "gpu", "worker-gpu"]]);
});
