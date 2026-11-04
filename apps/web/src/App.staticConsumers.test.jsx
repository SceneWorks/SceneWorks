import React, { act, Profiler } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";
import { AppLiveContext, AppStaticContext } from "./context/AppContext.js";
import { ControlOverlayPicker } from "./components/ControlOverlayPicker.jsx";
import { TrainedIrisModelSelect } from "./components/IrisRestoreControls.jsx";
import { SimpleAssets } from "./simple/SimpleAssets.jsx";
import { SimpleLoraSheet } from "./simple/SimpleLoraSheet.jsx";
import { SimplePreview } from "./simple/SimplePreview.jsx";
import { SimpleUiContext } from "./simple/SimpleUiContext.js";

it("keeps actual static-only consumers off live ticks while catalog updates remain reactive", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const counts = new Map();
  const noop = () => {};
  const cases = [
    ["overlay", <ControlOverlayPicker key="overlay" baseModel={null} />],
    ["preview", <SimplePreview key="preview" asset={{ id: "a", type: "image", status: {} }} onClose={noop} onUseAsReference={noop} onToggleFavorite={noop} onDelete={noop} />],
    ["loras", <SimpleLoraSheet key="loras" selectedModel={null} onAdd={noop} />],
    ["assets", <SimpleAssets key="assets" />],
    ["iris", <TrainedIrisModelSelect key="iris" task="depth" value="" onChange={noop} releaseLabel="Release" />],
  ];
  const children = <SimpleUiContext.Provider value={{ breakpoint: "desktop", toast: noop, openPreview: noop }}>
    {cases.map(([id, child]) => <Profiler key={id} id={id} onRender={() => counts.set(id, (counts.get(id) ?? 0) + 1)}>{child}</Profiler>)}
  </SimpleUiContext.Provider>;
  const catalog = { token: "", activeProject: { id: "p" }, assets: [], assetsReady: true, audioModels: [], loras: [], models: [] };
  const render = (staticValue, live) => act(async () => root.render(
    <AppStaticContext.Provider value={staticValue}><AppLiveContext.Provider value={live}>{children}</AppLiveContext.Provider></AppStaticContext.Provider>,
  ));
  try {
    await render(catalog, { jobs: [] });
    counts.clear();
    await render(catalog, { jobs: [{ id: "tick", status: "running" }], workersById: new Map() });
    expect([...counts.entries()]).toEqual([]);
    await render({ ...catalog, loras: [{ id: "new-lora" }], models: [{ id: "new-model" }] }, { jobs: [] });
    for (const [id] of cases) expect(counts.get(id), id).toBeGreaterThan(0);
  } finally {
    act(() => root.unmount());
    container.remove();
  }
});
