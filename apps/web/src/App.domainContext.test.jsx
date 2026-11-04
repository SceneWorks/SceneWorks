import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { useAppStatic } from "./context/AppContext.js";
import { App } from "./App.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
const captures = vi.hoisted(() => []);
vi.mock("./screens/LibraryScreen.jsx", () => ({ LibraryScreen: () => {
  captures.push(useAppStatic());
  return <div>Library probe</div>;
} }));

it("keeps the complete composed static context stable across a real App SSE update", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  FakeEventSource.instances = [];
  window.EventSource = FakeEventSource;
  vi.stubGlobal("fetch", vi.fn(async (url) => {
    const path = new URL(url).pathname;
    if (path.endsWith("/health")) return response({ status: "ok" });
    if (path.endsWith("/access")) return response({ authRequired: false });
    if (path.endsWith("/ui-preferences")) return response({ activeView: "Library", simpleUiDefault: false });
    if (path.endsWith("/capabilities/mac")) return response({ macGatingActive: false });
    if (path.endsWith("/projects")) return response([{ id: "p", name: "Project" }]);
    if (path.endsWith("/jobs/events/ticket")) return response({ ticket: "ticket" });
    return response([]);
  }));
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  try {
    await act(async () => root.render(<App />));
    await settle();
    const first = captures.at(-1);
    expect(first.activeProject.id).toBe("p");
    for (const key of ["createImageJob", "refinePrompt", "createTrainingJob", "saveTimeline", "deleteAsset", "createCharacter", "changeTheme"])
      expect(typeof first[key], key).toBe("function");
    const source = FakeEventSource.instances.at(-1);
    await act(async () => source.listeners["job.updated"]({ data: JSON.stringify({ id: "new", type: "image_generate", projectId: "p", status: "running" }) }));
    expect(captures.at(-1)).toBe(first);
  } finally {
    act(() => root.unmount());
    container.remove();
    vi.unstubAllGlobals();
  }
});
