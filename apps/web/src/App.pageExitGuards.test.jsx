import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { App } from "./App.jsx";
import { usePageLeaveGuard } from "./hooks/usePageLeaveGuard.js";
import { appConfirm } from "./appConfirm.jsx";
import { FakeEventSource, response, settle } from "./main.testSupport.jsx";
import { mountRoot, unmountRoot, click } from "./testUtils/dom.js";

// Exercise the shell's exits with a page that holds an unsaved document. The
// editor owns the actual dirty-document detection; the shell must honor its guard.
function DirtyEditor() {
  usePageLeaveGuard(() => appConfirm({ title: "Leave Image Editor?", message: "Discard unsaved edits?" }));
  return <div data-dirty-document>Unsaved document</div>;
}
vi.mock("./screens/ImageEditor.jsx", () => ({ ImageEditor: DirtyEditor }));
let root, container;
beforeEach(() => {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  ({ root, container } = mountRoot());
  localStorage.clear();
  window.innerWidth = 1440;
  FakeEventSource.instances = [];
  window.EventSource = FakeEventSource;
  vi.stubGlobal("fetch", vi.fn(async (url) => {
    const path = new URL(url).pathname;
    if (path.endsWith("/health")) return response({ status: "ok" });
    if (path.endsWith("/access")) return response({ authRequired: false });
    if (path.endsWith("/projects")) return response([{ id: "project", name: "Project" }]);
    if (path.endsWith("/ui-preferences")) return response({ simpleUi: false });
    if (path.endsWith("/jobs/events/ticket")) return response({ ticket: "ticket" });
    return response([]);
  }));
});
afterEach(() => { unmountRoot(root, container); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

it.each(["status", "simple", "collapsed-simple"])("honors a cancelled leave guard for the %s control", async (control) => {
  await act(async () => root.render(<App />));
  await settle();
  await click([...container.querySelectorAll(".nav-item")].find((button) => button.textContent === "Image Editor"));
  await settle();
  const documentBefore = container.querySelector("[data-dirty-document]");
  expect(documentBefore).not.toBeNull();
  if (control === "collapsed-simple") await click(container.querySelector('[aria-label="Collapse sidebar"]'));
  const exit = control === "status" ? container.querySelector(".status-summary")
    : control === "simple" ? container.querySelector(".sidebar-footer .su-switch button")
      : container.querySelector('[aria-label="Switch to Simple UI"]');
  await click(exit);
  expect(document.querySelector(".app-confirm-modal")).not.toBeNull();
  await click([...document.querySelectorAll(".app-confirm-modal button")].find((button) => button.textContent === "Cancel"));
  expect(container.querySelector("[data-dirty-document]")).toBe(documentBefore);
  expect(container.querySelector(".su-root")).toBeNull();
  // Accepting a subsequent exit releases the document as intended.
  await click(exit);
  await click(document.querySelector(".app-confirm-modal .primary-action"));
  expect(container.querySelector("[data-dirty-document]")).toBeNull();
});
