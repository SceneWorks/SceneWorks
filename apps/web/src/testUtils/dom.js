// Shared jsdom test helpers for the React studio/component suites (sc-8937).
//
// These were re-declared near-identically across a dozen *.test.jsx files. The
// dispatch helpers below are the byte-for-byte canonical forms those files used;
// centralizing them keeps a test-technique fix a one-file edit instead of a
// dozen-file sweep. Files whose local helper diverges (e.g. a synchronous `click`
// that runs inside the caller's own `act`) intentionally keep their own copy.
import { act } from "react";
import { createRoot } from "react-dom/client";

// Dispatch a bubbling click and flush the resulting React work inside `act`.
// Matches the `async function click(element)` form the studio suites shared.
export async function click(element) {
  await act(async () => {
    element.dispatchEvent(new window.MouseEvent("click", { bubbles: true }));
  });
}

// Set a controlled <input> value through the native setter (bypassing React's
// value-tracking) and fire the `input` event React listens for. Synchronous —
// callers wrap in `act` when they need the update flushed before asserting.
export function setInput(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("input", { bubbles: true }));
}

// Set a controlled <select> value through the native setter and fire `change`.
export function setSelect(element, value) {
  const setter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, "value").set;
  setter.call(element, value);
  element.dispatchEvent(new window.Event("change", { bubbles: true }));
}

// Attach a FileList to a file <input> and fire `change` so upload handlers run.
export function setFileInput(element, files) {
  Object.defineProperty(element, "files", {
    configurable: true,
    value: files,
  });
  element.dispatchEvent(new window.Event("change", { bubbles: true }));
}

// Create a detached container mounted on document.body plus a React root for it,
// the create half of the shared beforeEach scaffolding. Returns both so the test
// can render into `root` and query `container`.
export function mountRoot() {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  return { container, root };
}

// Tear a mountRoot() pair down, the cleanup half of the shared afterEach: unmount
// inside `act` (so effect cleanups run) and detach the container from the DOM.
export async function unmountRoot(root, container) {
  await act(async () => root.unmount());
  container.remove();
}
