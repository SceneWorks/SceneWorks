import React, { act } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot, click } from "../testUtils/dom.js";
import { VqaPanel } from "./assetPanels.jsx";
import { CharacterTest } from "../screens/characterPanels.jsx";

const feedback = vi.hoisted(() => ({ value: null }));
vi.mock("../hooks/usePortableFeedback.js", () => ({
  usePortableFeedback: (_kind, _payload, _entry, enabled = true) => enabled ? feedback.value : null,
}));
let container;
let root;
beforeEach(() => { global.IS_REACT_ACT_ENVIRONMENT = true; ({ container, root } = mountRoot()); });
afterEach(async () => { await unmountRoot(root, container); });
const message = (state) => state === "pending" ? "Checking options with the server" : state === "server-required" ? "Reconnect to continue" : "Fix the authored option";
const refusal = { code: "invalid_generation_option", field: "maxNewTokens", detail: "Fix the authored option" };

it.each(["pending", "server-required", "invalid"])("blocks VQA %s and permits confirmed feedback", async (state) => {
  feedback.value = { state, refusal };
  const onAsk = vi.fn();
  const ui = <VqaPanel asset={{ id: "image" }} project={{ id: "project" }} entries={[]} onAsk={onAsk} />;
  await act(async () => root.render(ui));
  await act(async () => {
    const textarea = container.querySelector("textarea");
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set.call(textarea, "Describe this image");
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(container.querySelector("button").disabled).toBe(true);
  expect(container.textContent).toContain(message(state));
  await click(container.querySelector("button"));
  expect(onAsk).not.toHaveBeenCalled();
  feedback.value = { state: "valid" };
  await act(async () => root.render(React.cloneElement(ui)));
  expect(container.querySelector("button").disabled).toBe(false);
  await click(container.querySelector("button"));
  expect(onAsk).toHaveBeenCalledWith({ id: "image" }, "Describe this image", 256);
});

it.each(["pending", "server-required", "invalid"])("blocks character test %s, including direct form submit", async (state) => {
  const submitTest = vi.fn((event) => event.preventDefault());
  const props = { selectedCharacter: { id: "character" }, imageModels: [], latestAssets: [], testPrompt: "portrait", testCount: 1, testResolution: "1024x1024", testLookId: "", testModel: "", submitTest };
  await act(async () => root.render(<CharacterTest {...props} testFeedback={{ state, refusal }} testRefusal={state === "invalid" ? refusal : null} />));
  expect(container.querySelector('[type="submit"]').disabled).toBe(true);
  expect(container.textContent).toContain(message(state));
  await act(async () => container.querySelector("form").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(submitTest).not.toHaveBeenCalled();
  await act(async () => root.render(<CharacterTest {...props} testFeedback={{ state: "valid" }} />));
  expect(container.querySelector('[type="submit"]').disabled).toBe(false);
  await act(async () => container.querySelector("form").dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })));
  expect(submitTest).toHaveBeenCalledTimes(1);
});
