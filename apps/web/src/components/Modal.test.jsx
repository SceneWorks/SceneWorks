import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Modal } from "./Modal.jsx";

describe("Modal keyboard focus", () => {
  let container;
  let root;
  let opener;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    opener = document.createElement("button");
    opener.textContent = "Open";
    document.body.appendChild(opener);
    opener.focus();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    opener.remove();
    vi.restoreAllMocks();
  });

  it("traps Tab within the dialog and restores the opener on unmount", async () => {
    await act(async () => {
      root.render(
        <Modal label="Test dialog" onClose={() => {}} className="test-modal">
          <button>First</button>
          <a href="/inside">Middle</a>
          <button>Last</button>
        </Modal>,
      );
    });

    const dialog = document.body.querySelector(".test-modal");
    const [first, , last] = dialog.querySelectorAll("button, a");
    expect(document.activeElement).toBe(dialog);

    await act(async () => {
      dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true }));
    });
    expect(document.activeElement).toBe(first);

    last.focus();
    await act(async () => {
      last.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true }));
    });
    expect(document.activeElement).toBe(first);

    await act(async () => {
      first.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, bubbles: true }),
      );
    });
    expect(document.activeElement).toBe(last);

    await act(async () => root.render(<div />));
    expect(document.activeElement).toBe(opener);
  });

  it("keeps focus on the dialog when it has no focusable children", async () => {
    await act(async () => {
      root.render(
        <Modal label="Empty dialog" onClose={() => {}} className="empty-modal">
          <p>Nothing interactive</p>
        </Modal>,
      );
    });
    const dialog = document.body.querySelector(".empty-modal");
    await act(async () => {
      dialog.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true }));
    });
    expect(document.activeElement).toBe(dialog);
  });
});
