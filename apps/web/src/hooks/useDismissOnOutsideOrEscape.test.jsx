import React, { act, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { expect, it } from "vitest";
import { useDismissOnOutsideOrEscape } from "./useDismissOnOutsideOrEscape.js";

it("dismisses only on outside mousedown or Escape while open and cleans up", () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  function Menu() {
    const [open, setOpen] = useState(false);
    const ref = useRef(null);
    useDismissOnOutsideOrEscape(ref, open, setOpen);
    return <div ref={ref}><button onClick={() => setOpen(true)}>Open</button>{open && <span>Menu</span>}</div>;
  }
  act(() => root.render(<Menu />));
  const button = host.querySelector("button");
  act(() => button.click());
  act(() => button.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
  expect(host.textContent).toContain("Menu");
  act(() => document.body.dispatchEvent(new MouseEvent("mousedown", { bubbles: true })));
  expect(host.textContent).not.toContain("Menu");
  act(() => button.click());
  act(() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
  expect(host.textContent).not.toContain("Menu");
  act(() => root.unmount());
  host.remove();
});
