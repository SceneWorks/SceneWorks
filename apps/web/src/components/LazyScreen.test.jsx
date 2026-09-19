import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { lazyInteraction, lazyScreen } from "./LazyScreen.jsx";

const mountedRoots = [];
let consoleError;

beforeEach(() => {
  consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(async () => {
  while (mountedRoots.length) {
    const root = mountedRoots.pop();
    await act(async () => root.unmount());
  }
  document.body.innerHTML = "";
  consoleError.mockRestore();
});

async function render(ui) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  mountedRoots.push(root);
  await act(async () => root.render(ui));
  return container;
}

describe("lazyScreen", () => {
  it("announces loading and then mounts the requested named export", async () => {
    let resolveImport;
    const importScreen = vi.fn(
      () =>
        new Promise((resolve) => {
          resolveImport = resolve;
        }),
    );
    const Screen = lazyScreen(importScreen, "ExampleScreen", "Example");
    const container = await render(<Screen value="ready" />);

    expect(container.querySelector('[role="status"]')?.textContent).toContain("Loading Example");
    expect(container.querySelector('[aria-busy="true"]')).not.toBeNull();

    await act(async () => {
      resolveImport({ ExampleScreen: ({ value }) => <p>{value}</p> });
      await Promise.resolve();
    });

    expect(container.querySelector('[role="status"]')).toBeNull();
    expect(container.textContent).toContain("ready");
  });

  it("surfaces an accessible error and retries a failed import", async () => {
    const importScreen = vi
      .fn()
      .mockRejectedValueOnce(new Error("chunk unavailable"))
      .mockResolvedValueOnce({ ExampleScreen: () => <p>loaded after retry</p> });
    const Screen = lazyScreen(importScreen, "ExampleScreen", "Example");
    const container = await render(<Screen />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    const alert = container.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain("Example could not be loaded");

    await act(async () => {
      alert.querySelector("button").click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(importScreen).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("loaded after retry");
  });
});

describe("lazyInteraction", () => {
  it("announces a failed import and retries with a fresh lazy payload", async () => {
    const importComponent = vi
      .fn()
      .mockRejectedValueOnce(new Error("chunk unavailable"))
      .mockResolvedValueOnce({ ExamplePanel: () => <p>loaded after retry</p> });
    const Panel = lazyInteraction(importComponent, "ExamplePanel", "Example panel");
    const container = await render(<Panel />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    const alert = container.querySelector('[role="alert"]');
    expect(alert?.textContent).toContain("Example panel could not be loaded");

    await act(async () => {
      alert.querySelector("button").click();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(importComponent).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("loaded after retry");
  });

  it("creates a fresh lazy payload after the interaction closes and reopens", async () => {
    const importComponent = vi.fn(async () => ({ ExamplePanel: () => <p>open</p> }));
    const Panel = lazyInteraction(importComponent, "ExamplePanel", "Example panel");
    let setOpen;
    function Host() {
      const [open, updateOpen] = React.useState(true);
      setOpen = updateOpen;
      return open ? <Panel /> : null;
    }
    const container = await render(<Host />);

    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(container.textContent).toContain("open");

    await act(async () => setOpen(false));
    expect(container.textContent).toBe("");
    await act(async () => {
      setOpen(true);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(container.textContent).toContain("open");
    expect(importComponent).toHaveBeenCalledTimes(2);
  });
});
