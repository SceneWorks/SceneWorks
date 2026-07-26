import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppContext } from "../context/AppContext.js";
import { SimpleUiContext } from "./SimpleUiContext.js";
import { click, mountRoot, unmountRoot } from "../testUtils/dom.js";

// sc-15136 — the Simple shell's own durable preference write must be authenticated.
//
// `PUT /api/v1/ui-preferences` is GATED (only the pre-auth GET is public; sc-8869, F-067). This screen
// sent the default-quality write with an empty token and swallowed the 401, so on a remote-auth
// deployment the Simple shell's quality preference never reached `ui-preferences.json` — hidden by the
// localStorage instant-paint cache that the same handler writes.
//
// `apiFetch` is mocked at the transport seam that `uiPreferences.js` uses. The App-level suite
// (`App.uiPreferences.test.jsx`) covers the real header on the wire; this pins the Simple shell's own
// call site, which is a second, independent handler from the advanced Settings screen.

const apiFetch = vi.fn(async () => ({}));
vi.mock("../api.js", () => ({
  apiFetch,
  isAbortError: () => false,
  API_BASE_URL: "",
  eventUrl: () => "",
}));
// `credentials.js` is left real: it routes through the same mocked `apiFetch`, so the
// credentials card loads empty without a second mock to keep in sync with its exports.

const { SimpleSettings } = await import("./SimpleSettings.jsx");

const APP_CONTEXT = {
  theme: "light",
  changeTheme: vi.fn(),
  visibleWorkers: [],
  macCapabilities: null,
};

const SIMPLE_UI = { toast: vi.fn(), breakpoint: "desktop" };

describe("SimpleSettings — durable default-quality write (sc-15136)", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    window.localStorage.clear();
    apiFetch.mockClear();
    ({ container, root } = mountRoot());
  });

  afterEach(async () => {
    await unmountRoot(root, container);
    window.localStorage.clear();
  });

  async function render() {
    await act(async () => {
      root.render(
        <AppContext.Provider value={APP_CONTEXT}>
          <SimpleUiContext.Provider value={SIMPLE_UI}>
            <SimpleSettings
              accent="violet"
              lockedToSimple={false}
              onAccentChange={vi.fn()}
              onSimpleDefaultChange={vi.fn()}
              simpleDefault={false}
            />
          </SimpleUiContext.Provider>
        </AppContext.Provider>,
      );
    });
    await act(async () => {});
  }

  const chooseQuality = async (label) => {
    const chip = [
      ...container.querySelectorAll('[aria-label="Default quality"] button'),
    ].find((button) => button.textContent.trim() === label);
    expect(chip, `quality chip "${label}"`).toBeTruthy();
    await click(chip);
  };

  it("carries the access token on the gated PUT (remote auth)", async () => {
    // A non-empty token is pinned deliberately: asserting the "" default passes with the fix
    // reverted. This call site had no token coverage at all before sc-15136.
    window.localStorage.setItem("sceneworks-token", "lan-password-1");
    await render();

    await chooseQuality("Q4");

    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/ui-preferences",
      "lan-password-1",
      expect.objectContaining({
        method: "PUT",
        body: JSON.stringify({ defaultGenerationQuality: "q4" }),
      }),
    );
  });

  it("still writes with no token on desktop/loopback, where none is stored", async () => {
    // `SCENEWORKS_TRUST_LOOPBACK` exempts loopback peers before any token comparison, so the
    // desktop shell must keep persisting with nothing to send.
    await render();

    await chooseQuality("Q4");

    expect(apiFetch).toHaveBeenCalledWith(
      "/api/v1/ui-preferences",
      "",
      expect.objectContaining({ method: "PUT" }),
    );
  });
});
