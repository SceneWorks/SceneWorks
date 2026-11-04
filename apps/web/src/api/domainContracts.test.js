import { afterEach, describe, expect, it, vi } from "vitest";
import { API_BASE_URL, setMediaTicket } from "../api.js";
import { assetsApi } from "./assetsResources.js";
import { charactersApi } from "./charactersResources.js";
import { modelsApi } from "./modelsResources.js";
import { trainingApi } from "./trainingResources.js";
import { mediaResponse, projectFileUrl } from "./media.js";
import { accessApi } from "./access.js";

const hostile = "org/item ?#%";
const segment = "org%2Fitem%20%3F%23%25";
afterEach(() => { vi.unstubAllGlobals(); setMediaTicket(""); });
function jsonFetch() {
  const fetch = vi.fn(async () => new Response(JSON.stringify({ ok: true })));
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

describe("domain route contracts", () => {
  it("encodes every nested identifier without altering payload fields, token or abort", async () => {
    const fetch = jsonFetch();
    const controller = new AbortController();
    const body = JSON.stringify({ assetId: hostile, prompt: "a/b ?#%" });
    await charactersApi.look(hostile, hostile, hostile, "token", { method: "PATCH", body, signal: controller.signal });
    expect(fetch.mock.calls[0][0]).toBe(`${API_BASE_URL}/api/v1/projects/${segment}/characters/${segment}/looks/${segment}`);
    const options = fetch.mock.calls[0][1];
    expect(options).toMatchObject({ method: "PATCH", body, signal: controller.signal });
    expect(options.headers.get("X-SceneWorks-Token")).toBe("token");
  });

  it("keeps plain asset lists, hidden-asset grids and permanent removal distinct", async () => {
    const fetch = jsonFetch();
    await assetsApi.list(hostile, "token");
    await assetsApi.listIncludingHidden(hostile, "token");
    await assetsApi.purgePermanently(hostile, hostile, "token", { method: "DELETE" });
    expect(fetch.mock.calls.map(([url]) => url.slice(API_BASE_URL.length))).toEqual([
      `/api/v1/projects/${segment}/assets`,
      `/api/v1/projects/${segment}/assets?includeRejected=true&includeTrashed=true`,
      `/api/v1/projects/${segment}/assets/${segment}/purge?permanent=true`,
    ]);
  });

  it("keeps slash-containing model ids as one parameter and multipart uploads unmodified", async () => {
    const fetch = jsonFetch();
    await modelsApi.conditionalDownload(hostile, "image/edit", "token", { method: "POST", body: "{}" });
    expect(fetch.mock.calls[0][0]).toBe(`${API_BASE_URL}/api/v1/models/${segment}/conditional-components/image%2Fedit/download`);
    const body = new FormData();
    body.append("file", new Blob(["data"]), "a.bin");
    await trainingApi.ltxPreparedBundle(hostile, hostile, hostile, "token", { method: "POST", body });
    expect(fetch.mock.calls[1][0]).toBe(`${API_BASE_URL}/api/v1/projects/${segment}/training/datasets/${segment}/items/${segment}/ltx-prepared-bundle`);
    expect(fetch.mock.calls[1][1].body).toBe(body);
    expect(fetch.mock.calls[1][1].headers.has("Content-Type")).toBe(false);
  });

  it("preserves empty ticket bodies and forwards ticketed media Response streams untouched", async () => {
    const fetch = jsonFetch();
    await accessApi.mediaTicket("token", { method: "POST" });
    expect(fetch.mock.calls[0][1]).not.toHaveProperty("body");
    setMediaTicket("ticket ?#");
    const url = projectFileUrl(hostile, "nested dir/file?#.png", { suffix: "?v=one" });
    expect(url).toBe(`${API_BASE_URL}/api/v1/projects/${segment}/files/nested%20dir/file%3F%23.png?v=one&ticket=ticket%20%3F%23`);
    const response = new Response("streamed bytes");
    fetch.mockResolvedValueOnce(response);
    const signal = new AbortController().signal;
    const actual = await mediaResponse(url, { signal, credentials: "include" });
    expect(actual).toBe(response);
    expect(actual.bodyUsed).toBe(false);
    expect(fetch.mock.calls[1]).toEqual([url, { signal, credentials: "include" }]);
    expect(await actual.text()).toBe("streamed bytes");
  });
});
