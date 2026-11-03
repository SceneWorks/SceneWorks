import React, { act } from "react";
import { beforeEach, afterEach, expect, it, vi } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
const { get, put, remove } = vi.hoisted(() => ({ get: vi.fn(), put: vi.fn(), remove: vi.fn() }));
vi.mock("../api/yue2.js", () => ({ getLicenseAcknowledgment: get, putLicenseAcknowledgment: put, deleteLicenseAcknowledgment: remove }));
import { useLicenseAcknowledgments } from "./useLicenseAcknowledgments.js";
const model = { id: "licensed", requiresLicenseAcknowledgment: true, licenseNotice: "terms 1" };
let container, root, state;
function Harness({ models = [model], token = "context-token" }) { state = useLicenseAcknowledgments(models, token); return <span>{String(state.licenseAcks.licensed)}</span>; }
const render = async (props = {}) => act(async () => root.render(<Harness {...props} />));
beforeEach(() => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  ({ container, root } = mountRoot());
  window.localStorage.clear();
  get.mockReset().mockResolvedValue({ acknowledged: true, termsSha256: "terms-sha" });
  put.mockReset().mockResolvedValue({ acknowledged: true });
  remove.mockReset().mockResolvedValue({ acknowledged: false });
});
afterEach(async () => { if (root) await unmountRoot(root, container); vi.restoreAllMocks(); });
it("hydrates server acceptance without browser storage, writes through, and reloads changed terms/token", async () => {
  await render();
  expect(get).toHaveBeenCalledWith("licensed", "context-token");
  expect(state.licenseAcks.licensed).toBe(true);
  await act(async () => state.setLicenseAck("licensed", false));
  expect(remove).toHaveBeenCalledWith("licensed", "context-token");
  expect(state.licenseAcks.licensed).toBe(false);
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).toHaveBeenCalledWith("licensed", "context-token", "terms-sha");
  expect(state.licenseAcks.licensed).toBe(true);
  get.mockResolvedValue({ acknowledged: false });
  await render({ models: [{ ...model, licenseNotice: "terms 2" }] });
  expect(state.licenseAcks.licensed).toBe(false);
  await render({ token: "replacement" });
  expect(get).toHaveBeenLastCalledWith("licensed", "replacement");
});
it("ignores forged paint cache, fails closed on server errors, and reports failed writes", async () => {
  window.localStorage.setItem("sceneworks-license-ack:licensed", "true");
  get.mockRejectedValue(new Error("offline"));
  await render();
  expect(state.licenseAcks.licensed).not.toBe(true);
  expect(state.licenseAckError).toContain("offline");
  get.mockResolvedValue({ acknowledged: false, termsSha256: "terms-sha" });
  await render({ token: "replacement" });
  put.mockRejectedValue(new Error("write failed"));
  await act(async () => state.setLicenseAck("licensed", true));
  expect(state.licenseAcks.licensed).toBe(false);
  expect(state.licenseAckError).toContain("write failed");
});
it("cannot accept before current server terms hydrate", async () => {
  let hydrate;
  get.mockImplementation(() => new Promise((resolve) => { hydrate = resolve; }));
  await render();
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).not.toHaveBeenCalled();
  expect(state.licenseAcks.licensed).toBe(false);
  await act(async () => hydrate({ acknowledged: false, termsSha256: "terms-sha" }));
  expect(state.licenseAcks.licensed).toBe(false);
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).toHaveBeenCalledWith("licensed", "context-token", "terms-sha");
  expect(state.licenseAcks.licensed).toBe(true);
});
it("serializes a slow acceptance before withdrawal and ignores its obsolete response", async () => {
  let accept;
  put.mockImplementation(() => new Promise((resolve) => { accept = resolve; }));
  await render();
  let first, last;
  await act(async () => { first = state.setLicenseAck("licensed", true); await Promise.resolve(); });
  await act(async () => { last = state.setLicenseAck("licensed", false); });
  expect(remove).not.toHaveBeenCalled();
  expect(state.licenseAcks.licensed).toBe(false);
  await act(async () => { accept({ acknowledged: true }); await Promise.all([first, last]); });
  expect(remove).toHaveBeenCalledOnce();
  expect(state.licenseAcks.licensed).toBe(false);
});

it("refuses to accept newer server terms than the displayed catalog notice", async () => {
  get.mockResolvedValue({ acknowledged: false, termsSha256: "new-terms", terms: { license: null, licenseUrl: null, licenseNotice: "new unseen terms" } });
  await render();
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).not.toHaveBeenCalled();
  expect(state.licenseAcks.licensed).toBe(false);
  expect(state.licenseAckError).toContain("review the current license terms");
});

it("hydrates terms without resurrecting an acknowledgment withdrawn during the GET", async () => {
  let hydrate;
  get.mockImplementation(() => new Promise((resolve) => { hydrate = resolve; }));
  await render();
  await act(async () => state.setLicenseAck("licensed", false));
  await act(async () => hydrate({ acknowledged: true, termsSha256: "terms-sha" }));
  expect(state.licenseAcks.licensed).toBe(false);
  expect(window.localStorage.getItem("sceneworks-license-ack:licensed")).toBeNull();
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).toHaveBeenCalledWith("licensed", "context-token", "terms-sha");
});

it("ignores obsolete hydration after token, terms, model, or unmount changes", async () => {
  const pending = [];
  get.mockImplementation((id, token) => new Promise((resolve) => pending.push({ id, token, resolve })));
  await render();
  await render({ token: "new-token", models: [{ ...model, licenseNotice: "terms 2" }] });
  await act(async () => pending[0].resolve({ acknowledged: true, termsSha256: "old" }));
  expect(state.licenseAcks.licensed).not.toBe(true);
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).not.toHaveBeenCalled();
  await act(async () => pending[1].resolve({ acknowledged: false, termsSha256: "new" }));
  await act(async () => state.setLicenseAck("licensed", true));
  expect(put).toHaveBeenCalledWith("licensed", "new-token", "new");
  await render({ token: "new-token", models: [{ ...model, id: "other" }] });
  await act(async () => pending[2].resolve({ acknowledged: true, termsSha256: "other" }));
  expect(state.licenseAcks.licensed).toBeUndefined();
  expect(state.licenseAcks.other).toBe(true);
  await render({ token: "last", models: [model] });
  window.localStorage.removeItem("sceneworks-license-ack:licensed");
  await unmountRoot(root, container);
  root = null;
  await act(async () => pending[3].resolve({ acknowledged: true, termsSha256: "late" }));
  expect(window.localStorage.getItem("sceneworks-license-ack:licensed")).toBeNull();
});
