// sc-12068: emptyTrash confirms a permanent purge through the shared desktop-safe
// appConfirm helper (a real in-app dialog) instead of the raw window.confirm, which
// silently no-ops inside the Tauri WebView. These tests mock appConfirm so they can
// control the user's choice and assert the guard fired with the destructive (danger)
// tone, and that nothing is purged when the confirm is declined.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { appConfirmMock } = vi.hoisted(() => ({ appConfirmMock: vi.fn(async () => true) }));
vi.mock("../appConfirm.jsx", () => ({
  appConfirm: appConfirmMock,
  useConfirm: () => appConfirmMock,
  ConfirmHost: () => null,
}));

import { emptyTrash } from "./assetPanels.jsx";

const trashed = (id) => ({ id, status: { trashed: true } });
const kept = (id) => ({ id, status: { trashed: false } });

describe("emptyTrash desktop-safe confirm (sc-12068)", () => {
  beforeEach(() => {
    appConfirmMock.mockClear();
    appConfirmMock.mockResolvedValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("confirms via appConfirm (danger tone) and purges every trashed item when accepted", async () => {
    const purgeAsset = vi.fn(async () => {});
    await emptyTrash([trashed("a"), trashed("b"), kept("c")], purgeAsset);

    // The guard went through appConfirm — not the raw window.confirm.
    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(appConfirmMock.mock.calls[0][0]).toMatchObject({ tone: "danger" });
    expect(appConfirmMock.mock.calls[0][0].message).toContain("2 discarded items");

    // Only the trashed items are purged, in order; the kept one is untouched.
    expect(purgeAsset).toHaveBeenCalledTimes(2);
    expect(purgeAsset.mock.calls.map((call) => call[0].id)).toEqual(["a", "b"]);
  });

  it("purges nothing when the confirm is declined", async () => {
    appConfirmMock.mockResolvedValue(false);
    const purgeAsset = vi.fn(async () => {});
    await emptyTrash([trashed("a")], purgeAsset);

    expect(appConfirmMock).toHaveBeenCalledTimes(1);
    expect(purgeAsset).not.toHaveBeenCalled();
  });

  it("uses the singular label for a single discarded item", async () => {
    const purgeAsset = vi.fn(async () => {});
    await emptyTrash([trashed("only")], purgeAsset);

    expect(appConfirmMock.mock.calls[0][0].message).toContain("1 discarded item?");
  });

  it("never prompts when there is nothing trashed", async () => {
    const purgeAsset = vi.fn(async () => {});
    await emptyTrash([kept("c")], purgeAsset);

    expect(appConfirmMock).not.toHaveBeenCalled();
    expect(purgeAsset).not.toHaveBeenCalled();
  });
});
