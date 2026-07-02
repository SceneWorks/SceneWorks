import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetAddDialog } from "./DatasetAddDialog.jsx";

// sc-8857 regression: the Character tab must surface an asset that is only in a
// character's `approvedReferences` list (not its plain `references`). Before the
// shared `assetMatchesCharacter` predicate, this dialog checked `references`
// only, so approved-reference-only assets silently failed to appear here even
// though the Image-Edit source picker matched them.

const approvedOnlyAsset = {
  id: "asset-approved",
  projectId: "project-1",
  displayName: "Approved Reference",
  type: "image",
  status: {},
  file: { path: "assets/images/approved.png", mimeType: "image/png" },
};
const nonMemberAsset = {
  id: "asset-other",
  projectId: "project-1",
  displayName: "Unrelated",
  type: "image",
  status: {},
  file: { path: "assets/images/other.png", mimeType: "image/png" },
};

const character = {
  id: "char-1",
  name: "Mira",
  approvedReferences: [{ assetId: "asset-approved" }],
  references: [],
};

let container;
let root;

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

function characterCards() {
  return [...document.body.querySelectorAll(".dataset-add-card span")].map((el) => el.textContent);
}

describe("DatasetAddDialog Character tab (sc-8857)", () => {
  it("surfaces an approvedReferences-only asset and hides non-members", () => {
    act(() => {
      root.render(
        <DatasetAddDialog
          assets={[approvedOnlyAsset, nonMemberAsset]}
          characters={[character]}
          onAdd={vi.fn()}
          onClose={vi.fn()}
          onImport={vi.fn()}
        />,
      );
    });

    // Switch to the Character source tab.
    const characterTab = [...document.body.querySelectorAll('[role="tab"]')].find(
      (tab) => tab.textContent === "Character",
    );
    act(() => characterTab.dispatchEvent(new MouseEvent("click", { bubbles: true })));

    const cards = characterCards();
    expect(cards).toContain("Approved Reference");
    expect(cards).not.toContain("Unrelated");
  });
});
