// sc-24216 — Supra2-IMG's model details must show each component's licence, the total download
// size and the supported limits. Bound to the SHIPPED manifest entry so a dropped per-download
// licence, a lost prompt limit or a renamed component row fails here rather than silently emptying
// the disclosure.
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import React, { act } from "react";
import { createRoot } from "react-dom/client";
import JSON5 from "json5";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ModelComponentDetails, hasModelComponentDetails, modelLimitLines } from "./ModelComponentDetails.jsx";

const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../../config/manifests/builtin.models.jsonc");
const manifestModels = (() => {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  return Array.isArray(parsed) ? parsed : parsed.models;
})();
const supra2 = manifestModels.find((model) => model.id === "supra2_img");

describe("ModelComponentDetails", () => {
  let container;
  let root;

  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
  });

  async function render(model, totalSizeText) {
    await act(async () => {
      root.render(<ModelComponentDetails model={model} totalSizeText={totalSizeText} />);
    });
  }

  it("lists every Supra2 component with its pinned revision, size and own licence", async () => {
    expect(supra2, "supra2_img is in the shipped catalog").toBeTruthy();
    await render(supra2, "1.6 GiB");
    const details = container.querySelector("details.model-component-details");
    expect(details).toBeTruthy();
    expect(details.open, "busy-card disclosures default collapsed").toBe(false);
    const rows = [...details.querySelectorAll("tbody tr")].map((row) =>
      [...row.querySelectorAll("td")].map((cell) => cell.textContent.trim()),
    );
    expect(rows).toEqual([
      ["Model weights", "SupraLabs/Supra2-IMG @ 10dec6e4b4b5", "397.3 MiB", "Apache-2.0"],
      ["Text encoder", "google/flan-t5-base @ 7bcac572ce56", "946.8 MiB", "Apache-2.0"],
      ["VAE decoder", "stabilityai/sd-vae-ft-mse @ 31f26fdeee13", "319.1 MiB", "MIT"],
    ]);
    // The full revision stays reachable on hover.
    expect(details.querySelector(".model-component-revision").title).toBe(
      "10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b",
    );
    expect(details.textContent).toContain("Total download: 1.6 GiB");
  });

  it("states the supported limits: native resolution, images per job and the prompt limit", async () => {
    expect(modelLimitLines(supra2)).toEqual([
      "Native resolution: 256×256",
      "Images per job: 1, 2, 4, 8",
      "Prompt: up to 128 tokens (longer prompts are truncated and flagged on the result)",
    ]);
    await render(supra2, null);
    const limits = container.querySelector('[aria-label="Supported limits"]');
    expect(limits.querySelectorAll("li")).toHaveLength(3);
    expect(container.textContent).not.toContain("Total download");
  });

  it("renders nothing for a model that declares no per-component licence or prompt limit", async () => {
    const plain = {
      id: "plain",
      downloads: [{ provider: "huggingface", repo: "org/plain" }],
      limits: { resolutions: ["1024x1024"] },
    };
    expect(hasModelComponentDetails(plain)).toBe(false);
    await render(plain, "1 GiB");
    expect(container.innerHTML).toBe("");
  });
});
