import React, { act } from "react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mountRoot, unmountRoot } from "../testUtils/dom.js";
import { SimpleJobCard } from "./SimpleJobCard.jsx";

describe("Simple job progress messages", () => {
  let root;
  let container;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    ({ root, container } = mountRoot());
  });
  afterEach(async () => unmountRoot(root, container));

  async function render(overrides = {}) {
    await act(async () => root.render(<SimpleJobCard job={{ id: "job_1", type: "image_generate", status: "loading_model", stage: "loading_model", progress: 0.1, payload: {}, ...overrides }} />));
  }

  it.each(["image_generate", "video_generate", "audio_generate", "lora_train"])(
    "shows current worker messages unchanged for %s even when stage and percentage do not change",
    async (type) => {
      const phases = [
        "Checking model files (28/28).",
        "Loading text encoder — 6.1 / 15.2 GB",
        "Loading transformer (weights-00001.safetensors) — 1.0 / 2.0 GB",
        "Image 1/1 — Encoding prompt.",
      ];
      for (const message of phases) {
        await render({ type, message });
        expect(container.querySelector(".su-job-message")?.textContent).toBe(message);
        for (const previous of phases.filter((phase) => phase !== message)) {
          expect(container.textContent).not.toContain(previous);
        }
      }
      await render({ type, status: "completed", message: phases[2] });
      expect(container.querySelector(".su-job-message")).toBeNull();
      expect(container.textContent).not.toContain(phases[2]);
    },
  );

  it.each(["failed", "canceled", "interrupted"])("keeps %s error semantics without a duplicate live loading label", async (status) => {
    await render({ status, message: "Loading vae — 1.0 / 2.0 GB", error: "The worker stopped this run." });
    expect(container.querySelector(".su-job-message")).toBeNull();
    expect(container.querySelector(".su-job-error")?.textContent).toBe("The worker stopped this run.");
    expect(container.textContent).not.toContain("Loading vae");
  });

  it("does not add an empty row or replace the existing worker error fallback", async () => {
    await render({ message: "" });
    expect(container.querySelector(".su-job-message")).toBeNull();
    await render({ status: "failed", message: "Model files are missing." });
    expect(container.querySelector(".su-job-error")?.textContent).toBe("Model files are missing.");
  });
});
