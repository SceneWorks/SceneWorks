import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { LicensesScreen } from "./LicensesScreen.jsx";
import { bundledLicenses } from "../data/bundledLicenses.js";

// The corpus is imported from apps/desktop/licenses/ at build time, so these tests
// assert against the real bundled notices rather than a mock.
describe("LicensesScreen", () => {
  let container;
  let root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  async function render() {
    await act(async () => {
      root.render(<LicensesScreen />);
    });
  }

  it("lists every bundled component", async () => {
    await render();
    const items = container.querySelectorAll(".licenses-item");
    expect(items.length).toBe(bundledLicenses.length);
    expect(container.textContent).toContain("FFmpeg");
    expect(container.textContent).toContain("ONNX Runtime");
  });

  it("shows the first component's license text by default", async () => {
    await render();
    // ffmpeg is first: its written-offer notice mentions GPLv3 §6.
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Written offer for corresponding source",
    );
  });

  it("switches the displayed component on selection", async () => {
    await render();
    const onnx = [...container.querySelectorAll(".licenses-item")].find((b) =>
      b.textContent.includes("ONNX Runtime"),
    );
    await act(async () => onnx.click());
    expect(container.textContent).toContain("Microsoft Corporation");
    expect(container.querySelector(".licenses-text").textContent).toContain("MIT License");
  });

  it("lists the re-hosted AI models with their upstream license text", async () => {
    await render();
    // A Wan2.2 model is redistributed under Apache-2.0.
    const wan = [...container.querySelectorAll(".licenses-item")].find((b) =>
      b.textContent.includes("Wan2.2-TI2V-5B"),
    );
    expect(wan).toBeTruthy();
    await act(async () => wan.click());
    expect(container.textContent).toContain("Wan-AI / Alibaba Tongyi Lab");
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Apache License",
    );
  });

  it("renders complete notices for ported Cephes source and embedded CMUDICT data", async () => {
    await render();
    const cases = [
      ["Cephes Math Library", "Steven Moshier", "Neither the name"],
      ["CMU Pronouncing Dictionary", "Carnegie Mellon University", "Defense Advanced"],
    ];
    for (const [name, copyright, condition] of cases) {
      const item = [...container.querySelectorAll(".licenses-item")].find((button) =>
        button.textContent.includes(name),
      );
      expect(item).toBeTruthy();
      await act(async () => item.click());
      const text = container.querySelector(".licenses-text").textContent;
      const normalized = text.replace(/\s+/g, " ");
      expect(text).toContain(copyright);
      expect(text).toContain(condition);
      expect(normalized).toContain("DIRECT, INDIRECT, INCIDENTAL");
    }
  });

  it("records the ACE-Step SFT Cover-restyle co-requisite under MIT (sc-13821)", async () => {
    await render();
    const sftCover = [...container.querySelectorAll(".licenses-item")].find((b) =>
      b.textContent.includes("ACE-Step v1.5 XL SFT"),
    );
    expect(sftCover).toBeTruthy();
    await act(async () => sftCover.click());
    // The usage copy names the Cover-only role and the three re-hosted modules.
    expect(container.textContent).toContain("Cover");
    expect(container.querySelector(".licenses-text").textContent).toContain("MIT License");
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "ACE Studio and StepFun",
    );
  });

  it("surfaces both LTX-2 and Gemma notices for the LTX bundle", async () => {
    await render();
    const ltx = [...container.querySelectorAll(".licenses-item")].find((b) =>
      b.textContent.includes("LTX-2.3"),
    );
    expect(ltx).toBeTruthy();
    await act(async () => ltx.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "LTX-2 Community License Agreement",
    );
    const gemmaTab = [...container.querySelectorAll(".segmented-control button")].find((b) =>
      b.textContent.includes("Gemma"),
    );
    await act(async () => gemmaTab.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Gemma Prohibited Use Policy",
    );
  });

  it("surfaces the LTX-2.5 Community License and the Gemma 4 Apache-2.0 + Prohibited Use Policy documents (sc-18785)", async () => {
    await render();
    const ltx25 = [...container.querySelectorAll(".licenses-item")].find((b) =>
      b.textContent.includes("LTX-2.5"),
    );
    expect(ltx25).toBeTruthy();
    await act(async () => ltx25.click());
    // Default document is the LTX-2.x Community License Agreement dated 2026-08-11 — a
    // different, later text than LTX-2.3's January 5, 2026 agreement.
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "LTX-2.x Community License Agreement",
    );
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "License date: August 11, 2026",
    );

    const tabs = [...container.querySelectorAll(".segmented-control button")];
    const apacheTab = tabs.find((b) => b.textContent.includes("Apache License 2.0"));
    expect(apacheTab).toBeTruthy();
    await act(async () => apacheTab.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Apache License",
    );

    const prohibitedTab = tabs.find((b) => b.textContent.includes("Prohibited Use Policy"));
    expect(prohibitedTab).toBeTruthy();
    await act(async () => prohibitedTab.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Gemma Prohibited Use Policy",
    );

    const noticeTab = tabs.find((b) => b.textContent.includes("provenance"));
    expect(noticeTab).toBeTruthy();
    await act(async () => noticeTab.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "Diff against LTX-2.3's bundled license",
    );
  });

  it("binds the public LTX IC-LoRA rehost to its full license and immutable provenance", async () => {
    await render();
    const loras = [...container.querySelectorAll(".licenses-item")].find((button) =>
      button.textContent.includes("LTX-2.3 IC-LoRA HDR and LipDub"),
    );
    expect(loras).toBeTruthy();
    await act(async () => loras.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "LTX-2 Community License Agreement",
    );

    const noticeTab = [...container.querySelectorAll(".segmented-control button")].find((button) =>
      button.textContent.includes("immutable provenance"),
    );
    expect(noticeTab).toBeTruthy();
    await act(async () => noticeTab.click());
    const notice = container.querySelector(".licenses-text").textContent;
    expect(notice).toContain("ca287bbae91f939481b3b36764d1e8b2cfb6160b");
    expect(notice).toContain("ltx-2.3-22b-ic-lora-hdr-0.9.safetensors");
    expect(notice).toContain("ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors");
    expect(notice).toContain("ltx-2.3-22b-ic-lora-dubit-0.9.safetensors");
  });

  it("switches between a component's license documents", async () => {
    await render();
    // ffmpeg has two docs (notice + GPL text); pick the GPL tab.
    const gplTab = [...container.querySelectorAll(".segmented-control button")].find((b) =>
      b.textContent.includes("General Public License"),
    );
    await act(async () => gplTab.click());
    expect(container.querySelector(".licenses-text").textContent).toContain(
      "GNU GENERAL PUBLIC LICENSE",
    );
  });
});
