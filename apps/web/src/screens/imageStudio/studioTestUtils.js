import { act } from "react";

// Test helpers for the Image Studio compose panel (studio layout simplification): the settings
// tab strip replaced the Advanced disclosure, and the Structure control became a prompt tool.

// Tab labels: "Model & output", "LoRAs", "Style & look", "Sampling", "Upscale & finish".
export function settingsTab(label, root = document.body) {
  return [...root.querySelectorAll('.image-settings-tabs [role="tab"]')].find((tab) =>
    tab.querySelector(".studio-settings-tab-label")?.textContent.includes(label),
  );
}

// Opens the named settings tab (a no-op when it is already open — a second click would close it).
export async function openSettingsTab(label, root = document.body) {
  const tab = settingsTab(label, root);
  if (!tab) throw new Error(`No Image Studio settings tab "${label}"`);
  if (tab.getAttribute("aria-selected") !== "true") {
    await act(async () => {
      tab.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  }
  return tab;
}

export function settingsPanel(root = document.body) {
  return root.querySelector(".image-settings-tabs [role='tabpanel']");
}

// Prompt tools in the prompt box footer: "Image reference", "Structure", "Prompt from image",
// "Refine", "Qwen rewrite", "Expand".
export function promptToolButton(label, root = document.body) {
  return [...root.querySelectorAll(".studio-prompt-tool")].find((button) =>
    button.textContent.trim().startsWith(label),
  );
}

// Opens the named prompt tool's assist panel (a no-op when it is already open).
export async function openPromptTool(label, root = document.body) {
  const button = promptToolButton(label, root);
  if (!button) throw new Error(`No Image Studio prompt tool "${label}"`);
  if (button.getAttribute("aria-pressed") !== "true") {
    await act(async () => {
      button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
  }
  return button;
}

// The Generate submit button (its label carries the keyboard-shortcut hint).
export function generateButton(root = document.body) {
  return root.querySelector("button.studio-generate[type='submit']");
}
