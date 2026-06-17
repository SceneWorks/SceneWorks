import { beforeEach, describe, expect, it } from "vitest";
import {
  ADVANCED_UI_MODE,
  persistUiMode,
  readStoredUiMode,
  SIMPLE_UI_MODE,
  UI_MODE_STORAGE_KEY,
} from "./uiMode.js";

describe("ui mode persistence", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("defaults to advanced mode when no explicit setting is stored", () => {
    expect(readStoredUiMode()).toBe(ADVANCED_UI_MODE);
  });

  it("persists simple mode when explicitly enabled", () => {
    expect(persistUiMode(SIMPLE_UI_MODE)).toBe(SIMPLE_UI_MODE);
    expect(window.localStorage.getItem(UI_MODE_STORAGE_KEY)).toBe(SIMPLE_UI_MODE);
    expect(readStoredUiMode()).toBe(SIMPLE_UI_MODE);
  });

  it("falls back to advanced mode for unsupported stored values", () => {
    window.localStorage.setItem(UI_MODE_STORAGE_KEY, "compact");
    expect(readStoredUiMode()).toBe(ADVANCED_UI_MODE);
  });
});
