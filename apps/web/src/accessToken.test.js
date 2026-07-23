import { afterEach, describe, expect, it } from "vitest";
import {
  ACCESS_TOKEN_KEY,
  clearAccessToken,
  readAccessToken,
  storeAccessToken,
} from "./accessToken.js";

describe("accessToken (sc-8880)", () => {
  afterEach(() => {
    // Restore any storage descriptor a test replaced, and clear the key.
    if (originalStorageDescriptor) {
      Object.defineProperty(globalThis, "localStorage", originalStorageDescriptor);
      originalStorageDescriptor = null;
    }
    try {
      globalThis.localStorage?.removeItem(ACCESS_TOKEN_KEY);
    } catch {
      // storage may be intentionally broken by a test — ignore.
    }
  });

  let originalStorageDescriptor = null;

  it("persists the token under the single canonical key so it survives a reload", () => {
    expect(ACCESS_TOKEN_KEY).toBe("sceneworks-token");
    storeAccessToken("hunter2");
    // Read back through the helper AND the raw key: both must agree, proving the
    // helper writes exactly the documented storage contract.
    expect(readAccessToken()).toBe("hunter2");
    expect(window.localStorage.getItem(ACCESS_TOKEN_KEY)).toBe("hunter2");
  });

  it("returns an empty string when no token is stored", () => {
    expect(readAccessToken()).toBe("");
  });

  it("clears the stored token (the lock/forget affordance)", () => {
    storeAccessToken("to-forget");
    clearAccessToken();
    expect(readAccessToken()).toBe("");
    expect(window.localStorage.getItem(ACCESS_TOKEN_KEY)).toBeNull();
  });

  it("degrades gracefully when Web Storage is unavailable (private mode / disabled)", () => {
    originalStorageDescriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      get() {
        throw new Error("storage disabled");
      },
    });
    // No throw from any helper; read yields the empty-token default.
    expect(() => storeAccessToken("x")).not.toThrow();
    expect(readAccessToken()).toBe("");
    expect(() => clearAccessToken()).not.toThrow();
  });

  it("degrades gracefully when individual storage operations throw", () => {
    originalStorageDescriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      value: {
        getItem() {
          throw new Error("read denied");
        },
        setItem() {
          throw new Error("quota exceeded");
        },
        removeItem() {
          throw new Error("write denied");
        },
      },
    });
    expect(readAccessToken()).toBe("");
    expect(() => storeAccessToken("x")).not.toThrow();
    expect(() => clearAccessToken()).not.toThrow();
  });
});
