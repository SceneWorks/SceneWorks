import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ACCESS_TOKEN_KEY,
  clearAccessToken,
  readAccessToken,
  storeAccessToken,
  subscribeAccessToken,
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

  // sc-15165: storage is shared across tabs, the gate's React state is not. Without this
  // subscription the two diverge for the rest of the session — see the module header.
  describe("cross-tab subscription (sc-15165)", () => {
    const unsubscribes = [];
    function subscribe(listener) {
      const off = subscribeAccessToken(listener);
      unsubscribes.push(off);
      return off;
    }
    // Mutate storage the way another tab would, then deliver the event this tab receives.
    // jsdom (like a real browser) does NOT fire `storage` for same-window writes, so the
    // dispatch is explicit — the write is what makes the assertion meaningful. `storageArea`
    // is set because real browsers always set it and the handler discriminates on it.
    function otherTabWrites(value) {
      if (value === null) {
        window.localStorage.removeItem(ACCESS_TOKEN_KEY);
      } else {
        window.localStorage.setItem(ACCESS_TOKEN_KEY, value);
      }
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: ACCESS_TOKEN_KEY,
          newValue: value,
          storageArea: window.localStorage,
        }),
      );
    }

    afterEach(() => {
      while (unsubscribes.length) {
        unsubscribes.pop()();
      }
      vi.restoreAllMocks();
    });

    it("notifies subscribers with the cleared token when another tab hits forget", () => {
      storeAccessToken("shared-token");
      const seen = [];
      subscribe((next) => seen.push(next));
      otherTabWrites(null);
      // The empty string, not null/undefined: subscribers must get exactly what every
      // other reader in the tab now sees.
      expect(seen).toEqual([""]);
      expect(readAccessToken()).toBe("");
    });

    it("notifies subscribers with the new token when another tab unlocks", () => {
      const seen = [];
      subscribe((next) => seen.push(next));
      otherTabWrites("promoted-elsewhere");
      expect(seen).toEqual(["promoted-elsewhere"]);
    });

    it("reports the live stored value, not the event's newValue", () => {
      // A stale/inconsistent event payload must not win over storage: the value handed to
      // subscribers has to match what serverToken()/uiPreferences.js will read next.
      storeAccessToken("actually-stored");
      const seen = [];
      subscribe((next) => seen.push(next));
      window.dispatchEvent(
        new StorageEvent("storage", { key: ACCESS_TOKEN_KEY, newValue: "stale-payload" }),
      );
      expect(seen).toEqual(["actually-stored"]);
    });

    it("reacts to a whole-store clear (key === null)", () => {
      storeAccessToken("shared-token");
      const seen = [];
      subscribe((next) => seen.push(next));
      window.localStorage.clear();
      window.dispatchEvent(new StorageEvent("storage", { key: null, newValue: null }));
      expect(seen).toEqual([""]);
    });

    it("ignores storage events for unrelated keys", () => {
      const listener = vi.fn();
      subscribe(listener);
      window.localStorage.setItem("sceneworks-theme", "dark");
      window.dispatchEvent(
        new StorageEvent("storage", { key: "sceneworks-theme", newValue: "dark" }),
      );
      expect(listener).not.toHaveBeenCalled();
    });

    it("ignores a sessionStorage write of the same key from a same-origin frame", () => {
      const listener = vi.fn();
      subscribe(listener);
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: ACCESS_TOKEN_KEY,
          newValue: "from-session-storage",
          storageArea: window.sessionStorage,
        }),
      );
      expect(listener).not.toHaveBeenCalled();
    });

    it("stops notifying after unsubscribe", () => {
      const listener = vi.fn();
      const off = subscribe(listener);
      otherTabWrites("first");
      expect(listener).toHaveBeenCalledTimes(1);
      off();
      otherTabWrites("second");
      expect(listener).toHaveBeenCalledTimes(1);
    });

    // The window listener is bookkeeping the notify assertions above CANNOT see: a
    // subscriber removed from the Set stops being called either way. Spy on the window so
    // a leaked `addEventListener` — which outlives every unmount and keeps the handler's
    // closures alive for the life of the page — actually fails something.
    it("attaches one window listener for many subscribers and detaches with the last", () => {
      const add = vi.spyOn(window, "addEventListener");
      const remove = vi.spyOn(window, "removeEventListener");
      const storageAdds = () => add.mock.calls.filter((call) => call[0] === "storage").length;
      const storageRemoves = () =>
        remove.mock.calls.filter((call) => call[0] === "storage").length;

      const offFirst = subscribeAccessToken(() => {});
      expect(storageAdds()).toBe(1);
      const offSecond = subscribeAccessToken(() => {});
      // Second subscriber rides the same listener.
      expect(storageAdds()).toBe(1);

      offFirst();
      // Still one subscriber left — detaching here would go deaf with a live listener.
      expect(storageRemoves()).toBe(0);
      offSecond();
      expect(storageRemoves()).toBe(1);

      // And it re-attaches cleanly for a later subscriber (a remounted gate).
      subscribe(() => {});
      expect(storageAdds()).toBe(2);
    });

    it("keeps other subscribers alive when one throws", () => {
      const healthy = vi.fn();
      subscribe(() => {
        throw new Error("subscriber blew up");
      });
      subscribe(healthy);
      expect(() => otherTabWrites("still-delivered")).not.toThrow();
      expect(healthy).toHaveBeenCalledWith("still-delivered");
    });

    it("does not call a subscriber that a sibling removed mid-delivery", () => {
      // An unmounting hook must go quiet immediately: its setState closures are dead the
      // moment it unsubscribes, even if the notify loop is already running.
      const second = vi.fn();
      let offSecond = null;
      subscribe(() => offSecond());
      offSecond = subscribe(second);
      otherTabWrites("removed-mid-delivery");
      expect(second).not.toHaveBeenCalled();
    });

    it("does not deliver the in-flight event to a subscriber added mid-delivery", () => {
      // It wasn't listening when the change happened, and it will read current storage on
      // mount anyway — delivering here would double-apply it.
      const late = vi.fn();
      subscribe(() => subscribe(late));
      otherTabWrites("added-mid-delivery");
      expect(late).not.toHaveBeenCalled();
    });

    it("degrades gracefully when Web Storage is unavailable", () => {
      // Same promise the other three helpers make (see the tests above): no throw, and the
      // empty-token default. A subscriber must not be the one thing that blows up in a
      // private-mode / storage-disabled context.
      originalStorageDescriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
      const seen = [];
      expect(() => subscribe((next) => seen.push(next))).not.toThrow();
      Object.defineProperty(globalThis, "localStorage", {
        configurable: true,
        get() {
          throw new Error("storage disabled");
        },
      });
      expect(() =>
        window.dispatchEvent(new StorageEvent("storage", { key: ACCESS_TOKEN_KEY })),
      ).not.toThrow();
      expect(seen).toEqual([""]);
    });
  });
});
