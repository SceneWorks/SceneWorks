import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  ACCESS_TOKEN_KEY,
  clearAccessToken,
  readAccessToken,
  resetAccessTokenForTests,
  storeAccessToken,
  subscribeAccessToken,
} from "./accessToken.js";
import { installUnavailableStorage, installWriteRejectingStorage } from "./testUtils/storage.js";

describe("accessToken (sc-8880)", () => {
  beforeEach(() => {
    // sc-15223: the live token is module state that outlives a test. Reset it so each test
    // starts from the fresh-page-load state (nothing in memory, read through to storage).
    resetAccessTokenForTests();
  });

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
    // No throw from any helper, and nothing to read before a write.
    expect(readAccessToken()).toBe("");
    // sc-15223: the token still goes live even though it cannot be persisted. It used to
    // read back as "" here — the divergence that left the gate open while every
    // `readAccessToken()` caller sent an empty credential.
    expect(() => storeAccessToken("x")).not.toThrow();
    expect(readAccessToken()).toBe("x");
    expect(() => clearAccessToken()).not.toThrow();
    expect(readAccessToken()).toBe("");
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
    // Same sc-15223 contract with the store present but hostile on every operation.
    expect(readAccessToken()).toBe("x");
    expect(() => clearAccessToken()).not.toThrow();
    expect(readAccessToken()).toBe("");
  });

  // sc-15223: `localStorage` is the PERSISTENCE CACHE, not the source. A write the browser
  // refuses must not be able to desync the session — the same-tab remainder of the sc-15165
  // divergence, and the one no `storage` event can ever repair.
  describe("live value vs persistence cache (sc-15223)", () => {
    // One slot per installed fixture rather than a shared variable: a test that installs
    // twice would otherwise overwrite the real descriptor with a broken one and leak it into
    // every later test in the file.
    const restores = [];
    function rejectWrites() {
      restores.push(installWriteRejectingStorage());
    }

    afterEach(() => {
      while (restores.length) {
        restores.pop()();
      }
    });

    it("serves the live token when the store refused to persist it", () => {
      rejectWrites();
      storeAccessToken("private-mode-password");
      // The store really is empty — this is what made the old read return "".
      expect(window.localStorage.getItem(ACCESS_TOKEN_KEY)).toBeNull();
      expect(readAccessToken()).toBe("private-mode-password");
    });

    it("forgets a live token the store never held", () => {
      rejectWrites();
      storeAccessToken("private-mode-password");
      clearAccessToken();
      // The higher-consequence direction: `removeItem` on an empty store is a no-op, so the
      // in-memory reset is the ONLY thing that can stop a forgotten password being sent.
      expect(readAccessToken()).toBe("");
    });

    it("distinguishes 'this session forgot it' from 'this session never set one'", () => {
      // A cleared session must NOT fall back to whatever storage still holds — otherwise a
      // "lock" that could not reach the store would resurrect the password on the next read.
      window.localStorage.setItem(ACCESS_TOKEN_KEY, "left-behind");
      clearAccessToken();
      window.localStorage.setItem(ACCESS_TOKEN_KEY, "left-behind");
      expect(readAccessToken()).toBe("");

      // ...while a session that has set nothing DOES read through, which is the reload path
      // the persistence cache exists for.
      resetAccessTokenForTests();
      expect(readAccessToken()).toBe("left-behind");
    });

    it("lets a cross-tab change overwrite the live value (sc-15165 still wins)", () => {
      // Pinning the live value would make the tab deaf to every `storage` event, undoing the
      // fix this one builds on. A test that only asserted the subscriber's argument would
      // miss it — `readAccessToken()` is what the other callers use.
      storeAccessToken("mine");
      const seen = [];
      const off = subscribeAccessToken((next) => seen.push(next));
      window.localStorage.setItem(ACCESS_TOKEN_KEY, "theirs");
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: ACCESS_TOKEN_KEY,
          newValue: "theirs",
          storageArea: window.localStorage,
        }),
      );
      off();

      expect(seen).toEqual(["theirs"]);
      expect(readAccessToken()).toBe("theirs");
    });

    it("adopts a peer tab's value over a live token the store itself never held", () => {
      // The write-rejecting posture does NOT make this tab's live value untouchable. A real
      // `storage` event means a peer DID write the shared store, so its value outranks ours —
      // pinning the live token here would make a private-mode tab deaf to sc-15165 forever.
      // (This is the case the `stored !== null` guard below does not cover, on purpose:
      // `getItem` works here and answers null-for-absent, which is a real answer.)
      rejectWrites();
      storeAccessToken("never-persisted");
      const seen = [];
      const off = subscribeAccessToken((next) => seen.push(next));
      // No `storageArea`: the installed store is a plain object, which jsdom refuses to
      // accept as one. The handler lets a `storageArea`-less event through by design (real
      // browsers always set it), which is the same allowance the sc-15165 tests above rely on.
      window.dispatchEvent(new StorageEvent("storage", { key: null }));
      off();

      expect(seen).toEqual([""]);
      expect(readAccessToken()).toBe("");
    });

    it("does not let an UNREADABLE store phantom-clear the live token", () => {
      // The narrow case the guard is actually for: no browser dispatches a `storage` event
      // while our own store is unreachable, so this is a synthetic/misdelivered one carrying
      // no truth to adopt. Treating that as "" would lock a tab whose in-memory value is the
      // only copy of the credential in existence.
      storeAccessToken("only-copy");
      const seen = [];
      const off = subscribeAccessToken((next) => seen.push(next));
      restores.push(installUnavailableStorage());
      window.dispatchEvent(new StorageEvent("storage", { key: ACCESS_TOKEN_KEY }));
      off();

      expect(seen).toEqual(["only-copy"]);
      expect(readAccessToken()).toBe("only-copy");
    });

    it("treats a nullish store argument as an empty token, not as 'unset'", () => {
      // `null`/`undefined` is the module's "no helper has run this session" sentinel, so an
      // uncoerced nullish store would silently mean "read through to storage" — the opposite
      // of what storing a value means.
      window.localStorage.setItem(ACCESS_TOKEN_KEY, "left-behind");
      storeAccessToken(null);
      expect(readAccessToken()).toBe("");
    });
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

    it("ignores a whole-store sessionStorage clear", () => {
      // The higher-consequence half of the same guard: `key === null` is the branch that
      // CLEARS the gate, so a same-origin frame calling sessionStorage.clear() must not be
      // able to lock this tab out.
      storeAccessToken("shared-token");
      const listener = vi.fn();
      subscribe(listener);
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: null,
          newValue: null,
          storageArea: window.sessionStorage,
        }),
      );
      expect(listener).not.toHaveBeenCalled();
      expect(readAccessToken()).toBe("shared-token");
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
      Object.defineProperty(globalThis, "localStorage", {
        configurable: true,
        get() {
          throw new Error("storage disabled");
        },
      });
      // Subscribe AFTER breaking storage, so this covers what it claims to.
      const seen = [];
      expect(() => subscribe((next) => seen.push(next))).not.toThrow();
      expect(() =>
        window.dispatchEvent(new StorageEvent("storage", { key: ACCESS_TOKEN_KEY })),
      ).not.toThrow();
      // The empty-token default, delivered rather than thrown.
      expect(seen).toEqual([""]);
    });
  });
});
