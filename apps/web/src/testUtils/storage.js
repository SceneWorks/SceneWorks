// Shared Web Storage fixtures for the access-token suites (sc-15223).
//
// `accessToken.js` promises to degrade through three distinct storage postures, and the
// difference between them is exactly what the sc-15223 contract turns on — so a suite that
// hand-rolls "private mode" can quietly test a different posture than it thinks:
//
//   - UNAVAILABLE  — the `window.localStorage` getter itself throws. Reads yield nothing to
//                    adopt, so the live value stands.
//   - WRITE-REJECTING — reads and removes work, `setItem` throws (WebKit private mode, an
//                    over-quota origin). Reads answer null-for-absent, which is a real
//                    answer. This is the posture sc-15223 is about.
//   - HOSTILE      — every operation throws.
//
// Each installer returns a restore function; callers must invoke it in `afterEach`, because
// a leaked descriptor breaks every later test in the file.

function install(value) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { configurable: true, ...value });
  return function restore() {
    if (original) {
      Object.defineProperty(globalThis, "localStorage", original);
    } else {
      delete globalThis.localStorage;
    }
  };
}

// A store that reads and clears normally but silently refuses to persist. Backed by a real
// Map so `getItem`/`removeItem` behave — if reads were broken too this would degenerate into
// `installUnavailableStorage` and stop covering what it claims.
export function installWriteRejectingStorage() {
  const backing = new Map();
  return install({
    value: {
      getItem: (key) => (backing.has(key) ? backing.get(key) : null),
      setItem: () => {
        throw new Error("QuotaExceededError");
      },
      removeItem: (key) => backing.delete(key),
      clear: () => backing.clear(),
    },
  });
}

// Storage disabled outright: touching `window.localStorage` throws.
export function installUnavailableStorage() {
  return install({
    get() {
      throw new Error("storage disabled");
    },
  });
}
