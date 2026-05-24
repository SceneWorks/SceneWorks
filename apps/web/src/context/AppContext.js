import { createContext, useContext } from "react";

// Shared app data/actions provider (sc-1651 Phase B). App composes the per-domain
// hooks (Phase A) and exposes the primitives here so screens read what they need via
// useAppContext() instead of receiving dozens of drilled props. Screens build any
// screen-specific wrappers (e.g. send-to-studio with a mode) from these primitives.
export const AppContext = createContext(null);

export function useAppContext() {
  const value = useContext(AppContext);
  if (value === null) {
    throw new Error("useAppContext must be used within an <AppContext.Provider>");
  }
  return value;
}
