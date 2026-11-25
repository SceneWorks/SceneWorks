import { createContext, useContext } from "react";

// Explicit visibility override for nested consumers and isolated renderers.
// Top-level pages now unmount on navigation; a mounted page defaults to active.
export const ScreenActiveContext = createContext(true);

// Read an explicit visibility override, defaulting to active.
export function useScreenActive() {
  return useContext(ScreenActiveContext);
}
