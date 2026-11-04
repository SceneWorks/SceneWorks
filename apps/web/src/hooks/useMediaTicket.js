import { useSyncExternalStore } from "react";
import { getMediaTicket, subscribeMediaTicket } from "../api.js";

// Subscribe at the component that constructs an element-driven media URL.
export function useMediaTicket() {
  return useSyncExternalStore(subscribeMediaTicket, getMediaTicket, getMediaTicket);
}
