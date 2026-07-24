import { createContext, useContext } from "react";

// Shell services the Simple screens use: the measured layout band plus the four
// shell-owned overlays (picker sheet, prompt guide, asset preview, toast) and
// navigation. Kept as its own tiny context — separate from the app-wide
// AppStaticContext/AppLiveContext — so a Simple screen reads shell chrome from
// here and real app data from there, with no overlap.
//
// Shape:
//   breakpoint  "phone" | "tablet" | "desktop"
//   goTo(screenId)
//   openSheet({ title, kind, options, onSelect })  — kind: "list" | "grid" | "pills"
//   closeSheet()
//   openGuide()
//   openPreview(asset)
//   toast(message)
//   openInAdvanced(view) — hand off to a full-workspace screen (returns false, with an
//                          explanatory toast, when the viewport has Simple locked on)
//   uiLocked            — true when the viewport forces Simple
//   dismissedJobIds     — Set of failed-run ids the user has dismissed from a studio.
//                         Shell-held, because the studios unmount on navigation.
//   dismissJob(id)
export const SimpleUiContext = createContext(null);

export function useSimpleUi() {
  const value = useContext(SimpleUiContext);
  if (!value) {
    throw new Error("useSimpleUi must be used within the Simple UI shell");
  }
  return value;
}
