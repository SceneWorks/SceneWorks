import { useEffect } from "react";

// Swallows file drops that no in-app dropzone claimed (issue #1308).
//
// The desktop shell runs the webview with Tauri's `dragDropEnabled: false`
// (apps/desktop/tauri.conf.json) so the real dropzones — AssetPicker, the
// dataset upload dialog, the Image Editor canvas — receive genuine
// `dataTransfer.files`. The tradeoff is that a file dropped *anywhere else*
// falls through to the webview's default handler, which navigates to the file
// and replaces the whole UI with the image. A plain browser tab behaves the
// same way. This installs a window-level fallback that accepts and then
// discards those stray drags so nothing navigates.
//
// Real dropzones keep working untouched: each already calls preventDefault()
// on its own `dragover`/`drop` as the event bubbles through it, so by the time
// the event reaches this window listener `defaultPrevented` is already true and
// the guard bows out. It only acts on drags nothing else claimed, where it
// prevents the default navigation and marks the drop as "not allowed" so the
// cursor reads as a rejection rather than a copy affordance.
export function useDropNavigationGuard() {
  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }
    const swallowStrayDrag = (event) => {
      if (event.defaultPrevented) {
        // A registered dropzone already claimed this drag; leave it alone.
        return;
      }
      // Preventing the default on `dragover` is what makes the window a valid
      // drop target, so the `drop` event fires here (instead of navigating);
      // preventing it on `drop` is what stops the navigation itself.
      event.preventDefault();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "none";
      }
    };
    window.addEventListener("dragover", swallowStrayDrag);
    window.addEventListener("drop", swallowStrayDrag);
    return () => {
      window.removeEventListener("dragover", swallowStrayDrag);
      window.removeEventListener("drop", swallowStrayDrag);
    };
  }, []);
}
