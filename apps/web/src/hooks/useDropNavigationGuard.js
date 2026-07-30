import { useEffect, useRef } from "react";

// Swallows file drops that no in-app dropzone claimed (issue #1308), and — since
// sc-15951 — hands that unclaimed file to an optional inspector first.
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
//
// `onUnclaimedFileDrop` (sc-15951) rides on exactly that "nothing else claimed it"
// branch — the same `defaultPrevented` test, unchanged, is what decides whether it is
// called at all, so a drop one of the five real targets handles can never reach it.
// It is invoked only for a drop carrying EXACTLY one file: a bare-text or URL drag
// carries none, and a multi-file drag is a bulk gesture no single-recipe offer could
// answer honestly. Both keep today's silent swallow. Whatever the handler does is
// asynchronous and cannot affect the swallow, which has already happened by the time it
// runs — so a slow or throwing inspector can never let the webview navigate.
export function useDropNavigationGuard({ onUnclaimedFileDrop } = {}) {
  // Held in a ref, refreshed post-commit, so the window listeners below subscribe once
  // for the app's lifetime rather than tearing down and re-attaching on every render of
  // the component that owns the handler. A drag in flight across such a re-subscribe
  // would otherwise land on no listener at all and navigate.
  const handlerRef = useRef(null);
  useEffect(() => {
    handlerRef.current = typeof onUnclaimedFileDrop === "function" ? onUnclaimedFileDrop : null;
  });

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
      if (event.type !== "drop") {
        return;
      }
      // Read synchronously: a `DataTransfer` is neutered once the handler returns, so the
      // `File` has to be taken out of it here even though the inspector is async.
      const files = event.dataTransfer?.files;
      if (files?.length === 1) {
        handlerRef.current?.(files[0]);
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
