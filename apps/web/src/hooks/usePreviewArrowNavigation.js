import { useEffect, useRef } from "react";

// Keys that already mean something to the focused element: typing, a slider/scrubber, or a
// media element's own seek. The arrow belongs to that element, not to collection navigation.
const OWNS_ARROWS_SELECTOR = 'input, textarea, select, video, audio, [contenteditable="true"], [role="slider"]';

// Left/Right arrow steps an asset viewer through its collection. Listens on the document so it
// works wherever focus sits inside the viewer (or nowhere — body), but stands down when focus is
// in another surface stacked on top (a confirm dialog, a context menu) or in an element that
// owns its arrow keys. `onPrevious` / `onNext` are null at the collection's ends.
export function usePreviewArrowNavigation({ rootRef, enabled = true, onPrevious, onNext }) {
  // Latest callbacks in a ref so the listener isn't torn down on every parent render.
  const handlers = useRef({ onPrevious, onNext });
  handlers.current = { onPrevious, onNext };

  useEffect(() => {
    if (!enabled || typeof document === "undefined") {
      return undefined;
    }
    const onKeyDown = (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      if (event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      const target = event.target;
      // Scope is the dialog holding `rootRef`, so any node inside the viewer can anchor it.
      const root = rootRef.current?.closest?.('[role="dialog"]') ?? rootRef.current;
      if (target && target !== document.body && root && !root.contains(target)) return;
      if (target?.closest?.(OWNS_ARROWS_SELECTOR)) return;
      const handler = event.key === "ArrowLeft" ? handlers.current.onPrevious : handlers.current.onNext;
      if (!handler) return;
      event.preventDefault();
      handler();
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [enabled, rootRef]);
}
