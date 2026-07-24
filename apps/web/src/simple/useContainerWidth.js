import { useCallback, useEffect, useRef, useState } from "react";

// Measure an element's own width (design handoff: "measure the mount, not `window` — it
// must work embedded"). Returns `[ref, width]`; `width` is null until the first
// observation lands, which `breakpointFor` resolves to the desktop band.
//
// The 4px dead-band mirrors the prototype: a sub-pixel scrollbar/zoom jitter must not
// re-render the whole shell, and the three breakpoints are hundreds of pixels apart so a
// 4px hysteresis can never straddle one.
const WIDTH_EPSILON = 4;

export function useContainerWidth() {
  const [width, setWidth] = useState(null);
  const observerRef = useRef(null);
  const widthRef = useRef(null);

  const ref = useCallback((element) => {
    observerRef.current?.disconnect();
    observerRef.current = null;
    if (!element || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver((entries) => {
      const next = Math.round(entries[0]?.contentRect?.width ?? 0);
      if (!next) {
        return;
      }
      if (widthRef.current !== null && Math.abs(next - widthRef.current) <= WIDTH_EPSILON) {
        return;
      }
      widthRef.current = next;
      setWidth(next);
    });
    observer.observe(element);
    observerRef.current = observer;
  }, []);

  useEffect(
    () => () => {
      observerRef.current?.disconnect();
      observerRef.current = null;
    },
    [],
  );

  return [ref, width];
}

// Viewport width for the app-level "phones are always served the Simple UI" rule.
// This one IS window-scoped on purpose: which SHELL to render is a property of the
// device the app is running on, not of a container inside it. The shell's own layout
// still measures its mount (useContainerWidth above), so an embedded Simple UI lays
// itself out correctly regardless of what this reports.
export function useViewportWidth() {
  const [width, setWidth] = useState(() =>
    typeof window === "undefined" ? null : window.innerWidth,
  );
  useEffect(() => {
    if (typeof window === "undefined") {
      return undefined;
    }
    const onResize = () => setWidth(window.innerWidth);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return width;
}
