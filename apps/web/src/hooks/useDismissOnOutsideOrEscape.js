import { useEffect, useRef } from "react";

export function useDismissOnOutsideOrEscape(ref, open, setOpen) {
  const closeRef = useRef(setOpen);
  closeRef.current = setOpen;
  useEffect(() => {
    if (!open) return undefined;
    const onMouseDown = (event) => {
      if (!ref.current?.contains(event.target)) closeRef.current(false);
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") closeRef.current(false);
    };
    document.addEventListener("mousedown", onMouseDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onMouseDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [ref, open]);
}
