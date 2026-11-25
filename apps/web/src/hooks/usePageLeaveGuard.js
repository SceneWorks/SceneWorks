import { useLayoutEffect, useRef } from "react";
import { useAppStatic } from "../context/AppContext.js";

// Only the foreground page is mounted. Editors must resolve unsaved work before
// navigation releases their document, canvas and undo history.
export function usePageLeaveGuard(guard) {
  const { registerLeaveGuard } = useAppStatic();
  const guardRef = useRef(guard);
  useLayoutEffect(() => { guardRef.current = guard; });
  useLayoutEffect(() => registerLeaveGuard?.(() => guardRef.current()), [registerLeaveGuard]);
}
