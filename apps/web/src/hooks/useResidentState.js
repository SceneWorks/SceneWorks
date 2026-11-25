import { useCallback, useLayoutEffect, useRef, useState } from "react";

// Catalogs belong to the visible page. Drop the stored value, rather than only
// hiding it from consumers, and refuse late mutation callbacks while inactive.
export function useResidentState(initialValue, resident = true) {
  const initial = useRef(initialValue);
  const [value, setValue] = useState(initialValue);
  const committedResident = useRef(resident);
  useLayoutEffect(() => {
    committedResident.current = resident;
    if (!resident) setValue(initial.current);
    return () => { committedResident.current = false; };
  }, [resident]);
  const update = useCallback((next) => {
    if (committedResident.current) setValue(next);
  }, []);
  return [resident ? value : initial.current, update];
}
