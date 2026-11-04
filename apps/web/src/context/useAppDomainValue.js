import { useMemo } from "react";

// Domain shapes are fixed at their owner. Including every own key AND value keeps
// context identity stable without a second, hand-maintained dependency list.
// This is a render-local memo: suspended renders cannot publish their actions.
export function useAppDomainValue(value) {
  const dependencies = Object.entries(value).flat();
  // Every captured value is represented by the fixed-shape object's entries above.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  return useMemo(() => value, dependencies);
}
