import { useCallback, useLayoutEffect, useRef } from "react";

// A consent lookup is permission for one committed caller/catalog/session only.
// A late server answer cannot submit or update UI after that owner changes.
export function useConsentRequestOwner(token, models, loras, scope) {
  const ownerRef = useRef(null);
  useLayoutEffect(() => {
    const owner = { token, live: true };
    ownerRef.current = owner;
    return () => { owner.live = false; };
  }, [token, models, loras, scope]);
  return useCallback((expectedToken) => {
    const owner = ownerRef.current;
    return () => owner?.live === true && owner.token === expectedToken && ownerRef.current === owner;
  }, []);
}
