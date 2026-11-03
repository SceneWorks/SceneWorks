import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { getLicenseAcknowledgment, putLicenseAcknowledgment, deleteLicenseAcknowledgment } from "../api/yue2.js";
import { licenseTermsMatch, requiresLicenseAcknowledgment, writeLicenseAck } from "../licenseAcknowledgment.js";

// One React transport for the Models card and onboarding. Cache data is never
// authorization: every mount/catalog-terms/token change hydrates the server view.
export function useLicenseAcknowledgments(models, token) {
  const [acks, setAcks] = useState({});
  const [error, setError] = useState("");
  const epoch = useRef(0);
  const mutations = useRef(new Map());
  const views = useRef(new Map());
  const key = JSON.stringify(models.filter(requiresLicenseAcknowledgment).map(({ id, license, licenseUrl, licenseNotice }) => [id, license, licenseUrl, licenseNotice]));
  useLayoutEffect(() => {
    const generation = ++epoch.current;
    let live = true;
    views.current = new Map();
    setAcks({});
    setError("");
    for (const [id, license, licenseUrl, licenseNotice] of JSON.parse(key)) {
      const mutation = mutations.current.get(id);
      getLicenseAcknowledgment(id, token).then((view) => {
        if (!live) return;
        const matches = typeof view?.termsSha256 === "string" && licenseTermsMatch({ license, licenseUrl, licenseNotice }, view);
        views.current.set(id, { ...view, matches });
        // A newer user action owns the acknowledgment, but the current catalog's
        // server terms are still needed for a later explicit acceptance.
        if (mutations.current.get(id) !== mutation) return;
        if (!matches) setError("License terms changed. Refresh the model catalog and review them before accepting.");
        const acknowledged = matches && view?.acknowledged === true;
        writeLicenseAck(id, acknowledged);
        setAcks((current) => ({ ...current, [id]: acknowledged }));
      }).catch((err) => {
        if (live && mutations.current.get(id) === mutation) setError(`Could not check license acknowledgment: ${err.message}`);
      });
    }
    return () => { live = false; epoch.current = generation + 1; };
  }, [key, token]);
  const setLicenseAck = useCallback(async (id, acknowledged) => {
    const generation = epoch.current;
    // Serialize toggles per model so a slow PUT cannot undo a later withdrawal.
    const previous = mutations.current.get(id) ?? Promise.resolve();
    const view = views.current.get(id);
    const pending = previous.catch(() => {}).then(() => {
      if (epoch.current !== generation) return null;
      if (acknowledged && (!view || !view.matches)) {
        throw new Error("Refresh the model catalog and review the current license terms before accepting.");
      }
      return acknowledged ? putLicenseAcknowledgment(id, token, view.termsSha256) : deleteLicenseAcknowledgment(id, token);
    });
    mutations.current.set(id, pending);
    setAcks((current) => ({ ...current, [id]: false }));
    setError("");
    try {
      const view = await pending;
      if (epoch.current !== generation || mutations.current.get(id) !== pending) return;
      const accepted = view?.acknowledged === true;
      writeLicenseAck(id, accepted);
      setAcks((current) => ({ ...current, [id]: accepted }));
    } catch (err) {
      if (epoch.current === generation && mutations.current.get(id) === pending) {
        writeLicenseAck(id, false);
        setError(`Could not save license acknowledgment: ${err.message}`);
      }
    }
  }, [token]);
  return { licenseAcks: acks, setLicenseAck, licenseAckError: error };
}
