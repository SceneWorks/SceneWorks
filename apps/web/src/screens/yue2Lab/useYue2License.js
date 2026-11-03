import { useCallback, useEffect, useState } from "react";
import { licenseTermsMatch, writeLicenseAck } from "../../licenseAcknowledgment.js";
import { deleteLicenseAcknowledgment, getLicenseAcknowledgment, putLicenseAcknowledgment } from "../../api/yue2.js";

// Extracted from Yue2SongLab along its existing concern boundary.
export function useYue2License({
  model,
  token,
  settings,
  update
}) {
  // Server-side licence acknowledgment (sc-22999): the lab reads the server's record, never a
  // browser flag, so the acceptance survives a relaunch and lapses when the terms change.
  const [ack, setAck] = useState({ status: "loading", acknowledged: false, error: null });

  const [ackBusy, setAckBusy] = useState(false);

  const [gateError, setGateError] = useState(null);

  // Keyed on the model id and its licence TERMS, not the catalog object: a catalog refresh hands
  // back a new object for the same entry, and re-reading on identity would reset the lab to
  // "checking" on every refresh. Changed terms do re-read (the server lapses the old acceptance).
  const modelId = model?.id ?? null;

  const termsKey = model ? JSON.stringify([model.license, model.licenseUrl, model.licenseNotice]) : "";

  const loadAck = useCallback(() => {
    if (!modelId) return undefined;
    let live = true;
    setAck((current) => ({ ...current, status: "loading", error: null }));
    getLicenseAcknowledgment(modelId, token)
      .then((view) => {
        if (!live) return;
        if (!licenseTermsMatch(model, view)) throw new Error("License terms changed. Refresh the model catalog and review them before accepting.");
        setAck({ status: "ready", acknowledged: view?.acknowledged === true, error: null, view });
        // Mirror the server's record into the download choke point's cache so an install from
        // here (or the Models card) is not refused for a stale browser flag.
        // …and clear it when the server says the acceptance lapsed or was withdrawn.
        writeLicenseAck(modelId, view?.acknowledged === true);
      })
      .catch((error) => {
        if (live) setAck({ status: "error", acknowledged: false, error });
      });
    return () => {
      live = false;
    };
    // termsKey re-reads the acknowledgment when the licence terms change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelId, termsKey, token]);

  useEffect(() => loadAck(), [loadAck]);

  const enabled = settings.optIn && ack.status === "ready" && ack.acknowledged;

  async function accept() {
    setAckBusy(true);
    setGateError(null);
    try {
      if (!ack.acknowledged) {
        const view = await putLicenseAcknowledgment(model.id, token, ack.view?.termsSha256);
        setAck({ status: "ready", acknowledged: view?.acknowledged === true, error: null, view });
      }
      writeLicenseAck(model.id, true);
      update({ optIn: true });
    } catch (error) {
      setGateError(error);
    } finally {
      setAckBusy(false);
    }
  }

  async function withdraw() {
    setGateError(null);
    try {
      await deleteLicenseAcknowledgment(model.id, token);
      writeLicenseAck(model.id, false);
      setAck({ status: "ready", acknowledged: false, error: null });
      update({ optIn: false });
    } catch (error) {
      setGateError(error);
    }
  }
  return { ack, setAck, ackBusy, gateError, loadAck, enabled, accept, withdraw };
}
