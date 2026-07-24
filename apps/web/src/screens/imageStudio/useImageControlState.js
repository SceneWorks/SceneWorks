import { useState } from "react";

export function useStrictControlState(saved) {
  const [controlMode, setControlMode] = useState(saved.controlMode ?? "pose");
  const [controlImageAssetId, setControlImageAssetId] = useState("");
  const [controlImagePassthrough, setControlImagePassthrough] = useState(false);
  const [controlScale, setControlScale] = useState(saved.controlScale ?? null);
  const [controlOverlayId, setControlOverlayId] = useState(null);
  return {
    controlMode, setControlMode, controlImageAssetId, setControlImageAssetId,
    controlImagePassthrough, setControlImagePassthrough, controlScale,
    setControlScale, controlOverlayId, setControlOverlayId,
  };
}

export function usePidState(saved) {
  const [usePid, setUsePid] = useState(saved.usePid ?? false);
  const [pidTarget, setPidTarget] = useState(saved.pidTarget === "2k" ? "2k" : "4k");
  return { usePid, setUsePid, pidTarget, setPidTarget };
}
