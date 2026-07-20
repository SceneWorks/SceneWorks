import { useEffect, useState } from "react";
import { apiFetch } from "../api.js";
import { serverToken } from "../credentials.js";
import { isDesktop, tauriInvoke } from "../runtime.js";

// The host's unified memory (GPU VRAM off Mac) in GB, or `null` until the probe resolves / when the
// signal is unavailable. Desktop reads the Tauri GPU probe (`get_gpu_info`); a remote LAN browser reads
// the auth-protected REST host-capabilities signal derived from the registered GPU worker (epic 4484
// story 9). Extracted verbatim from ModelManagerScreen's probe so the Models download suggestion and
// the studios' capability-aware default tier (`suggestTier`) budget against the SAME memory reading.
//
// `null` is a valid, safe value everywhere it flows: `tierFits`/`suggestTier` treat an unknown memory as
// "fits" (never withhold a tier on missing data), so the capability default leans to the highest tier
// until the reading lands — and the worker's own capability downtier (sc-10733) still clamps a
// non-explicit pick to what actually fits, so a brief high default never OOMs a constrained host.
export function useUnifiedMemoryGb() {
  const [unifiedMemoryGb, setUnifiedMemoryGb] = useState(null);
  useEffect(() => {
    let cancelled = false;
    if (isDesktop) {
      // Desktop: read unified memory straight from the Tauri GPU probe.
      tauriInvoke("get_gpu_info")
        .then((info) => {
          if (!cancelled && info && typeof info.unifiedMemoryMb === "number") {
            setUnifiedMemoryGb(info.unifiedMemoryMb / 1024);
          }
        })
        .catch(() => {});
    } else {
      // Remote LAN browser: the Tauri probe is unavailable, so read the host's memory from the
      // auth-protected REST signal (unified memory on macOS / GPU VRAM on Windows).
      apiFetch("/api/v1/host-capabilities", serverToken())
        .then((caps) => {
          if (cancelled || !caps) {
            return;
          }
          const gb = caps.unifiedMemoryGb ?? caps.gpuMemoryGb;
          if (typeof gb === "number") {
            setUnifiedMemoryGb(gb);
          }
        })
        .catch(() => {});
    }
    return () => {
      cancelled = true;
    };
  }, []);
  return unifiedMemoryGb;
}
