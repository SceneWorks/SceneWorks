import { useEffect, useState } from "react";
import { apiFetch } from "../api.js";
import { serverToken } from "../credentials.js";
import { isDesktop, tauriInvoke } from "../runtime.js";
import { hostMemoryFromCapabilities, hostMemoryFromGpuInfo } from "../hostMemory.js";

// The host's active accelerator-memory pool, typed as unified memory or dedicated VRAM, or `null`
// until the probe resolves / when the signal is unavailable. Desktop reads the Tauri GPU probe
// (`get_gpu_info`); a remote LAN browser reads the auth-protected REST host-capabilities signal
// derived from the registered GPU worker (epic 4484 story 9). All model and studio surfaces consume
// this hook so their tier and resolution gates budget against the same typed reading.
//
// `null` is a valid, safe value everywhere it flows: `tierFits`/`suggestTier` treat an unknown memory as
// "fits" (never withhold a tier on missing data), so the capability default leans to the highest tier
// until the reading lands — and the worker's own capability downtier (sc-10733) still clamps a
// non-explicit pick to what actually fits, so a brief high default never OOMs a constrained host.
export function useHostMemory() {
  const [hostMemory, setHostMemory] = useState(null);
  useEffect(() => {
    let cancelled = false;
    if (isDesktop) {
      // Desktop: read the typed accelerator-memory pool from the Tauri GPU probe.
      tauriInvoke("get_gpu_info")
        .then((info) => {
          if (!cancelled) {
            setHostMemory(hostMemoryFromGpuInfo(info));
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
          setHostMemory(hostMemoryFromCapabilities(caps));
        })
        .catch(() => {});
    }
    return () => {
      cancelled = true;
    };
  }, []);
  return hostMemory;
}
