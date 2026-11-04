import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { expect, it, vi } from "vitest";
import { setMediaTicket } from "./api.js";
import { useAudioPeaks } from "./audioWaveform.js";

it("retries an unauthorized shared waveform once after the ticket changes", async () => {
  global.IS_REACT_ACT_ENVIRONMENT = true;
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: false, status: 401 })
    .mockResolvedValueOnce({ ok: true, arrayBuffer: async () => new ArrayBuffer(4) });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("AudioContext", class {
    async decodeAudioData() { return { getChannelData: () => new Float32Array([0.1, 0.5]) }; }
    close() {}
  });
  const asset = { id: "ticket-waveform", type: "audio", projectId: "p", file: { path: "a.wav", mimeType: "audio/wav" } };
  function View() { const peaks = useAudioPeaks(asset); return <span>{peaks ? "ready" : "pending"}</span>; }
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  try {
    setMediaTicket("old");
    await act(async () => root.render(<View />));
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await act(async () => setMediaTicket("new"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[1][0]).toContain("ticket=new");
    expect(container.textContent).toBe("ready");
    await act(async () => setMediaTicket("later"));
    expect(fetchMock).toHaveBeenCalledTimes(2);
  } finally {
    await act(async () => root.unmount());
    container.remove();
    setMediaTicket("");
    vi.unstubAllGlobals();
  }
});
