import { useCallback, useLayoutEffect, useRef } from "react";

import { acquireAudioPreviewGain, reacquireAudioPreviewGain } from "./audioPreviewGain.js";

export function useAudioPreviewGain(externalMediaRef, { gain = 1, muted = false } = {}) {
  const leaseRef = useRef(null);
  const stateRef = useRef({ gain, muted });
  stateRef.current = { gain, muted };

  const mediaRef = useCallback((element) => {
    leaseRef.current?.release();
    leaseRef.current = null;
    externalMediaRef.current = element;
    // Callback refs can detach and immediately reattach the same live element
    // (including Strict Mode). Reclaim an existing graph without constructing
    // an AudioContext outside a user gesture.
    if (element?.tagName === "AUDIO") {
      leaseRef.current = reacquireAudioPreviewGain(element);
      leaseRef.current?.setGain(stateRef.current.gain, stateRef.current.muted);
    }
  }, [externalMediaRef]);

  useLayoutEffect(() => {
    leaseRef.current?.setGain(gain, muted);
  }, [gain, muted]);

  const prepareForPlayback = useCallback(() => {
    const element = externalMediaRef.current;
    if (!leaseRef.current && element?.tagName === "AUDIO") {
      // Construct the AudioContext from the user's pointer/keyboard handler.
      // Safari and Chromium can otherwise leave an eagerly-created context
      // suspended even after the media element starts.
      leaseRef.current = acquireAudioPreviewGain(element);
    }
    const lease = leaseRef.current;
    if (!lease || !lease.setGain(stateRef.current.gain, stateRef.current.muted)) {
      return { ok: false, resume: Promise.resolve() };
    }
    // Called from the transport's pointer/keyboard handler so AudioContext.resume
    // receives the browser's user activation.
    return { ok: true, resume: lease.resume() };
  }, [externalMediaRef]);

  return { mediaRef, prepareForPlayback };
}
