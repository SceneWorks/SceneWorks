const previewGraphs = new WeakMap();

function browserAudioContext() {
  return globalThis.AudioContext ?? globalThis.webkitAudioContext ?? null;
}

function releaseGraph(element, graph) {
  graph.references -= 1;
  if (graph.references > 0) return;

  const releaseToken = {};
  graph.releaseToken = releaseToken;
  queueMicrotask(() => {
    if (graph.references > 0 || graph.releaseToken !== releaseToken) return;
    graph.source.disconnect();
    graph.gain.disconnect();
    void graph.context.close();
    // A media element can never be bound to a second MediaElementSourceNode,
    // even after its first AudioContext closes. Keep a weak tombstone so an
    // accidentally reinserted DOM node fails closed instead of throwing while
    // a genuinely new <audio> element receives a new graph.
    graph.closed = true;
  });
}

function leaseGraph(element, graph) {
  if (graph) {
    graph.references += 1;
    graph.releaseToken = null;
  }

  let released = false;
  return {
    supported: Boolean(graph && !graph.closed),
    resume() {
      return graph && graph.context.state !== "running" ? graph.context.resume() : Promise.resolve();
    },
    setGain(value, muted = false) {
      const gain = Math.max(0, Number(value) || 0);
      if (graph && !graph.closed) {
        // The element stays at unity and unmuted. Muting and all gain, including
        // above-unity gain, live in the one Web Audio node.
        element.volume = 1;
        element.muted = false;
        graph.gain.gain.value = muted ? 0 : gain;
        return true;
      }

      if (gain > 1 && !muted) {
        // Do not audition a boosted edit at a misleading clamped level when the
        // browser has no supported above-unity gain path.
        element.volume = 1;
        element.muted = true;
        return false;
      }
      element.volume = Math.min(1, gain);
      element.muted = Boolean(muted);
      return true;
    },
    release() {
      if (released) return;
      released = true;
      if (graph && !graph.closed) releaseGraph(element, graph);
    },
  };
}

export function reacquireAudioPreviewGain(element) {
  const graph = previewGraphs.get(element);
  return graph && !graph.closed ? leaseGraph(element, graph) : null;
}

/**
 * Route one media element through a reusable Web Audio gain node.
 *
 * MediaElementAudioSourceNode may only be created once for an element. The
 * shared lease keeps React Strict Mode's release/reacquire cycle from creating
 * a duplicate node, while still closing the AudioContext after the final owner
 * releases it.
 */
export function acquireAudioPreviewGain(element, AudioContextConstructor = browserAudioContext()) {
  let graph = previewGraphs.get(element);
  if (graph?.closed) return leaseGraph(element, null);
  if (!graph && AudioContextConstructor) {
    const context = new AudioContextConstructor();
    const source = context.createMediaElementSource(element);
    const gain = context.createGain();
    source.connect(gain);
    gain.connect(context.destination);
    graph = { closed: false, context, gain, references: 0, releaseToken: null, source };
    previewGraphs.set(element, graph);
  }
  return leaseGraph(element, graph);
}
