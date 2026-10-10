import { timelineAudioState } from "./timelineAudio.js";

// One graph for the assembled cut, created/resumed synchronously by the transport
// gesture. Each DOM media element is bound once, including across Strict Mode.
export function createTimelineAudioPlayback(onError) {
  const layers = new Map();
  const graphs = new WeakMap();
  let context, output, playing = false, time = 0, disposed = false;
  let generation = 0;

  function graphFor(element) {
    let graph = graphs.get(element);
    if (!graph && context) {
      const source = context.createMediaElementSource(element);
      const gain = context.createGain();
      graph = { source, gain, connected: false };
      graphs.set(element, graph);
    }
    if (graph && !graph.connected) {
      graph.source.connect(graph.gain);
      graph.gain.connect(output);
      graph.connected = true;
    }
    return graph;
  }
  function apply(layer, seek = false) {
    const { element, placement } = layer;
    const state = timelineAudioState(placement, time);
    const graph = graphFor(element);
    element.playbackRate = placement.speed;
    element.preservesPitch = true;
    if (seek || !playing || !state.active || Math.abs(element.currentTime - state.currentTime) > 0.12) {
      element.currentTime = state.currentTime;
    }
    const gain = playing ? state.gain : 0;
    if (graph) {
      element.volume = 1;
      element.muted = false;
      graph.gain.gain.cancelScheduledValues(context.currentTime);
      graph.gain.gain.value = gain;
      graph.gain.gain.setValueAtTime(gain, context.currentTime);
      if (playing && state.active) {
        // The audio clock closes the gate even if a long render delays rAF.
        const remaining = Math.min(placement.start + placement.span - time, (placement.sourceOut - state.currentTime) / placement.speed);
        graph.gain.gain.setValueAtTime(0, context.currentTime + remaining);
      }
    } else {
      element.volume = Math.min(1, gain);
      element.muted = gain === 0;
    }
    return state;
  }
  function fail(error, token) {
    if (disposed || token !== generation || error?.name === "AbortError") return;
    api.pause();
    onError(error);
  }
  function play(layer) {
    if (layer.pending || (!layer.element.paused && !layer.element.ended)) return;
    const token = generation;
    layer.pending = true;
    Promise.resolve(layer.element.play()).then(() => {
      layer.pending = false;
      if (!playing || !timelineAudioState(layer.placement, time).active) layer.element.pause();
    }, (error) => { layer.pending = false; fail(error, token); });
  }
  const api = {
    attach(key, element, placement) {
      const old = layers.get(key);
      if (old?.element === element) { old.placement = placement; return; }
      if (old) {
        old.element.pause();
        const graph = graphs.get(old.element);
        if (graph) {
          graph.gain.gain.cancelScheduledValues(context.currentTime);
          graph.gain.gain.value = 0;
          graph.source.disconnect();
          graph.gain.disconnect();
          graph.connected = false;
        }
        layers.delete(key);
      }
      if (element) {
        const layer = { element, placement, pending: false };
        layers.set(key, layer);
        apply(layer, true);
      }
    },
    mediaError(key) {
      const layer = layers.get(key);
      if (!layer) return;
      api.pause();
      onError(new Error(`Cannot load ${layer.placement.item.displayName ?? layer.placement.assetId}.`));
    },
    update(placements, nextTime, nextPlaying, seek = false) {
      time = nextTime;
      playing = nextPlaying;
      for (const placement of placements) {
        const layer = layers.get(placement.key);
        if (!layer) continue;
        layer.placement = placement;
        const state = apply(layer, seek);
        if (playing && state.active) play(layer);
        else layer.element.pause();
      }
    },
    start(nextTime) {
      generation += 1;
      time = nextTime;
      playing = true;
      const token = generation;
      try {
        if (layers.size && !context) {
          const Constructor = globalThis.AudioContext ?? globalThis.webkitAudioContext;
          if (!Constructor) throw new Error("This browser does not support timeline audio mixing.");
          context = new Constructor();
          // Preserve the sum and relative layer gains. A final peak guard avoids
          // clipping; browser dynamics and FFmpeg's lookahead limiter are not
          // sample-identical. Below the ceiling neither changes the layer gains.
          output = context.createDynamicsCompressor();
          output.threshold.value = 20 * Math.log10(0.98);
          output.knee.value = 0;
          output.ratio.value = 20;
          output.attack.value = 0.005;
          output.release.value = 0.05;
          output.connect(context.destination);
        }
        if (context) Promise.resolve(context.resume()).catch((error) => fail(error, token));
        // Unlock every element in this SAME gesture, including later placements.
        // They start at zero gain and pause once play resolves until their slot.
        for (const layer of layers.values()) { apply(layer, true); play(layer); }
        return true;
      } catch (error) { fail(error, token); return false; }
    },
    pause() {
      playing = false;
      generation += 1;
      for (const layer of layers.values()) { apply(layer); layer.element.pause(); }
    },
    dispose() {
      if (disposed) return;
      api.pause();
      disposed = true;
      for (const layer of layers.values()) {
        const graph = graphs.get(layer.element);
        graph?.source.disconnect();
        graph?.gain.disconnect();
      }
      layers.clear();
      output?.disconnect();
      void context?.close();
    },
  };
  return api;
}
