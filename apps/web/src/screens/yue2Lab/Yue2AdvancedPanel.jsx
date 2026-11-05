import { SamplingFields, TriStateSelect } from "./LabFields.jsx";
import React from "react";
import { AdvancedSection } from "../../components/AdvancedSection.jsx";
import { MAX_DECODE_TILE_FRAMES, MIN_ATTENTION_CHUNK_ELEMENTS, SAMPLING_TOKEN_DEFAULTS } from "../../yue2Lab.js";

export function Yue2AdvancedPanel({
  update,
  settings,
  why
}) {
  return (
<AdvancedSection
              hint="Sampling, guidance, seed, synthesis steps, AR mode, compute precision and memory"
              onToggle={() => update({ advancedOpen: !settings.advancedOpen })}
              open={settings.advancedOpen}
            >
              <div className="yue2-grid">
                <label>
                  Seed
                  <input aria-label="Seed" disabled={Boolean(why("seed"))} title={why("seed") ?? undefined} onChange={(event) => update({ seed: event.target.value })} placeholder="Random" step="1" type="number" value={settings.seed} />
                </label>
                <label>
                  Guidance (CFG)
                  <input aria-label="Guidance" disabled={Boolean(why("cfgScale"))} title={why("cfgScale") ?? undefined} onChange={(event) => update({ cfgScale: event.target.value })} placeholder="Model default" step="0.1" type="number" value={settings.cfgScale} />
                </label>
                <label>
                  Acoustic ODE steps
                  <input aria-label="ODE steps" disabled={Boolean(why("steps"))} title={why("steps") ?? undefined} max="10000" min="1" onChange={(event) => update({ steps: event.target.value })} placeholder="Model default" step="1" type="number" value={settings.steps} />
                </label>
                <label>
                  AR mode
                  <select aria-label="AR mode" disabled={Boolean(why("arMode"))} title={why("arMode") ?? undefined} onChange={(event) => update({ arMode: event.target.value })} value={settings.arMode}>
                    <option value="">Native (default)</option>
                    <option value="experimentalFp8">Experimental FP8 AR (CUDA sm_89+, bf16 only)</option>
                  </select>
                </label>
                <label>
                  Offload
                  <select aria-label="Offload" disabled={Boolean(why("offloadPolicy"))} title={why("offloadPolicy") ?? undefined} onChange={(event) => update({ offloadPolicy: event.target.value })} value={settings.offloadPolicy}>
                    <option value="">Default (resident)</option>
                    <option value="resident">Resident</option>
                    <option value="sequential">Sequential (offload the AR stage)</option>
                  </select>
                </label>
              </div>
              <SamplingFields
                defaults={SAMPLING_TOKEN_DEFAULTS.scoreSampling}
                disabled={settings.planSource !== "sample" || settings.planning === "off"}
                label="Score planning sampling"
                reason={why("scoreSampling")}
                onChange={(value) => update({ scoreSampling: value })}
                value={settings.scoreSampling}
              />
              <SamplingFields
                defaults={SAMPLING_TOKEN_DEFAULTS.semanticSampling}
                label="Semantic sampling"
                reason={why("semanticSampling")}
                onChange={(value) => update({ semanticSampling: value })}
                value={settings.semanticSampling}
              />
              <fieldset className="yue2-fieldset">
                <legend>Memory</legend>
                <div className="yue2-grid">
                  <TriStateSelect label="Stage residency" reason={why("memory.acoustic")} onChange={(value) => update({ stageResidency: value })} value={settings.stageResidency} />
                  <TriStateSelect label="Chunked attention" reason={why("memory.acoustic")} onChange={(value) => update({ chunkAttention: value })} value={settings.chunkAttention} />
                  <label>
                    Attention chunk size
                    <input aria-label="Attention chunk size" disabled={settings.chunkAttention !== "on" || Boolean(why("memory.acoustic"))} title={why("memory.acoustic") ?? undefined} min={MIN_ATTENTION_CHUNK_ELEMENTS} onChange={(event) => update({ attentionChunkSize: event.target.value })} placeholder={`≥ ${MIN_ATTENTION_CHUNK_ELEMENTS} (with chunked attention on)`} type="number" value={settings.attentionChunkSize} />
                  </label>
                  <TriStateSelect label="Tiled decode" reason={why("memory.decode")} onChange={(value) => update({ tileVaeDecode: value })} value={settings.tileVaeDecode} />
                  <label>
                    Decode tile (frames)
                    <input aria-label="Decode tile" disabled={settings.tileVaeDecode !== "on" || Boolean(why("memory.decode"))} title={why("memory.decode") ?? undefined} max={MAX_DECODE_TILE_FRAMES} min="1" onChange={(event) => update({ decodeTileEdge: event.target.value })} placeholder="1–1024 (with tiled decode on)" type="number" value={settings.decodeTileEdge} />
                  </label>
                </div>
              </fieldset>
            </AdvancedSection>
  );
}
