import React from "react";
import { AssetMedia, assetCanRenderAsImage } from "../assetMedia.jsx";
import { Icon } from "../Icons.jsx";
import { itemDuration } from "../../timeline.js";
import { formatTimecode } from "../../formatting.js";
import { clipHue, isAiItem } from "./editorUtils.js";

// The storyboard strip that sits directly above the timeline (design 2a, epic 12798):
// key-image cards synced to the main-track clips, separated by connectors that read as
// generated (teal "▸ 1.8s") or pending (violet "＋ generate"). Selecting a card moves the
// playhead to that key's time and selects its clip.
//
// NOTE (backend gap, tracked): there is no persisted per-clip keyframe model yet, so keys
// are derived from the main-track clips and the "generate across keys" affordance is UI
// only where the backend can't yet drive it.
export function StoryboardStrip({ clips = [], assetsById, fps = 30, selectedItemId, onSelectKey, onAddKey }) {
  const nodes = [];
  clips.forEach((clip, index) => {
    nodes.push({ type: "key", clip });
    if (index < clips.length - 1) {
      const next = clips[index + 1];
      const generated = isAiItem(next, assetsById);
      nodes.push({
        type: "conn",
        key: `conn_${clip.id}`,
        generated,
        label: generated ? `▸ ${itemDuration(next).toFixed(1)}s` : "＋ generate",
      });
    }
  });

  return (
    <div className="ve-storyboard">
      <div className="ve-storyboard-hd">
        <Icon.Stars size={14} className="ve-ai-icon" />
        <strong>Storyboard</strong>
        <span className="ve-storyboard-hint">— key images synced to the timeline; generate the motion between them</span>
        <span className="ve-storyboard-legend">
          <span className="ve-legend-dot ve-dot-accent" />
          done
          <span className="ve-legend-dot ve-dot-ai" />
          generate
        </span>
      </div>
      <div className="ve-storyboard-row">
        {nodes.map((node) => {
          if (node.type === "conn") {
            return (
              <div className="ve-conn" key={node.key}>
                <span className={`ve-conn-pill${node.generated ? " generated" : " pending"}`}>{node.label}</span>
              </div>
            );
          }
          const clip = node.clip;
          const asset = assetsById?.get?.(clip.assetId) ?? null;
          const showImage = assetCanRenderAsImage(asset);
          const ai = isAiItem(clip, assetsById);
          return (
            <button
              className={`ve-key${selectedItemId === clip.id ? " selected" : ""}${ai ? " ai" : ""}`}
              key={clip.id}
              onClick={() => onSelectKey?.(clip)}
              style={{ "--key-hue": clipHue(clip.id) }}
              title={clip.displayName}
              type="button"
            >
              {showImage ? <AssetMedia asset={asset} className="ve-key-media" controls={false} /> : null}
              <span className="ve-key-tc">{formatTimecode(clip.timelineStart, fps)}</span>
              <span className="ve-key-label">{clip.displayName}</span>
            </button>
          );
        })}
        <button className="ve-key-add" onClick={onAddKey} title="Add key image" type="button">
          <Icon.Plus size={18} />
        </button>
      </div>
    </div>
  );
}
