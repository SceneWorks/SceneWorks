import React from "react";
import { Icon } from "../Icons.jsx";
import { FILM_SHOT_STATE_LABELS, filmRunSummary } from "./filmShotState.js";

function seconds(value) {
  const number = Number(value) || 0;
  return `${Number.isInteger(number) ? number : number.toFixed(1)}s`;
}

// Every planned shot in order, sized by its target duration, showing where it is in delivery.
// It keeps the film's timeline in view while authoring, and is the way into Timeline mode.
export function FilmCutStrip({ draft, run, onOpenTimeline, onSelectShot }) {
  const shots = draft?.productionPlan?.shots ?? [];
  if (!shots.length) return null;
  const summary = filmRunSummary(draft, run);
  const timelineId = run?.record?.timeline?.timelineId;
  return (
    <section aria-label="Film cut" className="ve-film-cut">
      <div className="ve-film-cut-head">
        <span className="ve-film-eyebrow">Cut</span>
        <strong>{draft.title}</strong>
        <span className="ve-film-cut-legend"><i className="accepted" />Accepted</span>
        <span className="ve-film-cut-legend"><i className="decide" />Needs decision</span>
        <span className="ve-film-cut-legend"><i className="rendering" />Rendering or queued</span>
        <span className="ve-film-cut-count">{summary.delivered} of {summary.total} delivered</span>
        <button disabled={!timelineId} onClick={() => onOpenTimeline?.(timelineId)} title={timelineId ? "" : "Render a shot to create the film timeline."} type="button">
          <Icon.Editor size={14} />
          Open in Timeline
        </button>
      </div>
      <ol className="ve-film-cut-shots">
        {shots.map((shot, index) => {
          const state = summary.states[index];
          return (
            <li key={shot.id} style={{ flexGrow: Math.max(Number(shot.targetDurationSeconds) || 1, 1) }}>
              <button aria-label={`${shot.id}, ${FILM_SHOT_STATE_LABELS[state]}`} className={`ve-film-cut-shot ${state}`} onClick={() => onSelectShot?.(shot.id, state)} title={shot.beat} type="button">
                <span className="ve-film-cut-thumb" />
                <span className="ve-film-cut-label"><b>{shot.id} · {seconds(shot.targetDurationSeconds)}</b><i className={state} /></span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
