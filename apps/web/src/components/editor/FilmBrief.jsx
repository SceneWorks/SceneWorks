import React from "react";

export function FilmBrief({ draft, disabled, onChange, onParse }) {
  const brief = draft.structuredBrief ?? {
    synopsis: "", styleNotes: "", targetTotalSeconds: 30, beats: [], dialogue: [],
  };
  function updateBrief(mutator) {
    onChange((next) => {
      next.structuredBrief ??= structuredClone(brief);
      mutator(next.structuredBrief);
    });
  }
  return (
    <section aria-labelledby="film-brief-heading" className="ve-film-section">
      <h3 id="film-brief-heading">Brief</h3>
      <label className="ve-film-wide">
        Original prose or screenplay
        <textarea aria-label="Original prose or screenplay" disabled={disabled} rows="8" value={draft.originalScript ?? ""} onChange={(event) => onChange((next) => { next.originalScript = event.target.value; })} />
      </label>
      <div className="ve-film-actions">
        <button disabled={disabled || !draft.originalScript?.trim()} onClick={onParse} type="button">Extract editable beats and dialogue</button>
      </div>
      <div className="ve-film-grid">
        <label className="ve-film-wide">Synopsis<textarea disabled={disabled} rows="3" value={brief.synopsis} onChange={(event) => updateBrief((next) => { next.synopsis = event.target.value; })} /></label>
        <label className="ve-film-wide">Visual direction<textarea disabled={disabled} rows="3" value={brief.styleNotes} onChange={(event) => updateBrief((next) => { next.styleNotes = event.target.value; })} /></label>
        <label>Target duration (seconds)<input disabled={disabled} min="0.1" step="0.1" type="number" value={brief.targetTotalSeconds} onChange={(event) => updateBrief((next) => { next.targetTotalSeconds = Number(event.target.value); })} /></label>
      </div>
      <div className="ve-film-document-columns">
        <div>
          <h4>Beats</h4>
          {brief.beats.length ? brief.beats.map((beat, index) => (
            <label key={beat.id}>{beat.id}<textarea aria-label={`Beat ${beat.id}`} disabled={disabled} rows="2" value={beat.summary} onChange={(event) => updateBrief((next) => { next.beats[index].summary = event.target.value; })} /></label>
          )) : <p className="ve-film-empty">Extract the script or add a manual shot below.</p>}
        </div>
        <div>
          <h4>Dialogue</h4>
          {brief.dialogue.length ? brief.dialogue.map((line, index) => (
            <div className="ve-film-dialogue" key={line.id}>
              <label>Speaker<input aria-label={`Dialogue ${line.id} speaker`} disabled={disabled} value={line.speaker} onChange={(event) => updateBrief((next) => { next.dialogue[index].speaker = event.target.value; })} /></label>
              <label>Line<textarea aria-label={`Dialogue ${line.id} text`} disabled={disabled} rows="2" value={line.text} onChange={(event) => updateBrief((next) => { next.dialogue[index].text = event.target.value; })} /></label>
            </div>
          )) : <p className="ve-film-empty">No screenplay dialogue detected. The shot plan remains editable.</p>}
        </div>
      </div>
    </section>
  );
}
