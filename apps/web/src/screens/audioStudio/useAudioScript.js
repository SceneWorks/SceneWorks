

// Extracted from AudioStudio along its existing concern boundary.
export function useAudioScript({
  setScript,
  speakerChoices
}) {
  // Segmented-script editor handlers (sc-13676). Each edits a single { speaker, text } turn; the
  // speaker dropdown is capped at `maxSpeakers` labels so a script can never name more speakers than
  // the model renders. A row can always be added (dialogue length is unbounded — max_speakers caps
  // DISTINCT speakers, not turns) and removed down to a single turn.
  function updateSegment(index, patch) {
    setScript((current) => current.map((segment, i) => (i === index ? { ...segment, ...patch } : segment)));
  }

  function addSegment() {
    setScript((current) => {
      // Default the new turn to the speaker that keeps a two-person dialogue alternating.
      const next = speakerChoices[current.length % speakerChoices.length]?.value ?? speakerChoices[0]?.value ?? "S1";
      return [...current, { speaker: next, text: "" }];
    });
  }

  function removeSegment(index) {
    setScript((current) => (current.length > 1 ? current.filter((_, i) => i !== index) : current));
  }
  return { updateSegment, addSegment, removeSegment };
}
