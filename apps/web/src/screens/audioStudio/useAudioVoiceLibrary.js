import { useMemo, useRef, useState } from "react";
import { appConfirm } from "../../appConfirm.jsx";

// Extracted from AudioStudio along its existing concern boundary.
export function useAudioVoiceLibrary({
  referenceAudioAssetId,
  savedVoices,
  createSavedVoice,
  setReferenceAudioAssetId,
  deleteSavedVoice
}) {
  // Register-a-voice affordance (sc-13517): a name for the saved voice built from the currently
  // selected reference clip, an in-flight guard, and the post-register notice (dedup warning / info).
  const [savedVoiceName, setSavedVoiceName] = useState("");

  const [registeringVoice, setRegisteringVoice] = useState(false);

  const [deletingVoiceId, setDeletingVoiceId] = useState("");

  const deletingVoiceRef = useRef("");

  const [savedVoiceNotice, setSavedVoiceNotice] = useState(null);

  // Saved voices (sc-13517): a saved voice is "selected" when the reference clip in play is the one
  // it points to. Picking a saved voice just sets referenceAudioAssetId, which the existing pipeline
  // routes to native chatterbox_tts / OpenVoice fallback — no new generation wiring.
  const selectedSavedVoiceId = useMemo(() => {
    if (!referenceAudioAssetId) return "";
    const match = savedVoices.find(
      (voice) => voice?.referenceAudioAssetId === referenceAudioAssetId,
    );
    return match?.id ?? "";
  }, [savedVoices, referenceAudioAssetId]);

  const canRegisterVoice =
    Boolean(createSavedVoice) &&
    referenceAudioAssetId.length > 0 &&
    savedVoiceName.trim().length > 0 &&
    !registeringVoice;

  async function handleRegisterVoice() {
    if (!canRegisterVoice) return;
    setRegisteringVoice(true);
    setSavedVoiceNotice(null);
    try {
      const created = await createSavedVoice({
        name: savedVoiceName.trim(),
        referenceAudioAssetId,
      });
      // A null result means the register failed — the error is surfaced by the app-level banner.
      if (!created) return;
      setSavedVoiceName("");
      const duplicate = created.nearDuplicate;
      if (duplicate) {
        setSavedVoiceNotice({
          tone: "warning",
          message: `Saved “${created.name}”, but it looks very similar to “${duplicate.name}” (${Math.round(
            duplicate.similarity * 100,
          )}% match) — you may already have this voice registered.`,
        });
      } else {
        setSavedVoiceNotice({
          tone: "info",
          message: `Saved “${created.name}” to this project’s voices.`,
        });
      }
    } finally {
      setRegisteringVoice(false);
    }
  }

  function handleSelectSavedVoice(voice) {
    setReferenceAudioAssetId(voice.referenceAudioAssetId);
    setSavedVoiceNotice(null);
  }

  async function handleDeleteSavedVoice(voice) {
    if (!deleteSavedVoice || deletingVoiceRef.current) return;
    deletingVoiceRef.current = voice.id;
    setDeletingVoiceId(voice.id);
    try {
      const proceed = await appConfirm({
        title: "Delete saved voice?",
        message: `Permanently delete “${voice.name}” from this project’s saved voices? This cannot be undone.`,
        confirmLabel: "Delete permanently",
        cancelLabel: "Cancel",
        tone: "danger",
      });
      if (!proceed) return;
      const deleted = await deleteSavedVoice(voice.id);
      if (deleted) {
        setSavedVoiceNotice(null);
      }
    } finally {
      deletingVoiceRef.current = "";
      setDeletingVoiceId("");
    }
  }
  return { savedVoiceName, setSavedVoiceName, registeringVoice, deletingVoiceId, savedVoiceNotice, selectedSavedVoiceId, canRegisterVoice, handleRegisterVoice, handleSelectSavedVoice, handleDeleteSavedVoice };
}
