import React, { useMemo, useState } from "react";
import { addFilmSound } from "../../api/films.js";
import { AssetMedia, assetCanRenderAsAudio } from "../assetMedia.jsx";
import { FindingList, SOUND_FIELD_PREFIX } from "./filmFindings.jsx";

const KINDS = ["dialogue", "ambience", "music", "sfx"];
const SPEECH_MODELS = ["kokoro_82m", "chatterbox_tts", "moss_tts_realtime", "moss_ttsd_v05"];

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function BedFields({ bed, disabled, entries, label, onBedChange, onRemove }) {
  return (
    <fieldset>
      <legend>{label}</legend>
      <label>Sound role<select disabled={disabled} value={bed.role} onChange={(event) => onBedChange((next) => { next.role = event.target.value; })}>{entries.map((entry) => <option key={entry.role} value={entry.role}>{entry.role}</option>)}</select></label>
      <label>Gain<input disabled={disabled} max="4" min="0" step="0.01" type="number" value={bed.gain ?? 1} onChange={(event) => onBedChange((next) => { next.gain = number(event.target.value, 1); })} /></label>
      <label><input checked={Boolean(bed.muted)} disabled={disabled} type="checkbox" onChange={(event) => onBedChange((next) => { next.muted = event.target.checked; })} /> Muted</label>
      {[['startSeconds', 'Timeline start'], ['sourceInSeconds', 'Source in'], ['fadeInSeconds', 'Fade in'], ['fadeOutSeconds', 'Fade out']].map(([key, fieldLabel]) => <label key={key}>{fieldLabel}<input disabled={disabled} min="0" step="0.01" type="number" value={bed[key] ?? 0} onChange={(event) => onBedChange((next) => { next[key] = number(event.target.value); })} /></label>)}
      <button disabled={disabled} onClick={onRemove} type="button">Remove {label.toLowerCase()}</button>
    </fieldset>
  );
}

function BedEditor({ disabled, kind, sound, onChange }) {
  const bed = sound[kind];
  const entries = sound.entries.filter((entry) => entry.kind === kind);
  if (!bed) return <button disabled={disabled || !entries.length} onClick={() => onChange((next) => { next.productionPlan.sound[kind] = { role: entries[0].role, gain: 1, muted: false, startSeconds: 0, sourceInSeconds: 0, fadeInSeconds: 0, fadeOutSeconds: 0 }; })} type="button">Add {kind} bed</button>;
  return <BedFields bed={bed} disabled={disabled} entries={entries} label={`${kind[0].toUpperCase() + kind.slice(1)} bed`} onBedChange={(mutate) => onChange((next) => mutate(next.productionPlan.sound[kind]))} onRemove={() => onChange((next) => { delete next.productionPlan.sound[kind]; })} />;
}

function SfxBeds({ disabled, sound, onChange }) {
  const entries = sound.entries.filter((entry) => entry.kind === "sfx");
  const beds = sound.sfx ?? [];
  return <div className="ve-film-sound-beds">
    {beds.map((bed, index) => <BedFields bed={bed} disabled={disabled} entries={entries} key={`${bed.role}:${index}`} label={`Sound effect ${index + 1}`} onBedChange={(mutate) => onChange((next) => mutate(next.productionPlan.sound.sfx[index]))} onRemove={() => onChange((next) => { next.productionPlan.sound.sfx.splice(index, 1); })} />)}
    <button disabled={disabled || !entries.length} onClick={() => onChange((next) => { next.productionPlan.sound.sfx ??= []; next.productionPlan.sound.sfx.push({ role: entries[0].role, gain: 1, muted: false, startSeconds: 0, sourceInSeconds: 0, fadeInSeconds: 0, fadeOutSeconds: 0 }); })} type="button">Add sound effect bed</button>
  </div>;
}

export function FilmSound({ activeProject, assets = [], disabled, draft, findings = [], onChange, onReplaceDraft, saveDraft, setNotice, token }) {
  const audioAssets = useMemo(() => assets.filter(assetCanRenderAsAudio), [assets]);
  // The sound half of the reference pack is authored HERE, so its findings are shown here
  // (sc-24028). The References step deliberately leaves them alone, and between the two panels
  // every pack-level finding the server reports is displayed somewhere.
  const soundFindings = findings.filter((finding) => (
    finding.shotId == null && finding.field.startsWith(SOUND_FIELD_PREFIX)
  ));
  const [assetId, setAssetId] = useState("");
  const [role, setRole] = useState("");
  const [kind, setKind] = useState("dialogue");
  const [description, setDescription] = useState("");
  const sound = { ...draft.productionPlan.sound, entries: draft.referencePack.sound ?? [] };

  async function stageAsset() {
    if (!assetId || !role.trim()) return;
    try {
      const saved = await saveDraft();
      const next = await addFilmSound(activeProject.id, saved.id, {
        assetId,
        description,
        draftRevision: saved.revision,
        kind,
        role: role.trim(),
      }, token);
      onReplaceDraft(next);
      setRole("");
      setDescription("");
      setNotice("Audio copied into the draft sound pack.");
    } catch (error) {
      setNotice(error.message);
    }
  }

  function addSynthesizedLine() {
    onChange((next) => {
      next.referencePack.sound ??= [];
      next.referencePack.sound.push({ role: `line_${next.referencePack.sound.length + 1}`, kind: "dialogue", description: "", text: "", model: "kokoro_82m" });
    });
  }

  return (
    <details className="ve-film-section" open>
      <summary>Sound and dialogue</summary>
      <p className="ve-film-help">Add recorded or synthesized dialogue and sequence sound beds.</p>
      <FindingList
        label="Sound findings"
        messages={soundFindings.map((finding) => finding.message)}
      />

      <div className="ve-film-form">
        <label>Generated picture audio<select aria-label="Generated picture audio" disabled={disabled} value={draft.productionPlan.sound.generatedAudio ?? "mute"} onChange={(event) => onChange((next) => { next.productionPlan.sound.generatedAudio = event.target.value; })}><option value="mute">Mute generated clip audio</option><option value="include">Include generated clip audio</option></select></label>
        <label>Dialogue bus gain<input aria-label="Dialogue bus gain" disabled={disabled} max="4" min="0" step="0.01" type="number" value={draft.productionPlan.sound.dialogue?.gain ?? 1} onChange={(event) => onChange((next) => { next.productionPlan.sound.dialogue ??= { gain: 1, muted: false }; next.productionPlan.sound.dialogue.gain = number(event.target.value, 1); })} /></label>
        <label><input aria-label="Mute dialogue bus" checked={Boolean(draft.productionPlan.sound.dialogue?.muted)} disabled={disabled} type="checkbox" onChange={(event) => onChange((next) => { next.productionPlan.sound.dialogue ??= { gain: 1, muted: false }; next.productionPlan.sound.dialogue.muted = event.target.checked; })} /> Mute dialogue bus</label>
      </div>
      <div className="ve-film-sound-list">
        {sound.entries.map((entry, index) => {
          const sourceAsset = entry.file ? audioAssets.find((asset) => entry.file.includes(asset.id)) : null;
          return <fieldset key={`${entry.role}:${index}`}>
            <legend>Sound {index + 1}</legend>
            <label>Role<input aria-label={`Sound ${index + 1} role`} disabled={disabled} value={entry.role} onChange={(event) => onChange((next) => { next.referencePack.sound[index].role = event.target.value; })} /></label>
            <label>Kind<select aria-label={`Sound ${index + 1} kind`} disabled={disabled} value={entry.kind} onChange={(event) => onChange((next) => { next.referencePack.sound[index].kind = event.target.value; })}>{KINDS.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
            <label>Description<input disabled={disabled} value={entry.description ?? ""} onChange={(event) => onChange((next) => { next.referencePack.sound[index].description = event.target.value; })} /></label>
            {entry.text !== undefined ? <>
              <label>Dialogue text<textarea aria-label={`Sound ${index + 1} dialogue text`} disabled={disabled} maxLength={1000} value={entry.text ?? ""} onChange={(event) => onChange((next) => { next.referencePack.sound[index].text = event.target.value; })} /></label>
              <label>Voice<input aria-label={`Sound ${index + 1} voice`} disabled={disabled} placeholder="Model default" value={entry.voice ?? ""} onChange={(event) => onChange((next) => { const value = event.target.value; if (value) next.referencePack.sound[index].voice = value; else delete next.referencePack.sound[index].voice; })} /></label>
              <label>Speech model<select aria-label={`Sound ${index + 1} speech model`} disabled={disabled} value={entry.model ?? "kokoro_82m"} onChange={(event) => onChange((next) => { next.referencePack.sound[index].model = event.target.value; })}>{SPEECH_MODELS.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
              <p className="ve-film-help">Preview is available on the timeline after synthesis.</p>
            </> : <>
              <p className="ve-film-help">Prerecorded · {entry.file}</p>
              {sourceAsset ? <AssetMedia asset={sourceAsset} className="ve-film-audio-preview" /> : null}
            </>}
            <button disabled={disabled} onClick={() => onChange((next) => { next.referencePack.sound.splice(index, 1); })} type="button">Remove sound</button>
          </fieldset>;
        })}
      </div>
      <button disabled={disabled} onClick={addSynthesizedLine} type="button">Add generated dialogue</button>
      <fieldset>
        <legend>Add prerecorded sound</legend>
        <label>Project audio<select aria-label="Project audio" disabled={disabled} value={assetId} onChange={(event) => setAssetId(event.target.value)}><option value="">Choose audio</option>{audioAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.displayName}</option>)}</select></label>
        <label>Role<input aria-label="Prerecorded sound role" disabled={disabled} value={role} onChange={(event) => setRole(event.target.value)} /></label>
        <label>Kind<select aria-label="Prerecorded sound kind" disabled={disabled} value={kind} onChange={(event) => setKind(event.target.value)}>{KINDS.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label>Description<input disabled={disabled} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
        {assetId ? <AssetMedia asset={audioAssets.find((asset) => asset.id === assetId)} className="ve-film-audio-preview" /> : null}
        <button disabled={disabled || !assetId || !role.trim()} onClick={stageAsset} type="button">Copy into sound pack</button>
      </fieldset>
      <div className="ve-film-sound-beds">
        <BedEditor disabled={disabled} kind="ambience" onChange={onChange} sound={sound} />
        <BedEditor disabled={disabled} kind="music" onChange={onChange} sound={sound} />
      </div>
      <SfxBeds disabled={disabled} onChange={onChange} sound={sound} />
    </details>
  );
}
