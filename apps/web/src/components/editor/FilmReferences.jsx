import React, { useMemo, useState } from "react";
import { apiFetch } from "../../api.js";
import { assetCanRenderAsImage } from "../assetMedia.jsx";

const REFERENCE_KINDS = ["character", "prop", "location", "style", "plate"];
const BINDABLE_KINDS = new Set(["character", "prop", "location"]);

function defaultRole(asset) {
  const name = asset?.displayName ?? asset?.title ?? asset?.name ?? "reference";
  return name
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64) || "reference";
}

export function FilmReferences({
  assets = [],
  busy = false,
  draft,
  importAsset,
  onDraftChange,
  onReplaceDraft,
  saveDraft,
  setNotice,
  token,
}) {
  const [assetId, setAssetId] = useState("");
  const [role, setRole] = useState("");
  const [kind, setKind] = useState("character");
  const [description, setDescription] = useState("");
  // Which subject in the image this role names (sc-24024). Required by the API once two roles
  // share one image, so the add form has to be able to say it.
  const [locator, setLocator] = useState("");
  const [approved, setApproved] = useState(false);
  const [selectedShotId, setSelectedShotId] = useState(draft.productionPlan.shots[0]?.id ?? "");
  const [bindingRole, setBindingRole] = useState("");
  const [working, setWorking] = useState(false);

  const imageAssets = useMemo(
    () => assets.filter((asset) => (
      asset.projectId === draft.projectId &&
      assetCanRenderAsImage(asset) &&
      !asset.status?.trashed &&
      !asset.status?.rejected
    )),
    [assets, draft.projectId],
  );
  const references = draft.referencePack.references;
  const selectedShot = draft.productionPlan.shots.find((shot) => shot.id === selectedShotId)
    ?? draft.productionPlan.shots[0];
  const bindable = references.filter((reference) => (
    reference.approved && BINDABLE_KINDS.has(reference.kind)
  ));

  function mutate(mutator) {
    onDraftChange((next) => mutator(next));
  }

  function mergeAddedReferences(saved, response) {
    const additions = response.referencePack.references.slice(saved.referencePack.references.length);
    mutate((current) => {
      current.revision = response.revision;
      current.updatedAt = response.updatedAt;
      current.productionPlan.version = response.productionPlan.version;
      current.referencePack.version = response.referencePack.version;
      for (const addition of additions) {
        const alreadyPresent = current.referencePack.references.some((reference) => (
          reference.role === addition.role
          && reference.sourceAssetId === addition.sourceAssetId
          && reference.file === addition.file
        ));
        if (!alreadyPresent) current.referencePack.references.push(structuredClone(addition));
      }
    });
  }

  async function addAssetReference(nextAssetId) {
    const chosen = imageAssets.find((asset) => asset.id === nextAssetId);
    const nextRole = role.trim() || defaultRole(chosen);
    if (!nextAssetId || !nextRole) {
      setNotice("Choose an image asset and name its reference role.");
      return;
    }
    setWorking(true);
    setNotice("");
    try {
      const saved = await saveDraft({ updateLocal: false });
      const next = await apiFetch(
        `/api/v1/projects/${draft.projectId}/films/${draft.id}/references`,
        token,
        {
          method: "POST",
          body: JSON.stringify({
            draftRevision: saved.revision,
            assetId: nextAssetId,
            role: nextRole,
            kind,
            description,
            locator: locator.trim() || null,
            approved,
          }),
        },
      );
      mergeAddedReferences(saved, next);
      setAssetId("");
      setRole("");
      setDescription("");
      setLocator("");
      setApproved(false);
      setNotice(`Reference ${nextRole} added to the draft.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setWorking(false);
    }
  }

  async function uploadReference(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || typeof importAsset !== "function") return;
    setWorking(true);
    setNotice("");
    try {
      const imported = await importAsset(file, { select: false, throwOnError: true });
      setAssetId(imported.id);
      setRole((current) => current || defaultRole(imported));
      // Use the returned asset directly: App state propagation is asynchronous.
      const saved = await saveDraft({ updateLocal: false });
      const next = await apiFetch(
        `/api/v1/projects/${draft.projectId}/films/${draft.id}/references`,
        token,
        {
          method: "POST",
          body: JSON.stringify({
            draftRevision: saved.revision,
            assetId: imported.id,
            role: role.trim() || defaultRole(imported),
            kind,
            description,
            locator: locator.trim() || null,
            approved,
          }),
        },
      );
      mergeAddedReferences(saved, next);
      setAssetId("");
      setRole("");
      setDescription("");
      setLocator("");
      setApproved(false);
      setNotice("Uploaded image added to the reference pack.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setWorking(false);
    }
  }

  function renameReference(index, nextRole) {
    mutate((next) => {
      const previous = next.referencePack.references[index].role;
      next.referencePack.references[index].role = nextRole;
      for (const shot of next.productionPlan.shots) {
        const conditioning = shot.conditioning;
        if (conditioning.firstFrameRole === previous) conditioning.firstFrameRole = nextRole;
        if (conditioning.lastFrameRole === previous) conditioning.lastFrameRole = nextRole;
        conditioning.referenceRoles = (conditioning.referenceRoles ?? [])
          .map((item) => (item === previous ? nextRole : item));
        shot.continuityRoles = (shot.continuityRoles ?? [])
          .map((item) => (item === previous ? nextRole : item));
      }
    });
  }

  function removeReference(index) {
    mutate((next) => {
      const removed = next.referencePack.references.splice(index, 1)[0];
      for (const shot of next.productionPlan.shots) {
        const conditioning = shot.conditioning;
        if (conditioning.firstFrameRole === removed.role) delete conditioning.firstFrameRole;
        if (conditioning.lastFrameRole === removed.role) delete conditioning.lastFrameRole;
        conditioning.referenceRoles = (conditioning.referenceRoles ?? [])
          .filter((item) => item !== removed.role);
        shot.continuityRoles = (shot.continuityRoles ?? [])
          .filter((item) => item !== removed.role);
        if (!conditioning.referenceRoles.length && conditioning.mode === "reference_to_video") {
          conditioning.mode = "text_to_video";
        }
      }
    });
  }

  function addBinding() {
    if (!selectedShot || !bindingRole || selectedShot.conditioning.referenceRoles.includes(bindingRole)) return;
    mutate((next) => {
      const shot = next.productionPlan.shots.find((item) => item.id === selectedShot.id);
      shot.conditioning.mode = "reference_to_video";
      delete shot.conditioning.firstFrameRole;
      delete shot.conditioning.lastFrameRole;
      shot.conditioning.referenceRoles.push(bindingRole);
    });
    setBindingRole("");
  }

  function moveBinding(index, offset) {
    mutate((next) => {
      const shot = next.productionPlan.shots.find((item) => item.id === selectedShot.id);
      const roles = shot.conditioning.referenceRoles;
      const target = index + offset;
      if (target < 0 || target >= roles.length) return;
      [roles[index], roles[target]] = [roles[target], roles[index]];
    });
  }

  function removeBinding(index) {
    mutate((next) => {
      const shot = next.productionPlan.shots.find((item) => item.id === selectedShot.id);
      shot.conditioning.referenceRoles.splice(index, 1);
      if (!shot.conditioning.referenceRoles.length) shot.conditioning.mode = "text_to_video";
    });
  }

  function exportPack() {
    const blob = new Blob([`${JSON.stringify(draft.referencePack, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${draft.referencePack.id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function importPack(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setWorking(true);
    setNotice("");
    try {
      const referencePack = JSON.parse(await file.text());
      const saved = await saveDraft();
      const next = await apiFetch(
        `/api/v1/projects/${draft.projectId}/films/${draft.id}/reference-pack`,
        token,
        {
          method: "PUT",
          body: JSON.stringify({ draftRevision: saved.revision, referencePack }),
        },
      );
      onReplaceDraft(next);
      setNotice("Reference pack imported.");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setWorking(false);
    }
  }

  const disabled = busy || working;
  return (
    <section aria-label="Film references" className="ve-film-references">
      <div className="ve-film-reference-heading">
        <strong>References</strong>
        <span>{references.length ? `${references.length} in this draft` : "Optional — this film can stay script-only"}</span>
        <button disabled={disabled} onClick={exportPack} type="button">Export pack</button>
        <label className="ve-film-file-button">Import pack<input accept="application/json,.json" disabled={disabled} onChange={importPack} type="file" /></label>
      </div>
      <div className="ve-film-reference-add">
        <label>Project image<select aria-label="Reference project image" disabled={disabled} onChange={(event) => { setAssetId(event.target.value); if (!role) setRole(defaultRole(imageAssets.find((asset) => asset.id === event.target.value))); }} value={assetId}>
          <option value="">Choose an existing asset</option>
          {imageAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.displayName ?? asset.title ?? asset.name ?? asset.id}</option>)}
        </select></label>
        <label>Role name<input aria-label="Reference role name" disabled={disabled} onChange={(event) => setRole(event.target.value)} value={role} /></label>
        <label>Kind<select aria-label="Reference kind" disabled={disabled} onChange={(event) => setKind(event.target.value)} value={kind}>{REFERENCE_KINDS.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
        <label>Description<input aria-label="Reference description" disabled={disabled} onChange={(event) => setDescription(event.target.value)} value={description} /></label>
        <label>Locator<input aria-label="Reference locator" disabled={disabled} onChange={(event) => setLocator(event.target.value)} placeholder="the woman on the left" value={locator} /></label>
        <label className="ve-film-reference-check"><input checked={approved} disabled={disabled} onChange={(event) => setApproved(event.target.checked)} type="checkbox" />Approved</label>
        <button disabled={disabled || !assetId} onClick={() => addAssetReference(assetId)} type="button">Add asset</button>
        <label className="ve-film-file-button">Upload image<input accept="image/png,image/jpeg,image/webp" disabled={disabled || typeof importAsset !== "function"} onChange={uploadReference} type="file" /></label>
      </div>
      {references.map((reference, index) => (
        <div className="ve-film-reference-row" key={`${reference.sourceAssetId ?? reference.file}-${index}`}>
          <input aria-label={`Reference ${index + 1} role`} disabled={disabled} onChange={(event) => renameReference(index, event.target.value)} value={reference.role} />
          <select aria-label={`Reference ${reference.role} kind`} disabled={disabled} onChange={(event) => mutate((next) => { next.referencePack.references[index].kind = event.target.value; })} value={reference.kind}>{REFERENCE_KINDS.map((item) => <option key={item} value={item}>{item}</option>)}</select>
          <input aria-label={`Reference ${reference.role} description`} disabled={disabled} onChange={(event) => mutate((next) => { next.referencePack.references[index].description = event.target.value; })} value={reference.description ?? ""} />
          <input aria-label={`Reference ${reference.role} locator`} disabled={disabled} onChange={(event) => mutate((next) => { next.referencePack.references[index].locator = event.target.value || undefined; })} placeholder="the woman on the left" value={reference.locator ?? ""} />
          <label><input checked={reference.approved} disabled={disabled} onChange={(event) => mutate((next) => { next.referencePack.references[index].approved = event.target.checked; })} type="checkbox" />Approved</label>
          <button disabled={disabled} onClick={() => removeReference(index)} type="button">Remove</button>
        </div>
      ))}
      {selectedShot ? (
        <div className="ve-film-bindings">
          <label>Shot<select aria-label="Reference binding shot" disabled={disabled} onChange={(event) => setSelectedShotId(event.target.value)} value={selectedShot.id}>{draft.productionPlan.shots.map((shot) => <option key={shot.id} value={shot.id}>{shot.id}</option>)}</select></label>
          <label>Approved subject<select aria-label="Reference binding role" disabled={disabled} onChange={(event) => setBindingRole(event.target.value)} value={bindingRole}><option value="">Choose a role</option>{bindable.map((reference) => <option key={reference.role} value={reference.role}>{reference.role}</option>)}</select></label>
          <button disabled={disabled || !bindingRole} onClick={addBinding} type="button">Add ordered binding</button>
          <ol aria-label={`Ordered bindings for ${selectedShot.id}`}>
            {selectedShot.conditioning.referenceRoles.map((item, index) => <li key={`${item}-${index}`}><span>{item}</span><button aria-label={`Move ${item} up`} disabled={disabled || index === 0} onClick={() => moveBinding(index, -1)} type="button">Up</button><button aria-label={`Move ${item} down`} disabled={disabled || index === selectedShot.conditioning.referenceRoles.length - 1} onClick={() => moveBinding(index, 1)} type="button">Down</button><button aria-label={`Remove ${item} binding`} disabled={disabled} onClick={() => removeBinding(index)} type="button">Remove</button></li>)}
          </ol>
          {references.some((reference) => !reference.approved) ? <small>Unapproved references stay in the pack but cannot be bound for conditioning.</small> : null}
        </div>
      ) : null}
    </section>
  );
}
