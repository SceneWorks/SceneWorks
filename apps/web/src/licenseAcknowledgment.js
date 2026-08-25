// The pre-download licence-acknowledgment gate, as ONE predicate shared by every surface
// that can start a model download (sc-17227).
//
// This module exists because the gate used to live inside `ModelManagerScreen.jsx` alone, and
// the Models screen is only one of the places a download can begin. The Simple UI's model
// manager, the first-run Setup Wizard, and the studio availability gates all call
// `createModelDownloadJob` directly, so a gate enforced on the card did not bind them.
//
// Why that became load-bearing with MiniMax-H3: every previously gated model was ALSO
// credential-gated, so Hugging Face answered 401 without a saved token and an unacknowledged
// download failed regardless of which button started it. `MiniMaxAI/MiniMax-H3` is a PUBLIC
// repo — nothing upstream refuses it — so the acknowledgment is the ONLY gate, and a surface
// that skips it lands the weights.
//
// The enforcement point is `createModelDownloadJob` (`hooks/useModelsAndLoras.js`), which every
// one of those surfaces funnels through, with a matching server-side rejection in
// `apps/rust-api/src/models.rs` for clients this module never runs in (a remote browser on the
// LAN, curl, a workflow envelope's suggested action).
//
// The ack is persisted per model id in localStorage (origin-scoped, same store as the theme and
// token caches). localStorage may be unavailable (private mode, quota) or may not survive a
// desktop relaunch, so the accessors swallow failures and default to "not acknowledged" — the
// gate fails CLOSED, and the user re-accepts.

const LICENSE_ACK_KEY_PREFIX = "sceneworks-license-ack:";

// The machine-readable code the API returns when it refuses an unacknowledged download
// (`ApiError.code`). Kept next to the client predicate so both halves of the gate name the
// same rejection.
export const LICENSE_ACK_ERROR_CODE = "license_acknowledgment_required";

// What the user is told when a download is refused. Names the screen that can actually take the
// acknowledgment, because most of the surfaces this refusal fires on have no licence UI of their
// own — a bare "not allowed" would be a dead end.
export function licenseAckRefusalMessage(model) {
  const name = model?.name ?? model?.id ?? "This model";
  return `${name} requires accepting its license first. Open Models and accept the license on the ${name} card before downloading.`;
}

// "Needs a licence acknowledgment" and "needs a Hugging Face credential" are DIFFERENT
// requirements, and the catalog used to have one flag for both.
//
//   * `gated` (sc-1898) means the download 401s without a token on `credentialHost`. It still
//     implies an acknowledgment, so every existing gated model behaves exactly as before.
//   * `requiresLicenseAcknowledgment` (sc-17227) is the other half ON ITS OWN: a public repo
//     whose licence nevertheless binds the user directly, so the terms must be shown and
//     accepted before the download — with no credential UI, because there is no token to add
//     and no access to request.
//
// MiniMax-H3 is the first of the second kind: the MiniMax H3 Community License grants a
// non-transferable licence (§II) and defines a "Licensee" as whoever uses the Works (§I.9), so
// SceneWorks' own authorization does not stand in for the user's. Coupling the two would have
// failed open (no gate at all) or demanded a credential that does not exist.
export function requiresLicenseAcknowledgment(model) {
  return model?.gated === true || model?.requiresLicenseAcknowledgment === true;
}

export function readLicenseAck(modelId) {
  if (!modelId) {
    return false;
  }
  try {
    return window.localStorage.getItem(`${LICENSE_ACK_KEY_PREFIX}${modelId}`) === "true";
  } catch {
    return false;
  }
}

export function writeLicenseAck(modelId, acknowledged) {
  if (!modelId) {
    return;
  }
  try {
    const key = `${LICENSE_ACK_KEY_PREFIX}${modelId}`;
    if (acknowledged) {
      window.localStorage.setItem(key, "true");
    } else {
      window.localStorage.removeItem(key);
    }
  } catch {
    // localStorage unavailable — the ack just won't persist. The gate stays closed, which is
    // the safe direction: the user is asked again rather than silently let through.
  }
}

// The choke-point predicate: true when a download of `model` must be refused because its licence
// has not been accepted. Read the STORED ack rather than any screen's in-memory state — the
// Models screen persists on toggle, so the store is authoritative for every caller, including
// the ones that never render a checkbox.
export function licenseAcknowledgmentBlocked(model) {
  return requiresLicenseAcknowledgment(model) && !readLicenseAck(model?.id);
}

// Whether a first-run / studio surface with NO licence UI may offer this model for download at
// all. Scoped to the standalone flag, NOT to `gated`: a gated model is separately backstopped by
// Hugging Face's own 401 + "accept on the model page" flow, and has been offered on these
// surfaces for years. A `requiresLicenseAcknowledgment` model has no such backstop — its repo is
// public — so offering it where the terms cannot be shown would be offering a download this
// module is about to refuse.
export function offerableWithoutLicenseUi(model) {
  return model?.requiresLicenseAcknowledgment !== true;
}

// A catalog row that is not itself a model can still sit behind a MODEL's acknowledgment: a LoRA
// whose Hugging Face `source.repo` is a repo some `requiresLicenseAcknowledgment` model declares
// downloads exactly those restricted weights, and `POST /api/v1/loras/:id/download` refuses it
// without the assertion (sc-17227).
//
// The row cannot answer "does this need an acknowledgment?" on its own, and this client must not
// try to answer it either: the model catalog it holds is filtered to the running OS, so a repo→model
// re-derivation here would come up empty on a host where the gating model's download rows were
// stripped — and silently drop the gate on exactly the hosts where it still binds. So the SERVER
// stamps the answer onto the row (`annotate_license_acknowledgment_sources`,
// `apps/rust-api/src/models.rs`), reading the unfiltered manifest, and this reads the stamp.
//
// Returns the `{ id, name }` of the model whose card carries the licence text and the checkbox —
// the LoRA rows have neither, so a refusal that did not name it would be a dead end.
export function licenseAcknowledgmentSource(entry) {
  const id = entry?.licenseAcknowledgmentModelId;
  if (typeof id !== "string" || id === "") {
    return null;
  }
  const name = entry?.licenseAcknowledgmentModelName;
  return { id, name: typeof name === "string" && name !== "" ? name : id };
}

// The `licenseAcknowledgmentBlocked` of a borrowed acknowledgment: blocked when the row names a
// gating model whose licence the user has not accepted. Reads the SAME store the model gate reads,
// so accepting on the Models screen clears both.
export function borrowedLicenseAcknowledgmentBlocked(entry) {
  const source = licenseAcknowledgmentSource(entry);
  return source !== null && !readLicenseAck(source.id);
}
