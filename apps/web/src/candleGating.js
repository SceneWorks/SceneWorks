// Off-Mac (candle) UI gating (sc-19570, epic 17137) — the Windows/Linux twin of `macGating.js`.
//
// `macGating.js` mapped the real `video_mode_is_mlx_eligible` predicate onto every model's
// `macSupport.features.videoModes`, so a Mac user never reaches a mode the MLX worker won't claim.
// There was no off-Mac equivalent: `macGatingActive` is false on Windows/Linux, every macGating
// helper is a no-op there, and `videoModelServesMode` collapsed to `capabilities.includes(mode)` —
// the manifest declaration alone, with nothing asking whether a lane exists on that platform.
//
// The measured consequence: the MLX-only, candle-unclaimable, Windows/Linux-installable pairs were
// all offered as Video Studio tabs off-Mac, and submitting one produced a job that sat `queued`
// with "Waiting for an available worker." and NEVER reached a terminal state — no mlx worker can
// register off-Mac to claim it, the candle worker's own gate refuses it, and both enforce sweeps
// default to warn.
//
// `GET /api/v1/capabilities/mac` now carries `candleGatingActive` (true exactly when the API host
// is not a Mac) and `GET /api/v1/models` carries a per-model `candleSupport` block built from the
// real candle claim predicate. On a Mac — and before the capabilities endpoint responds — every
// helper here is a no-op, so Mac behaviour is untouched.
//
// THIS FILE IS THE PLATFORM-AWARE HALF, AND IT IS THE ONLY ONE THAT MAY BE. A client is free to
// know what host it is talking to and hide a tab accordingly; the HTTP API is not. `POST
// /api/v1/video/jobs` answers `201` for these pairs on every platform — the server reports the gap
// as an EXECUTION outcome instead, failing the job terminal with a `platform_unreachable:` reason
// (`JobsStore::fail_platform_unreachable_jobs`). So the two layers are complementary rather than
// duplicated: hiding spares the user a dead end, and the terminal failure catches the MCP tool, a
// raw REST caller and a recipe replay, none of which see a tab. Do not "improve" this by expecting
// a platform-conditional status code — that shape was ruled out.

export const CANDLE_NOT_AVAILABLE_LABEL = "Not available on this platform (macOS/MLX only)";

// Whether off-Mac gating is engaged at all (the API host is Windows/Linux, so candle is the only
// lane that can register). Inert until the capabilities endpoint responds, mirroring
// `macGatingActive`.
export function candleGatingActive(caps) {
  return Boolean(caps?.candleGatingActive);
}

// A whole video model with no candle lane at all (no VIDEO_UI_MODES entry is candle-claimable) →
// hidden/disabled in pickers off-Mac. Returns `{ blocked, reason, text }` or `null`, the same shape
// `macModelBlock` returns.
export function candleModelBlock(model, caps) {
  if (!candleGatingActive(caps)) return null;
  if (model?.candleSupport?.supported === false) {
    const reason = model.candleSupport.reason ?? null;
    const detail = reason?.detail || reason?.feature || "";
    return {
      blocked: true,
      reason,
      text: detail
        ? `${CANDLE_NOT_AVAILABLE_LABEL} — ${detail}`
        : CANDLE_NOT_AVAILABLE_LABEL,
    };
  }
  return null;
}

// Whether a `video_generate` mode has a candle lane for the given video model. The off-Mac twin of
// `macVideoModeBlock`. Returns `{ blocked, text }` or `null`.
export function candleVideoModeBlock(model, caps, mode) {
  if (!candleGatingActive(caps)) return null;
  const modes = model?.candleSupport?.features?.videoModes;
  if (modes && modes[mode] === false) {
    return {
      blocked: true,
      // Off-Mac the runtime is candle-only — there is no MLX fallback and no torch fallback
      // (the Python backend was retired in Phase 7), so a blocked mode simply isn't served for
      // this model here. Name macOS, because that IS where the pair works: a user who sees this
      // is being told the combination exists, just not on this machine.
      text: `${CANDLE_NOT_AVAILABLE_LABEL} — this model only supports this mode on macOS.`,
    };
  }
  return null;
}

// Partition a model list into the ones with a candle lane (for the picker) — a no-op on Mac.
export function candleAvailableModels(models, caps) {
  if (!candleGatingActive(caps)) return models ?? [];
  return (models ?? []).filter((model) => model?.candleSupport?.supported !== false);
}
