export function refreshSuccess(value) {
  return { ok: true, value };
}

export function refreshFailure(reason, error = null) {
  return error ? { ok: false, reason, error } : { ok: false, reason };
}
