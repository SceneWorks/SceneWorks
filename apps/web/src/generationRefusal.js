// The worker keeps the readable error and a structured refusal beside existing result telemetry.
export function generationRefusal(value) {
  const refusal = value?.result?.validationRefusal ?? value;
  if (refusal?.code !== "invalid_generation_option" || typeof refusal.detail !== "string") {
    return null;
  }
  return {
    code: refusal.code,
    field: refusal.field ?? refusal.context?.field ?? null,
    detail: refusal.detail,
    context: refusal.context ?? null,
  };
}

export function jobFailureMessage(job) {
  const refusal = generationRefusal(job);
  if (!refusal) {
    return job?.error ?? job?.message ?? "";
  }
  const { field, detail } = refusal;
  return field && field !== "request" && !detail.includes(field) ? `${field}: ${detail}` : detail;
}
