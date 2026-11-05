export function backoffDelay(attempt, { baseMs, maxMs }) {
  return Math.min(maxMs, baseMs * 2 ** attempt);
}
