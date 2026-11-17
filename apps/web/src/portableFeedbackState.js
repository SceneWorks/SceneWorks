export function portableFeedbackBlocks(feedback) {
  return feedback != null && feedback.state !== "valid";
}

export function portableFeedbackStatus(feedback) {
  if (feedback?.state === "pending") return "Checking options with the server…";
  if (feedback?.state === "server-required") return "The server must check these options. Reconnect to continue.";
  if (feedback?.state === "partial") return "First 256 requests checked; each remaining request is checked before it enters the queue.";
  return null;
}
