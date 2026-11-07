import { fetchResource } from "./api.js";
export async function fetchPromptGuide(path, signal) {
  const response = await fetchResource(path, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.text();
}
