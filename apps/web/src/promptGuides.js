export async function fetchPromptGuide(path, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.text();
}
