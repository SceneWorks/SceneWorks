// Clipboard writes are normally available in the secure Tauri webview context.
// WebKitGTK can still deny navigator.clipboard (session policy / compositor
// integration), so keep the user-gesture-compatible execCommand fallback.
export async function writeClipboardText(
  text,
  {
    clipboard = globalThis.navigator?.clipboard,
    documentRef = globalThis.document,
  } = {},
) {
  if (clipboard?.writeText) {
    try {
      await clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the WebKit-compatible selection/copy path.
    }
  }

  if (!documentRef?.body || typeof documentRef.execCommand !== "function") {
    return false;
  }
  const textarea = documentRef.createElement("textarea");
  textarea.value = text;
  textarea.readOnly = true;
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  textarea.style.pointerEvents = "none";
  documentRef.body.appendChild(textarea);
  textarea.select();
  textarea.setSelectionRange(0, text.length);
  try {
    return documentRef.execCommand("copy");
  } catch {
    return false;
  } finally {
    textarea.remove();
  }
}
