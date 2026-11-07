import { isDesktop, tauriInvoke } from "./runtime.js";

// Save a browser-generated File (an edited image, an exported JSON plan, ...) to disk.
//
// Desktop routes the bytes through the shell's native save dialog (`save_image_export`, which
// writes any payload verbatim despite its name): WKWebView and WebKitGTK are inconsistent about
// honoring `<a download>` filenames on `blob:` URLs, and the dialog gives every platform the
// intended filename and a real destination (sc-6554). Resolves to the saved path, or null when the
// user cancels. A browser (server deployment or LAN remote) keeps the plain `<a download>` path and
// resolves to null.
export async function saveExportFile(
  file,
  {
    desktop = isDesktop,
    invoke = tauriInvoke,
    documentRef = globalThis.document,
    urlApi = globalThis.URL,
  } = {},
) {
  if (desktop) {
    const maxBytes = 256 * 1024 * 1024;
    const chunkBytes = 1024 * 1024;
    if (!Number.isSafeInteger(file.size) || file.size <= 0 || file.size > maxBytes) {
      throw new Error("The exported file must contain 1 byte to 256 MB.");
    }
    const exportId = await invoke("begin_image_export", {
      suggestedFilename: file.name,
      totalBytes: file.size,
    });
    if (!exportId) return null;
    try {
      for (let offset = 0; offset < file.size; offset += chunkBytes) {
        const chunk = await file.slice(offset, offset + chunkBytes).arrayBuffer();
        await invoke("save_image_export", chunk, {
          headers: { "x-sceneworks-export-id": exportId },
        });
      }
      return await invoke("finish_image_export", { exportId });
    } catch (error) {
      await invoke("cancel_image_export", { exportId }).catch(() => {});
      throw error;
    }
  }
  downloadBlob(file, file.name, { documentRef, urlApi });
  return null;
}

// A pretty-printed JSON export, the shape every plan/pack/batch export uses.
export function saveJsonExport(filename, value, options) {
  const file = new File([`${JSON.stringify(value, null, 2)}\n`], filename, {
    type: "application/json",
  });
  return saveExportFile(file, options);
}

export function downloadBlob(blob, filename, { documentRef = globalThis.document, urlApi = globalThis.URL } = {}) {
  const url = urlApi.createObjectURL(blob);
  const anchor = documentRef.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  documentRef.body.appendChild(anchor);
  try { anchor.click(); } finally {
    anchor.remove();
    setTimeout(() => urlApi.revokeObjectURL(url), 0);
  }
}
