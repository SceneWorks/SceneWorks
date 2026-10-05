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
    return invoke("save_image_export", {
      imageBytes: Array.from(new Uint8Array(await file.arrayBuffer())),
      suggestedFilename: file.name,
    });
  }
  const url = urlApi.createObjectURL(file);
  const anchor = documentRef.createElement("a");
  anchor.href = url;
  anchor.download = file.name;
  documentRef.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  urlApi.revokeObjectURL(url);
  return null;
}

// A pretty-printed JSON export, the shape every plan/pack/batch export uses.
export function saveJsonExport(filename, value, options) {
  const file = new File([`${JSON.stringify(value, null, 2)}\n`], filename, {
    type: "application/json",
  });
  return saveExportFile(file, options);
}
