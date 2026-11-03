const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "bmp", "gif", "tif", "tiff"]);

export function fileMatchesMediaKind(file, mediaKind) {
  const mime = String(file?.type ?? "").toLowerCase();
  if (mime.startsWith("image/") || mime.startsWith("video/") || mime.startsWith("audio/")) {
    return mime.startsWith(`${mediaKind}/`);
  }
  const name = String(file?.name ?? "").toLowerCase();
  if (mediaKind === "video") {
    return /\.(mp4|m4v|mov|webm|mkv|avi|ogv|mpeg|mpg|wmv|flv|3gp|3g2)$/i.test(name);
  }
  if (mediaKind === "audio") {
    return /\.(mp3|wav|flac|m4a|aac|ogg|oga|opus|wma|aiff?)$/i.test(name);
  }
  return /\.(png|jpe?g|webp|gif|bmp|tiff?|avif|heic|heif)$/i.test(name);
}



export function uploadFileBaseName(name) {
  return String(name ?? "").replaceAll("\\", "/").split("/").pop() ?? "";
}
export function uploadFileExtension(name) {
  const base = uploadFileBaseName(name);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(dot + 1).toLowerCase() : "";
}
export function isImageUpload(file) {
  return String(file?.type ?? "").startsWith("image/") || IMAGE_EXTENSIONS.has(uploadFileExtension(file?.name));
}
export function isCaptionUpload(file) {
  return uploadFileExtension(file?.name) === "txt" || file?.type === "text/plain";
}

// Results retain input order and individual failures. Abort stops scheduling new work.
export async function uploadFiles(files, upload, { signal, onProgress, concurrency = 3 } = {}) {
  const results = new Array(files.length);
  let next = 0;
  let completed = 0;
  async function worker() {
    while (!signal?.aborted && next < files.length) {
      const index = next++;
      try {
        const value = await upload(files[index], signal);
        if (!value?.id) throw new Error("Upload returned no asset");
        results[index] = { file: files[index], value };
      } catch (error) {
        results[index] = { file: files[index], error };
      }
      completed += 1;
      if (!signal?.aborted) onProgress?.(completed, files.length);
    }
  }
  await Promise.all(Array.from({ length: Math.min(Math.max(1, concurrency), files.length) }, worker));
  return results.filter(Boolean);
}
