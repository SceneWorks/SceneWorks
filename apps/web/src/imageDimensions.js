// Start a cancel-safe natural-dimension probe. Cleanup detaches handlers and
// releases the pending image request so a stale selection cannot update state.
export function probeImageDimensions(src, onDimensions) {
  if (!src || typeof Image === "undefined" || typeof onDimensions !== "function") {
    return () => {};
  }
  let active = true;
  const image = new Image();
  image.onload = () => {
    if (active && image.naturalWidth && image.naturalHeight) {
      onDimensions(image.naturalWidth, image.naturalHeight);
    }
  };
  image.src = src;
  return () => {
    active = false;
    image.onload = null;
    image.onerror = null;
    image.src = "";
  };
}

// Image probes settle even when a browser never reports load or error.
export function loadImage(src, { signal, timeoutMs = 15000 } = {}) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    let timer;
    let settled = false;
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      image.onload = image.onerror = null;
      if (error) { image.src = ""; reject(error); } else resolve(image);
    };
    const abort = () => finish(new DOMException("Image load aborted", "AbortError"));
    if (signal?.aborted) return abort();
    signal?.addEventListener("abort", abort, { once: true });
    timer = setTimeout(() => finish(new Error("Image load timed out")), timeoutMs);
    image.onload = () => finish(null);
    image.onerror = () => finish(new Error("Could not decode image"));
    image.src = src;
  });
}

// Caller owns the object URL on success; errors and aborts release it here.
export async function blobToImage(blob, options) {
  const objectUrl = URL.createObjectURL(blob);
  try { return { image: await loadImage(objectUrl, options), objectUrl }; }
  catch (error) { URL.revokeObjectURL(objectUrl); throw error; }
}
