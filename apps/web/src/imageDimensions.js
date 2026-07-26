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
