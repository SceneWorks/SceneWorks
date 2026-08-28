// Append a point to the active mask stroke without cloning its growing point buffer on
// every pointer event. The caller creates a new outer stroke list to notify React; the
// current stroke remains private to the active gesture.
export function appendMaskStrokePoint(lines, point) {
  const line = lines.at(-1);
  if (!line) return lines;
  line.points.push(point.x, point.y);
  return [...lines];
}
