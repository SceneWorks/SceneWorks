import { useEditorPanelActions } from "./useEditorPanelActions.js";

function usePanelProps({ state, actions }) {
  const events = useEditorPanelActions(actions);
  return { ...state, ...events };
}

// Each tool is an explicit producer/consumer boundary; render-derived helpers
// remain state inputs, while only user-event callbacks use committed trampolines.
export function useImageEditorPanels({ move, transform, crop, upscale, detail, color, cutout, edit, boxes }) {
  const moveProps = usePanelProps(move);
  const transformProps = usePanelProps(transform);
  const cropProps = usePanelProps(crop);
  const upscaleProps = usePanelProps(upscale);
  const detailProps = usePanelProps(detail);
  const colorProps = usePanelProps(color);
  const cutoutProps = usePanelProps(cutout);
  const editProps = usePanelProps(edit);
  const boxesProps = usePanelProps(boxes);
  return { move: moveProps, transform: transformProps, crop: cropProps, upscale: upscaleProps, detail: detailProps, color: colorProps, cutout: cutoutProps, edit: editProps, boxes: boxesProps };
}
