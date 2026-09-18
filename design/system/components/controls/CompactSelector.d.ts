import React from "react";

export interface CompactSelectorItem {
  id: string;
  name: string;
  [key: string]: unknown;
}

export interface CompactSelectorProps {
  items?: CompactSelectorItem[];
  /** id of the currently active item. */
  selectedId?: string;
  onSelect?: (item: CompactSelectorItem) => void;
  /** When provided, a "create new" row appears at the top of the menu. */
  onCreate?: () => void;
  createLabel?: string;
  /** Return the asset to draw for an item's thumbnail. */
  getThumbAsset?: (item: CompactSelectorItem) => unknown;
  /** Render a thumbnail node from the asset returned by getThumbAsset. */
  renderThumbnail?: (asset: unknown) => React.ReactNode;
  /** Secondary line under an item's name. */
  getSubtitle?: (item: CompactSelectorItem) => string;
  /** id of an item that is busy (shows "Opening…", disabled). */
  busyId?: string;
  label?: string;
  placeholder?: string;
  emptyLabel?: string;
  disabled?: boolean;
}

/**
 * Compact thumbnail + name switcher pill with a dropdown list. Used for the
 * active character / dataset / project switchers. Closes on outside-click + Esc.
 * @dsCard group="Components"
 * @startingPoint section="Controls" subtitle="Thumbnail + name switcher pill" viewport="700x150"
 */
export function CompactSelector(props: CompactSelectorProps): React.ReactElement;
