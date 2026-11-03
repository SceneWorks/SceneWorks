import React, { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";

const GAP = 12;
const ESTIMATED_HEIGHT = 340;
const OVERSCAN_ROWS = 2;
const WINDOW_THRESHOLD = 24;
const FOCUSABLE_CONTROLS = "button:not([disabled]), textarea:not([disabled]), input:not([disabled]):not([hidden]), select:not([disabled])";

function MeasuredItem({ asset, style, onMeasure, children }) {
  const ref = useRef(null);
  useLayoutEffect(() => {
    const element = ref.current;
    const measure = () => onMeasure(asset.id, element.getBoundingClientRect().height);
    measure();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [asset.id, onMeasure]);
  return <div data-caption-id={asset.id} ref={ref} style={style}>{children}</div>;
}

// Small sets keep the normal responsive grid. Large sets mount only viewport rows,
// two overscan rows on either side, and the focused row. Drafts and all-item actions
// stay with the owner; unmounting a card never discards its caption or references.
export function DatasetCaptionWindow({ assets, renderItem }) {
  const scrollRef = useRef(null);
  const [viewport, setViewport] = useState({ top: 0, width: 340, height: 720 });
  const [heights, setHeights] = useState(() => new Map());
  const [focusedId, setFocusedId] = useState(null);
  const [pendingFocus, setPendingFocus] = useState(null);
  const previous = useRef(null);
  const windowed = assets.length > WINDOW_THRESHOLD;
  const columns = Math.max(1, Math.floor((viewport.width + GAP) / (340 + GAP)));
  const measure = useCallback((id, height) => {
    if (!Number.isFinite(height) || height <= 0) return;
    const rounded = Math.ceil(height);
    setHeights((current) => {
      if (current.get(id) === rounded) return current;
      const next = new Map(current);
      next.set(id, rounded);
      return next;
    });
  }, []);
  const geometry = useMemo(() => {
    const offsets = [0];
    for (let index = 0; index < assets.length; index += columns) {
      const row = assets.slice(index, index + columns);
      const height = Math.max(174, ...row.map((asset) => heights.get(asset.id) ?? ESTIMATED_HEIGHT));
      offsets.push(offsets.at(-1) + height + GAP);
    }
    return { offsets, height: Math.max(0, offsets.at(-1) - GAP) };
  }, [assets, columns, heights]);
  const { offsets } = geometry;
  // Binary lookup remains cheap even for a dataset with thousands of items.
  const rowAt = useCallback((top) => {
    let low = 0;
    let high = offsets.length - 2;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      if (offsets[middle] <= top) low = middle;
      else high = middle - 1;
    }
    return low;
  }, [offsets]);
  const firstRow = rowAt(viewport.top);
  const startRow = Math.max(0, firstRow - OVERSCAN_ROWS);
  const endRow = Math.min(offsets.length - 1, rowAt(viewport.top + viewport.height) + OVERSCAN_ROWS + 1);

  useLayoutEffect(() => {
    if (!windowed) return;
    const element = scrollRef.current;
    const update = () => setViewport({
      top: element.scrollTop,
      width: element.clientWidth || 340,
      height: element.clientHeight || 720,
    });
    update();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, [windowed]);

  // Keep the first visible image anchored while measured card heights or column
  // counts change, including rails/badges that grow after an async report arrives.
  useLayoutEffect(() => {
    if (!windowed) { previous.current = null; return; }
    const old = previous.current;
    const element = scrollRef.current;
    let top = element.scrollTop;
    if (old && old.geometry !== geometry && old.assets === assets) {
      const anchorIndex = assets.findIndex((asset) => asset.id === old.anchorId);
      if (anchorIndex >= 0) {
        top = Math.max(0, offsets[Math.floor(anchorIndex / columns)] + old.inset);
        element.scrollTop = top;
        if (top !== viewport.top) setViewport((current) => ({ ...current, top }));
      }
    }
    const row = rowAt(top);
    previous.current = { geometry, assets, anchorId: assets[row * columns]?.id, inset: top - offsets[row] };
  }, [windowed, geometry, assets, rowAt, columns, offsets, viewport.top]);

  useLayoutEffect(() => {
    if (!pendingFocus) return;
    if (!scrollRef.current) { setPendingFocus(null); return; }
    const item = [...scrollRef.current.querySelectorAll("[data-caption-id]")]
      .find((element) => element.dataset.captionId === String(pendingFocus.id));
    const controls = item?.querySelectorAll(FOCUSABLE_CONTROLS);
    const target = pendingFocus.backward ? controls?.[controls.length - 1] : controls?.[0];
    target?.focus({ preventScroll: true });
    setPendingFocus(null);
  }, [pendingFocus]);

  const mountedRows = new Set(Array.from({ length: endRow - startRow }, (_, index) => startRow + index));
  const focusedIndex = assets.findIndex((asset) => asset.id === focusedId);
  if (focusedIndex >= 0) mountedRows.add(Math.floor(focusedIndex / columns));
  const indexes = windowed
    ? [...mountedRows].sort((a, b) => a - b)
      .flatMap((row) => assets.slice(row * columns, (row + 1) * columns).map((_, column) => row * columns + column))
    : assets.map((_, index) => index);
  const advance = (index, backward = false) => {
    const asset = assets[index];
    if (!asset) return;
    const top = offsets[Math.floor(index / columns)];
    scrollRef.current.scrollTop = top;
    setViewport((current) => ({ ...current, top }));
    setFocusedId(asset.id);
    setPendingFocus({ id: asset.id, backward });
  };
  return (
    <div
      className={windowed ? "training-caption-window" : undefined}
      ref={scrollRef}
      onScroll={(event) => {
        const top = event.currentTarget.scrollTop;
        setViewport((current) => ({ ...current, top }));
      }}
      onKeyDown={(event) => {
        if (!windowed || event.key !== "Tab") return;
        const item = event.target.closest(".training-caption-card");
        if (!item) return;
        const controls = [...item.querySelectorAll(FOCUSABLE_CONTROLS)];
        const edge = event.shiftKey ? controls[0] : controls.at(-1);
        if (event.target !== edge) return;
        const index = assets.findIndex((asset) => String(asset.id) === item.dataset.captionId);
        const next = index + (event.shiftKey ? -1 : 1);
        if (next < 0 || next >= assets.length || indexes.includes(next)) return;
        event.preventDefault();
        advance(next, event.shiftKey);
      }}
      onFocusCapture={(event) => {
        const id = event.target.closest("[data-caption-id]")?.dataset.captionId;
        if (id) setFocusedId(id);
      }}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocusedId(null);
      }}
    >
      <div className={windowed ? "training-caption-window-canvas" : "training-caption-grid"}
        style={windowed ? { height: geometry.height } : undefined}>
        {indexes.map((index) => (
          <MeasuredItem key={assets[index].id} asset={assets[index]} onMeasure={measure} style={windowed ? {
            position: "absolute", top: offsets[Math.floor(index / columns)],
            left: `calc(${index % columns} * (100% + ${GAP}px) / ${columns})`,
            width: `calc((100% - ${(columns - 1) * GAP}px) / ${columns})`,
          } : undefined}>{renderItem(assets[index])}</MeasuredItem>
        ))}
      </div>
    </div>
  );
}
