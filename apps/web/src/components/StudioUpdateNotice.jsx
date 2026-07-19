import React, { useEffect, useState } from "react";

export function updateOptionLabel(item) {
  const label = item?.name ?? item?.label ?? item?.id ?? "";
  return item?.updateAvailable ? `${label} • update` : label;
}

export function StudioUpdateBadge({ item }) {
  return item?.updateAvailable ? (
    <span className="studio-update-badge" aria-label={`${item.name ?? item.label ?? item.id} update available`}>
      Update
    </span>
  ) : null;
}

export function StudioUpdateNotice({ item, kind = "model", onUpdate }) {
  const [dismissedId, setDismissedId] = useState(null);
  const available = Boolean(item?.updateAvailable);
  useEffect(() => {
    if (!available) setDismissedId(null);
  }, [available]);
  if (!available || dismissedId === item?.id) return null;
  const name = item.name ?? item.label ?? item.id;
  return (
    <div className="studio-update-notice" role="status">
      <span><strong>{name}</strong> has an update. You can keep generating with the installed version.</span>
      <span className="studio-update-notice-actions">
        <button disabled={!onUpdate} onClick={() => onUpdate?.(item)} type="button">Update {kind}</button>
        <button aria-label={`Dismiss ${name} update notice`} onClick={() => setDismissedId(item?.id)} type="button">×</button>
      </span>
    </div>
  );
}
