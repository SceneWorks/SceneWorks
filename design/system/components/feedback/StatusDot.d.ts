import React from "react";

export interface StatusDotProps {
  /** true → success (green); false → danger (red). */
  ok?: boolean;
}

/**
 * Tiny status indicator dot. Green when ok, red otherwise.
 * @dsCard group="Components"
 */
export function StatusDot(props: StatusDotProps): React.ReactElement;
