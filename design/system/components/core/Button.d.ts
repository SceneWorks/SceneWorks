import React from "react";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  /**
   * Visual treatment:
   * - primary   — accent-filled CTA (.primary-action), taller 44px
   * - secondary — neutral surface button (.secondary-action) — DEFAULT
   * - danger    — destructive, danger-tinted (.danger-action)
   * - icon      — 34×34 square icon-only button (.icon-btn)
   */
  variant?: "primary" | "secondary" | "danger" | "icon";
  /** Leading icon element, e.g. <Icon.Sparkle />. */
  icon?: React.ReactNode;
  /** Trailing icon element, e.g. <Icon.ChevDown />. */
  iconRight?: React.ReactNode;
  children?: React.ReactNode;
}

/**
 * SceneWorks button. Wraps the real .primary-action / .secondary-action /
 * .danger-action / .icon-btn classes.
 * @dsCard group="Components"
 * @startingPoint section="Core" subtitle="Primary / secondary / danger / icon buttons" viewport="700x150"
 */
export function Button(props: ButtonProps): React.ReactElement;
