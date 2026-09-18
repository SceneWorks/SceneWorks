import React from "react";

// Button — a thin wrapper over SceneWorks' real action classes (.primary-action,
// .secondary-action, .danger-action, .icon-btn). These are pure-CSS in the
// SceneWorks source (no React <Button> is exported); this wrapper is an
// INTENTIONAL design-system addition so consumers get a typed, consistent API.
// See readme.md → "Intentional additions".
const VARIANT_CLASS = {
  primary: "primary-action",
  secondary: "secondary-action",
  danger: "danger-action",
  icon: "icon-btn",
};

export function Button({
  variant = "secondary",
  icon = null,
  iconRight = null,
  children,
  className = "",
  type = "button",
  ...rest
}) {
  const base = VARIANT_CLASS[variant] ?? VARIANT_CLASS.secondary;
  const cls = className ? `${base} ${className}` : base;
  return (
    <button type={type} className={cls} {...rest}>
      {icon}
      {variant !== "icon" ? children : (children ?? icon)}
      {iconRight}
    </button>
  );
}

export default Button;
