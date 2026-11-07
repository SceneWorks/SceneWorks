// Keep generated catalog/model rows and interned observation entries on individual lines.
// This preserves reviewable outer keys without hundreds of thousands of formatting-only lines.
export function renderBaseline(value, depth = 0) {
  const indent = "  ".repeat(depth);
  if (Array.isArray(value)) {
    if (!value.length) return "[]";
    return `[\n${value.map((entry) => `${indent}  ${depth >= 2 ? JSON.stringify(entry) : renderBaseline(entry, depth + 1)}`).join(",\n")}\n${indent}]`;
  }
  if (value && typeof value === "object" && depth < 2) {
    return `{\n${Object.entries(value).map(([key, entry]) => `${indent}  ${JSON.stringify(key)}: ${renderBaseline(entry, depth + 1)}`).join(",\n")}\n${indent}}`;
  }
  return JSON.stringify(value);
}
