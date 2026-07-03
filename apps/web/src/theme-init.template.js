// TEMPLATE for the pre-paint theme script — do NOT edit the generated
// public/theme-init.js by hand; edit this template. The vite-plugin-theme-init
// plugin substitutes the ACCENT_IDS placeholder below with the id list derived
// from accents.js (single source of truth) and serves/emits it as
// /theme-init.js.
//
// Applied before first paint to avoid a theme/accent flash. Kept as an external
// script (not inline) so the served CSP can use a strict script-src 'self'.
try {
  const root = document.documentElement;

  const savedTheme = window.localStorage.getItem("sceneworks-theme");
  if (savedTheme === "dark" || savedTheme === "light") {
    root.setAttribute("data-theme", savedTheme);
  }

  // Generated from src/accents.js (ACCENTS[].id) at build time.
  const ACCENT_IDS = /* @accent-ids */ [];
  const savedAccent = window.localStorage.getItem("sceneworks-accent");
  if (savedAccent && ACCENT_IDS.indexOf(savedAccent) !== -1) {
    root.setAttribute("data-accent", savedAccent);
  }
} catch {
  // ignore (private mode etc.)
}
