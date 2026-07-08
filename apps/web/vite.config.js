import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { defineConfig, searchForWorkspaceRoot } from "vite";

import themeInitPlugin from "./vite-plugin-theme-init.js";

// Expose the product version (kept in lockstep across the repo's package.json /
// tauri.conf.json by scripts/sync-version.mjs) to the frontend as a build-time
// constant. Reading package.json here works in both the desktop shell and the
// remote-LAN browser (epic 4484), where window.__TAURI__ / getVersion() is
// unavailable — see the sidebar footer in App.jsx.
const pkg = JSON.parse(
  readFileSync(fileURLToPath(new URL("./package.json", import.meta.url)), "utf8"),
);

// The About → Licenses screen (sc-3778) imports the bundled-license corpus
// directly from apps/desktop/licenses/ (single source of truth, no second copy).
// That dir is outside the web project root, so the dev server / vitest module
// loader would deny it unless it's on server.fs.allow. (The production embedded
// build resolves these via rollup and isn't gated by this.)
const licensesDir = fileURLToPath(new URL("../desktop/licenses", import.meta.url));

export default defineConfig({
  // Generate the pre-paint /theme-init.js from src/accents.js at dev/build time
  // (single source of truth for the accent-id list). See vite-plugin-theme-init.js.
  plugins: [themeInitPlugin()],
  // Matches the existing import.meta.env.VITE_* convention (see api.js). Defining
  // the specific key keeps it lint-safe (no bare global) under no-undef.
  define: {
    "import.meta.env.VITE_APP_VERSION": JSON.stringify(pkg.version),
  },
  // react-konva pulls its own react-reconciler; without deduping, Vite's dep
  // optimizer can hand it a second React copy, tripping "Invalid hook call /
  // cannot read useRef of null" when the canvas <Stage> mounts. Force a single
  // React and pre-bundle the canvas deps against it.
  resolve: {
    dedupe: ["react", "react-dom"],
  },
  optimizeDeps: {
    include: ["react", "react-dom", "react-dom/client", "konva", "react-konva"],
  },
  server: {
    headers: {
      "Cache-Control": "no-store",
    },
    fs: {
      // Keep the default workspace-root allowance and add the desktop license
      // corpus the Licenses screen imports from.
      allow: [searchForWorkspaceRoot(process.cwd()), licensesDir],
    },
  },
  build: {
    rollupOptions: {
      output: {
        // Peel third-party code (React et al.) into a separate, rarely-changing
        // chunk so the app bundle stays under Vite's size warning and the vendor
        // chunk caches across app-only deploys.
        manualChunks(id) {
          if (id.includes("node_modules")) {
            return "vendor";
          }
        },
      },
    },
  },
});
