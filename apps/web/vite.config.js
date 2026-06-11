import { fileURLToPath } from "node:url";

import { defineConfig, searchForWorkspaceRoot } from "vite";

// The About → Licenses screen (sc-3778) imports the bundled-license corpus
// directly from apps/desktop/licenses/ (single source of truth, no second copy).
// That dir is outside the web project root, so the dev server / vitest module
// loader would deny it unless it's on server.fs.allow. (The production embedded
// build resolves these via rollup and isn't gated by this.)
const licensesDir = fileURLToPath(new URL("../desktop/licenses", import.meta.url));

export default defineConfig({
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
