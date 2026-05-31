import { defineConfig } from "vite";

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
