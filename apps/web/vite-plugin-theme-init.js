import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { renderThemeInit } from "./src/accentIds.js";

// Generate the pre-paint theme script (theme-init.js) from a template, with the
// accent-id list derived from src/accents.js at build/dev time — a single source
// of truth, so editing accents.js alone keeps the pre-paint list correct (no
// hand-copied "keep in sync" array). See public/fonts note + sc-8956.
//
// theme-init.js must load before the module graph (it runs pre-paint from a
// plain <script src>), so it can't import accents.js at runtime. This plugin:
//   - dev:   serves a freshly generated /theme-init.js via middleware
//   - build: emits theme-init.js into the output root (like a public/ asset)

const ACCENTS_PATH = fileURLToPath(new URL("./src/accents.js", import.meta.url));
const TEMPLATE_PATH = fileURLToPath(new URL("./src/theme-init.template.js", import.meta.url));

/** Read accents.js + the template and produce the final theme-init.js source. */
export function generateThemeInit() {
  const accentsSource = readFileSync(ACCENTS_PATH, "utf8");
  const templateSource = readFileSync(TEMPLATE_PATH, "utf8");
  return renderThemeInit(templateSource, accentsSource);
}

/** @returns {import("vite").Plugin} */
export default function themeInitPlugin() {
  return {
    name: "sceneworks-theme-init",

    // Dev: intercept the request the shell makes (`<script src="/theme-init.js">`)
    // and serve a freshly generated script so edits to accents.js are picked up
    // on the next reload without a build step.
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url || "").split("?")[0];
        if (url !== "/theme-init.js") {
          next();
          return;
        }
        try {
          const body = generateThemeInit();
          res.setHeader("Content-Type", "application/javascript; charset=utf-8");
          res.setHeader("Cache-Control", "no-store");
          res.end(body);
        } catch (err) {
          next(err);
        }
      });
    },

    // Build: emit theme-init.js into the output root so the built index.html's
    // `<script src="/theme-init.js">` resolves, mirroring a public/ asset.
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "theme-init.js",
        source: generateThemeInit(),
      });
    },
  };
}
