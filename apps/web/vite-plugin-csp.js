import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { URL } from "node:url";

const baseline = readFileSync(new URL("./content-security-policy.txt", import.meta.url), "utf8").trim();

export function browserPolicy(apiBase, websocketOrigin) {
  let apiOrigin;
  if (apiBase) {
    const url = new URL(apiBase);
    if (!["http:", "https:"].includes(url.protocol) || url.username || url.password) {
      throw new Error("VITE_API_BASE_URL must be an HTTP(S) origin without credentials");
    }
    apiOrigin = url.origin;
  }
  return baseline.split(";").map((part) => {
    const directive = part.trim();
    const name = directive.split(" ")[0];
    const origins = [];
    if (apiOrigin && ["connect-src", "img-src", "media-src"].includes(name)) origins.push(apiOrigin);
    if (websocketOrigin && name === "connect-src") origins.push(websocketOrigin);
    return [directive, ...origins].join(" ");
  }).join("; ");
}

export default function cspPlugin() {
  let apiBase;
  let development = false;
  let hmr;
  return {
    name: "sceneworks-csp",
    configResolved(config) {
      apiBase = config.env.VITE_API_BASE_URL ?? "http://localhost:8000";
      development = config.command === "serve" && !config.isPreview;
      hmr = config.server.hmr;
      // Fail startup/build on an invalid origin instead of silently widening CSP.
      browserPolicy(apiBase);
    },
    transformIndexHtml: {
      order: "pre",
      handler() {
        // Headers enforce frame-ancestors in Vite/embedded hosts. The built HTML
        // also carries the enforceable subset for ordinary static-file hosts.
        // Development needs a request-specific websocket origin (header below).
        if (development) return [];
        const policy = browserPolicy(apiBase).split("; ").filter((part) => !part.startsWith("frame-ancestors ")).join("; ");
        return [{ tag: "meta", attrs: { "http-equiv": "Content-Security-Policy", content: policy }, injectTo: "head-prepend" }];
      },
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        let ws;
        if (hmr !== false) {
          const host = hmr?.host ?? req.headers.host;
          if (host) {
            const url = new URL(`${hmr?.protocol === "wss" || server.config.server.https ? "wss" : "ws"}://${host}`);
            if (hmr?.clientPort || hmr?.port) url.port = String(hmr.clientPort ?? hmr.port);
            ws = url.origin;
          }
        }
        res.setHeader("Content-Security-Policy", browserPolicy(apiBase, ws));
        next();
      });
    },
    generateBundle() {
      // Preview must use the API origin compiled into this bundle, even when the
      // build-only environment variable is absent in the serving process.
      this.emitFile({ type: "asset", fileName: "content-security-policy.txt", source: browserPolicy(apiBase) });
    },
    configurePreviewServer(server) {
      const policy = readFileSync(resolve(server.config.root, server.config.build.outDir, "content-security-policy.txt"), "utf8").trim();
      server.middlewares.use((_req, res, next) => {
        res.setHeader("Content-Security-Policy", policy);
        next();
      });
    },
  };
}
