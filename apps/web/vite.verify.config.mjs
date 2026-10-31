// VERIFICATION-ONLY, UNTRACKED. Launches the real apps/web frontend on one fixed
// localhost port with the controlled mock API in ./verify-fixtures.mjs. It never
// proxies or forwards to a real backend: the middleware answers every /api request
// itself and 404s unknown paths.
import { defineConfig } from "vite";
import baseConfig from "./vite.config.js";
import { closeVerifyEventStreams, verifyMiddleware } from "./verify-fixtures.mjs";

const base = typeof baseConfig === "function" ? null : baseConfig;

export default defineConfig({
  ...(base ?? {}),
  plugins: [...(base?.plugins ?? []), {
    name: "sc26687-verify-mock-api",
    configureServer(server) { server.middlewares.use(verifyMiddleware()); },
    // The mock API's event streams keep a socket open for as long as they live, so the server's
    // own close() waits on them. They live in THIS process — the driver's copy of the fixture
    // module is a different instance and cannot reach them — so they are ended here, where the
    // server that owns them is shutting down.
    closeBundle() { closeVerifyEventStreams(); },
  }],
  define: {
    ...(base?.define ?? {}),
    // Same-origin: api.js sends requests to window.location.origin, which is this
    // verification server, and never to localhost:8000.
    "import.meta.env.VITE_API_BASE_URL": JSON.stringify(""),
  },
  server: {
    ...(base?.server ?? {}),
    host: "127.0.0.1",
    port: 5317,
    strictPort: true,
  },
});
