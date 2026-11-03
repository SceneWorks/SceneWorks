// @vitest-environment node
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createServer } from "vite";
import { describe, expect, it } from "vitest";
import baseConfig from "./vite.config.js";
import { verificationConfig } from "./vite.verify.config.mjs";

const root = fileURLToPath(new URL(".", import.meta.url));
const env = { command: "serve", mode: "verification", isSsrBuild: false, isPreview: false };

describe("verification Vite configuration", () => {
  it("evaluates async factories with the complete Vite environment and retains shared options", async () => {
    let received;
    const shared = {
      plugins: [{ name: "shared-plugin" }],
      define: { "import.meta.env.VITE_APP_VERSION": '"fixture"', "import.meta.env.VITE_API_BASE_URL": '"remote"' },
      resolve: { dedupe: ["react", "react-dom"] },
      optimizeDeps: { include: ["react"] },
      build: { rolldownOptions: { output: { manualChunks: () => "shared" } } },
      server: {
        host: "0.0.0.0", port: 1234, strictPort: false,
        headers: { "Content-Security-Policy": "default-src 'self'", "Referrer-Policy": "no-referrer" },
        fs: { strict: true, allow: ["/web", "/licenses"] },
      },
    };
    const actual = await verificationConfig(env, async (value) => { received = value; return shared; });
    expect(received).toBe(env);
    expect(actual).toEqual({
      ...shared,
      plugins: [shared.plugins[0], expect.objectContaining({ name: "sc26687-verify-mock-api" })],
      define: { ...shared.define, "import.meta.env.VITE_API_BASE_URL": '""' },
      server: { ...shared.server, host: "127.0.0.1", port: 5317, strictPort: true },
    });
    for (const config of [shared, Promise.resolve(shared)]) {
      const value = await verificationConfig(env, config);
      expect(value.plugins[0]).toBe(shared.plugins[0]);
      expect(value.plugins.at(-1).name).toBe("sc26687-verify-mock-api");
      expect({ ...value, plugins: undefined }).toEqual({ ...actual, plugins: undefined });
    }
  });

  it("retains the real shared definitions, React deduplication, plugins and filesystem boundary", async () => {
    const shared = await baseConfig(env);
    const actual = await verificationConfig(env);
    expect(actual.plugins.slice(0, -1).map((plugin) => plugin.name)).toEqual(shared.plugins.map((plugin) => plugin.name));
    expect(actual.define).toEqual({ ...shared.define, "import.meta.env.VITE_API_BASE_URL": '""' });
    expect(actual.resolve).toEqual(shared.resolve);
    expect(actual.optimizeDeps).toEqual(shared.optimizeDeps);
    expect(actual.server.fs).toEqual(shared.server.fs);
    expect(actual.server.headers).toEqual(shared.server.headers);
    expect(typeof actual.build.rolldownOptions.output.manualChunks).toBe("function");
    expect(actual.build.rolldownOptions.output.manualChunks("/web/src/context/AppContext.jsx")).toBe("app-shared");
  });

  it("starts the actual verify config and transforms EditorGuides, virtual guides and licenses", async () => {
    const server = await createServer({
      root, mode: "verification", configFile: fileURLToPath(new URL("./vite.verify.config.mjs", import.meta.url)),
      // The production verification defaults stay fixed; this owned regression uses an ephemeral port.
      server: { port: 0 },
    });
    try {
      expect(server.config.server.host).toBe("127.0.0.1");
      expect(server.config.server.strictPort).toBe(true);
      expect(server.config.server.fs.strict).toBe(true);
      const shared = await baseConfig(env);
      const portableRoots = (files) => files.map((file) => path.resolve(file).replaceAll("\\", "/"));
      expect(portableRoots(server.config.server.fs.allow)).toEqual(expect.arrayContaining(portableRoots(shared.server.fs.allow)));
      expect(server.config.mode).toBe("verification");
      expect(server.config.resolve.dedupe).toEqual(expect.arrayContaining(["react", "react-dom"]));
      expect(server.config.define["import.meta.env.VITE_APP_VERSION"]).toBeTruthy();
      expect(server.config.server.headers["Cache-Control"]).toBe("no-store");
      expect((await server.pluginContainer.resolveId("virtual:editor-guides")).id).toBe("\0virtual:editor-guides");
      await server.listen();
      const port = server.httpServer.address().port;
      expect(port).toBeGreaterThan(0);
      expect(port).not.toBe(5317);
      const get = async (file) => {
        const response = await fetch(`http://127.0.0.1:${port}${file}`, { signal: AbortSignal.timeout(5000) });
        expect(response.status, file).toBe(200);
        expect(response.headers.get("cache-control")).toBe("no-store");
        return response.text();
      };
      expect(await get("/src/components/editor/EditorGuides.jsx")).toContain("virtual:editor-guides");
      const guides = await get("/@id/__x00__virtual:editor-guides");
      expect(guides).toContain("export const scriptGuide");
      expect(guides).toContain("export const operatorGuide");
      expect(await get("/src/data/bundledLicenses.js")).toContain("manifest");
    } finally { await server.close(); }
  }, 15000);
});
