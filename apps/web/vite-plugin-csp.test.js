// @vitest-environment node
import { afterEach, describe, expect, it } from "vitest";
import { createServer, build, preview } from "vite";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import process from "node:process";
import { join } from "node:path";
import cspPlugin, { browserPolicy } from "./vite-plugin-csp.js";

let server;
afterEach(async () => {
  if (server?.close) await server.close();
  else if (server?.httpServer) await new Promise((resolve) => server.httpServer.close(resolve));
  server = null;
});
const directive = (policy, name) => policy.split(";").map((part) => part.trim()).find((part) => part.startsWith(`${name} `));

describe("Docker/direct browser CSP (F-269)", () => {
  it("adds only configured API origins and preserves strict script/object policy", () => {
    const policy = browserPolicy("https://api.example.test:8443/prefix", "wss://ui.example.test:5173");
    expect(directive(policy, "connect-src")).toContain("https://api.example.test:8443");
    expect(directive(policy, "connect-src")).toContain("wss://ui.example.test:5173");
    expect(directive(policy, "img-src")).toContain("https://api.example.test:8443");
    expect(directive(policy, "media-src")).toContain("https://api.example.test:8443");
    expect(directive(policy, "script-src")).toBe("script-src 'self'");
    expect(directive(policy, "object-src")).toBe("object-src 'none'");
    expect(policy).not.toContain("*");
    expect(() => browserPolicy("https://user:secret@api.example.test")).toThrow();
    expect(() => browserPolicy("javascript:alert(1)")).toThrow();
  });

  it("enforces headers on the actual Vite shell including SPA fallback and HMR", async () => {
    server = await createServer({ server: { host: "127.0.0.1", port: 0 }, logLevel: "silent" });
    await server.listen();
    const { port } = server.httpServer.address();
    const origin = `http://127.0.0.1:${port}`;
    for (const path of ["/", "/models", "/theme-init.js"]) {
      const response = await globalThis.fetch(`${origin}${path}`);
      expect(response.status).toBe(200);
      const policy = response.headers.get("content-security-policy");
      expect(directive(policy, "script-src")).toBe("script-src 'self'");
      expect(directive(policy, "frame-ancestors")).toBe("frame-ancestors 'none'");
      expect(directive(policy, "connect-src")).toContain(`ws://127.0.0.1:${port}`);
      expect(directive(policy, "connect-src")).toContain("http://localhost:8000");
    }
  });
  it("serves the policy compiled into a configured production bundle", async () => {
    const root = await mkdtemp(join(tmpdir(), "sceneworks-csp-"));
    const original = process.env.VITE_API_BASE_URL;
    try {
      await writeFile(join(root, "index.html"), "<html><head></head><body><div>SceneWorks fixture</div></body></html>");
      process.env.VITE_API_BASE_URL = "https://configured-api.example.test:8443";
      await build({ root, configFile: false, plugins: [cspPlugin()], logLevel: "silent" });
      delete process.env.VITE_API_BASE_URL;
      const html = await readFile(join(root, "dist/index.html"), "utf8");
      expect(html).toContain('http-equiv="Content-Security-Policy"');
      expect(html).toContain("https://configured-api.example.test:8443");
      server = await preview({ root, configFile: false, plugins: [cspPlugin()], preview: { host: "127.0.0.1", port: 0 }, logLevel: "silent" });
      const { port } = server.httpServer.address();
      const response = await globalThis.fetch(`http://127.0.0.1:${port}`);
      const policy = response.headers.get("content-security-policy");
      expect(directive(policy, "connect-src")).toContain("https://configured-api.example.test:8443");
      expect(directive(policy, "connect-src")).not.toContain("http://localhost:8000");
      expect(directive(policy, "frame-ancestors")).toBe("frame-ancestors 'none'");
      await new Promise((resolve) => server.httpServer.close(resolve));
      server = null;
    } finally {
      if (original === undefined) delete process.env.VITE_API_BASE_URL;
      else process.env.VITE_API_BASE_URL = original;
      await rm(root, { recursive: true, force: true });
    }
  });

});
