import assert from "node:assert/strict";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { createServer } from "vite";

const webRoot = fileURLToPath(new URL("..", import.meta.url));
const repoRoot = path.resolve(webRoot, "../..");

function vitePath(file) {
  return path.resolve(file).replaceAll("\\", "/");
}

function fsUrl(file) {
  return `/@fs/${encodeURI(vitePath(file))}`;
}

const server = await createServer({
  configFile: path.join(webRoot, "vite.config.js"),
  root: webRoot,
  // Exercise the exposure boundary in the explicit LAN posture. Port zero
  // keeps this smoke test safe for concurrent runs.
  server: { host: "0.0.0.0", port: 0 },
});

try {
  assert.deepEqual(server.config.server.fs.allow, [
    vitePath(webRoot),
    vitePath(path.join(repoRoot, "apps", "desktop", "licenses")),
  ]);
  await server.transformRequest("/src/data/bundledLicenses.js");
  await server.listen();
  const address = server.httpServer.address();
  const origin = `http://127.0.0.1:${address.port}`;
  const configPath = vitePath(path.join(repoRoot, "config", "manifests", "builtin.styles.jsonc"));
  const deniedPaths = [
    fsUrl(configPath),
    fsUrl(path.join(repoRoot, "crates", "sceneworks-worker", "Cargo.toml")),
    fsUrl(path.join(repoRoot, "documents", "style.txt")),
    `/@fs/${configPath.replace("config", "con%66ig")}`,
    `/@fs/${configPath.replace("/config/", "/documents/../config/")}`,
  ];

  for (const requestPath of deniedPaths) {
    const response = await fetch(`${origin}${requestPath}`);
    assert.equal(response.ok, false, `${requestPath} must not be readable through /@fs`);
  }
} finally {
  await server.close();
}
