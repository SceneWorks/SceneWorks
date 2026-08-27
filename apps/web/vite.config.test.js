import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import path from "node:path";

import { describe, expect, it } from "vitest";

const webRoot = process.cwd();
const packageJson = JSON.parse(readFileSync(path.join(webRoot, "package.json"), "utf8"));

describe("Vite safe development boundary", () => {
  it("binds npm dev and preview to loopback unless --host explicitly overrides them", () => {
    expect(packageJson.scripts.dev).toBe("vite --host 127.0.0.1");
    expect(packageJson.scripts.preview).toBe("vite preview --host 127.0.0.1");
  });

  it("rejects sensitive /@fs reads from a LAN-bound Vite process", () => {
    const env = { ...process.env, VITEST: "1" };
    const result = spawnSync(process.execPath, ["scripts/vite-safe-dev-smoke.mjs"], {
      cwd: webRoot,
      encoding: "utf8",
      env,
    });

    expect(result.status, result.stderr || result.stdout).toBe(0);
  });
});
