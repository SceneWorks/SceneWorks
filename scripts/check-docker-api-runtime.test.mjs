import assert from "node:assert/strict";
import http from "node:http";
import { readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { waitForHealth } from "./check-docker-api-runtime.mjs";

const repository = fileURLToPath(new URL("../", import.meta.url));

function assertLocalPatchCopies(dockerfile, cargoManifest) {
  const patches = new Set();
  let inPatch = false;
  for (const line of cargoManifest.split(/\r?\n/)) {
    if (/^\s*\[/.test(line)) inPatch = /^\s*\[patch\./.test(line);
    if (inPatch) for (const match of line.matchAll(/\bpath\s*=\s*"([^"]+)"/g)) patches.add(match[1]);
  }
  assert(patches.size > 0, "actual root path patches must be discovered");
  const stages = [...dockerfile.matchAll(/^FROM\s+[^\n]+\s+AS\s+(\S+)\s*$/gm)];
  for (const name of ["builder", "candle-builder"]) {
    const index = stages.findIndex((stage) => stage[1] === name);
    assert(index >= 0, `${name} exists`);
    const body = dockerfile.slice(stages[index].index, stages[index + 1]?.index ?? dockerfile.length);
    const fetch = body.indexOf("cargo fetch --locked");
    assert(fetch >= 0, `${name} retains its locked dependency fetch`);
    const copies = [...body.slice(0, fetch).matchAll(/^COPY\s+([^\n]+)$/gm)].flatMap((match) => {
      const args = match[1].trim().split(/\s+/);
      if (args.some((arg) => arg.startsWith("--from="))) return [];
      const destination = args.pop();
      return args.filter((arg) => !arg.startsWith("--")).map((source) => ({ source, destination }));
    });
    for (const patch of patches) {
      assert(statSync(path.join(repository, patch, "Cargo.toml")).isFile());
      const covered = copies.some(({ source, destination }) => {
        // A manifest-only COPY is insufficient: local library/build targets and resources
        // must be present too. Directory COPY preserves its complete descendant closure.
        if (!statSync(path.join(repository, source)).isDirectory()) return false;
        const relative = path.posix.relative(source, patch);
        return relative !== ".." && !relative.startsWith("../")
          && path.posix.resolve("/app", destination, relative) === path.posix.resolve("/app", patch);
      });
      assert(covered, `${name}: root path patch ${patch} must be copied completely before cargo fetch`);
    }
  }
}

test("both Docker builders copy every actual local Cargo patch before locked fetch", () => {
  assertLocalPatchCopies(readFileSync(path.join(repository, "docker/rust.Dockerfile"), "utf8"),
    readFileSync(path.join(repository, "Cargo.toml"), "utf8"));
});

test("local patch COPY contract rejects missing delayed or misdirected dependency trees", () => {
  const dockerfile = readFileSync(path.join(repository, "docker/rust.Dockerfile"), "utf8");
  const manifest = readFileSync(path.join(repository, "Cargo.toml"), "utf8");
  for (const replacement of ["", "COPY vendor/glib-0.18.5 ./vendor/glib-0.18.5", "COPY vendor ./wrong-vendor"]) {
    const broken = dockerfile.replaceAll("COPY vendor ./vendor", replacement);
    assert.throws(() => assertLocalPatchCopies(broken, manifest), /root path patch .*before cargo fetch/);
  }
  const delayed = dockerfile.replaceAll("COPY vendor ./vendor", "")
    .replaceAll("    cargo fetch --locked", "    cargo fetch --locked\nCOPY vendor ./vendor");
  assert.throws(() => assertLocalPatchCopies(delayed, manifest), /root path patch .*before cargo fetch/);
});

test("waitForHealth retries after a single request stalls", async (t) => {
  let requests = 0;
  const server = http.createServer((_request, response) => {
    requests += 1;
    if (requests === 1) {
      // Accept the first connection without responding. The request timeout
      // must abort it so a subsequent health probe can succeed.
      return;
    }
    response.writeHead(200, { "content-type": "application/json" });
    response.end('{"runtime":"rust"}');
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });

  const { port } = server.address();
  // The request timeout also bounds the HEALTHY retry, so a 50 ms one aborted that too whenever a
  // loaded runner was slow to answer, and the count read 4 instead of 2.
  const requestTimeoutMs = 1_000;
  const readinessTimeoutMs = 10_000;
  const startedAt = Date.now();
  await waitForHealth({
    url: `http://127.0.0.1:${port}/api/v1/health`,
    readinessTimeoutMs,
    requestTimeoutMs,
    retryDelayMs: 10,
  });
  const elapsedMs = Date.now() - startedAt;

  assert.equal(requests, 2, "stalled request was aborted and retried");
  assert.ok(elapsedMs >= requestTimeoutMs - 10, `first request did not stall (${elapsedMs}ms)`);
  assert.ok(elapsedMs < readinessTimeoutMs, `retry escaped the readiness deadline (${elapsedMs}ms)`);
});

test("waitForHealth enforces its deadline when a request never responds", async (t) => {
  const server = http.createServer((_request, _response) => {
    // Reproduce the Docker failure mode: accept the connection but never send
    // headers or a body, as when create_app stalls before the API starts serving.
  });
  server.keepAliveTimeout = 1;
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });

  const { port } = server.address();
  const startedAt = Date.now();
  await assert.rejects(
    waitForHealth({
      url: `http://127.0.0.1:${port}/api/v1/health`,
      readinessTimeoutMs: 200,
      requestTimeoutMs: 50,
      retryDelayMs: 10,
    }),
    /did not become healthy within 200ms after \d+ attempt\(s\)/,
  );
  const elapsedMs = Date.now() - startedAt;

  assert.ok(elapsedMs >= 150, `deadline fired too early after ${elapsedMs}ms`);
  assert.ok(elapsedMs < 1_000, `hung request escaped the deadline (${elapsedMs}ms)`);
});
