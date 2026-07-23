import assert from "node:assert/strict";
import http from "node:http";
import test from "node:test";

import { waitForHealth } from "./check-docker-api-runtime.mjs";

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
  const startedAt = Date.now();
  await waitForHealth({
    url: `http://127.0.0.1:${port}/api/v1/health`,
    readinessTimeoutMs: 500,
    requestTimeoutMs: 50,
    retryDelayMs: 10,
  });
  const elapsedMs = Date.now() - startedAt;

  assert.equal(requests, 2, "stalled request was aborted and retried");
  assert.ok(elapsedMs >= 40, `first request did not stall (${elapsedMs}ms)`);
  assert.ok(elapsedMs < 500, `retry escaped the readiness deadline (${elapsedMs}ms)`);
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
