import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { parseEventStream, validateProxy } from "./validate-runpod-proxy.mjs";

test("SSE parser handles split fields, comments, and multiline data", async () => {
  const chunks = [
    Buffer.from(": keepalive\r\nevent: job."),
    Buffer.from('updated\r\ndata: {"id":\r\n'),
    Buffer.from('data: "job-1"}\r\n\r\n'),
  ];
  const events = [];
  for await (const event of parseEventStream(chunks)) {
    events.push(event);
  }
  assert.deepEqual(events, [
    { event: "job.updated", data: '{"id":\n"job-1"}' },
  ]);
});

test("SSE parser preserves CRLF framing when the pair is split across chunks", async () => {
  const chunks = [
    Buffer.from("event: ready\r"),
    Buffer.from('\ndata: {"status":"connected"}\r'),
    Buffer.from("\n\r"),
    Buffer.from("\n"),
  ];
  const events = [];
  for await (const event of parseEventStream(chunks)) {
    events.push(event);
  }
  assert.deepEqual(events, [
    { event: "ready", data: '{"status":"connected"}' },
  ]);
});

test("proxy validator observes live SSE, streams multipart, and cleans up", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "sceneworks-proxy-test-"));
  const fixture = join(tempDir, "representative.mp4");
  await writeFile(fixture, Buffer.alloc(4096, 0x5a));

  const clients = new Set();
  let uploadedBodyBytes = 0;
  let deletedAsset = false;
  let purgedAsset = false;
  let canceledJob = false;
  let clearedJob = false;
  const server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://localhost");
    const isTicketedSse =
      request.method === "GET" &&
      url.pathname === "/api/v1/jobs/events" &&
      url.searchParams.get("ticket") === "single-use-ticket";
    if (!isTicketedSse && request.headers["x-sceneworks-token"] !== "test-token") {
      response.writeHead(401).end();
      return;
    }
    if (request.method === "GET" && url.pathname === "/api/v1/projects") {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify([{ id: "project-1", name: "Proxy Test" }]));
      return;
    }
    if (
      request.method === "POST" &&
      url.pathname === "/api/v1/jobs/events/ticket"
    ) {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ ticket: "single-use-ticket" }));
      return;
    }
    if (
      isTicketedSse
    ) {
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
      });
      response.write('event: ready\ndata: {"status":"connected"}\n\n');
      clients.add(response);
      const interval = setInterval(
        () => response.write('event: heartbeat\ndata: {"status":"ok"}\n\n'),
        20,
      );
      request.on("close", () => {
        clearInterval(interval);
        clients.delete(response);
      });
      return;
    }
    if (request.method === "POST" && url.pathname === "/api/v1/jobs") {
      for await (const _chunk of request) {
        // Drain the JSON request.
      }
      const job = { id: "job-1", status: "queued" };
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify(job));
      for (const client of clients) {
        client.write(`event: job.updated\ndata: ${JSON.stringify(job)}\n\n`);
      }
      return;
    }
    if (
      request.method === "POST" &&
      url.pathname === "/api/v1/jobs/job-1/cancel"
    ) {
      canceledJob = true;
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ id: "job-1", status: "canceled" }));
      return;
    }
    if (
      request.method === "POST" &&
      url.pathname === "/api/v1/jobs/job-1/clear"
    ) {
      clearedJob = true;
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ id: "job-1", status: "canceled" }));
      return;
    }
    if (
      request.method === "POST" &&
      url.pathname === "/api/v1/projects/project-1/assets"
    ) {
      for await (const chunk of request) {
        uploadedBodyBytes += chunk.length;
      }
      response.writeHead(201, { "content-type": "application/json" });
      response.end(JSON.stringify({ id: "asset-1" }));
      return;
    }
    if (
      request.method === "DELETE" &&
      url.pathname === "/api/v1/projects/project-1/assets/asset-1"
    ) {
      deletedAsset = true;
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ id: "asset-1", deleted: true }));
      return;
    }
    if (
      request.method === "DELETE" &&
      url.pathname === "/api/v1/projects/project-1/assets/asset-1/purge" &&
      url.searchParams.get("permanent") === "true"
    ) {
      purgedAsset = true;
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ id: "asset-1", status: "purged" }));
      return;
    }
    response.writeHead(404).end();
  });

  try {
    await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
    const address = server.address();
    const result = await validateProxy({
      rawBaseUrl: `http://127.0.0.1:${address.port}`,
      token: "test-token",
      uploadFile: fixture,
      minimumUploadBytes: 4096,
      observationSeconds: 0.12,
      minimumHeartbeats: 2,
      maximumEventLatencyMs: 1_000,
      allowNonRunpod: true,
    });

    assert.equal(result.sse.remainedConnected, true);
    assert.equal(result.upload.fileBytes, 4096);
    assert.ok(uploadedBodyBytes > 4096, "multipart framing is sent with the file");
    assert.equal(deletedAsset, true);
    assert.equal(purgedAsset, true);
    assert.equal(canceledJob, true);
    assert.equal(clearedJob, true);
    assert.equal(result.job.cleanedUp, true);
    assert.equal(result.upload.cleanedUp, true);
  } finally {
    for (const client of clients) {
      client.end();
    }
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    await rm(tempDir, { recursive: true, force: true });
  }
});
