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

/**
 * Minimal SSE-only fake proxy: enough surface for `validateProxy` to reach the
 * observation-window assertions and then run its cleanup `finally`. Pass
 * `heartbeatIntervalMs: null` to serve a stream that never heartbeats.
 *
 * Deliberately *not* shared with the happy-path test below, which keeps its own
 * inline server: that one is the full-surface fixture (assets upload, delete,
 * purge, and the `canceledJob`/`clearedJob` witnesses) and asserting on those
 * witnesses is most of its point. This helper stops at the event stream because
 * the negative cases below never reach the upload. If a third caller ever wants
 * the full surface, merge the two rather than adding a third copy.
 *
 * Note both fakes answer `/cancel` with `status: "canceled"`, which is in
 * `TERMINAL_JOB_STATUSES`, so `cleanupProbeJob`'s 250 ms poll loop never runs a
 * single iteration. sc-17763 listed that poll as a flake candidate; it has no
 * flake surface here, and this comment is the record of that being checked.
 */
function startEventOnlyProxy({ heartbeatIntervalMs = 20, jobEventDelayMs = 0 } = {}) {
  const clients = new Set();
  const intervals = new Set();
  const timers = new Set();
  // Set when the validator reaches the multipart upload, which this stub deliberately does not
  // serve. Without it a validator that runs FURTHER than the test expects fails against a bare
  // `HTTP 404 Not Found` from the catch-all, which says nothing about what actually went wrong.
  const reached = { upload: false };
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
    if (request.method === "POST" && url.pathname === "/api/v1/jobs/events/ticket") {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ ticket: "single-use-ticket" }));
      return;
    }
    if (isTicketedSse) {
      response.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
      });
      response.write('event: ready\ndata: {"status":"connected"}\n\n');
      clients.add(response);
      let interval;
      if (heartbeatIntervalMs !== null) {
        interval = setInterval(
          () => response.write('event: heartbeat\ndata: {"status":"ok"}\n\n'),
          heartbeatIntervalMs,
        );
        intervals.add(interval);
      }
      request.on("close", () => {
        if (interval) {
          clearInterval(interval);
          intervals.delete(interval);
        }
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
      // `jobEventDelayMs` makes the OBSERVED latency exceed a ceiling by construction rather than
      // by hoping the machine is slow. `validateProxy` computes the latency as
      // `Math.round(now - jobStartedAt)`, so a loopback delivery under 0.5 ms rounds to exactly 0 —
      // and `0 > 0` is false, so a 0 ms ceiling never fires (sc-17803). A fixed server-side delay
      // is deterministic in the safe direction: load can only make it larger.
      const emit = () => {
        for (const client of clients) {
          client.write(`event: job.updated\ndata: ${JSON.stringify(job)}\n\n`);
        }
      };
      if (jobEventDelayMs > 0) {
        const timer = setTimeout(() => {
          timers.delete(timer);
          emit();
        }, jobEventDelayMs);
        timers.add(timer);
      } else {
        emit();
      }
      return;
    }
    if (
      request.method === "POST" &&
      (url.pathname === "/api/v1/jobs/job-1/cancel" ||
        url.pathname === "/api/v1/jobs/job-1/clear")
    ) {
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify({ id: "job-1", status: "canceled" }));
      return;
    }
    if (request.method === "POST" && /^\/api\/v1\/projects\/[^/]+\/assets$/.test(url.pathname)) {
      reached.upload = true;
    }
    response.writeHead(404).end();
  });

  return {
    server,
    reached,
    async listen() {
      await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
      return `http://127.0.0.1:${server.address().port}`;
    },
    async close() {
      for (const interval of intervals) {
        clearInterval(interval);
      }
      intervals.clear();
      for (const timer of timers) {
        clearTimeout(timer);
      }
      timers.clear();
      for (const client of clients) {
        client.end();
      }
      server.closeAllConnections();
      await new Promise((resolve) => server.close(resolve));
    },
  };
}

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
      // `stopWhenSatisfied` ends the window the moment the contract is met
      // (~40 ms with a 20 ms heartbeat interval), so this 10 s is a ceiling the
      // test never spends, not a dwell. sc-17763 flaked because the old 120 ms
      // wall-clock window had to survive 17 sibling test files competing for the
      // CPU, and an ~80 ms scheduling stall lost the second heartbeat.
      observationSeconds: 10,
      minimumHeartbeats: 2,
      // Slack, not a budget: the loop is already bounded by the 10 s ceiling
      // above, so this can never fire here. The latency branch is pinned
      // deterministically by its own test below instead of by a wall-clock
      // margin that a loaded box can blow through.
      maximumEventLatencyMs: 10_000,
      allowNonRunpod: true,
      stopWhenSatisfied: true,
    });
    // Pins the claim the comment above makes: deleting `stopWhenSatisfied` from
    // the validator turns this red instead of merely making CI mysteriously
    // slower. Deliberately expressed as a fraction of the ceiling rather than a
    // fixed millisecond bound -- a bare wall-clock threshold here would be the
    // exact defect sc-17763 is about. The only way to lose this is a multi-second
    // stall, which every other budget in the file would lose first.
    assert.ok(
      result.sse.observedSeconds < result.sse.observationSeconds / 2,
      `expected the satisfied contract to end the window early, observed ` +
        `${result.sse.observedSeconds}s of a ${result.sse.observationSeconds}s ceiling`,
    );
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

// The main test above waits for heartbeats rather than racing a 120 ms clock,
// so on its own it would still pass if the heartbeat path were deleted -- it
// would simply run to its 10 s ceiling and then fail, which no green run
// proves. These two negative tests pin the failure side directly: each drives
// a server that violates one contract and asserts the specific `fail()` fires.

test("proxy validator fails when the stream never heartbeats", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "sceneworks-proxy-test-"));
  const fixture = join(tempDir, "representative.mp4");
  await writeFile(fixture, Buffer.alloc(4096, 0x5a));
  const proxy = startEventOnlyProxy({ heartbeatIntervalMs: null });

  try {
    const rawBaseUrl = await proxy.listen();
    await assert.rejects(
      validateProxy({
        rawBaseUrl,
        token: "test-token",
        uploadFile: fixture,
        minimumUploadBytes: 4096,
        // This one genuinely dwells: heartbeats never arrive, so
        // `stopWhenSatisfied` can never fire and the window always runs to the
        // ceiling. That makes the ceiling a real budget for the *other*
        // requirement -- `job.updated` must land inside it, or validateProxy
        // throws "matching job.updated event was not observed" and the regex
        // below fails on the wrong message. At 250 ms that was sc-17763's own
        // flake class at 2x the budget (a ~300 ms stall flips the message);
        // 3 s buys ~12x margin for the cost of a 3 s dwell.
        observationSeconds: 3,
        minimumHeartbeats: 2,
        maximumEventLatencyMs: 10_000,
        allowNonRunpod: true,
        stopWhenSatisfied: true,
      }),
      /only 0 heartbeat events arrived; expected 2/,
    );
  } finally {
    await proxy.close();
    await rm(tempDir, { recursive: true, force: true });
  }
});

test("proxy validator fails when the job event exceeds the latency ceiling", async () => {
  const tempDir = await mkdtemp(join(tmpdir(), "sceneworks-proxy-test-"));
  const fixture = join(tempDir, "representative.mp4");
  await writeFile(fixture, Buffer.alloc(4096, 0x5a));
  // The event is held back a fixed 25 ms so the observed latency clears the ceiling BY
  // CONSTRUCTION. The previous fixture emitted immediately and relied on "any real delivery
  // exceeds a 0 ms ceiling" -- which is false: `validateProxy` rounds the latency to whole
  // milliseconds, a loopback delivery lands under 0.5 ms, and `0 > 0` does not fire. The run then
  // sailed past the ceiling check into the upload this stub does not serve and failed on the
  // catch-all 404. Measured 12/12 sub-millisecond in isolation, ~1-in-3 inside the suite, where
  // concurrent tests add just enough load to sometimes push it over (sc-17803).
  const proxy = startEventOnlyProxy({ jobEventDelayMs: 25 });

  try {
    const rawBaseUrl = await proxy.listen();
    await assert.rejects(
      validateProxy({
        rawBaseUrl,
        token: "test-token",
        uploadFile: fixture,
        minimumUploadBytes: 4096,
        observationSeconds: 10,
        minimumHeartbeats: 2,
        maximumEventLatencyMs: 0,
        allowNonRunpod: true,
        stopWhenSatisfied: true,
      }),
      /matching job\.updated event took \d+ ms \(limit 0 ms\)/,
    );
    // The ceiling must stop the run BEFORE the upload. Asserting this keeps a future regression
    // legible: without it, falling through surfaces as a bare `HTTP 404 Not Found`.
    assert.equal(
      proxy.reached.upload,
      false,
      "the latency ceiling must abort the run before the multipart upload is attempted",
    );
  } finally {
    await proxy.close();
    await rm(tempDir, { recursive: true, force: true });
  }
});
