import { randomUUID } from "node:crypto";
import { createReadStream, statSync } from "node:fs";
import { basename, extname } from "node:path";
import { Readable } from "node:stream";
import { pathToFileURL } from "node:url";

const TOKEN_HEADER = "x-sceneworks-token";
const DEFAULT_MIN_UPLOAD_BYTES = 100 * 1024 * 1024;
const DEFAULT_OBSERVATION_SECONDS = 110;
const REQUEST_TIMEOUT_MS = 120_000;
const RUNPOD_PROXY_HOST = /^[a-z0-9]+-\d+\.proxy\.runpod\.net$/;

function fail(message) {
  throw new Error(message);
}

function normalizeBaseUrl(rawBaseUrl, allowNonRunpod = false) {
  const url = new URL(rawBaseUrl);
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  if (!allowNonRunpod && (url.protocol !== "https:" || !RUNPOD_PROXY_HOST.test(url.hostname))) {
    fail(
      "SCENEWORKS_BASE_URL must be a RunPod HTTPS proxy origin such as " +
        "https://<podid>-8010.proxy.runpod.net",
    );
  }
  return url;
}

function endpoint(baseUrl, pathname) {
  return new URL(pathname, baseUrl);
}

async function responseError(response) {
  const text = (await response.text()).slice(0, 512);
  return new Error(
    `HTTP ${response.status} ${response.statusText}${text ? `: ${text}` : ""}`,
  );
}

async function jsonResponse(url, token, init = {}) {
  const headers = new Headers(init.headers);
  headers.set(TOKEN_HEADER, token);
  if (init.body && typeof init.body === "string" && !headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const response = await fetch(url, {
    ...init,
    headers,
    signal: init.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  if (response.status === 204) {
    return { data: null, status: response.status };
  }
  return { data: await response.json(), status: response.status };
}

async function jsonRequest(url, token, init = {}) {
  return (await jsonResponse(url, token, init)).data;
}

export async function* parseEventStream(stream) {
  const decoder = new TextDecoder();
  let buffer = "";
  let eventName = "message";
  let data = [];

  function emitEvent() {
    if (data.length === 0) {
      eventName = "message";
      return null;
    }
    const event = { event: eventName, data: data.join("\n") };
    eventName = "message";
    data = [];
    return event;
  }

  function consumeLine(line) {
    if (line === "") {
      return emitEvent();
    }
    if (line.startsWith(":")) {
      return null;
    }
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") {
      eventName = value;
    } else if (field === "data") {
      data.push(value);
    }
    return null;
  }

  function takeLine(final = false) {
    const lineEnd = buffer.search(/[\r\n]/);
    if (lineEnd === -1) {
      return null;
    }
    if (!final && buffer[lineEnd] === "\r" && lineEnd === buffer.length - 1) {
      return null;
    }
    const terminatorLength =
      buffer[lineEnd] === "\r" && buffer[lineEnd + 1] === "\n" ? 2 : 1;
    const line = buffer.slice(0, lineEnd);
    buffer = buffer.slice(lineEnd + terminatorLength);
    return { line };
  }

  for await (const chunk of stream) {
    buffer += decoder.decode(chunk, { stream: true });
    let framed;
    while ((framed = takeLine()) !== null) {
      const event = consumeLine(framed.line);
      if (event) {
        yield event;
      }
    }
  }
  buffer += decoder.decode();
  let framed;
  while ((framed = takeLine(true)) !== null) {
    const event = consumeLine(framed.line);
    if (event) {
      yield event;
    }
  }
  if (buffer) {
    const event = consumeLine(buffer);
    if (event) {
      yield event;
    }
  }
  const event = emitEvent();
  if (event) {
    yield event;
  }
}

function contentTypeFor(filePath) {
  switch (extname(filePath).toLowerCase()) {
    case ".mp4":
      return "video/mp4";
    case ".mov":
      return "video/quicktime";
    case ".webm":
      return "video/webm";
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".webp":
      return "image/webp";
    default:
      return "application/octet-stream";
  }
}

function multipartUpload(filePath) {
  const size = statSync(filePath).size;
  const boundary = `sceneworks-proxy-probe-${randomUUID()}`;
  const filename = basename(filePath).replaceAll('"', "_");
  const prefix = Buffer.from(
    `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="file"; filename="${filename}"\r\n` +
      `Content-Type: ${contentTypeFor(filePath)}\r\n\r\n`,
  );
  const suffix = Buffer.from(`\r\n--${boundary}--\r\n`);

  async function* bodyChunks() {
    yield prefix;
    yield* createReadStream(filePath);
    yield suffix;
  }

  return {
    body: Readable.from(bodyChunks()),
    contentLength: prefix.length + size + suffix.length,
    contentType: `multipart/form-data; boundary=${boundary}`,
    fileBytes: size,
  };
}

async function nextEvent(iterator, timeoutMs) {
  let timeout;
  const timeoutPromise = new Promise((resolve) => {
    timeout = setTimeout(() => resolve({ timeout: true }), timeoutMs);
  });
  try {
    return await Promise.race([iterator.next(), timeoutPromise]);
  } finally {
    clearTimeout(timeout);
  }
}

function parseEventJson(event) {
  try {
    return JSON.parse(event.data);
  } catch {
    return null;
  }
}

function redactedTarget(baseUrl) {
  if (!baseUrl.hostname.endsWith(".proxy.runpod.net")) {
    return `${baseUrl.protocol}//<test-host>`;
  }
  const internalPort =
    baseUrl.hostname.match(/-(\d+)\.proxy\.runpod\.net$/)?.[1] ?? "<port>";
  return `https://<redacted>-${internalPort}.proxy.runpod.net`;
}

const TERMINAL_JOB_STATUSES = new Set([
  "completed",
  "failed",
  "canceled",
  "interrupted",
]);

async function cleanupProbeJob(baseUrl, token, jobId) {
  let job = await jsonRequest(
    endpoint(baseUrl, `/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`),
    token,
    { method: "POST" },
  );
  const deadline = Date.now() + 30_000;
  while (!TERMINAL_JOB_STATUSES.has(job?.status) && Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 250));
    job = await jsonRequest(
      endpoint(baseUrl, `/api/v1/jobs/${encodeURIComponent(jobId)}`),
      token,
    );
  }
  if (!TERMINAL_JOB_STATUSES.has(job?.status)) {
    fail("placeholder job did not become terminal within 30 seconds of cleanup");
  }
  await jsonRequest(
    endpoint(baseUrl, `/api/v1/jobs/${encodeURIComponent(jobId)}/clear`),
    token,
    { method: "POST" },
  );
}

export async function validateProxy({
  rawBaseUrl,
  token,
  uploadFile,
  minimumUploadBytes = DEFAULT_MIN_UPLOAD_BYTES,
  observationSeconds = DEFAULT_OBSERVATION_SECONDS,
  minimumHeartbeats = 5,
  maximumEventLatencyMs = 10_000,
  allowNonRunpod = false,
  stopWhenSatisfied = false,
}) {
  if (!token) {
    fail("SCENEWORKS_ACCESS_TOKEN is required");
  }
  if (!uploadFile) {
    fail("--upload-file is required; use a representative image or video");
  }
  const uploadSize = statSync(uploadFile).size;
  if (uploadSize < minimumUploadBytes) {
    fail(
      `upload fixture is ${uploadSize} bytes; expected at least ${minimumUploadBytes} bytes`,
    );
  }

  const baseUrl = normalizeBaseUrl(rawBaseUrl, allowNonRunpod);
  const projects = await jsonRequest(endpoint(baseUrl, "/api/v1/projects"), token);
  if (!Array.isArray(projects) || projects.length === 0) {
    fail("SceneWorks has no project available for the proxy validation upload");
  }
  const project = projects[0];

  const ticket = await jsonRequest(
    endpoint(baseUrl, "/api/v1/jobs/events/ticket"),
    token,
    { method: "POST" },
  );
  if (!ticket?.ticket) {
    fail("event ticket response did not include a ticket");
  }

  const streamAbort = new AbortController();
  const streamUrl = endpoint(baseUrl, "/api/v1/jobs/events");
  streamUrl.searchParams.set("ticket", ticket.ticket);
  const streamStartedAt = performance.now();
  const streamResponse = await fetch(streamUrl, { signal: streamAbort.signal });
  if (!streamResponse.ok) {
    throw await responseError(streamResponse);
  }
  if (!streamResponse.headers.get("content-type")?.includes("text/event-stream")) {
    fail("job events endpoint did not return text/event-stream");
  }

  const events = parseEventStream(streamResponse.body)[Symbol.asyncIterator]();
  const first = await nextEvent(events, 20_000);
  if (first.timeout || first.done || first.value.event !== "ready") {
    streamAbort.abort();
    fail("SSE ready event did not arrive within 20 seconds");
  }
  const readyLatencyMs = Math.round(performance.now() - streamStartedAt);

  let job;
  try {
    const jobStartedAt = performance.now();
    job = await jsonRequest(endpoint(baseUrl, "/api/v1/jobs"), token, {
      method: "POST",
      body: JSON.stringify({
        type: "placeholder",
        projectId: project.id,
        projectName: project.name,
        payload: { proxyValidation: "sc-10368" },
        requestedGpu: "cpu",
      }),
    });
    if (!job?.id) {
      fail("placeholder job response did not include an id");
    }

    // `observationSeconds` is a *ceiling*, not a fixed dwell. A real proxy run
    // keeps `stopWhenSatisfied` false so the full window elapses -- that is what
    // proves RunPod's proxy does not sever an idle SSE stream. Callers that only
    // need to prove events flow (the tests) set it true, which turns the window
    // into "wait up to this long for the contract to be met" instead of a
    // wall-clock race that a scheduling stall can lose.
    const observationStartedAt = performance.now();
    const observationDeadline = observationStartedAt + observationSeconds * 1000;
    const heartbeatTimes = [];
    let jobEventLatencyMs = null;
    const observationSatisfied = () =>
      jobEventLatencyMs !== null && heartbeatTimes.length >= minimumHeartbeats;
    while (performance.now() < observationDeadline) {
      if (stopWhenSatisfied && observationSatisfied()) {
        break;
      }
      const remaining = observationDeadline - performance.now();
      const waitMs = Math.min(25_000, Math.max(1, remaining));
      const next = await nextEvent(events, waitMs);
      if (next.timeout) {
        if (waitMs < 25_000 || performance.now() >= observationDeadline) {
          break;
        }
        fail("SSE stream produced no event for 25 seconds");
      }
      if (next.done) {
        fail("SSE stream closed before the observation window completed");
      }
      if (next.value.event === "heartbeat") {
        heartbeatTimes.push(performance.now());
      }
      if (next.value.event === "job.updated" && parseEventJson(next.value)?.id === job.id) {
        jobEventLatencyMs ??= Math.round(performance.now() - jobStartedAt);
      }
    }

    if (jobEventLatencyMs === null) {
      fail("matching job.updated event was not observed through the proxy");
    }
    if (jobEventLatencyMs > maximumEventLatencyMs) {
      fail(
        `matching job.updated event took ${jobEventLatencyMs} ms (limit ${maximumEventLatencyMs} ms)`,
      );
    }
    if (heartbeatTimes.length < minimumHeartbeats) {
      fail(
        `only ${heartbeatTimes.length} heartbeat events arrived; expected ${minimumHeartbeats}`,
      );
    }
    const observedSeconds =
      Math.round(performance.now() - observationStartedAt) / 1000;

    const upload = multipartUpload(uploadFile);
    const uploadStartedAt = performance.now();
    let importedAsset;
    let uploadStatus;
    try {
      const uploadResponse = await jsonResponse(
        endpoint(
          baseUrl,
          `/api/v1/projects/${encodeURIComponent(project.id)}/assets`,
        ),
        token,
        {
          method: "POST",
          headers: {
            "content-type": upload.contentType,
            "content-length": String(upload.contentLength),
          },
          body: upload.body,
          duplex: "half",
        },
      );
      importedAsset = uploadResponse.data;
      uploadStatus = uploadResponse.status;
      if (!importedAsset?.id) {
        fail("upload response did not include an asset id");
      }
    } finally {
      if (importedAsset?.id) {
        await jsonRequest(
          endpoint(
            baseUrl,
            `/api/v1/projects/${encodeURIComponent(project.id)}/assets/${encodeURIComponent(importedAsset.id)}`,
          ),
          token,
          { method: "DELETE" },
        );
        await jsonRequest(
          endpoint(
            baseUrl,
            `/api/v1/projects/${encodeURIComponent(project.id)}/assets/${encodeURIComponent(importedAsset.id)}/purge?permanent=true`,
          ),
          token,
          { method: "DELETE" },
        );
      }
    }
    const uploadElapsedMs = Math.round(performance.now() - uploadStartedAt);

    return {
      target: redactedTarget(baseUrl),
      sse: {
        readyLatencyMs,
        jobEventLatencyMs,
        // `observationSeconds` is the configured ceiling; `observedSeconds` is
        // what the window actually spent. They are equal on a real proxy run
        // (`stopWhenSatisfied` false) and diverge when a caller opts into the
        // early break, so the logged record never overstates the observation.
        observationSeconds,
        observedSeconds,
        heartbeatCount: heartbeatTimes.length,
        remainedConnected: true,
      },
      job: { cleanedUp: true },
      upload: {
        fileBytes: upload.fileBytes,
        elapsedMs: uploadElapsedMs,
        status: uploadStatus,
        cleanedUp: true,
      },
    };
  } finally {
    streamAbort.abort();
    if (job?.id) {
      await cleanupProbeJob(baseUrl, token, job.id);
    }
  }
}

function parseCli(argv) {
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const value = argv[index + 1];
    if (arg === "--upload-file") {
      options.uploadFile = value;
      index += 1;
    } else if (arg === "--minimum-upload-bytes") {
      options.minimumUploadBytes = Number(value);
      index += 1;
    } else if (arg === "--observation-seconds") {
      options.observationSeconds = Number(value);
      index += 1;
    } else {
      fail(`unknown or incomplete argument: ${arg}`);
    }
  }
  return options;
}

async function main() {
  const cli = parseCli(process.argv.slice(2));
  const result = await validateProxy({
    rawBaseUrl: process.env.SCENEWORKS_BASE_URL,
    token: process.env.SCENEWORKS_ACCESS_TOKEN,
    ...cli,
  });
  console.log(JSON.stringify(result, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((error) => {
    console.error(`RunPod proxy validation failed: ${error.message}`);
    process.exitCode = 1;
  });
}
