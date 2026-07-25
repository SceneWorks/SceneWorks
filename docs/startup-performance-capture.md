# Startup and page-load timing capture

SceneWorks exposes a small, secret-safe timing surface for separating API work,
response transfer, and browser readiness. It is diagnostic telemetry in the
existing process log and browser Performance Timeline; it does not send data to
an external service.

## What is recorded

API responses emit an `api_request_duration` log event with only:

- the HTTP method;
- Axum's normalized route template, such as
  `/api/v1/projects/:project_id/assets`;
- the response status; and
- server duration in milliseconds.

Unknown API paths collapse to `/api/<unmatched>` and MCP paths collapse to
`/mcp/<unmatched>`. Raw URLs, query strings, IDs, tokens, media tickets,
prompts, names, and filesystem paths are never timing fields. Every API
response also carries `Server-Timing: app;dur=<milliseconds>`.

Before binding the listener, the API emits `startup_phase_duration` for these
fixed phases:

- `upload_sweeps`
- `jobs_retention_recovery`
- `reserved_project_initialization`
- `orphaned_asset_maintenance`

The web app records fixed `sceneworks.*` marks for bootstrap start, access
resolution, media authorization settlement, projects committed, active project
selection, asset request start/settle, and the first Assets-ready render.
Repeated asset refreshes clear and reuse the same entries. Startup marks are
one-shot, so the browser retains a fixed, bounded set.

## RunPod capture procedure

Use an immutable image tag or digest and record it with the Pod's GPU type,
region, volume state (cold or warm), browser version, and UTC capture time.
Never paste a Pod URL, access token, media ticket, project name, or full browser
network export into a ticket.

1. Start the Pod and save only `startup_phase_duration` log events. Copy the
   `phase` and `duration_ms` fields; omit unrelated log fields.
2. Open the RunPod proxy URL in a private browser window, start a Performance
   recording, authenticate normally, and wait until Assets renders.
3. In DevTools Console, run the sanitizing snippet below. It returns fixed mark
   names and allow-listed route templates only; it never returns resource URLs
   or query strings.
4. Repeat at least five cold browser loads. Use a fresh private window for each
   cold load. Keep cold-volume and warm-volume samples in separate tables.
5. Attach the sanitized tables to the Shortcut story before comparing any
   optimization result.

```js
const routeLabel = (pathname) => {
  if (pathname === "/api/v1/access") return pathname;
  if (pathname === "/api/v1/files/ticket") return pathname;
  if (pathname === "/api/v1/projects") return pathname;
  if (/^\/api\/v1\/projects\/[^/]+\/assets$/.test(pathname)) {
    return "/api/v1/projects/:project_id/assets";
  }
  return pathname.startsWith("/api/") ? "/api/<other>" : null;
};

const marks = performance
  .getEntriesByType("mark")
  .filter(({ name }) => name.startsWith("sceneworks."))
  .map(({ name, startTime }) => ({
    mark: name,
    millisecondsFromNavigation: Number(startTime.toFixed(3)),
  }));

const requests = performance
  .getEntriesByType("resource")
  .map((entry) => ({ entry, route: routeLabel(new URL(entry.name).pathname) }))
  .filter(({ route }) => route)
  .map(({ entry, route }) => {
    const server =
      entry.serverTiming.find(({ name }) => name === "app")?.duration ?? null;
    return {
      route,
      totalMs: Number(entry.duration.toFixed(3)),
      serverMs: server === null ? null : Number(server.toFixed(3)),
      transferMs: Number((entry.responseEnd - entry.responseStart).toFixed(3)),
      proxyNetworkAndQueueMs:
        server === null
          ? null
          : Number(
              Math.max(0, entry.responseStart - entry.requestStart - server).toFixed(3),
            ),
    };
  });

({ marks, requests });
```

Interpret the columns as follows:

- `serverMs` is application processing through response-header creation,
  reported by the API's `Server-Timing` header.
- `transferMs` is first response byte through final response byte and therefore
  includes proxy-to-browser body transfer.
- `proxyNetworkAndQueueMs` is an approximation of the remaining request wait
  outside the app. TLS, RunPod proxying, scheduling, and network latency are
  included, so do not label it server work.
- Client readiness is the delta from `sceneworks.bootstrap-start` to the
  relevant fixed mark, especially `sceneworks.assets-ready-render`.

For a same-origin RunPod deployment, `serverTiming` is available directly.
If it is absent, retain `serverMs: null` rather than estimating server time from
the resource duration; correlate the normalized API log event instead.

## Local pre-optimization evidence (not RunPod)

Captured 2026-07-25 on a Windows development host from an unoptimized Rust
debug build, loopback HTTP, no authentication, empty temporary data/config
directories, and ten sequential PowerShell requests per route:

| Normalized route | Samples | Total median (min-max) | Server median (min-max) |
| --- | ---: | ---: | ---: |
| `/api/v1/access` | 10 | 23.973 ms (23.372-24.940) | 0.138 ms (0.131-0.342) |
| `/api/v1/projects` | 10 | 22.188 ms (16.045-28.163) | 0.959 ms (0.878-1.094) |

The total includes substantial PowerShell `Invoke-WebRequest` overhead and is
not a browser transfer measurement. This local harness did not exercise the
RunPod proxy, TLS, a populated network volume, authentication, bundle loading,
or a real browser render. Its purpose is to prove that the server and
end-to-end values can be separated without secrets; it must not be used as the
RunPod performance baseline or compared directly with downstream RunPod
optimization results. The focused client timing test separately proves the
expected mark ordering and bounded repeated-asset behavior.
