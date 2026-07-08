# SceneWorks MCP server (epic 10231)

SceneWorks embeds a [Model Context Protocol](https://modelcontextprotocol.io)
server in the Rust API. Any MCP-capable client — Claude Code, Cursor, a custom
agent — can drive generation directly: list projects and the model/LoRA
catalog, generate images (returned inline), and submit/poll video jobs with
ticketed download links. This is the operator's guide to turning it on, wiring
a client to it (same machine or across the LAN), and the security posture of
exposing it beyond loopback.

## What is served where

The MCP endpoint is `/mcp` (MCP **streamable-HTTP** transport, via the
official `rmcp` SDK — primarily `POST`, but the transport also uses `GET` for
the SSE stream and `DELETE` for session teardown, which is why the auth gate
below covers every method). It is mounted inside the same axum app as `/api/v1/*`
(`apps/rust-api`). There is no separate process, port, or feature flag: if the
SceneWorks API is running, `/mcp` is running. It is served by:

- the **desktop app** (the `sceneworks-rust-api` sidecar),
- **Docker Compose** (`npm run dev` — API published on host port 8010 by default),
- a **direct binary run** (`cargo run -p sceneworks-rust-api`, default port 8000).

Every MCP request passes through the same `access_control` middleware as the
API routes — `/mcp` is gated for **every** HTTP method, exactly like a
`/api/v1` route (`apps/rust-api/src/auth.rs::requires_token`). The MCP layer
adds no authentication of its own, and its tools are a thin HTTP client over
the existing `/api/v1/*` surface (no side-door into the engine or DB).

### Tools

| Tool | Kind | What it does |
| --- | --- | --- |
| `list_projects` | read | Project ids/names — `projectId` for every other call |
| `list_models` | read | Generation model catalog: id, family, type (image/video), capabilities, `installState`, defaults, resolutions |
| `list_loras` | read | LoRA adapter catalog (filter by `modelFamily` / `projectId`) |
| `generate_image` | **blocking** | Submits an image job, relays progress notifications, returns the images inline as base64 (default `count` 1) |
| `submit_video_job` | non-blocking | Submits a video job (`generate` / `extend` / `bridge` / `person_replace`) and returns the job id immediately |
| `get_job_status` | read | Status/stage/progress/eta for any job id (video and image) |
| `get_job_result` | read | For a completed job: ticketed download links (media bytes are never inlined) |

`generate_image` blocks until the job is terminal (poll interval 1 s, overall
deadline 30 min — generous because a cold first run legitimately spends minutes
downloading/loading the model). Video runs for minutes, so it is submit → poll
`get_job_status` → `get_job_result`.

## Environment variables

All knobs are on the API process (`Settings::from_env` in
`apps/rust-api/src/server.rs`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCENEWORKS_API_HOST` | `127.0.0.1` | Bind address. Loopback by default — set `0.0.0.0` to serve the LAN. |
| `SCENEWORKS_API_PORT` | `8000` | Bind port (Docker Compose publishes `8010` by default). |
| `SCENEWORKS_ACCESS_TOKEN` | *(empty)* | The access token. Empty = auth off (loopback-only use). Required in practice for any non-loopback bind (see the open-bind guard). |
| `SCENEWORKS_API_URL` | derived from bind host/port | Base URL the MCP server uses to call back into its own API, and the **fallback** base for `get_job_result`'s ticket URLs (which now default to the incoming request's `Host` — see below). Defaults to the bound interface (`sc-10260`); override for reverse-proxy/container setups. |
| `SCENEWORKS_TRUST_LOOPBACK` | off | `1`/`true` = requests from `127.0.0.1`/`::1` bypass the token. The desktop sets it; never set it behind a reverse proxy or on a shared multi-user machine. |
| `SCENEWORKS_ALLOW_OPEN_BIND` | off | `1`/`true`/`yes` = allow a non-loopback bind with **no** token (the API refuses to start otherwise). Only for fully trusted networks. |
| `SCENEWORKS_MCP_JOB_POLL_INTERVAL` | `1` | Seconds between status polls for the blocking `generate_image` tool. A zero value falls back to the default. |
| `SCENEWORKS_MCP_JOB_TIMEOUT` | `1800` | Seconds the blocking `generate_image` tool waits for a job before returning a timeout error (the job itself keeps running / is not canceled). Clamped to ≥ the poll interval. |

### `SCENEWORKS_API_URL` on the LAN

`SCENEWORKS_API_URL` (`Settings.mcp_api_url`) is the base for the MCP tools'
outbound self-calls into `/api/v1/*`. When unset it is derived from the bind
host/port: a wildcard/loopback bind self-dials `127.0.0.1`, and a specific
interface bind (e.g. `SCENEWORKS_API_HOST=192.168.4.97`) self-dials that
interface (sc-10260). Override it only for reverse-proxy / container setups
where the process cannot reach its own `/api/v1` at the bound address:

```text
SCENEWORKS_API_URL=http://192.168.4.97:8000
```

**Ticket download URLs** (`get_job_result`) no longer depend on this variable:
because `/mcp` and `/api/v1` are the same app, the absolute URL host is derived
per-request from the `Host` (or `X-Forwarded-Host`/`-Proto`) the client used to
reach `/mcp`, so it is reachable by that client regardless of `SCENEWORKS_API_URL`
(sc-10290). `SCENEWORKS_API_URL` (or the loopback default) is only the fallback
when the request carries no usable Host. Each asset also carries a `relativeUrl`,
and the result `note` says to re-base it onto whatever host reaches `/mcp`, so an
edge case is still recoverable client-side.

## Security posture for LAN exposure

Short version: **token-authenticated plain HTTP on a trusted LAN.** There is
no TLS and no user model — one token, full access. Do not expose this to
untrusted networks or the internet; if you must go wider than a LAN, front it
with a TLS reverse proxy that does its own auth (and leave
`SCENEWORKS_TRUST_LOOPBACK` off, since every proxied request would look like
loopback).

What the API enforces, and where it lives in code:

- **Token gate** (`apps/rust-api/src/auth.rs`): with `SCENEWORKS_ACCESS_TOKEN`
  set, every `/mcp` request (any method) and every gated `/api/v1` route must
  present the token as `X-SceneWorks-Token: <token>` **or**
  `Authorization: Bearer <token>`. Comparison is constant-time. Missing/wrong
  token → `401 {"detail":"SceneWorks access token required","authRequired":true}`.
- **Brute-force throttle** (sc-8870): per-peer-IP rolling window — more than
  10 failed token attempts within 60 s → `429 Too Many Requests` for that IP
  until the window rolls off (~60 s after the last failure). A valid token
  clears the peer's budget; loopback-trusted peers never accrue failures.
  Failures are logged (`auth_rejected` / `auth_throttled` events), never the
  token itself.
- **Open-bind guard** (sc-4201/sc-5720): binding to a non-loopback address
  with an empty token makes the process **refuse to start**:
  `Refusing to bind to <addr> with no SCENEWORKS_ACCESS_TOKEN set …`. Override
  only with `SCENEWORKS_ALLOW_OPEN_BIND=1`, which starts anyway and logs a
  loud `open_bind_without_token` warning. Never combine the override with a
  LAN bind outside a fully trusted network — every endpoint (project files,
  credential writes, job creation, uploads, MCP) becomes anonymous.
- **Loopback trust** (epic 4484): with `SCENEWORKS_TRUST_LOOPBACK=1`,
  connections whose *peer address* is loopback bypass the token; all other
  source IPs stay gated. This is how the desktop keeps its embedded UI and
  local workers password-free while the LAN needs the pairing token. Caveats:
  trust is per-connection, not per-OS-user (any local process/user on the
  machine inherits it — see "Loopback trust and the multi-user-machine
  caveat" in the root README), and behind a reverse proxy every request looks
  like loopback, so leave it unset there.
- **Media tickets** (sc-8810 flavor): `get_job_result` mints one short-lived
  ticket (`POST /api/v1/files/ticket`, itself auth-gated) and embeds it as
  `?ticket=` in each download URL, so the links work from any machine without
  an auth header — but only for `GET` on the read-only media routes, and only
  for the ticket TTL (**300 s**, sliding). A leaked URL dies at most one TTL
  after the last refresh; call `get_job_result` again for fresh links.
- **DNS-rebinding note**: rmcp's built-in `allowed_hosts` check is
  deliberately disabled (its loopback-only default would 403 the supported
  LAN deployment); the token gate above is the access control for `/mcp`.

## Server setup

### Same machine only (no token)

Nothing to do — run the desktop app or `cargo run -p sceneworks-rust-api` and
point your client at `http://127.0.0.1:8000/mcp` (desktop) /
`http://127.0.0.1:8010/mcp` (Docker). With no token configured and a loopback
bind, requests from this machine are accepted as-is. If you *have* set a token
and want local clients to skip it, set `SCENEWORKS_TRUST_LOOPBACK=1`.

### LAN-reachable with a token

Direct binary / server (PowerShell shown; same variables everywhere):

```powershell
$env:SCENEWORKS_API_HOST   = "0.0.0.0"
$env:SCENEWORKS_API_PORT   = "8000"
$env:SCENEWORKS_ACCESS_TOKEN = "choose-a-private-token"
$env:SCENEWORKS_API_URL    = "http://192.168.4.97:8000"   # this machine's LAN address
cargo run --release -p sceneworks-rust-api
```

- **Host firewall**: binding `0.0.0.0` makes the API *listen* on the LAN, but
  Windows Defender Firewall (and Linux equivalents) typically still blocks
  inbound connections from **other** hosts to a fresh `cargo run` binary on a
  new port — and you won't notice from the server itself, because same-box
  traffic to the machine's own LAN IP never traverses the inbound firewall.
  Answer "Allow access" on the first-run firewall prompt, or open the port
  explicitly, e.g. (admin PowerShell):

  ```powershell
  New-NetFirewallRule -DisplayName "SceneWorks API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
  ```

  If a remote client times out while a local `curl` to the LAN IP works, the
  firewall is the first suspect.
- Generation needs at least one worker running against the same API
  (`SCENEWORKS_WORKER_ONLY=1` + `SCENEWORKS_API_URL` +
  `SCENEWORKS_ACCESS_TOKEN`, or just use the desktop app / Docker Compose,
  which manage workers for you).
- Docker Compose: set `SCENEWORKS_API_PUBLISH_HOST=0.0.0.0` and
  `SCENEWORKS_ACCESS_TOKEN` in `.env` (see `.env.example`), and point clients
  at port `8010`. **Caveat**: `docker-compose.yml` does not pass
  `SCENEWORKS_API_URL` into the `api` service's environment (it is only set
  for the worker services, as `http://api:<port>`), so putting it in `.env`
  has no effect on the API container — the MCP self-calls still work (loopback
  inside the container), but `get_job_result`'s absolute download URLs stay
  `http://127.0.0.1:8010/...`. Until that's plumbed, either lean on the
  `relativeUrl` re-base fallback described above, or add the variable via a
  `docker-compose.override.yml` (picked up automatically by compose):

  ```yaml
  services:
    api:
      environment:
        SCENEWORKS_API_URL: http://192.168.4.97:8010
  ```
- The desktop app's LAN remote-access mode already binds `0.0.0.0`, uses the
  pairing password as the token, and sets `SCENEWORKS_TRUST_LOOPBACK` for its
  own local processes.

## Client configuration

The examples use the LAN setup above (`http://192.168.4.97:8000/mcp`, token
`choose-a-private-token`). For a same-machine client, substitute
`http://127.0.0.1:8000/mcp` and drop the header if no token is configured (or
if `SCENEWORKS_TRUST_LOOPBACK=1`).

### Claude Code (CLI)

```bash
claude mcp add --transport http sceneworks http://192.168.4.97:8000/mcp --header "X-SceneWorks-Token: choose-a-private-token"
```

Keep `--header` **after** the name and URL — it is a variadic flag, and placed
before them it swallows the positional arguments (`error: missing required
argument 'name'`). Add `--scope user` to register it for all your projects
(default is project-local). `Authorization: Bearer choose-a-private-token`
works as the header too. Verify with `claude mcp get sceneworks` /
`claude mcp list`, which health-check the connection.

Or as a checked-in `.mcp.json` (project scope — use env expansion so the token
stays out of git):

```json
{
  "mcpServers": {
    "sceneworks": {
      "type": "http",
      "url": "http://192.168.4.97:8000/mcp",
      "headers": {
        "X-SceneWorks-Token": "${SCENEWORKS_ACCESS_TOKEN}"
      }
    }
  }
}
```

Token-free loopback variant (server started with
`SCENEWORKS_TRUST_LOOPBACK=1`, client on the same machine — no `--header`):

```bash
claude mcp add --transport http sceneworks http://127.0.0.1:8000/mcp
```

### Cursor

`.cursor/mcp.json` in the project (or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "sceneworks": {
      "url": "http://192.168.4.97:8000/mcp",
      "headers": {
        "X-SceneWorks-Token": "${env:SCENEWORKS_ACCESS_TOKEN}"
      }
    }
  }
}
```

(No `type` field — Cursor infers the HTTP transport from `url`. Cursor's env
expansion syntax is `${env:VAR}`, unlike Claude Code's `${VAR}`.)

### Smoke-testing without an MCP client

The transport is plain JSON-RPC over HTTP, so `curl` works for a quick check
(the server replies as an SSE stream; session id comes back in the
`Mcp-Session-Id` response header of `initialize`):

```bash
curl -si http://192.168.4.97:8000/mcp \
  -H "X-SceneWorks-Token: choose-a-private-token" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

Expected failure modes: no/wrong token → `401` with
`{"detail":"SceneWorks access token required","authRequired":true}`; more than
10 bad tokens in a minute from one IP → `429`.

## Typical flow

1. `list_projects` → pick a `projectId` (or create a project in the UI first).
2. `list_models` → pick an image model with `"installState": "installed"`
   (anything not installed downloads on first use, which can be tens of GB).
3. `generate_image` with `projectId`, `prompt`, `model`, and optionally
   `width`/`height`/`count`/`seed`/`loras` — images come back inline; the JSON
   summary block carries the created asset ids.
4. `submit_video_job` with `projectId`, `prompt`, and optionally a video
   `model` (type `"video"` in `list_models`), `duration`/`fps`/`width`/
   `height`/`quality` — returns `jobId`.
5. Poll `get_job_status` until `"status": "completed"`, then `get_job_result`
   → download each asset URL within the ticket TTL (300 s; re-call for fresh
   links). If the absolute URL host is not reachable from your machine, apply
   `relativeUrl` to the base you use to reach the MCP server.

## Known limitations

- End-to-end validation from a physically separate second machine on the LAN
  (true NIC-to-NIC + Windows Defender Firewall traversal) is still outstanding
  (sc-10301); the validation record below was run against the host's own LAN IP,
  which exercises the auth/throttle/ticket paths but not separate-hardware
  reachability.

Earlier follow-ups filed during epic execution are now resolved: ticket URLs
derive from the request Host (sc-10290), MCP-call cancellation propagates to the
job (sc-10276), the blocking-wait poll interval/deadline are operator-configurable
(sc-10277), and the self-call base auto-derives from the bound interface (sc-10260).

## Validation record

Validated live on 2026-07-07 on Windows 11 (2× RTX PRO 6000 Blackwell,
candle/CUDA worker built `--features backend-candle` sm_120): API bound
`0.0.0.0:8000` with a test token, `SCENEWORKS_TRUST_LOOPBACK=1` and
`SCENEWORKS_API_URL=http://192.168.4.97:8000`; all "LAN" calls were made
against the machine's LAN address (`192.168.4.97` — a non-loopback peer, the
same auth path a second machine exercises). **Scope**: these same-box LAN-IP
calls exercise the non-loopback auth/throttle/ticket paths, but not true
NIC-to-NIC reachability from separate hardware — notably the Windows Defender
Firewall inbound rules above, which same-box own-IP traffic never traverses.
A run from a physically second machine on the LAN is tracked as follow-up
story sc-10301 (filed, needs hands on a second box):

- Open-bind guard: `SCENEWORKS_API_HOST=0.0.0.0` with no token → the process
  refused to start with the documented `Refusing to bind …` error (exit 1).
- LAN `POST /mcp` and `GET /api/v1/projects` with no token → `401` +
  `authRequired`; with `X-SceneWorks-Token` → MCP `initialize` succeeds
  (session id issued); `Authorization: Bearer` accepted equivalently.
- Brute-force throttle: 12 bad-token requests from the LAN IP →
  `401 ×10` then `429`; a **valid** token while blocked also `429`; loopback
  stayed `200` throughout; the valid token was accepted again after the ~60 s
  window rolled off.
- Loopback `initialize` + tool calls with **no** token succeed under
  `SCENEWORKS_TRUST_LOOPBACK=1`.
- Claude Code CLI: `claude mcp add --transport http sceneworks-val
  http://192.168.4.97:8000/mcp --header "X-SceneWorks-Token: …"` →
  `claude mcp get` health-check reports `✓ Connected` (real MCP client
  initialize over the token-authenticated LAN path).
- `tools/list` returns all seven tools; `list_projects`, `list_models`
  (59 models with install state), `list_loras` return the compacted catalogs.
- `generate_image` (`z_image_turbo`, 1024×1024, count 1, seed 42) over the LAN
  → 10 progress notifications, one inline base64 `image/png` (valid PNG magic,
  1.4 MB) + asset-id summary; end-to-end on the candle GPU worker.
- `submit_video_job` (`wan_2_2_t2v_14b`, 832×480, 1 s @ 8 fps, draft) →
  `get_job_status` polls (`queued` → `running`/`generating` → `completed`,
  ~3 min) → `get_job_result` returned a `resource_link` whose absolute URL
  used the `SCENEWORKS_API_URL` LAN base; fetching it from the LAN address
  with **no auth header** returned the MP4 bytes (`ftyp` magic, 232 KB), and
  the same path without a `?ticket=` → `401`. `get_job_result` on a
  still-running job correctly returned `ready=false` instead of an error.
