# Deploy SceneWorks on a RunPod Pod

SceneWorks publishes a combined Linux/amd64 image containing the embedded web
UI, Rust API, candle CUDA workers, CPU utility worker, ffmpeg, and Model Manager
downloader:

```text
ghcr.io/sceneworks/sceneworks-runpod
```

The image is public, so RunPod does not need registry credentials. SceneWorks
still requires its own access token before it will listen on RunPod's
network-reachable interface.

## Before you deploy

1. In RunPod, create a **Secure Cloud network volume** with enough room for
   models and generated assets. Choose its region carefully: a network volume
   limits Pod selection to GPUs available in that datacenter. Network volumes
   must be attached when a Pod is created and cannot be attached later.
2. Open RunPod's **Secrets** section and create
   `sceneworks_access_token`. Give it a unique, high-entropy value; do not reuse
   a RunPod API key or Hugging Face token.
3. If you need gated Hugging Face repositories, also create an `hf_token`
   secret containing a Hugging Face read token for an account that has accepted
   the model's license.

RunPod expands a secret reference such as
`{{ RUNPOD_SECRET_sceneworks_access_token }}` into an environment variable
without putting the value in the template. See RunPod's
[environment-variable](https://docs.runpod.io/pods/templates/environment-variables)
and [Secret](https://docs.runpod.io/pods/templates/secrets) documentation.

## Choose an image version

For production, use an immutable release tag:

```text
ghcr.io/sceneworks/sceneworks-runpod:X.Y.Z
```

Exact Git release tags of the form `vX.Y.Z` publish both `:X.Y.Z` and
`:latest`. Prefer the immutable version tag for reproducible Pods. Do not assume
`:latest` exists before the first official release.

Manual publication tags (`manual-*`) are validation artifacts, not official
releases and are never promoted to `:latest`. Until the first official release,
the checked-in template uses the anonymously pullable validation tag
`manual-sc10367-2c4dd777560a`:

```text
ghcr.io/sceneworks/sceneworks-runpod:manual-sc10367-2c4dd777560a
```

That tag resolved during publication validation to this manifest:

```text
ghcr.io/sceneworks/sceneworks-runpod@sha256:fdd60e35655708915ea046a9db86093360a81c2946f08fe63cef58f59f9ab065
```

The manifest digest is the immutable verification evidence; the tag is used in
the template because RunPod's template API specifies `imageName` as an image
tag. After an official release, replace the template's `imageName` with the
desired `:X.Y.Z` tag before registering it. Do not substitute another
unreviewed `manual-*` tag for an official release.

## Create the reusable template

[`config/runpod-template.json`](../config/runpod-template.json) is a RunPod Pod
template payload with these safe defaults:

- the public SceneWorks image;
- NVIDIA Pod mode, not Serverless;
- `8010/http` as the only exposed port;
- the access token supplied through a RunPod Secret;
- `/workspace` as the volume base and mount point;
- no entrypoint override, start-command override, or raw TCP port.

Register it with RunPod's template API:

```bash
export RUNPOD_API_KEY='<your RunPod API key>'
curl --fail-with-body --request POST \
  --url https://rest.runpod.io/v1/templates \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary @config/runpod-template.json
```

The response includes the new template ID. The payload follows RunPod's current
[Create template API](https://docs.runpod.io/api-reference/templates/POST/templates).
You can instead create the same private template under **Templates → New
Template** by copying the image, port, environment, and storage values from the
JSON. If the template name already exists in your account, rename it before
registering.

## Deploy the Pod

1. Open **Pods → Deploy** and select the network volume created earlier.
2. Choose a supported NVIDIA GPU in the same datacenter.
3. Select the **SceneWorks GPU** custom template. Confirm that the network
   volume is mounted at `/workspace` and `8010` is exposed as an **HTTP** port.
   Attaching a network volume replaces the template's 20 GB fallback local
   volume; it does not change the `/workspace` mount path.
4. Confirm `SCENEWORKS_ACCESS_TOKEN` resolves from
   `{{ RUNPOD_SECRET_sceneworks_access_token }}`. If gated Hugging Face models
   are needed, add:

   ```text
   HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
   ```

5. Deploy the Pod. Do not override the image's entrypoint or start command.
6. Wait for the Pod telemetry and container logs to show that the API is
   healthy and the GPU and utility workers have started.

RunPod documents network-volume attachment and lifecycle constraints in
[Network volumes](https://docs.runpod.io/storage/network-volumes), and the
template fields in [Manage Pod templates](https://docs.runpod.io/pods/templates/manage-templates).

## Open and authenticate the UI

RunPod proxy URLs have the form
`https://<podid>-<port>.proxy.runpod.net`. With the template's default port,
open:

```text
https://<podid>-8010.proxy.runpod.net
```

RunPod's Connect panel also provides this link under **HTTP Services**. Its
[HTTP proxy URL](https://docs.runpod.io/pods/configuration/expose-ports) uses the
Pod's internal port and provides HTTPS in front of the container's HTTP service.

SceneWorks shows a login gate. Enter the same value stored in
`sceneworks_access_token`; the UI verifies it and uses it for authenticated API
requests. After login:

1. Open **Model Manager**.
2. Select an image or video model supported by the chosen GPU.
3. Choose **Download** and wait for the job to finish.
4. Run a small generation before starting a large job.

Model downloads, generated assets, imported models, projects, configuration,
and the Hugging Face cache persist below `/workspace`. The job queue database is
intentionally kept on ephemeral local storage at
`/tmp/sceneworks/cache/jobs.db`, because SQLite locking is unsafe on many
NFS-style network volumes.

## Environment reference

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `SCENEWORKS_ACCESS_TOKEN` | **Yes** | none | SceneWorks login/API credential. The public-bind entrypoint refuses to start if it is missing or blank. Supply it with a RunPod Secret reference. |
| `HF_TOKEN` | Only for gated Hugging Face repositories | unset | Hugging Face read token used by Model Manager. Add it as a separate RunPod Secret only when needed. |
| `SCENEWORKS_VOLUME` | No | `/workspace` | Absolute base for durable SceneWorks data, config, and model-cache directories. Keep it equal to the network-volume mount. |
| `SCENEWORKS_API_PORT` | No | `8010` | Internal API/UI port. If changed, update the template's HTTP port and proxy URL to the same value. |
| `SCENEWORKS_CORS_ORIGINS` | Only for a different-origin frontend | built-in local-development origins | Comma-separated, exact allowed origins such as `https://studio.example.com`. It is not needed for the default embedded UI served through the same RunPod proxy origin. |

The combined image intentionally defaults `SCENEWORKS_API_HOST` to `0.0.0.0` so
RunPod's proxy can reach it. Do not blank `SCENEWORKS_ACCESS_TOKEN` or set
`SCENEWORKS_ALLOW_OPEN_BIND`; the combined image removes the latter and refuses
an unauthenticated public bind.

## Security and troubleshooting

- Expose `8010/http`, not `8010/tcp`. RunPod's HTTP proxy provides HTTPS, but
  HTTPS is transport protection, not application authentication; keep the
  SceneWorks token enabled.
- Never publish a raw TCP port for SceneWorks without an additional,
  independently verified authentication and TLS boundary. A raw TCP mapping
  bypasses RunPod's HTTPS proxy.
- Treat the proxy URL as public. Do not put the SceneWorks token in its query
  string, browser history, screenshots, logs, template JSON, or support
  messages.
- A Pod can be marked Running before its service is ready. Check telemetry and
  logs, then retry the HTTP Services link.
- If startup reports that a managed directory is not writable, verify that the
  volume is attached at `/workspace` and that its export/ACL permits writes by
  the container's root user.
- If a model remains gated, verify that `HF_TOKEN` is present, the Hugging Face
  account has accepted the repository's license, and the Pod was restarted
  after adding the environment variable.
- RunPod documents a 100-second Cloudflare proxy timeout for a service that has
  not responded. SceneWorks generation uses short asynchronous API requests,
  and `/api/v1/jobs/events` responds immediately before carrying progress as
  SSE. The stream sends a 15-second heartbeat so an otherwise idle connection
  continues to produce traffic. The web client reconnects with a new
  short-lived event ticket if the connection is interrupted.
- RunPod does not currently document a separate Pod HTTP-proxy request-body
  limit. SceneWorks itself accepts a streaming multipart body up to 2 GiB, so
  the file must remain slightly below that ceiling to leave room for framing.
  Upload success still depends on sending the request body and receiving the
  response within the proxy's timeout; use a stable uplink and retry a failed
  import. Transcode or split very large video into parts that fit the tested
  envelope rather than exposing an unauthenticated raw TCP port.

## Validate the proxy path

Before treating a new Pod, region, or RunPod proxy revision as production
ready, verify the actual path with a real media file. The repository's probe:

- connects to the authenticated job-event stream and requires the initial
  `ready` event;
- creates a harmless placeholder job and requires its matching `job.updated`
  event within 10 seconds, then cancels if necessary and clears the probe job;
- keeps that same SSE connection open for 110 seconds, past the documented
  100-second boundary, and requires at least five 15-second heartbeat events;
- streams a representative image or video of at least 100 MiB as multipart
  data through the public proxy, requires a successful asset import, and
  deletes the imported probe asset afterward.

The token is accepted only through the environment and is never printed. The
result also redacts the Pod ID. On macOS/Linux:

```bash
export SCENEWORKS_BASE_URL='https://<podid>-8010.proxy.runpod.net'
read -rsp 'SceneWorks access token: ' SCENEWORKS_ACCESS_TOKEN
export SCENEWORKS_ACCESS_TOKEN
node scripts/validate-runpod-proxy.mjs \
  --upload-file /path/to/representative-large-video.mp4
unset SCENEWORKS_ACCESS_TOKEN
```

On PowerShell:

```powershell
$env:SCENEWORKS_BASE_URL = 'https://<podid>-8010.proxy.runpod.net'
$env:SCENEWORKS_ACCESS_TOKEN = Read-Host 'SceneWorks access token'
node scripts/validate-runpod-proxy.mjs `
  --upload-file C:\path\to\representative-large-video.mp4
Remove-Item Env:SCENEWORKS_ACCESS_TOKEN
```

Also keep the browser UI open while submitting a real generation job. Confirm
that its queue card and progress change before the job completes; the probe
validates the same transport, while this manual check confirms the UI consumes
it. Record the file size, elapsed upload time, event latency, heartbeat count,
observation duration, and any `524` or disconnect. Never record the access
token, event ticket, full proxy hostname, or Pod ID in committed evidence.

### Measured live result

On 2026-07-24, the probe passed through a real RunPod Pod HTTPS proxy:

- the initial SSE `ready` event arrived in 528 ms;
- the matching `job.updated` event arrived in 464 ms, before the job request
  completed its wider workflow;
- the same connection remained open for the full 110-second observation and
  delivered seven heartbeat events, with no buffering or premature disconnect;
- in the browser UI at 2026-07-24T22:45:51Z, the proxied queue showed the same
  placeholder job live at Preparing / 10% on the CPU utility worker, then
  Completed / 100% after six seconds, without using a direct-origin URL;
- a valid 105,000,054-byte (100.14 MiB) BMP completed as a streaming multipart
  upload in 33.851 seconds with HTTP 201; and
- the probe asset was trashed and permanently purged after validation.

This establishes the tested body-size envelope at 100.14 MiB; it does not claim
that RunPod has no higher undocumented cap. The 110-second connected stream
also shows that the documented 100-second timeout does not terminate an SSE
response that starts immediately and continues delivering heartbeat traffic.

RunPod's current [HTTP proxy documentation](https://docs.runpod.io/pods/configuration/expose-ports)
describes the proxy chain and timeout.
