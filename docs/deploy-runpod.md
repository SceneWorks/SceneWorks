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
- RunPod's HTTP proxy currently has a 100-second request limit. SceneWorks
  generation runs as asynchronous jobs, but uploads or unrelated synchronous
  clients should account for that proxy limit.
