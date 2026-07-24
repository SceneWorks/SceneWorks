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

## Supported GPUs

The v1 image uses `CUDA_COMPUTE_CAP=80`, and its complete support contract has
two parts:

- Candle's general CUDA kernels include `compute_80` PTX, which the NVIDIA
  driver can JIT for compute capability 8.0 or newer.
- The quantized GGUF/MoE kernels are a separate fat binary with `sm_80`,
  `sm_90`, and `sm_120` cubins plus `compute_120` PTX. NVIDIA's same-major
  binary compatibility lets the `sm_80` cubin run on 8.6 and 8.9 GPUs.

The image build inspects that quantized-kernel fat binary with `cuobjdump`
before packaging it. Based on that whole-image contract, use this matrix:

| Architecture | Compute capability | Common RunPod GPUs | v1 status |
| --- | --- | --- | --- |
| Ampere | 8.0, 8.6 | A100, A30, A40, A10, RTX A6000/A5000/A4000, RTX 30-series | **Supported.** `sm_80` covers the quantized path; the general path carries `compute_80` PTX. |
| Ada | 8.9 | L4, L40/L40S, RTX 6000/5000/4500 Ada, RTX 40-series | **Supported.** NVIDIA guarantees an `sm_80` cubin on later 8.x devices. |
| Hopper | 9.0 | H100, H200 | **Supported.** The quantized path includes native `sm_90`; the general path JITs `compute_80` PTX. |
| Consumer/workstation Blackwell | 12.0 | RTX PRO 6000/5000/4500/4000/2000 Blackwell, RTX 50-series | **Supported.** The quantized path includes native `sm_120`; the general path JITs `compute_80` PTX. |
| Turing | 7.5 | T4, RTX 20-series, Quadro RTX | **Unsupported in v1.** Neither `compute_80` PTX nor an 8.x-or-newer cubin can run backward on `sm_75`. |
| Datacenter Blackwell | 10.0, 10.3 | B200/GB200, B300/GB300 | **Unsupported in v1.** General kernels can JIT from `compute_80`, but the quantized fat binary has no `sm_100`/`sm_103` cubin and its `compute_120` PTX cannot JIT backward. |

On 2026-07-24, `cuobjdump` inspection of the worker executable extracted
from the public validation manifest
`sha256:fdd60e35655708915ea046a9db86093360a81c2946f08fe63cef58f59f9ab065`
reported `sm_80`, `sm_90`, and `sm_120` cubins plus `sm_120` PTX. This
checks the shipped artifact, while the Docker build guard prevents future
publications from dropping those targets unnoticed.

This is an image-level support statement, not a promise that every model fits
every supported card. Check the model's VRAM requirement separately. RunPod's
GPU inventory also changes over time; confirm the exact product name and
compute capability shown for the Pod. NVIDIA maintains the authoritative
[GPU compute-capability table](https://developer.nvidia.com/cuda/gpus), and its
[CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)
defines cubin and PTX compatibility.

For v1, SceneWorks deliberately keeps the `sm_80` baseline and does not add
Turing code to this already-large image. Expanding the general and quantized
kernel builds together for Turing and datacenter Blackwell is tracked in
`sc-14423`; changing only the top-level `CUDA_COMPUTE_CAP` would not cover every
native CUDA archive.

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
