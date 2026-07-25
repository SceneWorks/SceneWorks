# Deploy and run SceneWorks on RunPod

This guide takes you from an empty RunPod account to a working SceneWorks web
app on an NVIDIA GPU. No SSH session, Docker Compose stack, or container
registry login is required.

The RunPod image contains the complete application:

- the SceneWorks web interface;
- the authenticated Rust API;
- one candle/CUDA worker for each visible NVIDIA GPU;
- a CPU utility worker for model downloads and media tasks; and
- ffmpeg and the native Model Manager downloader.

You will create one secret, one persistent network volume, one private template,
and one Pod. The web interface and API share a single HTTPS address.

## Before you start

You need:

- a RunPod account with enough credit to deploy an on-demand GPU;
- a private, randomly generated SceneWorks access token;
- about 15 minutes for setup, plus the time needed to pull the image and
  download your chosen model; and
- optionally, a Hugging Face read token for gated models.

For a first deployment, use at least **100 GB** of network-volume storage.
Individual models are large, and keeping them on the network volume prevents
another download when you replace the Pod.

The image is public:

```text
ghcr.io/sceneworks/sceneworks-runpod
```

RunPod does not need GitHub Container Registry credentials. SceneWorks still
requires its own access token because the web service is reachable through a
public proxy URL.

## 1. Choose the image version

For an official release, use its immutable version tag:

```text
ghcr.io/sceneworks/sceneworks-runpod:X.Y.Z
```

Replace `X.Y.Z` with the SceneWorks release version. Prefer this versioned tag
over `latest` so a future Pod starts the exact version you tested.
Do not assume `:latest` exists before the first official release.

Until the first official release, the repository template uses this public
validation build:

```text
ghcr.io/sceneworks/sceneworks-runpod:manual-sc14427-4fe122b7dba3
```

Its immutable OCI index digest is:

```text
ghcr.io/sceneworks/sceneworks-runpod@sha256:376182ddbdf4c78d2a6f66a1e3ce66c573145b4c21b99300e42aeefba9f710ab
```

Tags beginning with `manual-` are validation artifacts, not official releases.
Do not replace a release tag with an unreviewed manual tag.

## 2. Create the SceneWorks secret

1. In the RunPod console, open **Settings -> Secrets**.
2. Select **New Secret**.
3. Name it `sceneworks_access_token`.
4. Paste a unique, high-entropy random value and save it.

This value is the password you will enter when SceneWorks opens. Do not reuse a
RunPod API key, GitHub token, or Hugging Face token.

The template will refer to the secret without containing its value:

```text
{{ RUNPOD_SECRET_sceneworks_access_token }}
```

If you plan to use a gated Hugging Face model:

1. Accept the model's license on Hugging Face.
2. Create a read-only Hugging Face access token.
3. Create a second RunPod secret named `hf_token`.

Do not add `HF_TOKEN` unless you need it. RunPod explains secret creation and
template references in its
[Secrets documentation](https://docs.runpod.io/pods/templates/secrets).

## 3. Create persistent storage

1. In RunPod, open **Storage -> Network Volumes**.
2. Create a **Secure Cloud** network volume.
3. Choose at least **100 GB** for a useful model cache.
4. Note the datacenter you selected.

The volume's datacenter controls which GPUs can use it. A network volume must be
selected when the Pod is created; it cannot be attached later.

SceneWorks uses `/workspace` for durable data:

| Path | Contents |
| --- | --- |
| `/workspace/data` | Projects, generated assets, imported models, and application data |
| `/workspace/config` | SceneWorks configuration and user manifests |
| `/workspace/cache/huggingface` | Downloaded Hugging Face model files |

The live job queue is intentionally stored on the Pod's local disk. Projects,
assets, configuration, and model files persist; completed queue cards do not
need to survive a Pod replacement.

See RunPod's
[Network volumes documentation](https://docs.runpod.io/storage/network-volumes)
for current region and lifecycle rules.

## 4. Create a private Pod template

The easiest route is the RunPod web console:

1. Open **Templates -> New Template**.
2. Name it `SceneWorks GPU`.
3. Enter the values below.
4. Leave the Docker entrypoint and start command empty.
5. Save it as a private Pod template.

| Template field | Value |
| --- | --- |
| Container image | The versioned image chosen in step 1 |
| Container disk | `50 GB` |
| Volume disk | `20 GB` fallback; the network volume selected at deployment replaces it |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8010` (stored as `8010/http` in the template API) |
| Expose TCP ports | Leave empty |
| `SCENEWORKS_ACCESS_TOKEN` | `{{ RUNPOD_SECRET_sceneworks_access_token }}` |
| `SCENEWORKS_API_PORT` | `8010` |
| `SCENEWORKS_VOLUME` | `/workspace` |

For gated Hugging Face models, also add:

```text
HF_TOKEN={{ RUNPOD_SECRET_hf_token }}
```

The checked-in [`config/runpod-template.json`](../config/runpod-template.json)
contains the same fields. RunPod documents the web form in
[Manage Pod templates](https://docs.runpod.io/pods/templates/manage-templates).

Important:

- expose `8010` as an **HTTP** port, not a raw TCP port;
- never paste a secret value directly into the template;
- do not override the image's entrypoint or start command; and
- keep the network-volume mount and `SCENEWORKS_VOLUME` set to the same path.

## 5. Deploy the Pod

1. Open **Pods -> Deploy**.
2. Select the network volume created in step 3.
3. Choose an NVIDIA GPU available in that volume's datacenter.
4. Select the private **SceneWorks GPU** template.
5. Confirm the volume is mounted at `/workspace`.
6. Confirm the template exposes `8010` under **HTTP Services**.
7. Deploy the Pod.

Two GPUs have passed end-to-end SceneWorks validation:

| GPU | VRAM | What was validated |
| --- | ---: | --- |
| NVIDIA RTX PRO 4500 Blackwell | 32 GB | Cold start, GPU and CPU worker registration, model download, image generation, reduced-memory video generation, stop/start persistence |
| NVIDIA A40 | 48 GB | Cold start, CUDA execution, GPU and CPU worker registration |

Other supported Ampere, Ada, Hopper, and consumer/workstation Blackwell GPUs are
listed in [Supported GPU architectures](#supported-gpu-architectures). Model
memory requirements still apply: an image may support a GPU architecture even
when a particular model is too large for that card.

## 6. Wait for SceneWorks to become ready

The Pod can show **Running** while it is still pulling the image or starting the
workers.

1. Expand the Pod and open its container logs.
2. Wait for messages saying the API is healthy and the candle GPU and utility
   workers are starting.
3. Open **Connect**.
4. Under **HTTP Services**, open the link for port `8010`.

The address has this general form:

```text
https://<podid>-<port>.proxy.runpod.net
```

With the default port, that becomes:

```text
https://<podid>-8010.proxy.runpod.net
```

Use the link supplied by RunPod instead of typing the Pod ID yourself. The proxy
provides transport encryption: RunPod's HTTP proxy provides HTTPS. The
SceneWorks token provides application authentication.

At the SceneWorks login screen, enter the value stored in
`sceneworks_access_token`.

## 7. Verify the deployment and run a generation

Before downloading a large model:

1. Open **Queue** and confirm SceneWorks reports one GPU worker and one CPU
   utility worker. A one-GPU Pod normally shows **2 workers / 1 GPU**.
2. Open **Model Manager**.
3. Select a model that fits the GPU's VRAM.
4. Choose **Download** and wait for the queue job to complete.
5. Open the matching Studio and run a small generation.
6. Confirm the result appears in the project and Asset Library.

For the validated 32 GB RTX PRO 4500 configuration, RealVisXL Q4 is a practical
first image-generation check.

The first Model Manager visit on a large, previously populated volume can take
longer while SceneWorks scans the cache. If the RunPod proxy returns `524`,
wait a moment and refresh. Check **Queue** before assuming the Pod or GPU worker
failed.

### Notes for larger models

- A gated model needs both an accepted license and the optional `HF_TOKEN`
  secret. Restart the Pod after adding the environment variable.
- Ideogram 4 Q4 downloads successfully with the appropriate Hugging Face access,
  but its measured inference peak is above 32 GB. Use a larger-memory GPU for
  generation.
- Stable Video Diffusion's default 25-frame, 1024x576 job can exceed 32 GB.
  Reducing the job to 8 frames and using a decode chunk of 1 passed validation
  on the RTX PRO 4500.

## 8. Stop, restart, update, or remove the deployment

### Stop when idle

Stop the Pod when you are not generating. GPU compute billing stops, while
storage can continue to incur charges under RunPod's current pricing.

Do not rely on files outside `/workspace`. RunPod can replace or reset a Pod's
container disk, but the separate network volume remains available for another
Pod in the same datacenter.

### Restart with the same data

Start the stopped Pod, or deploy a replacement Pod and select the same network
volume. SceneWorks will reuse the projects, assets, configuration, and model
cache stored under `/workspace`.

### Update SceneWorks

1. Find the new official SceneWorks version.
2. Edit the template's container image to
   `ghcr.io/sceneworks/sceneworks-runpod:X.Y.Z`.
3. Stop the current Pod.
4. Deploy a fresh Pod from the updated template with the same network volume.
5. Repeat the verification checklist in step 7.
6. Terminate the old Pod after the replacement is healthy.

Using an explicit version makes rollback straightforward: point the template at
the previous version and deploy another Pod with the same volume.

### Remove everything

Terminate the Pod first. Delete the template, secrets, and network volume only
when you are sure you no longer need them. Deleting the network volume removes
the persisted SceneWorks projects, generated assets, and downloaded models.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| HTTP Services link is missing | The template must expose `8010` as HTTP. Edit the template and redeploy. |
| SceneWorks never opens | The image may still be pulling. Check the Pod logs and retry after the API reports healthy. |
| Startup refuses a public bind | `SCENEWORKS_ACCESS_TOKEN` is missing or blank. Confirm the environment variable uses the RunPod secret reference. |
| Login is rejected | Enter the SceneWorks access-token value, not the RunPod API key. If you rotated the secret, restart or replace the Pod. |
| Only the CPU worker appears | Wait for GPU initialization, then inspect logs and RunPod GPU telemetry. Confirm an NVIDIA GPU is actually attached. |
| A managed directory is not writable | Confirm the network volume is mounted at `/workspace` and permits writes by the container's root user. |
| A model remains gated | Accept its license, provide a read-only `HF_TOKEN` through a RunPod secret, and restart the Pod. |
| A generation reports insufficient memory | Choose a smaller quantization tier or workload, reduce video frames/resolution, or use a GPU with more VRAM. |
| Model Manager returns `524` | A large cache scan can outlast the proxy request. Wait, refresh, and check Queue/worker health. |
| Data disappeared after a Pod edit | Only data below `/workspace` is intended to persist. Reattach the original network volume if it still exists. |

Treat the proxy URL as public. Never put the SceneWorks token in a URL, template,
screenshot, log, or support message. Do not expose `8010/tcp` as a workaround;
that bypasses RunPod's HTTPS proxy.

## Environment reference

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `SCENEWORKS_ACCESS_TOKEN` | **Yes** | none | SceneWorks login/API credential. The public-bind entrypoint refuses to start when it is missing or blank. |
| `HF_TOKEN` | Only for gated Hugging Face repositories | unset | Read-only Hugging Face token used by Model Manager. |
| `SCENEWORKS_VOLUME` | No | `/workspace` | Base for durable SceneWorks data, config, and model-cache directories. |
| `SCENEWORKS_API_PORT` | No | `8010` | Internal UI/API port. If changed, expose the same HTTP port in RunPod. |
| `SCENEWORKS_CORS_ORIGINS` | Only for a different-origin frontend | built-in development origins | Exact allowed origins. Not needed for the default same-origin embedded UI. |

The combined image intentionally binds to `0.0.0.0` so RunPod's proxy can reach
it. It refuses to start without a nonblank access token.
`SCENEWORKS_CORS_ORIGINS` is not needed for the default embedded UI.

## Supported GPU architectures

The v1 image supports these NVIDIA architecture families:

| Architecture | Compute capability | Common RunPod GPUs | Status |
| --- | --- | --- | --- |
| Ampere | 8.0, 8.6 | A100, A30, A40, A10, RTX A6000/A5000/A4000, RTX 30-series | Supported |
| Ada | 8.9 | L4, L40/L40S, RTX 6000/5000/4500 Ada, RTX 40-series | Supported |
| Hopper | 9.0 | H100, H200 | Supported |
| Consumer/workstation Blackwell | 12.0 | RTX PRO 6000/5000/4500/4000/2000 Blackwell, RTX 50-series | Supported |
| Turing | 7.5 | T4, RTX 20-series, Quadro RTX | Not supported in v1 |
| Datacenter Blackwell | 10.0, 10.3 | B200/GB200, B300/GB300 | Not supported in v1 |

The published worker contains `sm_80`, `sm_90`, and `sm_120` quantized kernels.
The Docker build checks those targets before publication. NVIDIA's
[compute-capability table](https://developer.nvidia.com/cuda/gpus) is the
authoritative reference for a specific GPU.

The v1 build sets `CUDA_COMPUTE_CAP=80`, so its general CUDA path carries
`compute_80` PTX. Its quantized-kernel fat binary contains `sm_80`, `sm_90`, and
`sm_120` cubins plus `compute_120` PTX. It does not contain `sm_75`, `sm_100`,
or `sm_103` cubins. This is why Turing and Datacenter Blackwell are marked
**Unsupported in v1** even though newer drivers can JIT some of the general
kernels. Extending every native CUDA archive together is tracked in `sc-14423`;
changing only the top-level compute-capability setting would be incomplete.

### Published-image validation record

The public validation image was checked on two fresh RunPod Pods:

- At `2026-07-25T01:32:36Z`, an NVIDIA RTX PRO 4500 Blackwell reported driver
  `580.126.20` and `32,623 MiB` VRAM. A `4,096 bytes` CUDA round trip passed,
  the authenticated API reported `authRequired=true`, and the runtime
  registered `runpod-worker-0` plus `runpod-worker-cpu`.
- At `2026-07-25T01:40:55Z`, an NVIDIA A40 reported driver `570.195.03` and
  `46,068 MiB` VRAM. The same CUDA, authenticated API, and two-worker checks
  passed.

Those checks used
`ghcr.io/sceneworks/sceneworks-runpod:manual-sc14427-4fe122b7dba3` at OCI index
`sha256:376182ddbdf4c78d2a6f66a1e3ce66c573145b4c21b99300e42aeefba9f710ab`.
They validate the image and worker startup, not that every model fits each
GPU's memory.

## Optional: create the template with the API

Instead of filling out the web form, register the checked-in payload with
RunPod's template API:

```bash
export RUNPOD_API_KEY='<your RunPod API key>'
curl --fail-with-body --request POST \
  --url https://rest.runpod.io/v1/templates \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data-binary @config/runpod-template.json
unset RUNPOD_API_KEY
```

The response includes the new template ID. Rename the template in the JSON first
if an object with the same name already exists.

## For maintainers: publish an official image

The [Publish RunPod image workflow](../.github/workflows/publish-runpod.yml)
is part of the existing release flow:

1. Create and push an exact Git tag of the form `vX.Y.Z`.
2. The workflow builds the combined `linux/amd64` RunPod image.
3. It publishes both
   `ghcr.io/sceneworks/sceneworks-runpod:X.Y.Z` and
   `ghcr.io/sceneworks/sceneworks-runpod:latest`.
4. Copy the published digest from the GitHub Actions job summary.
5. Update and validate the private RunPod template with the immutable version
   tag before announcing support for the release.

A manual workflow dispatch can publish only a `manual-*` validation tag. It
cannot update `latest`. The full CUDA build is intentionally expensive and can
take well over an hour on a cold hosted runner.

## Optional: validate the RunPod proxy

For a page-load performance capture that separates API server work, proxy/body
transfer, and client readiness without recording secrets, follow the
[startup and page-load timing procedure](startup-performance-capture.md).

The repository probe checks authenticated job events, SSE heartbeats, and a
large multipart upload through the real RunPod HTTPS proxy. Keep a representative
image or video of at least 100 MiB available.

macOS/Linux:

```bash
export SCENEWORKS_BASE_URL='https://<pod-id>-8010.proxy.runpod.net'
read -rsp 'SceneWorks access token: ' SCENEWORKS_ACCESS_TOKEN
export SCENEWORKS_ACCESS_TOKEN
node scripts/validate-runpod-proxy.mjs \
  --upload-file /path/to/representative-large-video.mp4
unset SCENEWORKS_ACCESS_TOKEN
```

PowerShell:

```powershell
$env:SCENEWORKS_BASE_URL = 'https://<pod-id>-8010.proxy.runpod.net'
$env:SCENEWORKS_ACCESS_TOKEN = Read-Host 'SceneWorks access token'
node scripts/validate-runpod-proxy.mjs `
  --upload-file C:\path\to\representative-large-video.mp4
Remove-Item Env:SCENEWORKS_ACCESS_TOKEN
```

The probe redacts the Pod ID and never prints the token. Also keep the browser
open during one real generation and confirm that its Queue card updates before
the job completes.

The live proxy validation on 2026-07-24 also established a reference result:

- the same SSE connection stayed open beyond the documented `100-second`
  boundary and delivered a `15-second heartbeat`, totaling
  **seven heartbeat events** over 110 seconds;
- the browser Queue moved through `Preparing / 10%` and completed without a
  direct-origin URL; and
- a valid `105,000,054-byte` (100.14 MiB) upload completed in `33.851 seconds`
  and the probe asset was removed afterward.

This is a measured envelope, not a promise that RunPod has no higher
undocumented limit.
