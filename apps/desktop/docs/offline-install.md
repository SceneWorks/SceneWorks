# Offline / air-gapped install (Windows)

SceneWorks Desktop on Windows needs ~2.7 GB of NVIDIA CUDA runtime, cuDNN, and the
ONNX Runtime GPU provider at runtime. These are **not** in the installer (the set
exceeds the NSIS installer format's datablock limit), so a normal first run
**downloads** them once into `%APPDATA%\SceneWorks\gpu-runtime` — see
[First run](../README.md#first-run).

On a machine with no internet access that download fails and startup stops. This
guide covers how to **pre-stage** the runtime so first run completes fully offline.

> This only makes *startup* work offline. Generation also needs model weights, which
> download from Hugging Face on first use — see [Model weights](#model-weights-are-separate)
> at the end.

---

## Which method to use

| Your situation | Use |
| --- | --- |
| You have a connected Windows machine you can run SceneWorks on first | [Method 1 — copy a provisioned folder](#method-1-copy-a-provisioned-folder-easiest) (easiest) |
| You can reach the internet to download files, but not from the target machine | [Method 2 — stage a redist bundle](#method-2-stage-a-redist-bundle) |
| You are scripting an enterprise / MSI rollout | [Method 2](#method-2-stage-a-redist-bundle) with `SCENEWORKS_GPU_RUNTIME_DIR` |

All methods land the same files in the same place. Under the hood the app skips the
download whenever the runtime is already present (`apps/desktop/src/cuda_provision.rs`).

---

## Method 1 — copy a provisioned folder (easiest)

If you have another Windows machine (same or newer NVIDIA driver) that can reach the
internet:

1. Install and launch SceneWorks on the **connected** machine and let first run
   finish the GPU-runtime download.
2. Copy its entire runtime folder to the **offline** machine, to the same path:

   ```
   %APPDATA%\SceneWorks\gpu-runtime
   ```

   That folder contains `cuda\`, `onnxruntime\`, and a `.redist-marker` file. Copy all
   of it (the marker tells the app it is already provisioned).
3. Install SceneWorks on the offline machine and launch it. First run detects the
   existing runtime and skips the download.

That is the whole procedure — no environment variables needed.

---

## Method 2 — stage a redist bundle

Use this when the target machine can never reach the internet and you cannot run
SceneWorks online first. You will assemble the DLLs from the pinned PyPI wheels on a
connected machine, then hand the bundle to the target.

### 2a. Download the wheels

Download each wheel below on a connected machine. Every entry is version-pinned; the
SHA-256 lets you verify each download. (This table mirrors the `COMPONENTS` manifest in
[`apps/desktop/src/cuda_provision.rs`](../src/cuda_provision.rs) — if it ever drifts,
the source is authoritative.)

| Component | Wheel | Approx | SHA-256 |
| --- | --- | --- | --- |
| CUDA runtime | [`nvidia_cuda_runtime_cu12-12.9.79`](https://files.pythonhosted.org/packages/59/df/e7c3a360be4f7b93cee39271b792669baeb3846c58a4df6dfcf187a7ffab/nvidia_cuda_runtime_cu12-12.9.79-py3-none-win_amd64.whl) | 3 MB | `8e018af8fa02363876860388bd10ccb89eb9ab8fb0aa749aaf58430a9f7c4891` |
| cuBLAS | [`nvidia_cublas_cu12-12.9.1.4`](https://files.pythonhosted.org/packages/45/a1/a17fade6567c57452cfc8f967a40d1035bb9301db52f27808167fbb2be2f/nvidia_cublas_cu12-12.9.1.4-py3-none-win_amd64.whl) | 530 MB | `1e5fee10662e6e52bd71dec533fbbd4971bb70a5f24f3bc3793e5c2e9dc640bf` |
| cuRAND | [`nvidia_curand_cu12-10.3.10.19`](https://files.pythonhosted.org/packages/e5/98/1bd66fd09cbe1a5920cb36ba87029d511db7cca93979e635fd431ad3b6c0/nvidia_curand_cu12-10.3.10.19-py3-none-win_amd64.whl) | 66 MB | `e8129e6ac40dc123bd948e33d3e11b4aa617d87a583fa2f21b3210e90c743cde` |
| NVRTC | [`nvidia_cuda_nvrtc_cu12-12.9.86`](https://files.pythonhosted.org/packages/52/de/823919be3b9d0ccbf1f784035423c5f18f4267fb0123558d58b813c6ec86/nvidia_cuda_nvrtc_cu12-12.9.86-py3-none-win_amd64.whl) | 73 MB | `72972ebdcf504d69462d3bcd67e7b81edd25d0fb85a2c46d3ea3517666636349` |
| cuDNN | [`nvidia_cudnn_cu12-9.23.0.39`](https://files.pythonhosted.org/packages/b7/ec/d95cc4204dd45f40f2d1512f8ff0d4c3fb1810a893fecc79fcea05dfec0e/nvidia_cudnn_cu12-9.23.0.39-py3-none-win_amd64.whl) | 660 MB | `357e5d59a1b79d27eef754aa79b3d9e7adf11baf86dc928dc114df0033c2c912` |
| cuFFT | [`nvidia_cufft_cu12-11.4.1.4`](https://files.pythonhosted.org/packages/20/ee/29955203338515b940bd4f60ffdbc073428f25ef9bfbce44c9a066aedc5c/nvidia_cufft_cu12-11.4.1.4-py3-none-win_amd64.whl) | 190 MB | `8e5bfaac795e93f80611f807d42844e8e27e340e0cde270dcb6c65386d795b80` |
| nvJitLink | [`nvidia_nvjitlink_cu12-12.9.86`](https://files.pythonhosted.org/packages/dd/7e/2eecb277d8a98184d881fb98a738363fd4f14577a4d2d7f8264266e82623/nvidia_nvjitlink_cu12-12.9.86-py3-none-win_amd64.whl) | 34 MB | `cc6fcec260ca843c10e34c936921a1c426b351753587fdd638e8cff7b16bb9db` |
| ONNX Runtime (GPU) | [`onnxruntime_gpu-1.26.0-cp312`](https://files.pythonhosted.org/packages/a4/e4/9b378a5466ea0bed65e5beb8e09254973c580a6522810a38afbcc45e5105/onnxruntime_gpu-1.26.0-cp312-cp312-win_amd64.whl) | 216 MB | `5f49c44689894650990e4c8a857d2edafc276fbd79bba57ceb224bd18d25d491` |

Verify a download's hash in PowerShell:

```powershell
Get-FileHash .\nvidia_cudnn_cu12-9.23.0.39-py3-none-win_amd64.whl -Algorithm SHA256
```

### 2b. Extract the DLLs into a bundle

A wheel is just a ZIP. Build a bundle folder with two subdirectories, `cuda` and
`onnxruntime`:

```
gpu-runtime-bundle\
  cuda\           <- every *.dll from the 7 nvidia_*-cu12 wheels
  onnxruntime\    <- three DLLs from the onnxruntime_gpu wheel
```

- From each of the **seven `nvidia_*-cu12`** wheels, copy **every `*.dll`** (they live
  under `nvidia\<component>\bin\` inside the wheel) into `cuda\`.
- From the **`onnxruntime_gpu`** wheel (DLLs under `onnxruntime\capi\`), copy exactly
  these three into `onnxruntime\`:
  - `onnxruntime.dll`
  - `onnxruntime_providers_cuda.dll`
  - `onnxruntime_providers_shared.dll`

A PowerShell sketch (run in a folder holding all eight `.whl` files):

```powershell
$bundle = "$PWD\gpu-runtime-bundle"
New-Item -ItemType Directory -Force "$bundle\cuda", "$bundle\onnxruntime" | Out-Null

foreach ($whl in Get-ChildItem *.whl) {
  $zip = "$($whl.FullName).zip"
  Copy-Item $whl.FullName $zip
  $out = Join-Path $env:TEMP ("whl-" + $whl.BaseName)
  Expand-Archive -Force $zip $out
  if ($whl.Name -like 'onnxruntime_gpu*') {
    'onnxruntime.dll','onnxruntime_providers_cuda.dll','onnxruntime_providers_shared.dll' |
      ForEach-Object { Copy-Item (Get-ChildItem -Recurse -Filter $_ $out).FullName "$bundle\onnxruntime" }
  } else {
    Get-ChildItem -Recurse -Filter *.dll $out | Copy-Item -Destination "$bundle\cuda"
  }
  Remove-Item $zip; Remove-Item -Recurse -Force $out
}
```

Sanity-check the bundle carries the key libraries: `cuda\` should contain
`cudart64_12.dll`, `cublas64_12.dll`, `curand64_10.dll`, an `nvrtc64_*.dll`,
`cudnn64_9.dll`, `cufft64_11.dll`, and an `nvJitLink_*.dll`; `onnxruntime\` should hold
the three DLLs above. (These are exactly the sentinels the app checks — an incomplete
bundle is rejected with a clear error, not silently accepted.)

### 2c. Point SceneWorks at the bundle

Copy `gpu-runtime-bundle` to the offline machine, then set the environment variable
**before launching SceneWorks** so it installs from the bundle instead of downloading:

```powershell
setx SCENEWORKS_GPU_RUNTIME_DIR "C:\path\to\gpu-runtime-bundle"
```

(`setx` persists it for future launches; open a new session for it to take effect. For
a machine-wide rollout, set it as a system environment variable.)

On first launch the app copies the bundle's DLLs into `%APPDATA%\SceneWorks\gpu-runtime`,
verifies the set is complete, marks the runtime provisioned, and starts — with no network
access. If the bundle is missing a subdirectory or a required DLL, setup stops with an
actionable message rather than starting a broken worker.

Once installed, `SCENEWORKS_GPU_RUNTIME_DIR` is no longer consulted (the provisioned
copy is marked complete), so you can leave it set or remove it.

---

## Manual fallback (marker file)

If you prefer not to set an environment variable, you can place the DLLs directly and
let the app adopt them:

1. Copy the bundle's `cuda\` and `onnxruntime\` folders into
   `%APPDATA%\SceneWorks\gpu-runtime\` so you have
   `%APPDATA%\SceneWorks\gpu-runtime\cuda\...` and `...\onnxruntime\...`.
2. Launch SceneWorks. It detects the complete DLL set and skips the download.

You do **not** need to hand-write the `.redist-marker` file — the app writes it once it
confirms the set is complete. (Older instructions that had you create the marker by hand
still work, but the DLL detection above is the supported path.)

---

## Model weights are separate

Pre-staging the GPU runtime gets the app to **start** offline. The first time you
generate with a given model, its weights download from Hugging Face into the model
cache (`%USERPROFILE%\.cache\huggingface`, overridable with `HF_HOME`). For a fully
offline workstation you must also pre-seed that cache:

- Point `HF_HOME` at a folder you have pre-populated (copy `~/.cache/huggingface`, or a
  subset, from a connected machine that already downloaded the models you need), or
- Download the models through **Model Manager** on a connected machine first, then copy
  the cache across.

Only the runtime DLLs are covered by this guide; the model cache is standard Hugging
Face layout and can be seeded with any of the usual tooling.

---

## See also

- [First run](../README.md#first-run) — the normal (online) provisioning flow.
- [`apps/desktop/src/cuda_provision.rs`](../src/cuda_provision.rs) — the pinned manifest
  and the offline install logic (authoritative source for the versions/hashes above).
- [sc-5560 CUDA bundling](../../../docs/sc-5560/cuda-bundling.md) — provenance and
  licensing of the redistributable CUDA runtime.
