# Issue #3017 / #1966: AppImage CUDA reproduction

Tested 2026-10-17. Reports: [#3017](https://github.com/SceneWorks/SceneWorks/issues/3017),
[original #1966](https://github.com/SceneWorks/SceneWorks/issues/1966).

## Finding

Both published AppImages examined bundle the build machine's NVIDIA **driver**
library, `usr/lib/libcuda.so.1`, version **575.57.08**. The packaged API directly
links `libcuda.so.1` and has `$ORIGIN/../lib` in its RUNPATH. AppRun also prepends
the AppDir's library directories to `LD_LIBRARY_PATH`.

Consequently the AppImage selects its own driver library ahead of the installed
host driver. This is a packaging defect: CUDA runtime libraries can travel with
the app, but the driver library must match the host's driver environment.
AppImage's [packaging concepts](https://docs.appimage.org/introduction/concepts.html)
explicitly warn that bundling hardware-dependent graphics driver libraries can
break an AppImage on the target system. NVIDIA's
[compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html#check-for-compatibility-support)
defines 803 as a mismatch between display-driver and CUDA-driver versions.

The original report's healthy `nvidia-smi` and failing SceneWorks CUDA path are
consistent with this mechanism: NVML can query the installed driver while the
AppImage loads a different `libcuda`. This evidence does **not** establish that
this is the reporter's exact failure, or that an immutable OS caused it.

## Published artifacts

Downloaded from GitHub Releases; SHA-256 values matched the release asset digests.

| Artifact | SHA-256 | Bundled driver |
| --- | --- | --- |
| `SceneWorks_0.8.1_amd64.AppImage` | `fec3bb373b3fc230d6d2a33db0b1c6de58b40262358fe6d9b6c51b282747606b` | 575.57.08 |
| `SceneWorks_0.9.0_amd64.AppImage` | `5f12dcd8baaa2395985c26f707111120ed3a9fecfd2b52431cda7b34ff02086f` | 575.57.08 |

Raw evidence: [0.8.1 inventory](v0.8.1-appimage.json),
[0.9.0 comparison and fault injection](v0.9.0-appimage.json),
[0.9.0 with the bundled driver omitted](without-bundled-driver.json).

## Real GPU comparison

Environment: Ubuntu 24.04 under WSL2, host NVIDIA driver 596.36, RTX PRO 6000
Blackwell. These are **not** the reporter's Bazzite/RTX 3090 Ti/610.43.03 environment.
No models or weights were loaded. The packaged sidecar's one-shot GPU check
acquires a device and runs a small CUDA kernel.

| Test | Result |
| --- | --- |
| Host `nvidia-smi` | Healthy; driver 596.36, compute capability 12.0 |
| 0.9.0 packaged sidecar, AppRun library directory order | Exit 2: `CUDA_ERROR_NO_DEVICE` |
| Same binary and libraries, host `/usr/lib/wsl/lib` first | Exit 0; GPU kernel check passes |
| Same binary, `libcuda.so.1` omitted from a separate diagnostic AppDir | Exit 0; system loader selects the host driver without an explicit override |
| Same binary, host driver first, injected `cuInit` result 803 | Exit 2; `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH` and reboot guidance |

The diagnostic AppDir uses the identical API executable (same SHA-256) and links
the other shared libraries from the original extraction. Published artifacts and
the original extraction were not modified to remove their driver.

The real failure on WSL is **100 / NO_DEVICE**, not **803 / DRIVER_MISMATCH**.
The 803 case is explicitly fault injection using an `LD_PRELOAD` library whose
`cuInit` returns 803. It verifies error handling, not a naturally occurring host
driver mismatch. Reboot guidance cannot remove a driver library inside an AppImage.

The 0.8.1 inventory is static only: its `SCENEWORKS_GPU_CHECK` implementation
checked Metal and did not exercise CUDA, so its successful exit would not be a
valid CUDA test.

## Repeat the artifact test

Use Linux with Python 3.9+ and an extracted AppImage. `--probe` requires a release
that implements the CUDA startup probe. `--inject-803` additionally requires `cc`.
The script isolates XDG data/config/cache/state in temporary directories and does
not launch the desktop, a server, or a generation job.

```sh
chmod +x SceneWorks_0.9.0_amd64.AppImage
./SceneWorks_0.9.0_amd64.AppImage --appimage-extract
python3 scripts/diagnose-linux-appimage.py squashfs-root \
  --artifact SceneWorks_0.9.0_amd64.AppImage --output appimage-inventory.json

# WSL comparison; on native Linux use the directory containing the host driver.
python3 scripts/diagnose-linux-appimage.py squashfs-root \
  --probe --host-driver-dir /usr/lib/wsl/lib --inject-803 \
  --output appimage-gpu-comparison.json
```

Exit 1 is expected for these released AppImages because the packaging check rejects
bundled `libcuda.so*` / `libnvidia-ml.so*`. It permits `libcudart.so*`. GPU probe
stdout, stderr, exit codes, and `ldd` results are recorded separately in the JSON.
Five temporary-fixture CLI checks passed: runtime allowed, driver rejected,
driver symlink rejected, NVML rejected, and invalid fault-injection options rejected.

## Exclusion experiment and regression check

Repacked the full 0.9.0 AppImage using appimagetool's `--exclude-file`, with
`usr/lib/libcuda.so.1` as its only exclusion. This experiment preserves the original
AppImage runtime and does not rebuild SceneWorks. The resulting AppImage is an
unsigned local test artifact, not a published release or signed updater payload.

Candidate: `SceneWorks_0.9.0_without-host-nvidia.AppImage` (932,862,456 bytes),
SHA-256 `cc249c25c7c32f76d2f331d50def8679850497e0263a75cebc138dfd37ee0420`.
The Windows-accessible copy is at
`C:\Users\Michael\.codex\tmp\sc3017\artifacts\SceneWorks_0.9.0_without-host-nvidia.AppImage`.

| Check | Published 0.9.0 | Repacked candidate |
| --- | --- | --- |
| Finished-AppImage driver inventory | Fails: `usr/lib/libcuda.so.1` | Passes: no host NVIDIA libraries |
| Actual sidecar GPU kernel check under AppRun library order | Exit 2, NO_DEVICE | Exit 0, host driver loaded |
| Injected CUDA 803 detection | Exit 2 with mismatch guidance | Exit 2 with mismatch guidance |

The candidate GPU test starts without an inherited `LD_LIBRARY_PATH` or
`LD_PRELOAD`; host-library precedence is not supplied for its default probe.
An inventory of every file and symlink after extracting the candidate found
**one removed file, zero added files, zero changed files, and 720 unchanged
entries**, compared with the source AppDir. The API executable, CUDA toolkit,
WebKitGTK, GStreamer, and other retained file contents match their source hashes.

Evidence: [repack command, checksums, regression results, and content comparison](exclusion-experiment.json),
[candidate GPU and 803 probe output](excluded-appimage-gpu.json).

The new check is `apps/desktop/scripts/check-linux-driver-libraries.mjs`:

```sh
# Linux: extract and check the finished AppImage without requiring FUSE or a GPU.
npm --prefix apps/desktop run check:linux-drivers -- --appimage /path/to/SceneWorks.AppImage

# All platforms: inspect an existing AppDir.
npm --prefix apps/desktop run check:linux-drivers -- --appdir /path/to/squashfs-root
```

It rejects CUDA/NVML and other NVIDIA driver sonames, including versioned files
and dangling driver symlinks, while permitting the CUDA toolkit and playback
libraries. Missing/invalid inputs and extraction failures produce a separate
error exit instead of a passing empty scan. Seven regression tests are registered
in the desktop suite. The full desktop suite passed on Windows (30 passed, two
Linux-specific cases skipped); all seven new tests passed on Linux. Test-list
registration checks also passed.

Production AppImages now use `apps/desktop/.appimageignore` to omit host driver
libraries during SquashFS creation, before updater signing. Its non-anchored
patterns use the `... pattern` syntax (a space, not a slash). A real mksquashfs
regression test exercises every excluded driver family and retained toolkit
libraries. Both Linux workflows run the finished-AppImage check before upload.

The actual linuxdeploy AppImage output plugin was also run from `apps/desktop`
against a fresh release extraction using only the checked-in ignore file.
[Production exclusion validation](production-ignore-validation.json) and
[GPU check](production-ignore-gpu.json) record the result. Native Bazzite and
real model generation remain outside this test.

## Remaining validation and correction

The original `bazzite-dx-nvidia-gnome:testing` image tag returned HTTP 404 when
queried on this date. Native Bazzite desktop behavior and a real driver/kernel
mismatch on the reporter's hardware remain untested. #3017 contains no new error
text or exact version; it cannot be closed from this evidence alone.

### smolvm attempt

smolvm 1.30.1 was downloaded and its archive checksums verified. The native Windows
Alpine boot timed out waiting for its agent. The Linux runtime under WSL2/KVM
successfully booted Alpine and Ubuntu. The Ubuntu run reached the packaged
SceneWorks GPU check and returned NO_DEVICE, as expected without GPU forwarding.

The Bazzite `stable` image was resolved and pinned to
`sha256:bbd8ba49ce2598de2811ba8e661c78ace07cfe08d6fc1e7e476ddaa82761897c`
(128 layers, 6,650,266,136 compressed bytes). smolvm completed image preparation,
but `/bin/sh` failed with an executable-not-found error. Retrying `/usr/bin/bash`
failed with `exec: No such file or directory`. Whether the missing object was the
shell or its interpreter was not established. **Neither attempt reached SceneWorks
inside Bazzite**, and neither is a reproduction of the user's OS failure.

[smolvm attempt evidence](smolvm-attempts.json). No VMs remained running after
these ephemeral tests. A full Bazzite installation was not performed.

Rebuild and compare on native Bazzite with NVIDIA hardware to confirm
the reporter's path. The unmerged PR #3058 changes dependency-error wording;
it does not remove this bundled driver.
