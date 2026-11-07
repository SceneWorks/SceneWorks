# Host and Runner Guidance

Verified **2026-10-24**. Confidence is **high** for GitHub registration, labels, repository access, and the primary Mac’s hardware. Windows hardware and paths below include host-reported facts; refresh them when relevant. The secondary Mac’s exact hardware was not verified.

## 1. Runner inventory

All six runners are **organization-scoped**, in the **`self-hosted-gpu`** group, and accessible to:

- `SceneWorks/SceneWorks`
- `SceneWorks/inference`
- `SceneWorks/ChatWorks`

| Runner | Platform labels | Additional labels |
|---|---|---|
| `cuda-windows` | `self-hosted`, `Windows`, `X64`, `cuda` | `real-weights`, `cuda-windows` |
| `cuda-windows-2` | `self-hosted`, `Windows`, `X64`, `cuda` | `real-weights`, `cuda-windows-2` |
| `cuda-windows-3` | `self-hosted`, `Windows`, `X64`, `cuda` | — |
| `cuda-windows-4` | `self-hosted`, `Windows`, `X64`, `cuda` | — |
| `nax-macos` | `self-hosted`, `macOS`, `ARM64`, `nax` | `signing`, `real-weights`, `weights`, `rw-starvector` |
| `nax-macos-2` | `self-hosted`, `macOS`, `ARM64`, `nax` | `weights`, `rw-mage`, `rw-sa3`, `rw-krea`, `rw-audio`, `rw-llm`, `rw-chroma` |

Runners 3 and 4 moved from repository scope to organization scope on 2026-10-24. Inference and ChatWorks can now use all four CUDA listeners.

**Routing details:**

- Ordinary `cuda` jobs can match all four Windows runners.
- Windows jobs requiring `real-weights` match only the first two.
- Both Macs carry `nax`; it does **not** uniquely select the primary Mac.
- Only `nax-macos` carries `signing`. Preserve signing-workflow routing.
- The secondary Mac supports workflows using `weights` and its role labels despite lacking the literal `real-weights` label.
- Runner names are not automatically labels. Match the labels actually advertised.

Labels establish routing eligibility; they do not prove that a particular model snapshot, toolchain, or application is ready.

## 2. Physical hosts

### Windows: `MICHAEL-TRX50`

All four CUDA listeners run on this one physical host:

```text
D:\actions-runner\     → cuda-windows
D:\actions-runner-2\   → cuda-windows-2
D:\actions-runner-3\   → cuda-windows-3
D:\actions-runner-4\   → cuda-windows-4
```

| Fact | Guidance |
|---|---|
| OS | Windows 11 Pro; reported build `10.0.26200`. Refresh before OS-specific diagnosis. |
| GPUs | Reported: 2 × NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition. |
| Reported memory per GPU | **97,887 MiB = approximately 95.59 GiB or 102.64 decimal GB.** This is total device memory, not an available workload budget. |
| Checkouts | Reported roots: `D:\repos\SceneWorks` and `D:\repos\inference`. Discover the current checkout and worktree rather than assuming these paths. |
| Model cache | Reported shared root: `E:\huggingface`, with the Hub cache commonly under `E:\huggingface\hub`. Workflows can use other explicit roots. |
| Shells | PowerShell, `cmd.exe`, and Git Bash. Resolve the actual executable and use syntax appropriate to the selected shell. |

Normal automatic Candle PR/push CI currently selects **physical GPU1**. Manual profiles can use different settings. Inspect the exact workflow, inputs, and task restrictions; do not assume GPU0 is available.

With `CUDA_VISIBLE_DEVICES=1`, a process can see physical GPU1 as logical CUDA device 0. Preserve the intended physical-device mapping and any required `SCENEWORKS_GPU_ID` setting.

Four runner listeners provide four job slots. They do **not** provide four GPUs or four independent memory budgets.

### Primary Mac: `nax-macos`

Verified locally:

| Fact | Value |
|---|---|
| Hardware | MacBook Pro, Apple M5 Max |
| CPU | 18 physical and logical cores |
| Unified memory | **128 GiB**, shared by CPU, GPU, applications, and the OS |
| OS | macOS 27.0.1 at verification |
| Runner installation | `/Users/michael/actions-runner-sceneworks` |
| Existing checkout roots | `/Users/michael/Repos/SceneWorks`, `/Users/michael/Repos/inference` |
| Roles | Development host, MLX/Metal CI, and signing-capable runner |

Local builds, applications, agents, and Actions jobs share this host’s resources. An idle GitHub runner does not establish that its GPU, memory, or CPU is idle.

### Secondary Mac: `nax-macos-2`

This is a separate Apple-silicon host with its own Metal GPU and unified memory. Its verified labels appear in the inventory above.

Do not copy the primary Mac’s hardware, username, paths, cache layout, or memory budget onto this host. Collect those facts on the secondary Mac when needed. It currently has no `signing` label.

## 3. Discover the live pool

Run these commands from either macOS or Windows with authenticated `gh`:

```sh
gh api --paginate orgs/SceneWorks/actions/runners
gh api --paginate repos/SceneWorks/REPOSITORY/actions/runners
gh api --paginate orgs/SceneWorks/actions/runner-groups
```

Replace `REPOSITORY` with `SceneWorks`, `inference`, or `ChatWorks`.

For each relevant group, inspect membership and access:

```sh
gh api --paginate orgs/SceneWorks/actions/runner-groups/GROUP_ID/runners
gh api --paginate orgs/SceneWorks/actions/runner-groups/GROUP_ID/repositories
```

The repositories endpoint applies to groups with selected-repository visibility. Also inspect any workflow restrictions.

Combine accessible organization runners with repository-scoped runners, retaining registration scope alongside each runner ID. Then match every required workflow label and group restriction.

Report separately:

- Registered runners.
- Online runners compatible with the requested job.
- Compatible runners currently idle.

A zero repository-scoped count does not mean the repository has no runners. Failed inventory queries mean discovery is incomplete, not that capacity is zero.

## 4. Execution lanes and shared resources

| Lane | Resource relationship |
|---|---|
| `gpu-local` | The executing host’s GPU: Metal on an Apple-silicon Mac, CUDA on Windows. It shares hardware with CI listeners on that same host. |
| `nax-ci` | The two macOS listeners, each on a separate physical Mac. Route using the workflow’s complete label set. |
| `cuda-ci` | Four Windows listeners on `MICHAEL-TRX50`, sharing its CPU, disks, caches, and task-permitted GPUs. |
| Hosted CI | GitHub-hosted resources selected by the workflow; separate from these self-hosted machines. |
| `cpu-only` | Work that performs no GPU execution. It still consumes host CPU, RAM, and disk bandwidth. |

A local Windows render and CUDA CI jobs can contend on the **same permitted card**. A local render on the primary Mac shares resources with `nax-macos`, while the secondary Mac is a separate host.

Do not add VRAM across cards into a single-model budget unless the actual runtime supports and validates the required distributed execution.

Submit dependency-ready GPU work through compatible Actions workflows and let it queue. Preserve existing device restrictions and workflow concurrency. Busy runners are a queueing condition; remembered runner counts should not become submission caps or new locking requirements.

## 5. Agents, caches, and host checks

The SceneWorks epic policy currently allows up to **10 concurrent subagents across all roles, excluding the coordinator**, subject to a lower live platform or task limit. This is a coordination ceiling, not a token cap. More CPU cores do not raise it. Workflow jobs and PRs do not consume subagent slots.

Preserve existing cache configuration. Some workflows explicitly select their cache; some tests require isolated caches. Avoid blanket rules either forbidding or globally exporting `HF_HOME`. Use the specific test or workflow’s supported overrides.

Refresh changing host facts when relevant:

**Windows PowerShell**

```powershell
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption, Version, BuildNumber
Get-CimInstance Win32_Processor |
  Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
Get-Volume |
  Select-Object DriveLetter, Size, SizeRemaining
Get-Command bash -All
nvidia-smi
```

**macOS**

```sh
sw_vers
system_profiler SPHardwareDataType SPDisplaysDataType
sysctl -n hw.physicalcpu hw.logicalcpu hw.memsize
vm_stat
df -h
diskutil apfs list
```

Keep free disk, reclaimable disk, physical RAM, unified memory, and VRAM distinct. Do not preserve old free-space figures as operating assumptions or treat total installed memory as available inference capacity.
