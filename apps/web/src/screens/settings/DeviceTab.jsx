import React from "react";
import { isDesktop } from "../../runtime.js";

export function DeviceTab({
  GPU_LIMIT_MIN_PERCENT,
  commitGpuMemoryLimit,
  cudaGpuMemory,
  device,
  gpu,
  gpuLimitPercent,
  gpuTargetLabel,
  gpuTelemetry,
  macGpuMemory,
  rerunSetupWizard,
  restartWorker,
  setGpuLimitPercent,
  telemetryGibLabel,
  unifiedGb,
}) {
  return (
<div className="settings-tab-body">
            <div className="settings-group-title">This machine</div>
            <div className="settings-kv-list">
              <div className="settings-kv">
                <span>Engine</span>
                <span>{device.engine}</span>
              </div>
              <div className="settings-kv">
                <span>GPU</span>
                <span>{device.gpu}</span>
              </div>
              {macGpuMemory ? (
                <div className="settings-kv">
                  <span>Unified memory</span>
                  <span>
                    {unifiedGb} GiB
                    {typeof gpu.wiredLimitMb === "number"
                      ? ` · system cap ${Math.round(gpu.wiredLimitMb / 1024)} GiB`
                      : ""}
                  </span>
                </div>
              ) : null}
            </div>

            {macGpuMemory || cudaGpuMemory ? (
              <>
                <div className="work-panel-divider" />
                <div className="settings-group-title">GPU memory</div>
                <div>
                  <div className="settings-row settings-row--baseline">
                    <div className="settings-row-title">Memory target</div>
                    <div className="settings-readout">{gpuTargetLabel}</div>
                  </div>
                  <input
                    className="settings-range"
                    type="range"
                    min={GPU_LIMIT_MIN_PERCENT}
                    max={100}
                    step={5}
                    value={gpuLimitPercent}
                    aria-label={macGpuMemory ? "GPU memory target for SceneWorks (percent of unified memory)" : "MOSS training GPU memory target (percent of selected GPU VRAM)"}
                    onChange={(event) => setGpuLimitPercent(Number(event.target.value))}
                    onMouseUp={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                    onKeyUp={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                    onTouchEnd={(event) => commitGpuMemoryLimit(Number(event.target.value))}
                  />
                  <div className="settings-scale">
                    <span>{GPU_LIMIT_MIN_PERCENT}%</span>
                    <span>Use all available</span>
                  </div>
                  <p className="settings-note settings-note--spaced">
                    {macGpuMemory ? (
                      <>An MLX soft target, not an OS memory quota. Native MOSS training also uses
                        it for memory admission and refuses jobs that cannot fit. Changes apply
                        to new jobs without a restart; active jobs keep their admitted cap.{" "}
                        <code>sudo sysctl iogpu.wired_limit_mb=&lt;bytes&gt;</code> raises the system cap
                        on 96/128 GB Macs.</>
                    ) : (
                      <>For native MOSS training only: limits its admission budget using the actual
                        selected GPU&apos;s total VRAM, not host RAM or another GPU. Other CUDA jobs
                        retain their existing memory policy. Changes apply to new jobs without a restart.</>
                    )}
                  </p>
                  {macGpuMemory && gpuTelemetry ? (
                    <div className="settings-inset settings-inset--spaced">
                      <div className="settings-inset-title">Live MLX memory</div>
                      <div className="settings-kv">
                        <span>Active</span>
                        <span className="settings-mono">
                          {telemetryGibLabel(gpuTelemetry.activeBytes)}
                        </span>
                      </div>
                      <div className="settings-kv">
                        <span>Peak</span>
                        <span className="settings-mono">
                          {telemetryGibLabel(gpuTelemetry.peakBytes)}
                        </span>
                      </div>
                      {gpuTelemetry.limitBytes !== null && gpuTelemetry.limitBytes !== undefined ? (
                        <div className="settings-kv">
                          <span>Limit</span>
                          <span className="settings-mono">
                            {telemetryGibLabel(gpuTelemetry.limitBytes, 0)}
                          </span>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </>
            ) : null}

            {gpu?.platform === "windows" || gpu?.platform === "linux" ? (
              <p className="settings-note">
                Requires current NVIDIA drivers with CUDA support.{" "}
                {!cudaGpuMemory ? "GPU memory target unavailable: NVIDIA VRAM metadata could not be read. Check the driver and retry." : "The MOSS training target is not a per-process quota for other CUDA jobs."}
              </p>
            ) : null}

            <div className="work-panel-divider" />
            <div className="settings-group-title">Maintenance</div>
            {/* Available in both modes: desktop restarts via Tauri, a remote admin via REST
                (epic 4484 story 12). */}
            <div className="settings-row">
              <div>
                <div className="settings-row-title">Inference worker</div>
                <div className="settings-row-sub">
                  Restart to pick up credential changes or clear a stuck job.
                </div>
              </div>
              <button className="settings-btn" type="button" onClick={restartWorker}>
                Restart
              </button>
            </div>
            {isDesktop ? (
              <div className="settings-row">
                <div>
                  <div className="settings-row-title">Setup wizard</div>
                  <div className="settings-row-sub">
                    Re-open the guided setup to download more models or create a project.
                  </div>
                </div>
                <button className="settings-btn" type="button" onClick={rerunSetupWizard}>
                  Re-run
                </button>
              </div>
            ) : null}
          </div>
  );
}
