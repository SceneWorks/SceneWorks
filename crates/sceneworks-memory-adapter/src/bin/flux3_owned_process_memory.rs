//! FLUX3 resource-only Windows process-accounting transport. Not an allocator or
//! physical-residency probe; no changes to the shared device-delta testkit.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn bounded_remaining(
    deadline_ms: u64,
    timeout_seconds: u64,
    now_ms: u64,
) -> Result<Duration, String> {
    if !(60..=10800).contains(&timeout_seconds) {
        return Err("owned collector fixed timeout is invalid".to_owned());
    }
    let remaining = deadline_ms
        .checked_sub(now_ms)
        .filter(|remaining| *remaining > 0 && *remaining <= timeout_seconds * 1000)
        .ok_or("owned collector fixed deadline expired or extended")?;
    Ok(Duration::from_millis(remaining))
}

fn fixed_admission_deadline() -> Result<(u64, Instant), String> {
    let invocation: Value = serde_json::from_str(
        &std::env::var("SCENEWORKS_FLUX3_INVOCATION")
            .map_err(|_| "owned collector invocation binding missing")?,
    )
    .map_err(|_| "owned collector invocation binding invalid")?;
    let file = invocation["admissionFile"]
        .as_str()
        .ok_or("owned collector admission path missing")?;
    let admission: Value = serde_json::from_slice(
        &std::fs::read(file).map_err(|_| "owned collector admission unreadable")?,
    )
    .map_err(|_| "owned collector admission invalid")?;
    let deadline_ms = admission["runDeadline"]
        .as_u64()
        .ok_or("owned collector fixed deadline missing")?;
    let timeout_seconds = admission["timeoutSeconds"]
        .as_u64()
        .ok_or("owned collector fixed timeout missing")?;
    // Take the monotonic origin FIRST: processing cannot extend the admitted wall deadline.
    let origin = Instant::now();
    let now_ms = u64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "owned collector wall clock invalid")?
            .as_millis(),
    )
    .map_err(|_| "owned collector wall clock overflow")?;
    let remaining = bounded_remaining(deadline_ms, timeout_seconds, now_ms)?;
    let deadline = origin
        .checked_add(remaining)
        .ok_or("owned collector deadline overflow")?;
    Ok((deadline_ms, deadline))
}

fn safe_protocol_refusal(response: &Value) -> String {
    let code = if response["failureCode"] == "FLUX3_OWNED_MEMORY_CAPTURE_REFUSED" {
        "FLUX3_OWNED_MEMORY_CAPTURE_REFUSED"
    } else {
        "INVALID_OWNED_COLLECTOR_FAILURE"
    };
    let phase = response["phase"]
        .as_str()
        .filter(|value| {
            [
                "startup",
                "capture",
                "hardware_monitor",
                "counter_sampler",
                "terminal_guard",
            ]
            .contains(value)
        })
        .unwrap_or("capture");
    let class = response["errorClass"]
        .as_str()
        .filter(|value| {
            [
                "Error",
                "AssertionError",
                "AbortError",
                "TypeError",
                "SyntaxError",
            ]
            .contains(value)
        })
        .unwrap_or("Error");
    format!("owned memory accounting refused: code={code} phase={phase} class={class}")
}

pub(super) struct OwnedProcessMemory {
    child: Child,
    input: ChildStdin,
    responses: Receiver<Result<Value, String>>,
    sequence: u64,
    finished: bool,
    run_deadline: Instant,
}
impl OwnedProcessMemory {
    pub(super) fn start() -> Result<Self, String> {
        if !cfg!(target_os = "windows") {
            return Err("FLUX3 owned-process accounting requires Windows WDDM".to_owned());
        }
        let (deadline_ms, run_deadline) = fixed_admission_deadline()?;
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/flux3-owned-process-memory.mjs");
        let mut child = Command::new(
            std::env::var("SCENEWORKS_FLUX3_BUDGET_NODE")
                .map_err(|error| format!("owned sampler Node binding: {error}"))?,
        )
        .arg(script)
        .arg(std::process::id().to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("start owned memory collector: {error}"))?;
        let input = child.stdin.take().ok_or("owned collector stdin missing")?;
        let output = child
            .stdout
            .take()
            .ok_or("owned collector stdout missing")?;
        let (sender, responses) = mpsc::sync_channel(2);
        std::thread::spawn(move || {
            let mut reader = BufReader::new(output);
            loop {
                let mut bytes = Vec::new();
                let response = match reader
                    .by_ref()
                    .take(1024 * 1024 + 1)
                    .read_until(b'\n', &mut bytes)
                {
                    Ok(0) => break,
                    Ok(_) if bytes.len() > 1024 * 1024 || bytes.last() != Some(&b'\n') => {
                        Err("owned collector response bound/truncation".to_owned())
                    }
                    Ok(_) => serde_json::from_slice(&bytes)
                        .map_err(|error| format!("owned collector invalid response: {error}")),
                    Err(error) => Err(format!("owned collector read: {error}")),
                };
                let failed = response.is_err();
                if sender.send(response).is_err() || failed {
                    break;
                }
            }
        });
        let mut collector = Self {
            child,
            input,
            responses,
            sequence: 0,
            finished: false,
            run_deadline,
        };
        let startup_wait = collector.remaining_wait()?.min(Duration::from_secs(45));
        let ready = collector.receive(startup_wait)?;
        if ready["ready"] != true
            || ready["identity"]["pid"] != std::process::id()
            || ready["runDeadline"] != deadline_ms
        {
            return Err("owned collector READY/PID/deadline mismatch".to_owned());
        }
        collector.boundary(Some("load"))?;
        Ok(collector)
    }
    fn remaining_wait(&self) -> Result<Duration, String> {
        self.run_deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| "owned collector fixed deadline expired".to_owned())
    }
    fn receive(&mut self, timeout: Duration) -> Result<Value, String> {
        let response = self
            .responses
            .recv_timeout(timeout)
            .map_err(|error| format!("owned collector missing response: {error}"))??;
        if response["ok"] != true {
            return Err(safe_protocol_refusal(&response));
        }
        Ok(response)
    }
    pub(super) fn check_health(&mut self) -> Result<(), String> {
        match self.responses.try_recv() {
            Err(mpsc::TryRecvError::Empty) => Ok(()),
            Err(mpsc::TryRecvError::Disconnected) => Err("owned collector disconnected".to_owned()),
            Ok(Err(error)) => Err(error),
            Ok(Ok(response)) => Err(safe_protocol_refusal(&response)),
        }
    }
    pub(super) fn boundary(&mut self, phase: Option<&str>) -> Result<Value, String> {
        if self.finished {
            return Err("owned collector already finished".to_owned());
        }
        self.sequence += 1;
        let command = json!({"op":"boundary","phase":phase,"id":self.sequence});
        serde_json::to_writer(&mut self.input, &command)
            .map_err(|error| format!("owned collector command: {error}"))?;
        self.input
            .write_all(b"\n")
            .and_then(|()| self.input.flush())
            .map_err(|error| format!("owned collector stdin: {error}"))?;
        let remaining = self.remaining_wait()?;
        let timeout = if phase.is_none() {
            remaining
        } else {
            remaining.min(Duration::from_secs(10))
        };
        let response = self.receive(timeout)?;
        if response["id"] != self.sequence {
            return Err("owned collector boundary identity mismatch".to_owned());
        }
        Ok(response)
    }
    pub(super) fn finish(mut self) -> Result<Value, String> {
        let response = self.boundary(None)?;
        let evidence = response["evidence"].clone();
        if evidence["kind"] != "flux3_owned_process_memory_accounting_v1"
            || evidence["physicalResidencyClaim"] != false
            || evidence["allocatorActivityClaim"] != false
        {
            return Err("owned collector typed accounting evidence missing".to_owned());
        }
        self.finished = true;
        Ok(evidence)
    }
}
impl Drop for OwnedProcessMemory {
    fn drop(&mut self) {
        // Only this adapter's own collector child. Its counter subprocess exits on
        // stdin EOF; never enumerate or terminate unrelated process names.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_deadline_wait_never_extends_the_approved_timeout() {
        assert_eq!(
            bounded_remaining(61000, 60, 1000).unwrap(),
            Duration::from_secs(60)
        );
        assert_eq!(
            bounded_remaining(61000, 60, 56000).unwrap(),
            Duration::from_secs(5)
        );
        for (deadline, timeout, now) in [
            (0, 60, 1000),
            (1000, 60, 1000),
            (999, 60, 1000),
            (61001, 60, 1000),
            (61000, 0, 1000),
            (61000, 59, 1000),
            (61000, 10801, 1000),
            (u64::MAX, 60, 1000),
        ] {
            assert!(bounded_remaining(deadline, timeout, now).is_err());
        }
    }
    #[test]
    fn protocol_failures_retain_safe_diagnosis_without_raw_data() {
        let response = json!({"ok":false,"failureCode":"FLUX3_OWNED_MEMORY_CAPTURE_REFUSED","phase":"hardware_monitor","errorClass":"AssertionError","error":"DECLARED_PRIVATE_SENTINEL"});
        let message = safe_protocol_refusal(&response);
        assert!(message.contains("FLUX3_OWNED_MEMORY_CAPTURE_REFUSED"));
        assert!(message.contains("hardware_monitor"));
        assert!(message.contains("AssertionError"));
        assert!(!message.contains("DECLARED_PRIVATE_SENTINEL"));
        let hostile = json!({"failureCode":"DECLARED_PRIVATE_SENTINEL","phase":"DECLARED_PRIVATE_SENTINEL","errorClass":"DECLARED_PRIVATE_SENTINEL"});
        assert!(!safe_protocol_refusal(&hostile).contains("DECLARED_PRIVATE_SENTINEL"));
    }
}
