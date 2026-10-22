//! Zero-budget native-boundary self-test of the SEALED FLUX 3 resource adapter (sc-26243).
//!
//! The terminal ledger refuses resource permits until each resource case is READY: bound to a
//! receipt proving that THIS sealed binary consumes exactly one permit immediately before exactly
//! one inner `Generator::generate`, and that a reused, missing or mismatched permit refuses before
//! any generate. This mode proves that on the CPU with no weights, no device and no provider: it
//! runs the PRODUCTION `plan`, `render_guarded` loop and reservation identity check
//! (`resource::reservation_input` + `resource::spawn_reservation`) against a counting generator
//! stub and a scripted single-use permit ledger. Only the Node ledger subprocess is replaced, so
//! it spends none of the campaign's permits.
//!
//! Hidden from production: reachable only through its own action AND the explicit environment
//! guard, with `CUDA_VISIBLE_DEVICES=-1`, and refused whenever any production invocation/provider
//! environment is present. The production launcher never sets the guard (and sets the invocation).
use super::*;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::rc::Rc;

pub(crate) const SELF_TEST_ACTION: &str = "flux3_native_boundary_self_test";
pub(crate) const SELF_TEST_ENV: &str = "SCENEWORKS_FLUX3_BOUNDARY_SELF_TEST";
/// Present on every production resource run (and the real ledger transport); never in self-test.
const PRODUCTION_ENV: [&str; 3] = [
    "SCENEWORKS_FLUX3_INVOCATION",
    "SCENEWORKS_FLUX3_BUDGET_NODE",
    "SCENEWORKS_FLUX3_KLEIN_ROOT",
];

/// The render function under test: production passes `render_guarded`; tests substitute mutants.
pub(super) type Render = fn(
    &Plan,
    &dyn Generator,
    &MemoryRunContext,
    &mut dyn FnMut(protocol::ReferencePhase),
    &mut dyn FnMut(&Plan, &GenerationRequest) -> Result<Value, String>,
) -> Result<Value, String>;

/// Refuse unless the explicit self-test guard is set, the device is hidden, and no production
/// invocation/provider environment exists.
pub(super) fn guard(env: &dyn Fn(&str) -> Option<String>) -> Result<(), String> {
    if env(SELF_TEST_ENV).as_deref() != Some("1") {
        return Err(format!(
            "BLOCKED: native-boundary self-test is unreachable without {SELF_TEST_ENV}=1"
        ));
    }
    if env("CUDA_VISIBLE_DEVICES").as_deref() != Some("-1") {
        return Err("BLOCKED: native-boundary self-test requires CUDA_VISIBLE_DEVICES=-1".into());
    }
    for key in PRODUCTION_ENV {
        if env(key).is_some() {
            return Err(format!(
                "BLOCKED: native-boundary self-test refused: production environment {key} is present"
            ));
        }
    }
    if env("SCENEWORKS_GPU_ID").is_some_and(|gpu| gpu != "cpu") {
        return Err("BLOCKED: native-boundary self-test refused: a GPU is selected".into());
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// SHA-256 of this running executable: the sealed binary under test.
pub(super) fn current_exe_sha256() -> Result<String, String> {
    let path = std::env::current_exe().map_err(|e| format!("current executable: {e}"))?;
    let bytes = std::fs::read(&path).map_err(|e| format!("read current executable: {e}"))?;
    Ok(sha256_hex(&bytes))
}

pub(crate) fn native_boundary_self_test(request: &Value, lane: &str) -> Result<Value, String> {
    guard(&|key| std::env::var(key).ok())?;
    run_self_test(request, lane, render_guarded, &current_exe_sha256()?)
}

/// Counting, tensor-neutral generator: records every `generate` and its order relative to permit
/// consumption; emits the production phase events; returns one image of the requested geometry.
struct CountingGenerator {
    descriptor: core::ModelDescriptor,
    log: Rc<RefCell<Vec<&'static str>>>,
    generate_calls: Cell<usize>,
}

impl CountingGenerator {
    fn new(log: Rc<RefCell<Vec<&'static str>>>) -> Self {
        Self {
            descriptor: core::ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: None,
                id: "flux3-native-boundary-self-test",
                family: "flux3",
                backend: "self-test",
                modality: core::Modality::Image,
                capabilities: core::Capabilities::default(),
                required_components: &[],
                control_kinds: None,
            },
            log,
            generate_calls: Cell::new(0),
        }
    }
}

struct CountingScope;

impl core::MemoryRequestScope for CountingScope {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> core::Result<()> {
        Ok(())
    }
    fn enter_phase(&mut self, _: MemoryPhase) -> core::Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _: MemoryPhase) -> core::Result<()> {
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> core::Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _: u32) -> core::Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> core::Result<()> {
        Ok(())
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> core::Result<()> {
        Ok(())
    }
}

impl Generator for CountingGenerator {
    fn descriptor(&self) -> &core::ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, _: &GenerationRequest) -> core::Result<()> {
        Ok(())
    }
    fn begin_memory_strategy_request(
        &self,
        _: &MemoryRunContext,
    ) -> core::Result<Option<Box<dyn core::MemoryRequestScope + '_>>> {
        Ok(Some(Box::new(CountingScope)))
    }
    fn generate(
        &self,
        request: &GenerationRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> core::Result<GenerationOutput> {
        self.generate_calls.set(self.generate_calls.get() + 1);
        self.log.borrow_mut().push("generate");
        if request.enhance_prompt {
            progress(Progress::PromptUpsamplingComplete);
        }
        let total = request.steps.unwrap_or(1).max(1);
        progress(Progress::ConditioningComplete);
        progress(Progress::Step { current: 1, total });
        progress(Progress::Decoding);
        Ok(GenerationOutput::Images(vec![core::Image {
            width: request.width,
            height: request.height,
            pixels: vec![0; request.width as usize * request.height as usize * 3],
        }]))
    }
}

/// Scripted single-use permit ledger (replaces ONLY the Node transport): consumes the permit or
/// refuses, and echoes the bound identity exactly like a real ledger event.
fn scripted_ledger(permits: &RefCell<BTreeSet<String>>, input: &Value) -> Result<Value, String> {
    let id = input["invocationId"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    if !permits.borrow_mut().remove(&id) {
        return Err("BLOCKED: scripted permit missing or already spent".into());
    }
    let mut event = input.clone();
    let canonical = serde_json::to_vec(input).map_err(|e| e.to_string())?;
    event["eventSha256"] = json!(sha256_hex(&canonical));
    event["selfTest"] = json!(true);
    event["hardwareObservation"] = json!({
        "adapterPid": input["adapterPid"], "sourceSha": input["sourceSha"],
        "binarySha256": input["binarySha256"], "approvedPlanSha256": input["approvedPlanSha256"],
        "invocationId": input["invocationId"], "caseId": input["caseId"],
        "requestSha256": input["requestSha256"], "selfTest": true});
    Ok(event)
}

struct Outcome {
    ok: bool,
    reserve_calls: usize,
    generate_calls: usize,
    order: Vec<&'static str>,
}

fn attempt(
    render: Render,
    plan: &Plan,
    permits: &RefCell<BTreeSet<String>>,
    invocation: &str,
) -> Outcome {
    let log = Rc::new(RefCell::new(Vec::new()));
    let generator = CountingGenerator::new(log.clone());
    let reserve_calls = Cell::new(0usize);
    let mut reserve = |plan: &Plan, request: &GenerationRequest| -> Result<Value, String> {
        reserve_calls.set(reserve_calls.get() + 1);
        log.borrow_mut().push("reserve");
        let input = resource::reservation_input(plan, request, invocation)?;
        resource::spawn_reservation(input, |input| scripted_ledger(permits, input))
    };
    let ok = render(
        plan,
        &generator,
        &context(plan, 1, 1 << 34),
        &mut |_| {},
        &mut reserve,
    )
    .is_ok();
    let order = log.borrow().clone();
    Outcome {
        ok,
        reserve_calls: reserve_calls.get(),
        generate_calls: generator.generate_calls.get(),
        order,
    }
}

fn exact_keys(value: &Value, keys: &[&str], label: &str) -> Result<(), String> {
    let mut actual: Vec<&str> = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?
        .keys()
        .map(String::as_str)
        .collect();
    actual.sort_unstable();
    let mut expected = keys.to_vec();
    expected.sort_unstable();
    if actual != expected {
        return Err(format!("{label} must carry exactly {expected:?}"));
    }
    Ok(())
}

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| format!("{key} must be a non-empty string"))
}

/// Run the boundary proofs for every case against `render`, binding the receipt to `binary_sha`.
pub(super) fn run_self_test(
    request: &Value,
    lane: &str,
    render: Render,
    binary_sha: &str,
) -> Result<Value, String> {
    exact_keys(
        request,
        &[
            "action",
            "lane",
            "expectedBinarySha256",
            "buildReceiptSha256",
            "buildReceiptBinarySha256",
            "sourceSha",
            "inferenceMainSha",
            "cases",
        ],
        "self-test request",
    )?;
    if text(request, "action")? != SELF_TEST_ACTION || text(request, "lane")? != lane {
        return Err("BLOCKED: self-test action/lane differs from this sealed binary".into());
    }
    let expected = text(request, "expectedBinarySha256")?;
    if expected != binary_sha || text(request, "buildReceiptBinarySha256")? != binary_sha {
        return Err(
            "BLOCKED: sealed binary hash mismatch: this executable is not the approved build"
                .into(),
        );
    }
    let build_receipt = text(request, "buildReceiptSha256")?;
    let source = text(request, "sourceSha")?;
    if text(request, "inferenceMainSha")? != protocol::INFERENCE_PIN {
        return Err("BLOCKED: self-test inference identity differs from this binary's pin".into());
    }
    let cases = request["cases"]
        .as_array()
        .filter(|cases| !cases.is_empty())
        .ok_or_else(|| "self-test requires at least one resource case".to_owned())?;
    let mut seen = BTreeSet::new();
    let mut proofs = Vec::new();
    for case in cases {
        exact_keys(
            case,
            &["invocationId", "caseId", "kind", "requestSha256", "request"],
            "self-test case",
        )?;
        let invocation_id = text(case, "invocationId")?.to_owned();
        if !seen.insert(invocation_id.clone()) {
            return Err("BLOCKED: duplicate self-test invocation".into());
        }
        let plan = plan(&case["request"], lane)?;
        let binding = |request_sha: &str, case_id: &str| {
            json!({"approvalFile":"SELF-TEST-NO-APPROVAL-FILE","admissionFile":"SELF-TEST-NO-ADMISSION-FILE",
                "adapterPid":0,"sourceSha":source,"binarySha256":binary_sha,"ledgerRoot":"SELF-TEST-SCRIPTED-LEDGER",
                "ledgerId":"SELF-TEST-SCRIPTED-LEDGER","approvedPlanSha256":"SELF-TEST-NOT-A-CAMPAIGN-PLAN",
                "invocationId":invocation_id,"backend":lane,"caseId":case_id,"kind":case["kind"],
                "requestSha256":request_sha})
            .to_string()
        };
        let case_id = text(case, "caseId")?;
        let request_sha = text(case, "requestSha256")?;
        let valid = binding(request_sha, case_id);
        let fresh = || RefCell::new(BTreeSet::from([invocation_id.clone()]));
        // 1) One permit -> reserve BEFORE exactly one generate, and the digest the planner fixed
        //    must equal the one this sealed binary computes for the real native invocation.
        let ledger = fresh();
        let first = attempt(render, &plan, &ledger, &valid);
        if !first.ok
            || first.reserve_calls != 1
            || first.generate_calls != 1
            || first.order != ["reserve", "generate"]
        {
            return Err(format!(
                "BLOCKED: {case_id}: expected one reservation before exactly one generate, observed ok={} reserve={} generate={} order={:?}",
                first.ok, first.reserve_calls, first.generate_calls, first.order
            ));
        }
        // 2) Reused permit (same ledger) refuses before generate.
        let reused = attempt(render, &plan, &ledger, &valid);
        // 3) Missing permit refuses before generate.
        let missing = attempt(render, &plan, &RefCell::new(BTreeSet::new()), &valid);
        // 4) Mismatched request digest / case refuse before generate (production identity check).
        let wrong_digest = attempt(render, &plan, &fresh(), &binding(&"0".repeat(64), case_id));
        let wrong_case = attempt(
            render,
            &plan,
            &fresh(),
            &binding(request_sha, "candle.distilled.not-this-case.bf16"),
        );
        for (label, outcome) in [
            ("reused", &reused),
            ("missing", &missing),
            ("mismatched-digest", &wrong_digest),
            ("mismatched-case", &wrong_case),
        ] {
            if outcome.ok || outcome.generate_calls != 0 {
                return Err(format!(
                    "BLOCKED: {case_id}: {label} permit must refuse before any generate (generate={})",
                    outcome.generate_calls
                ));
            }
        }
        proofs.push(json!({
            "invocationId": invocation_id, "caseId": case_id, "kind": case["kind"],
            "requestSha256": request_sha,
            "valid": {"reserveCalls": 1, "generateCalls": 1, "order": ["reserve", "generate"]},
            "reused": {"refused": true, "generateCalls": 0},
            "missing": {"refused": true, "generateCalls": 0},
            "mismatchedDigest": {"refused": true, "generateCalls": 0},
            "mismatchedCase": {"refused": true, "generateCalls": 0}}));
    }
    Ok(json!({
        "schemaVersion": 1, "kind": "flux3-native-boundary-self-test-receipt",
        "mode": "zero-budget-cpu-self-test: production plan/render/reservation path, counting generator, scripted single-use ledger; no weights, device or provider",
        "lane": lane, "binarySha256": binary_sha, "buildReceiptSha256": build_receipt,
        "sourceSha": source, "inferenceMainSha": protocol::INFERENCE_PIN,
        "campaignPermitsConsumed": 0, "cases": proofs}))
}
