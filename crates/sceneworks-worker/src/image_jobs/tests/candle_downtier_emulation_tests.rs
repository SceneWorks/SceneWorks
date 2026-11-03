use super::*;
use serde_json::json;

#[test]
#[ignore = "sc-10733 AC#6 e2e; run with SCENEWORKS_CUDA_VRAM_CAP_GB=30"]
fn candle_downtier_via_emulation_knob() {
    let cap = crate::vram_gate::cuda_vram_cap_gb().expect(
        "set SCENEWORKS_CUDA_VRAM_CAP_GB (e.g. 30) — between the q4 (~28) and q8 (~38) peaks",
    );
    // The knob-emulated budget: no real reading + the cap ⇒ a synthetic `free = total = cap` budget,
    // so this exercises the whole chain without a CUDA card.
    let budget = crate::vram_gate::apply_vram_cap(None, crate::vram_gate::cuda_vram_cap_gb());
    // Krea 2 Turbo candle tiers (builtin.models.jsonc, measured — sc-12126): q4 26.4, q8 35.9 (+2
    // headroom ⇒ peaks 28.4 / 37.9).
    let manifest = json!({ "candle": { "vramGbByTier": { "q4": 26.4, "q8": 35.9 } } })
        .as_object()
        .cloned()
        .unwrap();
    // Not sequential-capable (krea keeps the resident path) ⇒ resident-or-reject.
    let q8_fit = candle_tier_fit(&manifest, "q8", budget, false, 0);
    let q4_fit = candle_tier_fit(&manifest, "q4", budget, false, 0);
    assert!(
        matches!(q8_fit, TierFit::TooBig { .. }),
        "q8 (~38 GB) must exceed the {cap} GB emulated card: {q8_fit:?}"
    );
    assert_eq!(q4_fit, TierFit::Fits, "q4 (~28 GB) must fit {cap} GB");
    assert_eq!(
        choose_downtier("q8", &[("q8", q8_fit), ("q4", q4_fit)]),
        DowntierPick::Downtier("q4"),
        "q8 default must capability-downtier to q4 under the {cap} GB emulated cap"
    );
    eprintln!("emulation knob cap={cap} GB → q8 default DOWNTIERED to q4 (sc-10733 ✓)");
}
