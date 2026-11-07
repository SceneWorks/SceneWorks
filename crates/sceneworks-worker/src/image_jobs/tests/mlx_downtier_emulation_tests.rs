use super::super::mlx::mlx_tier_fit;
use super::super::LoadSpec;
use super::super::WeightsSource;
use super::choose_downtier;
use super::DowntierPick;
use super::TierFit;
use crate::PathBuf;
#[test]
#[ignore = "sc-10733 AC#6 e2e; run with SCENEWORKS_MLX_MEMORY_CAP_GB=21"]
fn mlx_downtier_via_emulation_knob() {
    let cap = crate::mlx_fit_gate::mlx_memory_cap_gb().expect(
        "set SCENEWORKS_MLX_MEMORY_CAP_GB (e.g. 21) — between the q4 (~19) and q8 (~23) peaks",
    );
    // Sparse tier dirs: q8 ~5 GiB, q4 ~1 GiB LOGICAL (set_len ⇒ no real disk on APFS). The gate sums
    // `metadata.len()`, so these read as 5/1 GiB → predicted peaks 23/19 GiB (+18 headroom).
    let root_guard = tempfile::Builder::new()
        .prefix("mlx_downtier_emu_")
        .tempdir()
        .expect("temp dir");
    let root = root_guard.path();
    let make_tier = |tier: &str, gib: u64| -> PathBuf {
        let dir = root.join(tier).join("transformer");
        std::fs::create_dir_all(&dir).expect("mk tier dir");
        let file = std::fs::File::create(dir.join("model.safetensors")).expect("mk weights");
        file.set_len(gib * 1024 * 1024 * 1024)
            .expect("sparse weights");
        root.join(tier)
    };
    let q8_dir = make_tier("q8", 5);
    let q4_dir = make_tier("q4", 1);
    // Unregistered engine ⇒ no footprint / not sequential-capable ⇒ resident-or-reject (te=0).
    let engine = "unregistered_downtier_probe";
    let probe = |dir: &PathBuf| LoadSpec::new(WeightsSource::Dir(dir.clone()));
    let q8_fit = mlx_tier_fit(engine, &probe(&q8_dir));
    let q4_fit = mlx_tier_fit(engine, &probe(&q4_dir));
    assert!(
        matches!(q8_fit, TierFit::TooBig { .. }),
        "q8 (~23 GiB) must exceed the {cap} GB emulated budget: {q8_fit:?}"
    );
    assert_eq!(q4_fit, TierFit::Fits, "q4 (~19 GiB) must fit {cap} GB");
    // The full decision: a q8 DEFAULT downtiers to the installed q4 that fits (acceptance #2).
    let candidates = [("q8", q8_fit), ("q4", q4_fit)];
    assert_eq!(
        choose_downtier("q8", &candidates),
        DowntierPick::Downtier("q4"),
        "q8 default must capability-downtier to q4 under the {cap} GB emulated cap"
    );
    eprintln!("emulation knob cap={cap} GB → q8 default DOWNTIERED to q4 (sc-10733 ✓)");
}
