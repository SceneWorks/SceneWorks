use super::*;
use crate::tests::support::isolate_hf_cache;
use std::os::unix::fs::symlink;

/// Bind cache resolution to this fixture's literal data directory before any seed or delete.
fn fixture_cache_root(data_dir: &FsPath) -> sceneworks_core::hf_home::test_support::CacheRootGuard {
    let hub = data_dir.join("cache/huggingface/hub");
    let guard = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
        data_dir.to_path_buf(),
        hub.clone(),
    );
    assert_eq!(huggingface_hub_cache_dir(data_dir), hub);
    guard
}

fn fixture_repo_cache_path(data_dir: &FsPath, repo: &str) -> PathBuf {
    let path = huggingface_repo_cache_path(data_dir, repo).unwrap();
    assert!(
        path.starts_with(data_dir),
        "fixture cache must be inside its temporary data directory before seeding or deletion: {}",
        path.display()
    );
    path
}

#[test]
fn external_cache_is_refused_before_fixture_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path().join("data");
    let external_cache = tmp.path().join("outside-fixture");
    let _cache_root = sceneworks_core::hf_home::test_support::CacheRootGuard::new(
        data_dir.clone(),
        external_cache.clone(),
    );
    assert!(
        std::panic::catch_unwind(|| { fixture_repo_cache_path(&data_dir, "Org/fixture") }).is_err()
    );
    assert!(!external_cache.exists(), "refusal must precede any writes");
}

/// Write (once) a `blobs/<etag>` file of `bytes.len()` bytes and return its path.
fn blob(repo: &FsPath, etag: &str, bytes: &[u8]) -> PathBuf {
    let blobs = repo.join("blobs");
    std::fs::create_dir_all(&blobs).unwrap();
    let path = blobs.join(etag);
    if !path.exists() {
        std::fs::write(&path, bytes).unwrap();
    }
    path
}

/// Materialize `snapshots/rev/<rel>` as a symlink to `blob_path` (the production cache links
/// relatively; an absolute link resolves identically under `canonicalize`).
fn link(repo: &FsPath, rel: &str, blob_path: &FsPath) {
    let link_path = repo.join("snapshots").join("rev").join(rel);
    std::fs::create_dir_all(link_path.parent().unwrap()).unwrap();
    symlink(blob_path, &link_path).unwrap();
}

/// Seed one snapshot file backed by its own fresh blob of `size` bytes.
fn seed(repo: &FsPath, rel: &str, etag: &str, size: usize) {
    let blob_path = blob(repo, etag, &vec![0u8; size]);
    link(repo, rel, &blob_path);
}

#[tokio::test]
async fn deletes_only_the_target_tiers_blobs_and_symlinks() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let repo = hub.join("models--Org--repo");
    seed(&repo, "q4/model.safetensors", "q4a", 100);
    seed(&repo, "q4/config.json", "q4b", 200);
    seed(&repo, "q8/model.safetensors", "q8a", 500);

    let removal = remove_tier_artifacts(
        Some(repo.clone()),
        None,
        &["q4/*".to_owned()],
        &[],
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    // q4's blobs AND snapshot symlinks are gone; the emptied q4 dir is pruned.
    assert!(!repo.join("blobs/q4a").exists());
    assert!(!repo.join("blobs/q4b").exists());
    assert!(!repo.join("snapshots/rev/q4").exists());
    // q8 is fully intact.
    assert!(repo.join("blobs/q8a").exists());
    assert!(repo.join("snapshots/rev/q8/model.safetensors").exists());
    // Reclaimed bytes = q4's blob sizes only (100 + 200), never the shared skeleton.
    assert_eq!(removal.reclaimed_bytes, 300);
    assert!(removal.trash_failed_paths.is_empty());
    assert!(!removal.removed_paths.is_empty());
}

#[tokio::test]
async fn retains_a_blob_shared_with_a_surviving_tier() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let repo = hub.join("models--Org--repo");
    // A single blob referenced by BOTH tiers (identical etag/content), plus a q4-only blob.
    let shared = blob(&repo, "shared", &vec![0u8; 400]);
    link(&repo, "q4/shared.safetensors", &shared);
    link(&repo, "q8/shared.safetensors", &shared);
    seed(&repo, "q4/only.safetensors", "q4only", 100);

    let removal = remove_tier_artifacts(
        Some(repo.clone()),
        None,
        &["q4/*".to_owned()],
        &[],
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    // The shared blob survives — q8 still references it — and q8's link still resolves.
    assert!(repo.join("blobs/shared").exists());
    assert!(repo.join("snapshots/rev/q8/shared.safetensors").exists());
    // q4's exclusive blob and all q4 links are removed.
    assert!(!repo.join("blobs/q4only").exists());
    assert!(!repo.join("snapshots/rev/q4").exists());
    // Only the exclusive blob's bytes count as reclaimed; the shared blob does not.
    assert_eq!(removal.reclaimed_bytes, 100);
}

/// sc-24112 — the SHIPPED `qwen_image_2_1` entry, as published at sc-24114 (no placeholder
/// flags left): deleting bf16 reclaims the whole upstream `Qwen/Qwen-Image-2.1` snapshot
/// (30.86 GiB in production, the tier's only bytes) and leaves the SceneWorks re-host holding
/// q8/q4 untouched. Before, the scope-less bf16 row could never be reclaimed per tier at all.
///
/// *Mutation that reds this:* `is_sole_repo_tier` answering `false` — the scope is refused.
#[tokio::test]
async fn deleting_the_qwen_image_2_1_bf16_tier_reclaims_only_the_upstream_snapshot() {
    let shipped = sceneworks_core::builtin_manifests::builtin_manifest("builtin.models.jsonc")
        .expect("valid embedded model catalog");
    let model = shipped["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|model| model["id"] == "qwen_image_2_1")
        .expect("qwen_image_2_1 is in the shipped catalog")
        .clone();
    let bf16 = model_download_for_variant(&model, "bf16").expect("bf16 row");
    let scope = tier_delete_scope(&model, &bf16, "bf16").expect("bf16 is reclaimable alone");
    assert_eq!(scope.repo, "Qwen/Qwen-Image-2.1");
    assert!(scope.retained_files.is_empty());

    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let upstream = hub.join("models--Qwen--Qwen-Image-2.1");
    let rehost = hub.join("models--SceneWorks--qwen-image-2-1-mlx");
    seed(
        &upstream,
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "u1",
        300,
    );
    seed(&upstream, "text_encoder/model.safetensors", "u2", 200);
    seed(&upstream, "model_index.json", "u3", 10);
    seed(&rehost, "q8/transformer/model.safetensors", "r8", 70);
    seed(&rehost, "q4/transformer/model.safetensors", "r4", 40);

    let removal = remove_tier_artifacts(
        Some(upstream.clone()),
        None,
        &scope.files,
        &scope.retained_files,
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    assert_eq!(
        removal.reclaimed_bytes, 510,
        "every upstream blob is the bf16 tier's"
    );
    assert!(!upstream.join("blobs/u1").exists());
    assert!(!upstream.join("snapshots/rev/model_index.json").exists());
    // The re-host — the q8/q4 tiers — is a different repo and is never scanned.
    assert!(rehost.join("blobs/r8").exists());
    assert!(rehost.join("blobs/r4").exists());
    assert!(rehost
        .join("snapshots/rev/q8/transformer/model.safetensors")
        .exists());
    assert!(rehost
        .join("snapshots/rev/q4/transformer/model.safetensors")
        .exists());
}

#[tokio::test]
async fn draining_the_last_tier_removes_the_repo_cache_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let repo = hub.join("models--Org--repo");
    seed(&repo, "q4/model.safetensors", "q4a", 100);

    let removal = remove_tier_artifacts(
        Some(repo.clone()),
        None,
        &["q4/*".to_owned()],
        &[],
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    // No tier remains → the whole models--repo dir is pruned (no bare refs/ skeleton left behind).
    assert!(!repo.exists());
    assert_eq!(removal.reclaimed_bytes, 100);
}

#[tokio::test]
async fn removes_real_tier_files_from_the_managed_mirror() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let managed = models.join("Org__repo");
    let write = |rel: &str, size: usize| {
        let path = managed.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, vec![0u8; size]).unwrap();
    };
    // A turnkey install writes REAL files (not blob symlinks) under the managed dir.
    write("q4/model.safetensors", 300);
    write("q8/model.safetensors", 500);

    let removal = remove_tier_artifacts(
        None,
        Some(managed.clone()),
        &["q4/*".to_owned()],
        &[],
        std::slice::from_ref(&models),
        true,
    )
    .await
    .unwrap();

    assert!(!managed.join("q4").exists());
    assert!(managed.join("q8/model.safetensors").exists());
    assert_eq!(removal.reclaimed_bytes, 300);
}

#[tokio::test]
async fn empty_tier_files_is_a_no_op() {
    // The "never scope a delete to everything" guard: an empty file filter removes nothing.
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let repo = hub.join("models--Org--repo");
    seed(&repo, "q4/model.safetensors", "q4a", 100);

    let removal = remove_tier_artifacts(
        Some(repo.clone()),
        None,
        &[],
        &[],
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    assert!(repo.join("blobs/q4a").exists());
    assert!(removal.removed_paths.is_empty());
    assert_eq!(removal.reclaimed_bytes, 0);
}

/// A model whose logical tiers SHARE one file predicate reclaims nothing and keeps the snapshot.
///
/// This is the load-time-quant contract, and it is still the right behavior for any family that
/// ships one dense snapshot per variant. Mage-Flow used to be the shipping example; since
/// sc-14980 its tiers are physically distinct, so the fixture below is a synthetic overlapping
/// model rather than a Mage row — see
/// `mage_flow_per_tier_delete_reclaims_only_that_tiers_dit` for Mage's actual shape.
#[tokio::test]
async fn overlapping_logical_tiers_retain_the_shared_snapshot_and_reclaim_zero() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let repo = hub.join("models--Org--overlapping-tiers");
    seed(
        &repo,
        "transformer/diffusion_pytorch_model.safetensors",
        "dit",
        100,
    );
    seed(&repo, "text_encoder/model.safetensors", "te", 200);
    seed(&repo, "vae/diffusion_pytorch_model.safetensors", "vae", 50);

    let complete = vec![
        "transformer/*".to_owned(),
        "text_encoder/*".to_owned(),
        "vae/*".to_owned(),
    ];
    let removal = remove_tier_artifacts(
        Some(repo.clone()),
        None,
        &complete,
        &complete,
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    assert_eq!(removal.reclaimed_bytes, 0);
    assert!(removal.removed_paths.is_empty());
    assert_eq!(removal.retained_paths.len(), 3);
    assert!(repo
        .join("snapshots/rev/transformer/diffusion_pytorch_model.safetensors")
        .exists());
    assert!(repo.join("blobs/te").exists());
    assert!(repo.join("blobs/vae").exists());
}

/// sc-14980 / sc-14979 — Mage-Flow's per-tier delete reclaims REAL bytes, and reclaims only its
/// own tier's DiT.
///
/// This is the physical-reclaim acceptance sc-14046 delegated. It replaces the honest-but-empty
/// zero-byte outcome the load-time-quant layout could offer, and it is the reason the tiers had
/// to become physically distinct artifacts. The byte counts are the REAL sizes measured on the
/// uploaded mirrors, so a re-host that silently changed the layout would move them.
///
/// Three things must hold simultaneously, and each is a distinct way the design could fail:
///   - the deleted tier's DiT bytes are actually reclaimed (not 0);
///   - the OTHER installed tier's DiT survives intact (predicates are disjoint);
///   - the SHARED text encoder + VAE survive — they live in a different repo entirely and are
///     co-requisites, so a variant/tier delete can never strand another installed variant.
#[tokio::test]
async fn mage_flow_per_tier_delete_reclaims_only_that_tiers_dit() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    // The variant mirror: q4 and q8 DiT tiers installed side by side.
    let variant = hub.join("models--SceneWorks--Mage-Flow-Base");
    // Real uploaded sizes, scaled down by 1e6 so the fixture stays a unit test; the RATIO and
    // the disjointness are what the assertions are about.
    const Q4_DIT: usize = 2316; // 2.317 GB
    const Q8_DIT: usize = 4374; // 4.374 GB
    seed(
        &variant,
        "q4/transformer/diffusion_pytorch_model.safetensors",
        "q4dit",
        Q4_DIT,
    );
    seed(&variant, "q4/model_index.json", "q4idx", 1);
    seed(
        &variant,
        "q8/transformer/diffusion_pytorch_model.safetensors",
        "q8dit",
        Q8_DIT,
    );
    seed(&variant, "q8/model_index.json", "q8idx", 1);
    // The SHARED components mirror — a different repo, reached only through co-requisite rows.
    let components = hub.join("models--SceneWorks--Mage-Flow-Components-mlx");
    seed(
        &components,
        "q4/text_encoder/model.safetensors",
        "q4te",
        2514,
    );
    seed(
        &components,
        "q4/vae/diffusion_pytorch_model.safetensors",
        "q4vae",
        345,
    );

    // Delete the q4 tier. `retained` carries the SURVIVING tiers' predicates, exactly as
    // `delete_model_variant` builds it (co-requisites are excluded there by construction, which
    // is why no components predicate appears here).
    let removal = remove_tier_artifacts(
        Some(variant.clone()),
        None,
        &["q4/*".to_owned()],
        &["q8/*".to_owned()],
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    // 1. Real bytes, not zero: exactly the q4 DiT + its tier index.
    assert_eq!(
        removal.reclaimed_bytes,
        (Q4_DIT + 1) as u64,
        "a Mage tier delete must reclaim that tier's own bytes — a 0 here means the tiers went \
         back to sharing one file predicate"
    );
    assert!(!removal.removed_paths.is_empty());
    assert!(!variant.join("blobs/q4dit").exists());
    assert!(!variant.join("snapshots/rev/q4").exists());

    // 2. The sibling tier is untouched.
    assert!(variant.join("blobs/q8dit").exists());
    assert!(variant
        .join("snapshots/rev/q8/transformer/diffusion_pytorch_model.safetensors")
        .exists());

    // 3. The shared components are untouched — different repo, never in the delete's scope.
    assert!(components.join("blobs/q4te").exists());
    assert!(components.join("blobs/q4vae").exists());
    assert!(components
        .join("snapshots/rev/q4/text_encoder/model.safetensors")
        .exists());
}

/// The `SceneWorks/minimax-h3-mlx` shape (sc-17150 / sc-17158): ONE repo holding two DiT
/// partitions per tier, each owned by a DIFFERENT catalog entry. Seeds `tier`'s `transformer/`
/// (owned by `minimax_h3`) and `transformer_ref/` (owned by `minimax_h3_ref`).
///
/// The two partitions ship a BYTE-IDENTICAL `config.json` (they carry the same architecture and
/// the same 638 tensor names; only the weights differ), so the hub cache stores it as ONE blob
/// that both snapshot entries symlink to. That shared blob is the trap: unlinking it with the base
/// partition would leave the reference partition's `config.json` dangling.
fn seed_minimax_tier(repo: &FsPath, tier: &str, base_etag: &str, ref_etag: &str, size: usize) {
    let shared_config = blob(repo, &format!("{tier}-config"), b"{}");
    link(
        repo,
        &format!("{tier}/transformer/config.json"),
        &shared_config,
    );
    link(
        repo,
        &format!("{tier}/transformer_ref/config.json"),
        &shared_config,
    );
    for partition in ["transformer", "transformer_ref"] {
        let etag = if partition == "transformer" {
            base_etag
        } else {
            ref_etag
        };
        seed(
            repo,
            &format!("{tier}/{partition}/diffusion_pytorch_model.safetensors.index.json"),
            &format!("{etag}-index"),
            1,
        );
        seed(
            repo,
            &format!("{tier}/{partition}/diffusion_pytorch_model-00001-of-00001.safetensors"),
            etag,
            size,
        );
    }
}

/// sc-19078 — a MiniMax-H3 per-tier delete reclaims that tier's own partition and NOTHING else.
///
/// Mirrors `mage_flow_per_tier_delete_reclaims_only_that_tiers_dit`, which is the shipping
/// physical-per-tier precedent. H3 adds a dimension Mage does not have: the sibling that must
/// survive is not only another TIER of the same entry but another CATALOG ENTRY's partition inside
/// the same tier of the same repo — and the two partitions share a blob.
///
/// Four things must hold at once, each a distinct way this could fail:
///   - the deleted tier's own partition bytes are actually reclaimed (not 0);
///   - the OTHER tier of the same entry survives (the tier predicates are disjoint);
///   - the SIBLING ENTRY's partition in the SAME tier survives (the partition predicates are
///     disjoint) — the case sc-17139's follow-ups flagged as reachable here for the first time;
///   - the blob the two partitions SHARE survives, so the sibling's `config.json` still resolves.
#[tokio::test]
async fn minimax_h3_per_tier_delete_reclaims_only_that_entrys_partition() {
    let tmp = tempfile::tempdir().unwrap();
    let hub = tmp.path().join("hub");
    let repo = hub.join("models--SceneWorks--minimax-h3-mlx");
    // Real hosted per-partition sizes scaled down by 1e6 (18,780,109,783 B q4 / 35,302,064,357 B
    // q8). The RATIO and the disjointness are what the assertions are about.
    const Q4_DIT: usize = 18780;
    const Q8_DIT: usize = 35302;
    seed_minimax_tier(&repo, "q4", "q4dit", "q4refdit", Q4_DIT);
    seed_minimax_tier(&repo, "q8", "q8dit", "q8refdit", Q8_DIT);

    // Delete `minimax_h3`'s q4 tier. `retained` carries the surviving tiers of THAT entry, exactly
    // as `delete_model_variant` builds it from the entry's own `downloads`.
    let removal = remove_tier_artifacts(
        Some(repo.clone()),
        None,
        &["q4/transformer/*".to_owned()],
        &[
            "q8/transformer/*".to_owned(),
            "bf16/transformer/*".to_owned(),
        ],
        std::slice::from_ref(&hub),
        true,
    )
    .await
    .unwrap();

    // 1. Real bytes: the q4 base partition's shard + its index, and nothing else. The shared
    //    `config.json` blob is NOT counted — it never left disk.
    assert_eq!(
        removal.reclaimed_bytes,
        (Q4_DIT + 1) as u64,
        "a MiniMax-H3 tier delete must reclaim that entry's own partition bytes"
    );
    assert!(!repo.join("blobs/q4dit").exists());
    assert!(!repo
        .join("snapshots/rev/q4/transformer/diffusion_pytorch_model-00001-of-00001.safetensors")
        .exists());

    // 2. The same entry's OTHER tier is untouched.
    assert!(repo.join("blobs/q8dit").exists());
    assert!(repo
        .join("snapshots/rev/q8/transformer/diffusion_pytorch_model-00001-of-00001.safetensors")
        .exists());

    // 3. The SIBLING ENTRY's partition inside the deleted tier is untouched — `minimax_h3_ref`
    //    stays installed at q4 even though its bytes live in the tier just deleted.
    assert!(repo.join("blobs/q4refdit").exists());
    assert!(repo
        .join("snapshots/rev/q4/transformer_ref/diffusion_pytorch_model-00001-of-00001.safetensors")
        .exists());

    // 4. The blob the two partitions SHARE survives and the sibling's link still resolves through
    //    it — a dangling `config.json` would make the reference entry unloadable while still
    //    reading installed.
    assert!(repo.join("blobs/q4-config").exists());
    let sibling_config = repo.join("snapshots/rev/q4/transformer_ref/config.json");
    assert!(
        std::fs::read(&sibling_config).is_ok(),
        "sibling config resolves"
    );
}

/// sc-19078 — the WHOLE-model delete is scoped when the download repo is shared.
///
/// This is the destructive half. `model_artifact_paths` resolves the repo's cache dir, so before
/// this the blanket `remove_dir_all` on `models--SceneWorks--minimax-h3-mlx` took every
/// `transformer_ref/` tier with it — up to 132.6 GB of an installed model the user never asked to
/// delete. `remove_whole_model_artifacts` removes the entry's own `files` scopes instead, with the
/// sibling entry's scopes retained.
#[tokio::test]
async fn whole_model_delete_on_a_shared_repo_keeps_the_sibling_entrys_partitions() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let _cache_root = fixture_cache_root(data_dir);
    let hub = data_dir.join("cache/huggingface/hub");
    let repo_name = "SceneWorks/minimax-h3-mlx";
    let repo = huggingface_repo_cache_path(data_dir, repo_name).unwrap();
    assert_eq!(repo, hub.join("models--SceneWorks--minimax-h3-mlx"));
    seed_minimax_tier(&repo, "q4", "q4dit", "q4refdit", 18780);
    seed_minimax_tier(&repo, "bf16", "bf16dit", "bf16refdit", 66280);

    let downloads = |partition: &str| {
        json!(["q4", "q8", "bf16"]
            .iter()
            .map(|tier| json!({
                "provider": "huggingface",
                "repo": repo_name,
                "variant": tier,
                "files": [format!("{tier}/{partition}/*")],
            }))
            .collect::<Vec<_>>())
    };
    let base = json!({ "id": "minimax_h3", "downloads": downloads("transformer") });
    let reference = json!({ "id": "minimax_h3_ref", "downloads": downloads("transformer_ref") });
    let catalog = vec![base.clone(), reference];
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

    let removal = remove_whole_model_artifacts(
        &catalog,
        "minimax_h3",
        &base,
        data_dir,
        &allowed_roots,
        true,
    )
    .await
    .unwrap();

    // Every tier of the deleted entry's own partition is gone…
    assert!(!removal.removed_paths.is_empty());
    for etag in ["q4dit", "bf16dit"] {
        assert!(!repo.join("blobs").join(etag).exists(), "{etag} removed");
    }
    assert!(!repo.join("snapshots/rev/q4/transformer").exists());
    assert!(!repo.join("snapshots/rev/bf16/transformer").exists());

    // …and every tier of the SIBLING entry's partition survives, blobs and links alike.
    for etag in ["q4refdit", "bf16refdit", "q4-config", "bf16-config"] {
        assert!(repo.join("blobs").join(etag).exists(), "{etag} retained");
    }
    for tier in ["q4", "bf16"] {
        let sibling = repo.join(format!("snapshots/rev/{tier}/transformer_ref"));
        assert!(sibling.join("config.json").exists(), "{tier} ref config");
        assert!(std::fs::read(sibling.join("config.json")).is_ok());
        assert!(sibling
            .join("diffusion_pytorch_model-00001-of-00001.safetensors")
            .exists());
    }
    // The repo cache dir itself must NOT be pruned — the sibling still lives in it.
    assert!(repo.is_dir(), "shared repo cache survives a scoped delete");
}

/// sc-25682 (E4): the generation and depth TASK installs share `speridlabs/iris-3b`; deleting
/// either entry removes only its own files and keeps the other task's, driven by the SHIPPED
/// catalog entries.
#[tokio::test]
async fn deleting_one_iris_task_keeps_the_other_tasks_files() {
    let _env = isolate_hf_cache();
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .unwrap();
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
    let catalog: Vec<Value> = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| matches!(entry["id"].as_str(), Some("iris_3b" | "iris_3b_depth")))
        .cloned()
        .collect();
    assert_eq!(catalog.len(), 2);
    let entry = |id: &str| catalog.iter().find(|e| e["id"] == id).unwrap().clone();

    let depth_files: &[&str] = &[
        "depth/config.yaml",
        "depth/model.safetensors",
        "depth/empty_prompt.safetensors",
    ];
    let generation_files: &[&str] = &["config.yaml", "model.safetensors"];
    for (deleted, kept_files, removed_files) in [
        ("iris_3b_depth", generation_files, depth_files),
        ("iris_3b", depth_files, generation_files),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let hub = data_dir.join("cache/huggingface/hub");
        let _cache_root = fixture_cache_root(data_dir);
        let repo = huggingface_repo_cache_path(data_dir, "speridlabs/iris-3b").unwrap();
        assert_eq!(repo, hub.join("models--speridlabs--iris-3b"));
        for (index, rel) in [
            "config.yaml",
            "model.safetensors",
            "depth/config.yaml",
            "depth/model.safetensors",
            "depth/empty_prompt.safetensors",
        ]
        .iter()
        .enumerate()
        {
            seed(&repo, rel, &format!("blob{index}"), 10 + index);
        }
        let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
        remove_whole_model_artifacts(
            &catalog,
            deleted,
            &entry(deleted),
            data_dir,
            &allowed_roots,
            true,
        )
        .await
        .unwrap();
        for rel in kept_files {
            assert!(
                repo.join("snapshots/rev").join(rel).exists(),
                "deleting {deleted} must keep {rel}"
            );
        }
        for rel in removed_files {
            assert!(
                !repo.join("snapshots/rev").join(rel).exists(),
                "deleting {deleted} removes its own {rel}"
            );
        }
    }
}

/// sc-25684 (E4): ALL THREE Iris task installs — generation (root files), depth (`depth/`) and
/// restoration (`upscaler/`) — share the `speridlabs/iris-3b` snapshot, and the two task exports
/// ship byte-identical `empty_prompt.safetensors` (one shared blob). Deleting any one task through
/// the SHIPPED catalog removes exactly its own files and keeps every other task's files AND their
/// blob contents — including a blob the deleted task shared with a kept one — and never touches
/// the generation task's Qwen3-VL text encoder when a depth or restoration task is deleted.
#[tokio::test]
async fn deleting_any_iris_task_keeps_every_other_tasks_files_and_shared_blobs() {
    let _env = isolate_hf_cache();
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .unwrap();
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
    let catalog = manifest["models"].as_array().unwrap().clone();
    let entry = |id: &str| catalog.iter().find(|e| e["id"] == id).unwrap().clone();
    let generation: &[&str] = &["config.yaml", "model.safetensors"];
    let depth: &[&str] = &[
        "depth/config.yaml",
        "depth/model.safetensors",
        "depth/empty_prompt.safetensors",
    ];
    let restoration: &[&str] = &[
        "upscaler/config.yaml",
        "upscaler/model.safetensors",
        "upscaler/empty_prompt.safetensors",
    ];
    for (deleted, removed, kept) in [
        ("iris_3b", generation, [depth, restoration]),
        ("iris_3b_depth", depth, [generation, restoration]),
        ("iris_3b_restore", restoration, [generation, depth]),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path();
        let hub = data_dir.join("cache/huggingface/hub");
        let _cache_root = fixture_cache_root(data_dir);
        let repo = huggingface_repo_cache_path(data_dir, "speridlabs/iris-3b").unwrap();
        assert_eq!(repo, hub.join("models--speridlabs--iris-3b"));
        for (index, rel) in generation
            .iter()
            .chain(depth)
            .chain(restoration)
            .enumerate()
        {
            if rel.ends_with("empty_prompt.safetensors") {
                // Both task exports point at ONE blob.
                seed(&repo, rel, "empty-prompt-blob", 7);
            } else {
                seed(&repo, rel, &format!("blob{index}"), 10 + index);
            }
        }
        let encoder = fixture_repo_cache_path(data_dir, "Qwen/Qwen3-VL-4B-Instruct");
        seed(&encoder, "config.json", "encoder-config", 5);
        let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];
        remove_whole_model_artifacts(
            &catalog,
            deleted,
            &entry(deleted),
            data_dir,
            &allowed_roots,
            true,
        )
        .await
        .unwrap();
        let snapshot = repo.join("snapshots/rev");
        for rel in removed {
            assert!(
                !snapshot.join(rel).exists(),
                "deleting {deleted} removes its own {rel}"
            );
        }
        for rel in kept.iter().flat_map(|files| files.iter()) {
            let bytes = std::fs::read(snapshot.join(rel))
                .unwrap_or_else(|error| panic!("deleting {deleted} broke {rel}: {error}"));
            assert!(!bytes.is_empty(), "deleting {deleted} emptied {rel}");
        }
        assert!(
            encoder.join("snapshots/rev/config.json").exists(),
            "deleting {deleted} must not touch the text encoder's own repo"
        );
    }
}

/// sc-25213 (finding 6): the two de-distill training adapter entries share ONE repo snapshot
/// (`ostris/zimage_turbo_training_adapter@654cd1bf`, one file each), so deleting one must take
/// only its own file — the sc-19078 sibling scoping applies because each entry declares its file.
/// Run against the SHIPPED catalog entries (their real `files` / `paths.model`). Mutation: give
/// both entries the same `files` or drop one entry's `files` ⇒ the sibling's blob is removed ⇒ red.
#[tokio::test]
async fn deleting_one_training_adapter_keeps_the_other_versions_file() {
    let _env = isolate_hf_cache();
    let raw = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .map(|(_, contents)| *contents)
        .unwrap();
    let manifest: Value =
        serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(raw)).unwrap();
    let catalog = manifest["models"].as_array().unwrap().clone();
    let entry = |id: &str| catalog.iter().find(|m| m["id"] == id).unwrap().clone();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let _cache_root = fixture_cache_root(data_dir);
    let hub = data_dir.join("cache/huggingface/hub");
    let repo =
        huggingface_repo_cache_path(data_dir, "ostris/zimage_turbo_training_adapter").unwrap();
    assert_eq!(
        repo,
        hub.join("models--ostris--zimage_turbo_training_adapter")
    );
    seed(
        &repo,
        "zimage_turbo_training_adapter_v1.safetensors",
        "v1blob",
        64,
    );
    seed(
        &repo,
        "zimage_turbo_training_adapter_v2.safetensors",
        "v2blob",
        128,
    );
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

    let v1 = entry("zimage_turbo_training_adapter_v1");
    remove_whole_model_artifacts(
        &catalog,
        "zimage_turbo_training_adapter_v1",
        &v1,
        data_dir,
        &allowed_roots,
        true,
    )
    .await
    .unwrap();

    assert!(!repo.join("blobs/v1blob").exists(), "v1 removed");
    assert!(!repo
        .join("snapshots/rev/zimage_turbo_training_adapter_v1.safetensors")
        .exists());
    assert!(repo.join("blobs/v2blob").exists(), "v2 blob retained");
    assert!(
        std::fs::read(repo.join("snapshots/rev/zimage_turbo_training_adapter_v2.safetensors"))
            .is_ok()
    );
}

/// The exclusive-repo case is UNCHANGED: with no sibling claiming the repo, a whole-model delete
/// still removes the repo cache wholesale, including files no `files` scope names.
///
/// This is the non-vacuity partner of the test above — without it, scoping could silently become
/// the universal path and quietly stop reclaiming the ~80 entries that own their repo outright.
#[tokio::test]
async fn whole_model_delete_on_an_exclusive_repo_still_removes_the_repo_cache() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let _cache_root = fixture_cache_root(data_dir);
    let hub = data_dir.join("cache/huggingface/hub");
    let repo_name = "Org/solo-model";
    let repo = huggingface_repo_cache_path(data_dir, repo_name).unwrap();
    assert_eq!(repo, hub.join("models--Org--solo-model"));
    seed(&repo, "q4/transformer/model.safetensors", "q4dit", 100);
    // A file NO declared scope names — only a blanket removal reaches it.
    seed(&repo, "README.md", "readme", 10);

    let model = json!({
        "id": "solo_model",
        "downloads": [{
            "provider": "huggingface",
            "repo": repo_name,
            "variant": "q4",
            "files": ["q4/transformer/*"],
        }],
    });
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

    remove_whole_model_artifacts(
        std::slice::from_ref(&model),
        "solo_model",
        &model,
        data_dir,
        &allowed_roots,
        true,
    )
    .await
    .unwrap();

    assert!(
        !repo.exists(),
        "an exclusively-owned repo cache is removed whole"
    );
}

/// A shared-repo entry that declares NO `files` scope keeps the blanket removal (`SceneWorks/bernini`
/// is the shipping example — both entries claim the whole repo with `files: []`). There is no
/// honest narrower scope for a whole-repo claim, so the documented behavior is preserved rather
/// than quietly reclaiming nothing.
#[tokio::test]
async fn whole_model_delete_keeps_the_blanket_path_for_an_unscoped_shared_claim() {
    let _env = isolate_hf_cache();
    let tmp = tempfile::tempdir().unwrap();
    let data_dir = tmp.path();
    let _cache_root = fixture_cache_root(data_dir);
    let hub = data_dir.join("cache/huggingface/hub");
    let repo_name = "SceneWorks/whole-repo-pair";
    let repo = huggingface_repo_cache_path(data_dir, repo_name).unwrap();
    assert_eq!(repo, hub.join("models--SceneWorks--whole-repo-pair"));
    seed(&repo, "model.safetensors", "dit", 100);

    let entry = |id: &str| {
        json!({
            "id": id,
            "downloads": [{ "provider": "huggingface", "repo": repo_name, "files": [] }],
        })
    };
    let first = entry("pair_a");
    let catalog = vec![first.clone(), entry("pair_b")];
    let allowed_roots = vec![data_dir.join("models"), huggingface_hub_cache_dir(data_dir)];

    remove_whole_model_artifacts(&catalog, "pair_a", &first, data_dir, &allowed_roots, true)
        .await
        .unwrap();

    assert!(!repo.exists());
}

#[test]
fn repo_file_scopes_union_tiers_and_reject_a_whole_repo_claim() {
    let model = json!({
        "id": "minimax_h3",
        "downloads": [
            { "repo": "SceneWorks/minimax-h3-mlx", "variant": "q4", "files": ["q4/transformer/*"] },
            { "repo": "SceneWorks/minimax-h3-mlx", "variant": "q8", "files": ["q8/transformer/*"] },
            { "repo": "MiniMaxAI/MiniMax-H3", "coRequisite": true, "files": ["vae/*"] },
        ],
    });
    // Only the named repo's rows, unioned across tiers — the co-requisite repo is a different repo
    // and contributes nothing to this repo's scope.
    assert_eq!(
        model_repo_file_scopes(&model, "SceneWorks/minimax-h3-mlx"),
        Some(vec![
            "q4/transformer/*".to_owned(),
            "q8/transformer/*".to_owned()
        ])
    );
    // A repo this entry does not claim at all has no scope.
    assert_eq!(model_repo_file_scopes(&model, "Org/unrelated"), None);
    // One unscoped row poisons the whole repo's scope: it is a claim on everything.
    let unscoped = json!({
        "id": "bernini",
        "downloads": [{ "repo": "SceneWorks/bernini", "files": [] }],
    });
    assert_eq!(
        model_repo_file_scopes(&unscoped, "SceneWorks/bernini"),
        None
    );
    // …and it poisons it even when a SCOPED sibling row is present in the same repo. This is the
    // case the empty-scopes fallback alone cannot express: without the early return the entry
    // would scope its delete to `q4/*` and strand everything else the unscoped row claims.
    let mixed = json!({
        "id": "mixed_claim",
        "downloads": [
            { "repo": "Org/mixed", "variant": "q4", "files": ["q4/*"] },
            { "repo": "Org/mixed", "files": [] },
        ],
    });
    assert_eq!(model_repo_file_scopes(&mixed, "Org/mixed"), None);

    // Sibling scopes exclude the entry itself and INCLUDE a sibling's co-requisite in that repo.
    let sibling = json!({
        "id": "minimax_h3_ref",
        "downloads": [
            { "repo": "SceneWorks/minimax-h3-mlx", "variant": "q4", "files": ["q4/transformer_ref/*"] },
            { "repo": "SceneWorks/minimax-h3-mlx", "coRequisite": true, "files": ["shared/*"] },
        ],
    });
    let catalog = vec![model.clone(), sibling];
    assert_eq!(
        other_entries_repo_file_scopes(&catalog, "minimax_h3", "SceneWorks/minimax-h3-mlx"),
        SiblingRepoScopes {
            primaries: vec!["q4/transformer_ref/*".to_owned()],
            co_requisites: vec!["shared/*".to_owned()],
        },
        "the two kinds must stay SEPARATE — only the co-requisite half may be subtracted from"
    );
    // Viewed from the sibling, the base entry's scopes are the ones retained.
    assert_eq!(
        other_entries_repo_file_scopes(&catalog, "minimax_h3_ref", "SceneWorks/minimax-h3-mlx"),
        SiblingRepoScopes {
            primaries: vec!["q4/transformer/*".to_owned(), "q8/transformer/*".to_owned()],
            co_requisites: Vec::new(),
        }
    );
    // A repo only this entry claims has no sibling scopes at all — the discriminator that keeps
    // the blanket path in force for the entries that own their repo outright.
    assert!(
        other_entries_repo_file_scopes(&catalog, "minimax_h3", "MiniMaxAI/MiniMax-H3").is_empty()
    );
}

/// The retained set may subtract the deleted entry's own scopes from the sibling's CO-REQUISITE
/// half only (sc-19573 review). Subtracting from the union destroys the sibling's weights.
#[test]
fn retained_files_never_subtracts_a_siblings_primary_scopes() {
    // The flux_dev ↔ pulid_flux_dev shape, and five more groups like it: the sibling names the
    // IDENTICAL primary `files`, because both entries really do load the same checkpoint. Every
    // one of those scopes must survive. Subtracting from the union yields `[]`, and then
    // `remove_tier_artifacts`'s `selected && !retained` unlinks the sibling's blobs with
    // `permanent=true` — tens of GB, unrecoverable without re-download.
    let own = vec!["q4/*".to_owned(), "q8/*".to_owned(), "bf16/*".to_owned()];
    let shared = SiblingRepoScopes {
        primaries: own.clone(),
        co_requisites: Vec::new(),
    };
    assert_eq!(
        shared.retained_files(&own),
        own,
        "an identically-scoped sibling primary must be retained IN FULL, not emptied"
    );

    // The anima trio: the sibling's primary is its own DiT, and the TE/VAE it shares with the
    // deleted entry ride the deleted entry's own primary rows too. Both must be retained.
    let anima_own = vec![
        "split_files/diffusion_models/anima-base-v1.0.safetensors".to_owned(),
        "split_files/text_encoders/qwen_3_06b_base.safetensors".to_owned(),
        "split_files/vae/qwen_image_vae.safetensors".to_owned(),
    ];
    let anima_sibling = SiblingRepoScopes {
        primaries: vec![
            "split_files/diffusion_models/anima-aesthetic-v1.0.safetensors".to_owned(),
            "split_files/text_encoders/qwen_3_06b_base.safetensors".to_owned(),
            "split_files/vae/qwen_image_vae.safetensors".to_owned(),
        ],
        co_requisites: Vec::new(),
    };
    assert!(
        anima_sibling
            .retained_files(&anima_own)
            .contains(&"split_files/text_encoders/qwen_3_06b_base.safetensors".to_owned()),
        "the shared text encoder anima_aesthetic/anima_turbo still need must be retained"
    );

    // MiniMax-H3, unchanged by the split: the sibling's PRIMARY is `transformer_ref`, and its
    // co-requisite claim on the deleted entry's own `transformer` is what gets subtracted, so
    // "delete minimax_h3" still frees the DiT the user asked to free.
    let mm_own = vec!["q4/transformer/*".to_owned()];
    let mm_sibling = SiblingRepoScopes {
        primaries: vec!["q4/transformer_ref/*".to_owned()],
        co_requisites: vec![
            "q4/transformer/*".to_owned(),
            "q4/text_encoder/*".to_owned(),
        ],
    };
    assert_eq!(
        mm_sibling.retained_files(&mm_own),
        vec![
            "q4/transformer_ref/*".to_owned(),
            "q4/text_encoder/*".to_owned()
        ],
        "the overlapping co-requisite is dropped; the sibling's primary and the shared TE stay"
    );
}

// Convert-at-install (Anima) tiers are real `<converted>/<tier>/` dirs with a packed DiT plus
// SYMLINKS to a shared TE/VAE source that lives outside the tier dirs (sc-12025).
fn seed_convert_tier(converted: &FsPath, tier: &str, dit_bytes: usize, shared_te: &FsPath) {
    let tier_dir = converted.join(tier);
    let dm = tier_dir.join("diffusion_models");
    std::fs::create_dir_all(&dm).unwrap();
    std::fs::write(dm.join("dit.safetensors"), vec![0u8; dit_bytes]).unwrap();
    let te = tier_dir.join("text_encoders");
    std::fs::create_dir_all(&te).unwrap();
    symlink(shared_te, te.join("te.safetensors")).unwrap();
}

#[tokio::test]
async fn removes_a_convert_tier_counting_only_real_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let converted = models.join("mlx").join("anima_base");
    std::fs::create_dir_all(&converted).unwrap();
    std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
    // The shared TE source lives OUTSIDE the tier dirs — deleting a tier must never free it.
    let shared = models.join("source").join("te.safetensors");
    std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
    std::fs::write(&shared, vec![0u8; 999]).unwrap();
    seed_convert_tier(&converted, "q4", 300, &shared);
    seed_convert_tier(&converted, "q8", 500, &shared);

    let removal = remove_converted_tier(converted.join("q4"), std::slice::from_ref(&models), true)
        .await
        .unwrap();

    assert!(!converted.join("q4").exists());
    assert!(converted
        .join("q8/diffusion_models/dit.safetensors")
        .exists());
    // The shared TE source and the converted marker both survive (q8 still installed).
    assert!(shared.exists());
    assert!(converted.join("model_index.json").exists());
    // Only q4's real DiT bytes count — the symlinked shared TE is not reclaimed.
    assert_eq!(removal.reclaimed_bytes, 300);
}

#[tokio::test]
async fn draining_the_last_convert_tier_removes_the_converted_dir() {
    let tmp = tempfile::tempdir().unwrap();
    let models = tmp.path().join("models");
    let converted = models.join("mlx").join("anima_base");
    std::fs::create_dir_all(&converted).unwrap();
    std::fs::write(converted.join("model_index.json"), b"{}").unwrap();
    let shared = models.join("source").join("te.safetensors");
    std::fs::create_dir_all(shared.parent().unwrap()).unwrap();
    std::fs::write(&shared, vec![0u8; 999]).unwrap();
    seed_convert_tier(&converted, "q4", 300, &shared);

    let removal = remove_converted_tier(converted.join("q4"), std::slice::from_ref(&models), true)
        .await
        .unwrap();

    // No tier remains → the whole converted dir (marker included) is gone; the shared source is
    // NOT (it belongs to the download, not the convert output).
    assert!(!converted.exists());
    assert!(shared.exists());
    assert_eq!(removal.reclaimed_bytes, 300);
}
