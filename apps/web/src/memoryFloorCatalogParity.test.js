import {
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  truncateSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import {
  CANDLE_HEADROOM_GB,
  DISK_TO_RESIDENT_MULTIPLIER,
  MEMORY_HEADROOM_FRACTION,
  MEASURED_SIBLING_RATIO_TOLERANCE,
  TRANSIENT_HEADROOM_BYTES,
  TRANSIENT_HEADROOM_MEASURED_PIXELS,
  blanketFloorGb,
  cheapestDeclaredTierPeakGb,
  hostGbForPeakGb,
  installedFloorHostGb,
  installedTierPeakGb,
  suggestTier,
  tierFits,
  variantFootprintBytes,
} from "./tierSuggestion.js";
import { isSelectableTier, tierHostEligible, tierQuantize } from "./quantTier.js";
import { needsLabel } from "./simple/SimpleModelManager.jsx";

// CATALOG-DRIVEN memory-floor guard (sc-15400 review). Sibling of controlModesParity.test.js /
// videoGeometryParity.test.js, which established the pattern of asserting against the REAL
// `config/manifests/builtin.models.jsonc` rather than a synthetic literal.
//
// WHY THIS FILE EXISTS. sc-15400 shipped two defects behind a fully green suite of 14 tests, because
// every one of them used a hand-written model literal:
//
//   1. The row displayed the RAW measured peak (`Math.ceil(peak)`), which does not satisfy the app's own
//      `peak <= host * MEMORY_HEADROOM_FRACTION` fit criterion. lens_turbo's 30.50 GiB q4 read
//      "needs 31 GB", and 31 x 0.9 = 27.9 GiB does not fit it. Four shipped models under-stated.
//   2. The MLX-measured footprint was read on the candle/CUDA lane. lens_turbo's q4 download serves
//      windows/linux and carries an MLX footprint, so a CUDA host read "needs 31 GB" against a measured
//      37.3 GB VRAM requirement — converting a pre-existing over-warn into an UNDER-warn.
//
// A synthetic literal cannot catch either, because the author picks both the input and the expected
// output. So every assertion below derives its expectation from the manifest and from the app's own
// constants — never from a transcribed number — and the suite fails if it ever covers zero models.
const HERE = dirname(fileURLToPath(import.meta.url));
const MANIFEST_PATH = resolve(HERE, "../../../config/manifests/builtin.models.jsonc");
const BYTES_PER_GB = 1024 ** 3;

// apps/rust-api/src/models.rs:4603 `is_supported_model_download`.
function isSupportedModelDownload(download) {
  return download?.provider === "huggingface" && typeof download?.repo === "string" && download.repo !== "";
}

// apps/rust-api/src/lib.rs:3080 `json_size_to_u64` — a non-negative integer, or a decimal STRING of one.
function jsonSizeToU64(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) && value >= 0 ? value : null;
  }
  if (typeof value === "string" && /^\d+$/.test(value.trim())) {
    return Number(value.trim());
  }
  return null;
}

// apps/rust-api/src/models.rs:5060 `manifest_download_size_bytes` — these three keys IN THIS ORDER off the
// selected download entry, then the same three off the model as a legacy fallback.
const DOWNLOAD_SIZE_KEYS = ["estimatedSizeBytes", "downloadSizeBytes", "sizeBytes"];
function firstDeclaredSize(source) {
  for (const key of DOWNLOAD_SIZE_KEYS) {
    const size = jsonSizeToU64(source?.[key]);
    if (size !== null) {
      return size;
    }
  }
  return null;
}

// The `variants[].downloadSizeBytes` the catalog emits, per models.rs:3489 —
// `manifest_download_size_bytes(model, entry).or_else(|| variant_footprint_disk_bytes(entry))` — and
// re-asserted verbatim by `refresh_variant_download_sizes` (models.rs:3595) after live HF estimation.
// The trailing `footprint.diskSizeBytes` leg cannot change any answer here (`variantFootprintBytes`
// already reads that field at its priority 3, AHEAD of `downloadSizeBytes` at priority 4), but the
// catalog emits it, so the mirror does too rather than a shape the client never receives.
function catalogDownloadSizeBytes(manifestModel, download) {
  return (
    firstDeclaredSize(download) ??
    firstDeclaredSize(manifestModel) ??
    jsonSizeToU64(download?.footprint?.diskSizeBytes)
  );
}

function loadManifestModels() {
  const parsed = JSON5.parse(readFileSync(MANIFEST_PATH, "utf8"));
  const models = Array.isArray(parsed) ? parsed : parsed.models;
  expect(Array.isArray(models), "manifest must expose a models array").toBe(true);
  return models;
}

// Project a manifest entry into the CATALOG shape the web client actually receives, for the memory
// fields only. Mirrors apps/rust-api/src/models.rs:
//   * `retain_downloads_for_os` — per-OS `platforms` filtering, and its early return when NO entry
//     declares `platforms` (in which case nothing is filtered).
//   * `is_co_requisite_download` — `coRequisite: true` rows are dependencies, never selectable tiers.
//   * `is_supported_model_download` (models.rs:4603) — `provider === "huggingface"` AND a non-empty
//     `repo`. `model_variant_downloads` applies this in the SAME filter as the co-requisite check, so a
//     row from another provider is not a selectable tier either.
//   * `model_variant_downloads` — duplicate `variant` keys collapse FIRST-wins.
//   * `apply_variant_fields` — emits `variants[]` carrying the raw `footprint` AND `downloadSizeBytes`
//     (`catalogDownloadSizeBytes` above), plus `hasVariantMatrix`.
// The rest of the entry passes through unchanged (the catalog decorates the manifest object in place),
// which is what makes `mlx` / `candle` — including `candle.vramGbByTier` — reach the client at all.
//
// AN HONEST LIMIT ON WHAT THIS MIRROR CAN CATCH — (A) DRIFT. It is a JS re-implementation, so it pins the
// MANIFEST against the mirror, NOT the mirror against the Rust. If someone changes
// `retain_downloads_for_os`, `is_co_requisite_download` or `is_supported_model_download` in models.rs,
// nothing here goes red — the two just silently disagree, and this file would then be asserting about a
// catalog shape the client never receives. The provider/repo filter above is a case in point: it was
// missing from the first version of this mirror and drifted zero rows only because every shipped download
// happens to be a huggingface entry with a repo. Cross-language enforcement would need the Rust to emit
// the projection (or a shared fixture); that is out of scope here and deliberately not claimed.
//
// (B) FIELDS AND SHAPES THE MIRROR DOES NOT MODEL. These are not drift risks, they are areas the sweep
// below simply cannot see. This file is this PR's central evidence, so its blind spots are written down
// next to it rather than left to be rediscovered:
//   * `downloadSizeBytes` WAS one of them, and its absence hid a real hole. `variantFootprintBytes`
//     consumes it as its LAST-RESORT priority 4 (tierSuggestion.js:92-98, reached via
//     `mlxEstimatedPeakGb`), and `krea_2_raw`/`krea_2_turbo` — the only two shipped matrix entries whose
//     downloads carry NO `footprint` at all — reach the fill through exactly that leg. Without the field
//     the mirror answered `needs 48 GB` (the `mlx.minMemoryGb` blanket) for all eight bf16-containing
//     install sets across the two models, where production answers 53; and because those tiers were
//     `variantFootprintBytes(...) === null` in the mirror they were `continue`d out of the invariant
//     sweep, so nothing asserted about them even though `tierFits` REFUSES bf16 at 48. It is modelled
//     now, and the sweep bounds those tiers like any other.
//   * `mlxTierStates` / `mlxTiers` — the convert-at-install tier shapes. `installedTiers`
//     (quantTier.js:196, :208) has a branch for each, and the general mirror emits neither, so BOTH are
//     unreachable from the sweep; every sweep row below goes through the `hasVariantMatrix` branch.
//     SC-16045 added runtime `mlxTierStates[].diskSizeBytes`; the targeted real-manifest + real-file
//     assertion below covers that path, while the broad combinatorial sweep deliberately stays manifest-only.
//   * SINGLE-VARIANT ENTRIES ARE DROPPED ENTIRELY. `tiersOn` returns [] for a model that projects to just
//     the `"default"` pseudo-variant, and the sweep `continue`s on that — 33 of the 86 manifest entries on
//     macOS, 37 of 84 on windows. For those entries `needsLabel`'s blanket fallthrough is the ONLY branch
//     reachable (no matrix ⇒ no installed tiers and no declared tiers), and none of them is swept. The
//     fallthrough itself is not uncovered — 94 sweep rows across 24 (model, os) pairs reach it, via matrix
//     models whose whole install set is gated out by host-eligibility or which declare no per-tier
//     evidence — but it is covered only in that flavour.
//   * `cacheState` / `installedPath` / `missingRequiredFiles` (models.rs:3566-3579) — the mirror emits a
//     binary installed/missing. A torn or incomplete install is therefore indistinguishable from a clean
//     one here, and any future rule that consults completeness would be invisible to this sweep.
//   * `catalogScope`, AND WITH IT EVERY NON-BUILTIN ENTRY. The catalog stamps every row
//     `catalogScope: "builtin" | "user"` (models.rs:3966), plus `"external"` for scanned externally-owned
//     bases (external_base_models.rs); no manifest entry declares the field, and the mirror emits none of
//     it. This projects `config/manifests/builtin.models.jsonc` only, so every claim below is about
//     BUILTIN entries — a user-scope or external model reaches the catalog by another route entirely and
//     is outside what this file can say anything about.
//
// `installedTiers` is the set to mark installed, so a caller can ask "what would this host be told if
// exactly these tiers were on disk".
function catalogEntry(manifestModel, os, installedTierKeys) {
  let downloads = manifestModel.downloads ?? [];
  if (downloads.some((entry) => entry.platforms !== undefined)) {
    downloads = downloads.filter(
      (entry) => !Array.isArray(entry.platforms) || entry.platforms.includes(os),
    );
  }
  const selectable = downloads.filter(
    (entry) => entry.coRequisite !== true && isSupportedModelDownload(entry),
  );
  const seen = new Set();
  const variants = [];
  for (const entry of selectable) {
    const key =
      typeof entry.variant === "string" && entry.variant.trim() ? entry.variant.trim() : "default";
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    const installed = installedTierKeys.includes(key);
    variants.push({
      variant: key,
      installed,
      installState: installed ? "installed" : "missing",
      downloadSizeBytes: catalogDownloadSizeBytes(manifestModel, entry),
      footprint: entry.footprint ?? null,
    });
  }
  return {
    ...manifestModel,
    hasVariantMatrix: selectable.some(
      (entry) => typeof entry.variant === "string" && entry.variant.trim() !== "",
    ),
    variants,
  };
}

// Every SELECTABLE tier a manifest entry offers on `os`, after the same filters — the same vocabulary
// `installedTiers` uses, which is wider than bf16/q8/q4: `krea_2_turbo` also offers the candle-only
// `int8-convrot`, and excluding it here would hide the install-set that exposed the blanket-fallback
// under-statement (32 shown against that tier's measured 34.7).
function tiersOn(manifestModel, os) {
  const entry = catalogEntry(manifestModel, os, []);
  return entry.variants.map((v) => v.variant).filter((key) => isSelectableTier(key));
}

// The MLX measured peak (GB) the manifest declares for a tier on `os`, or null.
function mlxPeakGb(manifestModel, os, tier) {
  const entry = catalogEntry(manifestModel, os, [tier]);
  const bytes = entry.variants.find((v) => v.variant === tier)?.footprint?.peakMemoryBytes;
  return typeof bytes === "number" && bytes > 0 ? bytes / BYTES_PER_GB : null;
}

const manifestModels = loadManifestModels();
const OSES = ["macos", "windows"];
// The lane each OS runs, matching every consumer's `macGatingActive ? "mlx" : "candle"`.
const BACKEND_FOR_OS = { macos: "mlx", windows: "candle" };
const MAX_MEASURED_SIBLING_RATIO_TOLERANCE = 1.05;

function estimateDiskBytes(variant) {
  return jsonSizeToU64(variant?.footprint?.diskSizeBytes) ?? jsonSizeToU64(variant?.downloadSizeBytes);
}

function measuredResidentForRatio(variant) {
  const resident = jsonSizeToU64(variant?.footprint?.residentMemoryBytes);
  if (resident !== null && resident > 0) {
    return resident;
  }
  const peak = jsonSizeToU64(variant?.footprint?.peakMemoryBytes);
  return peak !== null &&
    peak > TRANSIENT_HEADROOM_BYTES &&
    variant?.footprint?.measuredPixels === TRANSIENT_HEADROOM_MEASURED_PIXELS
    ? peak - TRANSIENT_HEADROOM_BYTES
    : null;
}

describe("catalog memory estimates: measured-sibling upper bound (sc-16046)", () => {
  it("bounds every estimable MLX tier with a measured sibling and audits the affected model shapes", () => {
    // The guard owns this ceiling independently of the implementation constant. Raising the production
    // tolerance therefore makes this test red instead of raising its own upper bound in lockstep.
    expect(MEASURED_SIBLING_RATIO_TOLERANCE).toBeLessThanOrEqual(
      MAX_MEASURED_SIBLING_RATIO_TOLERANCE,
    );
    const rows = [];
    for (const manifestModel of manifestModels) {
      const entry = catalogEntry(manifestModel, "macos", []);
      for (const target of entry.variants.filter((variant) => tierQuantize(variant.variant) !== null)) {
        if (measuredResidentForRatio(target) !== null) {
          continue;
        }
        const targetDisk = estimateDiskBytes(target);
        if (targetDisk === null || targetDisk <= 0) {
          continue;
        }
        const siblingRatios = entry.variants
          .filter(
            (sibling) =>
              sibling.variant !== target.variant && tierQuantize(sibling.variant) !== null,
          )
          .map((sibling) => {
            const disk = estimateDiskBytes(sibling);
            const resident = measuredResidentForRatio(sibling);
            return disk !== null && disk > 0 && resident !== null ? resident / disk : null;
          })
          .filter((ratio) => ratio !== null && ratio > 0 && Number.isFinite(ratio));
        if (siblingRatios.length === 0) {
          continue;
        }

        const siblingRatio = Math.max(...siblingRatios);
        const toleratedRatio = Math.min(
          DISK_TO_RESIDENT_MULTIPLIER,
          siblingRatio * MAX_MEASURED_SIBLING_RATIO_TOLERANCE,
        );
        const upperBound = Math.round(targetDisk * toleratedRatio) + TRANSIENT_HEADROOM_BYTES;
        const estimate = variantFootprintBytes(target, entry);
        const optedIn = entry?.mlx?.estimateResidentFromMeasuredSibling === true;
        const denseBytes =
          Math.round(targetDisk * DISK_TO_RESIDENT_MULTIPLIER) + TRANSIENT_HEADROOM_BYTES;

        expect(estimate, `${entry.id}/${target.variant} must remain estimable`).not.toBeNull();
        expect(estimate.measured, `${entry.id}/${target.variant} is still an estimate`).toBe(false);
        if (optedIn) {
          expect(
            estimate.bytes,
            `${entry.id}/${target.variant} exceeds its measured-sibling upper bound`,
          ).toBeLessThanOrEqual(upperBound);
        } else {
          expect(estimate.bytes, `${entry.id}/${target.variant} lacks compatible-topology opt-in`).toBe(
            denseBytes,
          );
        }
        rows.push({
          id: entry.id,
          tier: target.variant,
          estimate,
          denseBytes,
          optedIn,
        });
      }
    }

    // Pin the complete current corpus so a newly-measured matrix model forces an explicit audit instead
    // of silently entering (or evading) this upper-bound contract. Lens carries a packed MoE text encoder;
    // the two Wan A14B entries are the catalog's expert-swapped dual-transformer shapes. The dense Z-Image
    // and SDXL siblings remain at the 1.0 cap, proving the optimization is evidence-driven rather than a
    // family-name special case.
    expect([...new Set(rows.map((row) => row.id))].sort()).toEqual(
      [
        "lens_turbo",
        "sdxl",
        "wan_2_2_i2v_14b",
        "wan_2_2_t2v_14b",
        "z_image",
        "z_image_edit",
        "z_image_turbo",
      ].sort(),
    );
    expect(
      [...new Set(rows.filter((row) => row.estimate.bytes < row.denseBytes).map((row) => row.id))].sort(),
    ).toEqual(["wan_2_2_i2v_14b", "wan_2_2_t2v_14b"].sort());
    expect([...new Set(rows.filter((row) => row.optedIn).map((row) => row.id))].sort()).toEqual(
      ["wan_2_2_i2v_14b", "wan_2_2_t2v_14b"].sort(),
    );

    const wanBf16 = rows.filter(
      (row) => row.tier === "bf16" && row.id.startsWith("wan_2_2_"),
    );
    expect(wanBf16).toHaveLength(2);
    for (const row of wanBf16) {
      expect(hostGbForPeakGb(row.estimate.bytes / BYTES_PER_GB, "mlx"), row.id).toBe(46);
      expect(hostGbForPeakGb(row.denseBytes / BYTES_PER_GB, "mlx"), row.id).toBeGreaterThanOrEqual(
        87,
      );
    }
  });

  it("audits every current MoE-bearing manifest entry, including shapes with no measured sibling", () => {
    // Source audit of builtin.models.jsonc's MoE/dual-expert declarations. Lens has no measured tier;
    // Lens Turbo changes the MoE encoder's packing between tiers; Bernini has no measured sibling;
    // VACE-Fun is one default artifact. All retain the dense/no-sibling behavior. Only the two WAN A14B
    // matrices explicitly assert tier-compatible one-expert-at-a-time topology.
    const ids = [
      "bernini",
      "bernini_image",
      "lens",
      "lens_turbo",
      "wan_2_2_t2v_14b",
      "wan_2_2_i2v_14b",
      "wan_2_2_vace_fun_14b",
    ];
    const entries = new Map(
      ids.map((id) => {
        const model = manifestModels.find((candidate) => candidate.id === id);
        expect(model, `${id} must remain in the manifest`).toBeTruthy();
        return [id, catalogEntry(model, "macos", [])];
      }),
    );
    const relativeResidentScale = (entry, tier) => {
      const variant = entry.variants.find((candidate) => candidate.variant === tier);
      const disk = estimateDiskBytes(variant);
      const estimate = variantFootprintBytes(variant, entry);
      expect(disk, `${entry.id}/${tier} disk bytes`).toBeGreaterThan(0);
      expect(estimate, `${entry.id}/${tier} estimate`).not.toBeNull();
      return (estimate.bytes - TRANSIENT_HEADROOM_BYTES) / disk;
    };

    for (const tier of ["q4", "q8", "bf16"]) {
      for (const id of ["bernini", "bernini_image", "lens"]) {
        expect(relativeResidentScale(entries.get(id), tier), `${id}/${tier}`).toBe(1);
      }
    }
    for (const tier of ["q8", "bf16"]) {
      expect(relativeResidentScale(entries.get("lens_turbo"), tier), `lens_turbo/${tier}`).toBe(1);
      for (const id of ["wan_2_2_t2v_14b", "wan_2_2_i2v_14b"]) {
        expect(relativeResidentScale(entries.get(id), tier), `${id}/${tier}`).toBeLessThan(0.5);
      }
    }
    expect(
      manifestModels
        .filter((model) => model?.mlx?.estimateResidentFromMeasuredSibling === true)
        .map((model) => model.id)
        .sort(),
    ).toEqual(["wan_2_2_i2v_14b", "wan_2_2_t2v_14b"].sort());
    expect(
      entries
        .get("wan_2_2_vace_fun_14b")
        .variants.filter((variant) => tierQuantize(variant.variant) !== null),
    ).toEqual([]);
  });
});

describe("convert-at-install memory floors", () => {
  it("derives anima_base's bf16-only label from real catalog inputs instead of its blanket", () => {
    const anima = manifestModels.find((model) => model.id === "anima_base");
    expect(anima, "anima_base must still be in the real manifest").toBeTruthy();
    const blanket = blanketFloorGb(anima, "mlx");
    expect(blanket).toBeGreaterThan(0);

    // The Rust emitter independently tests the same contract from a converted directory. Here the
    // file's stat supplies runtime data instead of transcribing a model-size literal into the test.
    const converted = mkdtempSync(join(tmpdir(), "sc-16045-anima-"));
    const bf16Weights = join(converted, "anima-bf16.safetensors");
    try {
      writeFileSync(bf16Weights, "");
      truncateSync(bf16Weights, 2 * BYTES_PER_GB);
      const diskSizeBytes = statSync(bf16Weights).size;
      const entry = {
        ...anima,
        installState: "installed",
        hasVariantMatrix: false,
        variants: [],
        mlxTierStates: [
          {
            tier: "bf16",
            installState: "installed",
            cacheState: "complete",
            diskSizeBytes,
          },
        ],
      };
      const shown = shownGb(
        needsLabel(entry, {
          backend: "mlx",
          convRotEligible: true,
          nvfp4Eligible: true,
        }),
      );
      const estimatedPeak =
        variantFootprintBytes({ footprint: { diskSizeBytes } }).bytes / BYTES_PER_GB;
      expect(shown).toBe(hostGbForPeakGb(estimatedPeak, "mlx"));
      expect(shown).toBeGreaterThan(blanket);
    } finally {
      rmSync(converted, { recursive: true, force: true });
    }
  });
});

describe("catalog memory floors: the displayed number satisfies the app's own fit criterion", () => {
  // Every (model, tier) on macOS that carries a real measured MLX footprint. This is the set whose
  // numbers reach a user as "needs N GB", so every one of them must satisfy `tierFits`.
  const measuredMlx = [];
  for (const model of manifestModels) {
    for (const tier of tiersOn(model, "macos")) {
      const peak = mlxPeakGb(model, "macos", tier);
      if (peak !== null) {
        measuredMlx.push({ id: model.id, tier, peak, model });
      }
    }
  }

  it("covers a non-trivial set of really-measured tiers", () => {
    // Guards the whole describe against silently asserting nothing if the manifest shape moves.
    expect(measuredMlx.length).toBeGreaterThanOrEqual(8);
  });

  it.each(measuredMlx.map((row) => [`${row.id} ${row.tier}`, row]))(
    "%s: the derived host size fits the measured peak, and one GB less does not",
    (_label, row) => {
      const shown = hostGbForPeakGb(row.peak);
      expect(shown).toBeGreaterThan(0);
      // The app's OWN criterion, not a restatement of it: the peak must fit under the headroom budget.
      expect(row.peak, `${_label}: ${shown} GB must fit a ${row.peak.toFixed(2)} GiB peak`).toBeLessThanOrEqual(
        shown * MEMORY_HEADROOM_FRACTION,
      );
      // TIGHT: the displayed figure is the MINIMUM honest host, so one GB below must fail. This is the
      // assertion the raw-peak bug fails — ceil(peak) is always below peak/0.9 for a peak above ~0.
      expect(row.peak, `${_label}: ${shown - 1} GB must NOT fit`).toBeGreaterThan(
        (shown - 1) * MEMORY_HEADROOM_FRACTION,
      );
    },
  );

  it.each(measuredMlx.map((row) => [`${row.id} ${row.tier}`, row]))(
    "%s: agrees with tierFits, the gate the download suggestion already uses",
    (_label, row) => {
      const entry = catalogEntry(row.model, "macos", [row.tier]);
      const variant = entry.variants.find((v) => v.variant === row.tier);
      const shown = installedTierPeakGb(entry, { backend: "mlx" });
      expect(shown).not.toBeNull();
      const hostGb = hostGbForPeakGb(shown);
      expect(tierFits(variant, hostGb), `${_label}: tierFits must accept ${hostGb} GB`).toBe(true);
      expect(tierFits(variant, hostGb - 1), `${_label}: tierFits must reject ${hostGb - 1} GB`).toBe(
        false,
      );
    },
  );

  // The reported figure must never sit BELOW the lane's own measured requirement — the one direction
  // that causes the reported failure mode (a user downloads a model that then OOMs).
  //
  // ROUND 4 REMOVED THE BLIND SPOT THAT LET THE ROUND-3 BLOCKER SHIP GREEN THROUGH THIS VERY TEST. It read
  // `installedTierPeakGb` — the STRICT primitive, which returns null whenever any installed tier is
  // unevidenced — and then `if (peak === null) continue`. So a partially-evidenced install-set asserted
  // NOTHING here, and that is precisely the shape `z_image_edit` has on macOS. The test written to catch the
  // class could not see the instance.
  //
  // It now asserts on `installedFloorHostGb`, the COMPOSER the label actually calls, so no model is skipped:
  // every (model, os) with at least one selectable tier contributes an assertion, and the strict primitive is
  // checked as an ADDITIONAL constraint wherever it is non-null rather than as the gate for looking at all.
  it("never reports a host size below the lane's measured peak, on either lane", () => {
    let checked = 0;
    let skipped = 0;
    const silentWithEvidence = [];
    for (const os of OSES) {
      const backend = BACKEND_FOR_OS[os];
      for (const model of manifestModels) {
        const tiers = tiersOn(model, os);
        if (tiers.length === 0) {
          continue;
        }
        // The "everything installed" host — the widest set, so the ceiling is over every tier.
        const entry = catalogEntry(model, os, tiers);
        const options = { backend, convRotEligible: true, nvfp4Eligible: true };
        const shown = installedFloorHostGb(entry, options);
        const evidenced = tiers.filter((tier) => laneEvidenceGb(model, os, tier) !== null);
        if (shown === null) {
          // Silence is only ever correct when the lane has NO per-tier evidence and NO blanket. Anything
          // else here would be the round-3 defect in its other direction.
          skipped++;
          if (evidenced.length > 0 || blanketFloorGb(model, backend) !== null) {
            silentWithEvidence.push(`${model.id} on ${os}`);
          }
          continue;
        }
        checked++;
        // No single tier's own lane evidence may exceed the reported figure, judged by THAT LANE's own
        // criterion (MLX multiplies by 0.9, candle adds 2 — see `hostGbForPeakGb`).
        for (const tier of tiers) {
          const lanePeak = laneEvidenceGb(model, os, tier);
          if (lanePeak !== null) {
            expect(
              hostGbForPeakGb(lanePeak, backend),
              `${model.id} ${tier} on ${os} must not exceed the reported ceiling`,
            ).toBeLessThanOrEqual(shown);
          }
        }
        // Where the STRICT primitive can also speak, the composed answer must not sit below it.
        const strict = installedTierPeakGb(entry, options);
        if (strict !== null) {
          expect(shown, `${model.id} on ${os} vs the strict ceiling`).toBeGreaterThanOrEqual(
            Math.ceil(strict),
          );
        }
      }
    }
    expect(silentWithEvidence.join(", ")).toBe("");
    expect(checked, "must actually check some models on both lanes").toBeGreaterThanOrEqual(8);
    // Non-vacuity in the other direction: the skipped set is the genuinely-evidenceless one, and it must
    // not have quietly grown to swallow the corpus.
    expect(skipped).toBeLessThan(checked);
  });
});

// The lane's OWN per-tier evidence in GB, read straight off the manifest rather than through the module
// under test, so the assertions below are an independent check rather than a restatement.
//   * MLX     — `downloads[].footprint.peakMemoryBytes` (the footprint_measure.rs harness).
//   * candle  — `candle.vramGbByTier[tier]`.
// A candle-only tier with no row of its own returns null: the module degrades it to the q8 row
// (`vram_gate`'s sc-11042 rule) but there is no independent evidence here to assert against.
function laneEvidenceGb(manifestModel, os, tier) {
  if (BACKEND_FOR_OS[os] === "candle") {
    const value = (manifestModel.candle?.vramGbByTier ?? {})[tier];
    return typeof value === "number" && value > 0 ? value : null;
  }
  return mlxPeakGb(manifestModel, os, tier);
}

// Every non-empty subset of `list`, plus the empty one (the not-installed case).
function subsetsOf(list) {
  let out = [[]];
  for (const item of list) {
    out = [...out, ...out.map((subset) => [...subset, item])];
  }
  return out;
}

// Every (model, OS, installed-subset) the real catalog can produce, with the COMPOSED label for it.
// `installState` mirrors the catalog: a model with at least one installed tier reads installed, which is
// what selects `needsLabel`'s case 1 vs its "from" branch.
const LABEL_CASES = [];
for (const os of OSES) {
  const backend = BACKEND_FOR_OS[os];
  for (const model of manifestModels) {
    const tiers = tiersOn(model, os);
    if (tiers.length === 0) {
      continue;
    }
    for (const subset of subsetsOf(tiers)) {
      const entry = {
        ...catalogEntry(model, os, subset),
        installState: subset.length > 0 ? "installed" : "missing",
      };
      // Both candle-only tiers eligible: the widest install set a real host can be offered, and the one
      // where an under-statement would actually reach a user.
      const options = { backend, convRotEligible: true, nvfp4Eligible: true };
      LABEL_CASES.push({
        id: model.id,
        os,
        backend,
        subset,
        tiers,
        model,
        label: needsLabel(entry, options),
      });
    }
  }
}

// Pull the integer out of "needs N GB" / "from N GB". null when the row shows no number at all.
function shownGb(label) {
  const match = /(?:needs|from) (\d+) GB/.exec(label ?? "");
  return match ? Number(match[1]) : null;
}

describe("catalog memory floors: the COMPOSED label never under-states an installed tier", () => {
  // WHY THIS IS THE LOAD-BEARING TEST. Everything above asserts a HELPER against the manifest, and the
  // `SimpleModelManager` tests assert the label against hand-written literals. Neither can see a defect
  // in how the three cases COMPOSE — and that is where the previous round's under-statement lived: with
  // any installed tier unmeasured, `needsLabel` fell through to `blanketFloorGb` alone, and the blanket
  // is the DEFAULT (lightest) tier's peak, which heavier tiers are explicitly allowed to exceed (schema
  // `candle.minMemoryGb`; vram_gate.rs:180 "minMemoryGb is the WRONG floor"). Four real install-sets
  // read below their own measured evidence: `flux_dev` and `flux_schnell` showed 24 against a measured
  // q8 of 31.8, and `krea_2_turbo` showed 32 against a measured bf16 of 47.2 (and 32 against 34.7 with
  // only `int8-convrot` installed).
  //
  // So this asserts over the LABEL, for every install-subset the catalog can produce, against each
  // tier's own lane evidence and that lane's own criterion. No expected number is transcribed.
  it("covers every install-subset of every matrix model, on both lanes", () => {
    expect(LABEL_CASES.length).toBeGreaterThanOrEqual(300);
    // Both lanes really are exercised, and the biggest install sets really are enumerated.
    expect(LABEL_CASES.some((row) => row.backend === "candle")).toBe(true);
    expect(LABEL_CASES.some((row) => row.backend === "mlx")).toBe(true);
    expect(Math.max(...LABEL_CASES.map((row) => row.subset.length))).toBeGreaterThanOrEqual(4);
  });

  it("shows a host size that satisfies EVERY installed tier's lane criterion", () => {
    const failures = [];
    let asserted = 0;
    for (const row of LABEL_CASES) {
      if (row.subset.length === 0) {
        continue;
      }
      const shown = shownGb(row.label);
      for (const tier of row.subset) {
        const peak = laneEvidenceGb(row.model, row.os, tier);
        if (peak === null) {
          continue;
        }
        asserted++;
        // The host size THAT LANE's gate requires for this tier's evidence. The label must be at least
        // this, or a user who installed exactly this subset is told a number their own gate rejects.
        const required = hostGbForPeakGb(peak, row.backend);
        if (shown === null || shown < required) {
          failures.push(
            `${row.id} on ${row.os} [${row.subset.join(",")}]: label ${JSON.stringify(row.label)} but ${tier} peak ${peak} needs a ${required} GB host`,
          );
        }
      }
    }
    expect(asserted, "must actually compare some real evidence").toBeGreaterThanOrEqual(100);
    expect(failures.join("\n")).toBe("");
  });

  it("never falls silent while the lane HAS evidence for an installed tier", () => {
    // The other way to under-state is to say nothing at all. Silence is correct only when the lane has
    // neither a per-tier row nor a blanket (z_image / sdxl / illustrious_xl_* on candle).
    const failures = [];
    for (const row of LABEL_CASES) {
      if (row.subset.length === 0 || shownGb(row.label) !== null) {
        continue;
      }
      for (const tier of row.subset) {
        if (laneEvidenceGb(row.model, row.os, tier) !== null) {
          failures.push(`${row.id} on ${row.os} [${row.subset.join(",")}] showed nothing`);
        }
      }
    }
    expect(failures.join("\n")).toBe("");
  });

  it("states the uninstalled 'from' floor at or above the cheapest evidenced tier", () => {
    let checked = 0;
    for (const row of LABEL_CASES) {
      if (row.subset.length > 0) {
        continue;
      }
      const evidence = row.tiers
        .map((tier) => laneEvidenceGb(row.model, row.os, tier))
        .filter((gb) => gb !== null);
      if (evidence.length === 0) {
        continue;
      }
      checked++;
      const shown = shownGb(row.label);
      expect(shown, `${row.id} on ${row.os} must quote an entry floor`).not.toBeNull();
      // A FLOOR, so it answers for the CHEAPEST tier — not for every tier (which is what "from" means).
      expect(shown, `${row.id} on ${row.os} 'from' floor`).toBeGreaterThanOrEqual(
        hostGbForPeakGb(Math.min(...evidence), row.backend),
      );
    }
    expect(checked, "must cover some uninstalled models with evidence").toBeGreaterThanOrEqual(8);
  });
});

describe("catalog memory floors: the candle figure matches the gate that actually rejects", () => {
  // MAJOR 3. The candle lane's criterion is ADDITIVE — `vram_gate.rs` admits a load at
  // `candle.vramGbByTier[tier] + HEADROOM_GB` (2.0) — while MLX's is multiplicative (`peak / 0.9`).
  // Applying the MLX form to candle OVER-states against the gate: `flux2_dev` bf16 read 143 where the
  // gate requires 130, and 11 of the 95 shipped rows were >= 5 GB high. That re-creates the very
  // over-warning this story exists to remove, on the lane it just started reading.
  //
  // Asserted as an EQUALITY on the composed label for a single-tier install, so it is tight in both
  // directions: too high (the fractional form) and too low both fail.
  const singleTierCandle = LABEL_CASES.filter(
    (row) =>
      row.backend === "candle" &&
      row.subset.length === 1 &&
      // Only where the whole install set is measured evidence the module trusts alone — an entry the
      // manifest flags `measured: false` is deliberately floored at its blanket instead (MAJOR 4).
      row.model.candle?.measured !== false &&
      laneEvidenceGb(row.model, row.os, row.subset[0]) !== null,
  );

  it("covers a non-trivial set of candle rows", () => {
    expect(singleTierCandle.length).toBeGreaterThanOrEqual(20);
    // ...and enough of them are big enough that the two conversions really do disagree, so the equality
    // below is a live constraint rather than an arithmetic coincidence on small numbers.
    const disagree = singleTierCandle.filter((row) => {
      const peak = laneEvidenceGb(row.model, row.os, row.subset[0]);
      return Math.ceil(peak / MEMORY_HEADROOM_FRACTION) !== Math.ceil(peak + CANDLE_HEADROOM_GB);
    });
    expect(disagree.length).toBeGreaterThanOrEqual(15);
  });

  it.each(singleTierCandle.map((row) => [`${row.id} ${row.subset[0]}`, row]))(
    "%s: the label is exactly the host the candle gate requires",
    (_label, row) => {
      const peak = laneEvidenceGb(row.model, row.os, row.subset[0]);
      const shown = shownGb(row.label);
      // vram_gate's own arithmetic, restated from the constant rather than from a transcribed number.
      expect(shown).toBe(Math.ceil(peak + CANDLE_HEADROOM_GB));
      // Tight: one GB less would not clear the gate's own budget.
      expect(shown - 1).toBeLessThan(peak + CANDLE_HEADROOM_GB);
    },
  );
});

describe("catalog memory floors: an ESTIMATED lane is floored at its curated blanket", () => {
  // MAJOR 4. `candle.measured === false` means "ESTIMATED (scaled from weight size + a measured
  // sibling)" (schema), and the flag covers `minMemoryGb` AND `vramGbByTier` together — so neither is a
  // measurement and neither may lower the other. `tierSuggestion` takes the max; these are the five
  // shipped entries where that is load-bearing.
  const estimated = manifestModels.filter(
    (model) =>
      model.candle?.measured === false && Object.keys(model.candle?.vramGbByTier ?? {}).length > 0,
  );

  it("finds the shipped estimated-candle entries", () => {
    // lens_turbo, flux_schnell, krea_2_raw, sd3_5_large_turbo, sd3_5_medium.
    expect(estimated.map((model) => model.id)).toContain("lens_turbo");
    expect(estimated.length).toBeGreaterThanOrEqual(5);
  });

  const estimatedIds = new Set(estimated.map((model) => model.id));
  const estimatedRows = LABEL_CASES.filter(
    (row) => row.backend === "candle" && row.subset.length > 0 && estimatedIds.has(row.id),
  );

  it("never reports below the blanket for an estimated lane, on any install-subset", () => {
    expect(estimatedRows.length).toBeGreaterThanOrEqual(20);
    for (const row of estimatedRows) {
      const blanket = row.model.candle?.minMemoryGb;
      if (!Number.isFinite(blanket)) {
        continue;
      }
      expect(
        shownGb(row.label),
        `${row.id} [${row.subset.join(",")}] must not undercut its curated blanket`,
      ).toBeGreaterThanOrEqual(blanket);
    }
  });

  it("really does raise a number the per-tier row alone would have lowered", () => {
    // Non-vacuity: at least one estimated entry whose converted per-tier figure sits BELOW its blanket,
    // so the max is doing work rather than being a no-op. lens_turbo q4: 37.3 + 2 = 40, under the
    // curated 44 — which is precisely the "advertised 42 GB came from an estimate" case.
    const lifted = estimatedRows.filter((row) => {
      if (row.subset.length !== 1) {
        return false;
      }
      const peak = laneEvidenceGb(row.model, row.os, row.subset[0]);
      const blanket = row.model.candle?.minMemoryGb;
      return (
        peak !== null &&
        Number.isFinite(blanket) &&
        hostGbForPeakGb(peak, "candle") < blanket &&
        shownGb(row.label) === blanket
      );
    });
    expect(lifted.map((row) => `${row.id} ${row.subset[0]}`)).toContain("lens_turbo q4");
  });
});

describe("catalog memory floors: the two backend lanes are independent", () => {
  // THE BLOCKER-2 GUARD, proved structurally rather than by transcribing expected numbers: strip one
  // lane's evidence out of the catalog entry and the OTHER lane's answer must be unchanged. If any
  // consumer ever reads `footprint.peakMemoryBytes` for candle (or `candle.vramGbByTier` for MLX), one of
  // these goes red for whichever real models carry both — no new fixture needed.
  function stripMlxFootprints(entry) {
    return {
      ...entry,
      variants: entry.variants.map((variant) => ({
        ...variant,
        footprint: variant.footprint
          ? { ...variant.footprint, peakMemoryBytes: undefined, residentMemoryBytes: undefined }
          : variant.footprint,
      })),
    };
  }

  function stripCandle(entry) {
    const { candle: _candle, ...rest } = entry;
    return rest;
  }

  // Models carrying BOTH an MLX footprint and candle per-tier evidence on the candle OS — the entries
  // where crossing the lanes is silently plausible because both numbers exist.
  const bothLanes = manifestModels.filter((model) => {
    const tiers = tiersOn(model, "windows");
    const hasMlx = tiers.some((tier) => mlxPeakGb(model, "windows", tier) !== null);
    const hasCandle = Object.keys(model.candle?.vramGbByTier ?? {}).length > 0;
    return hasMlx && hasCandle;
  });

  it("finds real models that carry both lanes' evidence", () => {
    // z_image_turbo and lens_turbo at minimum. If this ever hits zero the two tests below are vacuous.
    expect(bothLanes.map((model) => model.id)).toContain("lens_turbo");
    expect(bothLanes.length).toBeGreaterThanOrEqual(2);
  });

  it.each(bothLanes.map((model) => [model.id, model]))(
    "%s: the candle answer ignores the MLX footprint entirely",
    (_id, model) => {
      const tiers = tiersOn(model, "windows");
      const entry = catalogEntry(model, "windows", tiers);
      const withMlx = installedTierPeakGb(entry, { backend: "candle" });
      const withoutMlx = installedTierPeakGb(stripMlxFootprints(entry), { backend: "candle" });
      expect(withoutMlx).toEqual(withMlx);
      // Same for the uninstalled "from" floor.
      expect(cheapestDeclaredTierPeakGb(stripMlxFootprints(entry), { backend: "candle" })).toEqual(
        cheapestDeclaredTierPeakGb(entry, { backend: "candle" }),
      );
    },
  );

  it.each(bothLanes.map((model) => [model.id, model]))(
    "%s: the MLX answer ignores the candle block entirely",
    (_id, model) => {
      const tiers = tiersOn(model, "macos");
      const entry = catalogEntry(model, "macos", tiers);
      expect(installedTierPeakGb(stripCandle(entry), { backend: "mlx" })).toEqual(
        installedTierPeakGb(entry, { backend: "mlx" }),
      );
      expect(blanketFloorGb(stripCandle(entry), "mlx")).toEqual(blanketFloorGb(entry, "mlx"));
    },
  );

  it("never answers the candle lane from MLX evidence when candle declares none", () => {
    // The pre-fix failure shape: a windows/linux-served tier with an MLX footprint and NO candle
    // evidence. z_image and sdxl are the shipped examples. Silence is required — NOT the MLX number.
    let checked = 0;
    for (const model of manifestModels) {
      const tiers = tiersOn(model, "windows");
      const hasMlx = tiers.some((tier) => mlxPeakGb(model, "windows", tier) !== null);
      const hasCandleTier = Object.keys(model.candle?.vramGbByTier ?? {}).length > 0;
      const hasCandleBlanket = Number.isFinite(model.candle?.minMemoryGb);
      if (!hasMlx || hasCandleTier || hasCandleBlanket) {
        continue;
      }
      checked++;
      const entry = catalogEntry(model, "windows", tiers);
      expect(installedTierPeakGb(entry, { backend: "candle" }), `${model.id}`).toBeNull();
      expect(cheapestDeclaredTierPeakGb(entry, { backend: "candle" }), `${model.id}`).toBeNull();
      expect(blanketFloorGb(entry, "candle"), `${model.id}`).toBeNull();
    }
    expect(checked, "z_image / sdxl / illustrious_xl_* have this shape").toBeGreaterThanOrEqual(3);
  });

  // The reason the MLX blanket may NOT be reused as a "conservative" candle default. This is a manifest
  // fact, so it belongs here: if it ever became true that mlx >= candle everywhere, the fallback choice
  // could be revisited — until then, cross-lane reuse can under-state.
  it("records that the MLX blanket is NOT always at or above the candle requirement", () => {
    const inversions = manifestModels
      .filter((model) => {
        const mlx = model.mlx?.minMemoryGb;
        const candle = model.candle?.minMemoryGb;
        return Number.isFinite(mlx) && Number.isFinite(candle) && mlx < candle;
      })
      .map((model) => `${model.id} (mlx ${model.mlx.minMemoryGb} < candle ${model.candle.minMemoryGb})`);
    // qwen_image is the shipped counterexample. Its existence is what makes silence, rather than the MLX
    // integer, the correct candle-lane fallback in SimpleModelManager's `needsLabel`.
    expect(inversions.join(", ")).toContain("qwen_image");
  });
});

describe("catalog memory floors: duplicate tier keys cannot reach the client", () => {
  // The invariant MINOR-6's max-over-duplicates change stops depending on silently. Three upstream
  // filters enforce it (coRequisite, per-OS platforms, first-wins de-dupe); this pins that they do, so
  // the client-side hardening is provably belt-and-braces rather than load-bearing.
  it.each(OSES)("emits at most one entry per tier key on %s", (os) => {
    let withDuplicatesUpstream = 0;
    for (const model of manifestModels) {
      const entry = catalogEntry(model, os, []);
      const keys = entry.variants.map((variant) => variant.variant);
      expect(new Set(keys).size, `${model.id} on ${os} emitted duplicate variant keys`).toBe(
        keys.length,
      );
      // ...while the RAW manifest really does declare duplicates, so the filters are doing work.
      const rawTierKeys = (model.downloads ?? [])
        .map((download) => download.variant)
        .filter((key) => tierQuantize(key) !== null);
      if (new Set(rawTierKeys).size !== rawTierKeys.length) {
        withDuplicatesUpstream++;
      }
    }
    // mage_flow x6 (coRequisite component rows) + wan_2_2* x3 (per-platform twins).
    expect(withDuplicatesUpstream).toBeGreaterThanOrEqual(6);
  });
});

// ---------------------------------------------------------------------------------------------------
// THE CONSISTENCY INVARIANT (sc-15400 round 4)
//
// Every sweep above compares the label to the lane's own per-tier EVIDENCE. That is necessary but it is
// structurally unable to express the round-3 blocker, because the offending tiers were the ones with NO
// evidence: `z_image_edit`'s q8/bf16 contributed nothing to compare against, so the label could quote q4's
// 22 GB for a `{q4,bf16}` install and every evidence-based assertion stayed green.
//
// So this sweep asserts a DIFFERENT property, over the same catalog: the number the label quotes must be a
// host at which the module's OWN PICKER would accept every installed tier. `tierFits` is the gate
// `suggestTier` uses to decide which tier to preselect on the Models screen, and it needs no per-tier
// measurement — it falls back to `variantFootprintBytes`'s disk estimate. That is what lets it see the tiers
// the evidence sweeps cannot, and it is why "label ≥ what tierFits demands" is the invariant that makes the
// class unrepeatable rather than merely fixing the instance.
//
// SC-15613 makes that picker lane-aware. MLX still compares its own footprint against a shared-pool
// budget; Candle reads only `candle.vramGbByTier` and applies the worker-owned dedicated-VRAM slack.
// Missing Candle evidence remains unknown/fail-open and must never borrow the MLX footprint.
// ---------------------------------------------------------------------------------------------------

// The candle-only tiers' host-eligibility gates, as `SimpleModelManager` derives them from live workers.
// Both flags matter because either can REMOVE a tier from the install set, and a tier that cannot contribute
// to the ceiling must also not be one the invariant demands coverage for.
const ELIGIBILITY = [
  { convRotEligible: true, nvfp4Eligible: true },
  { convRotEligible: true, nvfp4Eligible: false },
  { convRotEligible: false, nvfp4Eligible: true },
  { convRotEligible: false, nvfp4Eligible: false },
];

// Every (model, OS, install-subset, eligibility) the real catalog can produce.
const SWEEP = [];
for (const os of OSES) {
  const backend = BACKEND_FOR_OS[os];
  for (const model of manifestModels) {
    const tiers = tiersOn(model, os);
    if (tiers.length === 0) {
      continue;
    }
    for (const subset of subsetsOf(tiers)) {
      for (const eligibility of ELIGIBILITY) {
        const entry = {
          ...catalogEntry(model, os, subset),
          installState: subset.length > 0 ? "installed" : "missing",
        };
        const options = { backend, ...eligibility };
        SWEEP.push({
          id: model.id,
          os,
          backend,
          subset,
          tiers,
          model,
          entry,
          options,
          eligibility,
          label: needsLabel(entry, options),
        });
      }
    }
  }
}

describe("catalog memory floors: the label never contradicts the module's own picker", () => {
  it("sweeps every (model, OS, install-subset, eligibility) on both lanes", () => {
    // 4 eligibility combinations over every install-subset of every matrix model on both lanes.
    expect(SWEEP.length).toBeGreaterThanOrEqual(1200);
    expect(SWEEP.some((row) => row.backend === "mlx")).toBe(true);
    expect(SWEEP.some((row) => row.backend === "candle")).toBe(true);
    // All four eligibility combinations really are present...
    const combos = new Set(
      SWEEP.map((row) => `${row.eligibility.convRotEligible}/${row.eligibility.nvfp4Eligible}`),
    );
    expect(combos.size).toBe(4);
    // ...and at least one really CHANGES the install set, or the extra dimension would be inert.
    const gated = SWEEP.filter(
      (row) => row.subset.length > 0 && row.subset.some((tier) => !tierHostEligible(tier, row.eligibility)),
    );
    expect(gated.length).toBeGreaterThan(0);
  });

  it("MLX: every installed eligible tier FITS at the host the label quotes", () => {
    // THE INVARIANT. For every install-subset, for every installed host-eligible tier, `tierFits` must accept
    // the quoted host. A silent row asserts nothing (it makes no claim), so silence is counted separately and
    // required to be the evidenceless case only — otherwise this test could be satisfied by saying nothing.
    const violations = [];
    const silent = [];
    let asserted = 0;
    let subsetsCovered = 0;
    for (const row of SWEEP) {
      if (row.backend !== "mlx" || row.subset.length === 0) {
        continue;
      }
      const installed = row.subset.filter((tier) => tierHostEligible(tier, row.eligibility));
      if (installed.length === 0) {
        continue;
      }
      subsetsCovered++;
      const shown = shownGb(row.label);
      const estimable = installed.filter((tier) =>
        variantFootprintBytes(row.entry.variants.find((v) => v.variant === tier), row.entry) !== null,
      );
      if (shown === null) {
        if (estimable.length > 0) {
          silent.push(`${row.id} on ${row.os} [${row.subset.join(",")}] said nothing`);
        }
        continue;
      }
      for (const tier of installed) {
        const variant = row.entry.variants.find((v) => v.variant === tier);
        if (variantFootprintBytes(variant, row.entry) === null) {
          continue;
        }
        asserted++;
        if (!tierFits(variant, shown, { model: row.entry })) {
          const need = Math.ceil(
            variantFootprintBytes(variant, row.entry).bytes /
              BYTES_PER_GB /
              MEMORY_HEADROOM_FRACTION,
          );
          violations.push(
            `${row.id} on ${row.os} [${row.subset.join(",")}]: label ` +
              `${JSON.stringify(row.label)} but tierFits refuses ${tier} at ${shown} GB (needs ${need})`,
          );
        }
      }
      // ...and the two surfaces must agree on WHICH tier is servable: at the quoted host, `suggestTier` must
      // not have to fall below the heaviest installed tier. This is the form the round-3 defect took —
      // `suggestTier(z_image_edit, 22)` was "q4" while bf16 sat on disk.
      if (estimable.length > 0) {
        const heaviest = estimable
          .map((tier) => row.entry.variants.find((v) => v.variant === tier))
          .reduce((a, b) =>
            variantFootprintBytes(a, row.entry).bytes >=
            variantFootprintBytes(b, row.entry).bytes
              ? a
              : b,
          );
        if (!tierFits(heaviest, shown, { model: row.entry })) {
          violations.push(
            `${row.id} on ${row.os} [${row.subset.join(",")}]: suggestTier at ${shown} GB degrades ` +
              `below the installed ${heaviest.variant} (picked ${suggestTier(row.entry, shown)})`,
          );
        }
      }
    }
    expect(violations.join("\n")).toBe("");
    expect(silent.join("\n")).toBe("");
    // Non-vacuity: the sweep must really be comparing a large corpus, and every eligible subset must have
    // been visited (which is what the round-3 suite's `peak === null` skip prevented).
    expect(asserted, "must compare a real corpus of installed tiers").toBeGreaterThanOrEqual(500);
    expect(subsetsCovered).toBeGreaterThanOrEqual(400);
  });

  it("Candle: every evidenced installed tier fits at the dedicated-VRAM host the label quotes", () => {
    const violations = [];
    let asserted = 0;
    for (const row of SWEEP) {
      if (row.backend !== "candle" || row.subset.length === 0) {
        continue;
      }
      const shown = shownGb(row.label);
      if (shown === null) {
        continue;
      }
      const eligible = row.subset.filter((t) => tierHostEligible(t, row.eligibility));
      for (const tier of eligible) {
        const variant = row.entry.variants.find((v) => v.variant === tier);
        const evidence = laneEvidenceGb(row.model, row.os, tier);
        if (evidence === null) {
          continue;
        }
        asserted++;
        if (!tierFits(variant, shown, { backend: "candle", model: row.entry })) {
          violations.push(
            `${row.id} on ${row.os} [${row.subset.join(",")}]: ${tier} rejected at ${shown} GB`,
          );
        }
      }
    }
    expect(violations.join("\n")).toBe("");
    expect(asserted).toBeGreaterThanOrEqual(500);
  });

  it("pins that the linux projection is byte-identical to windows, so two OSes sweep exhaustively", () => {
    // OSES is macos + windows. linux runs the same candle lane, and if any download ever declared
    // `platforms: ["windows"]` without linux (or vice versa) the sweep would silently stop being exhaustive.
    for (const model of manifestModels) {
      const project = (os) =>
        catalogEntry(model, os, []).variants.map(
          (v) => `${v.variant}:${v.footprint?.diskSizeBytes ?? "-"}:${v.footprint?.peakMemoryBytes ?? "-"}`,
        );
      expect(project("linux"), `${model.id}: linux must project identically to windows`).toEqual(
        project("windows"),
      );
    }
  });
});

describe("catalog memory floors: the shapes the round-4 guards depend on", () => {
  it("has no candle entry with partial per-tier rows and no blanket", () => {
    // `installedFloorHostGb` case 4 returns null rather than quoting a ceiling over a strict subset. On the
    // MLX lane the disk-based fill almost always avoids that branch; on CANDLE there is no fill (the lane
    // rule), so the branch would be reachable for an entry that ships SOME `vramGbByTier` rows and no
    // `candle.minMemoryGb`. No such entry exists, which is why the label never falls silent while candle
    // evidence exists. If one ever ships, this reds — and the "never falls silent" sweep would red too, so
    // the failure mode is a red build, not a wrong number.
    const partialNoBlanket = manifestModels.filter(
      (model) =>
        Object.keys(model.candle?.vramGbByTier ?? {}).length > 0 &&
        !Number.isFinite(model.candle?.minMemoryGb),
    );
    expect(partialNoBlanket.map((model) => model.id).join(", ")).toBe("");
  });

  it("has exactly one shipped shape that reached the round-3 blocker, and it is z_image_edit on MLX", () => {
    // The blocker class, enumerated from the manifest: an install-set with SOME evidenced tier, SOME
    // unevidenced tier, and no blanket on the lane. Pinned so the fix's blast radius stays a known quantity.
    const reached = new Set();
    for (const row of SWEEP) {
      if (row.subset.length === 0) {
        continue;
      }
      const installed = row.subset.filter((tier) => tierHostEligible(tier, row.eligibility));
      const evidenced = installed.filter((tier) => laneEvidenceGb(row.model, row.os, tier) !== null);
      const unevidenced = installed.filter((tier) => laneEvidenceGb(row.model, row.os, tier) === null);
      if (
        evidenced.length > 0 &&
        unevidenced.length > 0 &&
        blanketFloorGb(row.model, row.backend) === null
      ) {
        reached.add(`${row.id}|${row.os}`);
      }
    }
    expect([...reached].sort()).toEqual(["z_image_edit|macos"]);
  });

  it("reads z_image_edit's four install-sets off the MANIFEST, and agrees with the picker", () => {
    // The round-4 headline, driven entirely from `builtin.models.jsonc` — no transcribed expectation. The
    // module's own estimator supplies the number, `tierFits`/`suggestTier` supply the cross-check, and the
    // only literal is the ORDERING claim (that the label rises as heavier tiers join the set).
    const model = manifestModels.find((entry) => entry.id === "z_image_edit");
    expect(model, "z_image_edit must still be in the manifest").toBeTruthy();
    // The precondition that made this the blocker: no blanket on the MLX lane to catch the partial ceiling.
    expect(blanketFloorGb(model, "mlx")).toBeNull();

    const options = { backend: "mlx", convRotEligible: true, nvfp4Eligible: true };
    const labelFor = (installed) => {
      const entry = { ...catalogEntry(model, "macos", installed), installState: "installed" };
      return { entry, shown: shownGb(needsLabel(entry, options)) };
    };
    // Each tier's requirement, derived from the module's own estimator rather than written down.
    const hostFor = (entry, tier) =>
      hostGbForPeakGb(
        variantFootprintBytes(entry.variants.find((v) => v.variant === tier), entry).bytes /
          BYTES_PER_GB,
        "mlx",
      );

    const sets = [["q4"], ["q4", "q8"], ["q4", "bf16"], ["q4", "q8", "bf16"]];
    const shownBySet = sets.map((installed) => {
      const { entry, shown } = labelFor(installed);
      // The label equals the host the HEAVIEST installed tier demands — the whole set, not a subset.
      const required = Math.max(...installed.map((tier) => hostFor(entry, tier)));
      expect(shown, `[${installed.join(",")}] must quote its heaviest installed tier`).toBe(required);
      // ...and the picker agrees at that host, in both directions.
      for (const tier of installed) {
        const variant = entry.variants.find((v) => v.variant === tier);
        expect(tierFits(variant, shown, { model: entry }), `[${installed.join(",")}] tierFits(${tier})`).toBe(
          true,
        );
      }
      const heaviest = installed.reduce((a, b) => (hostFor(entry, a) >= hostFor(entry, b) ? a : b));
      expect(
        tierFits(entry.variants.find((v) => v.variant === heaviest), shown - 1, { model: entry }),
      ).toBe(false);
      // `suggestTier` must not have to degrade below what is installed.
      expect(
        hostFor(entry, suggestTier(entry, shown)),
        `[${installed.join(",")}] suggestTier must not degrade below the installed set`,
      ).toBeGreaterThanOrEqual(hostFor(entry, heaviest));
      return shown;
    });

    // q4 alone is MEASURED, so it is unchanged by the fill; the rest rise strictly as heavier tiers join,
    // and {q4,q8,bf16} lands on the same figure as {q4,bf16} because bf16 dominates.
    const [q4, q4q8, q4bf16, all] = shownBySet;
    expect(q4).toBeLessThan(q4q8);
    expect(q4q8).toBeLessThan(q4bf16);
    expect(all).toBe(q4bf16);
    // The pre-fix answer was q4's figure for ALL FOUR. Pinned as an inequality so it cannot silently return.
    expect(q4q8).not.toBe(q4);
    expect(q4bf16).not.toBe(q4);
  });

  it("counts the candle.measured === false entries the way tierSuggestion.js describes them", () => {
    // MINOR 4. The header said `candle.measured` is false on "five shipped entries". It is false on 19;
    // eleven of those also carry `vramGbByTier`, which is the set the rule is about. Both numbers are
    // pinned so the prose cannot drift from the catalog again. sc-16025 adds the six Mage profiles to
    // this set because their q4/q8 numeric identity changed and the historical samples are no longer
    // valid current measurements.
    const estimatedLane = manifestModels.filter((model) => model.candle?.measured === false);
    expect(estimatedLane.length).toBe(19);
    const withPerTier = estimatedLane.filter(
      (model) => Object.keys(model.candle?.vramGbByTier ?? {}).length > 0,
    );
    expect(withPerTier.map((model) => model.id).sort()).toEqual(
      [
        "flux_schnell",
        "krea_2_raw",
        "lens_turbo",
        "mage_flow",
        "mage_flow_base",
        "mage_flow_edit",
        "mage_flow_edit_base",
        "mage_flow_edit_turbo",
        "mage_flow_turbo",
        "sd3_5_large_turbo",
        "sd3_5_medium",
      ].sort(),
    );
    // The other eight are blanket-only, so the flag has nothing per-tier to qualify on them.
    expect(estimatedLane.length - withPerTier.length).toBe(8);
  });

  it("pins the shipped shapes the SimpleModelManager fixtures claim", () => {
    // FIXTURE PROVENANCE. Two fixtures misstated what ships, and one of them concealed krea_2_raw's candle
    // under-statement through two review rounds. These assertions make the claims checkable from the manifest
    // instead of from a comment.
    const byId = new Map(manifestModels.map((model) => [model.id, model]));
    // sensenova_u1_8b is NOT "candle.minMemoryGb only" — it ships a full measured ladder.
    expect(byId.get("sensenova_u1_8b").candle).toMatchObject({
      minMemoryGb: 16,
      vramGbByTier: { q4: 14.8, q8: 22.5, bf16: 36.7 },
      measured: true,
    });
    expect(byId.get("sensenova_u1_8b").mlx?.minMemoryGb).toBeUndefined();
    // krea_2_raw ships a candle block, and ships NO footprint on any download.
    expect(byId.get("krea_2_raw").candle).toMatchObject({
      minMemoryGb: 32,
      vramGbByTier: { q4: 28.8, q8: 33.4, bf16: 46.4 },
      measured: false,
    });
    expect(byId.get("krea_2_raw").mlx?.minMemoryGb).toBe(48);
    for (const download of byId.get("krea_2_raw").downloads ?? []) {
      expect(download.footprint, "krea_2_raw ships no footprint at all").toBeUndefined();
    }
    // ...and it is NOT the only one: krea_2_turbo has the identical shape, and the two of them are the
    // complete set of footprint-less matrix entries. The fixture comment claimed "the ONE shipped entry".
    const footprintless = manifestModels
      .filter((model) => tiersOn(model, "macos").length > 0)
      .filter((model) =>
        (model.downloads ?? []).every((download) => download.footprint === undefined),
      )
      .map((model) => model.id)
      .sort();
    expect(footprintless).toEqual(["krea_2_raw", "krea_2_turbo"]);
    // The same fixture comment also claimed there is "nothing for `mlxEstimatedPeakGb` to read" on these
    // two. There is: `estimatedSizeBytes`, which the catalog emits as `variants[].downloadSizeBytes` and
    // `variantFootprintBytes` consumes at priority 4. This is the field whose absence from the mirror hid
    // the eight `needs 48 GB` answers production states as 53 — pinned so it cannot silently go away.
    for (const id of footprintless) {
      const bf16 = (byId.get(id).downloads ?? []).find((download) => download.variant === "bf16");
      expect(catalogDownloadSizeBytes(byId.get(id), bf16), `${id} bf16 size`).toBeGreaterThan(
        30 * BYTES_PER_GB,
      );
      const entry = { ...catalogEntry(byId.get(id), "macos", ["bf16"]), installState: "installed" };
      const estimated =
        variantFootprintBytes(entry.variants.find((variant) => variant.variant === "bf16")).bytes /
        BYTES_PER_GB;
      expect(
        needsLabel(entry, { backend: "mlx", convRotEligible: true, nvfp4Eligible: true }),
        `${id} bf16-only install`,
      ).toBe(`needs ${hostGbForPeakGb(estimated, "mlx")} GB`);
      // ...and that this really is ABOVE the blanket 48 the mirror used to answer, so the pin would red if
      // the fill ever silently reverted to it.
      expect(hostGbForPeakGb(estimated, "mlx")).toBeGreaterThan(byId.get(id).mlx.minMemoryGb);
    }
    // The genuinely blanket-only shape the corrected fixture uses instead.
    expect(byId.get("kokoro_82m").candle).toMatchObject({ minMemoryGb: 2, measured: false });
    expect(byId.get("kokoro_82m").candle.vramGbByTier).toBeUndefined();
    expect(byId.get("kokoro_82m").mlx ?? null).toBeNull();
    // The real single-variant image entry the "sana_sprint" fixture should have named. `sana_sprint` is not a
    // manifest id at all; the real SANA entries ship a full matrix on macOS.
    expect(byId.get("sana_sprint")).toBeUndefined();
    expect(byId.get("flux2_klein_9b_true_v2").mlx?.minMemoryGb).toBe(42);
    expect(
      (byId.get("flux2_klein_9b_true_v2").downloads ?? []).filter((d) => d.coRequisite !== true).length,
    ).toBe(1);
    // ...and the disk sizes the fixtures now carry, since the round-4 fill reads exactly this field.
    const diskOf = (id, tier) =>
      (byId.get(id).downloads ?? []).find((d) => d.variant === tier)?.footprint?.diskSizeBytes;
    expect(diskOf("z_image", "q8")).toBe(10996438540);
    expect(diskOf("z_image", "bf16")).toBe(20538412852);
    expect(diskOf("sensenova_u1_8b", "q4")).toBe(11824472971);
    expect(diskOf("flux_dev", "bf16")).toBe(33746402173);
    expect(diskOf("lens_turbo", "q8")).toBe(32753474289);
    expect(diskOf("z_image_edit", "bf16")).toBe(20538406851);
    expect(diskOf("chroma1_hd", "q4")).toBe(15121207891);
  });
});
