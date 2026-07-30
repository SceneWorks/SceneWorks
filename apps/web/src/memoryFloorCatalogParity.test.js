import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import JSON5 from "json5";
import { describe, expect, it } from "vitest";
import {
  CANDLE_HEADROOM_GB,
  MEMORY_HEADROOM_FRACTION,
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
//   * `apply_variant_fields` — emits `variants[]` carrying the raw `footprint`, plus `hasVariantMatrix`.
// The rest of the entry passes through unchanged (the catalog decorates the manifest object in place),
// which is what makes `mlx` / `candle` — including `candle.vramGbByTier` — reach the client at all.
//
// AN HONEST LIMIT ON WHAT THIS MIRROR CAN CATCH: it is a JS re-implementation, so it pins the MANIFEST
// against the mirror, NOT the mirror against the Rust. If someone changes `retain_downloads_for_os`,
// `is_co_requisite_download` or `is_supported_model_download` in models.rs, nothing here goes red — the
// two just silently disagree, and this file would then be asserting about a catalog shape the client
// never receives. The provider/repo filter above is a case in point: it was missing from the first
// version of this mirror and drifted zero rows only because every shipped download happens to be a
// huggingface entry with a repo. Cross-language enforcement would need the Rust to emit the projection
// (or a shared fixture); that is out of scope here and deliberately not claimed.
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
// WHY IT IS ASSERTED ON THE MLX LANE ONLY. `tierFits` is lane-BLIND and MLX-flavoured in both of its parts:
// its input is `variantFootprintBytes` (disk × 1.0 + a 14 GiB transient measured by the MLX
// `footprint_measure.rs` harness at 1024², an Apple unified-memory quantity) and its criterion is
// `peak <= host × 0.9`, the MLX budget. On MLX that makes the invariant hold BY CONSTRUCTION —
// `hostGbForPeakGb(_, "mlx")` is `ceil(peak / 0.9)`, literally `tierFits`'s own boundary.
//
// On candle it is not an invariant, it is a category error, and asserting it there would overwrite every
// number this PR's review confirmed. Measured against the shipped catalog: requiring it on candle demands
// `flux_dev` 51 where `vram_gate` requires 34, `sd3_5_medium` 42 against 36, `lens_turbo` 50 against 44 —
// 346 combinations in total, and all 294 of the fully-evidenced ones. Those are the figures the candle gate
// itself admits at (`peak + 2` on `candle.vramGbByTier`, a discrete-VRAM measurement), so raising them would
// re-create precisely the over-warning MAJOR 3 removed. The candle lane's correct analogue is its own gate's
// criterion, which is what the evidence sweeps above already assert with zero violations.
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
        variantFootprintBytes(row.entry.variants.find((v) => v.variant === tier)) !== null,
      );
      if (shown === null) {
        if (estimable.length > 0) {
          silent.push(`${row.id} on ${row.os} [${row.subset.join(",")}] said nothing`);
        }
        continue;
      }
      for (const tier of installed) {
        const variant = row.entry.variants.find((v) => v.variant === tier);
        if (variantFootprintBytes(variant) === null) {
          continue;
        }
        asserted++;
        if (!tierFits(variant, shown)) {
          const need = Math.ceil(
            variantFootprintBytes(variant).bytes / BYTES_PER_GB / MEMORY_HEADROOM_FRACTION,
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
          .reduce((a, b) => (variantFootprintBytes(a).bytes >= variantFootprintBytes(b).bytes ? a : b));
        if (!tierFits(heaviest, shown)) {
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

  it("records WHY the same invariant is not asserted on the candle lane", () => {
    // A manifest fact, pinned so the asymmetry above cannot be mistaken for an oversight — and so that if it
    // ever stopped being true, the choice would be revisited rather than silently inherited.
    //
    // `tierFits`'s MLX budget demands strictly MORE than the candle gate on the shipped corpus, including on
    // entries whose candle coverage is COMPLETE (so no fill is involved and the figure is purely the gate's
    // own arithmetic). Forcing agreement would raise those confirmed numbers.
    const wouldRaise = [];
    for (const row of SWEEP) {
      if (row.backend !== "candle" || row.subset.length === 0) {
        continue;
      }
      const shown = shownGb(row.label);
      if (shown === null) {
        continue;
      }
      for (const tier of row.subset.filter((t) => tierHostEligible(t, row.eligibility))) {
        const variant = row.entry.variants.find((v) => v.variant === tier);
        const footprint = variantFootprintBytes(variant);
        const evidence = laneEvidenceGb(row.model, row.os, tier);
        if (footprint === null || evidence === null || tierFits(variant, shown)) {
          continue;
        }
        // The candle GATE is satisfied at the quoted host; only the MLX-flavoured `tierFits` is not.
        expect(shown, `${row.id} ${tier}: the candle gate must still be satisfied`).toBeGreaterThanOrEqual(
          hostGbForPeakGb(evidence, "candle"),
        );
        wouldRaise.push(`${row.id} ${tier}`);
      }
    }
    // Real and numerous, on entries with full candle coverage — flux_dev among them.
    expect(wouldRaise.length).toBeGreaterThanOrEqual(50);
    expect(wouldRaise.join(",")).toContain("flux_dev");
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

  it("counts the candle.measured === false entries the way tierSuggestion.js describes them", () => {
    // MINOR 4. The header said `candle.measured` is false on "five shipped entries". It is false on 13; five
    // of those also carry `vramGbByTier`, which is the set the rule is about. Both numbers pinned so the
    // prose cannot drift from the catalog again.
    const estimatedLane = manifestModels.filter((model) => model.candle?.measured === false);
    expect(estimatedLane.length).toBe(13);
    const withPerTier = estimatedLane.filter(
      (model) => Object.keys(model.candle?.vramGbByTier ?? {}).length > 0,
    );
    expect(withPerTier.map((model) => model.id).sort()).toEqual(
      ["flux_schnell", "krea_2_raw", "lens_turbo", "sd3_5_large_turbo", "sd3_5_medium"].sort(),
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
