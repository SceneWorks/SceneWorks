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
  installedTierPeakGb,
  tierFits,
} from "./tierSuggestion.js";
import { isSelectableTier, tierQuantize } from "./quantTier.js";
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
  // NOTE the `peak === null` skip: this test only sees install-sets where EVERY tier is measured. That
  // is exactly the blind spot the label sweep below exists to close — the blanket-fallback path, where
  // the previous round's under-statement lived, never reaches this loop at all.
  it("never reports a host size below the lane's measured peak, on either lane", () => {
    let checked = 0;
    for (const os of OSES) {
      const backend = BACKEND_FOR_OS[os];
      for (const model of manifestModels) {
        const tiers = tiersOn(model, os);
        if (tiers.length === 0) {
          continue;
        }
        // The "everything installed" host: the ceiling over all tiers.
        const entry = catalogEntry(model, os, tiers);
        const peak = installedTierPeakGb(entry, { backend });
        if (peak === null) {
          continue;
        }
        const shown = hostGbForPeakGb(peak, backend);
        checked++;
        expect(shown, `${model.id} on ${os}`).toBeGreaterThanOrEqual(Math.ceil(peak));
        // ...and it is the ceiling, not some tier's number: no single tier may exceed it, judged by
        // THAT LANE's own criterion (MLX multiplies by 0.9, candle adds 2 — see `hostGbForPeakGb`).
        for (const tier of tiers) {
          const lanePeak = laneEvidenceGb(model, os, tier);
          if (lanePeak !== null) {
            expect(
              hostGbForPeakGb(lanePeak, backend),
              `${model.id} ${tier} on ${os} must not exceed the reported ceiling`,
            ).toBeLessThanOrEqual(shown);
          }
        }
      }
    }
    expect(checked, "must actually check some models on both lanes").toBeGreaterThanOrEqual(8);
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
