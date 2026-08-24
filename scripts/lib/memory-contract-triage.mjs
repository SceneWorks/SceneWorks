// Classify the memory-contract reconciliation's findings into "cannot be otherwise" and "two sources
// disagree" (sc-21505).
//
// WHY THIS EXISTS
// ---------------
// The reconciliation enumerates every coordinate on which the engine registries, the manifest
// declarations, the route witnesses and the rung-4 survey disagree. At the sc-20799 pin that is 471
// coordinates, and an undifferentiated 471 reads as 471 things to fix. It is not: the large majority
// are coordinates that CANNOT carry a declaration — the provider has no image catalog entry to
// declare it in, the catalog does not advertise the tier, production witnesses no route there, or the
// MLX lane would read the row as malformed and refuse the load. Writing those declarations is exactly
// the fabricated paperwork the deleted waiver ledger was made of.
//
// So this splits the enumeration by DISPOSITION:
//
//   * `by-construction` — the coordinate is a consequence of how the two sides are keyed, and the
//     honest count of work here is zero. Declaring it would either be unreachable or actively wrong.
//   * `drift` — the two sides make contradictory claims about the SAME fact, so one of them is
//     wrong. This is the worklist.
//
// Still not a gate. This classifies a report; nothing fails on it (Michael, 2026-08-17).
//
// The `cause` half comes from `collectMemoryContractMismatches`, recorded at each emit site. The
// engine_manifest leg needs a second input: WHY the manifest carries no declaration is a property of
// the projection, so `planProjection` from `manifest-memory-declarations.mjs` is joined on here.

/** @typedef {"by-construction" | "drift"} Disposition */

const CLASSES = Object.freeze({
  engine_provider_unhosted: {
    disposition: "by-construction",
    title: "engine provider has no image-manifest host",
    rationale:
      "The provider publishes a memory contract but no image catalog entry routes to it (video, " +
      "audio and renderer-only providers, plus the Kolors control/IP-adapter composites). " +
      "`config/manifests/builtin.models.jsonc` declares per-model, so there is no row these could " +
      "be written into. Nothing to fix in the manifest.",
  },
  tier_not_in_catalog: {
    disposition: "by-construction",
    title: "catalog entry does not advertise the tier",
    rationale:
      "The engine implements the rung at a tier the hosting catalog entry does not offer on that " +
      "backend. A declaration there would claim a capability no request can select.",
  },
  no_route_witness: {
    disposition: "by-construction",
    title: "no production route witnessed at that tier",
    rationale:
      "The engine implements the rung at the tier, but the deferred-route registry witnesses no " +
      "route reaching it. A declaration would be an unreachable capability claim — the failure " +
      "mode the waiver ledger was made of.",
  },
  mlx_request_context_lane: {
    disposition: "by-construction",
    title: "MLX lane requires `requestContexts`, which no dump publishes",
    rationale:
      "A non-legacy MLX route rule reads a declaration row without `requestContexts` as malformed " +
      "and fails the whole load closed to Refused + Eager. Declaring here would be worse than not " +
      "declaring: it would break the lane it claims to describe.",
  },
  manifest_declaration_withheld: {
    disposition: "by-construction",
    title: "declaration deliberately withheld by a recorded measured verdict",
    rationale:
      "`memoryDeclarationWithhold` names the story and reason on the model block. The absence is " +
      "the recorded verdict, not drift.",
  },
  survey_wildcard_axis: {
    disposition: "by-construction",
    title: "survey scope omits the axis, so the wildcard expands to every cell",
    rationale:
      "A rung-4 survey verdict that omits `implementedTiers` / `implementedModes` / " +
      "`implementedOverlays` claims every value of that axis, and the reconciliation expands it " +
      "against per-coordinate engine facts. A coordinate reached only through an omitted axis is " +
      "not an assertion the survey ever made about that coordinate — the two sides are keyed at " +
      "different grains. Narrowing the verdicts would change this class, but nothing here says the " +
      "survey is WRONG.",
  },
  survey_withholds_loaded_overlay: {
    disposition: "by-construction",
    title: "survey withholds a loaded overlay the engine dump cannot exclude",
    rationale:
      "The survey claims the coordinate at clean base and withholds the loaded overlay. A dump's " +
      "contract surface is keyed (tier, offloadPolicy, loadShape) with NO overlay axis, so the " +
      "engine side cannot represent an overlay-conditioned rung — and the route witnesses reach " +
      "the overlay because ROUTING is not RUNG CAPABILITY. The providers are explicit that rung 4 " +
      "is clean-base only (FLUX.1 `structurally_streamable`, FLUX.2 Klein `klein_streamable`, Mage " +
      "`surface_streamable` all require an empty adapter/identity/overlay spec), so the survey is " +
      "the more precise record here, not the wrong one.",
  },
  survey_records_none: {
    disposition: "drift",
    title: "survey records `implementation: none` where the engine implements and routes the rung",
    rationale:
      "A direct contradiction about one fact. The engine dump is the executable registry's own " +
      "output, so the survey verdict is stale and must be re-derived at the current pin.",
  },
  survey_scope_underclaims: {
    disposition: "drift",
    title: "survey scope is narrower than the engine implements and routes",
    rationale:
      "The survey names an explicit scope that excludes coordinates the engine both implements at " +
      "the tier and witnesses a route for. One of the two is wrong about the same fact.",
  },
  survey_scope_overclaims: {
    disposition: "drift",
    title: "survey names a coordinate the engine dump does not reach",
    rationale:
      "Not a wildcard: the verdict NAMES the axis they differ on, so one of the two is asserting " +
      "something false and a human has to say which. Do not assume it is the survey. " +
      "`qwen_image` is the standing counter-example — the dump carries " +
      "`bounded_transformer_residency` on its bf16 surface alone, while the checked-in calibration " +
      "evidence promotes the q4 AND q8 rung-4 cells to Verified, so there the DUMP under-reports " +
      "and the survey is right. Check for measured evidence on the coordinate before narrowing a " +
      "verdict to match a dump.",
  },
  manifest_overdeclares_cell: {
    disposition: "drift",
    title: "manifest declares a (mode, overlay) the matrix has no cell for",
    rationale:
      "A hand-authored declaration row names coordinates the catalog entry does not produce. The " +
      "engine-projected rows cannot land here — the projection intersects with the entry's own " +
      "axes — so these are rows that predate the projection and were never narrowed to it.",
  },
  manifest_overdeclares_route: {
    disposition: "drift",
    title: "manifest declares a coordinate with no witnessed route at its load profile",
    rationale:
      "The matrix cell exists, but the deferred-route registry witnesses no route for the declared " +
      "load profile. The declaration claims a path production does not take.",
  },
  unclassified: {
    disposition: "drift",
    title: "unclassified",
    rationale:
      "No triage rule matched. Deliberately filed as drift rather than benign: an unexplained " +
      "coordinate must surface, never be absorbed into a class that says there is nothing to do.",
  },
});

export const TRIAGE_CLASSES = CLASSES;

const PROJECTION_REASONS = new Map([
  ["no-route-witness", "no_route_witness"],
  ["tier-not-advertised", "tier_not_in_catalog"],
  ["requires-request-contexts", "mlx_request_context_lane"],
  ["withheld", "manifest_declaration_withheld"],
]);

const CAUSE_CLASSES = new Map([
  ["survey_wildcard_axis", "survey_wildcard_axis"],
  ["survey_records_none", "survey_records_none"],
  ["survey_scope_underclaims", "survey_scope_underclaims"],
  ["survey_withholds_loaded_overlay", "survey_withholds_loaded_overlay"],
  ["survey_scope_overclaims", "survey_scope_overclaims"],
  ["declared_cell_absent", "manifest_overdeclares_cell"],
  ["declared_route_unwitnessed", "manifest_overdeclares_route"],
  ["no_engine_contract", "unclassified"],
]);

/**
 * Which triage class does one finding belong to?
 *
 * `projection` is a `planProjection` result. The engine_manifest leg is the only one that needs it:
 * that leg says "the manifest has no declaration", and the projector is what knows why it could not
 * write one.
 */
export function classifyMemoryContractMismatch(finding, projection) {
  if (finding.leg !== "engine_manifest") {
    return CAUSE_CLASSES.get(finding.cause) ?? "unclassified";
  }
  const lane = `${finding.backend}:${finding.provider}`;
  if ((projection.unhosted ?? []).some((row) => `${row.backend}:${row.provider}` === lane)) {
    return "engine_provider_unhosted";
  }
  // `already-declared` is deliberately absent from PROJECTION_REASONS: it means the manifest DOES
  // carry the declaration, which contradicts the finding's own premise. Falling through to
  // `unclassified` surfaces that contradiction instead of laundering it into a benign class.
  const skipped = (projection.skipped ?? []).find(
    (row) =>
      row.backend === finding.backend &&
      row.provider === finding.provider &&
      row.rung === finding.rung &&
      (row.tiers ?? []).includes(finding.tier),
  );
  return PROJECTION_REASONS.get(skipped?.reason) ?? "unclassified";
}

/**
 * Group the whole enumeration by class, with the by-construction / drift split totalled.
 *
 * Returns `{ total, byDisposition, classes }`, where `classes` is ordered largest first and every
 * entry carries its findings so a caller can print or filter them.
 */
export function triageMemoryContractMismatches(findings, projection) {
  const classes = new Map();
  for (const finding of findings) {
    const name = classifyMemoryContractMismatch(finding, projection);
    if (!classes.has(name)) classes.set(name, []);
    classes.get(name).push({ ...finding, triageClass: name });
  }
  const byDisposition = { "by-construction": 0, drift: 0 };
  for (const [name, entries] of classes) {
    byDisposition[CLASSES[name].disposition] += entries.length;
  }
  return {
    total: findings.length,
    byDisposition,
    classes: [...classes]
      .map(([name, entries]) => ({ name, ...CLASSES[name], count: entries.length, findings: entries }))
      .sort((left, right) =>
        right.count - left.count || left.name.localeCompare(right.name),
      ),
  };
}
