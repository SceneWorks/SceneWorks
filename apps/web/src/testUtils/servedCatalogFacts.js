import candidate from "./catalogBaseline/candidate-catalog.json";

// The facts the API projection adds to a shipped manifest entry and the web reads instead of
// restating (SW-7032): the served memory guidance and the server's own capability verdicts. Read
// from the native catalog fixture (`dump-ui-catalog`), so a test fixture built from the raw manifest
// carries exactly what the catalog route serves.
const SERVED_FIELDS = ["memoryGuidance", "supportsLastOnlyImageToVideo", "supportsMinimaxH3Turbo"];

export function servedCatalogFacts(id) {
  const entry = candidate.models.find((model) => model.id === id);
  if (!entry) throw new Error(`native catalog has no model ${id}`);
  return Object.fromEntries(SERVED_FIELDS.filter((field) => field in entry).map((field) => [field, entry[field]]));
}

// A hand-specified `memoryGuidance` for a synthetic fixture, in the served shape: per lane, the
// per-tier fit thresholds (`{ q4: 20, q8: 36 }`, fidelity order taken from the order given) plus
// any lane-level fields (`{ entryFloorGb: 64 }`). The numbers are the test's own, standing in for
// what the API computes for that fixture.
export function servedGuidance({ mlx = {}, candle = {} } = {}, laneFields = {}) {
  const lane = (thresholds, fields = {}) => ({
    suggestion: Object.entries(thresholds).map(([tier, minHostGb]) => ({ tier, minHostGb })),
    tiers: Object.fromEntries(Object.entries(thresholds).map(([tier, minHostGb]) => [tier, { minHostGb }])),
    blanketGb: null,
    entryFloorGb: null,
    uncalibrated: false,
    resolutionMinHostGb: {},
    ...fields,
  });
  return { mlx: lane(mlx, laneFields.mlx), candle: lane(candle, laneFields.candle) };
}
