import { readFileSync } from "node:fs";

const contract = JSON.parse(readFileSync(new URL("./starvector-terminal-limit-cases.json", import.meta.url), "utf8"));
const tuples = new Set(["mlx:1b", "mlx:8b", "candle-cuda:1b", "candle-cuda:8b"]);
const scenarios = ["completion", "token", "byte", "wall_time", "queued_cancellation", "in_flight_cancellation"];
const controlKeys = ["cancel_after_create", "cancel_after_progress", "worker_unloaded"];
const die = (message) => { throw new Error(`starvector terminal limit cases: ${message}`); };
const stable = (value) => Array.isArray(value) ? `[${value.map(stable).join(",")}]` : value && typeof value === "object" ? `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stable(value[key])}`).join(",")}}` : JSON.stringify(value);

function tierBudget(tier) {
  const budget = contract.shipping_detail_budgets?.[tier];
  if (!budget) die(`unsupported tier ${tier}`);
  return structuredClone(budget);
}

function expectedRecords(tier, tuple) {
  if (contract.schema_version !== 1 || !Array.isArray(contract.scenarios) || contract.scenarios.length !== scenarios.length) die("shared scenario contract is malformed");
  if (stable(contract.sampling) !== stable({ temperature: 0, topP: 1, topK: 1, repetitionPenalty: 1, seed: 7 })) die("shared deterministic sampling contract drifted");
  return contract.scenarios.map((template, case_index) => {
    if (template.scenario !== scenarios[case_index]) die(`scenario ${case_index} ordering drifted`);
    const detailBudget = tierBudget(tier);
    Object.assign(detailBudget, template.detail_budget_overrides);
    const record = {
      case_id: `limit-${tuple}-${template.scenario}`,
      case_index,
      source_case_index: template.source_case_index,
      scenario: template.scenario,
      sampling: structuredClone(contract.sampling),
      detailBudget,
    };
    for (const key of controlKeys) if (template[key] !== undefined) record[key] = template[key];
    return record;
  });
}

export function materializeLimitCases(tuple) {
  if (!tuples.has(tuple)) die(`unsupported tuple ${tuple}`);
  return expectedRecords(tuple.split(":")[1], tuple);
}

export function validateLimitCases(records, tier) {
  if (!Array.isArray(records) || records.length !== scenarios.length) die("exactly six executable records are required");
  const expected = expectedRecords(tier, "<tuple>");
  records.forEach((record, index) => {
    const wanted = expected[index];
    if (!record || Object.hasOwn(record, "finish_reason")) die(`record ${index} is a label-only legacy scenario`);
    if (typeof record.case_id !== "string" || !record.case_id.endsWith(`-${wanted.scenario}`)) die(`record ${index} case id does not bind its scenario`);
    for (const key of ["case_index", "source_case_index", "scenario"]) if (record[key] !== wanted[key]) die(`record ${index} has invalid ${key}`);
    if (stable(record.sampling) !== stable(wanted.sampling)) die(`record ${index} sampling is not deterministic`);
    if (stable(record.detailBudget) !== stable(wanted.detailBudget)) die(`record ${index} Detail budget does not execute ${wanted.scenario}`);
    for (const key of controlKeys) if (record[key] !== wanted[key]) die(`record ${index} has invalid ${key}`);
  });
  return records;
}

export const LIMIT_CASE_SCENARIOS = Object.freeze([...scenarios]);
