import assert from "node:assert/strict"; import { readFileSync } from "node:fs"; import test from "node:test"; import { validatePlan } from "./starvector-terminal-campaign.mjs";
const plan=JSON.parse(readFileSync("release/starvector-terminal-campaign-v1.json"));
test("terminal campaign is fixed, serial, and fail closed",()=>assert.match(validatePlan(plan),/^[0-9a-f]{64}$/));
test("terminal campaign rejects count and metric drift",()=>{const bad=structuredClone(plan);bad.counts.hostile_sanitizer=199;assert.throws(()=>validatePlan(bad),/count/);bad.counts.hostile_sanitizer=200;bad.metrics.alexnet_sha256="0".repeat(64);assert.throws(()=>validatePlan(bad),/metric/);});
