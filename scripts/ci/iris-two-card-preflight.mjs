#!/usr/bin/env node
// sc-25690 (epic 25678 acceptance 5): the two-card Iris distributed smoke puts ONE rank on EACH
// card, so both cards must be free before anything is built. This check reads `nvidia-smi` and
// fails fast, naming every card another tenant holds; it never lets the smoke share a card.
// Dispatch-only tooling for windows-candle.yml's `iris-worker-smoke` job. Gates nothing.
//
//   node scripts/ci/iris-two-card-preflight.mjs            (cards from CUDA_VISIBLE_DEVICES)
//
// Env: CUDA_VISIBLE_DEVICES (the card indices, PCI bus order — must name two),
//      IRIS_TWO_CARD_MAX_USED_MIB (default 2048: a free card shows a few hundred MiB of driver /
//      desktop use; a tenant's model shows tens of thousands).

import { spawnSync } from "node:child_process";
import process from "node:process";
import { fileURLToPath } from "node:url";

export const DEFAULT_MAX_USED_MIB = 2048;
export const QUERY = ["--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"];

/** `index, name, memory.used, memory.total` rows (MiB) of the query above. */
export function parseCards(csv) {
  return csv
    .split(/\r?\n/)
    .map((line) => line.split(",").map((field) => field.trim()))
    .filter((fields) => fields.length === 4 && /^\d+$/.test(fields[0]))
    .map(([index, name, used, total]) => ({ index, name, usedMib: Number(used), totalMib: Number(total) }));
}

/**
 * The verdict for the cards `wanted` (index strings): `{ ok, lines, errors }`. A card is free when
 * its used memory is a finite number at or under `maxUsedMib`. Fewer than two wanted cards, a card
 * the driver does not list, an unreadable reading and an occupied card are each an error that
 * names the card.
 */
export function judgeCards(cards, wanted, maxUsedMib = DEFAULT_MAX_USED_MIB) {
  const errors = [];
  const lines = [];
  const unique = [...new Set(wanted)];
  if (unique.length < 2) {
    errors.push(`the two-card smoke needs two distinct cards in CUDA_VISIBLE_DEVICES, got "${wanted.join(",")}"`);
  }
  for (const index of unique) {
    const card = cards.find((candidate) => candidate.index === index);
    if (!card) {
      errors.push(`card ${index} is not listed by nvidia-smi (cards seen: ${cards.map((c) => c.index).join(", ") || "none"})`);
      continue;
    }
    const reading = `card ${card.index} (${card.name}): ${card.usedMib} MiB used of ${card.totalMib} MiB`;
    if (!Number.isFinite(card.usedMib)) {
      errors.push(`${reading} — its used memory could not be read, so it is not known to be free`);
    } else if (card.usedMib > maxUsedMib) {
      errors.push(
        `${reading} — OCCUPIED by another tenant (over the ${maxUsedMib} MiB free-card threshold). ` +
          `The two-card smoke needs card ${card.index} free; nothing was built and no rank was started.`,
      );
    } else {
      lines.push(`${reading} — free`);
    }
  }
  return { ok: errors.length === 0, lines, errors };
}

function main() {
  const wanted = (process.env.CUDA_VISIBLE_DEVICES ?? "")
    .split(",")
    .map((id) => id.trim())
    .filter(Boolean);
  const maxUsedMib = Number(process.env.IRIS_TWO_CARD_MAX_USED_MIB ?? DEFAULT_MAX_USED_MIB);
  const smi = spawnSync("nvidia-smi", QUERY, { encoding: "utf8" });
  if (smi.error || smi.status !== 0) {
    console.error(`::error::nvidia-smi failed (${smi.error?.message ?? `exit ${smi.status}`}); the cards cannot be shown free`);
    process.exit(1);
  }
  const verdict = judgeCards(parseCards(smi.stdout), wanted, maxUsedMib);
  for (const line of verdict.lines) console.log(line);
  for (const error of verdict.errors) console.error(`::error::${error}`);
  process.exit(verdict.ok ? 0 : 1);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
