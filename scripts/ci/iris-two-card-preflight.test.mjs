import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { judgeCards, parseCards } from "./iris-two-card-preflight.mjs";

// The campaign's box: card 0 held by another tenant, card 1 free.
const TENANT = "0, NVIDIA RTX PRO 6000 Blackwell, 95262, 97887\r\n1, NVIDIA RTX PRO 6000 Blackwell, 412, 97887\r\n";
const FREE = "0, NVIDIA RTX PRO 6000 Blackwell, 388, 97887\n1, NVIDIA RTX PRO 6000 Blackwell, 412, 97887\n";

test("nvidia-smi rows parse to cards in MiB", () => {
  assert.deepEqual(parseCards(TENANT), [
    { index: "0", name: "NVIDIA RTX PRO 6000 Blackwell", usedMib: 95262, totalMib: 97887 },
    { index: "1", name: "NVIDIA RTX PRO 6000 Blackwell", usedMib: 412, totalMib: 97887 },
  ]);
  assert.deepEqual(parseCards("index, name, memory.used [MiB], memory.total [MiB]\n\n"), []);
});

test("two free cards pass, and each is reported", () => {
  const verdict = judgeCards(parseCards(FREE), ["0", "1"]);
  assert.equal(verdict.ok, true);
  assert.equal(verdict.lines.length, 2);
  assert.deepEqual(verdict.errors, []);
});

test("a tenant-occupied card fails the check and is named", () => {
  const verdict = judgeCards(parseCards(TENANT), ["0", "1"]);
  assert.equal(verdict.ok, false);
  assert.equal(verdict.errors.length, 1);
  assert.match(verdict.errors[0], /^card 0 \(NVIDIA RTX PRO 6000 Blackwell\): 95262 MiB used of 97887 MiB — OCCUPIED/);
  assert.match(verdict.errors[0], /needs card 0 free/);
  // The free card is still reported free, so the log shows which one is the problem.
  assert.deepEqual(verdict.lines, ["card 1 (NVIDIA RTX PRO 6000 Blackwell): 412 MiB used of 97887 MiB — free"]);
});

test("the threshold is inclusive and configurable", () => {
  const cards = parseCards("0, A, 2048, 97887\n1, B, 2049, 97887\n");
  const verdict = judgeCards(cards, ["0", "1"]);
  assert.equal(verdict.ok, false);
  assert.match(verdict.errors[0], /^card 1 /);
  assert.equal(judgeCards(cards, ["0", "1"], 4096).ok, true);
});

test("one card, a repeated card, an unlisted card and an unreadable reading are refused", () => {
  const cards = parseCards(FREE);
  assert.match(judgeCards(cards, ["1"]).errors[0], /needs two distinct cards/);
  assert.match(judgeCards(cards, ["1", "1"]).errors[0], /needs two distinct cards/);
  assert.match(judgeCards(cards, ["1", "2"]).errors[0], /card 2 is not listed by nvidia-smi \(cards seen: 0, 1\)/);
  const unread = judgeCards(parseCards("0, A, [N/A], 97887\n1, B, 400, 97887\n"), ["0", "1"]);
  assert.equal(unread.ok, false);
  assert.match(unread.errors[0], /card 0 .*could not be read/);
});

test("the workflow runs the check before any build, only for the two-card dispatch, with the default card unchanged", () => {
  const workflow = readFileSync(new URL("../../.github/workflows/windows-candle.yml", import.meta.url), "utf8");
  const job = workflow.slice(workflow.indexOf("\n  iris-worker-smoke:\n"));
  const end = job.indexOf("\n  imported-nvfp4-worker-smoke:");
  const body = end === -1 ? job : job.slice(0, end);
  assert.match(workflow, /iris_smoke_cards:[\s\S]*?options:\n\s+- "1"\n\s+- "0,1"\n\s+default: "1"/);
  assert.match(body, /CUDA_VISIBLE_DEVICES: \$\{\{ inputs\.iris_smoke_cards \|\| '1' \}\}/);
  const check = body.indexOf("node scripts/ci/iris-two-card-preflight.mjs");
  assert.ok(check > 0, "the job runs the card check");
  for (const later of ["cargo fetch --locked", "cargo test -p sceneworks-worker"]) {
    assert.ok(body.indexOf(later) > check, `the card check runs before \`${later}\``);
  }
  // The check and the smoke run only for the two-card dispatch; the one-card steps never then.
  assert.equal(body.match(/if: \$\{\{ inputs\.iris_smoke_cards == '0,1' \}\}/g)?.length, 2);
  assert.equal(body.match(/inputs\.iris_smoke_cards != '0,1' &&/g)?.length, 3);
  assert.equal(body.match(/iris_two_card_distributed_smoke -- --ignored/g)?.length, 1);
  // The default training filter cannot select it on a one-card run.
  assert.ok(!"iris_two_card_distributed_smoke".includes("iris_training_gpu_smoke"));
  assert.ok(!"iris_two_card_distributed_smoke".includes("iris_worker_gpu_smoke"));
});
