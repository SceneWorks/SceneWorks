// `node --import ./scripts/fixtures/replay-download-pattern-network.mjs scripts/check-download-patterns.mjs …`
//
// Replays the COMMITTED config/download-pattern-evidence.json back as the huggingface.co API, so
// the script's live/recording modes can be driven end-to-end with no network at all. A re-record
// fed by this replay reproduces the committed artifact byte-for-byte, which is what makes the
// `--write --dry-run` reviewer tool testable: the UP-TO-DATE branch is otherwise only reachable
// against the real Hub.
//
// Why this exists (sc-18854 review): `--write --dry-run` used to exit 1 on a CLEAN tree, because
// the live zero-match/untranslatable verdicts wrote to the same `process.exitCode` as the artifact
// comparison. The runbook documents that exit code as the one signal separating an honest
// re-record from a hand edit, so it was wrong 100% of the time. Pinning that requires running the
// real `main()` — an in-process unit test of a pure function cannot see an exit-code collision.
//
// Env knobs (both optional):
//   REPLAY_EXTRA_FILE=<repo@revision key>  append a bogus file to that key's replayed listing, so
//                                          a re-record no longer matches the committed artifact.
//   REPLAY_ERROR_KEY=<repo@revision key>   answer HTTP 503 for that key, so the live modes see it
//                                          as UNREACHABLE.
//   REPLAY_UNKNOWN_IS_404=1                answer 404 for keys with no recorded listing instead of
//                                          throwing (default is to throw, so a silent catalog/
//                                          evidence drift cannot quietly become a passing test).
import { readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const evidence = JSON.parse(
  readFileSync(path.join(root, "config/download-pattern-evidence.json"), "utf8"),
);
const byKey = new Map((evidence.repos ?? []).map((entry) => [entry.key, entry]));

const jsonResponse = (body) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });

globalThis.fetch = async (input) => {
  const url = new URL(typeof input === "string" ? input : input.url);
  const rest = url.pathname.replace(/^\/api\/models\//, "");
  const marker = rest.indexOf("/revision/");
  const repo = marker === -1 ? rest : rest.slice(0, marker);
  const revision =
    marker === -1 ? null : decodeURIComponent(rest.slice(marker + "/revision/".length));
  const key = `${repo}@${revision ?? "main"}`;
  const entry = byKey.get(key);

  if (process.env.REPLAY_ERROR_KEY === key) {
    return new Response("replayed outage", { status: 503 });
  }
  if (!entry) {
    if (process.env.REPLAY_UNKNOWN_IS_404 === "1") {
      return new Response("not found", { status: 404 });
    }
    throw new Error(`replay has no recorded listing for ${key} — re-record before using it`);
  }
  // Errors were recorded as errors; replay them as the same status so the recorder transcribes
  // an identical `{ key, repo, revision, error }` row.
  if (entry.error) {
    const status = Number.parseInt(String(entry.error).replace(/^HTTP /, ""), 10);
    return new Response("recorded error", { status: Number.isNaN(status) ? 500 : status });
  }

  const files = [...entry.files];
  if (process.env.REPLAY_EXTRA_FILE === entry.key) files.push("zzz-replay-perturbation.bin");
  return jsonResponse({
    sha: entry.resolvedSha,
    id: entry.servedRepo,
    gated: entry.gated,
    siblings: files.map((rfilename) => ({ rfilename })),
  });
};
