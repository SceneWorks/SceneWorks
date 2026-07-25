import { appendFile } from "node:fs/promises";
import process from "node:process";
import { pathToFileURL } from "node:url";

const DOCKER_RELEVANT =
  /^(?:docker\/|config\/|docker-compose\.yml$|Cargo\.toml$|Cargo\.lock$|\.github\/workflows\/check\.yml$|scripts\/(?:check-(?:docker-api-runtime|compose-config)|docker-smoke-relevance)(?:\.test)?\.mjs$)/;

export function dockerSmokeDecision(changedFileCount, files) {
  if (!Number.isSafeInteger(changedFileCount) || changedFileCount < 0) {
    throw new Error(`invalid changed-file count: ${changedFileCount}`);
  }
  if (files.length !== changedFileCount) {
    return {
      run: true,
      reason:
        `GitHub reported ${changedFileCount} changed file(s), but pagination returned ` +
        `${files.length}; running conservatively.`,
    };
  }
  const relevant = files.find((file) => DOCKER_RELEVANT.test(file));
  return relevant
    ? { run: true, reason: `Docker-relevant file changed: ${relevant}` }
    : { run: false, reason: "No Docker-relevant files changed." };
}

async function main() {
  const countIndex = process.argv.indexOf("--expected-count");
  if (countIndex < 0) {
    throw new Error("usage: docker-smoke-relevance.mjs --expected-count <count>");
  }
  const changedFileCount = Number(process.argv[countIndex + 1]);
  let input = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) {
    input += chunk;
  }
  const files = input.split(/\r?\n/).filter(Boolean);
  const decision = dockerSmokeDecision(changedFileCount, files);
  console.log(decision.reason);
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(process.env.GITHUB_OUTPUT, `run=${decision.run}\n`, "utf8");
  } else {
    console.log(`run=${decision.run}`);
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
