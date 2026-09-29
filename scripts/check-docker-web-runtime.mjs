import { execFileSync } from "node:child_process";
import { setTimeout as sleep } from "node:timers/promises";
import process from "node:process";

const image = `sceneworks-web-smoke-${process.pid}`;
const container = `sceneworks-web-smoke-${process.pid}`;
const docker = (args, options = {}) => execFileSync("docker", args, { stdio: "inherit", ...options });

try {
  docker(["build", "-f", "docker/web.Dockerfile", "-t", image, "."]);
  docker(["run", "-d", "--name", container, "-v", `${process.cwd()}/apps/web:/app`,
    "-v", "/app/node_modules", image], { stdio: "ignore" });
  docker(["exec", container, "sh", "-ec",
    'test "$(id -u)" = 1000; test "$(id -g)" = 1000; test "$HOME" = /home/node; ' +
    'test -r /app/package.json; touch /app/node_modules/.docker-smoke; ' +
    'touch "$HOME/.docker-smoke"; rm /app/node_modules/.docker-smoke "$HOME/.docker-smoke"']);

  let healthy = false;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    try {
      docker(["exec", container, "node", "-e",
        "fetch('http://127.0.0.1:5173').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"],
      { stdio: "ignore" });
      healthy = true;
      break;
    } catch {
      await sleep(500);
    }
  }
  if (!healthy) {
    docker(["logs", container]);
    throw new Error("web dev server did not become healthy within 10 seconds");
  }
  console.log("Web image nonroot identity, bind mount, cache writes, and HTTP health passed.");
} finally {
  try { docker(["rm", "-f", "-v", container], { stdio: "ignore" }); } catch { /* container may not exist */ }
}
