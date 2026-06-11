#!/usr/bin/env node
// Fetches the `uv` binary for the host target triple and stages it as a Tauri
// sidecar (binaries/uv-<triple>) so the packaged app can bootstrap the Python
// venv on first run (sc-1348). Pinned for reproducibility; cached only when
// the staged sidecar metadata matches this script's pinned version. Wired into
// the tauri.conf.json beforeBuildCommand.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  copyFileSync,
  chmodSync,
  readdirSync,
  rmSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import process from "node:process";

const UV_VERSION = "0.11.15";
const UV_SHA256 = {
  "uv-aarch64-apple-darwin.tar.gz": "7e5b336108f8576eda1939920ca0a805b4a9a3c3d3eb2f6140e38b7092fbe4f3",
  "uv-aarch64-pc-windows-msvc.zip": "9eac2d68f3a66326c3e1fc97ef28bd54f1d13136ec092c2f0a8173ae12aaaf1e",
  "uv-aarch64-unknown-linux-gnu.tar.gz": "21a7dd1a03ea17ac0366887455dab15d215b31dba0870dcd65d3714e22f46c81",
  "uv-aarch64-unknown-linux-musl.tar.gz": "6505075cec3f551fad4fe9026922967ff9c895c9f513c97682b24e7a1c9becd3",
  "uv-arm-unknown-linux-musleabihf.tar.gz": "f9206848d617b7beec37c346624ad961d8d4110606990653ebbfc4c62b1f1741",
  "uv-armv7-unknown-linux-gnueabihf.tar.gz": "eb6a12e3e80e1474c1018edc9541bbe71cdf2248fa17b583dcbcc7bb391ad0c0",
  "uv-armv7-unknown-linux-musleabihf.tar.gz": "a40ee3c41443341846137afc5c7f29be766a9a677bd70c7ff91cbb4273e5383c",
  "uv-i686-pc-windows-msvc.zip": "6a9431f0044a1ff59fd6920f6f982b691acf336b6e26ac8cd40a02b5ab839cd1",
  "uv-i686-unknown-linux-gnu.tar.gz": "557e329e76072b513e47bcd8b50ca4bad07ec87cb325cbfc05e6069847af06c4",
  "uv-i686-unknown-linux-musl.tar.gz": "69490ca5580958cdee3353b54357925913ec0540dc8e09819294b9e5b6d48556",
  "uv-powerpc64le-unknown-linux-gnu.tar.gz": "6be3637ef86cdee3f5fcfbc66681ecbf6d57c6a123398a1bdd09786d65a06016",
  "uv-riscv64gc-unknown-linux-gnu.tar.gz": "a43e22243e3f3b1fb136a0998b730367fe2589ea98ce6cd4f0d7d20b9f77fb5b",
  "uv-riscv64gc-unknown-linux-musl.tar.gz": "2256c9b625d67a55986adda62b09782b5547e28a79fba472e7e93ac3ec0af258",
  "uv-s390x-unknown-linux-gnu.tar.gz": "df2b69ed893ce00e242d8cfe5b9fdc7b7a42d578df487d09aa624563a9801578",
  "uv-x86_64-apple-darwin.tar.gz": "42bca7cc879d117ed7139a0e26de8cab0b6f033ad439a32144f324d1f8580d8c",
  "uv-x86_64-pc-windows-msvc.zip": "04b98d414a9000e25e5e0e7c9f53749e66b790cdaffc582829e6f58c544ee11c",
  "uv-x86_64-unknown-linux-gnu.tar.gz": "b03e572f010bea94a4a52d42671ba72981e12894f71576181a1d26ff68546da7",
  "uv-x86_64-unknown-linux-musl.tar.gz": "200ccf2f351849c5d6698714e7e7eb9ead1e8c097dbdbb43730e1a4e059ceb87",
};

const scriptDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = resolve(scriptDir, ".."); // apps/desktop
const outDir = join(desktopDir, "binaries");

const triple = execFileSync("rustc", ["-vV"], { encoding: "utf8" }).match(
  /host:\s*(\S+)/,
)?.[1];
if (!triple) {
  console.error("fetch-uv: could not determine host target triple");
  process.exit(1);
}
const isWindows = triple.includes("windows");
const exe = isWindows ? ".exe" : "";
const dest = join(outDir, `uv-${triple}${exe}`);
const asset = isWindows ? `uv-${triple}.zip` : `uv-${triple}.tar.gz`;
const expectedSha256 = UV_SHA256[asset];
if (!expectedSha256) {
  console.error(`fetch-uv: no pinned SHA-256 for ${asset}`);
  process.exit(1);
}
const metadataPath = `${dest}.json`;
const expectedMetadata = {
  version: UV_VERSION,
  asset,
  sha256: expectedSha256,
};

function readJson(path) {
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
}

function isCurrentCachedUv(metadata) {
  return (
    metadata?.version === expectedMetadata.version &&
    metadata?.asset === expectedMetadata.asset &&
    metadata?.sha256 === expectedMetadata.sha256
  );
}

if (existsSync(dest)) {
  const metadata = readJson(metadataPath);
  if (isCurrentCachedUv(metadata)) {
    console.log(`fetch-uv: ${dest} already present for uv ${UV_VERSION} (cached)`);
    process.exit(0);
  }
  console.log(`fetch-uv: ${dest} cache metadata is missing or stale; refetching uv ${UV_VERSION}`);
  rmSync(dest, { force: true });
  rmSync(metadataPath, { force: true });
}
const url = `https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${asset}`;
const work = join(tmpdir(), `uv-fetch-${process.pid}`);
mkdirSync(work, { recursive: true });
const archive = join(work, asset);

console.log(`fetch-uv: downloading ${url}`);
execFileSync("curl", ["-fsSL", url, "-o", archive], { stdio: "inherit" });
const actualSha256 = createHash("sha256").update(readFileSync(archive)).digest("hex");
if (actualSha256 !== expectedSha256) {
  rmSync(work, { recursive: true, force: true });
  console.error(
    `fetch-uv: SHA-256 mismatch for ${asset}: expected ${expectedSha256}, got ${actualSha256}`,
  );
  process.exit(1);
}
console.log(`fetch-uv: verified ${asset} SHA-256 ${actualSha256}`);
if (isWindows) {
  // Use PowerShell Expand-Archive for the .zip rather than `tar`: depending on
  // PATH the `tar` on Windows may be GNU tar (no zip support — "does not look
  // like a tar archive"), and bsdtar treats a drive-letter path (`C:\...`) as a
  // remote host ("Cannot connect to C: resolve failed"). Expand-Archive is the
  // reliable built-in for zip on any Windows shell.
  execFileSync(
    "powershell",
    [
      "-NoProfile",
      "-Command",
      `Expand-Archive -LiteralPath '${archive}' -DestinationPath '${work}' -Force`,
    ],
    { stdio: "inherit" },
  );
} else {
  // macOS/Linux ship the .tar.gz; bsdtar/GNU tar both extract it fine.
  execFileSync("tar", ["-xf", asset], { cwd: work, stdio: "inherit" });
}

// Find the extracted uv binary (root for zip, uv-<triple>/ for tar.gz).
function findUv(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findUv(full);
      if (found) return found;
    } else if (entry.name === `uv${exe}`) {
      return full;
    }
  }
  return null;
}
const src = findUv(work);
if (!src) {
  console.error("fetch-uv: uv binary not found in archive");
  process.exit(1);
}
mkdirSync(outDir, { recursive: true });
copyFileSync(src, dest);
if (!isWindows) chmodSync(dest, 0o755);
writeFileSync(metadataPath, `${JSON.stringify(expectedMetadata, null, 2)}\n`);
rmSync(work, { recursive: true, force: true });
console.log(`fetch-uv: staged ${dest}`);
