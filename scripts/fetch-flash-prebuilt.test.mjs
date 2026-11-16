import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fetchPrebuilt, parseManifest, validateManifest } from "./fetch-flash-prebuilt.mjs";

const target = "x86_64-unknown-linux-gnu";
const sourceSha256 = "a".repeat(64);
const revision = "b".repeat(40);
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
const names = ["LICENSE-MIT", "LICENSE-APACHE", "LICENSE-FLASH-ATTENTION", "LICENSE-CUTLASS.txt", "AUTHORS-FLASH-ATTENTION"];

test("a matching package is cached and a corrupt cached archive fails closed", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "flash-prebuilt-test-"));
  try {
    const input = path.join(root, "input");
    await mkdir(input);
    const archive = Buffer.from("static archive fixture");
    await writeFile(path.join(input, "libflashattention.a"), archive);
    const manifest = [
      "format=1", `inference_revision=${revision}`, `source_sha256=${sourceSha256}`,
      `archive_sha256=${sha256(archive)}`, `target=${target}`, "cuda_compute_cap=80",
      "cuda_version=12.9", "host_compiler_version=11.4.0", "",
    ].join("\n");
    await writeFile(path.join(input, "flash-attn-prebuilt.txt"), manifest);
    for (const name of names) await writeFile(path.join(input, name), "license fixture");
    const bundle = path.join(root, "flash.tar.gz");
    execFileSync("tar", ["-czf", bundle, ...["libflashattention.a", "flash-attn-prebuilt.txt", ...names]], { cwd: input });
    let downloads = 0;
    const downloadAsset = async (_url, destination) => {
      downloads++;
      await writeFile(destination, await readFile(bundle));
      return true;
    };
    const cacheRoot = path.join(root, "cache");
    const trustedArchiveSha256 = sha256(archive);
    const directory = await fetchPrebuilt({ sourceSha256, target, cacheRoot, downloadAsset, trustedArchiveSha256 });
    assert.equal(downloads, 1);
    assert.equal(await fetchPrebuilt({ sourceSha256, target, cacheRoot, downloadAsset, trustedArchiveSha256 }), directory);
    assert.equal(downloads, 1);
    await writeFile(path.join(directory, "libflashattention.a"), "tampered");
    await assert.rejects(fetchPrebuilt({ sourceSha256, target, cacheRoot, downloadAsset, trustedArchiveSha256 }), /checksum mismatch/);
    await writeFile(path.join(directory, "flash-attn-prebuilt.txt"), manifest.replace(trustedArchiveSha256, sha256("tampered")));
    await assert.rejects(fetchPrebuilt({ sourceSha256, target, cacheRoot, downloadAsset, trustedArchiveSha256 }), /checksum mismatch/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("missing release uses source fallback; wrong source and duplicate keys are rejected", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "flash-prebuilt-test-"));
  try {
    assert.equal(await fetchPrebuilt({ sourceSha256, target, cacheRoot: root, downloadAsset: async () => false, trustedArchiveSha256: "d".repeat(64) }), null);
    assert.equal(await fetchPrebuilt({ sourceSha256, target, cacheRoot: root, downloadAsset: async () => { throw new Error("must not download without trust pin"); } }), null);
    assert.throws(() => parseManifest("format=1\nformat=2\n"), /duplicate/);
    assert.throws(() => validateManifest(parseManifest(`format=1\ninference_revision=${revision}\nsource_sha256=${"c".repeat(64)}\narchive_sha256=${"d".repeat(64)}\ntarget=${target}\ncuda_compute_cap=80\ncuda_version=12.9\nhost_compiler_version=11.4.0\n`), { sourceSha256, target, archiveSha256: "d".repeat(64), trustedArchiveSha256: "d".repeat(64) }), /does not match/);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
