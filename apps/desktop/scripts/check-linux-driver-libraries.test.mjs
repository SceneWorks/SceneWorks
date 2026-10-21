import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { inspectAppDir } from "./check-linux-driver-libraries.mjs";

const cli = fileURLToPath(new URL("./check-linux-driver-libraries.mjs", import.meta.url));
const desktopDir = fileURLToPath(new URL("../", import.meta.url));

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), "sceneworks-driver-fixture-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const put = (path) => {
    const target = join(root, path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, "fixture");
  };
  put("usr/bin/sceneworks-api");
  return { root, put };
}

test("CUDA toolkit and playback libraries remain allowed", (t) => {
  const { root, put } = fixture(t);
  for (const name of [
    "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9",
    "libnvrtc.so.12", "libnvJitLink.so.12", "libcurand.so.10", "libcufft.so.11",
    "libwebkit2gtk-4.1.so.0", "libgstnvcodec.so",
  ]) put(`usr/lib/${name}`);
  assert.deepEqual(inspectAppDir(root), { passed: true, forbidden: [] });
});

test("a runtime-only artifact passes the CLI", (t) => {
  const { root, put } = fixture(t);
  put("usr/lib/libcudart.so.12");
  put("usr/lib/libcudnn.so.9");
  const result = spawnSync(process.execPath, [cli, "--appdir", root], { encoding: "utf8" });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), { passed: true, forbidden: [] });
});

test("host driver libraries fail across nested and multiarch paths", (t) => {
  const { root, put } = fixture(t);
  const paths = [
    "usr/lib/libcuda.so.1", "usr/lib64/libcuda.so.575.57.08",
    "usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1", "usr/lib/libnvidia-ptxjitcompiler.so.1",
    "usr/lib/libnvcuvid.so.1", "usr/lib/libnvoptix.so.1", "usr/lib/libcudadebugger.so.1",
    "usr/lib/libGLX_nvidia.so.0", "usr/lib/libEGL_nvidia.so.0",
  ];
  for (const path of paths) put(path);
  assert.deepEqual(inspectAppDir(root), { passed: false, forbidden: [...paths].sort() });
  const result = spawnSync(process.execPath, [cli, "--appdir", root], { encoding: "utf8" });
  assert.equal(result.status, 1);
  assert.deepEqual(JSON.parse(result.stdout).forbidden, [...paths].sort());
});

test("a dangling driver symlink fails without following it", { skip: process.platform === "win32" }, (t) => {
  const { root, put } = fixture(t);
  put("usr/lib/libcudart.so.12");
  symlinkSync("/missing/host/libcuda.so.1", join(root, "usr/lib/libcuda.so.1"));
  symlinkSync("../lib", join(root, "usr/lib/loop"));
  assert.deepEqual(inspectAppDir(root), { passed: false, forbidden: ["usr/lib/libcuda.so.1"] });
});

test("missing AppDir and invalid CLI arguments fail instead of passing an empty scan", (t) => {
  const { root } = fixture(t);
  assert.throws(() => inspectAppDir(join(root, "absent")));
  const result = spawnSync(process.execPath, [cli, "--typo", root], { encoding: "utf8" });
  assert.equal(result.status, 2);
  assert.match(result.stderr, /Usage:/);
});

test("production squashfs exclusions remove host drivers and retain CUDA toolkit libraries", {
  skip: process.platform !== "linux",
}, (t) => {
  if (spawnSync("mksquashfs", ["-version"]).error?.code === "ENOENT") {
    t.skip("requires squashfs-tools (installed in both Linux packaging workflows)");
    return;
  }
  const { root, put } = fixture(t);
  const drivers = [
    "usr/lib/libcuda.so.1", "usr/lib/x86_64-linux-gnu/libcuda.so.575.57.08",
    "usr/lib/libcudadebugger.so.1", "usr/lib/libnvcuvid.so.1", "usr/lib/libnvoptix.so.1",
    "usr/lib64/libnvidia-ml.so.1", "usr/lib/libnvidia-ptxjitcompiler.so.1",
    "usr/lib/libEGL_nvidia.so.0", "usr/lib/libGLX_nvidia.so.0",
    "usr/lib/libGLESv1_nvidia.so.1", "usr/lib/libGLESv2_nvidia.so.2",
  ];
  const retained = ["libcudart.so.12", "libcublas.so.12", "libcudnn.so.9", "libnvrtc.so.12", "libwebkit2gtk-4.1.so.0"];
  for (const path of drivers) put(path);
  for (const name of retained) put(`usr/lib/${name}`);
  const scratch = mkdtempSync(join(tmpdir(), "sceneworks-squashfs-check-"));
  t.after(() => rmSync(scratch, { recursive: true, force: true }));
  const image = join(scratch, "artifact.squashfs");
  const packed = spawnSync("mksquashfs", [root, image, "-noappend", "-processors", "2", "-wildcards", "-ef", join(desktopDir, ".appimageignore")], { encoding: "utf8" });
  assert.equal(packed.status, 0, packed.stderr);
  const extracted = join(scratch, "extracted");
  const unpacked = spawnSync("unsquashfs", ["-d", extracted, image], { encoding: "utf8" });
  assert.equal(unpacked.status, 0, unpacked.stderr);
  assert.deepEqual(inspectAppDir(extracted), { passed: true, forbidden: [] });
  for (const path of drivers) assert.equal(existsSync(join(extracted, path)), false, path);
  for (const name of retained) assert.equal(readFileSync(join(extracted, "usr/lib", name), "utf8"), "fixture");
});

test("both Linux packaging lanes inspect finished AppImages before upload", () => {
  for (const [name, upload] of [
    ["desktop-linux.yml", "- name: Upload Linux desktop packages"],
    ["release.yml", "- name: Attach Linux packages + merge updater manifest"],
  ]) {
    const body = readFileSync(join(desktopDir, "../../.github/workflows", name), "utf8");
    const guard = body.indexOf("- name: Reject host NVIDIA drivers in finished AppImages");
    assert.ok(guard > body.indexOf("- name: Build Linux candle packages"), name);
    assert.ok(guard < body.indexOf(upload), name);
    const step = body.slice(guard, body.indexOf("\n      - name:", guard));
    assert.match(step, /"\$GITHUB_WORKSPACE"\/target\/release\/bundle\/appimage\/\*\.AppImage/);
    assert.match(step, /check:linux-drivers -- --appimage "\$appimage"/);
    assert.ok(body.includes("squashfs-tools"), name);
  }
});
