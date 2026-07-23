import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import Ajv from "ajv";

const scriptsDir = dirname(fileURLToPath(import.meta.url));
const desktopDir = join(scriptsDir, "..");

function readJson(relativePath) {
  return JSON.parse(readFileSync(join(desktopDir, relativePath), "utf8"));
}

function mergeConfig(base, overlay) {
  if (
    base !== null &&
    overlay !== null &&
    typeof base === "object" &&
    typeof overlay === "object" &&
    !Array.isArray(base) &&
    !Array.isArray(overlay)
  ) {
    const merged = { ...base };
    for (const [key, value] of Object.entries(overlay)) {
      merged[key] = key in base ? mergeConfig(base[key], value) : value;
    }
    return merged;
  }

  return overlay;
}

const baseConfig = readJson("tauri.conf.json");
const linuxOverlay = readJson("tauri.linux.conf.json");
const macosOverlay = readJson("tauri.macos.conf.json");
const packageJson = readJson("package.json");
const tauriSchema = readJson("node_modules/@tauri-apps/cli/config.schema.json");
const desktopReadme = readFileSync(join(desktopDir, "README.md"), "utf8");
const webStyles = readFileSync(join(desktopDir, "..", "web", "src", "styles.css"), "utf8");

test("the Linux overlay is valid against the locked Tauri v2 schema", () => {
  // Tauri's schema intentionally escapes `:` in a character class. Node 24's
  // Unicode RegExp mode rejects that harmless draft-07 pattern, while Tauri's
  // Rust schema parser accepts it.
  const ajv = new Ajv({
    allErrors: true,
    strict: false,
    unicodeRegExp: false,
    validateFormats: false,
  });
  const validate = ajv.compile(tauriSchema);
  const linuxConfig = mergeConfig(baseConfig, linuxOverlay);

  assert.equal(
    validate(linuxConfig),
    true,
    ajv.errorsText(validate.errors, { separator: "\n" }),
  );
});

test("the default Linux build produces only AppImage and deb bundles", () => {
  const linuxConfig = mergeConfig(baseConfig, linuxOverlay);

  assert.deepEqual(linuxConfig.bundle.targets, ["appimage", "deb"]);
  assert.equal(packageJson.scripts.build, "tauri build");
  assert.equal(
    packageJson.scripts["build:linux"],
    "tauri build --bundles appimage,deb",
  );
});

test("the deb declares its WebKitGTK, GTK, and H.264 playback dependencies", () => {
  assert.deepEqual(linuxOverlay.bundle.linux.deb.depends, [
    "libwebkit2gtk-4.1-0",
    "libgtk-3-0",
    "gstreamer1.0-libav",
  ]);
});

test("Linux bundles carry product metadata, category, and PNG icons", () => {
  const linuxBundle = mergeConfig(baseConfig, linuxOverlay).bundle;

  assert.equal(baseConfig.productName, "SceneWorks");
  assert.equal(baseConfig.identifier, "net.trefry.sceneworks");
  assert.match(baseConfig.version, /^\d+\.\d+\.\d+$/);
  assert.equal(linuxBundle.category, "GraphicsAndDesign");
  assert.equal(linuxBundle.publisher, "SceneWorks");
  assert.equal(linuxBundle.homepage, "https://github.com/SceneWorks/SceneWorks");
  assert.equal(linuxBundle.license, "AGPL-3.0-or-later");
  assert.match(linuxBundle.shortDescription, /image and video/i);
  assert.match(linuxBundle.longDescription, /locally on an NVIDIA GPU/i);
  assert.ok(linuxBundle.icon.length >= 3);

  for (const icon of linuxBundle.icon) {
    assert.equal(extname(icon), ".png", `${icon} must be a Linux PNG icon`);
    assert.equal(existsSync(join(desktopDir, icon)), true, `${icon} must exist`);
  }
});

test("Linux bundles carry the media framework needed for H.264 playback", () => {
  assert.equal(
    linuxOverlay.bundle.linux.appimage.bundleMediaFramework,
    true,
  );
  assert.match(desktopReadme, /AppImage carries its GTK\/WebKitGTK runtime/);
  assert.match(desktopReadme, /bundleMediaFramework: true/);
  assert.match(desktopReadme, /gstreamer1\.0-libav/);
  assert.match(desktopReadme, /H\.264/);
});

test("the WebKitGTK compatibility contract remains configured", () => {
  assert.equal(baseConfig.app.windows[0].dragDropEnabled, false);
  assert.match(desktopReadme, /WEBKIT_DISABLE_DMABUF_RENDERER=1/);
  assert.match(desktopReadme, /SCENEWORKS_WEBKIT_DMABUF=1/);

  const unprefixedBlur = /(?<!-webkit-)backdrop-filter:\s*([^;]+);/g;
  for (const match of webStyles.matchAll(unprefixedBlur)) {
    const declarationStart = match.index;
    const precedingRule = webStyles.slice(
      Math.max(0, declarationStart - 100),
      declarationStart,
    );
    assert.match(
      precedingRule,
      new RegExp(
        `-webkit-backdrop-filter:\\s*${match[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*;`,
      ),
      `backdrop-filter at byte ${declarationStart} needs a matching WebKit prefix`,
    );
  }
});

test("the Linux overlay does not regress macOS or Windows bundle config", () => {
  const linuxConfig = mergeConfig(baseConfig, linuxOverlay);
  const macosConfig = mergeConfig(baseConfig, macosOverlay);

  assert.deepEqual(linuxConfig.bundle.macOS, baseConfig.bundle.macOS);
  assert.deepEqual(linuxConfig.bundle.windows, baseConfig.bundle.windows);
  assert.equal(macosConfig.bundle.targets, "all");
  assert.deepEqual(macosConfig.bundle.windows, baseConfig.bundle.windows);
  assert.equal(macosConfig.bundle.macOS.minimumSystemVersion, "26.2");
  assert.equal(
    macosConfig.bundle.windows.webviewInstallMode.type,
    "downloadBootstrapper",
  );
});
