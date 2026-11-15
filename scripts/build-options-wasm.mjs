// Build the browser validator from the same Rust crate used by API/MCP/worker.
// Generated JS and WASM are checked in so desktop/web builds need no Rust-to-WASM
// toolchain; Check rebuilds them with the pinned tool and rejects drift.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("..", import.meta.url)));
const output = join(root, "apps/web/src/wasm/options-validator");
const target = process.env.CARGO_TARGET_DIR
  ? resolve(process.env.CARGO_TARGET_DIR)
  : join(root, "target");
const check = process.argv.includes("--check");
const tool = process.env.WASM_BINDGEN_BIN || "wasm-bindgen";
const version = execFileSync(tool, ["--version"], { encoding: "utf8" }).trim();
if (version !== "wasm-bindgen 0.2.121") {
  throw new Error(`Expected wasm-bindgen 0.2.121, got ${version}`);
}

execFileSync("cargo", [
  "build", "--locked", "--release", "--target", "wasm32-unknown-unknown",
  "-p", "sceneworks-options-wasm",
], { cwd: root, stdio: "inherit" });

const temporary = check ? mkdtempSync(join(tmpdir(), "sceneworks-options-wasm-")) : null;
const destination = temporary ?? output;
try {
  execFileSync(tool, [
    "--target", "web", "--out-dir", destination,
    "--out-name", "options_validator",
    join(target, "wasm32-unknown-unknown/release/sceneworks_options_wasm.wasm"),
  ], { cwd: root, stdio: "inherit" });
  if (check) {
    const files = ["options_validator.js", "options_validator.d.ts", "options_validator_bg.wasm", "options_validator_bg.wasm.d.ts"];
    for (const name of files) {
      const generated = readFileSync(join(destination, name));
      const committed = readFileSync(join(output, name));
      if (!generated.equals(committed)) throw new Error(`Generated browser validator differs: ${name}`);
    }
    const extra = readdirSync(output).filter((name) => !files.includes(name));
    if (extra.length) throw new Error(`Unexpected browser validator artifacts: ${extra.join(", ")}`);
  }
} finally {
  if (temporary) rmSync(temporary, { recursive: true, force: true });
}
