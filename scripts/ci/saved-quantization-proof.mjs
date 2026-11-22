// Dedicated numerical validation only; ordinary CI may use --prepare for metadata startup checks.
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync, openSync, readSync, closeSync, fstatSync, readdirSync } from "node:fs";
import { resolve, dirname, join, basename, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { lockedInferenceRevision } from "../select-windows-cargo-cache.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const args = process.argv.slice(2);
const get = (name) => {
  const index = args.indexOf(name);
  if (index < 0 || !args[index + 1] || args[index + 1].startsWith("--")) throw new Error(`${name} requires a value`);
  return args[index + 1];
};
const backend = get("--backend");
const root = resolve(get("--root") ?? "");
if (!["candle", "mlx"].includes(backend)) throw new Error("--backend must be candle or mlx");
if (!args.includes("--root") || root === resolve("/")) throw new Error("explicit owned --root required");
if (process.platform === "win32" && !/^D:[\\/]/i.test(root)) throw new Error("Windows proof outputs must be on D:");
const marker = join(root, ".saved-quantization-proof");
const owner = `${process.env.GITHUB_RUN_ID ?? "local-metadata"}:${process.env.GITHUB_RUN_ATTEMPT ?? "1"}`;
if (existsSync(root) && (!existsSync(marker) || readFileSync(marker, "utf8").trim() !== owner)) {
  throw new Error("proof root exists without this run's ownership marker");
}
if (args.includes("--stage-receipts")) {
  if (process.env.GITHUB_ACTIONS !== "true" || !isAbsolute(process.env.GITHUB_WORKSPACE ?? "") || !existsSync(marker)) {
    throw new Error("receipt staging requires an existing owned Actions proof");
  }
  const destination = join(process.env.GITHUB_WORKSPACE, "saved-quantization-receipts", owner.replace(":", "-"));
  mkdirSync(destination, { recursive: true });
  for (const file of readdirSync(root, { withFileTypes: true })) {
    if (file.isFile() && file.name.endsWith(".json")) copyFileSync(join(root, file.name), join(destination, file.name));
  }
  if (backend === "candle" && process.env.RUNNER_TEMP) {
    const receipt = join(process.env.RUNNER_TEMP, `sceneworks-cuda-execution-${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}-saved-quantization`, "resource-receipt.json");
    if (existsSync(receipt)) copyFileSync(receipt, join(destination, "resource-receipt.json"));
  }
  console.log(JSON.stringify({ destination, files: readdirSync(destination) }));
  process.exit(0);
}
mkdirSync(root, { recursive: true });
writeFileSync(marker, `${owner}\n`);
const revision = lockedInferenceRevision(readFileSync(join(repo, "Cargo.lock"), "utf8"));
if (!/^[a-f0-9]{40}$/.test(revision)) throw new Error("invalid paired inference revision");
if (backend === "candle") {
  const model = process.env.SCENEWORKS_BF16_CHECKPOINT;
  if (!model) throw new Error("SCENEWORKS_BF16_CHECKPOINT is required");
  const fd = openSync(model, "r");
  try {
    const prefix = Buffer.alloc(8);
    if (readSync(fd, prefix, 0, 8, 0) !== 8) throw new Error("checkpoint header is truncated");
    const size = Number(prefix.readBigUInt64LE());
    if (size < 2 || size > 32 * 1024 * 1024) throw new Error("invalid safetensors header size");
    const bytes = Buffer.alloc(size);
    if (readSync(fd, bytes, 0, size, 8) !== size) throw new Error("checkpoint header is truncated");
    const tensor = JSON.parse(bytes.toString("utf8"))["model.diffusion_model.blocks.0.attn.wq.weight"];
    if (tensor?.dtype !== "BF16" || tensor.shape?.length !== 2 || tensor.shape.some((n) => !Number.isSafeInteger(n) || n <= 0) || tensor.shape[1] % 32) {
      throw new Error("proof requires the real Krea BF16 attention projection");
    }
    const elements = tensor.shape.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(elements) || elements > 512 * 1024 * 1024 || tensor.data_offsets?.length !== 2 ||
        tensor.data_offsets[1] - tensor.data_offsets[0] !== elements * 2 || 8 + size + tensor.data_offsets[1] > fstatSync(fd).size) {
      throw new Error("projection payload is truncated or outside the bounded proof size");
    }
    writeFileSync(join(root, "source-header.json"), JSON.stringify({ model: basename(model), dtype: tensor.dtype, shape: tensor.shape }, null, 2));
  } finally { closeSync(fd); }
}
const harness = join(root, "harness");
mkdirSync(join(harness, "src"), { recursive: true });
const pathLiteral = JSON.stringify(join(repo, "crates/sceneworks-core").replaceAll("\\", "/"));
const git = `git = "https://github.com/SceneWorks/inference", rev = "${revision}"`;
const lock = readFileSync(join(repo, "Cargo.lock"), "utf8");
const mlxSource = /name = "pmetal-mlx-rs"[\s\S]*?source = "git\+([^"?]+)\?rev=([a-f0-9]{40})#[a-f0-9]{40}"/.exec(lock);
if (!mlxSource) throw new Error("locked MLX Rust source is missing");
writeFileSync(join(harness, "Cargo.toml"), `[package]
name = "saved-quantization-proof"
version = "0.0.0"
edition = "2021"
[workspace]
[features]
default = []
candle = ["dep:candle-gen"]
mlx = ["dep:mlx-gen", "dep:mlx-rs"]
[dependencies]
sceneworks-core = { path = ${pathLiteral} }
gen-core = { package = "sceneworks-gen-core", ${git} }
candle-gen = { ${git}, optional = true, features = ["cuda"] }
mlx-gen = { ${git}, optional = true }
mlx-rs = { package = "pmetal-mlx-rs", git = "${mlxSource[1]}", rev = "${mlxSource[2]}", optional = true }
serde_json = "1"
sha2 = "0.10"
[patch."https://github.com/huggingface/candle"]
candle-core = { ${git} }
candle-nn = { ${git} }
candle-kernels = { ${git} }
candle-metal-kernels = { ${git} }
[profile.release]
opt-level = 1
debug = 0
lto = false
`);
copyFileSync(join(repo, "scripts/ci/saved-quantization-proof.rs"), join(harness, "src/main.rs"));
copyFileSync(join(repo, "Cargo.lock"), join(harness, "Cargo.lock"));
writeFileSync(join(root, "preflight.json"), JSON.stringify({ backend, revision, owner, assertions: ["cold equals original packing", "new warm process hits both formats", "warm quantizer runs are zero", "packed digests match across processes"] }, null, 2));
if (args.includes("--prepare")) {
  console.log(JSON.stringify({ backend, revision, harness, preflight: "passed" }));
} else {
  if (process.env.GITHUB_ACTIONS !== "true") throw new Error("numerical execution belongs in GitHub Actions");
  const binary = process.env.SAVED_QUANTIZATION_BINARY;
  if (!binary || !existsSync(binary)) throw new Error("SAVED_QUANTIZATION_BINARY must name the compiled harness");
  for (const mode of ["cold", "warm"]) {
    const child = spawnSync(binary, [mode, root], { env: process.env, stdio: "inherit" });
    if (child.error) throw child.error;
    if (child.status !== 0) throw new Error(`${mode} proof exited ${child.status}`);
  }
}
