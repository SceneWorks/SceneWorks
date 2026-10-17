// sc-24163: opt-in reuse of the terminal MLX job's task-owned hub. The catalog's
// existing resolver selects exact manifest revisions and --download-missing only
// fetches missing snapshots. This step neither downloads nor deletes weights.
import { appendFile, lstat, mkdir, realpath } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// sc-25907: the distilled Turbo sibling is measured by the same terminal MLX job over the same
// task-owned hub, so it is accepted beside base. Each id's snapshots stay its own (the resolver pins
// each manifest revision); this only widens which ids may share the hub.
const QWEN_PERSISTENT_MODELS = new Set(["qwen_image_2_1", "qwen_image_2_1_turbo"]);

export async function bindQwenPersistentCache(env = process.env) {
  const models = (env.MODELS_INPUT ?? "").trim().split(/\s+/).filter(Boolean);
  if (
    env.BACKEND !== "mlx" ||
    models.length === 0 ||
    models.some((model) => !QWEN_PERSISTENT_MODELS.has(model))
  ) {
    throw new Error(
      "qwen_persistent_weights requires backend=mlx and models drawn from qwen_image_2_1 / qwen_image_2_1_turbo",
    );
  }
  if (env.HF_CACHE_INPUT?.trim()) {
    throw new Error("qwen_persistent_weights requires empty hf_cache_roots");
  }
  if (env.ANCHORS_INPUT?.trim()) {
    const anchors = env.ANCHORS_INPUT.split(",").map((anchor) => anchor.trim());
    const outOfScope = (anchor) => {
      const match = /^(qwen_image_2_1(?:_turbo)?):(q4|q8|bf16):mlx$/.exec(anchor);
      return !match || !models.includes(match[1]);
    };
    if (anchors.some(outOfScope)) {
      throw new Error("qwen_persistent_weights accepts only MLX anchors of the selected Qwen-Image 2.1 models");
    }
  }
  if (!env.HOME || !path.isAbsolute(env.HOME) || /[\r\n]/.test(env.HOME) || !env.GITHUB_ENV) {
    throw new Error("the persistent Qwen cache requires an absolute HOME and GITHUB_ENV");
  }
  // A linked HOME is normal on some hosts. Resolve it before constructing the
  // task-owned path, then reject a link that could send the hub to the full SSD.
  const home = await realpath(env.HOME);
  const taskRoot = path.join(home, "sceneworks-rw-weights");
  const root = path.join(taskRoot, "hub");
  // Validate each parent before creating its child, so a stale link to the
  // external cache cannot cause even a new empty directory to land there.
  for (const directory of [taskRoot, root]) {
    try { await mkdir(directory); } catch (error) { if (error.code !== "EEXIST") throw error; }
    const entry = await lstat(directory);
    if (!entry.isDirectory() || entry.isSymbolicLink() || await realpath(directory) !== directory) {
      throw new Error("the persistent Qwen hub must be a physical directory under HOME");
    }
  }
  await appendFile(env.GITHUB_ENV, `HF_CACHE_INPUT=${root}\n`);
  return root;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  bindQwenPersistentCache().then((root) => {
    console.log(`Qwen manifest-pinned snapshot hub: ${root}`);
  }).catch((error) => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
