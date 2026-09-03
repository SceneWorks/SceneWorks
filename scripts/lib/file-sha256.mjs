import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

const HASH_CHUNK_BYTES = 1024 * 1024;

export async function fileSha256(file, { openReadStream = createReadStream } = {}) {
  const digest = createHash("sha256");
  for await (const chunk of openReadStream(file, { highWaterMark: HASH_CHUNK_BYTES })) digest.update(chunk);
  return digest.digest("hex");
}
