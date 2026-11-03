import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";

/** Hash exact bytes/text. Serialization and domain separation belong to the caller. */
export function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

const HASH_CHUNK_BYTES = 1024 * 1024;

export async function fileSha256(file, { openReadStream = createReadStream, signal, onChunk } = {}) {
  const digest = createHash("sha256");
  signal?.throwIfAborted();
  for await (const chunk of openReadStream(file, { highWaterMark: HASH_CHUNK_BYTES, ...(signal ? { signal } : {}) })) {
    onChunk?.(chunk);
    digest.update(chunk);
    signal?.throwIfAborted();
  }
  return digest.digest("hex");
}
