// Verify the CPU/iOS old-tree source receipt without compiling or loading an engine.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

export function verifyProfileReceipt(receipt, readRevisionFile) {
  const sources = Object.fromEntries(Object.entries(receipt.sources).map(([path, expected]) => {
    const source = readRevisionFile(receipt.inferenceRevision, path);
    assert.equal(createHash('sha256').update(source).digest('hex'), expected, `source receipt changed: ${path}`);
    return [path, source.toString()];
  }));
  const array = (source, key) => {
    const marker = `snapshot.${key},`;
    assert.equal(source.split(marker).length, 2, `source assertion must be unique: ${key}`);
    const tail = source.split(marker)[1];
    return JSON.parse(tail.slice(tail.indexOf('['), tail.indexOf(']') + 1).replace(/,\s*\]/g, ']'));
  };
  const ios = sources['crates/bundles/runtime-ios/src/lib.rs'];
  for (const [key, expected] of Object.entries(receipt.ios)) assert.deepEqual(array(ios, key), expected, `iOS ${key}`);
  const cpu = sources['crates/bundles/runtime-cpu/src/lib.rs'];
  assert.deepEqual(array(cpu, 'text_llm_ids'), receipt.cpu.text_llm_ids);
  for (const shared of receipt.cpuSharedCatalogs) assert.ok(cpu.includes(shared), `CPU must compose ${shared}`);
  return { revision: receipt.inferenceRevision, files: Object.keys(sources).length, iosGenerators: receipt.ios.generator_ids.length };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  assert.equal(process.argv.length, 3, 'usage: node scripts/check-generation-route-profiles.mjs INFERENCE_CHECKOUT');
  const receipt = JSON.parse(readFileSync(new URL('../crates/sceneworks-core/tests/fixtures/generation-route-profiles.json', import.meta.url)));
  console.log(JSON.stringify(verifyProfileReceipt(receipt, (revision, path) =>
    execFileSync('git', ['-C', process.argv[2], 'show', `${revision}:${path}`])), null, 2));
}
