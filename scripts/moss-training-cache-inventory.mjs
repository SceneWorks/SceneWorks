// Cheap pre-build presence check only. Native preflight still authenticates every
// file's bytes and headers and owns device/memory admission before model loading.
import { readFileSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { stripJsoncComments } from './lib/jsonc.mjs';

const requiredPurposes = new Set(['moss_dmd_unmerged_bf16', 'moss_dmd_full_fp32']);
const nonempty = value => typeof value === 'string' && value.trim() ? value.trim() : null;

export function mossTrainingHubRoot({ root, env = process.env, home = homedir() }) {
  // Match hf_home::{ensure_default_huggingface_home,huggingface_hub_cache_dir}.
  const direct = nonempty(env.HF_HUB_CACHE) ?? nonempty(env.HUGGINGFACE_HUB_CACHE);
  const cache = direct ?? path.join(nonempty(env.HF_HOME) ?? path.join(home, '.cache', 'huggingface'), 'hub');
  return path.resolve(root, cache);
}

export function requiredMossTrainingCacheFiles(catalog, phase) {
  if (!['capture', 'cross-load', 'bounded-product'].includes(phase)) throw new Error('MOSS cache inventory: unknown execution phase');
  const selected = new Map();
  for (const id of (phase === 'bounded-product' ? ['moss_image_base'] : ['moss_image_base', 'moss_image_sft'])) {
    const entry = catalog.models?.find(model => model.id === id);
    if (!entry || !Array.isArray(entry.downloads)) throw new Error(`MOSS cache inventory: missing production entry ${id}`);
    // Match selected_components' ordinary download rows. Capture also exercises
    // both declared DMD purposes; cross-load consumes ordinary generation only.
    const rows = entry.downloads.filter(row => !row.componentId?.startsWith('moss-dmd-'));
    if (phase === 'capture' && id === 'moss_image_sft') {
      rows.push(...(entry.conditionalComponents ?? []).filter(row =>
        row.requiredFor?.some(purpose => requiredPurposes.has(purpose))));
    }
    for (const row of rows) {
      if (row.provider !== 'huggingface' || !/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(row.repo ?? '') ||
          !/^[a-f0-9]{40}$/.test(row.revision ?? '') || !Array.isArray(row.files) || row.files.length === 0) {
        throw new Error('MOSS cache inventory: production component must name exact Hub files and revision');
      }
      for (const file of row.files) {
        if (typeof file !== 'string' || !file || file.includes('\\') || file.includes(':') ||
            file.startsWith('/') || file.split('/').some(part => !part || part === '.' || part === '..') || /[*?\[]/.test(file)) {
          throw new Error('MOSS cache inventory: expected literal snapshot-relative production filename');
        }
        const item = { repository: row.repo, revision: row.revision, file };
        selected.set(JSON.stringify(item), item);
      }
    }
  }
  return [...selected.values()].sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b)));
}

function present(file, directory) {
  try {
    // Follow normal Hub snapshot symlinks, as the native inventory does.
    const value = statSync(file);
    return directory ? value.isDirectory() : value.isFile();
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return false;
    throw error;
  }
}

export function inventoryMossTrainingCache({ root, phase, env = process.env, home = homedir() }) {
  const catalog = JSON.parse(stripJsoncComments(readFileSync(path.join(root, 'config/manifests/builtin.models.jsonc'), 'utf8')));
  const hubRoot = mossTrainingHubRoot({ root, env, home });
  const requiredFiles = requiredMossTrainingCacheFiles(catalog, phase).map(item => ({ ...item,
    path: path.join(hubRoot, `models--${item.repository.replace('/', '--')}`, 'snapshots', item.revision, item.file) }));
  const missing = [], snapshots = new Set();
  for (const item of requiredFiles) {
    const snapshot = path.join(hubRoot, `models--${item.repository.replace('/', '--')}`, 'snapshots', item.revision);
    if (!snapshots.has(snapshot)) {
      snapshots.add(snapshot);
      if (!present(snapshot, true)) missing.push({ repository: item.repository, revision: item.revision, path: snapshot, expected: 'directory' });
    }
    if (!present(item.path, false)) missing.push({ ...item, expected: 'file' });
  }
  return { schemaVersion: 1, kind: 'moss-training-cache-presence', phase, hubRoot,
    verification: 'stat-only; native authentication remains required', ready: missing.length === 0, requiredFiles, missing };
}
