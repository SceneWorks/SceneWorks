// CPU-only interpreter selection. No tool-cache, host settings, credentials, or device work.
// Official immutable release: https://github.com/astral-sh/python-build-standalone/releases/tag/20261009
// Archive contract: https://gregoryszorc.com/docs/python-build-standalone/main/distributions.html
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFileSync, chmodSync, existsSync, lstatSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

const SOURCE = fileURLToPath(import.meta.url), ROOT = path.resolve(path.dirname(SOURCE), '..');
const MAX_ARCHIVE = 64 * 1024 * 1024, MAX_EXPANDED = 128 * 1024 * 1024;
export const artifacts = Object.freeze({
  'darwin-arm64': { version: '3.12.15', bytes: 25030095, sha256: '233f8e15255b3e1d1fd47ea5dd5230dfed803f371138e896e54d98c60ae19f6f', url: 'https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-aarch64-apple-darwin-install_only_stripped.tar.gz', executable: 'python/bin/python3.12' },
  'win32-x64': { version: '3.12.15', bytes: 22011208, sha256: '11ed9f8a6ac64fb1b5a55e40e305a6744a3f77a70863c8f8197a6e4349abc64a', url: 'https://github.com/astral-sh/python-build-standalone/releases/download/20261009/cpython-3.12.15%2B20261009-x86_64-pc-windows-msvc-install_only_stripped.tar.gz', executable: 'python/python.exe' },
});
const check = (ok, why) => { if (!ok) throw new Error(`H3 owned Python: ${why}`); };
export const sha256 = b => createHash('sha256').update(b).digest('hex');
const hashFile = p => sha256(readFileSync(p));
export function noSymlinkAncestors(p) {
  for (let current = path.resolve(p);;) {
    if (existsSync(current)) check(!lstatSync(current).isSymbolicLink(), `symlink ancestor: ${current}`);
    const parent = path.dirname(current); if (parent === current) break; current = parent;
  }
}
export function ownedRoot(runDir) {
  check(typeof runDir === 'string' && !/[\r\n\0]/.test(runDir) && path.isAbsolute(runDir) && path.resolve(runDir) === runDir && runDir !== path.parse(runDir).root, 'absolute normalized external run directory required');
  check(runDir !== ROOT && !runDir.startsWith(ROOT + path.sep), 'run directory inside checkout');
  noSymlinkAncestors(runDir); mkdirSync(runDir, { recursive: true, mode: 0o700 }); return runDir;
}
export function candidates(platform = process.platform, env = process.env) {
  const fixed = platform === 'darwin' ? ['/opt/homebrew/bin/python3.12', '/opt/homebrew/opt/python@3.12/bin/python3.12', '/usr/local/bin/python3.12', '/usr/local/opt/python@3.12/bin/python3.12', '/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12'] : platform === 'win32' ? [path.join(env.LOCALAPPDATA || 'C:\\Users\\invalid', 'Programs', 'Python', 'Python312', 'python.exe'), 'C:\\Python312\\python.exe'] : [];
  const names = platform === 'win32' ? ['python3.12.exe', 'python.exe'] : ['python3.12'];
  return [...new Set([env.S9_GUARD_PYTHON, ...fixed, ...(env.PATH || '').split(path.delimiter).filter(p => path.isAbsolute(p)).flatMap(p => names.map(n => path.join(p, n)))].filter(Boolean))];
}
export function verifyInterpreter(candidate, { platform = process.platform, arch = process.arch, execute = execFileSync } = {}) {
  check(path.isAbsolute(candidate) && !/[\r\n\0]/.test(candidate), 'invalid interpreter path');
  const resolved = realpathSync(candidate); check(!/[\r\n\0]/.test(resolved), 'invalid resolved interpreter path'); check(lstatSync(resolved).isFile(), 'interpreter is not a regular file');
  const before = hashFile(resolved);
  const probe = "import sys,json,platform,ssl,ctypes,venv; print(json.dumps({'implementation':sys.implementation.name,'version':list(sys.version_info[:3]),'executable':sys.executable,'platform':sys.platform,'machine':platform.machine()}))";
  const fact = JSON.parse(execute(resolved, ['-I', '-S', '-B', '-c', probe], { encoding: 'utf8', timeout: 15000, maxBuffer: 65536 }));
  check(fact.implementation === 'cpython' && fact.version?.[0] === 3 && fact.version?.[1] === 12 && Number.isInteger(fact.version[2]), 'exact CPython 3.12 required');
  check(fact.platform === platform && (arch === 'arm64' ? /^(arm64|aarch64)$/i : /^(amd64|x86_64)$/i).test(fact.machine), 'interpreter host architecture/platform differs');
  check(realpathSync(fact.executable) === resolved && hashFile(resolved) === before, 'interpreter changed during verification');
  return { path: resolved, sha256: before, version: fact.version.join('.'), platform, arch, implementation: fact.implementation };
}
// Strict subset of ustar used by these authenticated install_only archives: regular files,
// directories and internal relative symlinks. Reject metadata extensions, hardlinks/devices,
// duplicate entries and link ancestors before any extraction. Do not delegate unsafe tar flags.
export function parseArchive(bytes) {
  check(bytes.length <= MAX_ARCHIVE, 'archive byte bound exceeded');
  const tar = gunzipSync(bytes, { maxOutputLength: MAX_EXPANDED }), entries = [], names = new Set();
  const str = (h, a, n) => h.subarray(a, a+n).toString('utf8').split('\0', 1)[0];
  const oct = (h, a, n) => { const s = str(h,a,n).trim(); check(/^[0-7]*$/.test(s), 'invalid tar integer'); return parseInt(s || '0',8); };
  let offset = 0, ended = false;
  while (offset + 512 <= tar.length) {
    const h = tar.subarray(offset,offset+512); offset += 512;
    if (h.every(b => b === 0)) { ended = true; check(tar.subarray(offset).every(b => b === 0), 'data after archive terminator'); break; }
    let sum = 0; for (let i=0;i<512;i++) sum += i>=148 && i<156 ? 32 : h[i];
    check(sum === oct(h,148,8), 'tar header checksum differs');
    const prefix = str(h,345,155), name = (prefix ? prefix+'/' : '')+str(h,0,100), type = str(h,156,1) || '0';
    check(name.startsWith('python/') && !name.includes('\\') && !name.includes(':') && !/[\x00-\x1f\x7f]/.test(name) && name.split('/').every(x => x !== '..' && x !== '.') && path.posix.normalize(name).replace(/\/$/, '') === name.replace(/\/$/, ''), 'unsafe archive path');
    check(['0','2','5'].includes(type) && !names.has(name), 'unsupported or duplicate tar entry'); names.add(name);
    const size = oct(h,124,12); check(Number.isSafeInteger(size) && offset+size<=tar.length && entries.length<10000, 'archive extent/count invalid');
    const link = str(h,157,100);
    if (type === '2') { const target = path.posix.normalize(path.posix.join(path.posix.dirname(name),link)); check(link && !link.startsWith('/') && !link.includes('\\') && !link.includes(':') && !/[\x00-\x1f\x7f]/.test(link) && target.startsWith('python/'), 'unsafe symlink target'); }
    check(type==='0' || size===0, 'non-file payload');
    entries.push({ name, type, link, mode: oct(h,100,8) & 0o777, data: tar.subarray(offset,offset+size) }); offset += Math.ceil(size/512)*512;
  }
  check(ended && entries.length, 'truncated or empty archive');
  for (const e of entries) for (const link of entries.filter(x => x.type==='2')) check(!e.name.startsWith(link.name+'/'), 'symlink traversal entry');
  for (const e of entries.filter(x => x.type==='2')) check(names.has(path.posix.normalize(path.posix.join(path.posix.dirname(e.name),e.link))), 'dangling archive symlink');
  return entries;
}
export function extractArchive(bytes, destination) {
  const entries = parseArchive(bytes); noSymlinkAncestors(destination);
  check(!existsSync(destination), 'extraction destination already exists'); mkdirSync(destination,{mode:0o700});
  for (const e of entries.filter(x => x.type!=='2')) { const target=path.join(destination,...e.name.split('/')); mkdirSync(path.dirname(target),{recursive:true,mode:0o700}); if(e.type==='5') mkdirSync(target,{recursive:true}); else writeFileSync(target,e.data,{flag:'wx',mode:e.mode}); }
  for (const e of entries.filter(x => x.type==='2')) symlinkSync(e.link,path.join(destination,...e.name.split('/')));
  return sha256(Buffer.from(entries.map(e => JSON.stringify([e.name,e.type,e.link,e.mode,sha256(e.data)])).join('\n')));
}
export function treeDigest(root, freeze = false) {
  noSymlinkAncestors(root); const rows=[];
  function walk(dir) { for(const name of readdirSync(dir).sort()) { const p=path.join(dir,name),s=lstatSync(p),relative=path.relative(root,p);
    if(s.isSymbolicLink()) { const resolved=realpathSync(p);check(resolved.startsWith(root+path.sep),'installed symlink escapes');rows.push([relative,'link',path.relative(root,resolved)]); }
    else if(s.isDirectory()) { walk(p);if(freeze) chmodSync(p,0o555); }
    else { check(s.isFile(),'unsupported installed entry');if(freeze) chmodSync(p,s.mode&0o111 ? 0o555 : 0o444);rows.push([relative,'file',lstatSync(p).mode&0o777,hashFile(p)]); }
  } }
  walk(root);if(freeze) chmodSync(root,0o555);return sha256(Buffer.from(JSON.stringify(rows)));
}
async function download(artifact) {
  const response=await fetch(artifact.url,{signal:AbortSignal.timeout(120000)}); check(response.ok && response.body, 'pinned download failed');
  const parts=[];let count=0;for await(const part of response.body) { count+=part.length;check(count<=MAX_ARCHIVE,'download byte bound exceeded');parts.push(part); }
  const bytes=Buffer.concat(parts);check(bytes.length===artifact.bytes && sha256(bytes)===artifact.sha256,'pinned archive SHA/length differs');return bytes;
}
export async function bootstrap(runDir, { platform = process.platform, arch = process.arch, discover = candidates, verify = verifyInterpreter, fetchArchive = download } = {}) {
  ownedRoot(runDir); const tools=path.join(runDir,'tools');noSymlinkAncestors(tools);mkdirSync(tools,{recursive:true,mode:0o700});
  const sourceSha256=hashFile(SOURCE), receiptPath=path.join(tools,'python-3.12-attestation.json'), marker=path.join(tools,'.h3-owned-python.json');
  const context={schemaVersion:1,runDir,sourceSha256,platform,arch};
  if(existsSync(marker)) { check(!lstatSync(marker).isSymbolicLink(),'tools marker symlink');check(JSON.stringify(JSON.parse(readFileSync(marker)))===JSON.stringify(context),'owned tools context differs'); } else writeFileSync(marker,JSON.stringify(context),{flag:'wx',mode:0o600});
  if(existsSync(receiptPath)) { check(!lstatSync(receiptPath).isSymbolicLink(),'attestation symlink');const prior=JSON.parse(readFileSync(receiptPath));check(JSON.stringify(prior.context)===JSON.stringify(context),'attestation context differs');check(['existing_verified','owned_relocatable'].includes(prior.provenance?.kind),'attestation provenance invalid');if(prior.provenance.kind==='existing_verified') check(!prior.interpreter.path.startsWith(tools+path.sep),'owned install cannot masquerade as existing');const actual=verify(prior.interpreter.path,{platform,arch});check(JSON.stringify(actual)===JSON.stringify(prior.interpreter),'attested interpreter mutation');if(prior.provenance.kind==='owned_relocatable') { check(prior.provenance.installed===path.join(tools,'cpython-3.12.15-pbs-20261009'),'installed root differs');check(JSON.stringify(prior.provenance.artifact)===JSON.stringify(artifacts[`${platform}-${arch}`]),'archive provenance differs');check(treeDigest(prior.provenance.installed)===prior.provenance.installedTreeSha256,'installed runtime tree mutation'); }return {...prior,receiptPath}; }
  const failures=[];let interpreter,provenance={kind:'existing_verified'};
  for(const candidate of discover(platform)) { try { interpreter=verify(candidate,{platform,arch});check(!interpreter.path.startsWith(tools+path.sep),'unattested owned candidate');break; } catch(error) { failures.push({path:candidate,error:String(error.message)}); } }
  if(!interpreter) {
    const artifact=artifacts[`${platform}-${arch}`];check(artifact,'no pinned relocatable distribution for actual host');const bytes=await fetchArchive(artifact);check(bytes.length===artifact.bytes && sha256(bytes)===artifact.sha256,'pinned archive SHA/length differs');
    const staging=mkdtempSync(path.join(tools,'.python-stage-')),installed=path.join(tools,'cpython-3.12.15-pbs-20261009');
    try { const unpack=path.join(staging,'unpack'),archiveTreeSha256=extractArchive(bytes,unpack);check(!existsSync(installed),'unattested installation already exists');renameSync(unpack,installed);interpreter=verify(path.join(installed,artifact.executable),{platform,arch});check(interpreter.version===artifact.version,'pinned runtime version differs');provenance={kind:'owned_relocatable',artifact,archiveTreeSha256,installed,installedTreeSha256:treeDigest(installed,true)}; } finally { rmSync(staging,{recursive:true,force:true}); }
  }
  const receipt={context,interpreter,provenance,candidateFailures:failures};writeFileSync(receiptPath,JSON.stringify(receipt,null,2)+'\n',{flag:'wx',mode:0o400});return {...receipt,receiptPath};
}
if(process.argv[1] && path.resolve(process.argv[1])===SOURCE) {
  try { const [command,flag,runDir]=process.argv.slice(2);check(command==='bootstrap' && flag==='--run-dir' && process.argv.length===5,'usage: bootstrap --run-dir <absolute external owned root>');const r=await bootstrap(runDir);if(process.env.GITHUB_ENV) appendFileSync(process.env.GITHUB_ENV,`S9_GUARD_PYTHON=${r.interpreter.path}\nS9_GUARD_PYTHON_SHA256=${r.interpreter.sha256}\nS9_GUARD_PYTHON_ATTESTATION=${r.receiptPath}\nS9_GUARD_PYTHON_ATTESTATION_SHA256=${hashFile(r.receiptPath)}\n`);console.log(JSON.stringify(r)); } catch(error) { console.error(error.message);process.exitCode=1; }
}
