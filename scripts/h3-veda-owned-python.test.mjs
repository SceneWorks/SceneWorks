import assert from 'node:assert/strict';
import test from 'node:test';
import { mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { artifacts, bootstrap, candidates, extractArchive, ownedRoot, parseArchive, sha256, treeDigest, verifyInterpreter } from './h3-veda-owned-python.mjs';
function fixture(t) { const root=realpathSync(mkdtempSync(path.join(os.tmpdir(),'h3-python-')));t.after(()=>rmSync(root,{recursive:true,force:true}));return root; }
function tar(entries) { const blocks=[];for(const e of entries) { const h=Buffer.alloc(512),data=Buffer.from(e.data||'');h.write(e.name,0,100);h.write('0000755\0',100);h.write(data.length.toString(8).padStart(11,'0')+'\0',124);h.fill(32,148,156);h.write(e.type||'0',156);h.write(e.link||'',157,100);h.write('ustar\0',257);let sum=0;for(const b of h)sum+=b;h.write(sum.toString(8).padStart(6,'0')+'\0 ',148);blocks.push(h,data,Buffer.alloc((512-data.length%512)%512)); }blocks.push(Buffer.alloc(1024));return gzipSync(Buffer.concat(blocks)); }
function fact(executable,overrides={}) {return JSON.stringify({implementation:'cpython',version:[3,12,13],executable,platform:process.platform,machine:process.arch==='arm64'?'arm64':'x86_64',...overrides});}
function probe(p, options={}) {return verifyInterpreter(p,{...options,execute:(exe,args)=>{assert.deepEqual(args.slice(0,3),['-I','-S','-B']);return fact(exe,options.fact||{});}});}
test('existing interpreter verification binds actual executable bytes/version/platform and rejects mutation',t=>{const root=fixture(t),p=path.join(root,'python');writeFileSync(p,'original');const r=probe(p);assert.equal(r.sha256,sha256(Buffer.from('original')));for(const f of [{version:[3,11,9]},{version:[3,14,0]},{implementation:'pypy'},{machine:'unrelated'},{platform:'wrong'}])assert.throws(()=>probe(p,{fact:f}));assert.throws(()=>verifyInterpreter(p,{execute:exe=>{writeFileSync(exe,'changed');return fact(exe);}}),/changed/);});
test('candidate discovery names exact3.12 and includes explicit Homebrew locations',()=>{assert.ok(candidates('darwin',{PATH:'',LOCALAPPDATA:''}).includes('/opt/homebrew/bin/python3.12'));assert.ok(!candidates('darwin',{PATH:'/usr/bin'}).includes('/usr/bin/python3'));});
test('owned roots reject traversal/newlines/symlink ancestors',t=>{const root=fixture(t);assert.throws(()=>ownedRoot('relative'));assert.throws(()=>ownedRoot(root+'\n'));const link=path.join(root,'link');symlinkSync(root,link);assert.throws(()=>ownedRoot(path.join(link,'child')),/symlink/);});
test('strict archive supports real regular files and internal relative links without invoking tar',t=>{const root=fixture(t),bytes=tar([{name:'python/bin/python3.12',data:'exe'},{name:'python/bin/python3',type:'2',link:'python3.12'},{name:'python/links/python',type:'2',link:'../bin/python3.12'}]);assert.equal(parseArchive(bytes).length,3);const dest=path.join(root,'install');extractArchive(bytes,dest);assert.equal(readFileSync(path.join(dest,'python/bin/python3'),'utf8'),'exe');const hash=treeDigest(dest);writeFileSync(path.join(dest,'python/bin/python3.12'),'mutation');assert.notEqual(treeDigest(dest),hash);});
test('archive refuses path/link/metadata/hardlink/duplicate/checksum/truncation mutations before extraction',()=>{for(const entry of [{name:'../escape'},{name:'/absolute'},{name:'python/../../escape'},{name:'python/C:\\escape'},{name:'python/a',type:'1'},{name:'python/a',type:'x'},{name:'python/a',type:'2',link:'../../escape'}])assert.throws(()=>parseArchive(tar([entry])));assert.throws(()=>parseArchive(tar([{name:'python/a'},{name:'python/a'}])));assert.throws(()=>parseArchive(tar([{name:'python/a',type:'2',link:'target'},{name:'python/a/child'},{name:'python/target'}])));const raw=Buffer.alloc(512);raw.write('python/a');assert.throws(()=>parseArchive(gzipSync(raw)),/checksum/);assert.throws(()=>parseArchive(tar([{name:'python/a'}]).subarray(0,15)));});
test('bootstrap uses verified existing3.12 without download and seals reuse/mutation/context checks',async t=>{const root=fixture(t),p=path.join(root,'existing');writeFileSync(p,'interpreter');const options={discover:()=>[p],verify:probe,fetchArchive:()=>{throw Error('must not download');}};const r=await bootstrap(path.join(root,'owned'),options);assert.equal(r.provenance.kind,'existing_verified');assert.equal((await bootstrap(path.join(root,'owned'),options)).interpreter.sha256,r.interpreter.sha256);writeFileSync(p,'mutated');await assert.rejects(bootstrap(path.join(root,'owned'),options),/mutation/);await assert.rejects(bootstrap(path.join(root,'owned'),{...options,arch:'different'}),/context/);});
test('bootstrap rejects unauthenticated download and foreign symlink marker',async t=>{const root=fixture(t),run=path.join(root,'owned');await assert.rejects(bootstrap(run,{platform:'darwin',arch:'arm64',discover:()=>[],fetchArchive:async()=>Buffer.from('corrupt')}),/SHA\/length/);const marker=path.join(run,'tools/.h3-owned-python.json');const bytes=readFileSync(marker);rmSync(marker);const external=path.join(root,'external');writeFileSync(external,bytes);symlinkSync(external,marker);await assert.rejects(bootstrap(run,{platform:'darwin',arch:'arm64'}),/marker symlink/);});
test('fallback artifact contract pins immutable official release, exactminor and both physical platform families',()=>{assert.deepEqual(Object.keys(artifacts),['darwin-arm64','win32-x64']);for(const a of Object.values(artifacts)){assert.match(a.url,/^https:\/\/github.com\/astral-sh\/python-build-standalone\/releases\/download\/20261009\//);assert.match(a.sha256,/^[a-f0-9]{64}$/);assert.equal(a.version,'3.12.15');assert.ok(a.bytes<64*1024*1024);}});
function assertTerminalBootstrapSurface(w) {
 assert.doesNotMatch(w,/actions\/setup-python@|RUNNER_TOOL_CACHE|sudo/);
 assert.match(w,/options: \[inventory, stage, build, prepare, resume, review\]/);
 const calls=[...w.matchAll(/h3-veda-owned-python\.mjs bootstrap --run-dir/g)].length;
 const body=w.split(/\njobs:\n/)[1];assert.ok(body,'terminal jobs body required');
 if(/^  registration_only:/m.test(body)) {
  assert.deepEqual([...body.matchAll(/^  ([\w-]+):/gm)].map(m=>m[1]),['registration_only']);
  assert.match(body,/runs-on: ubuntu-latest/);assert.match(body,/timeout-minutes: 5/);
  assert.doesNotMatch(body,/self-hosted|secrets\.|h3-veda-(?:actions|acceptance)\.mjs/);
  assert.equal(calls,0,'registration stub must execute no Python bootstrap');
 } else assert.equal(calls,2,'full feature workflow must bootstrap both physical hosts');
}
test('publication stub and CPU diagnostic/cache routes preserve owned bootstrap and terminal six-mode scope',()=>{
 const w=readFileSync(new URL('../.github/workflows/h3-veda-terminal.yml',import.meta.url),'utf8');assertTerminalBootstrapSurface(w);
 const cache=readFileSync(new URL('../.github/workflows/h3-veda-cache-prepare.yml',import.meta.url),'utf8');
 assert.doesNotMatch(cache,/actions\/setup-python@|RUNNER_TOOL_CACHE|sudo/);assert.match(cache,/options: \[build, prepare\]/);assert.match(cache,/run: node scripts\/h3-veda-cache-diagnostic\.mjs "\$S9_RUN_DIR"/);assert.match(cache,/h3-veda-cache-prepare\.mjs prepare/);
 const controller=readFileSync(new URL('h3-veda-cache-prepare.mjs',import.meta.url),'utf8');assert.match(controller,/import \{bootstrap,noSymlinkAncestors\} from '\.\/h3-veda-owned-python\.mjs'/);assert.match(controller,/await bootstrap\(runDir\)/);
 const diagnostic=readFileSync(new URL('h3-veda-cache-diagnostic.mjs',import.meta.url),'utf8');assert.match(diagnostic,/import \{bootstrap,noSymlinkAncestors\} from '\.\/h3-veda-owned-python\.mjs'/);assert.match(diagnostic,/await bootstrap\(runDir\)/);
});
test('full feature bootstrap checks remain enforced after integration and stub cannot hide host jobs',()=>{
 const header=readFileSync(new URL('../.github/workflows/h3-veda-terminal.yml',import.meta.url),'utf8').split(/\njobs:\n/)[0];
 const full=header+'\njobs:\n  mac2:\n    steps:\n      - run: node scripts/h3-veda-owned-python.mjs bootstrap --run-dir "$S9_RUN_DIR"\n  windows:\n    steps:\n      - run: node scripts/h3-veda-owned-python.mjs bootstrap --run-dir "$S9_RUN_DIR"\n';
 assertTerminalBootstrapSurface(full);
 assert.throws(()=>assertTerminalBootstrapSurface(full.replaceAll('h3-veda-owned-python.mjs bootstrap --run-dir','other-bootstrap')));
 assert.throws(()=>assertTerminalBootstrapSurface(full.replace('[inventory, stage, build, prepare, resume, review]','[prepare]')));
 assert.throws(()=>assertTerminalBootstrapSurface(full+'  registration_only:\n    runs-on: ubuntu-latest\n    timeout-minutes: 5\n'));
 assert.throws(()=>assertTerminalBootstrapSurface(full+'# actions/setup-python@bad\n'));
});
