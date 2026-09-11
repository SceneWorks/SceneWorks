import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { lstat, mkdtemp, mkdir, readFile, realpath, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { installComponentBundle, packComponentBundle, validateComponentBundle } from "./starvector-terminal-components.mjs";
const hash = value => createHash("sha256").update(value).digest("hex");
async function fixture(t) {
  const root = await realpath(await mkdtemp(path.join(tmpdir(), "terminal-components-"))); t.after(() => rm(root, {recursive:true, force:true}));
  const source = path.join(root,"source"); await mkdir(source);
  const lock = {components:{}}, manifest={};
  for(const key of ["starcoder1","starcoder2","siglip"]){const bytes=Buffer.from(JSON.stringify({model_type:key,hidden_size:64})+'\n');lock.components[key]={repository:`owner/${key}`,revision:'a'.repeat(40),config_sha256:hash(bytes)};manifest[key]={...lock.components[key],config_path:`${key}.json`};await writeFile(path.join(source,`${key}.json`),bytes)}
  await writeFile(path.join(source,"components.json"),JSON.stringify(manifest));
  const output=path.join(root,"transport.json");await packComponentBundle(lock,source,output);
  return {root,source,lock,output,payload:await readFile(output,"utf8")};
}
test("confidential bundle round trip preserves exact bytes and returns metadata only",async t=>{
 const f=await fixture(t), destination=path.join(f.root,"installed");
 const result=await installComponentBundle(f.payload,f.lock,destination);assert.equal(result.component_count,3);assert.ok(!JSON.stringify(result).includes('content_base64'));
 for(const key of Object.keys(f.lock.components))assert.deepEqual(await readFile(path.join(destination,`${key}.json`)),await readFile(path.join(f.source,`${key}.json`)));
 await installComponentBundle(f.payload,f.lock,destination);
 if(process.platform!=="win32")assert.equal((await lstat(f.output)).mode&0o777,0o600);
 await writeFile(path.join(destination,"starcoder1.json"),'mutated');await assert.rejects(installComponentBundle(f.payload,f.lock,destination),/preserve the original/);
});
test("lock authority rejects changed bytes even if payload recomputes its own digest",async t=>{
 const f=await fixture(t), changed=JSON.parse(f.payload), bytes=Buffer.from('{"hidden_size":123}');changed.components.starcoder1.content_base64=bytes.toString('base64');changed.components.starcoder1.config_sha256=hash(bytes);
 assert.throws(()=>validateComponentBundle(JSON.stringify(changed),f.lock),/committed lock/);
 const wrong=JSON.parse(f.payload);wrong.components.starcoder1.content_base64=bytes.toString('base64');assert.throws(()=>validateComponentBundle(JSON.stringify(wrong),f.lock),/committed hash/);
 const pending=structuredClone(f.lock);delete pending.components.starcoder1.config_sha256;assert.throws(()=>validateComponentBundle(f.payload,pending),/not ready/);
});
test("bundle rejects missing extra traversal malformed and oversized entries without disclosing content",async t=>{
 const f=await fixture(t);
 for(const change of [b=>delete b.components.siglip,b=>b.components.extra=b.components.siglip,b=>b.components.siglip.config_path='../escape.json',b=>b.components.siglip.extra='confidential-sentinel',b=>b.components.siglip.revision='b'.repeat(40)]){const b=JSON.parse(f.payload);change(b);assert.throws(()=>validateComponentBundle(JSON.stringify(b),f.lock),error=>!error.message.includes('confidential-sentinel'))}
 assert.throws(()=>validateComponentBundle('confidential-sentinel',f.lock),/not JSON/);
 assert.throws(()=>validateComponentBundle('x'.repeat(32769),f.lock),/32 KiB/);
});
test("installation rejects symbolic links and extra durable files",async t=>{
 const f=await fixture(t), destination=path.join(f.root,"installed");await installComponentBundle(f.payload,f.lock,destination);
 await writeFile(path.join(destination,"extra"),'x');await assert.rejects(installComponentBundle(f.payload,f.lock,destination),/different inventory/);
 const alias=path.join(f.root,"alias");await symlink(destination,alias,process.platform==='win32'?'junction':'dir');await assert.rejects(installComponentBundle(f.payload,f.lock,alias),/symlink/);
});
test("existing provision transports the secret only to the Windows materializer",async()=>{
 const workflow=await readFile('.github/workflows/starvector-terminal-provision.yml','utf8');assert.match(workflow,/STARVECTOR_UPSTREAM_COMPONENTS_JSON:\n        required: true/);assert.equal((workflow.match(/secrets\.STARVECTOR_UPSTREAM_COMPONENTS_JSON/g)??[]).length,1);assert.match(workflow,/starvector-terminal-components\.mjs install/);assert.doesNotMatch(workflow,/path:[^\n]*upstream-components/);
});
