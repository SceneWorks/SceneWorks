// Normal-product task guardian adapter. A resume marker never authorizes the physical GPU.
import { createHash } from "node:crypto";
import { spawn, execFileSync } from "node:child_process";
import { open, readFile, writeFile, mkdir, chmod, rename } from "node:fs/promises";
import { createConnection } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { digest, boundFile } from "./h3-veda-acceptance.mjs";
export const protocol = "epic-25140-physical-guard-v1";
const check = (ok, why) => { if (!ok) throw new Error(`H3 guardian adapter: ${why}`); };
export function launchShape(m, backend, runDir) {
  const r=m.runners[backend], g=r.guardian;
  check(g && Number.isSafeInteger(g.maxSeconds) && g.maxSeconds > 0 && g.maxSeconds <= 3600 && Number.isSafeInteger(g.cleanupSeconds) && g.cleanupSeconds > 0 && g.cleanupSeconds <= 60, "bounded guardian/deadline unavailable");
  check(Number.isInteger(g.port) && g.port >= 1024 && g.port <= 65535 && r.apiUrl === `http://127.0.0.1:${g.port}`, "isolated loopback product endpoint required");
  for (const key of ["dataDir","configDir"]) check(path.isAbsolute(g[key]) && path.relative(runDir,g[key]) && !path.relative(runDir,g[key]).startsWith(".."), `task-owned ${key} must stay under external run directory`);
  check(path.isAbsolute(g.pythonPath) && g.guardianSource, "source-bound Python guardian interpreter/path missing");
  check(m.sourceFreeze.resourceContract?.[backend]?.canonicalLocks?.length && m.sourceFreeze.resourceContract[backend].physicalGpuId===r.physicalGpuId, "canonical physical resource contract missing");
  if (backend==="candle_cuda") for (const source of ["scripts/release/veda_cuda_guard.py","scripts/media_campaign_windows.py"]) check(/^[a-f0-9]{64}$/.test(m.sourceFreeze.inferenceFiles?.[source] ?? ""), "unchanged inference physical/WindowsJob policy hash missing");
  return g;
}
export function rpc(control, method, arg={}, { timeout=90000, connect=createConnection }={}) {
  return new Promise((resolve,reject)=> {
    const socket=connect({host:"127.0.0.1",port:control.port});let data="",done=false;
    const finish=(error,value)=>{if(done)return;done=true;socket.destroy();error?reject(error):resolve(value);};
    socket.setTimeout(timeout,()=>finish(new Error("physical guardian IPC timeout; ownership remains uncertain")));
    socket.on("error",(e)=>finish(e));socket.on("end",()=>{if(!done)finish(new Error("guardian IPC ended without readback"));});
    socket.on("connect",()=>socket.write(`${JSON.stringify({token:control.token,fingerprint:control.fingerprint,method,arg})}\n`));
    socket.on("data",(chunk)=>{data+=chunk; if(data.length>4*1024*1024)return finish(new Error("guardian IPC byte cap"));if(!data.includes("\n"))return;try { const r=JSON.parse(data.split("\n")[0]);check(r.ok,r.error??"guardian refused");finish(null,r.value); } catch(e){finish(e);} });
  });
}
export async function acquire({manifest:m,backend,runDir}) {
  const g=launchShape(m,backend,runDir), source=await boundFile(g.guardianSource), relative=path.relative(m.sourceFreeze.sceneworks.root,source.path);
  check(relative && !relative.startsWith("..") && m.sourceFreeze.files[relative]===source.sha256,"Python guardian source not frozen current code");
  check(path.resolve(source.path)===path.resolve(path.dirname(fileURLToPath(import.meta.url)),"h3-veda-guardian.py"),"guardian entrypoint differs");
  // Same bound interpreter must answer its identity before process launch; no shell commands.
  const version=JSON.parse(execFileSync(g.pythonPath,["-c","import sys,json; print(json.dumps(list(sys.version_info[:2])))"],{encoding:"utf8",timeout:5000}));check(version[0]===3 && version[1]>=10,"Python>=3.10 required for unchanged frozen policy");
  await mkdir(runDir,{recursive:true});await mkdir(path.join(runDir,"physical-guardian"),{recursive:true});
  const controlFile=path.join(runDir,"physical-guardian/control.json");let control, launch=false;
  try {
    control=JSON.parse(await readFile(controlFile,"utf8"));check(control.fingerprint===digest(m),"existing guardian input fingerprint differs");
    const closedFile=path.join(runDir,"physical-guardian/closed.json");let closed;
    try {closed=JSON.parse(await readFile(closedFile,"utf8"));}catch(e){if(e.code!=="ENOENT")throw e;}
    if(closed) {
      const tokenHash=createHash("sha256").update(control.token).digest("hex");
      check(closed.closed && closed.servicesDrained && closed.physicalLockReleased && closed.fingerprint===digest(m) && closed.ownerTokenSha256===tokenHash && closed.pid===control.pid,"guardian closure receipt identity incomplete");
      // Closure receipt is emitted after drainage/unlock; a reused PID is distinguished by birth identity.
      const current=JSON.parse(execFileSync(g.pythonPath,[source.path,"--owner-state",String(control.pid)],{encoding:"utf8",timeout:5000}));
      check(current.birth===null || current.birth!==control.birth,"old guardian still alive; wait for exact verified closure");
      const suffix=Date.now();await rename(controlFile,`${controlFile}.closed-${suffix}`);await rename(closedFile,`${closedFile}.${suffix}`);launch=true;
    } else {
      const status=await rpc(control,"status",{}, {timeout:5000});
      check(!status.closed && status.owned,"old guardian no longer owns physical resource; inspect journal before new launch");
      if(!status.controllerAlive)await rpc(control,"resume",{pid:process.pid});
      else check(status.controllerPid===process.pid,"physical guardian belongs to another live controller");
    }
  } catch(error) {if(error.code!=="ENOENT")throw error;launch=true;}
  if(launch) {
    const manifestFile=path.join(runDir,"physical-guardian/launch-input.json");await writeFile(manifestFile,JSON.stringify(m),{mode:0o600});await chmod(manifestFile,0o600);
    const log=await open(path.join(runDir,"physical-guardian/launcher.log"),"a",0o600);
    const child=spawn(g.pythonPath,[source.path,manifestFile,backend,runDir,String(process.pid)],{detached:true,stdio:["ignore",log.fd,log.fd],env:process.env});child.unref();await log.close();
    let exited;child.on("error",(e)=>{exited=e;});child.on("exit",(code)=>{exited=new Error(`guardian initialization exited ${code}`);});
    const until=Date.now()+60000;control=undefined;
    while(Date.now()<until) {if(exited)throw exited;try {control=JSON.parse(await readFile(controlFile,"utf8"));break;}catch(e){if(e.code!=="ENOENT")throw e;await delay(100);}}
    check(control,"guardian did not acknowledge actual product/physical readiness; do not submit");
  }
  check(control.fingerprint===digest(m),"guardian launch fingerprint differs");
  const verified=await rpc(control,"status",{}, {timeout:5000});check(verified.pid===control.pid && verified.birth===control.birth && verified.controllerPid===process.pid && verified.owned && !verified.closed,"guardian belongs to another controller/owner; do not adopt or release");
  const status=()=>rpc(control,"status",{}, {timeout:5000});
  return {owned:true,ownerToken:control.token,lockIdentities:control.locks,
    isOwnerAlive:async()=>{const s=await status();return s.pid===control.pid && !s.closed;},
    ownsCanonicalLocks:async()=>{const s=await status();return s.pid===control.pid && s.owned===true;},
    beforePhase:async(phase,context={})=>rpc(control,"phase",{phase,context}),
    submit:async(intent)=>rpc(control,"submit",{key:intent.key,requestSha256:intent.requestSha256,request:intent.request}),
    attachJob:async(intent)=>rpc(control,"attach",{key:intent.key,jobId:intent.jobId}),
    retainForOwnedJobs:async()=>rpc(control,"retain"),
    release:async()=>rpc(control,"release")};
}
