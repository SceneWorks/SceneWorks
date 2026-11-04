#!/usr/bin/env node
// Read-only native Windows interpreter discovery. Never install Python or edit its registry.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';

const probe = 'import ensurepip,json,platform,ssl,struct,sys,venv; print(json.dumps(dict(executable=sys.executable,implementation=sys.implementation.name,version=list(sys.version_info[:3]),platform=sys.platform,machine=platform.machine(),bits=struct.calcsize("P")*8)))';
function directories(directory, read = fs.readdirSync) {
  try { return read(directory, {withFileTypes:true}).filter(row => row.isDirectory()).map(row => row.name); }
  catch (error) { if (['ENOENT','ENOTDIR','EACCES','EPERM'].includes(error.code)) return []; throw error; }
}
export function windowsCandidates(env, read = fs.readdirSync) {
  const p = path.win32;
  const candidates = [];
  const add = (command, prefix = []) => {
    if (command && !/[\r\n]/.test(command) && !candidates.some(row => row.command === command)) candidates.push({command,prefix});
  };
  if (env.CV_NATIVE_PYTHON) { assert(p.isAbsolute(env.CV_NATIVE_PYTHON) && !/[\r\n]/.test(env.CV_NATIVE_PYTHON), 'explicit Python must be absolute and single-line'); add(env.CV_NATIVE_PYTHON); }
  if (env.LOCALAPPDATA) add(p.join(env.LOCALAPPDATA,'Programs','Python','Python312','python.exe'));
  if (env.ProgramFiles) add(p.join(env.ProgramFiles,'Python312','python.exe'));
  const caches = new Set([env.RUNNER_TOOL_CACHE,env.AGENT_TOOLSDIRECTORY].filter(value => value && p.isAbsolute(value)));
  // Listeners on one host have independent tool caches. Inspect their existing interpreters,
  // without trusting a .complete marker or invoking setup-python's privileged repair script.
  if (env.RUNNER_TOOL_CACHE && p.isAbsolute(env.RUNNER_TOOL_CACHE)) {
    const listener = p.dirname(p.dirname(env.RUNNER_TOOL_CACHE));
    const parent = p.dirname(listener);
    const stem = p.basename(listener).replace(/-\d+$/, '');
    for (const name of directories(parent,read)) {
      if (name === stem || (name.startsWith(stem+'-') && /^\d+$/.test(name.slice(stem.length+1)))) caches.add(p.join(parent,name,'_work','_tool'));
    }
  }
  for (const cache of caches) {
    const root = p.join(cache,'Python');
    for (const version of directories(root,read).filter(name => /^3\.12\.\d+$/.test(name)).sort((a,b) => b.localeCompare(a,undefined,{numeric:true}))) add(p.join(root,version,'x64','python.exe'));
  }
  add('py',['-3.12']);
  add('python3.12');
  add('python');
  return candidates;
}
export function selectWindowsPython(env, spawn = spawnSync, read = fs.readdirSync) {
  const attempts = [];
  for (const {command,prefix} of windowsCandidates(env,read)) {
    const result = spawn(command,[...prefix,'-I','-B','-c',probe],{encoding:'utf8',timeout:15000,maxBuffer:65536,windowsHide:true,shell:false});
    let identity;
    try { identity = JSON.parse(result.stdout); } catch { /* Record failed or non-JSON probes below. */ }
    const valid = !result.error && result.status === 0 && !result.signal && identity?.implementation === 'cpython' && identity.platform === 'win32' && identity.bits === 64 && identity.version?.[0] === 3 && identity.version?.[1] === 12 && ['amd64','x86_64'].includes(identity.machine?.toLowerCase()) && path.win32.isAbsolute(identity.executable ?? '') && !/[\r\n]/.test(identity.executable);
    attempts.push({command,status:result.status,error:result.error?.code,identity});
    if (valid) return {identity,attempts};
  }
  throw new Error('Native x64 Windows CPython3.12 with ssl/venv/ensurepip is required; existing-interpreter probes: '+JSON.stringify(attempts));
}
export function run(env = process.env) {
  assert(process.platform === 'win32', 'Windows native selection only');
  for (const name of ['GITHUB_ENV','GITHUB_PATH','RUNNER_TEMP','GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT']) assert(env[name] && !/[\r\n]/.test(env[name]),`invalid ${name}`);
  assert(/^\d+$/.test(env.GITHUB_RUN_ID) && /^\d+$/.test(env.GITHUB_RUN_ATTEMPT),'numeric run identity required');
  const record = path.join(env.RUNNER_TEMP,`cv-python-selection-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}.json`);
  try {
    const selected = selectWindowsPython(env);
    fs.writeFileSync(record,JSON.stringify(selected,null,2),{flag:'wx'});
    fs.appendFileSync(env.GITHUB_ENV,`CV_NATIVE_PYTHON=${selected.identity.executable}\n`);
    fs.appendFileSync(env.GITHUB_PATH,path.win32.dirname(selected.identity.executable)+'\n');
    console.log('Selected verified native CPython',JSON.stringify(selected.identity));
  } catch (error) {
    if (!fs.existsSync(record)) fs.writeFileSync(record,JSON.stringify({error:error.message}),{flag:'wx'});
    throw error;
  }
}
if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  try { run(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
