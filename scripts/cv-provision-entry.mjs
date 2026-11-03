#!/usr/bin/env node
// Native setup only. The Python provisioner owns manual admission before any downloads.
import assert from 'node:assert/strict';
import {appendFileSync} from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath, pathToFileURL} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export function provisionPaths(kind,env,platform=process.platform) {
  assert(['oracle','provider'].includes(kind),'unknown path setup');
  const paths=platform==='win32'?path.win32:path.posix;
  assert(paths.isAbsolute(env.RUNNER_TEMP??'')&&!/[\r\n]/.test(env.RUNNER_TEMP),'absolute runner temp required');
  for(const key of ['GITHUB_RUN_ID','GITHUB_RUN_ATTEMPT']) assert(/^\d+$/.test(env[key]??''),`invalid ${key}`);
  const names=kind==='provider'
    ? {CV_PROVISION_ROOT:'cv-provision',CV_ORACLE_ARTIFACT:'cv-oracle-artifact'}
    : {CV_ORACLE_INPUT:'cv-oracle-input',CV_ORACLE_TOOLS:'cv-oracle-tools',CV_ORACLE_OUTPUT:'cv-oracles',YOLO_CONFIG_DIR:'cv-ultralytics',MPLCONFIGDIR:'cv-matplotlib',TORCH_HOME:'cv-torch',XDG_CACHE_HOME:'cv-oracle-cache',PIP_CACHE_DIR:'cv-pip'};
  return Object.entries(names).map(([name,prefix])=>`${name}=${paths.join(env.RUNNER_TEMP,`${prefix}-${env.GITHUB_RUN_ID}-${env.GITHUB_RUN_ATTEMPT}`)}\n`).join('');
}
export function provisionCommand(env,platform=process.platform) {
  const paths=platform==='win32'?path.win32:path.posix;
  assert(paths.isAbsolute(env.CV_NATIVE_PYTHON??'')&&!/[\r\n]/.test(env.CV_NATIVE_PYTHON),'explicit native Python required');
  for(const key of ['CV_BACKEND','CV_SOURCE_SHA','CV_INFERENCE_SHA','CV_PROVISION_ROOT','CV_ORACLE_ARTIFACT']) assert(env[key],`missing ${key}`);
  return [env.CV_NATIVE_PYTHON,['-I','-B',path.join(root,'scripts/cv_provision.py'),'--mode','proof','--backend',env.CV_BACKEND,'--source-sha',env.CV_SOURCE_SHA,'--inference-sha',env.CV_INFERENCE_SHA,'--output',env.CV_PROVISION_ROOT,'--oracle-dir',env.CV_ORACLE_ARTIFACT,...(env.CV_BACKEND==='mlx'?['--public-only']:[])]];
}
export function run(args,env=process.env,spawn=spawnSync) {
  if(args.length===2&&args[0]==='paths') {
    assert(env.GITHUB_ENV,'GITHUB_ENV required');
    appendFileSync(env.GITHUB_ENV,provisionPaths(args[1],env),'utf8');
    return 0;
  }
  assert.deepEqual(args,['proof'],'expected paths oracle|provider or proof');
  const [command,argv]=provisionCommand(env);
  const result=spawn(command,argv,{cwd:root,env,stdio:'inherit',shell:false,windowsHide:true});
  if(result.error) throw result.error;
  assert.equal(result.signal,null,'provisioner terminated by signal');
  assert(Number.isInteger(result.status),'provisioner did not exit');
  return result.status;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) {
  try {process.exitCode=run(process.argv.slice(2));}
  catch(error) {console.error(error.message);process.exitCode=1;}
}
