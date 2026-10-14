import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const candidates=[process.env.S9_GUARD_TEST_PYTHON,'/opt/homebrew/bin/python3.12','python3.12'].filter(Boolean);
function python312() { for(const p of candidates) { try { if(execFileSync(p,['-I','-S','-c',"import sys; print('.'.join(map(str,sys.version_info[:2])))"],{encoding:'utf8'}).trim()==='3.12') return p; } catch {} } throw Error('actual CPython3.12 required for guard CPU policy tests'); }
for (const [name,file,count] of [['fixture source/refusals','h3-veda-mlx-fixtures-tests.py',26],['unchanged shared physical guard','h3-veda-guardian-tests.py',38]]) {
  test(`${name} CPU policy suite executes completely`,()=>{
    const r=spawnSync(python312(),[fileURLToPath(new URL(file,import.meta.url))],{encoding:'utf8',timeout:60000});
    assert.equal(r.status,0,r.stderr);assert.match(r.stderr,new RegExp(`Ran ${count} tests`));assert.match(r.stderr,/\nOK\s*$/);
  });
}
