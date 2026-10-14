"""CPU fake-command policy tests; no Rustup/Cargo/native/product execution."""
import importlib.util,json,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('cpu_build',Path(__file__).with_name('h3-veda-cpu-build.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class Tests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
 def tearDown(self):self.temp.cleanup()
 def command(self,code):return [sys.executable,'-I','-S','-c',code]
 def run_child(self,code,seconds=3):return b.supervise(self.command(code),self.root,os.environ.copy(),self.root/'child',seconds,os.getpid())
 def receipt(self):return json.loads((self.root/'child/receipt.json').read_text())
 def test_actual_cpu_success_requires_reaped_drained_birth_and_log_hash(self):
  r=self.run_child('import time;time.sleep(.2);print("CPU fixture")');self.assertTrue(r['prepared']);self.assertTrue(r['closure']['reaped']);self.assertTrue(r['closure']['descendantsDrained']);self.assertTrue(r['birth']);self.assertEqual(r['stdout']['sha256'],b.sha(r['stdout']['path']));self.assertFalse(r['physicalAdmission'])
 def test_original_timeout_drains_actual_owned_cpu_child_and_cannot_pass(self):
  start=time.monotonic()
  with self.assertRaises(RuntimeError):self.run_child('import time;time.sleep(60)',.6)
  r=self.receipt();self.assertIn('TimeoutError',r['error']);self.assertFalse(r['prepared']);self.assertTrue(r['closure']['reaped']);self.assertTrue(r['closure']['descendantsDrained']);self.assertLess(time.monotonic()-start,2)
 def test_controller_birth_loss_drains_actual_owned_child(self):
  real=b.pid_birth;calls=[0]
  def birth(pid):
   if pid==os.getpid():calls[0]+=1;return 'owner' if calls[0]==1 else None
   return real(pid)
  with patch.object(b,'pid_birth',side_effect=birth):
   with self.assertRaises(RuntimeError):self.run_child('import time;time.sleep(60)')
  r=self.receipt();self.assertIn('controller ended',r['error']);self.assertTrue(r['closure']['descendantsDrained']);self.assertFalse(r['prepared'])
 def test_child_launch_failure_is_durable_without_fabricated_closure(self):
  with patch.object(b,'pid_birth',return_value='owned'),patch.object(b.subprocess,'Popen',side_effect=OSError('spawn failure')):
   with self.assertRaises(RuntimeError):self.run_child('pass')
  r=self.receipt();self.assertIn('spawn failure',r['error']);self.assertNotIn('closure',r);self.assertFalse(r['prepared'])
 def test_foreign_pid_birth_never_signals_group(self):
  child=type('Child',(),{'pid':123,'poll':lambda s:None})()
  with patch.object(b,'pid_birth',return_value='foreign'),patch.object(b.os,'killpg') as kill:
   with self.assertRaisesRegex(RuntimeError,'birth changed'):b.child_closure(child,'owned',time.monotonic()+10)
   kill.assert_not_called()
 def test_surviving_descendants_refuse_closure_without_foreign_kill(self):
  child=type('Child',(),{'pid':123,'returncode':0,'poll':lambda s:0})()
  with patch.object(b,'pid_birth',return_value=None),patch.object(b,'group_alive',return_value=True),patch.object(b.os,'killpg') as kill:
   with self.assertRaisesRegex(RuntimeError,'descendants'):b.child_closure(child,'owned',time.monotonic()+10)
   kill.assert_not_called()
 def test_nonzero_cpu_command_is_closed_but_not_prepared(self):
  with self.assertRaises(RuntimeError):self.run_child('import time;time.sleep(.2);raise SystemExit(7)')
  r=self.receipt();self.assertEqual(r['rc'],7);self.assertTrue(r['closure']['reaped']);self.assertFalse(r['prepared'])
 def test_failure_receipt_seals_context_before_nonzero_child(self):
  context={'tools':{'rustc':{'path':'/owned/rustc','sha256':'a'*64}},'environment':{'RUSTFLAGS':'original'},'sourceBindings':[{'sha256':'b'*64}]}
  with self.assertRaises(RuntimeError):b.supervise(self.command('import time;time.sleep(.2);raise SystemExit(7)'),self.root,os.environ.copy(),self.root/'child',3,os.getpid(),context)
  r=self.receipt();self.assertEqual(r['tools'],context['tools']);self.assertEqual(r['environment'],context['environment']);self.assertEqual(r['sourceBindings'],context['sourceBindings']);self.assertFalse(r['prepared']);self.assertEqual(r['rc'],7)
 def test_context_cannot_forge_result_or_lifetime(self):
  with self.assertRaisesRegex(RuntimeError,'override'):b.supervise(self.command('pass'),self.root,os.environ.copy(),self.root/'child',3,os.getpid(),{'prepared':True})
 def test_context_survives_timeout_and_secret_environment_is_not_recorded(self):
  with self.assertRaises(RuntimeError):b.supervise(self.command('import time;time.sleep(60)'),self.root,os.environ.copy(),self.root/'child',.6,os.getpid(),{'tools':{'cargo':'actual'},'environment':b.environment_binding({'GH_TOKEN':'secret','RUSTC_WRAPPER':'/owned/wrapper'})})
  r=self.receipt();self.assertEqual(r['tools'],{'cargo':'actual'});self.assertEqual(r['environment'],{'RUSTC_WRAPPER':'/owned/wrapper'});self.assertTrue(r['closure']['descendantsDrained'])
 def plan(self,mode='toolchain'):

  source=self.root/'download-source';bindings=[]
  for name in ['Cargo.toml','Cargo.lock','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml','crates/sceneworks-worker/src/lib.rs']:
   p=source/name;p.parent.mkdir(parents=True,exist_ok=True)
   if not p.exists():p.write_bytes(b'owned CPU source fixture')
   bindings.append(b.binding(p))
  return {'schemaVersion':1,'kind':'epic25140_closed_CPU_build_plan','mode':mode,'runDir':str(self.root),'sourceRoot':str(self.root/'download-source'),'helper':b.binding(Path(b.__file__).resolve()),'python':b.binding(Path(sys.executable).resolve()),'sourceBindings':bindings}
 def test_closed_toolchain_argv_owned_homes_and_caps_without_installing(self):
  argv,cwd,env,seconds,tools=b.closed_command(self.plan());self.assertEqual(argv[1:],['toolchain','install','1.97.1','--profile','minimal','--no-self-update']);self.assertEqual(seconds,600);self.assertEqual(env['RUSTUP_HOME'],str(self.root/'tools/rustup'));self.assertEqual(env['CARGO_HOME'],str(self.root/'tools/cargo'));self.assertNotIn('GH_TOKEN',env);self.assertTrue(tools['rustup']['sha256'])
  for change in [lambda p:p.update(mode='arbitrary'),lambda p:p.update(argv=['rm']),lambda p:p.update(sourceRoot='/foreign')]:
   p=self.plan();change(p)
   with self.assertRaises(RuntimeError):b.closed_command(p)
 def test_closed_cargo_argv_uses_actual_owned_tool_bytes_and_original_source(self):
  source=self.root/'download-source';source.mkdir(exist_ok=True);bins=self.root/'tools/rustup/toolchains/1.97.1-aarch64-apple-darwin/bin';bins.mkdir(parents=True)
  for name in ['rustc','cargo']:(bins/name).write_bytes(b'CPU fake tool fixture')
  def output(argv,**kw):return b.SOURCE+'\n' if argv[0]=='git' else Path(argv[0]).name+' 1.97.1 (CPU fixture)\n'
  with patch.object(b.subprocess,'check_output',side_effect=output):argv,cwd,env,seconds,tools=b.closed_command(self.plan('cargo'))
  self.assertEqual(seconds,3600);self.assertEqual(argv,[str(bins/'cargo'),'build','--release','--locked','--message-format=json-render-diagnostics','-p','sceneworks-rust-api']);self.assertEqual(env['RUSTC'],str(bins/'rustc'));self.assertEqual(tools['cargo']['sha256'],b.sha(bins/'cargo'))
  with patch.object(b.subprocess,'check_output',return_value='wrong-head'):
   with self.assertRaisesRegex(RuntimeError,'source head'):b.closed_command(self.plan('cargo'))
if __name__=='__main__':unittest.main()
