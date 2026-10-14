"""Tiny CPU files/fake transports and owned CPU child processes; no API/GPU/model download."""
import importlib.util,os,tempfile,unittest,json,shutil,sys,time
from unittest.mock import patch
from pathlib import Path
spec=importlib.util.spec_from_file_location('cache',Path(__file__).with_name('h3-veda-cache-prepare.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
class Tests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.shared=self.root/'shared';self.cache=self.root/'owned';self.shared.mkdir();self.cache.mkdir();self.a={'repo':'author/model','revision':'a'*40,'file':'weight.safetensors','bytes':4,'sha256':__import__('hashlib').sha256(b'data').hexdigest(),'allowManagedFetch':True}
 def tearDown(self):self.temp.cleanup()
 def test_private_clone_is_independent_and_corruption_refused(self):
  p=c.snapshot(self.shared,self.a);p.parent.mkdir(parents=True);p.write_bytes(b'data');q,origin=c.stage_file(self.shared,self.cache,self.a,lambda:None,clone=shutil.copyfile);r=c.verify_file(q,self.a,lambda:None);self.assertTrue(r['contentVerified']);q.resolve().chmod(0o600);q.write_bytes(b'evil');self.assertEqual(p.read_bytes(),b'data')
  with self.assertRaises(RuntimeError):c.verify_file(q,self.a,lambda:None)
 def test_complete_private_blob_recovers_without_shared_source(self):
  blob=self.cache/'hub/models--author--model/blobs'/self.a['sha256'];blob.parent.mkdir(parents=True);blob.write_bytes(b'data');q,origin=c.stage_file(self.shared,self.cache,self.a,lambda:None,clone=lambda *_:self.fail('must not clone'));self.assertTrue(q.exists());self.assertEqual(origin['method'],'recovered_complete_private_blob_after_SHA')
 def test_foreign_symlink_and_hardlink_refused(self):
  q=c.snapshot(self.cache,self.a);q.parent.mkdir(parents=True);outside=self.root/'outside';outside.write_bytes(b'data');q.symlink_to(outside)
  with self.assertRaises(RuntimeError):c.stage_file(self.shared,self.cache,self.a,lambda:None)
  q.unlink();os.link(outside,q)
  with self.assertRaises(RuntimeError):c.stage_file(self.shared,self.cache,self.a,lambda:None)
 def test_timeout_streaming_hash_and_unapproved_missing_are_refused(self):
  p=self.root/'file';p.write_bytes(b'data')
  with self.assertRaises(TimeoutError):c.sha(p,lambda:(_ for _ in ()).throw(TimeoutError('deadline')))
  with self.assertRaises(RuntimeError):c.stage_file(self.shared,self.cache,{**self.a,'allowManagedFetch':False},lambda:None)
 def test_exact_membership_not_counts_and_changed_stat_cannot_pass(self):
  q=c.snapshot(self.cache,self.a);q.parent.mkdir(parents=True);q.write_bytes(b'data');r=c.verify_file(q,self.a,lambda:None);key='author/model@'+'a'*40+'/weight.safetensors';p={'files':{key:r}};self.assertTrue(c.complete_files(p,[self.a],self.cache));self.assertFalse(c.complete_files({'files':{'foreign':r}},[self.a],self.cache));q.chmod(0o600);q.write_bytes(b'evil');self.assertFalse(c.complete_files(p,[self.a],self.cache))
 def test_cancel_transport_error_still_drains_both_and_retains_error(self):
  class Services:
   children={'download_worker':1,'api':2}
   def stop(s,name):s.children.pop(name);return {'reaped':True,'descendantsDrained':True}
  p={'jobs':{'key':{}}};errors=c.cleanup(Services(),lambda *_:(_ for _ in ()).throw(EOFError('API response lost')),({'id':'job'},self.a,'key'),p);self.assertTrue(errors);self.assertTrue(p['servicesDrained']);self.assertEqual(len(p['servicesClosure']),2)
 def test_one_owned_child_failure_does_not_bypass_other_drain(self):
  class Services:
   children={'download_worker':1,'api':2}
   def stop(s,name):
    if name=='download_worker':raise RuntimeError('birth identity uncertain')
    s.children.pop(name);return {'reaped':True,'descendantsDrained':True}
  p={'jobs':{}};self.assertTrue(c.cleanup(Services(),None,None,p));self.assertFalse(p['servicesDrained']);self.assertEqual(p['servicesClosure'][0]['role'],'api')
 def test_wrong_job_or_predictor_digest_never_passes(self):
  a={**self.a,'repo':'Veda-Sparse/predictor'};j={'type':'model_download','requestedGpu':'cpu','payload':c.request_shape(a,'key')['payload']};c.verify_job(j,a,'key');j['payload']['expectedSha256']='b'*64
  with self.assertRaises(RuntimeError):c.verify_job(j,a,'key')
 def test_download_closure_requires_actual_completed_job_not_existing_bytes(self):
  key=__import__('hashlib').sha256(('author/model@'+'a'*40+'/weight.safetensors').encode()).hexdigest();j={'id':'job','type':'model_download','requestedGpu':'cpu','status':'completed','payload':c.request_shape(self.a,key)['payload']};p={'jobs':{key:{'jobId':'job','readback':j}}};self.assertTrue(c.complete_jobs(p,[self.a]))
  for change in [lambda p:p['jobs'][key].pop('readback'),lambda p:p['jobs'][key]['readback'].update(status='failed'),lambda p:p.update(uncertainPost={'key':key}),lambda p:p['jobs'].update(foreign={})]:
   q=__import__('copy').deepcopy(p);change(q);self.assertFalse(c.complete_jobs(q,[self.a]))
 def test_early_apfs_refusal_is_durable_nonzero_without_children(self):
  p={'runDir':str(self.root),'cacheRoot':str(self.cache),'sharedCache':str(self.shared),'sourceModules':{'guardian':{'path':str(Path(__file__).with_name('h3-veda-guardian.py'))}},'maxSeconds':30,'entries':[self.a],'sourceHead':'a'*40,'owner':{},'template':{'sha256':'b'*64},'catalog':{'sha256':'c'*64}}
  with patch.object(c,'prove_apfs',side_effect=RuntimeError('not APFS')):
   with self.assertRaises(RuntimeError):c.prepare_locked(p,os.getpid(),self.root,self.cache,self.shared,None)
  r=json.loads((self.root/'cache-progress.json').read_text());self.assertIn('not APFS',r['phaseEnded']);self.assertFalse(r['contentVerified']);self.assertTrue(r['servicesDrained']);self.assertFalse(r['campaignReady'])
 def test_timeout_is_durable_nonzero_not_success(self):
  p={'sourceModules':{'guardian':{'path':str(Path(__file__).with_name('h3-veda-guardian.py'))}},'maxSeconds':0,'entries':[self.a],'sourceHead':'a'*40,'owner':{},'template':{'sha256':'b'*64},'catalog':{'sha256':'c'*64}}
  with patch.object(c,'prove_apfs',return_value={}):
   with self.assertRaises(RuntimeError):c.prepare_locked(p,os.getpid(),self.root,self.cache,self.shared,None)
  r=json.loads((self.root/'cache-progress.json').read_text());self.assertIn('TimeoutError',r['phaseEnded']);self.assertFalse(r['contentVerified'])
 def test_foreign_inherited_hf_cache_priority_cannot_escape_private_root(self):
  base={k:'/foreign/shared/cache' for k in c.owned_hf_environment(self.cache)};base.update(HF_HUB_OFFLINE='1',HF_TOKEN='unpersisted fixture secret')
  for kind in ['api','download_worker']:
   env=c.cpu_environment(base,self.cache,kind);self.assertEqual({k:env[k] for k in c.owned_hf_environment(self.cache)},c.owned_hf_environment(self.cache));self.assertNotIn('HF_TOKEN',env);self.assertEqual(env['SCENEWORKS_GPU_ID'],'cpu');self.assertEqual(env['SCENEWORKS_WORKER_ONLY'],'1' if kind=='download_worker' else '0')
   env['HF_HUB_CACHE']='/foreign/shared/cache'
   with self.assertRaisesRegex(RuntimeError,'escaped'):c.require_owned_hf_environment(env,self.cache)
 def test_actual_owned_cpu_children_are_reaped_after_transport_failure(self):
  gs=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('h3-veda-guardian.py'));g=importlib.util.module_from_spec(gs);gs.loader.exec_module(g)
  binary=self.root/'cpu-child';binary.write_text('#!'+sys.executable+'\nimport time\ntime.sleep(60)\n');binary.chmod(0o700);shader=self.root/'shader';shader.write_bytes(b'cpu fixture')
  m={'runners':{'mlx':{'guardian':{'dataDir':str(self.root/'data'),'configDir':str(self.root/'config'),'port':58099,'cleanupSeconds':10},'apiUrl':'http://127.0.0.1:58099','workerId':'cpu_fixture','apiBinary':{'path':str(binary),'sha256':g.file_sha(binary)},'workerBinary':{'path':str(binary),'sha256':g.file_sha(binary)},'nativeResources':{'mlxMetallib':{'path':str(shader),'sha256':g.file_sha(shader),'bytes':shader.stat().st_size}}}}}
  (self.root/'config').mkdir();physical=type('CPU',(),{'locks':type('NoGPU',(),{'handles':[]})()})();s=g.Services(m,'mlx',self.root,physical);prior=os.environ.get('SCENEWORKS_TOKEN');os.environ['SCENEWORKS_TOKEN']='owned-test-token'
  try:
   s.start('api');s.start('download_worker');p={'jobs':{'key':{}}};self.assertTrue(c.cleanup(s,lambda *_:(_ for _ in ()).throw(EOFError('lost')),({'id':'job'},self.a,'key'),p));self.assertTrue(p['servicesDrained']);self.assertTrue(all(x['closure']['reaped'] and x['closure']['descendantsDrained'] for x in p['servicesClosure']))
  finally:
   s.close()
   if prior is None:os.environ.pop('SCENEWORKS_TOKEN',None)
   else:os.environ['SCENEWORKS_TOKEN']=prior
if __name__=='__main__':unittest.main()
