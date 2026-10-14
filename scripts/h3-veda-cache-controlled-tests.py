"""CPU source/receipt mutation tests. No compiler, native GPU, or API execution."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('controlled',Path(__file__).with_name('h3-veda-cache-controlled.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
F=Path(__file__).parent/'fixtures/h3-cache-failure-299'
class Tests(unittest.TestCase):
 def test_full_actual_old_syn_host_union_is_preserved(self):
  units=c.reference_units(F);features=units[c.SYNTAX_ID]['features'];self.assertEqual(features,['clone-impls','default','derive','extra-traits','fold','full','parsing','printing','proc-macro','visit','visit-mut']);self.assertEqual(units[c.DERIVE_ID]['profile']['opt_level'],'0')
 def test_actual_feature_or_optimization_mutation_refuses(self):
  expected=c.reference_units(F);actual=list(json.loads(json.dumps(expected)).values());c.validate_units(actual,expected)
  syn=next(x for x in actual if x['package_id']==c.SYNTAX_ID);syn['features'].remove('full')
  with self.assertRaisesRegex(RuntimeError,'feature/profile'):c.validate_units(actual,expected)
  actual=list(json.loads(json.dumps(expected)).values());actual[0]['profile']['opt_level']='3'
  with self.assertRaisesRegex(RuntimeError,'feature/profile'):c.validate_units(actual,expected)
 def test_matched_source_unifies_host_syn_without_changing_versions_or_release(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();source=root/'tiny';source.mkdir();(source/'Cargo.toml').write_text('[package]\nname="fixture"\nversion="0.0.0"\n');(source/'Cargo.lock').write_text('dependencies=["bytemuck"]\n')
   with patch.object(c.d,'vendor',return_value=(source,[])):s,archives,units=c.matched_source(F,root)
   text=(s/'Cargo.toml').read_text();self.assertIn('[build-dependencies]',text);self.assertIn('version="=2.0.119"',text);self.assertIn('visit-mut',text);self.assertNotIn('[profile',text);self.assertEqual((s/'build.rs').read_text(),'fn main() {}\n');self.assertIn('["bytemuck", "syn"]',(s/'Cargo.lock').read_text())
 def test_actual_macho_offsets_read_from_metadata_not_guessed(self):
  text='cmd LC_SYMTAB\n cmdsize 24\n symoff 2076080\n nsyms 9521\n stroff 2228876\n strsize 1071776\ncmd LC_BUILD_VERSION\ncmdsize 32\nplatform 1\nminos 26.2\nsdk 27.0\n';self.assertEqual(c.offsets(text)['stroff'],2228876);self.assertEqual(c.offsets(text)['minos'],'26.2')
  with self.assertRaisesRegex(RuntimeError,'unavailable'):c.offsets('unparseable')
 def test_artifact_outside_owned_target_or_symlink_refuses(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();outside=root/'foreign.dylib';outside.write_bytes(b'CPU fake metadata');target=root/'target';target.mkdir()
   with self.assertRaisesRegex(RuntimeError,'escaped'):c.macro_path({'filenames':[str(outside)]},target)
   (target/'link.dylib').symlink_to(outside)
   with self.assertRaisesRegex(RuntimeError,'symlink'):c.macro_path({'filenames':[str(target/'link.dylib')]},target)
 def test_internal_admission_refuses_wrong_uid_or_low_space(self):
  import os
  with tempfile.TemporaryDirectory() as tmp:
   temp=Path(tmp).resolve();head='a'*40;root=temp/('h3-cache-controlled-'+head[:9]);root.mkdir();p={'runDir':str(root),'head':head,'runnerTemp':{'path':str(temp),'uid':os.getuid()},'owner':{'runnerId':5281}}
   with patch.object(c.sys,'platform','darwin'),patch.object(c.os,'uname',return_value=type('Host',(),{'machine':'arm64'})()),patch.dict(c.os.environ,{'RUNNER_TEMP':str(temp)}),patch.object(c.shutil,'disk_usage',return_value=type('Disk',(),{'free':c.FLOOR-1})()):
    with self.assertRaisesRegex(RuntimeError,'free space'):c.root_admission(p)
   p['runnerTemp']['uid']+=1
   with patch.object(c.sys,'platform','darwin'),patch.object(c.os,'uname',return_value=type('Host',(),{'machine':'arm64'})()),patch.dict(c.os.environ,{'RUNNER_TEMP':str(temp)}):
    with self.assertRaisesRegex(RuntimeError,'UID'):c.root_admission(p)
 def test_conditional_comparison_requires_reproduced_baseline_and_successful_changed_loader(self):
  from unittest.mock import Mock
  good={'cargoRc':0,'loaderRc':0,'invalidLinkeditReproduced':False};runner=Mock(return_value=good)
  self.assertEqual(c.conditional_comparison({'loaderRc':0,'invalidLinkeditReproduced':False},runner),(None,False));runner.assert_not_called()
  self.assertEqual(c.conditional_comparison({'loaderRc':0,'invalidLinkeditReproduced':True},runner),(None,False));runner.assert_not_called()
  baseline={'loaderRc':1,'invalidLinkeditReproduced':True};self.assertEqual(c.conditional_comparison(baseline,runner),(good,True));runner.assert_called_once()
  for changed in [{'cargoRc':101,'loaderRc':0,'invalidLinkeditReproduced':False},{'cargoRc':0,'loaderRc':1,'invalidLinkeditReproduced':False},{'cargoRc':0,'loaderRc':1,'invalidLinkeditReproduced':True}]:
   self.assertFalse(c.conditional_comparison(baseline,lambda:changed)[1])
if __name__=='__main__':unittest.main()
