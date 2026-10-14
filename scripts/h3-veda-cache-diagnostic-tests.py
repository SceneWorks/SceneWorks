"""CPU metadata/source mutation tests. Never invoke Cargo/Rustc/product/native code."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).with_name('h3-veda-cache-diagnostic.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
F=Path(__file__).parent/'fixtures/h3-cache-failure-299'
class Tests(unittest.TestCase):
 def xattr_probe(self,root,names='com.example.binary\n',hex_value='00 FF 01 80\n',rc=0):
  calls=[]
  def probe(name,argv):
   calls.append((name,argv));out=root/(name+'.txt');out.write_text(names if name=='macro-xattr-list' else hex_value)
   return {'rc':rc,'stdout':d.b.binding(out)}
  return probe,calls
 def test_portable_collector_needs_no_os_xattr_api_and_hashes_binary_bytes(self):
  from types import SimpleNamespace
  import hashlib
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();file=root/'owned';file.write_bytes(b'CPU fixture');probe,calls=self.xattr_probe(root)
   with patch.object(d,'os',SimpleNamespace()):
    stat=d.dylib_stat(file);attrs=d.collect_xattrs(file,probe)
   self.assertTrue(stat['device']);self.assertEqual(attrs['attributes']['com.example.binary']['sha256'],hashlib.sha256(bytes([0,255,1,128])).hexdigest());self.assertEqual(calls[0][1],['/usr/bin/xattr',str(file)]);self.assertEqual(calls[1][1],['/usr/bin/xattr','-p','-x','com.example.binary',str(file)])
 def test_xattr_empty_census_is_explicit_and_not_an_omitted_collection(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();probe,calls=self.xattr_probe(root,names='');r=d.collect_xattrs(root/'owned',probe);self.assertEqual(r['attributes'],{});self.assertEqual(len(calls),1);self.assertEqual(r['mechanism'],'supervised_/usr/bin/xattr_hex')
 def test_xattr_cli_failure_or_invalid_hex_never_becomes_empty_metadata(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();probe,calls=self.xattr_probe(root,rc=1)
   with self.assertRaisesRegex(RuntimeError,'enumeration failed'):d.collect_xattrs(root/'owned',probe)
   probe,calls=self.xattr_probe(root,hex_value='NOT HEX')
   with self.assertRaisesRegex(RuntimeError,'invalid xattr'):d.collect_xattrs(root/'owned',probe)
 def test_xattr_duplicate_or_option_name_refuses_before_value_command(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve()
   for names in ['com.example\ncom.example\n','-w\n']:
    probe,calls=self.xattr_probe(root,names=names)
    with self.assertRaisesRegex(RuntimeError,'name census'):d.collect_xattrs(root/'owned',probe)
    self.assertEqual(len(calls),1)
 def test_actual_full_cargo_json_and_failure_recipe(self):

  t,c,a=d.frozen(F);self.assertEqual(c['rc'],101);self.assertEqual(a['target']['kind'],['proc-macro']);self.assertEqual(c['argv'][1:],['build','--release','--locked','--message-format=json-render-diagnostics','-p','sceneworks-rust-api']);self.assertNotIn('tools',c);self.assertTrue(t['prepared'])
 def test_tampered_original_logs_cannot_be_diagnostic_authority(self):
  import shutil
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/'fixture';shutil.copytree(F,f);(f/'cargo-stderr.txt').write_text('replacement')
   with self.assertRaisesRegex(RuntimeError,'fixture changed'):d.frozen(f)
 def test_prior_live_child_refuses_before_original_artifact_use(self):
  t,c,a=d.frozen(F)
  with patch.object(d,'frozen',return_value=(t,c,a)),patch.object(d.b,'no_links'),patch.object(d.b,'sha',return_value='same'),patch.object(d.b,'binding',side_effect=lambda p: t['stdout'] if 'stdout' in str(p) else t['stderr']),patch.object(d.b,'pid_birth',return_value=t['birth']):
   with self.assertRaisesRegex(RuntimeError,'still active'):d.verify_old(F)
 def test_vendor_requires_original_lock_exact_package_identity(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();old=root/'old';(old/'download-source').mkdir(parents=True);(old/'download-source/Cargo.lock').write_text('version=4\npackage=[]\n')
   with patch.object(d,'OLD',old):
    with self.assertRaisesRegex(RuntimeError,'original lock'):d.vendor(F,root)
 def make_vendor_fixture(self,root,bad_member=False):
  import hashlib,io,tarfile
  f=root/'fixture';f.mkdir();rows=json.loads((F/'tiny-dependencies.json').read_text());old=root/'old';cache=old/'tools/cargo/registry/cache/index.crates.io-1949cf8c6b5b557f';cache.mkdir(parents=True)
  for row in rows:
   name=row['name']+'-'+row['version'];p=cache/(name+'.crate')
   with tarfile.open(p,'w:gz') as t:
    member=tarfile.TarInfo(name+'/Cargo.toml');data=b'[package]\nname="fixture"\nversion="0.0.0"\n';member.size=len(data);t.addfile(member,io.BytesIO(data))
    if bad_member and row['name']=='bytemuck':
     member=tarfile.TarInfo(name+'/escape');member.type=tarfile.SYMTYPE;member.linkname='/foreign';t.addfile(member)
   row['checksum']=hashlib.sha256(p.read_bytes()).hexdigest()
  (f/'tiny-dependencies.json').write_text(json.dumps(rows));source=old/'download-source';source.mkdir();lock='version=4\n'
  for row in rows:
   lock+='\n[[package]]\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in row.items())
  (source/'Cargo.lock').write_text(lock);(source/'Cargo.toml').write_text('[workspace]\n');return f,old
 def test_vendor_preserves_exact_six_lock_rows_and_private_source_without_execution(self):
  import tomllib
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();f,old=self.make_vendor_fixture(root);dest=root/'destination';dest.mkdir()
   with patch.object(d,'OLD',old):source,archives=d.vendor(f,dest)
   rows=tomllib.loads((source/'Cargo.lock').read_text())['package'];self.assertEqual(len(rows),7);self.assertEqual(next(r for r in rows if r['name']=='syn')['version'],'2.0.119');self.assertEqual(len(archives),6);self.assertIn('offline=true',(source/'.cargo/config.toml').read_text());self.assertFalse((old/'vendor').exists())
 def test_vendor_refuses_archive_symlink_before_any_compiler_execution(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();f,old=self.make_vendor_fixture(root,True);dest=root/'destination';dest.mkdir()
   with patch.object(d,'OLD',old):
    with self.assertRaisesRegex(RuntimeError,'unsafe archive'):d.vendor(f,dest)
 def test_vendor_refuses_tampered_exact_archive(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();f,old=self.make_vendor_fixture(root);archive=next((old/'tools/cargo/registry/cache/index.crates.io-1949cf8c6b5b557f').iterdir());archive.write_bytes(b'tampered');dest=root/'destination';dest.mkdir()
   with patch.object(d,'OLD',old):
    with self.assertRaisesRegex(RuntimeError,'checksum'):d.vendor(f,dest)
 def test_source_bound_offline_recipe_has_no_remote_fetch_or_product_launch(self):

  s=Path(d.__file__).read_text();self.assertIn("'--locked','--offline','-vv'",s);self.assertIn('historicalCompilerEnvironmentRecorded',s);self.assertNotIn("'curl'",s);self.assertNotIn('sceneworks-rust-api\'',s);self.assertNotIn('chmod',s)
if __name__=='__main__':unittest.main()
