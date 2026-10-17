"""CPU fixtures for the inline, fixed-owned-state diagnostic; no native process."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import textwrap
import unittest
from unittest.mock import patch

WORKFLOW=Path(__file__).resolve().parents[1]/'.github/workflows/windows-candle.yml'

class KleinDiagnostic(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        script=WORKFLOW.read_text().split("$diagnostic = @'\n",1)[1].split("\n          '@",1)[0]
        self.ns={'__name__':'cpu_fixture'}
        with patch.dict(os.environ,dict(RUNNER_TEMP=str(self.root),GITHUB_RUN_ID='fixture',GITHUB_RUN_ATTEMPT='1')):
            exec(compile(textwrap.dedent(script),'inline-klein-diagnostic','exec'),self.ns)
        self.state=self.root/'state';self.clone=self.root/'clone';self.out=self.root/'out'
        self.clone.mkdir();self.out.mkdir();self.ns.update(STATE=self.state,CLONE=self.clone,OUT=self.out)
        self.project=self.state/'data/projects/test.sceneworks';self.project.mkdir(parents=True)
        self.job=self.ns['JOB'];self.asset={'id':'asset_fixture','projectId':'project_fixture','lineage':{'jobId':self.job},'file':{'path':'assets/images/fixture.png','mimeType':'image/png','width':1024,'height':1024},'recipe':{'model':'flux2_klein_9b','rawAdapterSettings':{'mlxQuantize':None}}}
        (self.state/'data/recent-projects.json').write_text(json.dumps([dict(id='project_fixture',path=str(self.project)),dict(id='unrelated',path='/not/read')]))
        (self.project/'project.json').write_text(json.dumps(dict(id='project_fixture')))
        self.png=b'\x89PNG\r\n\x1a\n'+b'\0'*8+(1024).to_bytes(4,'big')*2
        image=self.project/'assets/images/fixture.png';image.parent.mkdir(parents=True);image.write_bytes(self.png)
        image.with_suffix('.sceneworks.json').write_text(json.dumps(self.asset))
        recipes=self.project/'recipes';recipes.mkdir();(recipes/'asset_fixture.recipe.json').write_text(json.dumps(self.asset['recipe']))
        conn=sqlite3.connect(self.clone/'jobs.db')
        conn.executescript('CREATE TABLE jobs(id TEXT PRIMARY KEY,status TEXT,project_id TEXT,result_json TEXT); CREATE TABLE generation_metrics(job_id TEXT PRIMARY KEY,model TEXT,quant_label TEXT);')
        conn.execute('INSERT INTO jobs VALUES(?,?,?,?)',(self.job,'completed','project_fixture',json.dumps({'assets':[self.asset]})))
        conn.execute('INSERT INTO generation_metrics VALUES(?,?,?)',(self.job,'flux2_klein_9b','actual unclassified label'))
        conn.execute('INSERT INTO jobs VALUES(?,?,?,?)',('unrelated','completed','unrelated','{"secret":"do not export"}'))
        conn.commit();conn.close()
    def collect(self):return self.ns['collect']()
    def test_exact_rows_and_asset_bytes_only_with_original_hashes(self):
        result=self.collect();selected=json.loads((self.out/'selected-state.json').read_text())
        self.assertEqual(selected['job']['id'],self.job);self.assertNotIn('unrelated',(self.out/'selected-state.json').read_text())
        self.assertEqual(selected['metrics']['quant_label'],'actual unclassified label')
        self.assertEqual((self.out/'original.png').read_bytes(),self.png)
        self.assertEqual(result['png_dimensions'],[1024,1024]);self.assertEqual(len(result['files']),5)
        for row in result['files']:
            self.assertEqual(row['sha256'],hashlib.sha256(Path(row['source']).read_bytes()).hexdigest())
        self.assertFalse((self.out/'jobs.db').exists())
    def test_missing_metrics_is_explicit_failure(self):
        c=sqlite3.connect(self.clone/'jobs.db');c.execute('DELETE FROM generation_metrics');c.commit();c.close()
        with self.assertRaisesRegex(ValueError,'exact target job/metrics'):self.collect()
    def test_sql_clone_is_not_modified(self):
        path=self.clone/'jobs.db';before=path.read_bytes();self.collect();self.assertEqual(before,path.read_bytes())
    def test_readonly_snapshot_includes_uncheckpointed_copied_wal(self):
        conn=sqlite3.connect(self.clone/'jobs.db');self.addCleanup(conn.close)
        conn.execute('PRAGMA journal_mode=WAL');conn.execute('UPDATE generation_metrics SET quant_label=?',('wal-only value',));conn.commit()
        wal=self.clone/'jobs.db-wal';self.assertTrue(wal.exists());before=(self.clone/'jobs.db').read_bytes(),wal.read_bytes()
        self.collect();selected=json.loads((self.out/'selected-state.json').read_text())
        self.assertEqual(selected['metrics']['quant_label'],'wal-only value')
        self.assertEqual(before,((self.clone/'jobs.db').read_bytes(),wal.read_bytes()))
    def test_reparse_clone_sidecar_is_refused_before_sqlite(self):
        target=self.root/'unrelated';target.write_bytes(b'not a WAL')
        (self.clone/'jobs.db-wal').symlink_to(target)
        with self.assertRaisesRegex(ValueError,'reparse'):self.collect()
    def test_source_size_cap_refuses_copy(self):
        self.ns['MAX_PNG']=1;report={}
        with self.assertRaisesRegex(ValueError,'byte cap'):self.ns['collect'](report)
        self.assertFalse((self.out/'original.png').exists());self.assertTrue(report['files'])
    def test_count_cap_refuses_unbounded_selected_files(self):
        self.ns['MAX_FILES']=2
        with self.assertRaisesRegex(ValueError,'file count cap'):self.collect()
    def test_reparse_ancestor_is_refused(self):
        external=self.root/'external';external.mkdir();(external/'file').write_bytes(b'not read')
        link=self.root/'link';link.symlink_to(external,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'reparse'):self.ns['bounded'](link/'file',100,[])
    def test_foreign_asset_is_not_copied(self):
        path=self.project/'assets/images/fixture.sceneworks.json';a=json.loads(path.read_text());a['lineage']['jobId']='foreign';path.write_text(json.dumps(a))
        with self.assertRaisesRegex(ValueError,'unrelated sidecar'):self.collect()
        self.assertFalse((self.out/'asset.sceneworks.json').exists())
    def test_project_escape_is_refused(self):
        registry=self.state/'data/recent-projects.json';registry.write_text(json.dumps([dict(id='project_fixture',path=str(self.root))]))
        with self.assertRaisesRegex(ValueError,'project escaped'):self.collect()

if __name__=='__main__':unittest.main()
