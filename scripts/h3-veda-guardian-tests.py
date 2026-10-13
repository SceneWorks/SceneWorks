"""CPU failure proofs. No product, GPU, NVIDIA/Metal probe or network server starts."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("guard",Path(__file__).with_name("h3-veda-guardian.py"));g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

class Physical:
    def __init__(self,events):self.events=events;self.held=True;self.reject=False
    def probe(self,phase,owned):
        self.events.append("probe:"+phase)
        if self.reject:raise RuntimeError("foreign queue or physical activity")
        return {"phase":phase,"admitted":True,"observedAt":int(time.time()*1000),"queueObservedAt":int(time.time()*1000)}
    def owned(self):return self.held
    def close(self):self.events.append("unlock");self.held=False

class Services:
    def __init__(self,events):self.events=events;self.children={"api":{}}
    def start(self,name):self.events.append("start:"+name);self.children[name]={}
    def stop(self,name):self.events.append("stop:"+name);self.children.pop(name,None)
    def close(self):self.events.append("drain-owned-descendants");self.children={}
    def owned_pids(self):return {111}

class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup);self.root=Path(self.dir.name)
        self.m={"projectId":"owned","runners":{"mlx":{"workerId":"worker","guardian":{"maxSeconds":10,"cleanupSeconds":1}}}}
        self.events=[];self.alive=True;self.now=0;self.physical=Physical(self.events);self.services=Services(self.events);self.jobs={};self.variant="ok"
        self.owner=g.Guardian(self.m,"mlx",self.root,self.physical,self.services,self.api,lambda:self.alive,lambda:self.now)
    def intent(self,key="key",arm="dense"):
        req={"projectId":"owned","advanced":{"s9Harness":{"key":key,"fingerprint":g.digest(self.m),"arm":arm}}}
        return {"key":key,"request":req,"requestSha256":g.digest(req),"status":"submitting"}
    def persist(self,intent):
        g.atomic(self.root/"campaign.json",{"fingerprint":g.digest(self.m),"cells":{"cell":{"arms":{"dense":intent}}}})
        self.owner.before_phase("submit-or-reconcile")
    def api(self,method,route,body=None):
        self.events.append(method+":"+route)
        if route=="/api/v1/workers":return [{"id":"worker","status":"idle"}]
        if route=="/api/v1/video/jobs":
            journal=json.loads((self.root/"guardian-journal.json").read_text());self.assertEqual(journal["intents"][body["advanced"]["s9Harness"]["key"]]["status"],"submitting")
            self.assertIn("start:worker",self.events)
            job={"id":"job","projectId":"owned","payload":body,"status":"running"};self.jobs["job"]=job
            if self.variant=="transport":raise OSError("transport lost after commit")
            if self.variant=="refusal":raise g.ProductError(400,{"error":"declared for model"})
            return job
        if "?" in route:
            if self.variant=="duplicate":return [self.jobs["job"],self.jobs["job"]]
            if self.variant=="full":return [{}]*500
            return list(self.jobs.values())
        job_id=route.split("/")[4]
        if route.endswith("/cancel"):
            self.assertEqual(job_id,"job");self.jobs[job_id]["status"]="canceled";return {}
        return self.jobs[job_id]
    def test_acknowledgement_is_fsynced_before_ordinary_post_and_id_readback(self):
        intent=self.intent();self.persist(intent);r=self.owner.submit(intent)
        self.assertEqual(r["status"],201);self.assertEqual(json.loads((self.root/"guardian-journal.json").read_text())["intents"]["key"]["jobId"],"job")
        self.assertLess(self.events.index("start:worker"),self.events.index("POST:/api/v1/video/jobs"))
    def test_undurable_and_mutated_intents_never_start_worker_or_post(self):
        intent=self.intent()
        with self.assertRaises(Exception):self.owner.submit(intent)
        self.persist(intent);intent["request"]["projectId"]="foreign"
        with self.assertRaises(Exception):self.owner.submit(intent)
        self.assertNotIn("start:worker",self.events);self.assertFalse(self.jobs)
    def test_controller_sigkill_semantics_guardian_cancels_drains_before_unlock(self):
        i=self.intent();self.persist(i);self.owner.submit(i);self.alive=False;self.owner.tick()
        self.assertEqual(self.jobs["job"]["status"],"canceled");self.assertTrue(self.owner.closed)
        self.assertLess(self.events.index("POST:/api/v1/jobs/job/cancel"),self.events.index("drain-owned-descendants"));self.assertLess(self.events.index("drain-owned-descendants"),self.events.index("unlock"))
    def test_lost_post_response_reconciles_actual_product_job_without_second_post(self):
        self.variant="transport";i=self.intent();self.persist(i)
        with self.assertRaises(OSError):self.owner.submit(i)
        self.alive=False;self.owner.tick();self.assertTrue(self.owner.closed)
        self.assertEqual(self.events.count("POST:/api/v1/video/jobs"),1)
    def test_absent_duplicate_and_full_history_keep_lock_after_owned_cleanup(self):
        for variant in ["absent","duplicate","full"]:
            self.owner=g.Guardian(self.m,"mlx",self.root,self.physical,self.services,self.api,lambda:False,lambda:self.now);i=self.intent();self.owner.intents={"key":{**i,"status":"ambiguous"}};self.variant=variant;self.physical.held=True
            self.jobs={} if variant=="absent" else {"job":{"id":"job","status":"running","projectId":"owned","payload":i["request"]}}
            with self.assertRaises(RuntimeError):self.owner.tick()
            self.assertTrue(self.physical.held);self.assertFalse(self.services.children)
            self.assertNotIn("unlock",self.events)
    def test_expired_guardian_uses_same_owned_cleanup(self):
        i=self.intent();self.persist(i);self.owner.submit(i);self.now=11;self.owner.tick();self.assertTrue(self.owner.closed)
    def test_foreign_attach_and_cancel_never_touch_foreign_job(self):
        i=self.intent();self.persist(i);self.owner.submit(i);self.jobs["foreign"]={"id":"foreign","projectId":"someone-else","payload":i["request"],"status":"running"}
        with self.assertRaises(RuntimeError):self.owner.attach("key","foreign")
        self.owner.intents["key"]["jobId"]="foreign"
        with self.assertRaises(RuntimeError):self.owner.drain()
        self.assertNotIn("POST:/api/v1/jobs/foreign/cancel",self.events);self.assertTrue(self.physical.held)
    def test_next_phase_cannot_probe_or_restart_worker_while_owned_job_active(self):
        i=self.intent();self.persist(i);self.owner.submit(i)
        count=self.events.count("probe:submit-or-reconcile")
        with self.assertRaises(RuntimeError):self.owner.before_phase("submit-or-reconcile")
        self.assertEqual(self.events.count("probe:submit-or-reconcile"),count)
        self.jobs["job"]["status"]="completed";self.owner.before_phase("submit-or-reconcile")
        self.assertLess(self.events.index("stop:worker"),self.events.index("probe:submit-or-reconcile"))
    def test_physical_conflict_keeps_canonical_lock(self):
        self.physical.reject=True
        with self.assertRaises(RuntimeError):self.owner.before_phase("next")
        self.assertTrue(self.physical.held)
    def test_source_helper_same_size_corruption_refused_before_import(self):
        p=self.root/"policy.py";p.write_text("VALUE=1\n");sha=g.file_sha(p);p.write_text("VALUE=2\n")
        with self.assertRaises(RuntimeError):g.source_module(self.root,"policy.py",sha)
    @unittest.skipIf(os.name=="nt","POSIX flock CPU proof runs on POSIX only; WindowsJob policy tested below with fake API")
    def test_real_cpu_flock_contention_and_path_inode_replacement_refused(self):
        p=self.root/"global.lock";one=g.Locks([p]);two=g.Locks([p]);one.claim();self.addCleanup(one.close)
        with self.assertRaises(OSError):two.claim()
        self.assertTrue(one.owned());p.unlink();p.touch();self.assertFalse(one.owned())
    def test_prior_unresolved_journal_is_preserved_on_new_owner(self):
        i=self.intent();self.owner.intents={"key":i};self.owner.save()
        owner=g.Guardian(self.m,"mlx",self.root,self.physical,self.services,self.api,lambda:True,lambda:self.now)
        with self.assertRaises(RuntimeError):owner.before_phase("next")
        self.assertEqual(owner.intents["key"]["requestSha256"],i["requestSha256"])

    def test_actual_frozen_cuda_policy_keeps_five_timed_samples_and_all_negative_refusals(self):
        source=Path(__file__).with_name("fixtures")/"h3-veda"/"cuda-policy.py"
        cuda=g.source_module(source.parent,"cuda-policy.py",g.file_sha(source))
        from types import SimpleNamespace
        device={"index":0,"uuid":"GPU-a","used_mib":1000,"utilization_percent":0}
        base_rows=[["GPU-a","10","desktop","C+G"]]
        def run(variant):
            times=[0.0];calls=[]
            cuda.time=SimpleNamespace(monotonic=lambda:times[0],sleep=lambda n:times.__setitem__(0,times[0]+n))
            def inventory(*args):
                calls.append(args);d=dict(device);rows=[list(x) for x in base_rows]
                if variant=="pureC":rows[0][3]="C"
                if variant=="unknown":rows[0][3]="unknown"
                if variant=="high-memory":d["used_mib"]=2048
                if variant=="high-util":d["utilization_percent"]=5
                if variant=="rising" and len(calls)>1:d["used_mib"]+=1
                if variant=="changedPID" and len(calls)>1:rows[0][1]="11"
                if variant=="changedUUID" and len(calls)>1:d["uuid"]="GPU-other"
                if variant=="timeout":times[0]=6
                return [d],rows
            cuda.inventory=inventory
            result=cuda.sample_baseline(device,self.root,"frozen-policy")
            return calls,result
        calls,_=run("ok");self.assertEqual(len(calls),5)
        receipt=json.loads((self.root/"frozen-policy-samples.json").read_text());self.assertEqual([x["elapsed_seconds"] for x in receipt],[0,.5,1,1.5,2])
        for variant in ["pureC","unknown","high-memory","high-util","rising","changedPID","changedUUID","timeout"]:
            with self.subTest(variant=variant),self.assertRaises(RuntimeError):run(variant)
    def test_normal_worker_launch_ownership_failure_kills_only_suspended_owned_child(self):
        from types import SimpleNamespace
        events=self.events;self.physical.device={"index":0};binary=self.root/"fake-api";binary.write_bytes(b"never executed")
        m={"runners":{"candle_cuda":{"guardian":{"dataDir":str(self.root/"data"),"configDir":str(self.root/"config"),"port":18000},"apiUrl":"http://127.0.0.1:18000","workerId":"owned-worker","requestedGpu":"0","physicalGpuId":"GPU-a","apiBinary":{"path":str(binary),"sha256":g.file_sha(binary)},"workerBinary":{"path":str(binary),"sha256":g.file_sha(binary)}}},"sourceFreeze":{"inference":{"root":"/frozen"},"inferenceFiles":{"scripts/media_campaign_windows.py":"frozen"}}}
        class Job:
            def assign_and_resume(self,child):events.append("assign-suspended");raise RuntimeError("job assignment refused")
            def close(self):events.append("close-job")
        class Child:
            pid=111
            def kill(self):events.append("kill-owned-111")
            def wait(self,timeout):events.append("reap-owned-111")
        def spawn(args,**kw):
            self.assertEqual(args,[str(binary)]);self.assertEqual(kw["creationflags"],0x00000004);self.assertEqual(kw["env"]["SCENEWORKS_WORKER_ONLY"],"1");self.assertEqual(kw["env"]["SCENEWORKS_DATA_DIR"],str(self.root/"data"));self.assertEqual(kw["env"]["SCENEWORKS_PARENT_PID"],"999");return Child()
        os_fake=SimpleNamespace(name="nt",environ={"SCENEWORKS_TOKEN":"private-token","SCENEWORKS_GPU_CHECK":"1"},getpid=lambda:999)
        with patch.object(g,"os",os_fake),patch.object(g,"source_module",return_value=SimpleNamespace(WindowsJob=Job)),patch.object(g.subprocess,"Popen",side_effect=spawn):
            services=g.Services(m,"candle_cuda",self.root,self.physical)
            with self.assertRaises(RuntimeError):services.start("worker")
            self.assertFalse(services.children)
            for log in services.logs:log.close()
        self.assertEqual(events,["assign-suspended","kill-owned-111","reap-owned-111","close-job"])
    def test_cleanup_reap_failure_never_closes_physical_token(self):
        def unsafe_close():raise RuntimeError("owned descendant still alive")
        self.services.close=unsafe_close
        with self.assertRaises(RuntimeError):self.owner.drain()
        self.assertTrue(self.physical.held);self.assertNotIn("unlock",self.events)

    def test_stale_phase_and_lost_actual_lock_prevent_any_product_post(self):
        i=self.intent();self.persist(i);self.owner.last_phase["observedAt"]=0
        with self.assertRaises(RuntimeError):self.owner.submit(i)
        self.assertNotIn("POST:/api/v1/video/jobs",self.events)
        self.owner.before_phase("submit-or-reconcile");self.physical.held=False
        with self.assertRaises(RuntimeError):self.owner.submit(i)
        self.assertFalse(self.jobs)
    def test_duplicate_ack_cannot_erase_already_submitted_job_identity(self):
        i=self.intent();self.persist(i);self.owner.submit(i)
        with self.assertRaises(RuntimeError):self.owner.submit(i)
        self.assertEqual(self.owner.intents["key"]["jobId"],"job");self.assertEqual(self.owner.intents["key"]["status"],"running")

    def test_physical_owner_secret_is_not_persisted_in_evidence_journal(self):
        i=self.intent();self.persist(i);self.owner.submit(i)
        journal=(self.root/"guardian-journal.json").read_text()
        self.assertNotIn(self.owner.token,journal)

if __name__=="__main__":unittest.main()
