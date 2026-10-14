"""CPU failure proofs. No product, GPU, NVIDIA/Metal probe or network server starts."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location("guard",Path(__file__).with_name("h3-veda-guardian.py"));g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

class GlobalMacProfileTests(unittest.TestCase):
    def profile(self, root):
        from types import SimpleNamespace
        home = root / "home"; home.mkdir()
        epic = home / "Documents/Codex/2026-10-10/epic-25140"; epic.mkdir(parents=True)
        paths = [epic / "s0-gpu-driver.lock", epic / "s2s4-aggregate-target.lock"]
        account = SimpleNamespace(pw_name="MTrefry", pw_dir=str(home))
        return home, paths, account

    def test_profiles_are_fixed_host_global_and_never_follow_generic_overrides(self):
        mac1 = list(g.MAC_LOCKS); mac2 = list(g.MAC2_LOCKS)
        with patch.dict(os.environ, {"EPIC": "/tmp/arbitrary", "HOME": "/tmp/arbitrary", "S9_RUN_DIR": "/tmp/arbitrary"}):
            self.assertEqual(g.canonical_mac_locks("D299B954-4C63-5322-86A5-BD5667051634"), mac1)
            self.assertEqual(g.canonical_mac_locks(g.MAC2_UUID), mac2)
            with self.assertRaisesRegex(RuntimeError, "no source-owned"): g.canonical_mac_locks("unknown")
        self.assertEqual(str(mac2[0]), "/Users/MTrefry/Documents/Codex/2026-10-10/epic-25140/s0-gpu-driver.lock")
        self.assertNotIn(".h3-veda-epic", str(mac2[0]))

    def test_initial_profile_binds_actual_uid_home_uuid_and_refuses_foreign_host(self):
        import pwd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); home, paths, account = self.profile(root)
            with patch.object(g, "MAC2_HOME", home), patch.object(g, "MAC2_LOCKS", paths), patch.object(g, "MAC_LOCKS", [root / "absent-legacy"]), patch.object(g.sys, "platform", "darwin"), patch.object(pwd, "getpwuid", return_value=account), patch.dict(os.environ, {"HOME": str(home), "RUNNER_NAME": "nax-macos-2"}), patch.object(g, "command", return_value=json.dumps({"SPHardwareDataType": [{"platform_UUID": g.MAC2_UUID}]})), patch.object(g.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "p123\nn/ordinary\n", "")):
                fact = g.qualify_mac2_global_locks(g.MAC2_UUID)
                self.assertFalse(fact["previouslyObservedProfile"]); self.assertEqual(fact["account"]["uid"], os.getuid()); self.assertEqual([x["path"] for x in fact["canonicalLocks"]], list(map(str, paths)))
                with patch.dict(os.environ, {"HOME": "/foreign"}):
                    with self.assertRaisesRegex(RuntimeError, "account/HOME/UID"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                with patch.object(g, "command", return_value=json.dumps({"SPHardwareDataType": [{"platform_UUID": "foreign"}]})):
                    with self.assertRaisesRegex(RuntimeError, "actual UUID"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                with self.assertRaisesRegex(RuntimeError, "exact admitted"): g.qualify_mac2_global_locks("foreign")

    def test_initial_profile_refuses_legacy_namespace_symlink_hardlink_and_held_descriptors(self):
        import pwd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); home, paths, account = self.profile(root); legacy = root / "legacy"
            with patch.object(g, "MAC2_HOME", home), patch.object(g, "MAC2_LOCKS", paths), patch.object(g, "MAC_LOCKS", [legacy]), patch.object(g.sys, "platform", "darwin"), patch.object(pwd, "getpwuid", return_value=account), patch.dict(os.environ, {"HOME": str(home), "RUNNER_NAME": "nax-macos-2"}), patch.object(g, "command", return_value=json.dumps({"SPHardwareDataType": [{"platform_UUID": g.MAC2_UUID}]})), patch.object(g.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "p123\nn/ordinary\n", "")):
                legacy.write_text("legacy")
                with self.assertRaisesRegex(RuntimeError, "Mac1 literal"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                legacy.unlink(); paths[0].symlink_to(root / "alternate")
                with self.assertRaisesRegex(RuntimeError, "alternate/symlink"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                paths[0].unlink(); paths[0].write_text("lock"); os.link(paths[0], root / "alias")
                with self.assertRaisesRegex(RuntimeError, "hardlink alias"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                (root / "alias").unlink()
                with patch.object(g.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "p123\nn" + str(paths[0]) + "\n", "")):
                    with self.assertRaisesRegex(RuntimeError, "held descriptors"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                with patch.object(g.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "p123\nn" + str(legacy) + " (deleted)\n", "")):
                    with self.assertRaisesRegex(RuntimeError, "held descriptors"): g.qualify_mac2_global_locks(g.MAC2_UUID)
                with patch.object(g.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "p123\nn/ordinary\n", "")):
                    self.assertTrue(g.qualify_mac2_global_locks(g.MAC2_UUID)["canonicalLocks"][0]["existed"])

    def test_actual_cpu_held_flock_blocks_global_claim_and_path_replacement_loses_identity(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / "s0-global.lock", Path(temporary) / "aggregate-global.lock"]
            held = paths[0].open("a+b"); fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaises(BlockingIOError): g.Locks(paths).claim()
            finally: held.close()
            locks = g.Locks(paths); locks.claim()
            try:
                self.assertTrue(locks.owned()); paths[0].unlink(); paths[0].write_text("replacement"); self.assertFalse(locks.owned())
            finally: locks.close()


class Physical:
    def __init__(self,events):self.events=events;self.held=True;self.reject=False;self.services=None
    def probe(self,phase,owned,drained_api=None):
        self.events.append("probe:"+phase)
        if self.services is not None: g.require(not self.services.children,"probe observed live owned API/worker")
        if self.reject:raise RuntimeError("foreign queue or physical activity")
        return {"phase":phase,"admitted":True,"observedAt":int(time.time()*1000),"queueObservedAt":int(time.time()*1000),"drainedApi":drained_api}
    def owned(self):return self.held
    def close(self):self.events.append("unlock");self.held=False

class PythonBindingTests(unittest.TestCase):
    def test_actual_cpu_interpreter_and_original_attestation_mutations(self):
        with tempfile.TemporaryDirectory() as root:
            executable=Path(g.sys.executable).resolve();source='a'*64
            value={'context':{'sourceSha256':source},'interpreter':{'path':str(executable),'sha256':g.file_sha(executable)}}
            raw=json.dumps(value);att=Path(root)/'attestation.json';att.write_text(raw)
            p={'path':str(executable),'sha256':g.file_sha(executable),'bytes':executable.stat().st_size,'version':[3,12],'bootstrapAttestation':{'path':str(att),'originalJson':raw,'sha256':g.file_sha(att),'bytes':att.stat().st_size}}
            m={'runners':{'mlx':{'guardian':{'pythonPath':str(executable),'pythonBinding':p}}},'sourceFreeze':{'files':{'scripts/h3-veda-owned-python.mjs':source}}}
            g.validate_python_binding(m,'mlx')
            m['sourceFreeze']['files']['scripts/h3-veda-owned-python.mjs']='b'*64
            with self.assertRaises(RuntimeError):g.validate_python_binding(m,'mlx')
            m['sourceFreeze']['files']['scripts/h3-veda-owned-python.mjs']=source;att.write_text(raw+' ')
            with self.assertRaises(RuntimeError):g.validate_python_binding(m,'mlx')

class Services:
    def __init__(self,events):self.events=events;self.children={"api":{"pid":100,"birth":"mock-birth-100","apiBinarySha256":"a"*64}};self.pid=100
    def start(self,name):
        self.events.append("start:"+name);self.pid+=1;self.children[name]={"pid":self.pid,"birth":"mock-birth-"+str(self.pid),"apiBinarySha256":"a"*64,"workerBinarySha256":"b"*64,"workerConfigBinding":{"configFiles":[],"environment":{"MOCK":"CPU_ONLY"}}}
    def api_receipt(self):
        value=self.children["api"];binding={"configFiles":[{"path":"CPU_ONLY.json","sha256":"c"*64}],"environment":{"SCENEWORKS_GPU_ID":"cpu"}}
        return {"pid":value["pid"],"birth":value["birth"],"apiBinarySha256":value["apiBinarySha256"],"apiConfigBinding":binding,"apiConfigSha256":g.digest(binding)}
    def worker_receipt(self):
        value=self.children["worker"];return {**value,"workerConfigSha256":g.digest(value["workerConfigBinding"])}
    def stop(self,name):
        self.events.append("stop:"+name);item=self.children.pop(name,None)
        return None if item is None else {"pid":item["pid"],"birth":item["birth"],"reaped":True,"descendantsDrained":True,"exitCode":0}
    def close(self):self.events.append("drain-owned-descendants");self.children={}
    def owned_pids(self):return {111}

class ActionsOwnership(unittest.TestCase):
    def setUp(self):
        self.uuid="BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF"
        self.physical=g.Physical.__new__(g.Physical)
        self.physical.backend="mlx"
        self.physical.r={"physicalGpuId":self.uuid}
        self.physical.contract={"hostRunnerNames":["nax-macos-2"]}
        self.physical.m={"sourceFreeze":{"sceneworks":{"head":"a"*40}}}
        self.runners={"id":2310,"name":"nax-macos","status":"online","busy":True},{"id":5281,"name":"nax-macos-2","status":"online","busy":True}
        self.run={"id":123,"run_attempt":1,"event":"workflow_dispatch","head_sha":"a"*40,"status":"in_progress","conclusion":None,"name":"H3 Veda terminal owned campaign"}
        self.job={"id":456,"name":"mac2","runner_name":"nax-macos-2","run_id":123,"run_attempt":1,"head_sha":"a"*40,"status":"in_progress","conclusion":None,"runner_id":5281}
        self.env={"GITHUB_ACTIONS":"true","GH_TOKEN":"CPU_FAKE","GITHUB_REPOSITORY":"SceneWorks/SceneWorks","GITHUB_EVENT_NAME":"workflow_dispatch","GITHUB_SHA":"a"*40,"GITHUB_WORKFLOW_REF":"SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@refs/heads/main","GITHUB_JOB":"mac2","RUNNER_NAME":"nax-macos-2","GITHUB_RUN_ID":"123","GITHUB_RUN_ATTEMPT":"1"}
    def command(self,args,timeout=0):
        route=args[-1]
        if "orgs/SceneWorks/actions/runners" in route:return json.dumps([{"runners":self.runners}])
        if "lensRunning" in route or "lensQueued" in route:raise AssertionError("unexpected alias")
        if "michaeltrefry/LensWorks" in route:return json.dumps([{"workflow_runs":[]}])
        if route.endswith("/runs/123"):return json.dumps(self.run)
        if "/runs/123/jobs?" in route:return json.dumps([{"jobs":[self.job]}])
        if args[:2]==["system_profiler","SPHardwareDataType"]:return json.dumps({"SPHardwareDataType":[{"platform_UUID":self.uuid}]})
        raise AssertionError("unexpected command " + repr(args))
    def test_exact_actions_owner_allows_only_its_own_busy_mac2_on_distinct_host(self):
        with patch.dict(os.environ,self.env,clear=True),patch.object(g,"command",side_effect=self.command):
            self.assertEqual(self.physical.actions_owner({"orgRunners":[{"runners":self.runners}]}),5281)
            self.assertIsInstance(self.physical.queue()["observedAt"],int)
            self.job["runner_id"]=2310
            with self.assertRaisesRegex(RuntimeError,"identity ambiguous|runner not busy") : self.physical.queue()
    def test_stale_or_unknown_mac_topology_refuses_even_with_live_own_job(self):
        with patch.dict(os.environ,self.env,clear=True),patch.object(g,"command",side_effect=self.command):
            self.physical.r["physicalGpuId"]="D299B954-4C63-5322-86A5-BD5667051634"
            with self.assertRaisesRegex(RuntimeError,"topology unbound"):self.physical.queue()
            self.physical.r["physicalGpuId"]=self.uuid;self.runners[0]["id"]=999
            with self.assertRaisesRegex(RuntimeError,"topology changed"):self.physical.queue()
            self.runners[0]["id"]=2310;self.run["status"]="completed"
            with self.assertRaisesRegex(RuntimeError,"run identity"):self.physical.queue()

class ActionsAncestry(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name).resolve();(root/"bin").mkdir();self.exe=root/"bin/Runner.Worker";self.exe.write_text("CPU fake executable never launched");self.config=root/".runner";self.config.write_text(json.dumps({"agentId":5281,"agentName":"nax-macos-2"}));self.physical=g.Physical.__new__(g.Physical);self.physical.out=root;self.physical.actions_identity={"runId":123,"runAttempt":1,"runnerId":5281,"jobId":456,"sourceHead":"a"*40,"jobName":"mac2","workflowRef":"exact CPU test tuple"};self.argv=f"100 90 /python python CPU\n90 80 /node node CPU\n80 1 {self.exe} {self.exe} spawnclient CPU\n70 1 /foreign/bin/Runner.Worker /foreign/bin/Runner.Worker spawnclient foreign"
    def command(self,argv,timeout=0):return {100:"90",90:"80",80:"1"}[int(argv[2])]
    def test_only_exact_live_owned_runner_ancestor_is_exempted(self):
        with patch.dict(os.environ,{"RUNNER_NAME":"nax-macos-2"}),patch.object(g.os,"getpid",return_value=100),patch.object(g,"pid_birth",side_effect=lambda pid:"birth-"+str(pid)),patch.object(g,"command",side_effect=self.command):
            self.assertEqual(self.physical.actions_actor_proof(self.argv),{80});proof=json.loads((self.physical.out/"actions-ancestor-proof.json").read_text());self.assertEqual(proof["jobId"],456);self.assertEqual(len(proof["ancestry"]),3);self.assertNotIn(70,{x["pid"] for x in proof["ancestry"]})
    def test_foreign_runner_config_and_broken_ancestry_cannot_exempt(self):
        with patch.dict(os.environ,{"RUNNER_NAME":"nax-macos-2"}),patch.object(g.os,"getpid",return_value=100),patch.object(g,"pid_birth",return_value="birth"),patch.object(g,"command",side_effect=self.command):
            self.config.write_text(json.dumps({"agentId":2310,"agentName":"nax-macos"}))
            with self.assertRaisesRegex(RuntimeError,"another runner"):self.physical.actions_actor_proof(self.argv)
            with self.assertRaisesRegex(RuntimeError,"configuration unavailable|ancestry absent|path unbound"):self.physical.actions_actor_proof(self.argv.replace("100 90","100 70"))
    def test_pid_birth_or_parent_link_change_refuses_exemption(self):
        births={100:0,90:0,80:0}
        def birth(pid):births[pid]+=1;return "birth" if births[pid]==1 else "reused"
        with patch.dict(os.environ,{"RUNNER_NAME":"nax-macos-2"}),patch.object(g.os,"getpid",return_value=100),patch.object(g,"pid_birth",side_effect=birth),patch.object(g,"command",side_effect=self.command):
            with self.assertRaisesRegex(RuntimeError,"PID/birth/link changed"):self.physical.actions_actor_proof(self.argv)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.addCleanup(self.dir.cleanup);self.root=Path(self.dir.name)
        self.m={"campaignId":"cpu-campaign","sourceFreeze":{"CPU_ONLY":True},"projectId":"owned","runners":{"mlx":{"workerId":"worker","physicalGpuId":"CPU_ONLY_NOT_GPU","guardian":{"maxSeconds":10,"cleanupSeconds":1}}}}
        self.events=[];self.alive=True;self.now=0;self.physical=Physical(self.events);self.services=Services(self.events);self.jobs={};self.variant="ok"
        self.physical.services=self.services
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
        post=self.events.index("POST:/api/v1/video/jobs");stop=self.events.index("stop:api",post);start=self.events.index("start:api",post)
        probes=[n for n,event in enumerate(self.events) if event=="probe:submit-or-reconcile"]
        self.assertEqual(len(probes),2);self.assertLess(stop,probes[1]);self.assertLess(probes[1],start)
    def test_api_restart_failure_clears_prior_admission_without_unlocking(self):
        self.owner.before_phase("submit-or-reconcile")
        original=self.services.start
        def refuse(name):
            if name=="api":raise RuntimeError("owned API restart refused")
            return original(name)
        self.services.start=refuse
        with self.assertRaisesRegex(RuntimeError,"owned API restart refused"):self.owner.before_phase("submit-or-reconcile")
        self.assertIsNone(self.owner.last_phase);self.assertTrue(self.physical.held)
    def test_api_restart_cannot_refresh_stale_drained_probe(self):
        original=self.physical.probe
        def stale(phase,owned,drained_api=None):
            proof=original(phase,owned,drained_api)
            proof["queueObservedAt"]-=11000
            return proof
        self.physical.probe=stale
        with self.assertRaisesRegex(RuntimeError,"not ready; no physical admission"):self.owner.before_phase("submit-or-reconcile")
        self.assertIsNone(self.owner.last_phase);self.assertTrue(self.physical.held)
        self.assertLess(self.events.index("probe:submit-or-reconcile"),self.events.index("start:api"))
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
        (self.root/"config").mkdir()
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

    def test_completed_arm_exact_worker_lifetime_is_bound_and_drained_before_admission(self):
        i=self.intent();self.persist(i);self.owner.submit(i);self.jobs["job"]["status"]="completed"
        phase=self.owner.before_phase("collect-completed-evidence",{"key":"key","jobId":"job"});receipt=phase["drainedWorker"]
        self.assertEqual(receipt["admissionReceipt"],self.owner.intents["key"]["admissionReceipt"])
        self.assertEqual(receipt["sourceFreezeSha256"],g.digest(self.m["sourceFreeze"]));self.assertEqual(receipt["sourceFingerprint"],g.digest(self.m))
        self.assertEqual(receipt["workerConfigSha256"],g.digest(receipt["workerConfigBinding"]))
        self.assertTrue(receipt["reaped"] and receipt["descendantsDrained"]);self.assertNotIn(self.owner.token,json.dumps(receipt))
        self.assertLess(self.events.index("stop:worker",self.events.index("POST:/api/v1/video/jobs")),self.events.index("probe:collect-completed-evidence"))
        self.assertTrue(phase["drainedApi"]["reaped"] and phase["drainedApi"]["descendantsDrained"])
        self.assertLess(self.events.index("stop:api",self.events.index("POST:/api/v1/video/jobs")),self.events.index("probe:collect-completed-evidence"))

    def test_missing_wrong_unknown_and_mutated_worker_closure_never_authorize_collection(self):
        i=self.intent();self.persist(i);self.owner.submit(i);self.jobs["job"]["status"]="completed"
        for context in [{},{"key":"unknown","jobId":"job"},{"key":"key","jobId":"foreign"}]:
            with self.assertRaises(RuntimeError):self.owner.before_phase("collect-completed-evidence",context)
        self.services.children["worker"]["birth"]="wrong-birth"
        with self.assertRaises(RuntimeError):self.owner.before_phase("collect-completed-evidence",{"key":"key","jobId":"job"})
        self.assertNotIn("probe:collect-completed-evidence",self.events);self.assertIn("worker",self.services.children)

    def test_lost_collection_reply_reuses_only_exact_durable_drained_lifetime(self):
        i=self.intent();self.persist(i);self.owner.submit(i);self.jobs["job"]["status"]="completed"
        context={"key":"key","jobId":"job"};first=self.owner.before_phase("collect-completed-evidence",context);one=first["drainedWorker"]
        second=self.owner.before_phase("collect-completed-evidence",context);two=second["drainedWorker"]
        self.assertEqual(one,two);self.assertNotIn("worker",self.services.children)
        self.assertNotEqual(first["drainedApi"]["pid"],second["drainedApi"]["pid"])
        self.jobs["job"]["projectId"]="foreign"
        with self.assertRaises(RuntimeError):self.owner.before_phase("collect-completed-evidence",context)

    def test_actual_config_binding_hashes_bytes_refuses_symlinks_and_omits_secrets(self):
        config=self.root/"config";config.mkdir();p=config/"builtin.models.jsonc";p.write_bytes(b"catalog-a")
        resource={"schemaVersion":1,"mode":"locked_cargo_source_build","sourceOutput":{"sha256":"a"*64},"mlxMetallib":{"sha256":"b"*64}}
        m={"runners":{"mlx":{"guardian":{"configDir":str(config)},"nativeResources":resource}}}
        service=g.Services(m,"mlx",self.root,self.physical)
        env={"SCENEWORKS_DATA_DIR":"owned","SCENEWORKS_ACCESS_TOKEN":"never-persist","SCENEWORKS_PARENT_PID":"123","HF_HUB_OFFLINE":"1"}
        one=service.config_binding(env);self.assertEqual(one["nativeResources"],resource);self.assertNotIn("never-persist",json.dumps(one));self.assertNotIn("SCENEWORKS_PARENT_PID",one["environment"])
        p.write_bytes(b"catalog-b");self.assertNotEqual(g.digest(one),g.digest(service.config_binding(env)))
        (config/"alias").symlink_to(p)
        with self.assertRaises(RuntimeError):service.config_binding(env)

    def test_actions_runner_read_token_is_not_inherited_by_normal_product_children(self):
        shader=self.root/"mlx.metallib";shader.write_bytes(b"test-shader")
        m={"runners":{"mlx":{"guardian":{"dataDir":str(self.root/"data"),"configDir":str(self.root/"config"),"port":18501},"apiUrl":"http://127.0.0.1:18501","workerId":"owned","nativeResources":{"mlxMetallib":{"path":str(shader),"bytes":shader.stat().st_size,"sha256":g.file_sha(shader)}}}}}
        service=g.Services(m,"mlx",self.root,self.physical)
        with patch.dict(os.environ,{"SCENEWORKS_TOKEN":"product-secret","GH_TOKEN":"runner-read-secret","GITHUB_TOKEN":"default-secret","PMETAL_METALLIB_PATH":"/ambient/wrong.metallib","PMETAL_MLX_PREBUILT_DIR":"/ambient/prebuilt"}):
            env=service.env()
        self.assertNotIn("GH_TOKEN",env);self.assertNotIn("GITHUB_TOKEN",env);self.assertNotIn("SCENEWORKS_TOKEN",env)
        self.assertEqual(env["SCENEWORKS_ACCESS_TOKEN"],"product-secret")
        self.assertEqual(env["PMETAL_METALLIB_PATH"],str(shader))
        self.assertNotIn("PMETAL_MLX_PREBUILT_DIR",env)
        shader.write_bytes(b"changed")
        with patch.dict(os.environ,{"SCENEWORKS_TOKEN":"product-secret"}):
            with self.assertRaisesRegex(RuntimeError,"metallib descriptor missing or changed"):service.env()

    def test_native_resource_closure_requires_exact_locked_output_and_owned_copy(self):
        import re
        source_root=Path(__file__).resolve().parent.parent
        blocks=re.split(r"(?=^\[\[package\]\]$)",(source_root/"Cargo.lock").read_text(),flags=re.M)
        packages=[b for b in blocks if re.search(r'^name = "pmetal-mlx-sys"$',b,flags=re.M)]
        revision=re.search(r'^source = "git\+https://github.com/michaeltrefry/mlx-rs\?rev=([a-f0-9]{40})#\1"$',packages[0],flags=re.M).group(1)
        version=re.search(r'^version = "([^\"]+)"$',packages[0],flags=re.M).group(1)
        run=self.root.resolve();out=run/"cargo-target"/"release"/"build"/"pmetal-mlx-sys-fixture"/"out"
        source=out/"build"/"lib"/"mlx.metallib";copy=run/"product-resources"/"mlx"/"mlx.metallib"
        source.parent.mkdir(parents=True);copy.parent.mkdir(parents=True);source.write_bytes(b"exact locked shader");copy.write_bytes(b"exact locked shader")
        item={"path":str(source),"bytes":source.stat().st_size,"sha256":g.file_sha(source)}
        native={"schemaVersion":1,"mode":"locked_cargo_source_build","sourceOutput":{"packageId":f"git+https://github.com/michaeltrefry/mlx-rs?rev={revision}#pmetal-mlx-sys@{version}","revision":revision,"outDir":str(out),**item},"mlxMetallib":{"path":str(copy),"bytes":copy.stat().st_size,"sha256":g.file_sha(copy)}}
        g.validate_native_resources({"nativeResources":native},"mlx",run,source_root)
        copy.write_bytes(b"same-size or changed wrong shader")
        with self.assertRaisesRegex(RuntimeError,"missing or changed"):g.validate_native_resources({"nativeResources":native},"mlx",run,source_root)
        with self.assertRaisesRegex(RuntimeError,"descriptor differs"):
            bad={"nativeResources":{**native,"sourceOutput":{**native["sourceOutput"],"revision":"0"*40}}}
            g.validate_native_resources(bad,"mlx",run,source_root)
        g.validate_native_resources({"nativeResources":{"schemaVersion":1,"mode":"not_applicable","sourceOutput":None,"mlxMetallib":None}},"candle_cuda",run,source_root)
        with self.assertRaisesRegex(RuntimeError,"not applicable"):
            g.validate_native_resources({"nativeResources":{"schemaVersion":1,"mode":"locked_cargo_source_build","sourceOutput":{},"mlxMetallib":{}}},"candle_cuda",run,source_root)

    def test_cpu_import_api_explicitly_disables_engine_and_inprocess_worker_for_both_hosts(self):
        m={"runners":{"candle_cuda":{"guardian":{"dataDir":str(self.root/"data"),"configDir":str(self.root/"config"),"port":18502},"apiUrl":"http://127.0.0.1:18502","workerId":"owned","requestedGpu":"0"}}}
        service=g.Services(m,"candle_cuda",self.root,None)
        with patch.dict(os.environ,{"SCENEWORKS_TOKEN":"product-secret","SCENEWORKS_GPU_CHECK":"1","SCENEWORKS_WORKER_ONLY":"1","SCENEWORKS_RUN_UTILITY_INPROCESS":"1","CUDA_VISIBLE_DEVICES":"0","NVIDIA_VISIBLE_DEVICES":"0","GH_TOKEN":"runner-secret"}):
            env=service.env("api")
        self.assertEqual([env[k] for k in ["SCENEWORKS_GPU_CHECK","SCENEWORKS_WORKER_ONLY","SCENEWORKS_RUN_UTILITY_INPROCESS","SCENEWORKS_GPU_ID","SCENEWORKS_RUST_WORKER_GPU_ID"]],["0","0","0","cpu","cpu"])
        for key in ["CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","GH_TOKEN","GITHUB_TOKEN"]:self.assertNotIn(key,env)
        self.assertEqual(env["SCENEWORKS_ACCESS_TOKEN"],"product-secret")

    def test_cpu_import_supervisor_consumes_only_job_owned_api_token(self):
        env={"SCENEWORKS_ACCESS_TOKEN":"job-owned-secret","GH_TOKEN":"runner-read-secret"}
        self.assertEqual(g.cpu_import_supervisor_token(env),"job-owned-secret")
        self.assertNotIn("SCENEWORKS_ACCESS_TOKEN",env);self.assertNotIn("job-owned-secret",json.dumps(env))
        for value in [None,"", "with whitespace"]:
            with self.assertRaises(RuntimeError):g.cpu_import_supervisor_token({"SCENEWORKS_ACCESS_TOKEN":value})

if __name__=="__main__":unittest.main()
