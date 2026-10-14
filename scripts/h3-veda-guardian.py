"""Task-scoped normal-product guardian. No benchmark/generator entrypoint.
CUDA policy/WindowsJob are loaded from SHA-bound frozen inference source unchanged.
The guardian outlives a dead controller and owns physical locks and service children.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from urllib.parse import quote

TERMINAL = {"completed", "failed", "canceled", "interrupted", "refused", "not_submitted"}
EPIC = Path("/Users/michael/Documents/Codex/2026-10-10/epic-25140")
MAC_LOCKS = [EPIC / "s0-gpu-driver.lock", EPIC / "s2s4-aggregate-target.lock"]

def require(ok, why):
    if not ok:
        raise RuntimeError("H3 physical guard: " + why)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

def file_sha(p):
    with Path(p).open("rb") as f:
        h=hashlib.sha256()
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
        return h.hexdigest()

def atomic(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)

def source_module(root, name, sha):
    p = Path(root) / name
    require(re.fullmatch("[a-f0-9]{64}", sha or "") and file_sha(p) == sha, "frozen inference helper hash differs")
    spec = importlib.util.spec_from_file_location("s9_" + p.stem, p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def command(argv, timeout=15):
    return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()

def pid_birth(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]; k.OpenProcess.restype = wintypes.HANDLE
        k.GetProcessTimes.argtypes=[wintypes.HANDLE,*[ctypes.POINTER(wintypes.FILETIME)]*4];k.GetProcessTimes.restype=wintypes.BOOL
        k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.GetExitCodeProcess.restype=wintypes.BOOL
        k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
        h = k.OpenProcess(0x1000, False, pid)
        if not h: return None
        try:
            stamps = [wintypes.FILETIME() for _ in range(4)]
            if not k.GetProcessTimes(h, *[ctypes.byref(s) for s in stamps]): return None
            code = wintypes.DWORD(); k.GetExitCodeProcess(h, ctypes.byref(code))
            if code.value != 259: return None
            return (stamps[0].dwHighDateTime << 32) | stamps[0].dwLowDateTime
        finally: k.CloseHandle(h)
    try:
        identity=command(["/bin/ps","-p",str(pid),"-o","stat=","-o","lstart="],3).split(None,1)
        return identity[1] if len(identity)==2 and not identity[0].startswith("Z") else None
    except subprocess.CalledProcessError: return None

class Locks:
    def __init__(self, paths): self.paths, self.handles, self.identities = list(map(Path, paths)), [], []
    def claim(self):
        try:
            for p in self.paths:
                require(not p.is_symlink(), "canonical lock is a symlink")
                p.parent.mkdir(parents=True, exist_ok=True)
                f = p.open("a+b"); self.handles.append(f)
                if os.name == "nt":
                    import msvcrt
                    if f.seek(0, 2) == 0: f.write(b"0"); f.flush()
                    f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                s = os.fstat(f.fileno()); self.identities.append((s.st_dev, s.st_ino))
        except Exception: self.close(); raise
    def owned(self):
        return len(self.handles) == len(self.paths) and all(not f.closed and (p.stat().st_dev, p.stat().st_ino) == ident for p, f, ident in zip(self.paths, self.handles, self.identities))
    def close(self):
        for f in reversed(self.handles): f.close()
        self.handles = []

class Physical:
    def __init__(self, manifest, backend, out):
        self.m, self.backend, self.out = manifest, backend, Path(out)
        self.r = manifest["runners"][backend]; self.contract = manifest["sourceFreeze"]["resourceContract"][backend]
        self.baseline = None; self.device = None; self.locks = None; self.sequence=0
        if backend == "candle_cuda":
            require(os.name == "nt", "CUDA guardian must run on the selected Windows host")
            require(re.fullmatch("GPU-[0-9a-f-]+", self.r["physicalGpuId"], re.I), "exact physical UUID required")
            roots = manifest["sourceFreeze"]
            self.cuda = source_module(roots["inference"]["root"], "scripts/release/veda_cuda_guard.py", roots["inferenceFiles"].get("scripts/release/veda_cuda_guard.py"))
            paths = [Path(os.environ["PUBLIC"]) / "SceneWorks" / "gpu-locks" / (self.r["physicalGpuId"] + ".lock")]
        else:
            require(sys.platform == "darwin", "MLX guardian must run on the admitted Mac")
            require(self.contract.get("metalDeviceName"), "exact Metal device name required")
            paths = MAC_LOCKS
        require([str(p) for p in paths] == self.contract["canonicalLocks"], "canonical global lock identity differs")
        require(self.contract["physicalGpuId"] == self.r["physicalGpuId"], "physical identity differs")
        self.locks = Locks(paths)
    def claim(self): self.locks.claim()
    def owned(self):return self.locks.owned()
    def mac_observation(self, owned):
        argv = command(["/bin/ps", "-axo", "pid=,ppid=,comm=,args="])
        actors = []
        for line in argv.splitlines():
            fields = line.strip().split(None, 3)
            if len(fields) != 4: continue
            pid, ppid, comm, args = fields; name = Path(args.split(None, 1)[0]).name.lower()
            if int(pid) in owned: continue  # exact owned PID + birth identity, never a name whitelist
            blocked = (name in ["runner.worker", "xcodebuild", "lensworks", "mlx-gen", "candle-gen", "lm studio", "lms", "ollama", "llama-server", "vllm", "sceneworks", "sceneworks-rust-api", "sceneworks-rust-worker"] or name.startswith("mlx_gen_") or name.startswith("nax") or "/examples/veda_s0" in args.split(None, 1)[0] or args.startswith("/Applications/LM Studio.app/"))
            if name.startswith("python") and any(x in args for x in ["inference_self_test.py", "--device mlx", "--device mps", "run-metal-validation", "run-metal-readback"]): blocked = True
            if blocked: actors.append({"pid":int(pid),"ppid":int(ppid),"executable":comm})
        pressure = command(["memory_pressure", "-Q"])
        free = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
        require(free and int(free[1]) >= 10 and not actors, "Mac foreign GPU actors or physical memory admission refused")
        devices = json.loads(command(["system_profiler", "SPDisplaysDataType", "-json"]))["SPDisplaysDataType"]
        matches = [d for d in devices if d.get("sppci_model") == self.contract["metalDeviceName"] and d.get("spdisplays_mtlgpufamilysupport")]
        require(len(matches) == 1, "actual Metal driver/device not observed")
        hardware=json.loads(command(["system_profiler","SPHardwareDataType","-json"]))["SPHardwareDataType"]
        require(len(hardware)==1 and hardware[0].get("platform_UUID")==self.r["physicalGpuId"],"actual Mac physical platform UUID changed/unbound")
        return {"observedAt": int(time.time()*1000), "admitted":True,"driverAvailable":True,"metalDevice":matches[0]["sppci_model"],"platformUuid":hardware[0]["platform_UUID"],"memoryFreePercent":int(free[1]),"actors":actors}
    def queue(self):
        # Same pre-existing Mac org/nax and Lens queue admission. CUDA host still requires runner/queue authority.
        queries = {"orgRunners":"orgs/SceneWorks/actions/runners?per_page=100", "lensRunning":"repos/michaeltrefry/LensWorks/actions/runs?status=in_progress&per_page=100", "lensQueued":"repos/michaeltrefry/LensWorks/actions/runs?status=queued&per_page=100"}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            fs = {k:pool.submit(command,["gh","api","--paginate","--slurp",q],15) for k,q in queries.items()}
            raw = {k:json.loads(f.result()) for k,f in fs.items()}
        owned_runner = self.actions_owner(raw)
        if self.backend == "mlx":
            runners = [r for page in raw["orgRunners"] for r in page["runners"] if r["name"].startswith("nax")]
            if owned_runner is not None and os.environ.get("RUNNER_NAME")=="nax-macos-2":
                # This checkout is on Mac1: local .runner id 2310 / current
                # SPHardwareDataType UUID D299... . Mac2 is admitted as a
                # different physical lane only when the actual live platform
                # UUID is bound and differs, and both org runner identities
                # remain the observed pair. A changed topology refuses.
                primary=[r for r in runners if r["name"]=="nax-macos"]
                secondary=[r for r in runners if r["name"]=="nax-macos-2"]
                require(len(runners)==2 and len(primary)==len(secondary)==1 and primary[0]["id"]==2310 and secondary[0]["id"]==5281 and secondary[0]["id"]==owned_runner,"Mac runner topology changed/unknown")
                require(self.contract.get("hostRunnerNames")==["nax-macos-2"] and self.r["physicalGpuId"]!="D299B954-4C63-5322-86A5-BD5667051634","Mac2 physical lane topology unbound/same as primary")
                observed=json.loads(command(["system_profiler","SPHardwareDataType","-json"],15))["SPHardwareDataType"]
                require(len(observed)==1 and observed[0].get("platform_UUID")==self.r["physicalGpuId"],"Mac2 actual host UUID differs from frozen physical identity")
                runners=secondary
            require(runners and all(r["status"] == "online" and (not r["busy"] or r["id"] == owned_runner) for r in runners), "nax busy/offline admission refused")
            require(not any(p["workflow_runs"] for k in ["lensRunning","lensQueued"] for p in raw[k]), "Lens competing queue/run admission refused")
        else:
            # A standalone terminal guardian cannot assert CI-host ownership from a UUID lock alone.
            runner = self.contract.get("runnerName")
            host_runners=self.contract.get("hostRunnerNames",[])
            require(host_runners and set(host_runners).issubset(self.cuda.ALLOWED_RUNNERS) and runner in host_runners,"host listener inventory not frozen")
            for name in host_runners:
                rows=[r for p in raw["orgRunners"] for r in p["runners"] if r["name"]==name]
                require(len(rows)==1 and rows[0]["status"]=="online" and (not rows[0]["busy"] or rows[0]["id"]==owned_runner),"shared Windows host listener busy/offline")
            matches = [r for p in raw["orgRunners"] for r in p["runners"] if r["name"] == runner]
            require(runner in self.cuda.ALLOWED_RUNNERS and len(matches) == 1 and matches[0]["status"] == "online" and (not matches[0]["busy"] or matches[0]["id"]==owned_runner), "Windows host runner busy/unowned; no scheduled CI process may host this standalone campaign")
        return {"observedAt":int(time.time()*1000),"raw":raw}
    def actions_owner(self, raw):
        # A workflow-owned guardian necessarily sees its own listener as busy.
        # Exempt only the one live runner ID proven by the current exact Actions
        # run/job/source; every other listener and Lens queue retains the old rule.
        if os.environ.get("GITHUB_ACTIONS") != "true": return None
        env=os.environ
        require(env.get("GH_TOKEN") and env.get("GITHUB_REPOSITORY")=="SceneWorks/SceneWorks" and env.get("GITHUB_EVENT_NAME")=="workflow_dispatch", "Actions owner authorization environment incomplete")
        require(env.get("GITHUB_SHA")==self.m["sourceFreeze"]["sceneworks"]["head"] and env.get("GITHUB_WORKFLOW_REF","").startswith("SceneWorks/SceneWorks/.github/workflows/h3-veda-terminal.yml@"),"Actions workflow/source identity differs")
        expected_job="mac2" if self.backend=="mlx" else "windows_cuda"
        expected_runner="nax-macos-2" if self.backend=="mlx" else self.contract.get("runnerName")
        require(env.get("GITHUB_JOB")==expected_job and env.get("RUNNER_NAME")==expected_runner,"Actions job/physical runner identity differs")
        run_id=env.get("GITHUB_RUN_ID","");attempt=env.get("GITHUB_RUN_ATTEMPT","")
        require(run_id.isdecimal() and attempt.isdecimal(),"Actions run/attempt identity missing")
        run=json.loads(command(["gh","api",f"repos/SceneWorks/SceneWorks/actions/runs/{run_id}"],15))
        jobs=json.loads(command(["gh","api","--paginate","--slurp",f"repos/SceneWorks/SceneWorks/actions/runs/{run_id}/jobs?per_page=100"],15))
        require(run.get("id")==int(run_id) and run.get("run_attempt")==int(attempt) and run.get("event")=="workflow_dispatch" and run.get("head_sha")==env["GITHUB_SHA"] and run.get("status")=="in_progress" and run.get("conclusion") is None and run.get("name")=="H3 Veda terminal owned campaign","live Actions run identity/terminal state differs")
        matches=[j for page in jobs for j in page.get("jobs",[]) if j.get("name")==expected_job and j.get("runner_name")==expected_runner and j.get("run_id")==int(run_id) and j.get("run_attempt")==int(attempt) and j.get("head_sha")==env["GITHUB_SHA"] and j.get("status")=="in_progress" and j.get("conclusion") is None]
        require(len(matches)==1 and isinstance(matches[0].get("runner_id"),int),"live Actions job/runner identity ambiguous")
        runner_id=matches[0]["runner_id"]
        inventory=[r for page in raw["orgRunners"] for r in page["runners"] if r.get("id")==runner_id and r.get("name")==expected_runner]
        require(len(inventory)==1 and inventory[0]["status"]=="online" and inventory[0]["busy"] is True,"live Actions owner runner not busy/unique")
        return runner_id
    def probe(self, phase, owned, drained_api=None):
        require(re.fullmatch(r"[a-z][a-z0-9-]{0,63}",phase),"invalid phase identity")
        require(self.locks.owned(), "physical lock descriptor/path identity lost")
        queue = self.queue(); samples = []; self.sequence+=1; prefix=f"{phase}-{self.sequence:06d}"
        if self.backend == "candle_cuda":
            # Reuse unchanged policy arithmetic/types/caps/timing. Only restrict UUID selection.
            devices, processes = self.cuda.inventory(self.out, prefix + "-inventory")
            matches = [d for d in devices if d["uuid"] == self.r["physicalGpuId"]]
            require(len(matches) == 1, "selected UUID changed")
            self.device = matches[0]
            _, self.baseline = self.cuda.sample_baseline(self.device, self.out, prefix, self.baseline)
            raw = json.loads((self.out / (prefix + "-samples.json")).read_text(encoding="utf-8"))
            ended = int(time.time()*1000); elapsed = raw[-1]["elapsed_seconds"]
            samples = [{"observedAt":ended-round((elapsed-s["elapsed_seconds"])*1000),"admitted":True,"driverAvailable":True,"sourceSample":s} for s in raw]
        else:
            for index in range(5):
                if index: time.sleep(.5)
                samples.append(self.mac_observation(owned))
        result = {"phase":phase,"admitted":True,"physicalGpuId":self.r["physicalGpuId"],"observedAt":int(time.time()*1000),"queueObservedAt":queue["observedAt"],"conflictingJobs":0,"driverSamples":samples}
        if drained_api is not None:result["drainedApi"]=drained_api
        receipt=self.out/(prefix+"-phase.json")
        atomic(receipt,{**result,"queue":queue})
        result["sourceReceipt"]={"path":str(receipt),"sha256":file_sha(receipt)}
        return result
    def close(self): self.locks.close()

class Services:
    def __init__(self, m, backend, run_dir, physical):
        self.m,self.r,self.backend,self.root,self.physical = m,m["runners"][backend],backend,Path(run_dir),physical
        self.children = {}; self.logs = []; self.windows = None; self.api_config = None
        if os.name == "nt":
            f=m["sourceFreeze"]; self.windows=source_module(f["inference"]["root"],"scripts/media_campaign_windows.py",f["inferenceFiles"].get("scripts/media_campaign_windows.py")).WindowsJob
    def env(self,kind="api"):
        g=self.r["guardian"]; token=os.environ.get("SCENEWORKS_TOKEN"); require(token,"SCENEWORKS_TOKEN required; never persist API secret")
        env={k:v for k,v in os.environ.items() if not k.startswith("SCENEWORKS_") and k not in ["CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","GH_TOKEN","GITHUB_TOKEN","PMETAL_METALLIB_PATH","PMETAL_MLX_PREBUILT_DIR"]}
        env.update(SCENEWORKS_DATA_DIR=g["dataDir"],SCENEWORKS_JOBS_DB_PATH=str(Path(g["dataDir"])/"cache/jobs.db"),SCENEWORKS_CONFIG_DIR=g["configDir"],SCENEWORKS_OWN_MANIFESTS="1",SCENEWORKS_API_HOST="127.0.0.1",SCENEWORKS_API_PORT=str(g["port"]),SCENEWORKS_API_URL=self.r["apiUrl"],SCENEWORKS_PARENT_PID=str(os.getpid()),SCENEWORKS_ACCESS_TOKEN=token,SCENEWORKS_GPU_ID="cpu",SCENEWORKS_RUST_WORKER_GPU_ID="cpu",SCENEWORKS_WORKER_ID=self.r["workerId"],SCENEWORKS_GPU_CHECK="0",SCENEWORKS_WORKER_ONLY="0",SCENEWORKS_RUN_UTILITY_INPROCESS="0",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
        if self.backend=="mlx":
            resource=self.r.get("nativeResources",{}).get("mlxMetallib")
            require(resource and Path(resource["path"]).is_absolute() and file_sha(resource["path"])==resource["sha256"] and Path(resource["path"]).stat().st_size==resource["bytes"],"owned MLX metallib descriptor missing or changed")
            env["PMETAL_METALLIB_PATH"]=resource["path"]
        if kind=="worker":env.update(SCENEWORKS_WORKER_ONLY="1",SCENEWORKS_GPU_ID=self.r["requestedGpu"],SCENEWORKS_UTILITY_JOBS="0")
        if self.backend=="candle_cuda" and kind=="worker":
            index=str(self.physical.device["index"])
            require(self.r["requestedGpu"]==index,"normal worker GPU selector does not match locked physical UUID/index")
            env.update(CUDA_VISIBLE_DEVICES=self.r["physicalGpuId"],NVIDIA_VISIBLE_DEVICES=index)
        return env
    def start(self, kind):
        require(kind not in self.children,"owned service already running")
        binding=self.r["apiBinary" if kind=="api" else "workerBinary"]
        require(file_sha(binding["path"])==binding["sha256"],"normal product binary changed")
        env=self.env(kind)
        if kind=="worker":
            require(self.backend!="mlx" or self.r["requestedGpu"]=="mlx","normal MLX worker must explicitly select mlx, never auto")
        config=self.config_binding(env)
        if kind=="api":
            require(self.api_config is None or self.api_config==config,"owned API config changed across physical phases")
            self.api_config=config
        log=(self.root/(kind+".guardian.log")).open("ab"); self.logs.append(log)
        kwargs={"env":env,"stdout":log,"stderr":log,"stdin":subprocess.DEVNULL}
        job=None
        if os.name=="nt": kwargs["creationflags"]=0x00000004; job=self.windows()
        else: kwargs.update(start_new_session=True,pass_fds=tuple(f.fileno() for f in self.physical.locks.handles))
        try:child=subprocess.Popen([binding["path"]],**kwargs)
        except Exception:
            if job:job.close()
            log.close();self.logs.remove(log);raise
        try:
            if job: job.assign_and_resume(child)
        except Exception:
            child.kill();child.wait(timeout=10);job.close();log.close();self.logs.remove(log);raise
        birth=pid_birth(child.pid)
        self.children[kind]={"child":child,"birth":birth,"job":job,"log":log,"configBinding":config,"binarySha256":binding["sha256"]}
        require(birth is not None,"owned service has no live process birth identity")
    def config_binding(self,env=None):
        # Hash the actual byte-bound launch configuration without persisting secrets or per-process parent PID.
        if env is None:
            env=self.env("worker")
        root=Path(self.r["guardian"]["configDir"])
        require(root.is_dir() and not root.is_symlink(),"owned worker config directory unavailable")
        files=[]
        for p in sorted(root.rglob("*")):
            require(not p.is_symlink(),"worker config symlink refused")
            if p.is_file():files.append({"path":p.relative_to(root).as_posix(),"sha256":file_sha(p)})
        public={k:v for k,v in env.items() if (k.startswith("SCENEWORKS_") and k not in ["SCENEWORKS_ACCESS_TOKEN","SCENEWORKS_PARENT_PID"]) or k in ["CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","PMETAL_METALLIB_PATH"]}
        return {"configFiles":files,"environment":public,"nativeResources":self.r.get("nativeResources")}
    def worker_receipt(self):
        item=self.children.get("worker");require(item is not None,"owned worker descriptor missing")
        self.owned_pids()
        require(item["configBinding"]==self.config_binding(),"actual worker configuration changed after launch")
        require(file_sha(self.r["workerBinary"]["path"])==item["binarySha256"],"actual worker binary changed after launch")
        return {"pid":item["child"].pid,"birth":item["birth"],"workerBinarySha256":item["binarySha256"],"workerConfigSha256":digest(item["configBinding"]),"workerConfigBinding":item["configBinding"]}
    def api_receipt(self):
        item=self.children.get("api");require(item is not None,"owned API descriptor missing")
        self.owned_pids()
        require(file_sha(self.r["apiBinary"]["path"])==item["binarySha256"],"actual API binary changed after launch")
        require(item["configBinding"]==self.api_config==self.config_binding(self.env()),"actual API config changed after launch")
        return {"pid":item["child"].pid,"birth":item["birth"],"apiBinarySha256":item["binarySha256"],"apiConfigSha256":digest(item["configBinding"]),"apiConfigBinding":item["configBinding"]}
    def owned_pids(self):
        require(all(pid_birth(x["child"].pid)==x["birth"] and x["birth"] is not None for x in self.children.values()),"owned service died/identity changed")
        return {os.getpid(),*[x["child"].pid for x in self.children.values()]}
    def stop(self,kind,deadline=None):
        grace=10 if deadline is None else max(.1,min(10,deadline-time.monotonic()))
        item=self.children.get(kind)
        if not item:return None
        child=item["child"]
        if item["job"]:
            item["job"].terminate_and_reap(child,grace);item["job"].close()
        else:
            current=pid_birth(child.pid)
            require(current is None or current==item["birth"],"owned root PID birth changed; never signal unrelated process group")
            live=child.poll() is None
            if live:
                require(current==item["birth"] and child.pid==os.getpgid(child.pid),"owned process group identity changed")
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    require(pid_birth(child.pid)==item["birth"] and os.getpgid(child.pid)==child.pid,"owned process group identity changed before kill")
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=grace)
            # A group that persists after its verified leader exits cannot be
            # re-identified by PID alone; retain the lock instead of signaling it.
            try: os.killpg(child.pid,0)
            except ProcessLookupError: pass
            else:raise RuntimeError("owned process group descendants or foreign reuse not verifiably drained")
        self.children.pop(kind)
        item["log"].close();self.logs.remove(item["log"])
        return {"pid":child.pid,"birth":item["birth"],"reaped":child.poll() is not None,"descendantsDrained":True,"exitCode":child.returncode}
    def close(self):
        deadline=time.monotonic()+self.r["guardian"]["cleanupSeconds"]
        self.stop("worker",deadline);self.stop("api",deadline)
        for f in self.logs:f.close()

class Guardian:
    """State machine has injectable physical/process/product interfaces for CPU failure tests."""
    def __init__(self,m,backend,run_dir,physical,services,api,alive,clock=time.monotonic):
        self.m,self.backend,self.root,self.physical,self.services,self.api,self.alive,self.clock=m,backend,Path(run_dir),physical,services,api,alive,clock
        self.started=clock();self.deadline=self.started+m["runners"][backend]["guardian"]["maxSeconds"]
        self.token=os.urandom(24).hex();self.last_phase=None;self.intents={};self.closed=False;self.retain=False;self.mu=threading.RLock()
        old=self.root/"guardian-journal.json"
        if old.exists():
            journal=json.loads(old.read_text(encoding="utf-8"));require(journal["fingerprint"]==digest(m),"prior guardian journal source/inputs differ")
            self.intents=journal["intents"]
    def save(self):atomic(self.root/"guardian-journal.json",{"fingerprint":digest(self.m),"intents":self.intents,"retain":self.retain,"closed":self.closed})
    def active(self):
        active=False
        for intent in self.intents.values():
            if intent.get("status") in TERMINAL:continue
            job_id=intent.get("jobId")
            if job_id:
                job=self.api("GET","/api/v1/jobs/"+job_id)
                require(job["id"]==job_id and job["payload"]["advanced"]["s9Harness"]["key"]==intent["key"],"owned job readback identity changed")
                intent["status"]=job["status"]
            if intent.get("status") not in TERMINAL:active=True
        self.save();return active
    def before_phase(self,phase,context=None):
        require(not self.closed and not self.retain and self.clock()<self.deadline and self.alive(),"dead controller/expired owner deadline")
        self.last_phase=None  # A failed readback/restart/probe must never leave prior admission usable.
        # Never run idle baseline while the known owned job may be running. Both
        # owned service process groups are reaped before each fresh admission;
        # a CPU-configured API may still have acquired a CUDA context.
        require(not self.active(),"owned job still active or ambiguous; phase unavailable")
        context={} if context is None else context
        require(isinstance(context,dict),"phase context must be an object")
        record=None
        if phase=="collect-completed-evidence":
            require(set(context)=={"key","jobId"},"completed evidence requires exact acknowledged key/job context")
            record=self.intents.get(context["key"])
            require(record is not None and record.get("jobId")==context["jobId"] and record.get("status")=="completed","completed evidence job identity/status differs")
            job=self.api("GET","/api/v1/jobs/"+context["jobId"])
            require(job.get("id")==context["jobId"] and job.get("projectId")==self.m["projectId"] and job.get("status")=="completed" and job.get("payload",{}).get("advanced",{}).get("s9Harness")==record["request"]["advanced"]["s9Harness"],"fresh completed product readback differs")
            if "drainedWorker" in record:
                require("worker" not in self.services.children,"new worker cannot reuse prior closure")
            else:require(record.get("workerProcess")==self.services.worker_receipt(),"acknowledged worker process/configuration differs")
        else:require(not context,"unexpected phase context")
        closure=self.services.stop("worker")
        if record is not None and "drainedWorker" not in record:
            process=record["workerProcess"]
            require(closure and closure["pid"]==process["pid"] and closure["birth"]==process["birth"] and closure["reaped"] and closure["descendantsDrained"],"owned worker lifetime not verifiably drained")
            r=self.m["runners"][self.backend]
            record["drainedWorker"]={"schemaVersion":1,**context,**process,**closure,"workerId":r["workerId"],"physicalGpuId":r["physicalGpuId"],"backend":self.backend,"campaignId":self.m["campaignId"],"sourceFingerprint":digest(self.m),"guardianOwnerTokenSha256":hashlib.sha256(self.token.encode()).hexdigest(),"sourceFreezeSha256":digest(self.m["sourceFreeze"]),"lifecycle":"fresh_per_arm_process_drained_before_next_admission","admissionReceipt":record["admissionReceipt"]}
            self.save()
        api_process=self.services.api_receipt()
        api_closure=self.services.stop("api")
        require(api_closure and api_closure["pid"]==api_process["pid"] and api_closure["birth"]==api_process["birth"] and api_closure["reaped"] and api_closure["descendantsDrained"],"owned API lifetime not verifiably drained")
        api_lifetime={"schemaVersion":1,**api_process,**api_closure,"sourceFingerprint":digest(self.m),"physicalGpuId":self.m["runners"][self.backend]["physicalGpuId"]}
        proof=self.physical.probe(phase,self.services.owned_pids(),api_lifetime)
        # Admit only the fully drained process tree. Restarting the authenticated
        # normal API must fit within this same queue/driver receipt's freshness.
        self.services.start("api")
        until=min(self.deadline,self.clock()+10)
        def fresh():
            now=int(time.time()*1000)
            return all(0<=now-proof[field]<=10000 for field in ["observedAt","queueObservedAt"])
        while self.clock()<until and fresh():
            try:
                self.services.owned_pids();self.api("GET","/api/v1/workers");break
            except Exception:time.sleep(.2)
        else:raise RuntimeError("restarted owned normal API not ready; no physical admission")
        require(fresh() and self.clock()<self.deadline,"restarted owned API exceeded original physical/queue freshness; no admission")
        self.last_phase={**proof,"ownerToken":self.token}
        if record is not None:self.last_phase["drainedWorker"]=record["drainedWorker"]
        return self.last_phase
    def acknowledge(self,intent):
        require(not self.retain and self.clock()<self.deadline and self.alive(),"controller/deadline expired before submission")
        now=int(time.time()*1000)
        require(self.physical.owned() and self.last_phase and self.last_phase["phase"]=="submit-or-reconcile" and self.last_phase.get("admitted") is True and 0<=now-self.last_phase.get("observedAt",0)<=10000 and 0<=now-self.last_phase.get("queueObservedAt",0)<=10000,"fresh actual physical/queue phase required before enqueue")
        require(not self.active(),"earlier owned/ambiguous intent unresolved; no new ordinary enqueue")
        key=intent["key"]; require(key not in self.intents,"duplicate intent acknowledgement; resume reconciles, never blindly resubmits")
        request=intent["request"]; marker=request["advanced"]["s9Harness"]
        require(marker["key"]==key and marker["fingerprint"]==digest(self.m) and intent["requestSha256"]==digest(request),"submission intent identity differs")
        require(request["projectId"]==self.m["projectId"],"foreign project submission refused")
        stored=json.loads((self.root/"campaign.json").read_text(encoding="utf-8")); require(stored["fingerprint"]==digest(self.m),"durable campaign identity differs")
        found=[a for c in stored["cells"].values() for a in c["arms"].values() if a.get("key")==key]
        require(len(found)==1 and found[0]["status"]=="submitting" and found[0]["requestSha256"]==intent["requestSha256"],"intent not durably recorded before enqueue")
        self.intents[key]={"key":key,"requestSha256":intent["requestSha256"],"request":request,"status":"submitting","admissionReceipt":{k:v for k,v in self.last_phase.items() if k!="ownerToken"}};self.save()
        if marker["arm"] == "refusal": return {"key":key,"acknowledged":True,"ownerToken":self.token}
        self.services.start("worker")
        self.intents[key]["workerProcess"]=self.services.worker_receipt();self.save()
        until=min(self.deadline,self.clock()+30)
        while self.clock()<until:
            try:
                workers=self.api("GET","/api/v1/workers");self.services.owned_pids()
                if any(w.get("id")==self.m["runners"][self.backend]["workerId"] and w.get("status") not in ["offline","unhealthy"] for w in workers):return {"key":key,"acknowledged":True,"ownerToken":self.token}
            except Exception:pass
            time.sleep(.2)
        raise RuntimeError("owned normal product worker not ready; acknowledged intent remains uncertain, never submit")
    def submit(self,intent):
        existing=intent["key"] in self.intents
        try:self.acknowledge(intent)  # fsync guardian journal and owned worker readiness before ordinary POST
        except Exception:
            if not existing and intent["key"] in self.intents:self.intents[intent["key"]]["status"]="not_submitted";self.save()
            raise
        record=self.intents[intent["key"]]
        try:
            if not self.alive() or int(time.time()*1000)-self.last_phase["observedAt"]>10000:
                record["status"]="not_submitted";self.save();raise RuntimeError("controller/admission expired before ordinary POST")
            value=self.api("POST","/api/v1/video/jobs",record["request"])
            require(isinstance(value,dict) and value.get("id"),"ordinary enqueue response has no job identity")
            record.update(jobId=value["id"],status=value["status"]);self.save()
            return {"status":201,"value":value}
        except ProductError as error:
            if error.status in [400,422]: record.update(status="refused",response={"status":error.status,"value":error.value});self.save()
            else:record.update(status="ambiguous");self.save()
            return {"status":error.status,"value":error.value}
        except Exception:
            if record["status"]!="not_submitted":record["status"]="ambiguous"
            self.save();raise
    def attach(self,key,job_id):
        require(key in self.intents and isinstance(job_id,str) and job_id,"unacknowledged job identity")
        job=self.api("GET","/api/v1/jobs/"+job_id)
        require(job.get("id")==job_id and job.get("projectId")==self.m["projectId"] and job.get("payload",{}).get("advanced",{}).get("s9Harness",{}).get("key")==key,"foreign attach job refused")
        self.intents[key]["jobId"]=job_id;self.save()
    def reconcile(self,intent):
        jobs=self.api("GET","/api/v1/jobs?projectId="+quote(self.m["projectId"],safe="")+"&limit=500")
        require(isinstance(jobs,list) and len(jobs)<500,"incomplete product history; retain physical ownership")
        found=[j for j in jobs if j.get("payload",{}).get("advanced",{}).get("s9Harness",{}).get("key")==intent["key"]]
        require(len(found)==1,"absent/duplicate ambiguous job; cannot prove absence or free ownership")
        job=found[0];require(digest(job["payload"]["advanced"]["s9Harness"])==digest(intent["request"]["advanced"]["s9Harness"]),"ambiguous job marker differs")
        intent["jobId"]=job["id"];return job
    def drain(self):
        # Only guardian-acknowledged requests may be canceled. No cancellation of foreign jobs/processes.
        unresolved=[]; cleanup_deadline=self.clock()+self.m["runners"][self.backend]["guardian"]["cleanupSeconds"]
        for intent in self.intents.values():
            if intent.get("status") in TERMINAL:continue
            try:
                require(self.clock()<cleanup_deadline,"owned cleanup deadline exhausted")
                job=self.api("GET","/api/v1/jobs/"+intent["jobId"]) if intent.get("jobId") else self.reconcile(intent)
                require(job.get("projectId")==self.m["projectId"] and job.get("payload",{}).get("advanced",{}).get("s9Harness",{}).get("key")==intent["key"],"foreign job cancellation refused")
                if job["status"] not in TERMINAL:
                    self.api("POST","/api/v1/jobs/"+job["id"]+"/cancel",{})
                    until=min(cleanup_deadline,self.clock()+10)
                    while self.clock()<until:
                        job=self.api("GET","/api/v1/jobs/"+job["id"])
                        if job["status"] in TERMINAL:break
                        time.sleep(.2)
                require(job["status"] in TERMINAL,"owned cancel not terminal")
                intent["status"]=job["status"]
            except Exception as error:unresolved.append(str(error))
        self.services.close()  # bounded termination and descendant reap precede physical lock close
        self.save()
        require(not unresolved,"ambiguous persisted queue requires reconciliation; services drained, canonical locks retained: "+"; ".join(unresolved))
        self.physical.probe("after-owned-drain",{os.getpid()})
        self.physical.close();self.closed=True;self.save()
        atomic(self.root/"physical-guardian/closed.json",{"closed":True,"servicesDrained":True,"physicalLockReleased":True,"fingerprint":digest(self.m),"ownerTokenSha256":hashlib.sha256(self.token.encode()).hexdigest(),"pid":os.getpid()})
    def tick(self):
        if not self.alive() or self.clock()>=self.deadline or self.retain:
            self.drain()

class ProductError(RuntimeError):
    def __init__(self,status,value): self.status,self.value=status,value;super().__init__("ordinary product HTTP "+str(status))

class Product:
    def __init__(self,url,token):self.url,self.token=url,token
    def __call__(self,method,route,body=None):
        data=None if body is None else json.dumps(body).encode()
        request=urllib.request.Request(self.url+route,data=data,method=method,headers={"Content-Type":"application/json","X-SceneWorks-Token":self.token})
        try:
            with urllib.request.urlopen(request,timeout=10) as response:
                require(response.geturl()==self.url+route,"unexpected product redirect")
                return json.load(response)
        except urllib.error.HTTPError as error:raise ProductError(error.code,json.load(error)) from error

def validate_native_resources(r,backend,run_dir,source_root):
    native=r.get("nativeResources",{});root=Path(run_dir).resolve()
    if backend=="mlx":
        source=native.get("sourceOutput",{});resource=native.get("mlxMetallib",{})
        cargo_lock=Path(source_root)/"Cargo.lock";blocks=re.split(r"(?=^\[\[package\]\]$)",cargo_lock.read_text(encoding="utf-8"),flags=re.M);pins=[b for b in blocks if re.search(r'^name = "pmetal-mlx-sys"$',b,flags=re.M)];pin=re.search(r'^source = "git\+https://github.com/michaeltrefry/mlx-rs\?rev=([a-f0-9]{40})#\1"$',pins[0],flags=re.M) if len(pins)==1 else None;version=re.search(r'^version = "([^\"]+)"$',pins[0],flags=re.M) if len(pins)==1 else None
        target=root/"cargo-target";out=Path(source.get("outDir",""));source_path=Path(source.get("path",""));resource_path=Path(resource.get("path",""));safe_out=False
        try:
            current=target
            for part in out.relative_to(target).parts:
                current=current/part
                require(current.is_dir() and not current.is_symlink(),"MLX Cargo output path contains a symlink or missing directory")
            safe_out=True
        except (ValueError,OSError):pass
        require(native.get("schemaVersion")==1 and native.get("mode")=="locked_cargo_source_build" and pin and source.get("revision")==pin.group(1) and version and source.get("packageId")==f"git+https://github.com/michaeltrefry/mlx-rs?rev={pin.group(1)}#pmetal-mlx-sys@{version.group(1)}" and out.is_absolute() and target.resolve() in out.resolve().parents and safe_out and source_path==out/"build"/"lib"/"mlx.metallib" and resource_path==root/"product-resources"/"mlx"/"mlx.metallib","source-bound MLX metallib descriptor differs")
        for binding in [source,resource]:
            path=Path(binding["path"]);require(path.is_file() and not path.is_symlink() and path.resolve()==path and path.stat().st_size==binding["bytes"] and file_sha(path)==binding["sha256"],"source-bound MLX metallib missing or changed")
        require(source["bytes"]==resource["bytes"] and source["sha256"]==resource["sha256"],"durable MLX metallib differs from exact Cargo output")
    else:require(native=={"schemaVersion":1,"mode":"not_applicable","sourceOutput":None,"mlxMetallib":None},"Candle host must declare MLX metallib not applicable")

def validate_config(m,backend,run_dir):
    r=m["runners"][backend];g=r.get("guardian",{});root=Path(run_dir).resolve()
    for name in ["sceneworks","inference"]:
        repo=m["sourceFreeze"][name]; require(command(["git","-C",repo["root"],"rev-parse","HEAD"])==repo["head"] and not command(["git","-C",repo["root"],"status","--porcelain"]),"stale/dirty frozen source "+name)
    require(m["sourceFreeze"]["inference"]["head"]==m["sourceFreeze"]["inferencePin"],"actual inference head/pin differs")
    for name,sha in m["sourceFreeze"]["files"].items():require(file_sha(Path(m["sourceFreeze"]["sceneworks"]["root"])/name)==sha,"source file changed "+name)
    require(m["sourceFreeze"]["stage"]=="permanent-inference-main-pin","permanent source/pin freeze required")
    require(1<=g.get("maxSeconds",0)<=3600 and 1<=g.get("cleanupSeconds",0)<=60,"bounded guardian lifecycle/cleanup required")
    require(1024<=g.get("port",0)<=65535 and r["apiUrl"]=="http://127.0.0.1:"+str(g["port"]),"isolated loopback product API required")
    for key in ["dataDir","configDir"]:
        p=Path(g.get(key,"")).resolve();require(p!=root and root in p.parents and not p.is_symlink(),"task-owned isolated "+key+" required inside external run directory")
    marker=Path(g["dataDir"])/"s9-workspace-owner.json"
    owner=json.loads(marker.read_text(encoding="utf-8"))
    require(owner=={"campaignId":m["campaignId"],"sourceFreezeSha256":digest(m["sourceFreeze"]),"projectId":m["projectId"]},"isolated workspace ownership unbound")
    for name in ["builtin.models.jsonc","builtin.loras.jsonc"]:
        require(file_sha(Path(g["configDir"])/"manifests"/name)==m["sourceFreeze"]["files"]["config/manifests/"+name],"isolated normal product catalog differs")
    validate_native_resources(r,backend,root,m["sourceFreeze"]["sceneworks"]["root"])
    for binding in [r["apiBinary"],r["workerBinary"]]:
        require(Path(binding["path"]).is_absolute() and file_sha(binding["path"])==binding["sha256"],"source-attested normal binary differs")
    return g

def serve(manifest_file,backend,run_dir,controller_pid):
    m=json.loads(Path(manifest_file).read_text(encoding="utf-8"));g=validate_config(m,backend,run_dir)
    root=Path(run_dir);out=root/"physical-guardian";out.mkdir(exist_ok=True)
    physical=Physical(m,backend,out);physical.claim()
    services=Services(m,backend,root,physical)
    identity=pid_birth(controller_pid);require(identity is not None,"controller birth identity unavailable")
    controller={"pid":controller_pid,"birth":identity}
    alive=lambda:pid_birth(controller["pid"])==controller["birth"]
    product=Product(m["runners"][backend]["apiUrl"],os.environ.get("SCENEWORKS_TOKEN",""))
    owner=Guardian(m,backend,root,physical,services,product,alive)
    blocked=None
    def request_stop(signum,frame):owner.retain=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        physical.probe("initial-claim",{os.getpid()})
        with socket.socket() as probe: probe.bind(("127.0.0.1",g["port"]))
        services.start("api")
        # Before declaring ready, actual normal API must answer; an existing listener is never adopted.
        until=min(owner.deadline,time.monotonic()+30)
        while time.monotonic()<until:
            try:
                services.owned_pids();product("GET","/api/v1/workers");break
            except Exception:time.sleep(.2)
        else:raise RuntimeError("isolated normal API not ready")
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                nonlocal blocked
                self.connection.settimeout(90)
                try:
                    raw=self.rfile.readline(2*1024*1024+1);require(0<len(raw)<=2*1024*1024,"invalid guardian IPC size")
                    q=json.loads(raw);require(q.get("token")==owner.token and q.get("fingerprint")==digest(m),"foreign guardian IPC identity")
                    with owner.mu:
                        method=q["method"];arg=q.get("arg",{})
                        if method=="status":result={"owned":physical.locks.owned(),"pid":os.getpid(),"birth":pid_birth(os.getpid()),"controllerAlive":alive(),"controllerPid":controller["pid"],"closed":owner.closed,"blocked":blocked,"intents":{k:{x:v for x,v in a.items() if x!="request"} for k,a in owner.intents.items()}}
                        elif method=="phase":result=owner.before_phase(arg["phase"],arg.get("context",{}))
                        elif method=="submit":result=owner.submit(arg)
                        elif method=="attach":owner.attach(arg["key"],arg["jobId"]);result={"attached":True}
                        elif method=="retain":owner.retain=True;owner.save();result={"retained":True}
                        elif method=="release":owner.drain();result={"released":True}
                        elif method=="resume":
                            require(not owner.closed,"guardian already closed")
                            require(not alive(),"previous controller still alive")
                            birth=pid_birth(arg["pid"]);require(birth is not None,"new controller identity unavailable")
                            controller.update(pid=arg["pid"],birth=birth);owner.retain=False
                            if "api" not in services.children:services.start("api")
                            result={"resumed":True,"blocked":blocked}
                        else:raise RuntimeError("unknown guardian method")
                    response={"ok":True,"value":result}
                except Exception as error:response={"ok":False,"error":str(error)}
                self.wfile.write((json.dumps(response)+"\n").encode())
        class Server(socketserver.ThreadingTCPServer):daemon_threads=True;allow_reuse_address=False
        with Server(("127.0.0.1",0),Handler) as server:
            server.timeout=.5
            control={"pid":os.getpid(),"birth":pid_birth(os.getpid()),"port":server.server_address[1],"token":owner.token,"fingerprint":digest(m),"locks":physical.contract["canonicalLocks"]}
            atomic(out/"control.json",control);os.chmod(out/"control.json",0o600)
            while not owner.closed:
                server.handle_request()
                if not blocked:
                    with owner.mu:
                        try:owner.tick()
                        except Exception as error:
                            blocked=str(error);atomic(out/"blocked.json",{"error":blocked,"ownedServicesDrained":not services.children,"locksRetained":physical.locks.owned(),"qualityAcceptance":False})
            atomic(out/"closed.json",{"closed":True,"servicesDrained":not services.children,"physicalLockReleased":not physical.locks.owned(),"fingerprint":digest(m),"ownerTokenSha256":hashlib.sha256(owner.token.encode()).hexdigest(),"pid":os.getpid()})
    except Exception:
        # Launch/initialization refusal cannot release a physical token while our child still exists.
        services.close();physical.close();raise

def serve_cpu_import_api(manifest_file,backend,run_dir):
    """Own only the normal API process tree for CPU corpus import; never claim GPU locks."""
    from types import SimpleNamespace
    token=cpu_import_supervisor_token(os.environ);os.environ["SCENEWORKS_TOKEN"]=token
    m=json.loads(Path(manifest_file).read_text(encoding="utf-8"));root=Path(run_dir).resolve();r=m["runners"][backend];g=r["guardian"]
    validate_native_resources(r,backend,root,m["sourceFreeze"]["sceneworks"]["root"])
    require(backend in ["mlx","candle_cuda"] and Path(manifest_file).is_absolute() and root.is_absolute(),"absolute selected import inputs required")
    require(m["sourceFreeze"]["stage"]=="permanent-inference-main-pin" and m["sourceFreeze"]["inference"]["head"]==m["sourceFreeze"]["inferencePin"],"permanent source/pin freeze required")
    for name in ["sceneworks","inference"]:
        repo=m["sourceFreeze"][name];require(command(["git","-C",repo["root"],"rev-parse","HEAD"])==repo["head"] and not command(["git","-C",repo["root"],"status","--porcelain"]),"stale/dirty frozen source "+name)
    require(command(["git","-C",m["sourceFreeze"]["inference"]["root"],"rev-parse","refs/remotes/origin/main"])==m["sourceFreeze"]["inferencePin"],"inference pin is not permanent main")
    helper=Path(__file__).resolve();require(m["sourceFreeze"]["files"].get("scripts/h3-veda-guardian.py")==file_sha(helper),"CPU import supervisor is not bound to frozen source")
    frozen_roots=[Path(m["sourceFreeze"][name]["root"]).resolve() for name in ["sceneworks","inference"]]
    require(all(root!=frozen and frozen not in root.parents and root not in frozen.parents for frozen in frozen_roots),"owned run directory must stay outside frozen source")
    for key in ["dataDir","configDir"]:
        p=Path(g[key]).resolve();require(root in p.parents and p!=root and not p.is_symlink(),"CPU import API data/config must be task-owned under run directory")
    require(r["apiUrl"]=="http://127.0.0.1:"+str(g["port"]),"CPU import API must bind loopback only")
    binary=r["apiBinary"];worker=r["workerBinary"];require(binary["path"]==worker["path"] and binary["sha256"]==worker["sha256"] and binary["bytes"]==worker["bytes"] and Path(binary["path"]).is_absolute() and file_sha(binary["path"])==binary["sha256"],"durable source-bound API/worker binary differs")
    att_binding=r["attestation"];att_path=Path(att_binding["path"]);require(att_path.is_absolute() and file_sha(att_path)==att_binding["sha256"],"durable normal API build attestation bytes differ")
    att=json.loads(att_path.read_text(encoding="utf-8"));freeze_sha=digest(m["sourceFreeze"])
    require(att.get("schemaVersion")==3 and att.get("method")=="actions_source_build_durable_same_binary_api_worker_and_native_resources" and att.get("backend")==backend and att.get("sceneworksHead")==m["sourceFreeze"]["sceneworks"]["head"] and att.get("inferencePin")==m["sourceFreeze"]["inferencePin"] and att.get("sourceFreezeSha256")==freeze_sha and att.get("apiBinarySha256")==binary["sha256"] and att.get("workerBinarySha256")==r["workerBinary"]["sha256"] and att.get("workerId")==r["workerId"] and att.get("buildIdentitySha256")==digest(att.get("buildIdentity")) and att.get("nativeResources")==r.get("nativeResources") and att.get("buildIdentity",{}).get("nativeResources")==r.get("nativeResources") and att.get("buildIdentity",{}).get("environment",{}).get("PMETAL_METALLIB_PATH") is None and att.get("buildIdentity",{}).get("environment",{}).get("PMETAL_MLX_PREBUILT_DIR") is None,"normal API build attestation source/pin/binary differs")
    control=root/"cpu-import-api";control.mkdir(parents=True,exist_ok=False)
    class NoPhysicalLock:
        handles=[]
    services=Services(m,backend,root,SimpleNamespace(locks=NoPhysicalLock(),device=None))
    stopping=threading.Event()
    def stop_signal(signum,frame):stopping.set()
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    try:
        with socket.socket() as probe:probe.bind(("127.0.0.1",g["port"]))
        services.start("api");api=services.api_receipt()
        atomic(control/"ready.json",{"schemaVersion":1,"managerPid":os.getpid(),"managerBirth":pid_birth(os.getpid()),"api":api,"sourceFreezeSha256":digest(m["sourceFreeze"]),"physicalLockClaimed":False,"cpuOnlyEnvironment":True})
        while not stopping.wait(.2):services.owned_pids()
        drained=services.stop("api")
        require(drained and drained["reaped"] and drained["descendantsDrained"],"CPU import API process tree not verifiably drained")
        closure={"schemaVersion":1,"backend":backend,"managerPid":os.getpid(),"managerBirth":pid_birth(os.getpid()),"api":{**api,**drained},"sourceFreezeSha256":digest(m["sourceFreeze"]),"physicalLockClaimed":False,"cpuOnlyEnvironment":True}
        atomic(control/"closure.json",closure)
    except Exception:
        services.close();raise

def cpu_import_supervisor_token(env):
    token=env.pop("SCENEWORKS_ACCESS_TOKEN",None);require(token and not any(c.isspace() for c in token),"job-owned CPU import API token required");return token

if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--owner-state":
        print(json.dumps({"birth":pid_birth(int(sys.argv[2]))}));sys.exit(0)
    if len(sys.argv)==5 and sys.argv[1]=="--cpu-import-api":
        try:serve_cpu_import_api(sys.argv[2],sys.argv[3],sys.argv[4])
        except Exception as error:print("H3 CPU import API: "+str(error),file=sys.stderr);sys.exit(1)
        sys.exit(0)
    parser=argparse.ArgumentParser();parser.add_argument("manifest");parser.add_argument("backend",choices=["mlx","candle_cuda"]);parser.add_argument("run_dir");parser.add_argument("controller_pid",type=int)
    args=parser.parse_args();serve(args.manifest,args.backend,args.run_dir,args.controller_pid)
