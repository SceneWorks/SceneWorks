"""Closed CPU Rustup/Cargo preparation commands, owned process lifetime and receipts.

POSIX closure policy follows h3-veda-mlx-fixtures.py at 51eb9b41c:
PID birth + process-group identity, reap, then refuse any surviving descendants.
This module imports no native runtime or physical admission implementation.
"""
import hashlib,json,os,pwd,shutil,signal,stat,subprocess,sys,time
from pathlib import Path

CARGO_SECONDS=3600
RUSTUP_SECONDS=600
SOURCE='16567882f0e07b5d6d95941adc42210d41dff7ab'
CONFIG_SHA='8d60b170aa783b4487b3ae07a09ccef7116d03ddf7aeb79895537296098a1713'
BUILD_MIN_FREE=32*1024**3
BUILD_RESERVE_FREE=8*1024**3
EXPECTED_TOOLS={'rustc':'210df6794001b73ec3d453878707fa1e0bdcb63c427024a6e6574bbe5615a4da','cargo':'7672ead309d505577c018fff2cafb3433601f073e38cbe87359ac1f7b944bbf5','rustcDriver':'57f6b5280f218ed25ba6b7e4bc1d74da3ac8e4fd85bee1b8982ad4114d757c11'}

def require(ok,why):
    if not ok:raise RuntimeError('H3 CPU build refusal: '+why)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def binding(p):
    p=Path(p);s=p.lstat();require(p.is_absolute() and stat.S_ISREG(s.st_mode),'regular absolute file required')
    return {'path':str(p),'bytes':s.st_size,'sha256':sha(p)}
def atom(p,v):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    t.replace(p)
def no_links(p):
    p=Path(p);require(p.is_absolute(),'absolute owned path required')
    for a in [p,*p.parents]:require(not a.is_symlink(),'symlink ancestor refused')
def pid_birth(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','lstart='],capture_output=True,text=True,timeout=5)
    return r.stdout.strip() if r.returncode==0 and r.stdout.strip() else None
def group_alive(pid):
    try:os.killpg(pid,0);return True
    except ProcessLookupError:return False

def child_closure(child,birth,deadline):
    current=pid_birth(child.pid);require(current is None or current==birth,'child birth changed; never signal foreign group')
    if child.poll() is None:
        require(current==birth and child.pid==os.getpgid(child.pid),'owned child process group changed')
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=max(.1,min(10,deadline-time.monotonic())))
        except subprocess.TimeoutExpired:
            require(pid_birth(child.pid)==birth and os.getpgid(child.pid)==child.pid,'owned birth/group changed before forced drain')
            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=max(.1,deadline-time.monotonic()))
    require(child.poll() is not None and not group_alive(child.pid),'owned descendants not verifiably drained')
    return {'pid':child.pid,'birth':birth,'reaped':True,'descendantsDrained':True,'exitCode':child.returncode}

ENV_ALLOWLIST=("PATH","RUSTUP_HOME","CARGO_HOME","CARGO_TARGET_DIR","CARGO_BUILD_JOBS","RUSTUP_TOOLCHAIN","RUSTC","RUSTC_WRAPPER","RUSTC_WORKSPACE_WRAPPER","RUSTFLAGS","CARGO_ENCODED_RUSTFLAGS","RUSTDOCFLAGS","RUSTDOC","CARGO_BUILD_TARGET","CC","CXX","CFLAGS","CXXFLAGS","LDFLAGS","SDKROOT","MACOSX_DEPLOYMENT_TARGET","DYLD_LIBRARY_PATH","DYLD_FALLBACK_LIBRARY_PATH","LD_LIBRARY_PATH","LIBRARY_PATH","TMPDIR","CARGO_PROFILE_RELEASE_OPT_LEVEL","CARGO_PROFILE_RELEASE_LTO","CARGO_PROFILE_RELEASE_CODEGEN_UNITS","CARGO_PROFILE_RELEASE_DEBUG","CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS","CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS","CARGO_PROFILE_RELEASE_PANIC","CARGO_PROFILE_RELEASE_STRIP")
def environment_binding(env):
    return {k:env[k] for k in ENV_ALLOWLIST if k in env}

def supervise(argv,cwd,env,output,max_seconds,parent_pid,context=None):
    """Primitive accepts fake CPU commands in tests; production argv is closed below."""
    output=Path(output);no_links(output);output.mkdir(parents=True,exist_ok=False)
    parent_birth=pid_birth(parent_pid);require(parent_birth is not None,'controller birth unavailable')
    deadline=time.monotonic()+max_seconds;reserve=min(10,max_seconds/4)
    receipt={'schemaVersion':1,'kind':'epic25140_owned_CPU_build_child','argv':argv,'cwd':str(cwd),'maxSeconds':max_seconds,'controller':{'pid':parent_pid,'birth':parent_birth},'startedAt':int(time.time()*1000),'prepared':False,'physicalAdmission':False,'campaignReady':False}
    if context is not None:
        require(set(context).issubset({'tools','helper','python','plan','sourceRoot','sourceBindings','environment','diagnosticContext','buildAuthority','sourceCensus','diskAdmission'}),'CPU context cannot override lifetime/result fields')
        receipt.update(context)
    atom(output/'receipt.json',receipt);child=None;birth=None;error=None
    def interrupted(signum,frame):raise InterruptedError('CPU controller received signal '+str(signum))
    old={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    stdout=output/'stdout.log';stderr=output/'stderr.log'
    try:
        with stdout.open('xb') as out,stderr.open('xb') as err:
            child=subprocess.Popen(argv,cwd=cwd,env=env,stdout=out,stderr=err,stdin=subprocess.DEVNULL,start_new_session=True)
            birth=pid_birth(child.pid);require(birth is not None,'owned CPU child birth unavailable')
            receipt.update(pid=child.pid,birth=birth);atom(output/'receipt.json',receipt)
            while child.poll() is None:
                require(pid_birth(parent_pid)==parent_birth,'controller ended or birth changed')
                if context and context.get('diskAdmission'):require(shutil.disk_usage(context['diskAdmission']['path']).free>=context['diskAdmission']['reserveFreeBytes'],'owned CPU disk reserve exhausted; drain begins')
                if time.monotonic()>=deadline-reserve:raise TimeoutError('original CPU child deadline; reserved drain begins')
                time.sleep(min(.2,max_seconds/10))
            receipt['rc']=child.returncode
    except BaseException as e:error=type(e).__name__+': '+str(e)
    finally:
        try:
            if child is not None:
                require(birth is not None,'unidentified child; closure cannot be fabricated')
                receipt['closure']=child_closure(child,birth,deadline)
        except BaseException as e:
            receipt['closureError']=type(e).__name__+': '+str(e)
        for s,handler in old.items():signal.signal(s,handler)
        receipt.update(error=error,endedAt=int(time.time()*1000),elapsedSeconds=max_seconds-(deadline-time.monotonic()))
        for name,p in [('stdout',stdout),('stderr',stderr)]:
            if p.exists():receipt[name]=binding(p)
        receipt['prepared']=error is None and 'closureError' not in receipt and receipt['elapsedSeconds']<=max_seconds and pid_birth(parent_pid)==parent_birth and receipt.get('rc')==0 and receipt.get('closure',{}).get('reaped') is True and receipt['closure'].get('descendantsDrained') is True
        atom(output/'receipt.json',receipt)
    require(receipt['prepared'],error or receipt.get('closureError') or 'CPU child failed/unclosed')
    return receipt

def validate_build_authority(plan):
    a=plan['buildAuthority'];root=Path(plan['runDir']);home=Path(a['home']['path']);no_links(home);no_links(root)
    require(sys.platform=='darwin' and os.uname().machine=='arm64','actual Apple-silicon CPU build host required')
    require(str(home)==pwd.getpwuid(os.getuid()).pw_dir and a['home']['uid']==os.getuid()==home.stat().st_uid==root.stat().st_uid,'actual account/home/root UID differs')
    require(root==home/'.h3-veda-epic-25140'/('cpu-build-'+a['sourceHead'][:9]) and root.stat().st_dev==home.stat().st_dev==a['home']['device'],'exact durable internal source/root device required')
    require(a['control']['sourceHead']=='351e828ea7f1d43f433d3267a25869d9a2e0024d' and a['control']['runId']==38545586867 and a['control']['jobId']==115790708657 and a['control']['verificationSha256']=='39fb5d4b2e615daec201a932449078697d35ed091ffe8d48de0b980e04857f4f','actual matched control authority differs')
    require(a['owner']['runnerId']==5281 and a['owner']['sourceHead']==a['sourceHead'],'exact source/runner ownership required')
    marker=a['marker'];no_links(marker['path']);require(Path(marker['path'])==root/'.h3-cpu-build-owner.json' and binding(marker['path'])==marker,'actual durable owner marker changed')
    require(json.loads(Path(marker['path']).read_text())=={k:v for k,v in a.items() if k!='marker'},'actual durable marker authority differs')
    require(shutil.disk_usage(root).free>=BUILD_MIN_FREE,'fresh internal build capacity below32GiB operational floor')

def closed_command(plan):
    require(set(plan)=={'schemaVersion','kind','mode','runDir','sourceRoot','helper','python','sourceBindings','sourceConfiguration','buildAuthority'},'closed CPU plan fields differ')
    require(plan['schemaVersion']==1 and plan['kind']=='epic25140_closed_CPU_build_plan' and plan['mode'] in ['toolchain','cargo'],'unknown CPU plan')
    root=Path(plan['runDir']);no_links(root);require(root.is_dir(),'owned root missing');validate_build_authority(plan)
    source=Path(plan['sourceRoot']);require(source==root/'download-source','ordinary source escaped owned root');no_links(source);no_links(root/'download-cargo-target')
    expected=['Cargo.toml','Cargo.lock','crates/sceneworks-worker/Cargo.toml','crates/sceneworks-memory-adapter/Cargo.toml','crates/sceneworks-worker/src/lib.rs']
    require([b['path'] for b in plan['sourceBindings']]==[str(source/f) for f in expected],'exact CPU source binding census differs')
    for b in plan['sourceBindings']:require(binding(b['path'])==b,'actual source bytes changed')
    config=plan['sourceConfiguration'];no_links(config['path']);require(Path(config['path'])==source/'.cargo/config.toml' and binding(config['path'])==config and config['sha256']==CONFIG_SHA,'actual original Cargo configuration changed')
    for b in [plan['helper'],plan['python']]:require(binding(b['path'])==b,'actual source/Python bytes changed')
    require(Path(plan['helper']['path'])==Path(__file__).resolve() and Path(plan['python']['path'])==Path(sys.executable).resolve(),'helper/interpreter not actual selected bindings')
    homes={k:str(root/'tools'/name) for k,name in [('RUSTUP_HOME','rustup'),('CARGO_HOME','cargo')]}
    for p in homes.values():no_links(p);Path(p).mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if k not in ['GH_TOKEN','GITHUB_TOKEN','INFERENCE_READ_TOKEN','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','PMETAL_METALLIB_PATH','PMETAL_MLX_PREBUILT_DIR']}
    require(not any(env.get(k) for k in ['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER']+[k for k in ENV_ALLOWLIST if k.startswith('CARGO_PROFILE_')]),'unbound inherited compiler/profile override refused')
    require(env.get('MACOSX_DEPLOYMENT_TARGET') in [None,'26.2'],'inherited deployment target differs')
    env.update(homes,RUSTUP_TOOLCHAIN='1.97.1',CARGO_TARGET_DIR=str(root/'download-cargo-target'),CARGO_BUILD_JOBS='2',SCENEWORKS_GPU_CHECK='0')
    if plan['mode']=='toolchain':
        rustup=Path(shutil.which('rustup') or '').resolve();tool=binding(rustup)
        return [str(rustup),'toolchain','install','1.97.1','--profile','minimal','--no-self-update'],root,env,RUSTUP_SECONDS,{'rustup':tool}
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True,timeout=15).strip()==SOURCE,'ordinary source head differs')
    bins=Path(homes['RUSTUP_HOME'])/'toolchains/1.97.1-aarch64-apple-darwin/bin';tools={}
    for name,expected in [('rustc','rustc 1.97.1'),('cargo','cargo 1.97.1')]:
        actual=bins/name;no_links(actual);b=binding(actual);version=subprocess.check_output([str(actual),'--version'],env=env,text=True,timeout=15).strip();require(version.startswith(expected+' ('),'actual owned toolchain version differs');tools[name]={**b,'version':version}
    driver=bins.parent/'lib/librustc_driver-e03ed1db822dfa4c.dylib';no_links(driver);tools['rustcDriver']=binding(driver)
    for key,wanted in EXPECTED_TOOLS.items():require(tools[key]['sha256']==wanted,'owned compiler/runtime bytes differ from actual qualified1.97.1 control: '+key)
    env.update(PATH=str(bins)+os.pathsep+env.get('PATH',''),RUSTC=tools['rustc']['path'])
    return [tools['cargo']['path'],'build','--release','--locked','-vv','--message-format=json-render-diagnostics','-p','sceneworks-rust-api','--config','profile.release.build-override.strip="none"'],source,env,CARGO_SECONDS,tools

if __name__=='__main__':
    try:
        plan_path=Path(sys.argv[1]);require(sha(plan_path)==sys.argv[3],'CPU plan bytes changed');plan=json.loads(plan_path.read_text())
        argv,cwd,env,seconds,tools=closed_command(plan);output=Path(plan['runDir'])/'cpu-build'/plan['mode']
        context={'tools':tools,'helper':plan['helper'],'python':plan['python'],'plan':binding(plan_path),'sourceRoot':plan['sourceRoot'],'sourceBindings':plan['sourceBindings'],'environment':environment_binding(env),'buildAuthority':plan['buildAuthority'],'sourceCensus':plan['sourceBindings']+[plan['sourceConfiguration']],'diskAdmission':{'path':plan['runDir'],'admittedFreeBytes':shutil.disk_usage(plan['runDir']).free,'minimumAdmissionFreeBytes':BUILD_MIN_FREE,'reserveFreeBytes':BUILD_RESERVE_FREE}}
        receipt=supervise(argv,cwd,env,output,seconds,int(sys.argv[2]),context=context)
        print(json.dumps(receipt))
    except Exception as e:print(str(e),file=sys.stderr);raise SystemExit(1)
