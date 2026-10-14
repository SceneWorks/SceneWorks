"""One source-owned, offline CPU diagnostic. Old failed preparation remains read-only."""
import hashlib,importlib.util,json,os,shutil,sys,tarfile,time,tomllib
from pathlib import Path
spec=importlib.util.spec_from_file_location('cpu',Path(__file__).with_name('h3-veda-cpu-build.py'));b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OLD=Path('/Volumes/Models/epic-25140-owned-cache-299c99e8d')
DYLIB=OLD/'download-cargo-target/release/deps/libbytemuck_derive-eee249504c32ecab.dylib'

def frozen(fixture):
    hashes=json.loads((fixture/'files.json').read_text())
    for name,digest in hashes.items():b.require(b.sha(fixture/name)==digest,'source-owned failure fixture changed')
    tool=json.loads((fixture/'toolchain-receipt.json').read_text());cargo=json.loads((fixture/'cargo-receipt.json').read_text())
    for r,rc in [(tool,0),(cargo,101)]:
        b.require(r['rc']==rc and r['closure']['reaped'] and r['closure']['descendantsDrained'],'prior CPU lifetime not closed')
    messages=[json.loads(x) for x in (fixture/'cargo-stdout.txt').read_text().splitlines() if x.strip()]
    artifacts=[m for m in messages if m.get('reason')=='compiler-artifact' and m.get('package_id','').endswith('#bytemuck_derive@1.10.2')]
    b.require(len(artifacts)==1 and artifacts[0]['filenames']==[str(DYLIB)] and artifacts[0]['target']['kind']==['proc-macro'],'exact actual derive artifact differs')
    b.require("error[E0463]: can't find crate for `bytemuck_derive`" in (fixture/'cargo-stderr.txt').read_text(),'actual primary diagnostic missing')
    return tool,cargo,artifacts[0]

def verify_old(fixture):
    tool,cargo,artifact=frozen(fixture)
    for mode,r in [('toolchain',tool),('cargo',cargo)]:
        receipt=OLD/'cpu-build'/mode/'receipt.json';b.no_links(receipt)
        b.require(b.sha(receipt)==b.sha(fixture/(mode+'-receipt.json')),'prior actual receipt changed')
        for stream in ['stdout','stderr']:
            p=OLD/'cpu-build'/mode/(stream+'.log');b.no_links(p)
            b.require(b.binding(p)==r[stream],'prior log bytes changed')
        b.require(b.pid_birth(r['pid'])!=r['birth'],'prior CPU child still active')
    for v in tool['sourceBindings']+[tool['plan'],tool['python']]:
        b.no_links(v['path']);b.require(b.binding(v['path'])==v,'prior source/tool receipt changed')
    b.no_links(DYLIB);return tool,cargo,artifact,b.binding(DYLIB)

def vendor(fixture,root):
    rows=json.loads((fixture/'tiny-dependencies.json').read_text())
    original=tomllib.loads((OLD/'download-source/Cargo.lock').read_text())['package']
    for row in rows:b.require(row in original,'tiny dependency differs from actual original lock')
    b.require({(p['name'],p['version']) for p in rows}=={('bytemuck','1.25.0'),('bytemuck_derive','1.10.2'),('proc-macro2','1.0.106'),('quote','1.0.47'),('syn','2.0.119'),('unicode-ident','1.0.24')},'exact six-package derive closure differs')
    v=root/'vendor';v.mkdir()
    receipts=[]
    for row in rows:
        name=row['name']+'-'+row['version'];archive=OLD/'tools/cargo/registry/cache/index.crates.io-1949cf8c6b5b557f'/(name+'.crate');b.no_links(archive)
        b.require(b.sha(archive)==row['checksum'],'original exact .crate checksum differs');receipts.append(b.binding(archive))
        with tarfile.open(archive,'r:gz') as t:
            members=t.getmembers();b.require(sum(m.size for m in members)<=32*1024*1024,'tiny archive oversized')
            for m in members:
                parts=Path(m.name).parts;b.require(parts and parts[0]==name and '..' not in parts and not Path(m.name).is_absolute() and (m.isdir() or m.isfile()),'unsafe archive member')
                dest=v/m.name
                if m.isdir():dest.mkdir(parents=True,exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    with t.extractfile(m) as src,dest.open('xb') as out:shutil.copyfileobj(src,out)
        files={str(p.relative_to(v/name)):b.sha(p) for p in (v/name).rglob('*') if p.is_file()};(v/name/'.cargo-checksum.json').write_text(json.dumps({'files':files,'package':row['checksum']}))
    source=root/'tiny';source.mkdir();(source/'src').mkdir()
    (source/'src/lib.rs').write_text('#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]\n#[repr(C)]\npub struct ExactDerive(pub u32);\n')
    # Ordinary source has no custom release profile: retain Cargo release defaults.
    ordinary=tomllib.loads((OLD/'download-source/Cargo.toml').read_text());b.require('release' not in ordinary.get('profile',{}),'unexpected original release profile; refuse guessed profile')
    (source/'Cargo.toml').write_text('[package]\nname="h3-exact-bytemuck-diagnostic"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nbytemuck={version="=1.25.0",features=["derive"]}\n')
    lock='version = 4\n\n[[package]]\nname="h3-exact-bytemuck-diagnostic"\nversion="0.0.0"\ndependencies=["bytemuck"]\n'
    for r in rows:
        lock+='\n[[package]]\n'+''.join(f'{k} = {json.dumps(r[k])}\n' for k in ['name','version','source','checksum'])
        if r.get('dependencies'):lock+='dependencies = '+json.dumps([d.replace('syn 2.0.119','syn') for d in r['dependencies']])+'\n'
    (source/'Cargo.lock').write_text(lock);(source/'.cargo').mkdir();(source/'.cargo/config.toml').write_text('[source.crates-io]\nreplace-with="owned-vendor"\n[source.owned-vendor]\ndirectory='+json.dumps(str(v))+'\n[net]\noffline=true\n')
    return source,receipts

def runtime_libraries(bins):
    files=sorted((bins.parent/'lib').glob('librustc_driver-*.dylib'));b.require(0<len(files)<=16,'owned rustc-driver library census missing/oversized')
    for p in files:b.no_links(p)
    b.require(sum(p.stat().st_size for p in files)<=1024*1024*1024,'owned compiler libraries exceed metadata hashing bound')
    return [b.binding(p) for p in files]

def main(plan,parent):
    root=Path(plan['runDir']);internal=Path(plan['internal']);b.no_links(root);b.no_links(internal)
    b.require(root!=OLD and OLD not in root.parents and root not in OLD.parents,'diagnostic root overlaps failed preparation')
    b.require(not internal.exists(),'internal target must be fresh');internal.mkdir(parents=True)
    deadline=time.monotonic()+1800
    fixture=Path(plan['fixtureDirectory']);result={'schemaVersion':1,'kind':'epic25140_CPU_failure_diagnostic','owner':plan['owner'],'prior':plan['prior'],'sourceHead':plan['head'],'prepared':False,'campaignReady':False,'physicalAdmission':False,'diagnosticComplete':False,'cause':'unknown','historicalToolByteHashesRecorded':False,'historicalCompilerEnvironmentRecorded':False,'steps':[]};out=root/'diagnostic-receipt.json';b.atom(out,result)
    try:
        tool,cargo,artifact,dylib=verify_old(fixture);result.update(priorArtifact=artifact,actualDylib=dylib)
        bins=OLD/'tools/rustup/toolchains/1.97.1-aarch64-apple-darwin/bin';tools={}
        for n in ['rustc','cargo']:
            b.no_links(bins/n);tools[n]=b.binding(bins/n)
        env={k:v for k,v in os.environ.items() if k not in ['GH_TOKEN','GITHUB_TOKEN','INFERENCE_READ_TOKEN','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','PMETAL_METALLIB_PATH','PMETAL_MLX_PREBUILT_DIR']}
        env.update(RUSTUP_HOME=str(OLD/'tools/rustup'),CARGO_HOME=str(root/'tools/cargo'),RUSTUP_TOOLCHAIN='1.97.1',CARGO_BUILD_JOBS='2',RUSTC=tools['rustc']['path'],PATH=str(bins)+os.pathsep+env.get('PATH',''))
        Path(env['CARGO_HOME']).mkdir(parents=True);result.update(tools=tools,inheritedEnvironment=b.environment_binding(os.environ),runtimeLibraries=runtime_libraries(bins),volumeDevices={'external':root.stat().st_dev,'internal':internal.stat().st_dev,'failedRoot':OLD.stat().st_dev},environment=b.environment_binding(env),commandResolution={n:shutil.which(n,path=env['PATH']) for n in ['rustc','cargo','cc','clang','ld']},configurationBindings=[b.binding(p) for p in [OLD/'download-source/.cargo/config',OLD/'download-source/.cargo/config.toml',OLD/'tools/cargo/config',OLD/'tools/cargo/config.toml',Path.home()/'.cargo/config',Path.home()/'.cargo/config.toml'] if p.is_file() and not p.is_symlink()],dylibStat={'device':DYLIB.stat().st_dev,'mode':DYLIB.stat().st_mode,'mtimeNs':DYLIB.stat().st_mtime_ns,'xattrs':{n:hashlib.sha256(os.getxattr(DYLIB,n)).hexdigest() for n in os.listxattr(DYLIB)}},controlledDifferences=['private CARGO_HOME and target paths','tiny six-package locked graph, not whole API graph','historical flags/config not recorded; current allowlisted values captured'])
        b.atom(out,result)
        b.require(not env.get('RUSTC_WRAPPER') and not env.get('RUSTC_WORKSPACE_WRAPPER'),'inherited unbound compiler wrapper; metadata retained, no arbitrary execution')
        def probe(name,argv,cwd=root,seconds=60,probe_env=None):
            b.require(time.monotonic()<deadline-10,'original diagnostic deadline exhausted');target=root/'diagnostic'/name
            context={'tools':tools,'helper':plan['helper'],'python':plan['python'],'environment':b.environment_binding(probe_env or env),'diagnosticContext':{'sourceHead':plan['head'],'priorRunId':38534168074,'probe':name}}
            try:b.supervise(argv,cwd,probe_env or env,target,min(seconds,max(1,deadline-time.monotonic())),parent,context)
            except RuntimeError:pass
            r=json.loads((target/'receipt.json').read_text());result['steps'].append({'name':name,'receipt':b.binding(target/'receipt.json'),'rc':r.get('rc'),'closure':r.get('closure'),'error':r.get('error')});b.atom(out,result)
            b.require(r.get('closure',{}).get('reaped') and r['closure'].get('descendantsDrained'),'probe closure missing; stop diagnostic')
            b.require(r.get('error') is None,'probe timeout/controller/launch failure; diagnostic incomplete')
        for name,argv in [('rustc-version',[tools['rustc']['path'],'-Vv']),('cargo-version',[tools['cargo']['path'],'-Vv']),('rustc-sysroot',[tools['rustc']['path'],'--print','sysroot']),('rustc-linkage',['/usr/bin/otool','-L',tools['rustc']['path']]),('cargo-linkage',['/usr/bin/otool','-L',tools['cargo']['path']]),('macro-file',['/usr/bin/file',str(DYLIB)]),('macro-linkage',['/usr/bin/otool','-L',str(DYLIB)]),('macro-load-commands',['/usr/bin/otool','-l',str(DYLIB)]),('macro-signature',['/usr/bin/codesign','-dv','--verbose=4',str(DYLIB)]),('source-head',['/usr/bin/git','-C',str(OLD/'download-source'),'rev-parse','HEAD']),('disk-state',['/bin/df','-h',str(OLD),str(internal)])]:
            probe(name,argv)
            if name in ['rustc-version','cargo-version','source-head']:
                r=json.loads((root/'diagnostic'/name/'receipt.json').read_text());b.require(r.get('rc')==0,'actual tool/source version probe failed')
                value=Path(r['stdout']['path']).read_text().strip()
                b.require(value==b.SOURCE if name=='source-head' else value.startswith(name.split('-')[0]+' 1.97.1 ('),'actual prior source/toolchain identity differs')
        source=root/'macro-load.rs';source.write_text('extern crate bytemuck_derive;\n')
        probe('existing-macro-loader',[tools['rustc']['path'],'--edition=2021','--crate-type=lib','--crate-name','h3_existing_macro_probe',str(source),'--extern','bytemuck_derive='+str(DYLIB),'-L','dependency='+str(DYLIB.parent),'-o',str(root/'macro-load.rlib')])
        tiny,archives=vendor(fixture,root);result['vendoredArchives']=archives;result['tinySourceBindings']=[b.binding(tiny/f) for f in ['Cargo.toml','Cargo.lock','src/lib.rs','.cargo/config.toml']]
        for name,target in [('tiny-external',root/'target'),('tiny-internal',internal/'target')]:
            e={**env,'CARGO_TARGET_DIR':str(target)};probe(name,[tools['cargo']['path'],'build','--release','--locked','--offline','-vv','--message-format=json-render-diagnostics'],tiny,600,e)
        verify_old(fixture);b.require(all(b.binding(t['path'])==t for t in tools.values()),'tool bytes changed during probes');b.require(b.binding(DYLIB)==dylib,'original macro bytes changed');result['diagnosticComplete']=True
    except Exception as e:result['refusal']=type(e).__name__+': '+str(e);raise
    finally:b.atom(out,result)

if __name__=='__main__':
    try:
        p=Path(sys.argv[1]);b.require(b.sha(p)==sys.argv[3],'diagnostic plan hash changed');plan=json.loads(p.read_text())
        b.require(set(plan)=={'schemaVersion','kind','head','owner','prior','runDir','internal','helper','python','fixtureDirectory','supervisor'} and plan['schemaVersion']==1 and plan['kind']=='epic25140_cache_failure_299_diagnostic','closed diagnostic plan differs')
        for x in [plan['helper'],plan['python'],plan['supervisor']]:b.no_links(x['path']);b.require(b.binding(x['path'])==x,'source/runtime binding changed')
        b.require(plan['python']['path']==str(Path(sys.executable).resolve()) and plan['helper']['path']==str(Path(__file__).resolve()),'actual interpreter/helper mismatch');main(plan,int(sys.argv[2]))
    except Exception as e:print(str(e),file=sys.stderr);raise SystemExit(1)
