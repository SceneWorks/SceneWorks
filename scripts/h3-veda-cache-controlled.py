"""Closed internal-only matched CPU macro reproduction and conditional strip control."""
import importlib.util,json,os,re,shutil,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).with_name('h3-veda-cache-diagnostic.py'));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d);b=d.b
COMPLETED=Path('/Volumes/Models/epic-25140-cache-diagnostic-698e4406c')
COMPLETED_SHA='85c3d8cebacd3cef6fb73855357948767face64d3b1379be048e629ef5ddb95e'
SYNTAX_ID='registry+https://github.com/rust-lang/crates.io-index#syn@2.0.119'
DERIVE_ID='registry+https://github.com/rust-lang/crates.io-index#bytemuck_derive@1.10.2'
FLOOR=1073741824
collect_xattrs=d.collect_xattrs

def parse_messages(path):
    b.require(Path(path).stat().st_size<=64*1024*1024,'bounded Cargo JSON parser exceeded')
    result=[]
    for line in Path(path).read_text().splitlines():
        try:result.append(json.loads(line))
        except json.JSONDecodeError:pass
    return result

def reference_units(fixture):
    d.frozen(fixture)
    ids=[SYNTAX_ID,DERIVE_ID]+['registry+https://github.com/rust-lang/crates.io-index#'+p for p in ['proc-macro2@1.0.106','quote@1.0.47','unicode-ident@1.0.24']]
    units={x['package_id']:x for x in parse_messages(fixture/'cargo-stdout.txt') if x.get('reason')=='compiler-artifact' and x['package_id'] in ids and x['target']['kind']!=['custom-build']}
    b.require(set(units)==set(ids),'complete original host feature/profile units missing');return units

def matched_source(fixture,root):
    source,archives=d.vendor(fixture,root);units=reference_units(fixture)
    flags=[x for x in units[SYNTAX_ID]['features'] if x not in ['default']]
    # A root build-dependency joins the same host graph as bytemuck_derive under resolver2.
    with (source/'Cargo.toml').open('a') as f:f.write('[build-dependencies]\nsyn={version="=2.0.119",features='+json.dumps(flags)+'}\n')
    (source/'build.rs').write_text('fn main() {}\n')
    lock=(source/'Cargo.lock').read_text().replace('dependencies=["bytemuck"]','dependencies=["bytemuck", "syn"]');(source/'Cargo.lock').write_text(lock)
    return source,archives,units

def validate_units(messages,expected):
    units={x['package_id']:x for x in messages if x.get('reason')=='compiler-artifact' and x.get('package_id') in expected and x['target']['kind']!=['custom-build']}
    b.require(set(units)==set(expected),'matched original host units absent')
    for key,wanted in expected.items():
        b.require(units[key]['features']==wanted['features'] and units[key]['profile']==wanted['profile'],'actual host feature/profile union differs from original: '+key)
    return units

def macro_path(unit,target):
    files=unit['filenames'];b.require(len(files)==1,'single actual macro artifact required');p=Path(files[0]);b.no_links(p)
    b.require(target in p.parents and p.suffix=='.dylib' and p.stat().st_size<=64*1024*1024,'actual macro artifact escaped/oversized');return p

def offsets(text):
    sym=re.search(r'cmd LC_SYMTAB\s+cmdsize \d+\s+symoff (\d+)\s+nsyms (\d+)\s+stroff (\d+)\s+strsize (\d+)',text)
    version=re.search(r'cmd LC_BUILD_VERSION\s+cmdsize \d+\s+platform \d+\s+minos ([\d.]+)\s+sdk ([\d.]+)',text)
    b.require(sym and version,'actual LC_SYMTAB/build-version metadata unavailable')
    return {'symoff':int(sym[1]),'nsyms':int(sym[2]),'stroff':int(sym[3]),'strsize':int(sym[4]),'minos':version[1],'sdk':version[2]}

def conditional_comparison(baseline,run_changed):
    if baseline['invalidLinkeditReproduced'] and baseline['loaderRc']!=0:
        changed=run_changed()
        return changed,changed['cargoRc']==0 and changed['loaderRc']==0 and not changed['invalidLinkeditReproduced']
    return None,False

def root_admission(plan):
    b.require(sys.platform=='darwin' and os.uname().machine=='arm64','actual Apple-silicon CPU host required')
    root=Path(plan['runDir']);temp=Path(plan['runnerTemp']['path']);b.no_links(root);b.no_links(temp)
    b.require(root==temp/('h3-cache-controlled-'+plan['head'][:9]) and not str(temp).startswith('/Volumes/'),'actual internal source-prefixed root required')
    b.require(os.getuid()==plan['runnerTemp']['uid']==temp.stat().st_uid==root.stat().st_uid,'actual runner UID ownership differs')
    b.require(os.environ.get('RUNNER_TEMP')==str(temp) and plan['owner']['runnerId']==5281,'actual runner-temp/runner authority changed')
    b.require(shutil.disk_usage(temp).free>=FLOOR,'actual fresh internal free space below1GiB operational floor');return root

def main(plan,parent):
    root=root_admission(plan);deadline=time.monotonic()+1800;fixture=Path(plan['fixtureDirectory']);out=root/'diagnostic-receipt.json'
    result={'schemaVersion':1,'kind':'epic25140_matched_CPU_macro_control','sourceHead':plan['head'],'owner':plan['owner'],'prior':plan['prior'],'completedDiagnostic':plan['completedDiagnostic'],'runnerTemp':plan['runnerTemp'],'steps':[],'arms':{},'diagnosticComplete':False,'stripWorkaroundSupported':False,'cause':'unknown','prepared':False,'campaignReady':False,'physicalAdmission':False};b.atom(out,result)
    try:
        d.verify_old(fixture);previous=COMPLETED/'diagnostic-receipt.json';b.no_links(previous);b.require(b.sha(previous)==COMPLETED_SHA,'actual completed698 evidence changed')
        old=json.loads(previous.read_text());b.require(old['diagnosticComplete'] is True and len(old['steps'])==16 and all(x['closure']['reaped'] and x['closure']['descendantsDrained'] for x in old['steps']),'prior698 lifetime evidence incomplete')
        config=old['configurationBindings'][0];b.no_links(config['path']);b.require(b.binding(config['path'])==config,'actual original Cargo configuration changed')
        tools=old['tools'];libraries=old['runtimeLibraries']
        for x in list(tools.values())+libraries:b.no_links(x['path']);b.require(b.binding(x['path'])==x,'actual same698 compiler/tool/runtime bytes changed')
        env={k:v for k,v in os.environ.items() if k not in ['GH_TOKEN','GITHUB_TOKEN','INFERENCE_READ_TOKEN','HF_TOKEN','HUGGING_FACE_HUB_TOKEN','PMETAL_METALLIB_PATH','PMETAL_MLX_PREBUILT_DIR']}
        controls=['RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER']+[k for k in b.ENV_ALLOWLIST if k.startswith('CARGO_PROFILE_')]
        b.require(all(not env.get(k) for k in controls),'unbound inherited compiler/profile override; matched experiment refused')
        b.require(env.get('MACOSX_DEPLOYMENT_TARGET') in [None,'26.2'],'inherited deployment target conflicts with original')
        env.update(RUSTUP_HOME=str(d.OLD/'tools/rustup'),CARGO_HOME=str(root/'tools/cargo'),RUSTUP_TOOLCHAIN='1.97.1',CARGO_BUILD_JOBS='2',RUSTC=tools['rustc']['path'],PATH=str(Path(tools['rustc']['path']).parent)+os.pathsep+env.get('PATH',''),MACOSX_DEPLOYMENT_TARGET='26.2')
        Path(env['CARGO_HOME']).mkdir(parents=True,exist_ok=True);source,archives,units=matched_source(fixture,root)
        result.update(tools=tools,runtimeLibraries=libraries,environment=b.environment_binding(env),archives=archives,referenceUnits=units,sourceBindings=[b.binding(source/f) for f in ['Cargo.toml','Cargo.lock','src/lib.rs','build.rs','.cargo/config.toml']],originalProductConfigSha256=old['configurationBindings'][0]['sha256'],fullProductBuildExecuted=False)
        def probe(name,argv,cwd=root,seconds=60,probe_env=None):
            b.require(time.monotonic()<deadline-10,'original1800s matched experiment deadline exhausted')
            context={'tools':tools,'helper':plan['helper'],'python':plan['python'],'environment':b.environment_binding(probe_env or env),'sourceBindings':result['sourceBindings'],'diagnosticContext':{'sourceHead':plan['head'],'probe':name,'same698Tools':True}}
            target=root/'diagnostic'/name
            try:b.supervise(argv,cwd,probe_env or env,target,min(seconds,max(1,deadline-time.monotonic())),parent,context)
            except RuntimeError:pass
            r=json.loads((target/'receipt.json').read_text());result['steps'].append({'name':name,'receipt':b.binding(target/'receipt.json'),'rc':r.get('rc'),'closure':r.get('closure'),'error':r.get('error')});b.atom(out,result)
            b.require(r.get('closure',{}).get('reaped') and r['closure'].get('descendantsDrained') and r.get('error') is None,'matched probe timeout/launch/lifetime failure');return r
        def arm(label,strip_none):
            target=root/'targets'/label;b.no_links(target);b.require(not target.exists(),'fresh distinct target required')
            e={**env,'CARGO_TARGET_DIR':str(target)};argv=[tools['cargo']['path'],'build','--release','--locked','--offline','-vv','--message-format=json-render-diagnostics']
            b.require(shutil.disk_usage(root).free>=FLOOR,'actual fresh arm free-space admission below1GiB')
            if strip_none:argv+=['--config','profile.release.build-override.strip="none"']
            cargo=probe(label+'-cargo',argv,source,600,e);actual=validate_units(parse_messages(cargo['stdout']['path']),units);macro=macro_path(actual[DERIVE_ID],target)
            info={'cargoRc':cargo['rc'],'artifact':b.binding(macro),'hostUnits':actual,'override':{'profile.release.build-override.strip':'none'} if strip_none else None}
            for name,cmd in [('file',['/usr/bin/file',str(macro)]),('linkage',['/usr/bin/otool','-L',str(macro)]),('load-commands',['/usr/bin/otool','-l',str(macro)]),('signature',['/usr/bin/codesign','-dv','--verbose=4',str(macro)])]:
                r=probe(label+'-'+name,cmd,probe_env=e);b.require(r['rc']==0,'actual macro metadata command failed');info[name]=r['stdout'] if name!='signature' else r['stderr']
            info['machO']=offsets(Path(info['load-commands']['path']).read_text());b.require(info['machO']['minos']=='26.2','actual matched deployment target differs')
            info['xattrs']=collect_xattrs(macro,lambda name,argv:probe(label+'-'+name,argv,probe_env=e))
            src=root/(label+'-loader.rs');src.write_text('extern crate bytemuck_derive;\n')
            load=probe(label+'-loader',[tools['rustc']['path'],'--edition=2021','--crate-type=lib','--crate-name','h3_matched_macro_probe',str(src),'--extern','bytemuck_derive='+str(macro),'-L','dependency='+str(macro.parent),'-o',str(root/(label+'-loader.rlib'))],probe_env=e)
            info.update(loaderRc=load['rc'],loaderStderr=load['stderr'],invalidLinkeditReproduced=load['rc']!=0 and 'mis-aligned LINKEDIT string pool' in Path(load['stderr']['path']).read_text());result['arms'][label]=info;b.atom(out,result);return info
        baseline=arm('baseline',False)
        changed,supported=conditional_comparison(baseline,lambda:arm('strip-none',True));result['stripWorkaroundSupported']=supported
        if changed is None:result['comparisonSkippedReason']='matched baseline did not reproduce the actual LINKEDIT loader refusal; cause/workaround unproved and whole-product repro still required'
        for x in list(tools.values())+libraries+result['sourceBindings']:b.require(b.binding(x['path'])==x,'tool/runtime/source bytes changed during comparison')
        d.verify_old(fixture);b.require(b.sha(previous)==COMPLETED_SHA,'prior698 evidence changed');result['diagnosticComplete']=True
    except Exception as e:result['refusal']=type(e).__name__+': '+str(e);raise
    finally:b.atom(out,result)

if __name__=='__main__':
    try:
        p=Path(sys.argv[1]);b.require(b.sha(p)==sys.argv[3],'controlled plan bytes changed');plan=json.loads(p.read_text())
        b.require(set(plan)=={'schemaVersion','kind','head','owner','prior','completedDiagnostic','runnerTemp','runDir','internal','helper','python','fixtureDirectory','supervisor','collector'} and plan['schemaVersion']==1 and plan['kind']=='epic25140_internal_matched_macro_control','closed internal experiment plan required')
        for x in [plan['helper'],plan['python'],plan['supervisor'],plan['collector']]:b.no_links(x['path']);b.require(b.binding(x['path'])==x,'actual source/runtime changed')
        b.require(plan['collector']['path']==str(Path(d.__file__).resolve()) and plan['supervisor']['path']==str(Path(b.__file__).resolve()),'actual imported collector/supervisor differs from bindings')
        b.require(plan['helper']['path']==str(Path(__file__).resolve()) and plan['python']['path']==str(Path(sys.executable).resolve()),'actual source/interpreter mismatch');main(plan,int(sys.argv[2]))
    except Exception as e:print(str(e),file=sys.stderr);raise SystemExit(1)
