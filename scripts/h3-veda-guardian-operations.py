"""Actual CPU process/HTTP/flock proof; not a native SceneWorks or GPU lease proof."""
import argparse,importlib.util,json,os,signal,socket,socketserver,subprocess,sys,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('guardian',ROOT/'h3-veda-guardian.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

def wait_for(predicate,timeout=20):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        try:
            value=predicate()
            if value:return value
        except (FileNotFoundError,ConnectionError,OSError,json.JSONDecodeError):pass
        time.sleep(.05)
    raise RuntimeError('CPU process proof bounded wait expired')
def rpc(control,method,arg=None):
    with socket.create_connection(('127.0.0.1',control['port']),timeout=5) as s:
        s.settimeout(15);s.sendall((json.dumps({'token':control['token'],'method':method,'arg':arg})+'\n').encode());data=b''
        while b'\n' not in data:
            chunk=s.recv(65536)
            if not chunk:raise RuntimeError('CPU fixture IPC EOF')
            data+=chunk
        result=json.loads(data)
        if not result['ok']:raise RuntimeError(result['error'])
        return result['value']
class Physical:
    def __init__(self,root):self.root=root;self.locks=g.Locks([root/'CPU-ONLY-NOT-GPU.lock']);self.locks.claim();self.sequence=[]
    def owned(self):return self.locks.owned()
    def probe(self,phase,owned):
        self.sequence.append(phase)
        return {'phase':phase,'admitted':True,'observedAt':int(time.time()*1000),'queueObservedAt':int(time.time()*1000),'cpuFixtureOnly':True,'GPU':False}
    def close(self):
        state=json.loads((self.root/'cpu-worker.json').read_text()) if (self.root/'cpu-worker.json').exists() else {}
        api=json.loads((self.root/'cpu-api.json').read_text()) if (self.root/'cpu-api.json').exists() else {}
        for pid in [api.get('pid'),state.get('pid'),state.get('descendant')]:
            if pid is not None:g.require(g.pid_birth(pid) is None,'actual owned CPU child still live before unlock')
        self.locks.close()

def daemon(root,controller_pid):
    os.environ.update(S9_CPU_PROOF_ROOT=str(root),SCENEWORKS_TOKEN='dummy-cpu-fixture-token')
    m=json.loads((root/'input.json').read_text());physical=Physical(root);services=g.Services(m,'mlx',root,physical)
    birth=g.pid_birth(controller_pid);owner=g.Guardian(m,'mlx',root,physical,services,g.Product(m['runners']['mlx']['apiUrl'],'dummy-cpu-fixture-token'),lambda:g.pid_birth(controller_pid)==birth)
    services.start('api');wait_for(lambda:owner.api('GET','/api/v1/workers'))
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:
                q=json.loads(self.rfile.readline());g.require(q['token']==owner.token,'foreign CPU fixture token')
                with owner.mu:
                    if q['method']=='phase':value=owner.before_phase(q['arg']['phase'],q['arg'].get('context',{}))
                    elif q['method']=='submit':value=owner.submit(q['arg'])
                    elif q['method']=='cancel':owner.drain();value={'released':True}
                    elif q['method']=='status':value={'closed':owner.closed,'held':physical.owned()}
                    else:raise RuntimeError('unknown fixture method')
                reply={'ok':True,'value':value}
            except Exception as e:reply={'ok':False,'error':str(e)}
            try:self.wfile.write((json.dumps(reply)+'\n').encode())
            except BrokenPipeError:pass
    class Server(socketserver.ThreadingTCPServer):daemon_threads=True
    with Server(('127.0.0.1',0),Handler) as server:
        server.timeout=.05;g.atomic(root/'control.json',{'port':server.server_address[1],'token':owner.token,'pid':os.getpid()})
        blocked=False
        while not owner.closed:
            server.handle_request()
            if not blocked:
                with owner.mu:
                    try:owner.tick()
                    except Exception as e:
                        blocked=True;g.atomic(root/'blocked.json',{'error':str(e),'held':physical.owned(),'servicesDrained':not services.children})
        g.atomic(root/'done.json',{'closed':owner.closed,'held':physical.owned(),'servicesDrained':not services.children,'phaseSequence':physical.sequence,'GPU':False,'nativeProduct':False})

def controller(root):
    control=wait_for(lambda:json.loads((root/'control.json').read_text()));m=json.loads((root/'input.json').read_text())
    if os.environ.get('S9_CPU_PROOF_VARIANT')=='absent':
        while True:time.sleep(.1)
    rpc(control,'phase',{'phase':'submit-or-reconcile'})
    request={'projectId':'owned-cpu-proof','advanced':{'s9Harness':{'key':'one-owned-intent','fingerprint':g.digest(m),'arm':'dense'}}}
    intent={'key':'one-owned-intent','requestSha256':g.digest(request),'request':request,'status':'submitting'}
    g.atomic(root/'campaign.json',{'fingerprint':g.digest(m),'cells':{'cpu-case':{'arms':{'dense':intent}}}})
    try:r=rpc(control,'submit',intent);g.atomic(root/'controller-result.json',r)
    except Exception as e:g.atomic(root/'controller-result.json',{'transportError':str(e)})
    try:rpc(control,'submit',intent);duplicate='unexpected_success'
    except Exception as e:duplicate=str(e)
    g.atomic(root/'duplicate.json',{'result':duplicate})
    if os.environ.get('S9_CPU_PROOF_VARIANT')=='paired_completed':
        wait_for(lambda:(root/'cpu-worker.json').exists())
        one=rpc(control,'phase',{'phase':'collect-completed-evidence','context':{'key':intent['key'],'jobId':r['value']['id']}})['drainedWorker']
        rpc(control,'phase',{'phase':'submit-or-reconcile'})
        request['advanced']['s9Harness'].update(key='second-owned-intent',arm='sparse')
        second={'key':'second-owned-intent','requestSha256':g.digest(request),'request':request,'status':'submitting'}
        g.atomic(root/'campaign.json',{'fingerprint':g.digest(m),'cells':{'cpu-case':{'arms':{'sparse':second}}}})
        reply=rpc(control,'submit',second);wait_for(lambda:json.loads((root/'cpu-worker.json').read_text())['pid']!=one['pid'])
        two=rpc(control,'phase',{'phase':'collect-completed-evidence','context':{'key':second['key'],'jobId':reply['value']['id']}})['drainedWorker']
        g.require((one['pid'],one['birth'])!=(two['pid'],two['birth']),'paired arms reused process lifetime')
        for field in ['workerBinarySha256','workerConfigSha256','workerConfigBinding','backend','physicalGpuId','campaignId','sourceFingerprint','sourceFreezeSha256','workerId']:
            g.require(one[field]==two[field],'paired immutable identity differs: '+field)
        g.atomic(root/'paired-lifetimes.json',{'dense':one,'sparse':two,'actualCPUProcesses':True,'GPU':False,'nativeProduct':False})
    while True:time.sleep(.1)

def contender(root):
    locks=g.Locks([root/'CPU-ONLY-NOT-GPU.lock'])
    try:locks.claim();locks.close();return True
    except OSError:return False

def run(output):
    g.require(os.name!='nt','this local CPU operational proof is POSIX; native Windows Job proof remains separately required')
    output.mkdir(parents=True,exist_ok=False);cases=[]
    for variant in ['normal','response_loss','death_during_post','absent','paired_completed']:
        root=output/variant;root.mkdir();port_socket=socket.socket();port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1];port_socket.close()
        fixture=ROOT/'fixtures/h3-veda/cpu-owned-product.py';binding={'path':str(fixture),'sha256':g.file_sha(fixture)}
        (root/'config').mkdir();g.atomic(root/'config/CPU_ONLY.json',{'mockProduct':True,'GPU':False})
        m={'campaignId':'cpu-only-operational-proof','sourceFreeze':{'CPU_ONLY':True},'projectId':'owned-cpu-proof','runners':{'mlx':{'physicalGpuId':'CPU_ONLY_NOT_GPU','apiUrl':f'http://127.0.0.1:{port}','workerId':'owned-cpu-worker','requestedGpu':'mlx','guardian':{'dataDir':str(root/'data'),'configDir':str(root/'config'),'port':port,'maxSeconds':20,'cleanupSeconds':5},'apiBinary':binding,'workerBinary':binding}}}
        g.atomic(root/'input.json',m);env=dict(os.environ,S9_CPU_PROOF_VARIANT=variant,S9_CPU_PROOF_ROOT=str(root))
        ctrl=subprocess.Popen([sys.executable,__file__,'controller',str(root)],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=(root/'controller.stderr').open('w'))
        if variant=='absent':
            request={'projectId':'owned-cpu-proof','advanced':{'s9Harness':{'key':'one-owned-intent','fingerprint':g.digest(m),'arm':'dense'}}}
            g.atomic(root/'guardian-journal.json',{'fingerprint':g.digest(m),'intents':{'one-owned-intent':{'key':'one-owned-intent','requestSha256':g.digest(request),'request':request,'status':'ambiguous'}}})
        guardian=subprocess.Popen([sys.executable,__file__,'daemon',str(root),str(ctrl.pid)],env=env,start_new_session=True,stdout=subprocess.DEVNULL,stderr=(root/'guardian.stderr').open('w'))
        try:
            wait_for(lambda:(root/'control.json').exists());g.require(not contender(root),'actual CPU lock not held')
            if variant=='death_during_post':wait_for(lambda:(root/'cpu-product-state.json').exists())
            elif variant=='paired_completed':wait_for(lambda:(root/'paired-lifetimes.json').exists())
            elif variant!='absent':wait_for(lambda:(root/'duplicate.json').exists())
            g.require(g.pid_birth(ctrl.pid) is not None,'controller must still be alive before actual SIGKILL')
            os.kill(ctrl.pid,signal.SIGKILL)  # real controller loss; deliberately observe before parent reap
            g.require(wait_for(lambda:g.pid_birth(ctrl.pid) is None),'zombie/controller identity still live');ctrl.wait(timeout=5)
            if variant=='absent':
                receipt=wait_for(lambda:json.loads((root/'blocked.json').read_text()));g.require(receipt['held'] and receipt['servicesDrained'] and not contender(root),'ambiguity unlocked or failed to drain actual CPU services')
                # Test cleanup of our own fixture guardian only, after recording retained-lock proof. No product job existed.
                os.kill(guardian.pid,signal.SIGKILL);guardian.wait(timeout=5)
            else:
                receipt=wait_for(lambda:json.loads((root/'done.json').read_text()));guardian.wait(timeout=5)
                g.require(receipt['closed'] and receipt['servicesDrained'] and contender(root),'owned CPU services not drained before actual lock release')
                state=json.loads((root/'cpu-product-state.json').read_text())
                if variant=='paired_completed':g.require(state['postCount']==2 and all(j['status']=='completed' for j in state['jobs'].values()),'paired CPU jobs were duplicated/not completed')
                else:g.require(state['postCount']==1 and state['jobs']['cpu-job']['status']=='canceled','duplicate POST or uncanceled owned CPU job')
            case={'variant':variant,'acceptedCPUProof':True,'receipt':receipt,'controllerSIGKILLActual':True,'GPU':False,'nativeProduct':False}
            if variant=='paired_completed':case['pairedLifetimes']=json.loads((root/'paired-lifetimes.json').read_text())
            cases.append(case)
        finally:
            for child in [ctrl,guardian]:
                if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
    g.atomic(output/'receipt.json',{'source':{'guardianSha256':g.file_sha(ROOT/'h3-veda-guardian.py'),'fixtureSha256':g.file_sha(fixture),'operationsSha256':g.file_sha(__file__)},'cases':cases,'CPUProofOnly':True,'nativeProduct':False,'GPU':False})
    print(str(output/'receipt.json'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','daemon','controller']);parser.add_argument('directory');parser.add_argument('pid',nargs='?',type=int);a=parser.parse_args()
    if a.mode=='run':run(Path(a.directory))
    elif a.mode=='daemon':daemon(Path(a.directory),a.pid)
    else:controller(Path(a.directory))
