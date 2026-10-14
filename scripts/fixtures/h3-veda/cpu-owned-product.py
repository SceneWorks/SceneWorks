#!/usr/bin/env python3
"""TEST ONLY CPU HTTP service/worker. Never imports native engines or claims GPU.
Used as a byte-bound executable by real guardian.Services process ownership proof.
"""
import json,os,signal,subprocess,sys,time,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
root=Path(os.environ['S9_CPU_PROOF_ROOT']);root.mkdir(exist_ok=True)
state_file=root/'cpu-product-state.json'
def birth(pid):
    try:
        value=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','stat=','-o','lstart='],text=True).strip().split(None,1)
        return value[1] if len(value)==2 and not value[0].startswith('Z') else None
    except subprocess.CalledProcessError:return None
def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data));temp.replace(path)
def load():return json.loads(state_file.read_text()) if state_file.exists() else {'postCount':0,'jobs':{}}
if os.environ.get('SCENEWORKS_WORKER_ONLY')=='1':
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])
    atomic(root/'cpu-worker.json',{'pid':os.getpid(),'birth':birth(os.getpid()),'descendant':child.pid,'descendantBirth':birth(child.pid),'nativeProduct':False,'GPU':False})
    def stop(*args):
        child.terminate();child.wait(timeout=5);sys.exit(0)
    signal.signal(signal.SIGTERM,stop)
    parent=int(os.environ['SCENEWORKS_PARENT_PID']);identity=birth(parent)
    while True:
        if birth(parent)!=identity:stop()
        time.sleep(.1)
else:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,value,status=200):
            body=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            state=load()
            if self.path=='/api/v1/workers':self.send([{'id':os.environ['SCENEWORKS_WORKER_ID'],'status':'idle'}]);return
            if '?' in self.path:self.send(list(state['jobs'].values()));return
            key=self.path.split('/')[-1]
            if key in state['jobs']:self.send(state['jobs'][key]);return
            self.send({'error':'not found'},404)
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));state=load()
            if self.path=='/api/v1/video/jobs':
                variant=os.environ.get('S9_CPU_PROOF_VARIANT');state['postCount']+=1
                job={'id':('cpu-job-'+str(state['postCount'])) if variant=='paired_completed' else 'cpu-job','projectId':body['projectId'],'payload':body,'status':'completed' if variant=='paired_completed' else 'running','cpuFixtureOnly':True}
                state['jobs'][job['id']]=job;atomic(state_file,state)
                if variant=='response_loss':self.close_connection=True;return
                if variant=='death_during_post':time.sleep(2)
                self.send(job,201);return
            if self.path.endswith('/cancel'):
                key=self.path.split('/')[-2];state['jobs'][key]['status']='canceled';atomic(state_file,state);self.send({});return
            self.send({'error':'unsupported CPU fixture route'},400)
    atomic(root/'cpu-api.json',{'pid':os.getpid(),'birth':birth(os.getpid()),'nativeProduct':False,'GPU':False})
    parent=int(os.environ['SCENEWORKS_PARENT_PID']);identity=birth(parent)
    def parent_watch():
        while birth(parent)==identity:time.sleep(.1)
        os._exit(0)
    threading.Thread(target=parent_watch,daemon=True).start()
    HTTPServer(('127.0.0.1',int(os.environ['SCENEWORKS_API_PORT'])),Handler).serve_forever()
