"""Idempotent background start; launch lock prevents double clicks racing."""
from pathlib import Path
import subprocess,os,sys,time,json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'runtime_base'))
import reporting as rep

def running(pid):
    try:
        os.kill(int(pid),0)
        p=Path('/proc')/str(pid)/'cmdline'
        return not p.exists() or b'run.py' in p.read_bytes()
    except (OSError,ValueError,TypeError):return False

with rep.file_lock(ROOT/'.launch.lock'):
    previous=rep.read(ROOT/'launch.json') if (ROOT/'launch.json').exists() else {}
    if running(previous.get('pid')):print('Already running:',previous['pid'],'log:',previous['log']);sys.exit(0)
    (ROOT/'logs').mkdir(exist_ok=True)
    log=ROOT/'logs'/time.strftime('run_%Y%m%d_%H%M%S.log')
    action=sys.argv[1] if len(sys.argv)>1 else 'all'
    with log.open('ab',buffering=0) as f:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/'run.py'),action],cwd=ROOT,
                            stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                            start_new_session=True)
    rep.save(ROOT/'launch.json',dict(pid=p.pid,log=str(log),started=time.time()))
    time.sleep(1)
    if p.poll() is not None and p.returncode!=0:
        print(log.read_text(errors='replace')[-4000:]);sys.exit(p.returncode)
    print('Background PID:',p.pid,'log:',log)
