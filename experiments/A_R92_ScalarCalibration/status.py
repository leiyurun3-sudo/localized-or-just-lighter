"""Three-second lightweight monitor. No torch import or repeated nvidia-smi."""
from pathlib import Path
import json,time,os,sys
ROOT=Path(__file__).resolve().parent
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}
def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,ValueError,TypeError):return False
def show():
    s=read(ROOT/'status.json');p=read(ROOT/'runtime_base/results/progress.json');e=read(ROOT/'status_evaluation.json');l=read(ROOT/'launch.json')
    print(time.strftime('%Y-%m-%d %H:%M:%S'),flush=True)
    print('A-R92:',s.get('status','starting'),'driver alive:',alive(l.get('pid')))
    print('Generation:',str(p.get('completed',0))+'/80','failed:',p.get('failed',0),'stage:',p.get('stage','preflight'))
    print('Job:',p.get('job','-'),'step:',str(p.get('step','-'))+'/'+str(p.get('steps',28)))
    if p.get('updated'):print('Generation update age: %.1fs'%(time.time()-p['updated']))
    if e:print('Evaluation:',e)
    for key in ['error','next_action']:
        if s.get(key):print(key+':',s[key])
    if p.get('error'):print('Generation error:',p['error'])
    if not alive(l.get('pid')) and s.get('status') not in ['complete','generation_complete','evaluation_pending','paused']:
        print('Driver is not running. Last status may be stale. Restart with python start.py; committed images are resumed.')
    if l.get('log'):
        print('Log:',l['log'])
        try:
            with Path(l['log']).open('rb') as f:f.seek(max(0,Path(l['log']).stat().st_size-1600));tail=f.read().decode(errors='replace')
            print('\n'.join(tail.splitlines()[-6:]))
        except OSError:pass
    print('Ctrl+C stops this monitor only; the background experiment continues.',flush=True)
try:
    while True:
        if sys.stdout.isatty():print('\033[2J\033[H',end='')
        show()
        if '--once' in sys.argv:break
        time.sleep(3)
except KeyboardInterrupt:print('\nMonitor closed; background job unchanged.')
