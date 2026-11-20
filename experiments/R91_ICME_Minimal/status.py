from pathlib import Path
import json,time,os
R=Path(__file__).resolve().parent
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
s=read(R/'status.json');p=read(R/'runtime_base/results/progress.json');print('Pipeline:',s.get('status','not started'))
pid=s.get('pid');alive=False
if pid:
 try:os.kill(pid,0);alive=True
 except OSError:pass
print('PID:',pid,'alive:',alive,'updated:',time.strftime('%F %T',time.localtime(s.get('time',0))))
print('Generation:',p.get('completed',0),'/ 724; stage:',p.get('stage','-'),'job:',p.get('job','-'),'step:',p.get('step','-'))
if s.get('error'):print('ERROR:',s['error'])
if p.get('error'):print('Generation error:',p['error'])
c=read(R/'runtime_base/results/generation_checkpoint.json');done=[j for j in c.get('jobs',[]) if j.get('status')=='completed']
dur=[j['runtime_seconds_synchronized'] for j in done if j.get('runtime_seconds_synchronized')]
if len(dur)>=8:print('Rough remaining generation hours (measured image mean, excludes model load/evaluation):',round((724-len(done))*sum(dur)/len(dur)/3600,2))
if (R/'last_log.txt').exists():
 path=R/(R/'last_log.txt').read_text().strip();print('Log:',path)
 if path.exists():print('\n'.join(path.read_text(errors='replace').splitlines()[-6:]))
