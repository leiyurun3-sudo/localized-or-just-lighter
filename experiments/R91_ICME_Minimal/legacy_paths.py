"""R88 Fast: exact-equivalence-gated generation, evaluation, compact package."""
import argparse,csv,hashlib,importlib.metadata,importlib.util,json,os,shlex,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime_base'
DEFAULT_PROJECT=Path('/root/autodl-tmp/MINT_Flow_Jupyter_Ready')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def paths():
 bases=[Path(os.environ.get('R88_SERVER_PROJECT',str(DEFAULT_PROJECT))),DEFAULT_PROJECT]
 for base in bases:
  for n in ('R85_paths.env','paths.env','R84_paths.env'):
   f=base/'evaluation_models'/n
   if not f.is_file():continue
   for line in f.read_text(encoding='utf-8').splitlines():
    tokens=shlex.split(line,comments=True)
    if tokens and tokens[0]=='export':tokens=tokens[1:]
    for token in tokens:
     key,sep,value=token.partition('=')
     if sep and key in ('CLIP_PATH','DINO_PATH','LPIPS_PATH','LPIPS_TRUNK_PATH') and not os.environ.get(key) and Path(value).exists():os.environ[key]=value
 spec=importlib.util.find_spec('lpips');cal=[];trunk=[]
 for base in bases:
  cal.append(base/'evaluation_models/lpips/alex_v0.1.pth');trunk.append(base/'evaluation_models/lpips/alexnet-owt-7be5be79.pth')
 if spec and spec.origin:cal.append(Path(spec.origin).parent/'weights/v0.1/alex.pth')
 trunk.append(Path(os.environ.get('TORCH_HOME',Path.home()/'.cache/torch'))/'hub/checkpoints/alexnet-owt-7be5be79.pth')
 for key,candidates in [('LPIPS_PATH',cal),('LPIPS_TRUNK_PATH',trunk)]:
  if os.environ.get(key) and Path(os.environ[key]).is_file():continue
  found=next((str(p.resolve()) for p in candidates if p.is_file()),None)
  if found:os.environ[key]=found
def preflight_evaluation():
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_DATASETS_OFFLINE='1')
 if str(RUNTIME) not in sys.path:sys.path.insert(0,str(RUNTIME))
 import eval_pipeline
 paths();ready=eval_pipeline.readiness(Path(os.environ.get('R88_SERVER_PROJECT',str(DEFAULT_PROJECT))));os.environ.update(ready['paths'])
 for package in ('lpips','torchvision','scikit-image'):importlib.metadata.version(package)
 for name in ('LPIPS_PATH','LPIPS_TRUNK_PATH'):
  if not os.environ.get(name) or not Path(os.environ[name]).is_file():raise RuntimeError('Offline evaluator weight missing: '+name)
 return ready

def preflight_generation(plan):
 ready=preflight_evaluation()
 import model_loader
 model_loader.preflight_runtime(plan)
 # Imports before GPU queue; no model loading or network.
 import torch,numpy,PIL,diffusers,transformers,accelerate,safetensors,lpips,torchvision,skimage,einops
 import engine,controls,rf_kv,attention,mint_backend,spatial_controls
 return ready

def collect_delivery(plan,manifest_path):
 results=RUNTIME/'results'; profiles={}
 for job in plan['jobs']:
  if job['arm'] not in ('REQ','U25'):continue
  folder=results/job['id'];record=read(folder/'record.json');stepsfile=folder/'steps.json'
  if record.get('status')!='completed' or sha(stepsfile)!=record.get('artifact_sha256',{}).get('steps.json'):raise RuntimeError('Anchor profile absent or mismatched: '+job['id'])
  steps=read(stepsfile)['edit']['steps']
  if len(steps)!=28:raise RuntimeError('Anchor profile needs 28 steps: '+job['id'])
  profiles[job['id']]={'case':job['case'],'seed':job['seed'],'steps_sha256':sha(stepsfile),'sigma_sha256':record['sigma_sha256'],'nfe_standalone':record['nfe_standalone'],'steps':[{k:s[k] for k in ('step_index','sigma','h','applied_correction_rms')} for s in steps]}
 expected=sum(j['arm'] in ('REQ','U25') for j in plan['jobs'])
 if len(profiles)!=expected:raise RuntimeError(f'Expected {expected} committed REQ/U25 profiles')
 write(results/'anchor_profiles.json',profiles)
 shutil.copy2(manifest_path,results/'source_selection_manifest.json')
 for path in [ROOT/'protocol.json', ROOT/'stats_plan.json']:
  if path.is_file() and path.suffix.lower() in ('.json','.md','.txt','.tex'):
   shutil.copy2(path,results/path.name)
 # Analysis authors can write small outputs under results/evidence or results/analysis.
 for name in ('analysis','evidence'):
  source=ROOT/name
  if source.is_dir():
   for path in source.rglob('*'):
    if path.is_file() and path.suffix.lower() in ('.json','.csv','.txt','.tex'):
     target=results/name/path.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)

def materialize(manifest_path,data_dir):
 manifest=read(manifest_path);tasks=manifest.get('tasks');template=read(RUNTIME/'r85_template.json')
 if not isinstance(tasks,list) or len(tasks)!=40:raise ValueError('manifest.json must contain exactly 40 tasks')
 seen=set();normalized=[];inputs={}
 for raw in tasks:
  t=dict(raw);case=t.get('case')
  if not isinstance(case,str) or not case or any(c in case for c in '/\\:') or case in seen:raise ValueError('Invalid or duplicate task case: '+str(case))
  seen.add(case);t.setdefault('key','pie_'+case);t.setdefault('task',t['key']);t['seeds']=[0,17]
  for k in ('key','source_prompt','target_prompt','pattern_roi','detail_roi','edit_rubric'):
   if k not in t:raise ValueError(case+' missing task field '+k)
  folder=data_dir/case
  for f in ('source.png','outer.png','full_mask.png','core.png'):
   if not (folder/f).is_file():raise FileNotFoundError(folder/f)
  out=RUNTIME/'data'/case;out.mkdir(parents=True,exist_ok=True)
  for file in folder.iterdir():
   if not file.is_file():continue
   dest=out/file.name
   if dest.exists():
    if sha(dest)!=sha(file):raise RuntimeError('Runtime input differs; use a fresh runtime: '+str(dest))
   else:
    try:os.link(file,dest)
    except OSError:shutil.copy2(file,dest)
   inputs['data/'+case+'/'+file.name]=sha(file)
  normalized.append(t)
 ordered_jobs=[];base=template['anchor_template']
 for t in normalized:
  for seed in (0,17):
   a=dict(base)
   for k in ('case','source_prompt','target_prompt','pattern_roi','detail_roi','edit_rubric'):a[k]=t[k]
   a.update(task=t['key'],seed=seed,id=f"R88_pie_{t['case']}_s{seed}_REQ",arm='REQ',label='REQ',cohort='fresh_cross_anchor',correction_budget_mode='local_coefficient')
   for k in ('r86_arm','r86_anchor','r86_beta','budget_reference_id','region_gain_cap','expected_nfe_standalone'):a.pop(k,None)
   for k in ('target_category','target_category_id','eval_pixel_region','roi_scope'):
    if k in t:a[k]=t[k]
   ordered_jobs.append(a)
   glb=dict(a);glb.update(id=f"R88_pie_{t['case']}_s{seed}_GLB_R",arm='GLB_R',label='GLB_R',r86_arm='GLOBAL',r86_beta=1.,r86_anchor=a['id'],correction_budget_mode='r86_fixed_profile');ordered_jobs.append(glb)
   u=dict(a);u.update(id=f"R88_pie_{t['case']}_s{seed}_U25",arm='U25',label='U25',correction_shape='uniform',correction_budget_mode='none')
   for k in list(u):
    if k.startswith('route_') or k in ('response_sigmas','memory_response_mode','memory_response_floor'):u.pop(k,None)
   ordered_jobs.append(u)
   for label,anchor,mode in [('REQ_U',u['id'],None),('SHUF_R',a['id'],'wrong_prompt_histmatch'),('REV_R',a['id'],'reverse')]:
    j=dict(a);j.update(id=f"R88_pie_{t['case']}_s{seed}_{label}",arm=label,label=label,r86_arm='CORR',r86_beta=1.,r86_anchor=anchor,correction_budget_mode='r86_fixed_profile')
    if mode:j['r88_map_mode']=mode
    if label=='SHUF_R':
     j['probe_target_prompt']=t['probe_target_prompt'];j['probe_donor_case']=t['probe_donor_case']
    ordered_jobs.append(j)
 plan={k:template[k] for k in ('model_revision','steps','resolution','max_length','numerics')}
 plan.update(schema_version=1,name='R88_AnalysisClosure',round_id='R88',planned=len(ordered_jobs),tasks=normalized,jobs=ordered_jobs,input_sha256=inputs,design={
  'sources':40,'seeds':[0,17],'arms':['REQ','GLB_R','U25','REQ_U','SHUF_R','REV_R'],'outputs':480,
  'factorial':'request/global direction by request-derived/uniform-derived realized per-step amplitude profile',
  'anchors':'REQ and U25 generated on their own trajectories; dependent branches replay the selected recorded RMS sequence',
  'sampling':'source/seed pairs and wrong probe targets fixed in supplied manifest',
  'source_manifest_sha256':sha(manifest_path),'protocol_sha256':sha(ROOT/'protocol.json'),
  'stats_plan_sha256':sha(ROOT/'stats_plan.json')})
 target=RUNTIME/'plan.json'
 if (RUNTIME/'results/manifest.json').exists() and target.exists() and read(target)!=plan:raise RuntimeError('Existing runtime plan differs; preserve results and use a separate package directory')
 write(target,plan)
 return plan

def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare-only',action='store_true');g.add_argument('--generate',action='store_true');g.add_argument('--evaluate',action='store_true');g.add_argument('--package',action='store_true');g.add_argument('--download-url',type=Path);p.add_argument('--manifest',type=Path,default=ROOT/'manifest.json');p.add_argument('--data-dir',type=Path,default=ROOT/'data');args=p.parse_args()
 if args.download_url:
  print_download_url(args.download_url);return
 plan=materialize(args.manifest.resolve(),args.data_dir.resolve())
 planned=len(plan['jobs'])
 os.environ.update(OUTPUT_NAME='R88_AnalysisClosure_Fast_results',R88_QUEUE_PROJECT=os.environ.get('R88_SERVER_PROJECT',str(DEFAULT_PROJECT)),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
 os.environ.setdefault('R88_REFERENCE_CACHE','on');os.environ.setdefault('R88_SOURCE_KV_CACHE','on');os.environ.setdefault('R88_KV_CACHE_MAX_MB','3072')
 if args.package:
  collect_delivery(plan,args.manifest.resolve())
  subprocess.run([sys.executable,'-u',str(RUNTIME/'runner.py'),'--pack'],cwd=RUNTIME,check=True);return
 if args.prepare_only or args.generate:preflight_generation(plan)
 else:preflight_evaluation()
 if args.prepare_only:print('Prepared 40 fresh sources / 2 seeds / 6 arms = 480 outputs; offline preflight complete.');return
 if args.generate:
  import equivalence
  equivalence.ensure(plan,require_only=True)
  os.environ['R88_PREFLIGHT_PLAN_SHA256']=sha(RUNTIME/'plan.json')
  subprocess.run([sys.executable,'-u',str(RUNTIME/'runner.py')],cwd=RUNTIME,check=True)
  progress=read(RUNTIME/'results/progress.json')
  if progress.get('status')!='completed' or progress.get('completed')!=planned:raise RuntimeError('Generation incomplete; inspect progress.json and resume')
  return
 results=RUNTIME/'results'
 if not (results/'progress.json').is_file() or read(results/'progress.json').get('completed')!=planned:raise RuntimeError(f'Evaluation requires {planned} completed results')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 command=[sys.executable,'-u',str(RUNTIME/'evaluate.py'),'--from-results',str(results),'--clip-path',os.environ['CLIP_PATH'],'--dino-path',os.environ['DINO_PATH'],'--device','cpu','--data-dir',str(RUNTIME/'data'),'--lpips-path',os.environ['LPIPS_PATH'],'--lpips-trunk-path',os.environ['LPIPS_TRUNK_PATH']]
 subprocess.run(command,cwd=RUNTIME,env=env,check=True)
 sys.path.insert(0,str(RUNTIME));import eval_pipeline
 with (results/'evaluation.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 if len(rows)!=planned or {r['id'] for r in rows}!={j['id'] for j in plan['jobs']} or any(not eval_pipeline.has_required(r) for r in rows):raise RuntimeError('Evaluation incomplete; check evaluation.csv')
 # Materialize source-paired analysis inputs once, without generating a gallery/ZIP.
 import reporting as report
 provenance=report.fingerprint(RUNTIME);report.guard_existing(RUNTIME,provenance)
 rows_by_id={r['id']:r for r in rows};combined=report.rows_for(RUNTIME,plan,provenance['fingerprint'])
 for r in combined:
  extra=rows_by_id[r['id']]
  r.update({k:v for k,v in extra.items() if k.startswith(('eval_','clip_','dino_','lpips_','pixel_'))})
 fields=list(dict.fromkeys(k for r in combined for k in r))
 for name in ('metrics.csv','comparison_metrics.csv'):
  dest=results/name;tmp=dest.with_suffix('.csv.tmp')
  with tmp.open('w',encoding='utf-8-sig',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(combined)
  os.replace(tmp,dest)
 print('Evaluation completed for 480 outputs; run analyze.py then --package for compact ZIP.')

def print_download_url(path):
 """Print an observed Jupyter URL, or a host-free path with its real base prefix."""
 from urllib.parse import urlsplit,urlunsplit,quote
 path=Path(path).resolve();servers=[]
 runtime_dirs=[]
 if os.environ.get('JUPYTER_RUNTIME_DIR'):runtime_dirs.append(Path(os.environ['JUPYTER_RUNTIME_DIR']))
 try:
  from jupyter_core.paths import jupyter_runtime_dir
  runtime_dirs.append(Path(jupyter_runtime_dir()))
 except ImportError:pass
 for folder in dict.fromkeys(runtime_dirs):
  for pattern in ('jpserver-*.json','nbserver-*.json'):
   for entry in folder.glob(pattern):
    try:servers.append(read(entry))
    except (OSError,ValueError):pass
 server=next((s for s in servers if path.is_relative_to(Path(s.get('root_dir') or s.get('notebook_dir') or '/root').resolve())),{})
 root=Path(os.environ.get('JUPYTER_ROOT_DIR') or server.get('root_dir') or server.get('notebook_dir') or '/root').resolve()
 try:relative=path.relative_to(root).as_posix()
 except ValueError:
  print('Result is outside the detected Jupyter root: '+str(path));return
 observed=os.environ.get('JUPYTER_BASE_URL') or os.environ.get('JUPYTER_SERVER_URL') or os.environ.get('JUPYTER_URL') or ''
 if not observed and server.get('url'):
  candidate=urlsplit(server['url'])
  if candidate.scheme in ('https','http') and candidate.hostname not in (None,'localhost','127.0.0.1','0.0.0.0','::1'):observed=server['url']
 parsed=urlsplit(observed)
 base=parsed.path or os.environ.get('JUPYTERHUB_SERVICE_PREFIX') or server.get('base_url') or '/jupyter/'
 for marker in ('/lab','/tree','/files'):
  if marker in base:base=base.split(marker,1)[0]
 base='/'+base.strip('/') if base.strip('/') else ''
 route=base+'/files/'+quote(relative,safe='/')+'?download=1'
 if parsed.scheme in ('https','http') and parsed.netloc:
  # Drop tokens/query/userinfo; never fabricate a public hostname.
  host=parsed.hostname or ''
  netloc=('['+host+']') if ':' in host else host
  if parsed.port:netloc+=':'+str(parsed.port)
  url=urlunsplit((parsed.scheme,netloc,route.split('?',1)[0],'download=1',''))
  print('DOWNLOAD: '+url)
 else:
  print('JUPYTER DOWNLOAD (relative to your current server): '+route)
  print('For a full clickable link, set JUPYTER_BASE_URL to your browser address through /jupyter.')
if __name__=='__main__':main()
