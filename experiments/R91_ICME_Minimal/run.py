"""Offline resumable AutoDL driver. Generation and CPU evaluation are separate stages."""
from pathlib import Path
import argparse,json,os,sys,subprocess,time,zipfile,traceback
ROOT=Path(__file__).resolve().parent;RT=ROOT/'runtime_base';sys.path.insert(0,str(RT))
os.environ['OUTPUT_NAME']='R91_delivery'
import reporting as rep
from r91_controls import validate_plan

def state(status,**kw):rep.save(ROOT/'status.json',dict(status=status,time=time.time(),pid=os.getpid(),**kw));print(status,kw,flush=True)
def discover_model(plan):
    from locate_flux import find
    return find(plan['model_revision'])

def preflight():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    counts=validate_plan(RT);plan=rep.read(RT/'plan.json')
    path=discover_model(plan);os.environ['FLUX_MODEL_PATH']=str(path)
    import model_loader
    model_loader.preflight_runtime(plan)
    cfg=rep.read(path/'transformer/config.json')
    if cfg.get('in_channels')!=64 or not cfg.get('guidance_embeds'):raise RuntimeError('Expected base FLUX.1-dev transformer with guidance embeddings')
    import torch,diffusers,transformers,accelerate,einops
    import engine,controls,mint_backend,rf_kv,spatial_controls
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    rep.save(ROOT/'model_binding.json',dict(path=str(path),historical_revision=plan['model_revision'],revision_in_path=(path.name==plan['model_revision']),validation='local full weight headers/lengths at load; four historical reproduction images before scientific jobs'))
    print('Existing model:',path,'jobs:',counts,flush=True)

def evaluate():
    import legacy_paths
    legacy_paths.preflight_evaluation()
    from r89_evaluation import evaluate_all
    evaluate_all(RT)

def pack():
    for full in (False,True):
        target=ROOT/('R91_native_images.zip' if full else 'R91_results.zip');temp=target.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for base in [RT/'results',ROOT/'analysis']:
                if not base.exists():continue
                for p in base.rglob('*'):
                    if not p.is_file() or p.suffix=='.tmp':continue
                    native=p.name=='raw.png' or p.name.endswith('_source_eval.png')
                    if full==native:z.write(p,p.relative_to(ROOT))
            for p in [ROOT/'protocol.json',ROOT/'model_binding.json',ROOT/'status.json',RT/'plan.json',ROOT/'historical_edit_mae.csv']:
                if p.exists():z.write(p,p.relative_to(ROOT))
            if not full:
                for p in (RT/'historical').glob('*'):
                    if p.is_file() and p.suffix in ('.json','.csv'):z.write(p,p.relative_to(ROOT))
        os.replace(temp,target)
    print('Saved R91_results.zip + R91_native_images.zip (retain both).',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['all','generate','evaluate','analyze','pack','preflight']);p.add_argument('--through',choices=['sentinel','maps','seed1','seed2','seed3'],default='seed3');args=p.parse_args()
    os.environ['R91_THROUGH']=args.through
    with rep.file_lock(ROOT/'.driver.lock'):
        try:
            if args.action in ('all','generate','preflight'):
                state('preflight');preflight()
                if args.action=='preflight':state('ready');return
                (RT/'PAUSE').unlink(missing_ok=True)
                state('generating',through=args.through)
                import runner
                import signal
                for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,runner.request_pause)
                runner.work(RT)
                progress=rep.read(RT/'results/progress.json')
                if progress.get('completed')!=724:
                    state('paused',completed=progress.get('completed'),through=args.through);pack();return
                state('generation_complete',completed=724);pack()
                if args.action=='generate':return
            if args.action in ('all','evaluate'):
                state('evaluating_cpu')
                try:evaluate()
                except Exception as exc:
                    state('evaluation_pending',error=str(exc));pack();raise
            if args.action in ('all','evaluate','analyze'):
                state('analyzing');subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
                state('complete');pack()
            if args.action=='pack':pack()
        except BaseException as exc:
            if not (ROOT/'status.json').exists() or rep.read(ROOT/'status.json')['status']!='evaluation_pending':state('failed',error=str(exc))
            raise
if __name__=='__main__':main()
