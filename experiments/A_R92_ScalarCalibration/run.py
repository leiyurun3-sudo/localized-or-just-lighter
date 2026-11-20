"""A-R92 driver: generation, offline CPU evaluation, paired analysis, compact ZIP."""
from pathlib import Path
import argparse,json,os,sys,time,subprocess,zipfile,traceback
ROOT=Path(__file__).resolve().parent;RT=ROOT/'runtime_base'
sys.path.insert(0,str(RT))
import reporting as rep

def state(status,**kw):
    value=dict(status=status,pid=os.getpid(),updated=time.time(),**kw)
    rep.save(ROOT/'status.json',value);print(status,kw,flush=True)

def preflight():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    from r92_controls import validate_plan
    plan=validate_plan(RT)
    from locate_flux import find,check
    explicit=os.environ.get('FLUX_MODEL_PATH')
    if explicit:
        path=Path(explicit).expanduser()
        error=check(path)
        if error:raise RuntimeError('FLUX_MODEL_PATH: '+error)
    else:path=find(plan['model_revision'])
    os.environ['FLUX_MODEL_PATH']=str(path)
    import model_loader
    model_loader.preflight_runtime(plan)
    import torch,einops
    if not torch.cuda.is_available():raise RuntimeError('Generation requires CUDA; no generation was run')
    rep.save(ROOT/'model_binding.json',dict(path=str(path),expected_historical_revision=plan['model_revision'],
        revision_directory_matches=path.name==plan['model_revision'],note='Existing local model reused; no download or quantization'))
    print('Model:',path,'new outputs:',len(plan['jobs']),flush=True)
    return plan

def pack():
    final=ROOT/'A_R92_results.zip';tmp=final.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for base in [RT/'results',ROOT/'analysis']:
            if not base.exists():continue
            for p in base.rglob('*'):
                if not p.is_file() or p.name.endswith('.tmp'):continue
                # Native final PNGs are needed to finish evaluation locally if offline models are absent.
                if p.name in ['pattern.png','pattern.jpg']:continue
                z.write(p,p.relative_to(ROOT))
        for name in ['protocol.json','scalar_coefficients.csv','status.json','status_evaluation.json','model_binding.json','local_evaluation_environment.json','runtime_base/plan.json','runtime_base/historical/anchors.json','runtime_base/historical/scores.csv','historical_edit_mae.csv']:
            p=ROOT/name
            if p.is_file():z.write(p,name)
        for p in (RT/'data').glob('*/outer.png'):z.write(p,p.relative_to(ROOT))
        for name in ['analyze.py','evaluate.py','metric_helpers.py']:
            if (ROOT/name).exists():z.write(ROOT/name,name)
    os.replace(tmp,final)
    print('Result ZIP:',final,round(final.stat().st_size/1e6,1),'MB',flush=True)

def child(script,*args):
    p=subprocess.Popen([sys.executable,'-u',str(ROOT/script),*args],cwd=ROOT)
    s=rep.read(ROOT/'status.json');s.update(child_pid=p.pid,updated=time.time());rep.save(ROOT/'status.json',s)
    return p.wait()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['all','generate','evaluate','analyze','pack','preflight','_generate'],default='all',nargs='?')
    args=parser.parse_args()
    if args.action=='_generate':
        preflight()
        os.environ['OUTPUT_NAME']='A_R92_delivery'
        # A shared parent lock prevents overlap with other generation packages.
        os.environ.setdefault('R88_QUEUE_PROJECT','/root/autodl-tmp')
        import runner,signal
        for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,runner.request_pause)
        runner.work(RT)
        progress=rep.read(RT/'results/progress.json')
        return 0 if progress.get('completed')==80 else 3
    if args.action=='preflight':preflight();return 0
    if args.action=='pack':pack();return 0
    with rep.file_lock(ROOT/'.driver.lock'):
        try:
            if args.action in ['all','generate']:
                (RT/'PAUSE').unlink(missing_ok=True)
                state('generating')
                code=child('run.py','_generate')
                if code:
                    state('paused' if code==3 else 'generation_failed',exit_code=code)
                    pack();return code
                state('generation_complete',completed=80);pack()
                if args.action=='generate':return 0
            if args.action in ['all','evaluate']:
                state('evaluating_cpu',completed=80)
                # Generation process has exited, releasing FLUX and CUDA memory first.
                code=child('evaluate.py')
                if code:
                    state('evaluation_pending',completed=80,exit_code=code,
                          next_action='Download A_R92_results.zip for local evaluation; native outputs are included. No regeneration needed.')
                    child('analyze.py','--diagnostics-only');pack();return 0
            state('analyzing')
            diagnostics_only=args.action=='analyze' and not (RT/'results/evaluation.csv').exists()
            code=child('analyze.py',*(['--diagnostics-only'] if diagnostics_only else []))
            if code:raise RuntimeError('Analysis stopped with exit code '+str(code))
            state('evaluation_pending' if diagnostics_only else 'complete',completed=80);pack();return 0
        except BaseException as e:
            state('failed',error=str(e));pack();traceback.print_exc();return 1

if __name__=='__main__':sys.exit(main())
