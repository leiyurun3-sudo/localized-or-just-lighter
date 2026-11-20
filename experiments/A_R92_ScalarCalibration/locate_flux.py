"""Read-only discovery of complete local FLUX.1-dev pipelines; no downloads."""
from pathlib import Path
import os,json,sys

def check(p):
    try:
        cfg=json.loads((p/'transformer/config.json').read_text())
        if cfg.get('in_channels')!=64 or not cfg.get('guidance_embeds'):
            return 'not base FLUX.1-dev'
        for sub in ('transformer','text_encoder','text_encoder_2','vae'):
            folder=p/sub
            if not (folder/'config.json').is_file():return sub+'/config.json missing'
            names=set()
            for idx in folder.glob('*.index.json'):
                names.update(json.loads(idx.read_text())['weight_map'].values())
            if not names:names={f.name for f in folder.glob('*.safetensors')}
            if not names:return sub+': no safetensors weights'
            for name in sorted(names):
                f=folder/name
                if not f.is_file():return 'missing/broken link: '+str(f)
                with f.open('rb') as h:
                    prefix=h.read(8)
                    if len(prefix)!=8:return 'invalid weight header: '+str(f)
                    n=int.from_bytes(prefix,'little')
                    if not 2<n<100_000_000:return 'invalid weight header: '+str(f)
                    header=json.loads(h.read(n))
                end=max(v['data_offsets'][1] for k,v in header.items() if k!='__metadata__')
                if f.stat().st_size!=8+n+end:return 'incomplete weight: '+str(f)
        for sub in ('scheduler','tokenizer','tokenizer_2'):
            if not (p/sub).is_dir():return sub+' missing'
        return None
    except (OSError,ValueError,KeyError,TypeError) as e:return str(e)

def find(revision=None):
    roots=[Path('/root/autodl-tmp'),Path('/root/autodl-fs'),Path('/root/.cache/huggingface')]
    roots += [Path(os.environ[k]).expanduser() for k in ('FLUX_MODEL_PATH','HF_HOME','HF_HUB_CACHE') if os.environ.get(k)]
    found=set()
    for root in roots:
        if not root.is_dir():continue
        for folder,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in ('node_modules','.git','__pycache__','blobs')]
            if 'model_index.json' in files:
                found.add(Path(folder).resolve());dirs[:]=[]
    valid=[]
    for p in sorted(found):
        error=check(p)
        print(('SKIP '+str(p)+' : '+error) if error else 'COMPLETE '+str(p),file=sys.stderr)
        if error is None:valid.append(p)
    explicit=os.environ.get('FLUX_MODEL_PATH')
    if explicit and Path(explicit).expanduser().resolve() in valid:return Path(explicit).expanduser().resolve()
    pinned=[p for p in valid if p.name==revision]
    if pinned:return pinned[0]
    if len(valid)==1:return valid[0]
    if not valid:raise RuntimeError('No complete local pipeline found. See missing file paths above; no files were modified or downloaded.')
    raise RuntimeError('Multiple complete models: set FLUX_MODEL_PATH explicitly to the intended model directory.')

if __name__=='__main__':
    plan=Path(__file__).resolve().parent/'runtime_base/plan.json'
    revision=json.loads(plan.read_text())['model_revision'] if plan.is_file() else None
    try:print(find(revision))
    except RuntimeError as e:print(str(e),file=sys.stderr);sys.exit(1)
