"""Pinned SAM-Flow FLUX adapter. No weights are downloaded or quantized.

Scientific changes are confined to the two effective state projections. The
vendored upstream files stay intact; exact source substitutions create an
instrumented in-memory function with an explicit optional control argument.
"""
from pathlib import Path
import contextlib, copy, hashlib, inspect, json, random, time
import numpy as np
import torch
from PIL import Image
import yaml
from projection_control import ProjectionControl

ROOT=Path(__file__).resolve().parent
COMMIT='8167978666e6df11e987a453934b89ec422bd3a4'

def save_json(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    tmp.replace(path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _replace_once(source, old, new):
    if source.count(old)!=1:
        raise RuntimeError('Pinned-source hook anchor differs: '+old[:70])
    return source.replace(old,new,1)

def build_functions(module):
    source=inspect.getsource(module.run_sam_flow_flux_case)
    source=_replace_once(source,'    case_dir: str | Path,\n) -> Path:',
                         '    case_dir: str | Path,\n    control=None,\n    capture=None,\n) -> Path:')
    source=_replace_once(source,'    device = next(pipe.transformer.parameters()).device',
                         '    device = pipe._execution_device')
    source=_replace_once(source,'    unpacked_out = pipe._unpack_latents(',
                         '    if capture is not None:\n        capture(zt_visible.detach().cpu().clone())\n\n    unpacked_out = pipe._unpack_latents(')
    reference_source=source
    source=_replace_once(source,'        zt_edit_full = zt_edit_full.to(torch.float32)',
                         '        _previous_model = zt_model\n        _previous_visible = zt_visible\n        zt_edit_full = zt_edit_full.to(torch.float32)')
    source=_replace_once(source,'        zt_edit_full = zt_edit_full.to(V_delta_avg.dtype)',
        '''        if control is not None:
            _p_model = anchor_rho * W_soft_state
            _p_visible = anchor_rho * (core_keep * W_soft_state + ring_only * ring_blend * W_hard_state)
            zt_model, zt_visible = control.apply(
                step=i, sigma=t_i, next_sigma=t_im1, anchor=source_anchor,
                residual=zt_edit_full-source_anchor, p_model=_p_model, p_visible=_p_visible,
                native_model=zt_model, native_visible=zt_visible,
                previous_model=_previous_model, previous_visible=_previous_visible, velocity=V_delta_avg)
        zt_edit_full = zt_edit_full.to(V_delta_avg.dtype)''')
    reference_ns=dict(module.__dict__); instrumented_ns=dict(module.__dict__)
    exec(compile(reference_source,'<pinned_sam_reference>','exec'),reference_ns)
    exec(compile(source,'<pinned_sam_instrumented>','exec'),instrumented_ns)
    return reference_ns['run_sam_flow_flux_case'],instrumented_ns['run_sam_flow_flux_case'],instrumented_ns

class Runtime:
    def __init__(self,model_path,offload='auto',dtype='bfloat16'):
        import diffusers, transformers
        from diffusers import FluxPipeline
        from sam_flow import sam_flow_flux as module
        from sam_flow import flowedit_flux as scout_module
        if not torch.cuda.is_available():
            raise RuntimeError('This generation package requires CUDA; CPU analysis is separate')
        self.dtype=getattr(torch,dtype)
        if self.dtype not in (torch.float16,torch.bfloat16,torch.float32): raise ValueError(dtype)
        if dtype=='bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('GPU lacks BF16 support; explicitly choose float16 and re-run engineering')
        if diffusers.__version__!='0.35.1':
            raise RuntimeError('Use diffusers==0.35.1 for this pinned adapter')
        self.model_path=str(Path(model_path).resolve())
        self.pipe=FluxPipeline.from_pretrained(self.model_path,torch_dtype=self.dtype,local_files_only=True)
        if offload=='auto':
            offload='model' if torch.cuda.get_device_properties(0).total_memory>=26*1024**3 else 'sequential'
        self.offload=offload
        if offload=='model': self.pipe.enable_model_cpu_offload()
        elif offload=='sequential': self.pipe.enable_sequential_cpu_offload()
        elif offload=='none': self.pipe.to('cuda')
        else: raise ValueError(offload)
        self.pipe.set_progress_bar_config(disable=True)
        self.module=module
        self.reference,self.controlled,self.controlled_globals=build_functions(module)
        scout_source=inspect.getsource(scout_module.generate_scout_image_flux)
        scout_source=_replace_once(scout_source,'    device = next(pipe.transformer.parameters()).device','    device = pipe._execution_device')
        scout_ns=dict(scout_module.__dict__)
        exec(compile(scout_source,'<pinned_scout_device_adapter>','exec'),scout_ns)
        self.scout=scout_ns['generate_scout_image_flux']
        self.config=yaml.safe_load((ROOT/'configs/flux.yaml').read_text(encoding='utf8'))
        self.config['run']['save_debug_maps']=False
        self.scheduler_config=dict(self.pipe.scheduler.config)
        self.original_encoder=self.pipe.encode_prompt
        self.embeddings={}; self.embedding_case=None
        self.map_cache={}; self.map_hits=0; self.cache_maps=False
        original_map=self.controlled_globals['build_union_edit_map']
        def map_cached(**kwargs):
            key=(float(kwargs['t']),float(kwargs['t_i']),float(kwargs['current_sigma']),
                 tuple(kwargs['src_indices']),tuple(kwargs['tgt_indices']))
            if self.cache_maps and key in self.map_cache:
                self.map_hits+=1
                return self.map_cache[key].to(kwargs['x_src_packed'].device),{'mode':'cached_union'}
            cpu_rng=torch.random.get_rng_state()
            gpu_rng=torch.cuda.get_rng_state()
            value,debug=original_map(**kwargs)
            if self.cache_maps and torch.equal(cpu_rng,torch.random.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state()):
                self.map_cache[key]=value.detach().cpu().clone()
            return value,debug
        self.controlled_globals['build_union_edit_map']=map_cached
        def encode_cached(*args,**kwargs):
            if args: return self.original_encoder(*args,**kwargs)
            key=json.dumps({k:v for k,v in kwargs.items() if k!='device'},sort_keys=True,default=str)
            if key not in self.embeddings:
                out=self.original_encoder(**kwargs)
                self.embeddings[key]=tuple(t.detach().cpu().clone() for t in out)
                return out
            device=kwargs.get('device',self.pipe._execution_device)
            return tuple(t.to(device) for t in self.embeddings[key])
        self.pipe.encode_prompt=encode_cached
        self.environment={'torch':torch.__version__,'diffusers':diffusers.__version__,
                          'transformers':transformers.__version__,'dtype':dtype,'offload':offload,
                          'gpu':torch.cuda.get_device_name(0),'upstream_commit':COMMIT}
        self.nfe=0; self.forward_calls=0; self.callback=lambda x:None; self.phase='idle'; self.last_heartbeat=0
        def counted(model,args,kwargs):
            x=kwargs.get('hidden_states')
            self.nfe += int(x.shape[0]) if x is not None else 1
            self.forward_calls += 1
            now=time.monotonic()
            if now-self.last_heartbeat>2:
                self.callback({'stage':self.phase,'nfe':self.nfe,'forward_calls':self.forward_calls})
                self.last_heartbeat=now
        self.counter=self.pipe.transformer.register_forward_pre_hook(counted,with_kwargs=True)
        code_hash=hashlib.sha256(Path(__file__).read_bytes()+Path(module.__file__).read_bytes()+(ROOT/'projection_control.py').read_bytes()).hexdigest()
        identity={'environment':self.environment,'model_directory':self.model_path,'config':self.config,'adapter_code':code_hash}
        # Configuration fingerprint, not an expensive full-model weight hash.
        self.cache_identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]

    def case(self,case):
        c=dict(case); case_id=str(c.get('case',c.get('image_name')))
        c.update(case=case_id,image_name=case_id,target_code='edit',record_index=c.get('record_index',0),target_index=0)
        p=Path(c.get('image_path',c.get('source_image',f'data/{case_id}/source.png')))
        c['image_path']=str(p if p.is_absolute() else ROOT/p)
        c.setdefault('unchanged_tokens',[]); c.setdefault('seed',0)
        if self.embedding_case!=case_id:
            self.embeddings.clear(); self.map_cache.clear(); self.embedding_case=case_id
        return c

    def validate_case(self,case):
        c=self.case(case)
        if not Path(c['image_path']).is_file(): raise FileNotFoundError(c['image_path'])
        with Image.open(c['image_path']) as im:
            if im.size!=(512,512): raise ValueError('Expected native 512x512 input')
        from types import SimpleNamespace
        # Official helper falls back to CLIP tokenizer; attention columns are T5.
        # Validate T5 directly so a fallback cannot silently point at wrong tokens.
        token_pipe=SimpleNamespace(tokenizer_2=self.pipe.tokenizer_2,tokenizer=None)
        result={}
        for side in ('source','target'):
            tokens=c[side+'_mask_tokens']
            idx=self.module.get_token_indices(token_pipe,c[side+'_prompt'],tokens)
            if not tokens or not idx: raise ValueError(f'{c["case"]} {side} mask tokens not found in T5 prompt')
            result[side+'_indices']=idx
        return result

    def reset(self,seed):
        from diffusers import FlowMatchEulerDiscreteScheduler
        self.pipe.scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
        seed_all(int(seed))

    def scout_path(self,c,progress):
        identity={'runtime':self.cache_identity,'case':c,'flowedit':self.config['flowedit'],
                  'source_file':hashlib.sha256(Path(c['image_path']).read_bytes()).hexdigest()}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
        folder=ROOT/'outputs'/'_cache'/c['case']/key
        path=folder/'edit_scout.png'; marker=folder/'scout.json'
        if path.exists() and marker.exists(): return path,json.loads(marker.read_text(encoding='utf8'))
        folder.mkdir(parents=True,exist_ok=True)
        self.phase='scout'; progress({'stage':'scout','case':c['case']})
        self.reset(c['seed']); before=self.nfe; started=time.monotonic()
        with torch.inference_mode():
            path=self.scout(self.pipe,c,self.config,folder,folder)
        metadata={'nfe':self.nfe-before,'seconds':time.monotonic()-started,'identity':key}
        save_json(marker,metadata)
        return Path(path),metadata

    def engineering_check(self,case,output_dir,progress_callback):
        folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
        self.callback=progress_callback; c=self.case(case); self.validate_case(c)
        self.cache_maps=False
        scout,_=self.scout_path(c,progress_callback)
        tensors={}; timings={}; nfes={}
        for name,function,control in [('reference',self.reference,None),
                                      ('observe',self.controlled,ProjectionControl({'arm':'sam_native'},progress=progress_callback))]:
            self.phase='engineering_'+name; progress_callback({'stage':self.phase,'case':c['case']})
            out=folder/name; out.mkdir(exist_ok=True)
            self.reset(c['seed']); before=self.nfe; started=time.monotonic()
            with torch.inference_mode():
                function(self.pipe,c,self.config,scout,out,control=control,capture=lambda t:tensors.update({name:t}))
            torch.cuda.synchronize()
            timings[name]=time.monotonic()-started; nfes[name]=self.nfe-before
            if control is not None: control.save(out)
        delta=(tensors['reference'].float()-tensors['observe'].float()).abs()
        report={'passed':bool(torch.equal(tensors['reference'],tensors['observe'])),
                'max_abs_latent_difference':float(delta.max()),'criterion':'bitwise equality of final visible latent',
                'seconds':timings,'nfe':nfes,'environment':self.environment,
                'scope':'upstream native reference with device adapter versus observation-only instrumentation; shared dtype/scout/prompts',
                'not_claimed':'paper-quality reproduction, other dtype parity, or validated controlled effects'}
        save_json(folder/'parity.json',report)
        return report

    def run_job(self,case,job,output_dir,progress_callback):
        c=self.case(case); self.callback=progress_callback
        self.cache_maps=True
        folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic(); before=self.nfe; previous_map_hits=self.map_hits
        config=copy.deepcopy(self.config)
        report={'case':c['case'],'job_id':job['job_id'],'arm':job['arm'],'scale':job.get('scale',1),
                'seed':c['seed'],'stage':job.get('stage'),'environment':self.environment,
                'source_prompt':c['source_prompt'],'target_prompt':c['target_prompt'],
                'gt_mask_used_for_generation':False,'status':'running'}
        if job['arm']=='flowedit':
            nmax=int(job.get('n_max',24)); report['n_max']=nmax
            if nmax==24:
                raw,meta=self.scout_path(c,progress_callback)
                report['reused_scout']=True; report['reference_cost']=meta
            else:
                self.phase='flowedit'; progress_callback({'stage':'flowedit','case':c['case'],'n_max':nmax})
                config['flowedit']['n_max']=nmax
                self.reset(c['seed'])
                with torch.inference_mode(): raw=self.scout(self.pipe,c,config,folder,folder)
                report['reused_scout']=False
            report['baseline_label']='FlowEdit configuration from SAM-Flow official repository; n_avg=8, shared declared dtype'
        else:
            scout,_=self.scout_path(c,progress_callback)
            anchor=None
            if job.get('anchor_job_id'):
                anchor_folder=folder.parent/job['anchor_job_id']
                marker=anchor_folder/'report.json'
                if not marker.exists() or json.loads(marker.read_text(encoding='utf8')).get('status')!='completed':
                    raise ValueError('Required anchor job is not completed: '+job['anchor_job_id'])
                anchor_report=json.loads(marker.read_text(encoding='utf8'))
                if anchor_report.get('case')!=c['case'] or anchor_report.get('seed')!=c['seed'] or anchor_report.get('environment')!=self.environment:
                    raise ValueError('Anchor source/seed/execution environment differs from candidate')
                anchor=json.loads((anchor_folder/'profile.json').read_text(encoding='utf8'))
            if job['arm']=='sam_native': config['sam_flow']['anchor_rho']=float(job.get('scale',1.0))
            control=ProjectionControl(job,anchor,progress_callback)
            self.phase='editing'; self.reset(c['seed'])
            with torch.inference_mode():
                raw=self.controlled(self.pipe,c,config,scout,folder,control=control)
            report['profile']=control.save(folder)
            report['anchor_job_id']=job.get('anchor_job_id')
            report['matching_roundoff']='FP32 projection RMS matched; state dtype cast RMS separately logged'
        torch.cuda.synchronize()
        with Image.open(raw) as im:
            im=im.convert('RGB')
            if im.size!=(512,512): raise ValueError('Output image is not 512x512')
            im.save(folder/'native.png')
            preview=im.copy(); preview.thumbnail((256,256)); preview.save(folder/'preview.jpg',quality=90)
        report.update(status='completed',nfe=self.nfe-before,seconds=time.monotonic()-started,
                      reused_attention_maps=self.map_hits-previous_map_hits,
                      native_path=str(folder/'native.png'),config=config)
        save_json(folder/'report.json',report)
        return report

def load_runtime(model_path,offload='auto',dtype='bfloat16'):
    return Runtime(model_path,offload,dtype)
