"""Two-projection controls for the pinned SAM-Flow source-relative state.

All arithmetic is on the candidate's own displacement. Native returns the
original upstream tensors unchanged. No generation or evaluation mask is read.
"""
import json, math
from pathlib import Path
import torch

def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt().cpu())

class ProjectionControl:
    def __init__(self, job, anchor_rows=None, progress=None, retain_latent=False):
        self.job = job
        self.rows = []
        self.anchor = {int(r['step']):r for r in (anchor_rows or [])}
        self.progress = progress or (lambda x: None)
        self.retain_latent = retain_latent
        self.final_latent = None
        self.constants = None
        if job['arm']=='two_constant':
            rows=list(self.anchor.values())
            denominator=sum(r['residual_rms'] for r in rows)
            if not rows or denominator <= 1e-12:
                raise ValueError('Native profile cannot calibrate constants: zero residual')
            self.constants=[sum(r[k+'_rms'] for r in rows)/denominator for k in ('model','visible')]
            if any(not (0 <= p <= 1+1e-6) for p in self.constants):
                raise ValueError('Constant calibration exceeds the native convex factor range')

    @staticmethod
    def match(direction, target):
        if not math.isfinite(target) or target<0:
            raise ValueError('Target RMS must be finite and nonnegative')
        magnitude=rms(direction)
        if target==0:
            return torch.zeros_like(direction), 0.0
        if magnitude<=1e-12:
            raise ValueError('Positive target RMS with zero direction: infeasible')
        gain=target/magnitude
        return direction*gain, gain

    def apply(self, step, sigma, next_sigma, anchor, residual, p_model, p_visible,
              native_model, native_visible, previous_model, previous_visible, velocity):
        arm=self.job['arm']; scale=float(self.job.get('scale',1.0))
        target_model=target_visible=None
        gains=[None,None]
        if arm=='sam_native':
            # Preserve upstream association/order, including visible projection.
            model, visible=native_model,native_visible
        elif arm=='uniform':
            model=anchor+scale*residual
            visible=model
            gains=[scale,scale]
        elif arm=='two_constant':
            model=anchor+self.constants[0]*residual
            visible=anchor+self.constants[1]*residual
            gains=self.constants
        elif arm in ('uniform_replay','spatial_replay'):
            reference=self.anchor.get(int(step))
            if reference is None:
                raise ValueError('Anchor has no step '+str(step))
            if abs(float(sigma)-reference['sigma'])>1e-6 or abs(float(next_sigma)-reference['next_sigma'])>1e-6:
                raise ValueError('Anchor/candidate time-grid mismatch')
            target_model=reference['model_rms']; target_visible=reference['visible_rms']
            dm,dv=(residual,residual) if arm=='uniform_replay' else (p_model*residual,p_visible*residual)
            dm,gains[0]=self.match(dm,target_model)
            dv,gains[1]=self.match(dv,target_visible)
            model=anchor+dm; visible=anchor+dv
        else:
            raise ValueError('Unknown projection arm '+arm)
        tensors=(model,visible,residual)
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise FloatingPointError('Nonfinite projection state')
        row=dict(step=int(step),sigma=float(sigma),next_sigma=float(next_sigma),
                 residual_rms=rms(residual),model_rms=rms(model-anchor),visible_rms=rms(visible-anchor),
                 model_state_rms=rms(model.to(velocity.dtype).float()-anchor),
                 visible_state_rms=rms(visible.to(velocity.dtype).float()-anchor),
                 state_dtype=str(velocity.dtype),
                 model_increment_rms=rms(model-previous_model),visible_increment_rms=rms(visible-previous_visible),
                 velocity_rms=rms(velocity),euler_increment_rms=abs(float(next_sigma)-float(sigma))*rms(velocity),
                 model_factor_min=float(p_model.min()),model_factor_max=float(p_model.max()),
                 visible_factor_min=float(p_visible.min()),visible_factor_max=float(p_visible.max()),
                 target_model_rms=target_model,target_visible_rms=target_visible,
                 model_gain=gains[0],visible_gain=gains[1])
        for key,target in [('model',target_model),('visible',target_visible)]:
            row[key+'_relative_error']=None if target is None or target==0 else (row[key+'_rms']-target)/target
            row[key+'_absolute_error']=None if target is None else row[key+'_rms']-target
            if target is not None:
                state_target=self.anchor[int(step)][key+'_state_rms']
                row[key+'_target_state_rms']=state_target
                row[key+'_state_match_absolute_error']=row[key+'_state_rms']-state_target
                row[key+'_state_match_relative_error']=(row[key+'_state_rms']-state_target)/state_target if state_target else None
        if arm.endswith('replay'):
            # FP32 normalized projection. Error measured after anchor addition.
            for key,target in [('model',target_model),('visible',target_visible)]:
                tolerance=1e-6+1e-4*target
                if abs(row[key+'_absolute_error'])>tolerance:
                    raise ValueError(f'{key} replay RMS error exceeds 1e-6 + 1e-4*target')
        self.rows.append(row)
        if self.retain_latent:
            self.final_latent=visible.detach().cpu().clone()
        self.progress({'stage':'editing','step':int(step)+1,'steps':28,'arm':arm})
        return model,visible

    def summary(self):
        if not self.rows:
            raise ValueError('No active projection steps were observed')
        result={'active_steps':len(self.rows),'constants':self.constants,
                'profile_units':'RMS of source-relative projected latent displacement; active-step mean',
                'profile_stage':'FP32 projection before upstream state dtype cast; quantized state RMS also recorded',
                'euler_dose':sum(r['euler_increment_rms'] for r in self.rows),
                'replay_scope':'two projection displacement profiles, not equal final edit response',
                'convex_constraint':'replay gains may exceed 1; extended intervention is recorded, never clipped'}
        for key in ('model','visible'):
            realized=[r[key+'_rms'] for r in self.rows]
            result[key+'_mean_rms']=sum(realized)/len(realized)
            targets=[r['target_'+key+'_rms'] for r in self.rows]
            if all(t is not None for t in targets):
                total=sum(targets); actual=sum(realized)
                result[key+'_cumulative_relative_error']=(actual-total)/total if total else None
                result[key+'_integrated_absolute_relative_error']=sum(abs(a-b) for a,b in zip(realized,targets))/total if total else None
                result[key+'_shape_l1']=sum(abs(a/actual-b/total) for a,b in zip(realized,targets)) if actual and total else None
                result[key+'_maximum_stepwise_relative_error']=max([abs(a-b)/b for a,b in zip(realized,targets) if b>0],default=None)
                result[key+'_gain_above_one_steps']=sum((r[key+'_gain'] or 0)>1 for r in self.rows)
                result[key+'_max_state_relative_error']=max([abs(r[key+'_state_match_relative_error']) for r in self.rows if r[key+'_state_match_relative_error'] is not None],default=None)
                result[key+'_realized_state_match_passed']=all(abs(r[key+'_state_match_absolute_error'])<=1e-6+0.01*r[key+'_target_state_rms'] for r in self.rows)
        return result

    def save(self, folder):
        folder=Path(folder)
        (folder/'profile.json').write_text(json.dumps(self.rows,indent=2,allow_nan=False)+'\n',encoding='utf8')
        return self.summary()
