"""R88 immutable fixed-reference caches; no current-state/target-field cache.

The runner owns one instance per source/seed group. Content keys additionally
cover the actual CPU reference states, source embeddings, image IDs, sigma
grid, model instance, precision and original attention processors. Source
native probes and memory-capture forwards are separate execution namespaces.
"""
import hashlib
import json
import os

import torch


def _flag(name, default='on'):
    value = os.environ.get(name, default).strip().lower()
    if value in ('on', '1', 'true', 'yes'):
        return True
    if value in ('off', '0', 'false', 'no'):
        return False
    raise ValueError(name + ' must be on/off')


def cache_enabled():
    return _flag('R88_REFERENCE_CACHE')


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), str(value.dtype))).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _clone_tree(value, device=None):
    if torch.is_tensor(value):
        result = value.detach().to('cpu' if device is None else device)
        return result.clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(v, device) for v in value)
    if isinstance(value, dict):
        return {k: _clone_tree(v, device) for k, v in value.items()}
    raise TypeError('Reference cache supports only tensors, tuples and dictionaries')


def _tree_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    return sum(_tree_bytes(v) for v in (value.values() if isinstance(value, dict) else value))


class ReferenceCache:
    """Bounded CPU-only cache, invalidated on any fixed-reference scope change.

    R88_REFERENCE_CACHE=off disables both caches. R88_SOURCE_KV_CACHE=off
    retains native source-probe sharing while retaining original KV capture.
    The 3 GiB default bank limit covers only this one source/seed group.
    """
    def __init__(self):
        self.enabled = cache_enabled()
        self.kv_enabled = self.enabled and _flag('R88_SOURCE_KV_CACHE')
        self.max_bank_bytes = int(float(os.environ.get('R88_KV_CACHE_MAX_MB', '3072')) * 1024**2)
        if self.max_bank_bytes < 0:
            raise ValueError('R88_KV_CACHE_MAX_MB must be nonnegative')
        self.clear()

    def clear(self):
        self.scope = None
        self.source_fields = {}
        self.captures = {}
        self.bank_bytes = 0
        self.stats = dict(native_source_hits=0, native_source_saved_nfe=0,
                          capture_hits=0, capture_saved_nfe=0,
                          capture_misses=0, capture_capacity_skips=0)

    def bind(self, b, ids, src, job, trajectory, sigmas):
        if not self.enabled:
            return
        model = b.pipe.transformer
        if model.training:
            raise ValueError('Fixed-reference cache requires eval-mode transformer')
        processors = []
        for index, block in enumerate(model.single_transformer_blocks):
            processor = block.attn.processor
            if getattr(processor, 'owner', None) is not None:
                raise ValueError('Bind reference cache before installing attention memory')
            state = tuple(sorted((k, repr(v)) for k, v in vars(processor).items()
                                 if isinstance(v, (str, int, float, bool, type(None)))))
            processors.append((index, id(processor), type(processor).__module__,
                               type(processor).__qualname__, state))
        # Revision is fixed by the loader; identity prevents crossing model loads.
        # Parameter version counters also detect in-process weight mutations.
        weights = [(name, int(p._version), tuple(p.shape), str(p.dtype))
                   for name, p in model.named_parameters()]
        hooks = []
        for name, module in model.named_modules():
            pre = tuple((str(k), id(v)) for k, v in module._forward_pre_hooks.items())
            post = tuple((str(k), id(v)) for k, v in module._forward_hooks.items())
            offload = getattr(module, '_hf_hook', None)
            if pre or post or offload is not None:
                hooks.append((name, pre, post, None if offload is None else id(offload)))
        spec = dict(model_instance=id(model), model_revision=getattr(b, 'model_revision', None),
                    dtype=str(b.dtype), device=str(b.device), width=b.width, height=b.height,
                    max_length=b.max_length, processors=processors, weight_versions=weights, hooks=hooks,
                    attention_backends=(torch.backends.cuda.flash_sdp_enabled(),
                                        torch.backends.cuda.mem_efficient_sdp_enabled(),
                                        torch.backends.cuda.math_sdp_enabled()),
                    matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                    case=job['case'], seed=int(job['seed']), source_prompt=job['source_prompt'],
                    guidance=float(job['guidance']), gamma=float(job['gamma']),
                    inv_substeps=int(job['inv_substeps']),
                    strength=float(getattr(b, 'experiment_strength', 1.0)),
                    sigma_values=[float(s) for s in sigmas],
                    image_ids=_tensor_digest(ids), source_embeddings=[_tensor_digest(t) for t in src],
                    reference_states=[_tensor_digest(t) for t in trajectory])
        scope = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        if scope != self.scope:
            self.clear()
            self.scope = scope

    def snapshot(self):
        return dict(self.stats, enabled=self.enabled, kv_enabled=self.kv_enabled,
                    scope=self.scope, cpu_bank_bytes=self.bank_bytes,
                    cpu_bank_limit_bytes=self.max_bank_bytes,
                    cached_native_nodes=len(self.source_fields), cached_capture_nodes=len(self.captures))

    def native_source(self, b, ref, sigma, src, ids, guidance, node):
        """Only source-prompt, fixed-reference, native memory-off probe calls."""
        key = (int(node), float(sigma), float(guidance))
        if self.enabled and self.scope is None:
            raise RuntimeError('Reference cache must be bound before use')
        if self.enabled and key in self.source_fields:
            field, cost = self.source_fields[key]
            self.stats['native_source_hits'] += 1
            self.stats['native_source_saved_nfe'] += cost
            return _clone_tree(field, b.device), 0, cost
        before = b.nfe
        field = b.field(ref, sigma, src, ids, guidance)
        cost = int(b.nfe - before)
        if self.enabled:
            self.source_fields[key] = (_clone_tree(field), cost)
        return field, cost, cost

    def capture_source(self, b, memory, ref, sigma, src, ids, guidance, node, job):
        """Reuse a complete capture's output and banks, never a native probe.

        Caller still performs memory.capture() then memory.off()/apply() in the
        original order. Current-state correction/target calls are never here.
        """
        eligible = (self.kv_enabled and self.scope is not None
                    and memory.mode == 'capture' and memory.variant == 'attn_budget'
                    and job.get('memory_pairing') == 'paired'
                    and job.get('memory_source_k') is True and job.get('memory_source_v') is True
                    and not memory.handles and not memory.inputs
                    and 0 <= job['src_start'] < job['src_steps'] <= 7)
        key = (int(node), float(sigma), float(guidance), memory.variant,
               tuple(job['attention_blocks']), int(memory.tokens))
        if eligible and key in self.captures:
            entry = self.captures[key]
            # Fresh clones isolate the cache from every target forward and close().
            memory.bank.update(_clone_tree(entry['bank']))
            memory.residual_bank.update(_clone_tree(entry['residual_bank']))
            self.stats['capture_hits'] += 1
            self.stats['capture_saved_nfe'] += entry['cost']
            return _clone_tree(entry['field'], b.device), 0, entry['cost']
        before = b.nfe
        field = b.field(ref, sigma, src, ids, guidance)
        cost = int(b.nfe - before)
        if eligible:
            self.stats['capture_misses'] += 1
            if set(memory.bank) != set(job['attention_blocks']):
                raise RuntimeError('Source capture did not populate every requested block')
            if memory.residual_bank or any(not isinstance(v, tuple) or len(v) != 2
                                           for v in memory.bank.values()):
                raise RuntimeError('R88 complete source K/V bank layout changed')
            size = _tree_bytes(memory.bank) + _tree_bytes(field)
            if self.bank_bytes + size <= self.max_bank_bytes:
                self.captures[key] = dict(bank=_clone_tree(memory.bank),
                                          residual_bank=_clone_tree(memory.residual_bank),
                                          field=_clone_tree(field), cost=cost)
                self.bank_bytes += size
            else:
                self.stats['capture_capacity_skips'] += 1
        return field, cost, 0
