"""R86 fixed-profile contract; independent of model weights and CUDA."""
import math

MODE = 'r86_fixed_profile'
VERSION = 'r88-dual-anchor-profile-v1'


def expected_anchor_arm(job):
    return 'U25' if job.get('arm') == 'REQ_U' else 'REQ'


def validate_r88_job(job):
    """Native anchors and four replay allocations; no sampler parameters are rewritten."""
    arm = job.get('arm')
    if arm not in ('REQ', 'GLB_R', 'U25', 'REQ_U', 'SHUF_R', 'REV_R'):
        return
    if float(job.get('corr_lambda', 0)) != 0.25 or job.get('corr_policy', 'constant') != 'constant':
        raise ValueError('R88 requires constant corr_lambda=0.25')
    if (job.get('memory_response_mode') is not None or job.get('query_selection') != 'all_image_queries'
            or not job.get('global_update')):
        raise ValueError('R88 requires full update support and fixed all-query memory')
    if (job.get('mode') != 'attn_budget' or job.get('gate_mode') != 'uniform'
            or not job.get('memory_source_k') or not job.get('memory_source_v')
            or job.get('src_start') != 0 or job.get('src_steps') != 7
            or int(job.get('steps', 28)) != 28):
        raise ValueError('R88 keeps the 28-step schedule and original uniform source-KV window [0,7)')
    map_mode = job.get('r88_map_mode')
    expected_map = {'SHUF_R': 'wrong_prompt_histmatch', 'REV_R': 'reverse'}.get(arm)
    if map_mode != expected_map:
        raise ValueError('R88 arm/map mode mismatch')
    prompt = job.get('probe_target_prompt', job['target_prompt'])
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('probe_target_prompt must be a nonempty string')
    if (prompt != job['target_prompt']) != (arm == 'SHUF_R'):
        raise ValueError('Only SHUF_R uses a distinct frozen probe_target_prompt')
    if arm in ('REQ', 'U25'):
        if job.get('r86_arm') or job.get('r86_anchor') or job.get('budget_reference_id'):
            raise ValueError('Native R88 anchors cannot depend on a recorded profile')
        if arm == 'REQ':
            if job.get('correction_shape') != 'roi_uniform' or job.get('correction_budget_mode') != 'local_coefficient':
                raise ValueError('REQ must use native local_coefficient correction')
            if not job.get('route_mask') or job.get('route_mask_source') != 'prompt_response_reference' or float(job.get('route_floor', -1)) != 0.25:
                raise ValueError('REQ requires the native request map with floor=0.25')
        elif (job.get('correction_shape', 'uniform') != 'uniform'
              or job.get('correction_budget_mode', 'none') != 'none'
              or job.get('route_mask') or job.get('route_mask_source')):
            raise ValueError('U25 must use native uniform correction without E/P probes')
    else:
        expected = 'GLOBAL' if arm == 'GLB_R' else 'CORR'
        if job.get('r86_arm') != expected or job.get('correction_budget_mode') != MODE or float(job.get('r86_beta', 0)) != 1.0:
            raise ValueError('R88 replay allocation/beta mismatch')


def validate_profile(job, profile, sigmas):
    validate_r88_job(job)
    if job.get('correction_budget_mode') != MODE:
        raise ValueError('R86 requires correction_budget_mode=' + MODE)
    if job.get('r86_arm') not in ('CORR', 'GLOBAL', 'SHIFT', 'PERM'):
        raise ValueError('Unknown R86 allocation')
    beta = float(job.get('r86_beta', 0))
    if not math.isfinite(beta) or beta not in (0.7, 1.0):
        raise ValueError('R86 beta must be 0.7 or 1.0')
    if (job.get('correction_shape') != 'roi_uniform'
            or job.get('route_mask_source') != 'prompt_response_reference'
            or not job.get('route_mask') or not job.get('global_update')
            or float(job.get('corr_lambda', 0)) != 0.25
            or float(job.get('route_floor', -1)) != 0.25
            or job.get('budget_reference_id') or job.get('region_gain_cap') is not None):
        raise ValueError('R86 fixed-profile allocation configuration mismatch')
    if not job.get('r86_anchor'):
        raise ValueError('Missing R86 archived anchor identifier')
    if profile is None or len(profile) != len(sigmas) - 1:
        raise ValueError('Missing or incomplete R86 frozen step profile')
    for i, row in enumerate(profile):
        sigma, h = float(sigmas[i]), float(sigmas[i] - sigmas[i+1])
        if row.get('step_index') != i:
            raise ValueError('R86 profile step index mismatch at ' + str(i))
        for key in ('sigma', 'h', 'applied_correction_rms'):
            if not math.isfinite(float(row[key])):
                raise ValueError('Nonfinite R86 profile ' + key)
        if h <= 0 or float(row['applied_correction_rms']) <= 0:
            raise ValueError('Invalid R86 profile step size or RMS')
        if abs(float(row['sigma']) - sigma) > 1e-8 or abs(float(row['h']) - h) > 1e-8:
            raise ValueError('R86 frozen profile schedule mismatch at ' + str(i))


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('R86 source-version mismatch: ' + old[:100])
    return text.replace(old, new)


def fix_engine(text):
    text = once(text, 'from controls import pack',
                'from controls import pack\nfrom r86_compat import validate_profile')
    text = once(text, "\n    if request_global:\n", """
    r86_fixed = budget_mode == 'r86_fixed_profile'
    if bool(job.get('r86_arm')) != r86_fixed:
        raise ValueError('R86 arm and fixed-profile mode must be specified together')
    if r86_fixed:
        validate_profile(job, budget_profile, sigmas)
        if not bool((mask == 1).all()) or not bool((allowed > 0).all()):
            raise ValueError('R86 requires full update support')
    if request_global:
""")
    text = once(text, "budget_mode not in ('external_u25', 'online_same_state', 'local_coefficient')",
                "budget_mode not in ('external_u25', 'online_same_state', 'local_coefficient', 'r86_fixed_profile')")
    text = once(text, '            elif shaped:\n', '            elif shaped and not r86_fixed:\n')
    text = once(text, "    effective_route = None if route_mask64 is None else floor + (1.0-floor)*route_mask64",
                "    if r86_fixed:\n        control_kind = 'actual_profile_' + job['r86_arm'].lower()\n    effective_route = None if route_mask64 is None else floor + (1.0-floor)*route_mask64")
    text = once(text, "                weighted = candidate * gain\n", "                weighted = candidate * gain\n                desired_rms = r86_target\n")
    text = once(text, "matched_step_budget_rms=desired_rms if budget_mode == 'external_u25' else None",
                "matched_step_budget_rms=desired_rms if budget_mode in ('external_u25', 'r86_fixed_profile') else None")
    # Both the per-step and run-level diagnostics reflect the actual profile dependency.
    text = text.replace("external_trajectory_budget_matched=shaped and budget_mode=='external_u25'",
                        "external_trajectory_budget_matched=shaped and budget_mode in ('external_u25', 'r86_fixed_profile')")
    text = once(text, "correction_budget_status=('same_state_local_release_energy_control' if request_global",
                "correction_budget_status=(('fixed_profile_matched' if relative <= 1e-5 else 'budget_mismatch') if r86_fixed\n            else 'same_state_local_release_energy_control' if request_global")
    text = once(text, "(row['region_gain'] if request_global else lam)",
                "(row['region_gain'] if request_global or r86_fixed else lam)")
    return text


def fix_runner(text):
    text = once(text, '    from engine import execute',
                '    from engine import execute\n    from r86_compat import validate_profile')
    text = once(text, '                budget_reference_nfe = 97\n',
                "                budget_reference_nfe = int(anchor_record['nfe_standalone'])\n                validate_profile(job, budget_profile, sigmas)\n")
    text = once(text, 'budget_reference_reused_in_batch=external_matched,',
                "budget_reference_reused_in_batch=external_matched or bool(job.get('r86_arm')),")
    text = once(text, "            traceback.clear_frames(exc.__traceback__)\n", """            # Keep the error and partial results, then stop this batch immediately.
            progress.update(status='failed', stage='stopped on first generation error',
                            completed=len(done), failed=1, error=str(exc))
            raise RuntimeError('R86 stopped at '+job['id']+'; inspect saved error.json before resume') from exc
""")
    return text
