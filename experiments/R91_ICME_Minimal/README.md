# R91 ICME minimal evidence package

Uses the existing FLUX.1-dev model, offline. No model downloads, weight deletion,
environment replacement, or other-project process termination.

## Start and resume

```bash
cd /root/autodl-tmp
unzip -q R91_ICME_Minimal_Ready.zip
cd R91_ICME_Minimal
# Only if auto-discovery misses your existing model:
# export FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev
bash start.sh
```

`FLUX_MODEL_PATH` is the directory containing `model_index.json`, not a single
weight shard. Existing runtime versions: diffusers 0.35.1, transformers 4.55.4,
accelerate 1.10.1. Keep the existing torch/CUDA environment. Missing or mismatched
dependencies are reported before inference; nothing is silently installed.
Optional CPU analysis dependencies: pandas, scipy, statsmodels, matplotlib.

```bash
python status.py
watch -n 10 python status.py
tail -f "$(cat last_log.txt)"
touch runtime_base/PAUSE        # finish current image and pause
bash start.sh                   # resume, skips committed outputs
```

For a smaller first run: `bash start.sh --through seed1` executes four sentinels,
240 map/floor outputs and 160 seed-1 outputs (404 total). `--through maps` stops at
244. Run `bash start.sh` later to finish the unchanged plan. Stop points do not
alter planned comparisons. Do not edit the plan or code after generation starts.

## Work and outputs

Order: four historical U25/REQ-U reproduction images → random P + floor → seed 1
→ seed 2 → seed 3 → CPU evaluation → paired analysis → two ZIPs.

- Three new seeds × 40 sources × four cells: 480 new outputs.
- Two deterministic full 4096-cell permutations × 40 sources × old seeds 0/17:
  160 new outputs, exact P histograms and archived per-pair b_R.
- Floors 0 and .5 × 20 frozen sources × old seeds: 80 new outputs. The .25 floor
  and U25 are existing controls; floor uses archived b_U. Thus three floor points
  cover 120 observations, of which 80 require generation.
- Total: 720 scientific outputs plus four reproduction checks = 724.

The 20-source subset is fixed by category and identifier hash, independent of
outcomes. Model calls, memory, prompts, inversion, resolution and schedule follow
R88. New seeds get new native REQ/U25 anchors. Cell permutations are distinct
from R88/R90 block shuffles. Zero-norm floor directions stop with an explicit
error rather than inventing a fallback direction.

Existing models may differ numerically across GPU/software versions. Four
archived native images supply an initial numerical compatibility check. A failed
check stops before new scientific outputs; inspect `sentinel_check.json` and
`model_binding.json`. Passing checks is not proof of equality for every image.

`R91_results.zip`: configurations, native-image metrics, profiles, maps, previews,
planned contrasts, per-seed/sign summaries and mixed-model output.
`R91_native_images.zip`: every new full-resolution PNG and evaluation source.
Keep and return BOTH ZIPs. No latent/trajectory dumps. The package does not claim
to recover the 960 missing historical PNGs or recreate missing metadata.

Generation completion exports both ZIPs before evaluation. If evaluator weights
are missing, generation stays complete and status becomes `evaluation_pending`.
Set the existing local `CLIP_PATH`, `DINO_PATH`, `LPIPS_PATH`, and
`LPIPS_TRUNK_PATH`, then run `python run.py evaluate`. The script also checks the
original project's `evaluation_models/*paths.env` and shared Hugging Face caches.
Generation can be run alone with `python -u run.py generate`.

## Inference

The original two-seed result, independent three-new-seed replication, and pooled
five-seed sensitivity are reported separately. Seeds/draws are averaged within
source. Marginal intervals resample sources within category; all pairwise seed
sign-disagreement fractions are retained. An any-of-five-seeds sign-change rate
is not compared with the original two-seed rate.

Crossed source/seed random-intercept models fit paired allocation differences,
so variance components describe effect heterogeneity, not baseline image
difficulty. These are secondary model-based summaries with only five seed
levels; boundary and convergence warnings remain in output. Implementation:
[statsmodels variance components](https://www.statsmodels.org/stable/examples/notebooks/generated/variance_components.html).

Planned multiplicity families: two new-seed LPIPS contrasts; three permutation
LPIPS contrasts; their three CLIP counterparts; four floor LPIPS/CLIP contrasts.
MAE is descriptive. The primary endpoint remains preservation LPIPS. Random maps
retain target-derived histograms/anchors; recovery ratios are descriptive, not a
source-only/request-only decomposition. Nonsignificance is not equivalence.
Tested floor points are plotted without asserting a Pareto frontier. No new
results are inserted into the paper before execution and analysis.

Exact hours depend on the GPU/offloading regime; `status.py` estimates remaining
generation time after completed images. CPU evaluation time is separate.
