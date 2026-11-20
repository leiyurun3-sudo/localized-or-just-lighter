A-R92: one-shot scalar calibration for the A analysis paper

80 new outputs: 40 original sources x seeds 0 and 17. No new baseline runs.
Each pair uses lambda_SC = 0.25 * I_U / I_R from its own archived native runs.
It remains fixed throughout generation. No stepwise normalization or tuning.

AutoDL: unzip under /root/autodl-tmp, then bash START_AND_MONITOR.sh.
The model is reused locally. Background generation continues if the monitor or
browser closes. The monitor refreshes every 3 seconds and imports no model.
Rerun the same command to resume. Do not modify runtime files after starting.
To pause after the current image: touch runtime_base/PAUSE

Download A_R92_results.zip after completion. It includes final native PNGs,
previews, small step records, original anchors/scores, and paired analysis.
No latent trajectories or duplicate full-image crops are included in the ZIP.

If LPIPS/CLIP dependencies or offline weights are unavailable, generation is
retained and status becomes evaluation_pending. Download the same result ZIP;
local CPU evaluation can finish without regenerating any image. Model overrides:
LPIPS_TRUNK_PATH, LPIPS_CALIBRATION_PATH, CLIP_MODEL_PATH.

Generation and CPU scoring run in separate processes to release FLUX memory.
The exact existing R91 inference branch is reused with a changed fixed scalar;
model precision, scheduler, reference construction, memory and maps stay fixed.
Generation model versions: diffusers 0.35.1, transformers 4.55.4, accelerate 1.10.1.

protocol.json fixes comparisons, multiplicity and the stopping point before data.
Source, not output/seed, is the inference unit. Native REQ edit MAE is unavailable
and is not reconstructed from previews. A zero-containing interval is not proof
of equivalence. Record both outcome differences and actual magnitude deviations.

This package has not been executed on your AutoDL GPU at delivery.
