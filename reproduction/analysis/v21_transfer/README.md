# Transfer of measured attenuation

Run from this directory with Python 3 and NumPy:

    python -B analyze.py

This reads only source_inputs.csv and writes results.json and predictions.csv. It needs no model weights, images, network access, GPU or private paths. PIE source IDs identify matched observations. The optional extract_inputs.py reconstructs the compact input in an archive-equipped checkout, using Pillow and the archived evaluation tables. Pass the directory containing the original result ZIPs as its argument. No reconstruction is needed to reproduce the analysis.

## Input and target

Each row is one operator, source and seed. The 102 rows describe 82 distinct eligible images. There are 40 correction rows, 20 DirectEdit rows paired to a subset of the correction images, and 42 distinct SAM rows. We verified decoded RGB pixels, binary mask pixels, source and target prompt text and category for each of the 20 pairs. File-byte hashes can differ for identical decoded images.

All fits use $G=L_U-L_S$. Lower background error on the spatial arm makes the target positive for every operator. The primary target uses the AlexNet v0.1 spatial feature-map scorer weighted by the official background. Correction uses the original per-output score with the same metric definition and trunk/calibration hashes. The extracted 160 historical images with common-scorer re-evaluations reproduced their archived values within 1.5e−7. The complete original native spatial scores for correction come from the score archive. DirectEdit and SAM use the per-output spatial column. Only SAM has a secondary target in this input, the original RGB-masked scalar LPIPS used in the previous leave-one-source analysis.

| Operator | Arms in S, U order | Step measurements included in the magnitude columns | Source |
|---|---|---|---|
| Restoring correction | REQ, U25, both at 0.25 | Sum of h × actual_rms over 28 steps | R88_AnalysisClosure_Fast_results.zip, source_selection_manifest.json, evaluation.csv and each arm's step_summary.json |
| DirectEdit blend toward source | LOCAL_025, GLOBAL_025 | Sum of actual_delta_rms over 30 steps, with no second step-size factor | R89_DirectEdit_Results.zip, arm record.json and steps.json, plus DirectEdit_scores.csv |
| SAM projection away from source anchor | sam_native, uniform at default | Mean visible_rms over 26 active steps | Scan report ZIP, arm profile.json and sam_core_scores.csv |

The attenuation predictor is one minus the ratio of the spatial to uniform magnitude on their own trajectories. This is a realized, dimensionless arm comparison. The instantaneous same-state mask-energy factor and the area of a mask measure different quantities. Absolute RMS in the three editors is not pooled. DirectEdit's coefficient-one published local arm, both crossed replay arms, later floor runs, the two SAM constants and sweeps do not enter the regression. FlowEdit is a separate whole-editor scan without the required native gate counterfactual.

## Prediction design and uncertainty

The operator transfer trains intercept-only, area-only and area-plus-attenuation ordinary least squares on the 20 correction sources absent from DirectEdit. Coefficients are frozen and applied without recentering on the same 20 fresh images for correction and DirectEdit. There is no fitting on DirectEdit outcomes or on the correction outcomes of those test images. The training and test image sets are distinct. Lower paired RMSE or MAE is expressed as area-only error minus full-model error. The 5,000 refitted percentile bootstrap draws resample sources within observed categories in each split separately. The two test operators share draw indices. Inference is conditional on these operators and the observed category composition.

For SAM, one split withholds a source, the other its entire edit category. Predictions use training folds only. Pooled errors weight sources equally. The 5,000 category-cluster draws resample the ten observed categories, retain their source rows, and refit models with the test category and all its copies removed. These intervals reflect variation over the observed categories, including one category with one eligible image. The fixed predictors were taken from the earlier within-SAM analysis. Neither test-set outcomes nor category labels are predictors. The background annotation and both test-image traces must be available. The analysis is retrospective.

## Inventory and coverage

The inventory used the archived 98-package development index, the research result and delivery directories, local download/backup archives, and the SAM pilot and full-run trees. Completed numeric job reports, generation seeds, model identifiers and job-set inclusion were checked in all three SAM report archives. Archive and arm-level counts are in archive_inventory.json. Image ZIPs accompany the same reports. The 245-job and 300-job snapshots are subsets of the 900-job scan. All completed SAM jobs in these records use generation seed 0 on FLUX.1-dev. The internal reference-noise seed 999 is a sampler parameter. The 600 additional scan settings and cached scouts do not supply independent seeds.

The restoring R87 development run used seeds 0 and 17. R88 used 0 and 17, and R91 added 1, 2 and 3 on the correction cohort. R89 scale/map controls and R92 one-shot calibration act on the correction system. DirectEdit has 20 deterministic single-run cases at seed 0. Its local/global 0.25 pair is a controlled comparison on FLUX.1-dev. The archived SAM source tree includes SD3 code/configuration and the DirectEdit tree includes an SD3.5 runner. The inspected result archives contain no corresponding second-backbone spatial/uniform runs with compatible profiles, masks and scored outputs. The historical comparisons span different gate inputs, trajectory units, seed sets and sometimes different scorers. Regression results apply to the specified pairs and populations.
