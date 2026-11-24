# Restoring-map diagnostics

## Anonymous CSV reproduction

Use Python 3.11 or later, NumPy and SciPy. Place `analyze.py` beside the following three anonymous CSVs. These four files suffice for every number in the paper insert.

| Input | Rows | Contents |
|---|---:|---|
| `observed_source_seed.csv` | 560 | Actual maps on 40 sources × two seeds × seven conditions. Mask AUROC, map TV, source saliency correlation, recorded initial-state RMS before and after gain, and archived background LPIPS. |
| `static_controls.csv` | 80 | Measured native maps, seven equal-TV orientations, mask-stratified shuffles and image-only saliency. |
| `native_step_summary.csv` | 80 | Native trajectories summarized from 28 logged steps with their recorded step sizes. |

Run the following command in the anonymous CSV directory:

~~~text
python -B analyze.py
~~~

The script looks for `inputs.npz` in `--input-dir`, which defaults to the directory containing the script. If the file is present, it measures the dense inputs. If it is absent, it reads the three CSVs. An explicit `--from-records` always selects those CSVs, even when the NPZ is present.

For separate input and output directories run the following command. The CSV path loads only those three tables. It validates the complete source/seed/arm grid and accepts rows in any order.

~~~text
python -B analyze.py --from-records --input-dir records --output-dir results
~~~

Outputs are `summary.csv` (paired estimates, medians, signs and percentile intervals), `conditional_associations.csv` (two adjusted slopes per model), and `threshold_ordering.csv` (one row per source). The `positive` column for the two `P4_ordering` rows counts 38 mixed-sign sources and 14 within-source ordering violations. A positive gain means lower background LPIPS for that map than uniform allocation at the spatial profile. Counts in other rows refer to the sign of the named quantity.

From the anonymous reproduction root, the verification command is `python -B analysis/v21_map/analyze.py --from-records --output-dir new/map`. The automatic CSV path and this explicit invocation produce the same three result tables as the dense run.

## From the compact author-held inputs

Place `inputs.npz` beside `analyze.py` and run:

~~~text
python -B analyze.py
~~~

The default mode computes all three input CSVs before invoking the same summary routines. The NPZ holds the observed native FP32 protection fields, lossless ranks for six other archived fields sharing their histogram, per-cell counts of official preservation pixels, 8×8-cell RGB sums, four raw scalars per native solver step, three first-step norm scalars per arm, and archived losses. Grid-level inputs reproduce the defined calculations. The archive inventory and benchmark/source digests are optional provenance files.

To rebuild the NPZ from the original archives, use NumPy and Pillow. Archive, image and score paths are supplied locally:

~~~text
python -B extract.py --archives ARCHIVES --images R88_SOURCE_EVAL_DIRECTORY --scores R91_COMBINED_SCORES_CSV
python -B inventory.py --archives ARCHIVES
~~~

`extract.py` reads actual maps from `request_maps.npz:P_correction` and steps from `step_summary.json` inside the matching job directories. It checks the complete 28-step applied-RMS profile for all comparison arms and the uniform reference against native spatial. Archive lookup uses the original job basename with batch, public PIE-Bench source, seed and arm. `sources.csv` contains only ordinal aliases, SHA-256 of “PIE/” followed by the public zero-padded source ID, and input PNG digests. The aliases follow sorted source IDs. No local paths are required by the CSV reproduction.

| Archive or input | Use |
|---|---|
| R88_AnalysisClosure_Fast_results.zip | Native request (REQ), first histogram-matched wrong prompt (SHUF_R), uniform reference at the native spatial profile (GLB_R), 28 native step records. |
| R89_AnalysisEvidenceClosure_results.zip | Second wrong prompt (SHUF2_R), source-ranked map (SOURCE_R), histogram-matched reversal (REV_HIST_R), and official_preserve.png from outputs/r88_native/masks. |
| R91_results.zip | Two independent archived cell permutations (RAND_A, RAND_B). Every map has exactly the native sorted FP32 entries for its source/seed. |
| Recovered R88 source-evaluation PNGs | 512×512 RGB input, reduced by exact integer sums into 64×64 cells for the contrast diagnostic. |
| R91 analysis/combined_scores.csv | Original per-output lpips_preserved, joined by public case, seed and arm. The shared-S uniform score is reused within that source/seed. |

## Quantities and controls

**Spatial alignment and smoothness.** High protection $P$ should rank preservation pixels above edit pixels. Equivalently the release field $1-P$ ranks edit pixels. Each map cell broadcasts to its own 8×8 pixels. We use the archived official preservation mask and exclude the outermost one-pixel image frame as in the historical alignment analysis. Weighted AUROC gives half credit for ties and retains partially covered cells. TV is the mean absolute difference across all horizontal and vertical grid edges, with no wraparound. The seven nonidentity square-grid symmetries preserve the histogram, grid-edge TV and distance to the image center. They change placement relative to the same fixed image and evaluation mask. The complementary control permutes values within each distinct pair of edit/preservation pixel counts, including edge cells. This preserves the full score distribution in both evaluation regions and thus AUROC exactly. We average 16 within-stratum permutations per source/seed (NumPy seed sequence [2718, zero-based source ordinal, generation seed]). These transformed fields enter map diagnostics only. The RAND arms in the outcome analysis are separately archived editing runs.

**Independent appearance reference.** The source-only score is computed before reference to the evaluation mask. Average 8×8 sRGB pixels per cell in [0,1]. Form Gaussian-smoothed colour fields with standard deviations 1, 4 and 12 cells and reflected image borders. Compute channel RMS differences for scale pairs (1,4) and (1,12) and average the two contrast magnitudes. This fixed nonlearned center–surround score highlights local colour/brightness contrasts, including texture and background boundaries. It has no prompt, object model, trained parameters or semantic target. Its AUROC uses edit pixels as positives. Release-to-saliency correlations are Spearman correlations on the 4,096 cells. The archived SOURCE_R condition uses a different source-derived field.

**Fixed state and energy.** The archived engine initializes every arm from the first inversion state before the first solver update. At step zero the seven fields have the same measured unweighted correction RMS for each source/seed. The numerator is candidate_global_rms before replay gain and the denominator is current_correction_global_rms. Squaring their ratio gives $q$. The applied post-gain RMS is recorded separately. An unrestricted random permutation has expected $q_0=\mathrm{mean}(P^2)$ at a fixed correction tensor. The two archived cell permutations give an empirical check. In the 28-step extension, each native value is compared with its histogram baseline on its own current state. We average differences with the recorded solver-step sizes normalized to sum to one, then average the two seeds. The post-initial states of different arms evolve separately.

**Outcome analysis.** The actual outcome records cover seven archived conditions. Average seeds within source and average the two cell-permutation draws to form a single family. The six-family analysis uses native, two distinct wrong-prompt families, source-rank, reversal and cell permutation. Each source contributes six family means. The four-family sensitivity omits reversal and cell permutation. Regress shared-profile background-LPIPS gain on AUROC and TV after removing source and family means from all three variables. Slopes refer to a 0.10 increase in AUROC and a 0.01 increase in TV. A positive slope associates a larger gain with a higher predictor. The fitted slopes are descriptive associations across the archived interventions. `within_source_spearman_auc_gain` instead averages within-source rank correlations across map families. For an ordering violation the same source must have one strictly positive gain at a lower AUROC and one strictly negative gain at a higher AUROC. All uncertainty intervals use 10,000 equal-weight paired source draws, NumPy seed 0, marginal percentile endpoints. The fixed-effects fit recenters and refits for every draw.

**Evidence boundary.** Across the four main result archives, the 1,722 NPZ files hold maps and, in R89, a three-probe mean source-velocity RMS field. The 1,924 step logs store scalar measurements. The inventory inspects numerical payload names in every ZIP and NPZ and first-step field names in one log per arm. These archives provide no per-step correction tensor with spatial and channel axes. The logged roi_conditional_rms uses the internal soft routing field. The probe average describes source velocities. Neither supplies the official edit/background correction RMS or regional cross-terms in the full update. Those quantities are left unreported. P4 here concerns archived background preservation at a fixed intervention profile.
