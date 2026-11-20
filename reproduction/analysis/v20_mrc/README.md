# Matched-response sensitivity analysis

Keep this directory as a unit, for example `analysis/v20_mrc/` inside an anonymous evidence package. The portable entry point is `reproduce_mrc.py`. Its only data input is the adjacent `observations.csv`. The reference outputs are `summary.csv`, `per_source.csv` and `historical_inverse.csv`. The LaTeX insertion is outside this directory at `paper/v20_mrc.tex`.

Run `python -B reproduce_mrc.py --output rerun` from this directory with Python 3 and NumPy. The output option creates a separate results directory. All paths used by the entry point are relative to the script or supplied by the caller. There are no model, image, archive, TeX or network dependencies.

## Observations

The 750 SAM observations cover 50 source IDs, three arms and five knob settings, all at seed 0. S and U use scales 0.25, 0.5, 0.75, 1 and 1.25. F uses 12, 16, 20, 24 and 28. Eight sources have no background and have a blank feature-map LPIPS at every setting. The 240 restoring observations cover 40 source IDs, request (S) and uniform (U) directions on the uniform-origin profile, at scales 0.7, 1 and 1.3. Each restoring point already averages the two matched seeds 0 and 17, representing 480 outputs. Both operators measure full-image AlexNet-v0.1 feature-map LPIPS averaged over background pixels. Positive U−S, F−S and F−U differences favor S, S and U, respectively. Edit MAE is within-mask mean absolute RGB change from the source, on the 0–1 intensity scale.

The SAM response and edit MAE come from `analysis/curves/scan_points.csv`, joined by job ID to `spatial` in `analysis/harmonized_lpips/sam_900_scores.csv`. The source, arm, category, directional CLIP and background eligibility agree at the join. For restoring, the source is `analysis/history/supp_r89_seed_mean_points.csv`. Source IDs are public benchmark identifiers and must be read as strings. The two SAM flags are the previously identified discrepancies between dataset instruction and generation prompt (611000000001, raven/otter; 812000000002, girl/house ordering).

## Estimands and outputs

`summary.csv` contains 36 forward estimates or support diagnostics across the SAM and restoring cohorts. The forward outcome is always background feature-map LPIPS. Each pair code gives the subtracted arm first and the minuend second. The primary estimate averages sources equally. All three-arm restrictions apply within SAM. `per_source.csv` records the source ID, membership, support endpoints and integration width for each analysis; for source-first analyses it also gives the signed difference and aggregation weight. For a disjoint interval the lower bound exceeds the upper bound and the width is zero. Blank differences on aggregate-first rows are intentional because that estimator yields a cohort contrast. `historical_inverse.csv` reproduces the archived, separate endpoint with directional CLIP as the outcome and feature-map LPIPS as the matching coordinate.

For each source, connect outcomes linearly in ascending setting order. Intersect the observed response ranges, partition at all response knots, and integrate the signed difference using the midpoint of each linear subinterval. Repeated and vertical knots have zero integration width. At a fold, first and last select the earliest and latest segment in setting order covering the midpoint. Touching or disjoint ranges have no source estimate. No extrapolation is used. The source contrast divides the signed integral by the positive overlap width.

| Variant | Population and estimator |
|---|---|
| primary | First directional-CLIP crossing; pair-specific source set and overlap; equal source weights. |
| last_crossing | Same ranges and source set; traverse each curve in reverse setting order. |
| width_weighted | Same first-crossing source differences. Divide their summed signed areas by their summed widths. |
| exclude_ambiguous | Exclude the two flagged SAM IDs before the primary calculation. |
| common_sources | Restrict each comparison to the intersection of all three pair-specific sets; use each pair's own overlap. |
| common_interval | On those same sources integrate each pair on the single per-source interval common to all three arms. |
| common_interval_width_weighted | Use the three-arm interval and weight each source by its width. |
| edit_mae | Use edit-region RGB MAE as the horizontal coordinate, retaining feature-map LPIPS as the outcome. First crossing, own positive-width source set, equal source weights. |
| cohort_curve_pair_sources | Average coordinates at each setting over the primary pair's supported sources, then intersect and integrate the two mean curves. |
| cohort_curve_eligible | Same aggregate-first comparison over every background-eligible source. |

The interval columns give marginal percentile 95% intervals. Each standalone estimate uses 10,000 paired-source draws of its own contributing sources. Cases are in ascending ID order and NumPy's default RNG restarts at seed 0. A draw carries each source's arms and settings together. Source-first analyses keep the observed source-level integration results and resample whole sources. Weighted ratios recompute both sums within each draw. Cohort-curve resamples recompute all setting means and then the support, crossing and integral of the resampled mean curves. When any bootstrap draw lacks a positive-width mean-curve overlap, the interval is left blank and the number of finite draws is reported. A disjoint point-estimate support also leaves the mean blank.

`vs_primary` is variant minus primary, each estimated on its own source set. Its interval draws the union of contributing IDs once per replicate, restricting the shared draw to each estimator's members before renormalizing. This preserves pairing and includes changes in membership. `vs_primary_on_shared_sources` recomputes both on their source intersection, with identical IDs and draws. For a restriction that changes membership alone, the latter contrast is zero by construction. Sample counts, sign counts, median individual widths and the endpoints of the cohort-curve interval are in separate columns. An estimate using matched edit MAE still has background-LPIPS units; its support width is in MAE units.

The inverse historical comparison has the original category-stratified 20,000 source resamples, seed 20261106. Each draw rebuilds the three mean points per arm, requires the archived monotonic LPIPS ordering and integrates uniform minus request directional CLIP over the overlap in background LPIPS.

### Integration handoff

For an evidence packet, copy `reproduce_mrc.py`, `observations.csv`, `summary.csv`, `per_source.csv`, `historical_inverse.csv` and this README together. For document integration use `paper/v20_mrc.tex`. Figures and prior-revision result archives are unnecessary for the rerun.
