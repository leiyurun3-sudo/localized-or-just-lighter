# Observed-grid interpolation check

Copy this directory as a unit. Python 3 and NumPy are sufficient. Run `python -B grid_check.py --output rerun` with `observations.csv` next to the script. The six reference outputs are listed below. IDs are public source IDs; read them as strings.

## Observations and scope

The input is a field selection from the SAM rows of `analysis/v20_mrc/observations.csv`: all 50 sources, one generation seed (0), three arms, five settings, 750 measured outputs. S and U are spatial and uniform SAM at scales 0.25, 0.5, 0.75, 1 and 1.25. F is FlowEdit at window limits 12, 16, 20, 24 and 28. The outcome is the full-image AlexNet-v0.1 feature-map LPIPS averaged on background pixels; eight full edit masks have a missing background outcome. Directional CLIP is available for all 50 sources. The principal accuracy summaries use the same 42 background-eligible sources for both coordinates. A separate CLIP-only summary includes all 50.

The partial, core-complete and scan-complete DoseLens report archives share the same jobs manifest (SHA-256 `8140c8bc92b21a7f8db6953adc67750e520dc0be93865a91f788d2b4967e8d26`) and prompt-token manifest (`a3f7bfcab31860e5b9c66c12269bde4338e07629cffdf840fa176037e8ecd4cf`). The 245 partial and 300 core output IDs are subsets of the 900-output archive. The pilot manifest lists these same sweep settings; the Scan600 delivery archive contains its launch script. The 150 scalar/replay controls have different allocation or magnitude rules. No additional same-condition intermediate setting was located in these releases. The check reuses the five measured points and measures interpolation error at their interior settings. Testing newly generated intermediate settings would be a separate prospective experiment.

## Holdout estimands

Remove each interior rank (2, 3, 4) separately from one source and arm. The output at the omitted rank supplies the observed target. For the knob-space test, use its two surviving adjacent knob values and interpolate both directional CLIP and LPIPS linearly in knob space. These grids are evenly spaced, so each predicted coordinate is the arithmetic mean of its two neighbors. For the response-space test, use the omitted output's measured CLIP as the query and predict only LPIPS from the four retained points. Connect them in original knob order. A query outside their closed response range is unsupported and has a blank prediction. Within the range use the first segment reaching the response; the last is reported as a sensitivity. Reverse traversal of a flat segment selects its other endpoint. A local fold is a strict direction reversal across the omitted rank. A folded query can still be supported by a different retained segment. At such a query, the prediction error against the held image also reflects branch selection. The input contains no within-curve repeated response nodes (checked before delivery).

Signed residuals are predicted minus observed. Source-level MAE averages the absolute residuals available on that source and arm. The cohort mean weights contributing sources equally, including sources with fewer supported response queries. The interval is a 10,000-draw percentile source bootstrap, with NumPy seed 0 restarted for each estimate on ascending source IDs. Holdouts on the same source are not independent replicates.

## Grid resolution and support

The coarse grid retains ranks 1, 3 and 5 (SAM 0.25, 0.75, 1.25; FlowEdit 12, 20, 28). Grid-deletion diagnostics omit one interior rank from both arms together, or from only one arm. The pair codes SU, SF and UF denote U minus S, F minus S and F minus U. Evaluate each grid on the positive-width intersection of the two retained CLIP ranges within source. Partition it at all response knots and integrate the first-crossing LPIPS difference exactly by summing midpoint values times subinterval widths, then divide by the width. On a fold the last-crossing value is also recorded. Folded width sums the subinterval lengths on which either arm crosses more than once. Point contact and disjoint ranges supply zero width and no integrated estimate; no held or pairwise response is extrapolated.

The cohort contrast averages supported sources equally. `own_interval_shift` is reduced minus five-grid MRC on the same sources, each grid on its own interval. `fixed_interval_shift` integrates both grids over the narrower, reduced-grid interval on those sources. Differences are bootstrapped as paired source differences. `narrower` counts all eligible sources with a width reduction above numerical tolerance, including any loss of support. For F minus U, five and three ranks have 41 and 40 supported sources; the paired shift is computed on the common 40. Bootstrap intervals describe variation over these sources. A single source has a deterministic deletion range, without an individual confidence interval.

## Output files

| File | Rows and contents |
|---|---|
| `held_points.csv` | 450 actual holdouts, observed and predicted responses and outcomes, signed errors, retained support and crossing segment endpoints. |
| `held_source_errors.csv` | Mean absolute and signed errors within each source, with the number of scored holdouts. |
| `held_summary.csv` | Equal-source mean errors, 95% intervals and counts of attempts, scored holdouts and folds by arm. |
| `grid_sources.csv` | Per-source intervals, widths, first/last-crossing contrasts, folded width and full-grid contrast on each restricted interval. |
| `grid_summary.csv` | Coverage, signs, source means and paired shifts for all three pairs and all eleven grid variants. |
| `counterexamples.csv` | All four negative U minus S sources from the five-point analysis, including the jar. Gives widths and first/last contrasts on five, three and three jointly reduced grids, and sign counts for the six single-arm deletions. |
