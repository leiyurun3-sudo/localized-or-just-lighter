# Realized output points and placement share

From the extracted review ZIP, run the following commands in its root.

    python analysis/v21_operating_points/compare_points.py --points cpu/selectivity/operating_sources.csv --output rerun/points
    python analysis/v21_operating_points/compare_share.py --points cpu/selectivity/operating_sources.csv --output rerun/share

Both read the same archived feature-map LPIPS, directional CLIP and region records. Source IDs are strings. The single-point comparison uses no extrapolation, category-stratified resampling with 10,000 draws and seed 20261119 on each supported source set. Positive differences favor the point treatment. The editable-share calculation uses the integration and supported-source bootstrap from analysis/v20_mrc, with uniform minus spatial as in the main text. Edit share is a spatial summary and directional CLIP is an embedding score. These are post hoc comparisons. Each original seed-0 decoded output is used only at its recorded setting.
