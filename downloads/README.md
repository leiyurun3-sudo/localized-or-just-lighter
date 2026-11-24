# Experiment downloads

The original 900-output scan and the full seed 1 run are expanded here. The scan contains 900 native PNGs and its collection manifest. Seed 1 contains the 420 generated outputs, reports, projection traces and experiment manifest. The partial seed 2 ZIP preserves the 48 outputs from the 12 completed sources.

The LaTeX source ZIP, restoring-map inputs and SAM generation kit are included alongside the outputs. Ready-to-read PDFs and all 43 references are at the project root under `manuscript/` and `REFERENCES.md`. Recompute the manuscript statistics from the CSV files in `reproduction/`.

If a scorer expects a ZIP file, run the command below in this directory to repackage the expanded seed 1 folder with the original internal paths.

```bash
python -c "import shutil; shutil.make_archive('A_SAM_seed1_full', 'zip', 'A_SAM_seed1_full')"
```
