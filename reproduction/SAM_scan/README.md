# Statistical reproduction

Extract all files together. From the extracted directory, run `python reproduce.py` with Python 3.10+ and NumPy 1.22+ installed. The measured reference environment used Python 3.12 and NumPy 2.4.4. Use `--out DIRECTORY` to choose an output directory. The calculation reads only `data/`; `reference/` contains archived comparison values.

| File | Contents |
|---|---|
| `data/scores.csv` | 900 job-level observations, all nine metrics. Read case IDs as strings to preserve leading zeros. Blank regional scores denote empty regions. |
| `data/sources.csv` | 50 PIE-Bench source IDs, category, generation seed, and edit/preservation mask pixel counts. |
| `data/protocol.json` | Frozen settings, defaults, dependencies, public upstream revision, dates, and protocol and control digests. |
| `data/core_steps.csv` | 26 active steps for each of 250 core projection jobs. Each row records the time interval, residual RMS, and model/visible displacement RMS before and after the BF16 state cast. |
| `reference/` | Reference summary, 150 source/pair overlap records and 50 sourcewise core records. |
| `control/` | The generation-time projection control and upstream attribution. The statistical entry point imports only NumPy and Python standard-library modules. |

## Reading the outputs

- `results/summary.json` contains counts, operating points, pairwise overlap, core contrasts, two-constant gain recovery, crossed effects, and profile checks. The four output CSVs give the corresponding source-level values.
- The 50 sources span 10 categories with five sources per category, one generation seed and one backbone. Each source has six core jobs and twelve further scan jobs. Three core defaults also serve as curve points. The curves use 750 distinct observations. Preservation metrics use 42 sources with nonempty preservation regions.
- MAE measures absolute RGB change from the source. Regional MAE uses the region pixel count as denominator. LPIPS uses AlexNet v0.1; regional LPIPS masks both RGB images to black outside the selected region. Directional CLIP uses the cosine between image and text changes in normalized ViT-B/32 embedding space. Target CLIP uses the edited image and the target prompt. Image CLIP uses the edited and source images.
- Core recovery divides the mean uniform-minus-two-constant preservation LPIPS by the mean uniform-minus-native preservation LPIPS on the same 42 sources. Both means use the same bootstrap draws. The core analysis uses 20,000 source resamples and a single RNG stream initialized with 20261118. `reference/core_per_source.csv` uses the same subtraction order for each metric's native and constant gain columns.
- The crossed four cells are native spatial, uniform on the native profile, spatial on the uniform profile, and default uniform. Each allocation effect is spatial minus uniform within the specified profile. The interaction subtracts the uniform-profile allocation effect from the native-profile allocation effect. The post hoc crossed analysis uses 20,000 source resamples with seed 1208 restarted for each metric. Its intervals and the original core intervals have separate RNG streams.
- Scan overlap uses directional CLIP as the response and preservation LPIPS as the outcome. The five points retain knob order. At each response the earliest crossing supplies the interpolated LPIPS. The integral uses the union of both paths' response breakpoints within common support. Each source's difference is divided by its positive overlap width, then averaged equally over valid sources. SU is uniform minus native, SF is FlowEdit minus native, UF is FlowEdit minus uniform. Missing, disjoint and touching curves are counted separately. Intervals use 10,000 source resamples, with seed 0 restarted per comparison in ascending source-ID order.
- Profile checks pair each replay with its same-source anchor on the same time grid. FP32 projection tolerance is `1e-6 + 1e-4 * target`; post-cast BF16 tolerance is `1e-6 + 0.01 * target`. A job passes a stage when both branches pass on every active step. The two constants use each branch's sum of native projected RMS divided by the sum of native residual RMS.
- The reported uncertainty represents source variation on this frozen panel. Support can vary by comparison. The core archive closed on 2026-11-18 UTC and the full scan on 2026-11-19 UTC.

