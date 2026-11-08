# VTaC data attribution

The processed real-data cache, event metadata, and waveform figures in `real_smoke/` are derived from VTaC v1.1. The raw full dataset is not bundled. These data derivatives retain the source Creative Commons Attribution-ShareAlike 4.0 license; see `DATA_LICENSE.txt`.

Source: https://physionet.org/content/vtac/1.1/
DOI: https://doi.org/10.13026/jg2e-tf67

Lehman, L., Moody, B., McCullum, L., Saeed, H., Deep, H., Perry, D., Struja, T., Li, Q., Clifford, G., and Mark, R. (2026). VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors (version 1.1). PhysioNet.

Original study: Lehman et al. (2023), VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors, NeurIPS Datasets and Benchmarks.

Changes: selected a small software-verification subset, cropped to 30 seconds before alarm onset, filtered, interpolated permitted short gaps, normalized, and calculated feature/quality summaries. The subset is not an unbiased evaluation cohort. No endorsement by the original authors is implied.
