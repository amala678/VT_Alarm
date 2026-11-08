# Reducing False Ventricular Tachycardia Alarms

Implementation of the attached CS-GY 6923 proposal: compare ECG with ECG + PPG on identical VTaC alarm events. True VT is the positive class.

## Start here

Use Python 3.10+ (3.11 or 3.12 recommended). Open a terminal inside this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m vt_alarm.cli demo
```

On Windows, activate with `.venv\Scripts\activate` instead. For Google Colab, unzip the project, change into its folder with `%cd`, install with `!pip install -e .`, and prefix the commands below with `!`.

The demo runs all model branches with three seeds on generated toy signals. Its report is `demo_run/models/evaluation/REPORT.md`. The included `demo_run` is an already executed demo. To rerun, choose a new path, for example `demo --out demo_run_2`. Synthetic signals are deliberately simple and do not establish physiological validity or model effectiveness.

## Run the real study

Dataset: https://physionet.org/content/vtac/1.1/

The downloader retrieves the two official metadata CSVs, license, and **annotated** waveforms only. It skips the additional unlabeled corpus. It resumes existing files and writes downloads atomically. Read and comply with the dataset license, and cite the original VTaC paper and release. Leave the official CSVs unchanged.

```bash
python -m vt_alarm.cli download --data data/vtac
python -m vt_alarm.cli prepare --data data/vtac --cache cache/vtac --seconds 30
python -m vt_alarm.cli train --cache cache/vtac --out runs/main --epochs 40 --seeds 11 22 33
```

Before running the next command, inspect the training/validation output and freeze any justified revisions. Record a revision's observed validation failure, explanation, single change, and outcome in `ITERATION_LOG.md`. Test results must not drive revisions.

```bash
python -m vt_alarm.cli evaluate --cache cache/vtac --out runs/main --bootstrap 1000
```

For CUDA, add `--device cuda` to train/evaluate. This has been exercised on CPU; GPU determinism depends on the installed stack. To reduce compute while checking setup, use `train --quick --epochs 3 --seeds 11 --out runs/smoke`. A quick run is not the completed study. Prepared cache folders are immutable: use a fresh `--cache` directory for each preprocessing revision. `download --limit-per-split 4` loads a few real files to inspect I/O; that subset may lack a class and is not intended for model comparison.

If you already downloaded VTaC, `--data` should point to the directory containing:

```text
event_labels.csv
benchmark_data_split.csv
annotated_waveforms/
  RECORD/
    EVENT.hea
    EVENT.dat
```

Actual data filenames are read from each WFDB header. The official split column is recognized by its textual train/validation/test values; `--split-column NAME` resolves ambiguous metadata. Unrecognized numeric split codes fail explicitly rather than guessing a mapping.

## What is implemented

- Official record-level allocation; no random event-level resplit.
- Label join on record/event, duplicate key checks, cross-split record checks.
- Fixed ECG priority: II, I, V, V1, III, aVF, aVL, aVR, V2–V6.
- PPG/PLETH detection; shared paired cohort; separate PPG-missing ECG test analysis.
- Final 30 pre-alarm seconds at 250 Hz; 10/60 second windows supported.
- Crop before filtering. Every read ends at sample 75,000 (300 seconds), exclusively.
- Reject >5% non-finite samples, any gap >200 ms, constant signals, and inadequate windows; interpolate permitted short gaps inside the pre-alarm window.
- ECG .5–40 Hz and PPG .3–8 Hz third-order Butterworth filters; per-window median/MAD normalization; clipping normalized inputs to ±12.
- Features: approximate peak rate, RR variation, RMSSD, amplitude, derivative variability, spectral ratios, missing fraction, longest flatline, clipping proxy, template consistency.
- Train-only standardized, class-balanced logistic regression; C tuned on validation AUPRC.
- Compact 1-D CNNs with 16/32/64 channels, batch normalization, pooling, dropout, and binary logits. Fusion and shuffled models have identical parameter counts; ECG differs only in the first convolution's input channels.
- Same CNN optimization protocol; train-derived positive class weighting, validation AUPRC early stopping, limited learning-rate/weight-decay search on the first seed; winning configuration repeated at three seeds. All trials retained.
- Fixed cross-record PPG permutations separately in training, validation, and test; no label conditioning; event-to-donor maps saved. A mathematical impossibility (one record has >half the events) fails explicitly.
- Each model receives a threshold maximizing validation specificity with sensitivity ≥.95, lower threshold for ties. `--target .99` supports a separately predeclared secondary analysis.
- Test AUROC, AUPRC, sensitivity, specificity, precision, F1, TP/TN/FP/FN; 1,000 record-clustered paired bootstrap resamples by default with fixed thresholds.
- Overall, low/high ECG quality, low/high PPG quality, and joint ECG×PPG quality results. Cutoffs are training medians. Quality is a proxy, not a clinical label.
- Reproducible model weights, validation predictions, training curves, cohort audit, provenance hashes, test predictions, figures, and Markdown report.

## Files and results

| File | Purpose |
| --- | --- |
| `vt_alarm/data.py` | Download, official metadata join, WFDB crop, cohort audit, synthetic demo |
| `vt_alarm/signals.py` | Filtering, gap handling, normalization, features and quality |
| `vt_alarm/models.py` | CNN, weighted training, early stopping, reproducible seeds |
| `vt_alarm/experiment.py` | Tuning, repeat seeds, freeze/checkpoint checks, held-out evaluation |
| `vt_alarm/evaluation.py` | Thresholds, metrics, cross-record shuffle, paired cluster bootstrap |
| `vt_alarm/report.py` | Figures and report generation |
| `tests/test_integrity.py` | Leakage, threshold, gap, shuffle and bootstrap checks |
| `RUN_PROJECT.ipynb` | Guided notebook launcher for the same package |

`cache/vtac/audit.csv` records every event and exclusion reason. `channel_counts.json` reports channels among readable headers. `paired.csv` defines the final cohort. Official versus eligible counts must be reported: a partial download is recorded as missing files, not silently relabeled as the full study.

`runs/main/frozen.json` stores selected settings, train-only quality cutoffs, versions, model sizes, hashes, seeds, and thresholds. All seed metrics appear in `evaluation/metrics.csv`. Bootstrap JSONs include valid replicate counts and excluded replicates for each metric. Replicates missing a required class have undefined metrics and are excluded only for those metrics. Precision/F1 use zero when their denominators vanish.

`REPORT.md` is generated from actual predictions and includes overall metrics, paired sensitivity/specificity differences, graphs, and error examples. Subgroup confidence intervals live in the accompanying JSONs. Do not claim benefit from specificity alone if achieved test sensitivities differ. Evaluate small subgroups cautiously. No automatic post-test pooling is performed.

## Scope and limits

The real-data loader and small real-file smoke check are distinct from completing the full study. See `VALIDATION.md` for what was actually run in the delivered package. The full VTaC training run must produce its own results before a final scientific conclusion can be written. Never submit synthetic demo metrics as VTaC findings.

PPG that exists but fails validity checks is logged as invalid PPG; it is not called missing. Only absent PPG channels enter the separate PPG-missing analysis. No imputed PPG or quality fallback is used in the main comparison. Feature peak detection is a simple baseline and can fail in artifacts/VT. Quality scores may reflect physiology; their validity needs visual inspection using training data before freezing the study. Duplicate detection uses exact cropped paired waveform bytes and does not establish absence of near duplicates or overlap across hospitals.

ABP, external CinC2015 validation, self-supervised pretraining, and clinical integration are optional future work and are not implemented as completed experiments. The pipeline is for offline research; validation sensitivity targets are experimental settings, not clinical guarantees.

## References

- Lehman et al. VTaC v1.1, 2026. DOI: https://doi.org/10.13026/jg2e-tf67
- Lehman et al. VTaC: A Benchmark Dataset of Ventricular Tachycardia Alarms from ICU Monitors, NeurIPS 2023. Reference code: https://github.com/ML-Health/VTaC
- PhysioNet/CinC2015: https://physionet.org/challenge/2015/
