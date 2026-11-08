# Verification status

This package implements the core proposal. It does not contain a completed full-dataset scientific experiment.

## Passed checks

- Eight automated integrity tests: threshold sensitivity and tie rules, retain-all operating point, cross-record one-to-one shuffle, impossible-shuffle rejection, paired bootstrap identity, gap rejection/interpolation, official split leakage rejection, and a mocked WFDB loader check verifying that every read ends at sample 75,000 exclusively.
- Synthetic end-to-end run: 128 toy events; logistic regression and ECG/fusion/shuffled CNNs; three seeds; three training epochs; 100 record-clustered bootstrap resamples. Saved artifacts, predictions, report and plots are included in `demo_run/`.
- Real official metadata audit: 5,037 labeled events. Train: 4,060 events / 1,163 true / 1,808 records. Validation: 495 / 141 / 226. Test: 482 / 137 / 226. Patient waveform record IDs are disjoint across these sets.
- Real WFDB header, ECG and PPG loading: 250 Hz, 30-second crop ending just before alarm onset. Constant PPG is excluded; absent PPG is tracked separately.

## Small real-data smoke run

`real_smoke/` contains a deliberately small subset run to check execution, not estimate generalization. The sampled events were selected using the first eight events in each official split/label cell; this is an artificial class-balanced convenience sample, not the dataset's natural prevalence. Official split membership is unchanged. It downloaded 48 alarms; 44 were paired-eligible (14 train / 15 validation / 15 test). Constant or invalid signals and absent PPG were handled by the logged eligibility rules. Cache provenance and evaluation files retain exact counts. The run used one seed, three epochs, the quick configuration, and 100 bootstrap resamples.

The first smaller setup revealed a useful edge case: after exclusions, one test record held more than half the events, making a one-to-one cross-record PPG permutation mathematically impossible. The code correctly refused that shuffle. The package now checks shuffle feasibility for every split before training. The smoke check was enlarged for I/O verification, without using prediction performance to choose the sample. This is not a study cohort revision.

## Still required for the scientific study

1. Download all annotated waveforms and preprocess into a new cache directory.
2. Inspect training waveform examples and quality proxies; document any justified quality-rule changes before final evaluation.
3. Run the full three-configuration validation search and three seeds with the default 40-epoch maximum/early stopping.
4. Freeze the study and run the final test evaluation with 1,000 bootstrap resamples.
5. Interpret the sensitivity–specificity tradeoff, uncertainty, and quality subgroups from those full-study results.

The delivered demos use a short training budget and 100 bootstrap resamples to verify execution. They must not be described as the final VTaC benchmark result. ABP, external validation and pretraining are optional extensions, not completed deliverables.
