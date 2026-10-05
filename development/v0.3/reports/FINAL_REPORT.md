# v0.3 validation and independent real-recording analysis

The v0.2.0 parent is preserved. v0.3 is an isolated development copy, not a production release. See [exact changes](v02-to-v03.patch), [implementation details](IMPLEMENTATION.md), [unit log](validation/unit-tests.log), and [RF validation results](validation/release-final/results.json).

## Independent validation

25 unit tests passed. All 14 end-to-end RF cases completed. The computational resume shortcut also passed exact comparison of all 26 candidate reports in the independent low-deviation case, and bitwise audio-array comparison for tone, Gaussian-noise and impulsive controls. These execution-equivalence checks validate the shortcut; the decoder-free encoder and null cases validate protocol behavior. The RF generator imports no decoder modules. Tests use unsigned 8-bit I,Q at 1.024 MS/s, a −100 kHz capture offset, +6 kHz oscillator error, curved Doppler, fades and quantization. CNR is referenced to 48 kHz before fades; these numbers are not audio SNR. At 1.024 MS/s, full-capture CNR is 13.29 dB lower: the +6 dB test is −7.29 dB over the entire sampled RF bandwidth.

| Injected CNR (dB / 48 kHz) | Verdict | RF hypotheses | Supported hypotheses |
|---:|---|---:|---:|
| +6 | CONFIRMED ROBOT36 | 35 | 18 |
| +0 | CONFIRMED ROBOT36 | 69 | 23 |
| -3 | CONFIRMED ROBOT36 | 52 | 17 |
| -6 | PROBABLE | 54 | 3 |
| -9 | INCONCLUSIVE | 65 | 0 |
| -12 | NO EVIDENCE | 59 | 0 |

These are one seed/profile per SNR, not detection-probability estimates. Multiple supported RF hypotheses often represent sidebands/fragments of one injected transmitter. Detection does not measure recovered image quality.

| CNR (dB) | 11 kHz | 15 kHz | 25 kHz | 0 us | 75 us |
|---:|---|---|---|---|---|
| +6 | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 |
| +0 | PROBABLE | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 | PROBABLE | CONFIRMED ROBOT36 |
| -3 | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 | PROBABLE | CONFIRMED ROBOT36 | CONFIRMED ROBOT36 |
| -6 | PROBABLE | PROBABLE | INCONCLUSIVE | INCONCLUSIVE | PROBABLE |
| -9 | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| -12 | NO EVIDENCE | NO EVIDENCE | NO EVIDENCE | NO EVIDENCE | NO EVIDENCE |

Each column is the best outcome over all other tested choices; it is not an independently calibrated detection rate.

| Independent control | Verdict | Supported hypotheses |
|---|---|---:|
| colored_noise | NO EVIDENCE | 0 |
| unrelated_fm | INCONCLUSIVE | 0 |
| cw | NO EVIDENCE | 0 |
| periodic_non_sstv | INCONCLUSIVE | 0 |
| constant_tone | NO EVIDENCE | 0 |
| invalid_vis | PROBABLE | 12 |
| low_deviation | CONFIRMED ROBOT36 | 8 |
| mixed_fm_cw | CONFIRMED ROBOT36 | 12 |

Invalid parity is never a valid VIS confirmation. A genuine Robot 36 body with intentionally corrupted parity may remain PROBABLE. The half-rate pulse regression, wrong porch, wrong sync width, constant-tone overrides and measured-confidence controls are also in the unit suite. No negative control was accepted as PROBABLE or CONFIRMED.

## Real recordings

Both files were scanned in full, after independent validation passed, with the same pre-frozen settings: 1,024,000 complex samples/s, u8 I,Q, center 437,650,000 Hz, nominal 437,550,000 Hz, ±30 kHz residual search and 20 ppm declared uncertainty. No timing hint, old classification, orbital trajectory, selected time region or image reference influenced acquisition. Input SHA-256 and original file size/mtime are recorded.

All 1,933 retained trajectories were independently evaluated. There were no VIS hypotheses meeting the measured header detector criteria, no credible Robot 36 VIS headers, no PROBABLE/CONFIRMED candidates and no recovered images. Real evaluation used the unchanged production loop and a verified computational shortcut that skips duplicated, unused off-gate diagnostics. The separate FFT gate experiment was not used.

| Recording | Duration (s) | RF hypotheses | Verdict | Images |
|---|---:|---:|---|---:|
| 16-54 | 719.360 | 1040 | INCONCLUSIVE | 0 |
| 18-30 | 749.312 | 893 | INCONCLUSIVE | 0 |

### 16-54

Candidate verdict counts: {'NO EVIDENCE': 1039, 'INCONCLUSIVE': 1}.
Every RF hypothesis, rejection reason and best structural measurement: [candidate table](real/16-54/CANDIDATES.md), [candidate CSV](real/16-54/candidates.csv). Every measured time/frequency/CNR point: [track CSV](real/16-54/tracks.csv), [trajectory plot](real/16-54/frequency_tracks.png). Every bandwidth/preprocessing/window evaluation and measured VIS bit/parity decision: [full report](real/16-54/candidates.json). Input integrity: [SHA-256 and metadata](real/16-54/input-integrity.json).

Reported RF frequencies include oscillator offset. Linear slew is a summary of the full measured trajectory, not an ISS identity test. The list contains threshold-limited spectral placements, not necessarily distinct physical transmitters.

Whole-file ADC RMS about each component mean: I=0.957203, Q=0.957200 byte codes; rail fractions [0.0, 0.0]. [All ADC histograms/statistics](real/16-54/adc-statistics.json). This very small variation makes low-level quantization an important sensitivity limitation. It does not by itself prove a receiver fault or absence of RF.

### 18-30

Candidate verdict counts: {'NO EVIDENCE': 891, 'INCONCLUSIVE': 2}.
Every RF hypothesis, rejection reason and best structural measurement: [candidate table](real/18-30/CANDIDATES.md), [candidate CSV](real/18-30/candidates.csv). Every measured time/frequency/CNR point: [track CSV](real/18-30/tracks.csv), [trajectory plot](real/18-30/frequency_tracks.png). Every bandwidth/preprocessing/window evaluation and measured VIS bit/parity decision: [full report](real/18-30/candidates.json). Input integrity: [SHA-256 and metadata](real/18-30/input-integrity.json).

Reported RF frequencies include oscillator offset. Linear slew is a summary of the full measured trajectory, not an ISS identity test. The list contains threshold-limited spectral placements, not necessarily distinct physical transmitters.

Whole-file ADC RMS about each component mean: I=0.975733, Q=0.975706 byte codes; rail fractions [0.0, 0.0]. [All ADC histograms/statistics](real/18-30/adc-statistics.json). This very small variation makes low-level quantization an important sensitivity limitation. It does not by itself prove a receiver fault or absence of RF.

### Independent review of 400–450 s in the 18:30 file

This region was reviewed only after the whole-recording scan. Below are the ten largest combined sync/pulse-geometry scores among overlapping evaluations, including rejected hypotheses; these rankings do not change any verdict.

| Candidate | Window start (s) | RF BW (kHz) | De-emphasis (us), gate | Period (ms) | Sync z | Pulse z | Porch z | Separator z | Alternation z | Verdict / reason |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| RF0582 | 414.00 | 15 | 0, off | 149.850000 | 3.52 | 4.76 | -2.24 | -2.82 | 0.13 | INCONCLUSIVE: failed: porch, separator, alternation, sync_parity_0, sync_parity_1 |
| RF0582 | 414.00 | 15 | 0, auto | 149.850000 | 3.52 | 4.76 | -2.24 | -2.82 | 0.13 | INCONCLUSIVE: failed: porch, separator, alternation, sync_parity_0, sync_parity_1 |
| RF0564 | 409.50 | 11 | 75, off | 150.120000 | 3.74 | 3.75 | -2.34 | -1.04 | 2.00 | INCONCLUSIVE: failed: porch, separator, alternation, sync_parity_0, sync_parity_1, sync_width |
| RF0564 | 409.50 | 11 | 75, auto | 150.120000 | 3.74 | 3.75 | -2.34 | -1.04 | 2.00 | INCONCLUSIVE: failed: porch, separator, alternation, sync_parity_0, sync_parity_1, sync_width |
| RF0582 | 414.00 | 15 | 75, off | 149.977500 | 3.01 | 4.08 | -4.64 | -2.97 | -0.09 | NO EVIDENCE: failed: sync, porch, separator, alternation, sync_parity_1 |
| RF0582 | 414.00 | 15 | 75, auto | 149.977500 | 3.01 | 4.08 | -4.64 | -2.97 | -0.09 | NO EVIDENCE: failed: sync, porch, separator, alternation, sync_parity_1 |
| RF0575 | 412.00 | 11 | 75, off | 150.123750 | 3.48 | 3.55 | -1.90 | -0.70 | 2.55 | NO EVIDENCE: failed: sync, porch, separator, alternation, sync_parity_0, sync_parity_1 |
| RF0575 | 412.00 | 11 | 75, auto | 150.123750 | 3.48 | 3.55 | -1.90 | -0.70 | 2.55 | NO EVIDENCE: failed: sync, porch, separator, alternation, sync_parity_0, sync_parity_1 |
| RF0575 | 412.00 | 11 | 0, off | 150.150000 | 2.90 | 3.88 | -2.13 | -0.11 | 2.52 | NO EVIDENCE: failed: sync, porch, separator, alternation, sync_parity_0, sync_parity_1 |
| RF0575 | 412.00 | 11 | 0, auto | 150.150000 | 2.90 | 3.88 | -2.13 | -0.11 | 2.52 | NO EVIDENCE: failed: sync, porch, separator, alternation, sync_parity_0, sync_parity_1 |

**Region verdict: INCONCLUSIVE.** The two strongest partial candidates are RF0564 (411.5–415.0 s) and RF0582 (416.0–420.5 s). Their best fits are 150.120 ms and 149.850 ms, respectively; these short, searched fits do not independently establish the previously reported 150.0029 ms cadence. Neither passes porch, separator, alternating-chroma and both-parity sync checks. No credible VIS header supports them. I cannot positively classify this region as Robot 36 or confidently identify its transmitter as unrelated RF.

[Frequency and sync-profile illustration](real/18-30/region_400_450.png). The elevated, irregular off-pulse baseline in the highest-scoring profiles is another reason not to treat a fitted comb period as a verified 9 ms sync train.

A near-150 ms best-fit period alone is insufficient: the algorithm explicitly searches that period neighborhood. The z values are engineering statistics after multiple searched hypotheses, not calibrated p-values.

## Representative moving trajectories and partial structure

These illustrate the observed motion and strongest partial evidence; every retained trajectory and point remains in the linked full inventories. Endpoints are spectral placement estimates, and fitted slew includes oscillator error, association and sideband effects.

| File | Candidate | Time (s) | First RF (MHz) | Last RF (MHz) | Fitted slew (Hz/s) | Verdict |
|---|---|---:|---:|---:|---:|---|
| 16-54 | RF0384 | 279.5–309.5 | 437.556349 | 437.554650 | -57.2 | NO EVIDENCE |
| 16-54 | RF0669 | 438.0–457.5 | 437.544193 | 437.541381 | -104.1 | NO EVIDENCE |
| 16-54 | RF0686 | 440.0–448.5 | 437.544432 | 437.544051 | -65.5 | INCONCLUSIVE |
| 18-30 | RF0533 | 402.0–427.0 | 437.551410 | 437.549231 | -100.3 | NO EVIDENCE |
| 18-30 | RF0542 | 403.0–427.0 | 437.545877 | 437.545415 | -63.7 | NO EVIDENCE |
| 18-30 | RF0564 | 411.5–415.0 | 437.547977 | 437.549519 | 425.8 | INCONCLUSIVE |
| 18-30 | RF0582 | 416.0–420.5 | 437.551494 | 437.550632 | -209.4 | INCONCLUSIVE |

## Remaining reasons a verdict could be wrong

Known injected Robot 36 is missed at sufficiently weak CNR. Spectral-component CNR in the real reports is not a total transmitter-power estimate and cannot be directly compared to the injected full-signal CNR without summing the relevant components. Noise-model errors, narrow filtering, FM threshold effects, front-end clipping, frequency uncertainty beyond the scanned range, IQ impairments, sample-rate error beyond the cadence search, short/partial frames, fades, simultaneous unresolved transmitters, sideband centroid bias, and greedy track association remain limitations. The synthetic tests use u8 quantization but do not establish sensitivity at the approximately one-code ADC RMS measured in these files; a separate amplitude/dither/gain sweep is still needed. The null suite is finite and does not calibrate false alarms across a long recording. RF motion alone cannot identify ISS without receiver calibration and ephemeris. The inherited tone/image reconstruction is heavily blurred and has not been validated across independent color reference patterns. A PROBABLE body without valid VIS has frame-origin ambiguity.

**I would not trust this decoder to establish that a real-world recording contains no recoverable Robot 36 image.** Its reports can establish what the tested hypotheses support; negative results remain bounded by measured sensitivity and search assumptions.


## Recommended next work, ranked

1. Calibrate false alarms and detection probability with many independent seeds, long null recordings, multiple image patterns and multiple RF deviation profiles; include sample-clock error jointly with oscillator error.
2. Add receiver gain/amplitude/dither sweeps down to sub-code RF amplitude, IQ imbalance/DC/spurs/clipping and adjacent strong interferers. The approximately one-code ADC RMS in these files is not adequately characterized by the current sensitivity sweep.
3. Improve joint association of fragmented/sideband RF trajectories and quantify frequency uncertainty; compare independently implemented FM demodulation and acquisition. A threshold-limited bank and greedy linkage can still miss weak signals.
4. Extend declared RF uncertainty for uncalibrated receivers when justified by receiver metadata; the present results are conditional on the fixed ±30 kHz search and default 20 ppm declaration. Add independent orbital/receiver calibration before attributing motion to ISS.
5. Validate full luminance/chroma reconstruction against independent color test patterns and real reference decoders. The inherited long tone-analysis window limits image detail; body detection is not an image-quality guarantee.
6. Refactor candidate execution for performance and reproducible checkpoints in a separately tested change. The execution helpers used here preserve the original measured results, but the production CLI remains computationally expensive.
