# robot36dec — a Robot 36 SSTV decoder that does not need the header

`robot36dec` (0.2.0) decodes Robot 36 SSTV pictures from FM-demodulated audio (48 kHz mono WAV, rtl_fm style) or directly
from RTL-SDR unsigned 8-bit IQ captures. It was built from the analysis of the ISS recording
`audio/ISS_2026-10-04_10-23-00.wav` (a pass received below the FM threshold with only partial carrier capture) and it
keeps the two properties that recording demanded:

* **a valid VIS header is not required** — a normal path reads the calibration leader / break / VIS code and a recovery
  path assumes Robot 36 and recovers the line timing from the 1200 Hz sync pulses, the ~150 ms line period, the
  autocorrelation comb and inter-line coherence;
* **the timing is redundant** — line 0 and the line period are estimated by several independent methods (VIS stop-bit
  end, per-line sync-pulse fit, ACF comb fit, phase fold, header-end signature, picture presence, separator parity, image
  coherence) and combined with explicit agreement and confidence numbers.

Input files are opened read-only and never modified. No reference image is ever read by the decoder; the only place
references may be used is the separate blind-validation tool `tools/validate_against_references.py`, which the package
does not import.

## Install

```sh
# Python >= 3.11 with numpy, scipy, soundfile, pillow (matplotlib only for --plots)
uv pip install --python <your-python> numpy scipy soundfile pillow matplotlib pytest
# or: pip install numpy scipy soundfile pillow matplotlib pytest
```

The package is pure Python; run it from this directory (`python -m robot36dec ...`) or add the directory to `PYTHONPATH`.
The project venv used for development is `../.venv/bin/python` (Python 3.13, numpy 2.5, scipy 1.18, soundfile 0.14,
Pillow 12, matplotlib 3.11).

## Usage

### FM-demodulated WAV (48 kHz mono, rtl_fm output)

```sh
python -m robot36dec decode recording.wav --out out/ --plots
python -m robot36dec decode recording.wav --out out/ --mode recovery          # ignore the header, sync-pulse timing only
python -m robot36dec decode recording.wav --out out/ --start 300 --end 380     # analyse a window (times are kept absolute)
python -m robot36dec decode recording.wav --out out/ --t0 323.705 --period 0.1499964   # force the timing
```

Audio is converted to **instantaneous frequency deviation in Hz**: an rtl_fm int16 full scale corresponds to a
deviation of `fs_out` = 48000 Hz (`--wav-fullscale-hz`, default 48000). Other sample rates are resampled to 48 kHz (the
original rate and the resampling ratio are reported). For generic sound-card audio the scale is unknown; all estimators
are scale-free, only the FM-threshold gate depends on it and its `auto` mode detects when it does not apply.

### RTL-SDR IQ (unsigned 8-bit interleaved I/Q)

```sh
python -m robot36dec decode capture.u8 --iq --iq-rate 1024000 --center-freq 437.540e6 --signal-freq 437.550e6 --out out/
```

The IQ front end (`iqdemod.py`) is chunked (buffered reads, the file is never loaded whole): DC removal, coarse channel
selection around `--signal-freq` (default 437.550 MHz), carrier search within ±15 kHz by 0.5 s block spectra with a
narrowband-interferer test (a CW carrier whose power sits in ~125 Hz is rejected and the search repeated with its channel
masked), a Doppler-tolerant carrier tracker (hold/predict through gaps, slew limit ~300 Hz/s), fine mixing by the smoothed
track, polyphase resampling to 48 kHz, a 15 kHz channel low-pass, polar discriminator and 75 µs de-emphasis. The carrier
track (time, offset, SNR, lock) is written to the diagnostics JSON and summarised in the log; when no block locks (wrong
`--center-freq`, `--signal-freq` or `--iq-rate`) the decoder says so explicitly. Sample rates 240 kS/s to 2.4 MS/s were
tested; the chain runs 28–40× real time on 5–10 minute files.

### Options

| option | meaning |
|---|---|
| `--mode auto\|normal\|recovery` | `auto` (default): normal path when a credible Robot 36 VIS is found, else recovery; `normal`: use a Robot 36 VIS candidate of any confidence ≥ 0.2 (else recovery); `recovery`: ignore the header entirely |
| `--t0 S`, `--period S` | override line-0 time (seconds into the input) and line period |
| `--start S`, `--end S` | analyse only this window (absolute times are preserved) |
| `--gate auto\|on\|off` | FM-threshold gate (default `auto`) |
| `--estimator chain\|soft` | luminance estimator (default `chain`, see below) |
| `--window-ms`, `--wav-fullscale-hz`, `--params FILE.json` | analysis window, WAV scale, parameter overrides |
| `--plots` | diagnostic plots (needs matplotlib) |
| `--save-audio` | also write the demodulated and the preprocessed audio as float WAV into `--out` |
| `--basename NAME`, `--quiet` | output prefix (default: input stem), suppress the log |

Exit code 0 = a credible Robot 36 signal was decoded, 2 = no credible Robot 36 signal (diagnostics are still written and
the summary names the reason, e.g. `VIS 12 = Robot 72 identified ... sync pulses only on every other line`), 1 = error
(including usage errors).

## Outputs (all inside `--out`, prefixed with the input stem)

| file | content |
|---|---|
| `<base>_image.png`, `_image_x3.png` | RGB reconstruction (320×240; ×3 nearest-neighbour) |
| `<base>_luma.png`, `_luma_x3.png` | luminance (grayscale) reconstruction, one-line estimator |
| `<base>_luma_fused.png`, `_luma_fused_masked(_x3).png` | reliability-gated 3-line fusion, with/without the unreliable pixels greyed |
| `<base>_image_fused_masked.png`, `<base>_mask.png` | fused RGB with unreliable pixels grey; the reliability mask |
| `<base>_diagnostics.json` | everything numeric: timing (t_line0, period, source, confidence, agreement of all estimators incl. placement anchor and ambiguity, period standard error, pulse-pattern / Robot 36 structure test, signal confidence), VIS candidates (code, mode name, parity, bit tones, z-score, confidence, whether used for the timing), period estimate (ACF comb z at 150 and 300 ms, comb-fit period and its standard error, candidates for other modes), line-sync summary on the final timing (detected pulse times, inliers, residual sd) and the fit's statistics, gate decision and statistics, whitening decision, per-block SNR / quieting / reliable fraction / row coherence, the signal decision with its criteria and reason, parameters, software version, IQ carrier track and lock summary |
| `<base>_line_sync.csv` | per line of the final frame: expected and detected sync-pulse time, residual, strength, inlier flag |
| `<base>_line_metrics.csv` | per line: start time, audio SNR, wideband quieting, separator parity, reliable fraction |
| `<base>_plots/` (`--plots`) | `header_vis.png` (spectrogram around the header with the ideal VIS tone sequence), `sync_acf.png` (ACF comb), `sync_raster.png` (sync likelihood folded at the final timing), `line_metrics.png`, `freq_raster.png` (dominant-frequency raster next to the decoded luminance) |

## How it works

Both paths share the audio chain: DC/Doppler removal (running trimmed mean, 0.5 s) → FM-threshold gate (see below) →
400–3200 Hz band-pass.

**Normal path.** `header.find_vis` correlates a template of the whole 910 ms VIS header (leader 1900 Hz 300 ms, break
1200 Hz 10 ms, leader, start bit 1200 Hz 30 ms, eight bit slots in the 1100-or-1300 Hz band, stop bit) with band-energy
ratios z-scored against the file's no-tone frames (so that a clean picture does not inflate the false-alarm statistics;
the header z is ~16 from CNR 0 to +20 dB and 25 on the reference recording), refines each peak at 1 ms hop jointly over
all 128 codes, decodes the bits with a sampling-offset scan (stability plateau), checks parity and reports z-score, noise
maximum, mode name and a confidence in [0, 1]. A Robot 36 candidate (code 8) with confidence ≥ 0.6 defines line 0 at the
end of the stop bit; other codes are reported by name but not decoded.

**Recovery path** (`--mode recovery`, or `auto` without a credible header). `sync.sync_likelihood` turns the audio into
the share of the 900–2600 Hz energy that lies in the 1200 Hz sync band (9 ms Hann windows, 0.5 ms hop).
`estimate_period_acf` detects the Robot 36 comb (ACF peaks at k·150 ms minus the half-period troughs, in robust-sd
units; the same statistic at 300 ms exposes a Robot 72 train) and refines the period by a weighted fit of the comb-peak
lags over long lags; `fold_phase` finds the pulse phase; `track_lines` searches every line for its pulse (±20 ms), fits a
robust line through the inliers (tolerance shrinking 12 → 4 ms) and rejects 30 ms VIS-bit plateaus by a pulse-shape
test. `timing.resolve_timing` then places the 240-line frame on the pulse grid: the feasible range of integer placements
comes from the detected pulses (a pulse that cannot share a frame with the last one is noise) and the carrier-presence
span, and the anchor is, in order of priority, the header-end signature (a 1200 Hz stop bit of tone-like strength
immediately before a line start that is followed by picture tones), a Robot 36 VIS candidate with valid parity and
agreeing bit tones (auto mode only), the *dense start* (the first pulse followed by enough further pulses within 12 or 24
lines, 'enough' calibrated against the measured false-inlier rate so that isolated noise pulses never start a frame), the
first supported pulse, or the midpoint. The per-line tone SNR then checks the start: when the picture level is resolvable
above the pre-frame noise level (pure receiver noise gives −8.9 dB, a picture at CNR −6 dB about −6 dB), a pulse on a
noise-level line cannot be the first line and the start moves to the first pulse on a picture line; the separator-tone
parity (1500 Hz even / 2300 Hz odd lines) fixes an even/odd error, using the same picture evidence to break ties. A
pulse train on every other line, or a comb that is stronger at 300 ms than at 150 ms, marks a non-Robot-36 line structure
(Robot 72): the confidence is cut and the transmission is reported, not decoded. The period is the inverse-variance
combination of the sync fit and the ACF comb fit; a coherence refinement (sync-column contrast + adjacent-row correlation
of a quick decode on a t0 ±6 ms / ±60 ppm grid) is always reported and adopted only when the sync fit is weak. The
confidence carries the integer-line ambiguity (expected number of undetected leading lines from the local detection
density, fade-in at the start, excess of pulses before the chosen start) and the period's standard error; the signal
evidence alone (`confidence_signal`) decides the exit code. The feasible range and every alternative placement are written
to the diagnostics.

**Image estimation** (`decode.decode_robot36`). 12 ms Hann spectra at every pixel centre on a 1350–2450 Hz grid, whitened
by the receiver-noise spectral shape measured on the file's own no-signal stretches (when there is none, the whole-file
shape is used only if it spans ≤ 8 dB: a clean picture's whole-file "shape" is its own tone leakage); luminance by a
Gaussian Markov chain along the line (forward–backward posterior mean, evidence temperature T = 8, σ = 80 Hz/px; T = 16,
σ = 40 when the gate is off) in the FM-threshold regime, or by the soft posterior-mean estimator when the frame's median
per-line SNR is ≥ 8 dB (above the FM threshold the chain prior only blurs sharp content); chroma by the soft estimator
with 3-line-pair averaging; JPEG YCrCb → RGB. A reliability mask comes from the smoothed posterior sd, thresholded at the
0.5th percentile of a *control decode* of the file's own noise with the identical pipeline (so pure noise keeps 0.5 % of
its pixels by construction); the fused display is a reliability-gated 3-line inverse-variance fusion. Per line the decoder
reports audio SNR, wideband (8–20 kHz) quieting as carrier evidence, and separator parity.

**FM-threshold gate.** An rtl_fm discriminator below threshold outputs the tone when the carrier is captured and a ±fs/2
noise burst when it is not. The gate `w = exp(-(e/L²)²)` with `e` the 3-sample local power and `L` = 3840 Hz (0.08 FS)
suppresses the bursts; it is the single largest gain on the ISS recording (band row-correlation 0.60 → 0.89). In `auto`
mode it is applied only when gating raises the in-band tone contrast by ≥ 1 dB somewhere in the file and the signal has
the expected rtl_fm scale.

## Validation

Protocol (SPEC.md §5): synthetic tests first (`tests/test_synthetic.py`: encoder, rtl_fm-faithful FM channel at CNR
+6 / 0 / −2 / −4 dB with and without VIS, both paths, pure noise, IQ round trips at 250 kS/s and 1.024 MS/s with a
Doppler ramp), then the reference recording (`tests/test_real_recording.py`: VIS 8, leader 322.7945 s, line 0 323.705 s,
period 0.1499964 s, recovery path, VIS region replaced by noise, pure-noise cut), then the parameters were frozen
(`robot36dec/params_frozen.json`, `FROZEN.md` with the hashes, the change log and all validation numbers). Only after the
freeze may `tools/validate_against_references.py` compare a decode with the reference slides:

```sh
../.venv/bin/python -m pytest tests -q        # 39 tests, ~2 min (the real-recording tests skip if the WAV is absent)
../.venv/bin/python tools/validate_against_references.py runs/real_normal/ISS_2026-10-04_10-23-00_luma.png ../event_images \
    --control-luma runs/noise_control/ISS_control_430_luma.png --out VALIDATION.md     # blind comparison, after the freeze only
```

The decodes of the reference recording used for the freeze are in `runs/` (`real_normal`, `real_recovery`,
`real_vis_removed`, `noise`, `noise_control`); `FROZEN.md` has the full table. Headline numbers (oracle: leader 322.7945 s,
line 0 323.7050 s, period 0.14999640 s, VIS 8):

| run | line 0 | period | source / confidence | VIS | band 62-108 row-corr (excess over the noise control) |
|---|---|---|---|---|---|
| normal path | 323.7043 s (−0.7 ms) | 150.00066 ms (+28 ppm vs oracle, within its 34 ppm 1σ) | vis+sync / 0.94 | 8, parity OK, leader 322.7954 s, z 24.8, conf 1.00 | 0.895 (0.926) |
| recovery (header ignored) | 323.7040 s (−1.0 ms, 0 lines) | 150.00041 ms | sync-fit (header-end anchor z 15) / 0.90 | ignored | 0.899 (0.930) |
| VIS region replaced by noise | 323.7040 s (0 lines) | 149.99990 ms | sync-fit (dense start + parity) / 0.90 | none credible | 0.899 (0.930) |
| pure noise 430–470 s | — | — | sync-weak / 0.00, exit 2 | none | ≈ 0 |

Beyond the suite, the 0.2.0 fix pass was checked against the verifiers' adversarial inputs (`.scratch/fix/notes.md`):
header-less synthetic frames at CNR −6 dB (flat and fading) are placed on the right line (previously 8–14 lines late),
a header-less −2 dB frame is placed correctly for 10/10 seeds (previously 2/6 two lines early), a Robot 72 transmission
is rejected with its VIS named (previously "decoded"), clean signals reach a luma correlation of 0.976 with the VIS
credited (previously 0.93 and no VIS), a CW interferer 15 dB above the SSTV carrier is rejected and the SSTV tracked, and
the 660 s recording decodes in 1.65 GB of RAM (previously 2.68 GB).

### Blind comparison with the Series 33 reference slides (after the freeze only)

`tools/validate_against_references.py` was run **after** the freeze (FROZEN.md 2026-10-04T20:32:31Z, run 2026-10-04T20:45:58Z; the
report records both timestamps and the sha256 of `params_frozen.json`, which matches the frozen hash). The reference
images were used **only for scoring the finished decodes**: nothing in the package reads, compares with or is conditioned
on a reference image, and no number from the comparison was fed back into any parameter. Method (ported from the earlier
analysis scripts): each reference (every file in `../event_images/`: received copies of all 12 slides, received images of
unknown slide and non-slide distractors; this project's own `*_pipeline_*` decodes of other stations' audio excluded as
not independent) is converted to luminance, resampled to 320×240 and passed through the decoder's channel blur (horizontal
Gaussian σ = 9 px; a 3-line average only for the fused displays); the statistic is the Pearson correlation (NCC) per line
block at zero shift and at the best small shift (±3 lines, ±4 px); the null distributions are other slides, the same
reference at non-overlapping vertical positions (also mirrored) and the decoder's own pure-noise control decodes. Full
report: `VALIDATION.md`; figure `runs/validation/VALIDATION_figure.png`; per-line curves and the full ranking as CSV next to it.

| decode | lines 0-20 | 20-62 | 62-108 | 108-240 | all | z (band, vs wrong positions) | best other slide (band) |
|---|---|---|---|---|---|---|---|
| normal path, 1-line | 0.878 | 0.628 | 0.926 | 0.056 | 0.389 | 6.1 / 7.4 | 0.505 (slide 7) |
| recovery path (header ignored), 1-line | 0.877 | 0.626 | 0.924 | 0.056 | 0.388 | 6.0 / 7.4 | 0.515 (slide 7) |
| fused 3-line (normal / recovery) | 0.851 / 0.850 | 0.644 / 0.642 | 0.928 / 0.927 | 0.070 / 0.071 | 0.397 / 0.396 | 6.0 / 7.2 | 0.516 / 0.527 |
| pure-noise control decode (same pipeline) | 0.02 | −0.06 | 0.04 | 0.02 | 0.01 | — | — |
| earlier analysis (take2 final, 1-line) | 0.87 | 0.65 | 0.93 | 0.06 | 0.41 | 6.4 | — |

Both paths identify **slide 10** (13 of the top 19 ranks are copies of slide 10, the band 62-108 margin over the best copy
of any other slide is 0.40-0.42, the other three named copies of slide 10 agree within 0.01); lines 108-240 are at the
null (the lower half of the slide is not in the recording); the normal and recovery paths agree within 0.004 in every
block; and the frozen decoder reproduces the earlier take2 numbers (0.87 / 0.65 / 0.93 / 0.06) within 0.03, with 86
lines (10-19, 26-35, 42-106) above a per-line NCC of 0.5 against 76 before.

## Limitations

* Robot 36 only; other VIS codes are identified and named but not decoded. A Robot 72 transmission is recognised by its
  300 ms pulse train and refused.
* A frame that begins inside a deep fade starts at its first detectable line: when the first lines carry neither pulses
  nor picture content above the noise level (instantaneous CNR ≲ −7 dB) they are lost and the reported ambiguity grows;
  nothing in the signal distinguishes them from the pre-frame noise. With a destroyed header the even/odd parity fixes
  odd offsets only. The feasible range and all alternative placements are reported.
* The line period of a real recording is known only as well as its pulses allow (the reference recording: ±34 ppm with 62
  usable pulses); the ±20 ppm requirement is met on synthetic full frames. Short frames (≲ 60 lines) give periods
  60–120 ppm off with a formal standard error of 25–100 ppm, which lowers the reported confidence.
* The FM-threshold gate and the chain-estimator settings were verified on an rtl_fm 48 kHz discriminator output; the
  `auto` gate refuses inputs at a different scale and the decoder then falls back to the ungated settings. The gate's
  auto decision is marginal (noise-driven) at CNR ≥ +6 dB, where the estimator choice makes it irrelevant.
* The IQ carrier tracker holds the last smoothed offset through gaps and interpolates linearly across them; a gap of a
  minute around closest approach would be bridged by a straight line (harmless inside the 15 kHz channel). A CW
  interferer is rejected by its spectral narrowness; a strongly modulated neighbouring FM signal is not.
* Lock thresholds of the IQ front end come from synthetic captures only; no real RTL-SDR IQ of this event exists.
* Memory: the 660 s WAV peaks at 1.65 GB RSS, a 10-minute 250 kS/s IQ file at 1.63 GB and a 5.5-minute 1.024 MS/s file
  at 1.56 GB (the intermediate stream of pass 1 is cached in RAM when it is below 768 MB).
* Only one frame per file is decoded (the highest-confidence VIS or the best comb window); multi-slide recordings need
  `--start`/`--end`.
