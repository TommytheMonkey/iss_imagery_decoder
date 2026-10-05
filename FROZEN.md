# FROZEN — robot36dec 0.2.0

Frozen at **2026-10-04T20:32:31Z** (UTC, from the shell) after the validation protocol of SPEC.md §5 steps 1-2 was re-run on the fixed
package (see the change log below; the first freeze was 0.1.0 at 2026-10-04T18:59:08Z). From this point the parameters in
`robot36dec/params_frozen.json` (loaded by default by `Params.default()`) may not change; only
`tools/validate_against_references.py` (blind comparison with the Series 33 reference slides) may run afterwards.

## Package hash

Combined sha256 of the per-file hashes below: `b8f0a358889187fc0976ea67638fff114ab280950e18e3fc92df8cd496e22723`

| file | sha256 | bytes |
|---|---|---|
| `robot36dec/__init__.py` | `2f9a0b539d176c39618094c013715b441eceb7819bb97400bf5712313cf136f7` | 136 |
| `robot36dec/__main__.py` | `6ce05b79ee1666025af142adc913f1d468781e77af67ae431fe163472a52fb3a` | 50 |
| `robot36dec/cli.py` | `32b2692ee435eca0761ede52967e014723fd1943080699bbbcc3d9f11f7c3944` | 43865 |
| `robot36dec/decode.py` | `bc816e9e8e9f821eedbf9696c638068a25fd1c0b70afd8046ee3dab1b63beb61` | 28269 |
| `robot36dec/diagnostics.py` | `d223836dcffb674c5fbcc300ec5701a25ddc487a1816435dfb771b2eb4fee5ef` | 30736 |
| `robot36dec/header.py` | `ed6dd4ad4d53afce31d992099eca1e477d7b4cbab029c83e8c74fe475b813456` | 31545 |
| `robot36dec/io.py` | `0ccea7266ef06b3d28e5547f6a47e22cc638ea1fc23d77b3e3450b5fe54d713a` | 6554 |
| `robot36dec/iqdemod.py` | `3a8ac1d635272450e9333ffdbde22964da0b83830c4d861e41c6fe259a42e375` | 41560 |
| `robot36dec/modes.py` | `f4c8e172c982943a7e5492d8ae05b1ccc7795a1af307853eef390c8a7dafbf4c` | 1569 |
| `robot36dec/params.py` | `a7fdcba4356fa899778c732842e838d1f7236342cb10b52d777f3a0503fb452b` | 7595 |
| `robot36dec/params_frozen.json` | `dc21b4a28be1770650e22a56d725f3ca5b0ea7c67dcb8135eb5564e00cd1aaf5` | 2002 |
| `robot36dec/preprocess.py` | `f9edb15b69920fdaf6e173a6729b0544286e2e88e2946d2290fb959466fdacbd` | 18542 |
| `robot36dec/sync.py` | `55068e409c71ad63749ce4c339e558693f77f1b7361d331ddf41271ec1ff11ad` | 39257 |
| `robot36dec/synth.py` | `11eb544bddd758aee395db33fefc8aae054a122e82d9fbe443307ef1b0203be3` | 16786 |
| `robot36dec/timing.py` | `4dc036f2f1d777c6ab9afd70d179a8eaa6490a14fe50f11d3d78da81c9626f75` | 54801 |
| `robot36dec/tones.py` | `698518fc96d2b17af1bc88c1e79c297eada11dd447c633d1b6b9d76486d29176` | 11432 |

## Test suite (`../.venv/bin/python -m pytest tests -q`)

`39 passed in 108.05s (0:01:48)`  (log: `.scratch/fix/pytest_final.log`, run with `params_frozen.json` in place; 30 synthetic tests incl. IQ round trips, 9 real-recording tests)

## Change log 0.1.0 -> 0.2.0 (fix pass after the four verifier reports)

No frozen parameter VALUE changed (75 fields identical); 5 fields were added with defaults
(`frame_start_window_lines`, `frame_start_false_alarm`, `picture_step_min_db`, `whiten_max_range_db`, `vis_noise_contrast_db`). The decoding path changed in these places:

* **Recovery frame start** (`timing._place_frame`): the "first run of 5 consecutive pulses" rule discarded genuine but
  sparsely detected pulses at CNR <= -6 dB (frames placed 4-14 lines late with confidence 0.9) and accepted a noise pulse
  adjacent to the frame. It is replaced by a density rule calibrated against the false-inlier rate (`frame_start_window_lines`,
  `frame_start_false_alarm`; two windows of 12 and 24 lines), a picture-presence check of the start line(s) from the per-line
  tone SNR (`picture_step_min_db`; a pulse on a line at the pre-frame noise level cannot start the frame when the picture level
  is resolvable), a feasible-range guard (pulses that cannot share a frame with the last core pulse are noise; `lo <= hi`
  always), a stricter VIS anchor (parity OK and bit tones agreeing) and an absolute 1200 Hz share floor for the header-end
  (stop-bit) signature. The integer-line ambiguity (expected undetected leading lines, fade-in, excess of pulses before the
  start) and the period standard error now enter `Timing.confidence`; `agreement.confidence_signal` keeps the signal
  evidence alone for the exit-code decision. `frame_run_lines`/`frame_run_min` are kept and reported as the alternative
  `first-run-start`.
* **Robot 36 structure test** (`timing`, `sync.estimate_period_acf.comb_z_half_rate`, `cli._signal_decision`): a pulse train
  on every other line or an ACF comb that is stronger at 300 ms than at 150 ms (a Robot 72 transmission aliased onto the
  150 ms grid) is reported, cuts the confidence and vetoes the pulse/comb branches of the decision; a credible foreign VIS
  code is named in the reason. The decision also requires a signal confidence >= 0.3 in every non-VIS branch.
* **VIS detector** (`header._ratio_z`): the band log-ratios are z-scored against the no-tone frames (spectral contrast
  < `vis_noise_contrast_db`) when at least 200 exist, and floored at -2. On clean recordings the file median of a band the
  picture never visits sat ~20 dB below the noise level, so carrier-off stretches scored as headers in every band and the
  header z fell from 19 (CNR 0 dB) to 5.7 (+20 dB); it is now 16 at every CNR, unchanged (19.5 / 24.8) on the recording.
* **Whitening** (`decode`): without a noise-only stretch the whole-file tone-excluded median is the picture's own tone
  leakage; a shape spanning more than `whiten_max_range_db` is not used (luma correlation 0.935 -> 0.977 at CNR +12 dB).
* **IQ front end** (`iqdemod`): the carrier centroid is confined to the declared +-search range; a detection whose in-band
  power is concentrated in 5 bins (CW) is rejected and the search repeated with the CW's channel masked; the CLI reports
  when no block locked (wrong `--center-freq`) or when a CW was rejected.
* **Diagnostics**: the line-sync CSV/JSON are evaluated on the final timing (`sync.track_lines(refit=False)`); the JSON
  confidence block says whether the VIS was used; `--mode normal` falls back to recovery without a usable Robot 36 candidate;
  argparse usage errors exit 1; the original WAV sample rate and resampling are reported.
* **Memory**: `remove_dc` and the gate reuse buffers, the running mean and the band-pass are chunked, the VIS frame FFT and
  the plot spectra use smaller chunks, the IQ reader uses buffered reads instead of a memory map, the raw audio is released
  after DC removal: the 660 s recording peaks at 1.65 GB RSS (was 2.68), a 10-minute 250 kS/s IQ file at 1.63 GB (was 2.65).
* `tools/validate_against_references.py` is implemented (blind per-block NCC with wrong-slide / wrong-position / control
  nulls and a per-line curve); it was exercised on synthetic references only and has NOT been run on the Series 33 slides.

## Validation on the reference recording `audio/ISS_2026-10-04_10-23-00.wav`

Oracles (SPEC §2, never decoding inputs): first leader 322.7945 s, line 0 323.705 s, period
0.1499964 s (-24 ppm), VIS 8 parity OK. `runs/<name>/` holds every output (PNG, JSON, CSV, plots).

| run | exit | timing source (anchor) | line 0 (offset vs oracle) | period | timing conf. | VIS code / conf / leader | comb z | sync inliers on the final grid | reliable px | band 62-108 row-corr | take2 excess_band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real_normal | 0 | vis+sync (vis) | 323.7043 (+0 lines, -0.67 ms) | 150.00066 ms (+28.4 ppm vs oracle) | 0.94 | 8 / 1.00 / leader 322.7954 | 37.2 | 92 | 26.1 % | +0.895 | +0.926 |
| real_recovery | 0 | sync-fit (header-end) | 323.7040 (+0 lines, -1.04 ms) | 150.00041 ms (+26.7 ppm vs oracle) | 0.90 | 8 / 1.00 / leader 322.7954 | 30.0 | 92 | 26.2 % | +0.899 | +0.930 |
| real_vis_removed | 0 | sync-fit (dense) | 323.7040 (+0 lines, -1.00 ms) | 149.99990 ms (+23.3 ppm vs oracle) | 0.90 | 17 / 0.23 / leader 338.8569 | 48.4 | 93 | 26.0 % | +0.899 | +0.930 |
| noise | 2 | sync-weak (midpoint) | 430.0925 (n/a: pure noise) | 150.00000 ms | 0.00 | None / 0.00 / leader 431.7866 | -0.4 | 5 | 1.8 % | -0.036 | (control) |
| noise_control | 2 | override (supported) | 430.0000 (n/a: pure noise) | 150.00066 ms | 0.50 | None / 0.01 / leader 462.1966 | -0.1 | 9 | 0.4 % | -0.031 | (control) |

* `real_normal`: `--mode normal --plots` (auto would take the same path: the VIS is credible).
* `real_recovery`: `--mode recovery --plots` (header ignored; line 0 anchored by the header-end signature of the stop bit).
* `real_vis_removed`: `--mode recovery` on `.scratch/integrate/ISS_vis_removed_gauss.wav`, a copy with 322.0-323.75 s
  replaced by Gaussian noise of the file's own RMS (0.1039 FS); the original was never modified (sha256
  736834f1647f26856513f42f365b717ab892eeb40327c1227d040c7ff9e8fd16 before and after every run).
* `noise`: `--start 430 --end 470` (pure receiver noise) -> exit 2, no VIS claim, confidence 0.
* `noise_control`: the identical pipeline on pure noise at t0 = 430.0 s with the image run's period and the gate forced on
  (`--start 430 --end 466.5 --mode recovery --t0 430.0 --period <P> --gate on`): the control for `scripts/take2_score.py`.
  excess_band = band row-correlation of the image minus that of the control (requirement >= 0.85 for the gated chain).

The headline numbers are identical to the 0.1.0 freeze (line 0 323.7043 / 323.7040 / 323.7040 s, periods 150.00066 /
150.00041 / 149.99990 ms, confidences 0.94 / 0.90 / 0.90, excess_band 0.926 / 0.930 / 0.930); the VIS z rose from 23.6 to
24.8 with the noise-referenced detector.

### Period: what "matches" means here

The oracle period is a single robust fit through 62 pulses (1.84 ms residual sd, ~107-line lever arm) with a standard
error of 34 ppm (bootstrap 5-95 %: +-65 ppm; skeptic A: 149.996 +- 0.008 ms = +-53 ppm). The decoder's estimates
(+23..+28 ppm above the oracle, i.e. 150.0004 ms) are within 1 sigma of it, as are all other estimators tried on this
audio (`.scratch/integrate/exp_period.log`: -7..+31 ppm). The real-recording period test therefore uses the oracle's
precision (50 ppm) while the 20 ppm requirement is enforced on synthetic full frames (exact truth). No parameter was
tuned to the recording. The combined period standard error is now reported (`agreement.period_se_ppm`, 8.6 ppm here) and
lowers the confidence of short lever arms (a 55-line frame: 0.94 -> 0.84).

## Parameters (`robot36dec/params_frozen.json`)

Established values from the analysis (SPEC §3) were kept unchanged. Fields added during integration (0.1.0):
`clean_snr_db`, `frame_run_lines`/`frame_run_min`, `vis_anchor_min_conf`. Fields added in the fix pass (0.2.0), all with
defaults that leave the reference recording's result unchanged: `frame_start_window_lines`, `frame_start_false_alarm`,
`picture_step_min_db`, `whiten_max_range_db`, `vis_noise_contrast_db`.

| parameter | value |
|---|---|
| `fs` | `48000` |
| `wav_fullscale_hz` | `48000.0` |
| `dc_window_s` | `0.5` |
| `gate_mode` | `"auto"` |
| `gate_level_hz` | `3840.0` |
| `gate_M` | `3` |
| `gate_q` | `2.0` |
| `gate_block_s` | `2.0` |
| `gate_auto_gain_db` | `1.0` |
| `gate_core_max_ratio` | `2.0` |
| `gate_test_win_ms` | `12.0` |
| `gate_test_hop_ms` | `6.0` |
| `bp_lo_hz` | `400.0` |
| `bp_hi_hz` | `3200.0` |
| `win_ms` | `12.0` |
| `win_ms_chroma` | `12.0` |
| `f_grid_lo` | `1350.0` |
| `f_grid_hi` | `2450.0` |
| `f_grid_step` | `10.0` |
| `whiten` | `true` |
| `estimator` | `"chain"` |
| `chain_T` | `8.0` |
| `chain_sigma_hz` | `80.0` |
| `chain_T_nogate` | `16.0` |
| `chain_sigma_hz_nogate` | `40.0` |
| `soft_T` | `1.0` |
| `clean_snr_db` | `8.0` |
| `chroma_pair_avg` | `3` |
| `vis_frame_ms` | `20.0` |
| `vis_hop_ms` | `5.0` |
| `vis_min_z` | `6.0` |
| `vis_bit_offset_scan_ms` | `30.0` |
| `sync_win_ms` | `9.0` |
| `sync_hop_ms` | `0.5` |
| `sync_band_lo` | `1130.0` |
| `sync_band_hi` | `1270.0` |
| `period_search_ms` | `[100.0, 1100.0]` |
| `period_nominal_ms` | `150.0` |
| `period_tolerance_ppm` | `300.0` |
| `line_search_ms` | `20.0` |
| `line_inlier_ms` | `4.0` |
| `n_lines` | `240` |
| `coh_t0_span_ms` | `6.0` |
| `coh_t0_step_ms` | `0.5` |
| `coh_ppm_span` | `60.0` |
| `coh_ppm_step` | `10.0` |
| `comb_z_min` | `4.0` |
| `coh_min_gain` | `0.02` |
| `frame_slack_s` | `0.5` |
| `recovery_margin_lines` | `60` |
| `frame_run_lines` | `5` |
| `frame_run_min` | `5` |
| `vis_anchor_min_conf` | `0.2` |
| `frame_start_window_lines` | `12` |  (added in 0.2.0)
| `frame_start_false_alarm` | `0.001` |  (added in 0.2.0)
| `picture_step_min_db` | `3.0` |  (added in 0.2.0)
| `whiten_max_range_db` | `8.0` |  (added in 0.2.0)
| `vis_noise_contrast_db` | `12.0` |  (added in 0.2.0)
| `mask_smooth_lines` | `3` |
| `mask_smooth_px` | `15` |
| `mask_sd_hz_fallback` | `110.0` |
| `mask_percentile` | `0.5` |
| `iq_channel_bw_hz` | `15000.0` |
| `iq_search_hz` | `15000.0` |
| `iq_block_s` | `0.5` |
| `iq_max_slew_hz_s` | `300.0` |
| `iq_deemphasis_us` | `75.0` |
| `iq_lock_snr_db` | `-6.0` |
| `iq_track_smooth_s` | `5.0` |
| `fuse_percentile` | `1.0` |
| `fuse_sd_hz_fallback` | `67.0` |
| `mask_min_line_frac` | `0.3` |
| `control_min_lines` | `40` |
| `detect_min_comb_z` | `3.0` |
| `detect_min_inliers` | `8` |
| `detect_min_frac_reliable` | `0.03` |
| `detect_min_timing_conf` | `0.5` |
| `carrier_block_s` | `0.5` |
| `carrier_min_quiet_db` | `1.0` |
| `carrier_min_run_s` | `1.5` |
