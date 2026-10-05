"""All tunable parameters in one place. Serialisable to/from JSON. `params_frozen.json` is loaded by default once it exists."""
from dataclasses import dataclass, field, asdict
import json, os

@dataclass
class Params:
    # audio / preprocessing
    fs: int = 48000
    wav_fullscale_hz: float = 48000.0        # rtl_fm int16 full scale = fs_out Hz of deviation
    dc_window_s: float = 0.5                 # running trimmed-mean DC (Doppler) removal
    gate_mode: str = "auto"                  # auto | on | off
    gate_level_hz: float = 3840.0            # 0.08 FS at 48 kHz full scale; ~2x the max SSTV tone deviation
    gate_M: int = 3                          # local-power window (samples) of the gate
    gate_q: float = 2.0                      # gate exponent w = exp(-(e/L^2)^q)  [dsp]
    gate_block_s: float = 2.0                # auto mode: block length of the gate-effect test  [dsp]
    gate_auto_gain_db: float = 1.0           # auto mode: in-band tone-contrast gain (gated vs ungated, dB) needed to apply  [dsp]
    gate_core_max_ratio: float = 2.0         # auto mode: veto if the robust width 1.4826*MAD(x) exceeds this x gate_level_hz  [dsp]
    gate_test_win_ms: float = 12.0           # auto mode: window of the contrast test  [dsp]
    gate_test_hop_ms: float = 6.0            # auto mode: hop of the contrast test  [dsp]
    bp_lo_hz: float = 400.0
    bp_hi_hz: float = 3200.0
    # tone estimation
    win_ms: float = 12.0
    win_ms_chroma: float = 12.0
    f_grid_lo: float = 1350.0
    f_grid_hi: float = 2450.0
    f_grid_step: float = 10.0
    whiten: bool = True
    estimator: str = "chain"                 # chain | soft
    chain_T: float = 8.0
    chain_sigma_hz: float = 80.0
    chain_T_nogate: float = 16.0
    chain_sigma_hz_nogate: float = 40.0
    soft_T: float = 1.0
    clean_snr_db: float = 8.0                # [integrate] median per-line SNR (dB re 2400 Hz) at/above which the audio is above the FM
                                             # threshold (no click noise: the gate's own test stops firing at CNR >= +6 dB); the chain
                                             # prior (T=8/sigma=80, verified at CNR -2.5 dB) then over-smooths and the soft estimator is used
    chroma_pair_avg: int = 3
    # header
    vis_frame_ms: float = 20.0
    vis_hop_ms: float = 5.0
    vis_min_z: float = 6.0                   # matched-filter z-score (vs rest of file) to accept a header
    vis_bit_offset_scan_ms: float = 30.0
    # sync / period
    sync_win_ms: float = 9.0
    sync_hop_ms: float = 0.5
    sync_band_lo: float = 1130.0
    sync_band_hi: float = 1270.0
    period_search_ms: tuple = (100.0, 1100.0)
    period_nominal_ms: float = 150.0
    period_tolerance_ppm: float = 300.0
    line_search_ms: float = 20.0
    line_inlier_ms: float = 4.0
    n_lines: int = 240
    # coherence refinement
    coh_t0_span_ms: float = 6.0
    coh_t0_step_ms: float = 0.5
    coh_ppm_span: float = 60.0
    coh_ppm_step: float = 10.0
    # --- added by the sync agent (sync/timing); read with getattr fallbacks to the same defaults
    comb_z_min: float = 4.0                  # estimate_period_acf: minimum robust comb z for a Robot 36 period to be reported as found
    coh_min_gain: float = 0.02               # resolve_timing: minimum coherence-score gain to adopt the refined t0/period (weak fits only)
    frame_slack_s: float = 0.5               # resolve_timing: slack (s) around signal_span when placing the 240-line frame
    recovery_margin_lines: int = 60          # resolve_timing: extra lines tracked before/after the provisional frame (needs sync_signal)
    frame_run_lines: int = 5                 # [integrate] recovery placement: the frame starts at the first inlier pulse that begins a run of ...
    frame_run_min: int = 5                   # ... at least this many inlier pulses within frame_run_lines lines (isolated pulses before it are noise)
    vis_anchor_min_conf: float = 0.2         # [integrate] recovery placement (mode auto): a Robot 36 VIS candidate below this confidence never anchors line 0
    # --- added by the fix pass (0.2.0): recovery frame start by pulse density instead of a run of 5 consecutive pulses
    frame_start_window_lines: int = 12       # [fix] the frame starts at the first inlier pulse followed by enough further inliers within this many lines ...
    frame_start_false_alarm: float = 1e-3    # ... 'enough' = the count a uniform false-inlier background (calibrated rate) reaches with at most this probability
    picture_step_min_db: float = 3.0         # [fix] minimum per-line tone-SNR step (pre-frame noise -> picture) for the picture-presence check of the start line(s)
    whiten_max_range_db: float = 8.0         # [fix] a noise shape estimated WITHOUT a noise-only stretch is not used for whitening when its range exceeds this (it is the image's own tone leakage)
    vis_noise_contrast_db: float = 12.0      # [fix] VIS detector: frames whose 900-2600 Hz spectrum has a peak/median contrast below this are no-tone frames; the band ratios are z-scored against them
    # reliability
    mask_smooth_lines: int = 3
    mask_smooth_px: int = 15
    mask_sd_hz_fallback: float = 110.0
    mask_percentile: float = 0.5
    # IQ front end
    iq_channel_bw_hz: float = 15000.0
    iq_search_hz: float = 15000.0
    iq_block_s: float = 0.5
    iq_max_slew_hz_s: float = 300.0
    iq_deemphasis_us: float = 75.0
    iq_lock_snr_db: float = -6.0             # carrier detection: minimum SNR (dB, in iq_channel_bw_hz) of a 0.5 s block  [io-iq]
    iq_track_smooth_s: float = 5.0           # carrier track smoother: half width (s) of the local linear fit  [io-iq]
    # --- added by the pipeline agent (decode/cli): fusion calibration and the 'signal found' decision; defaults = established values
    fuse_percentile: float = 1.0             # fusion neighbour gate = this percentile of the control's (unsmoothed) posterior sd
    fuse_sd_hz_fallback: float = 67.0        # used when the file has no no-signal stretch to calibrate on (T8/sigma80, 12 ms value)
    mask_min_line_frac: float = 0.3          # a line counts as 'reliable' when this fraction of its pixels passes the mask
    control_min_lines: int = 40              # minimum control-decode length (lines) for calibrating mask/fusion/SNR floor
    detect_min_comb_z: float = 3.0           # exit-code-2 decision: ACF comb significance (noise gives ~N(0,1); the image window 4.8) ...
    detect_min_inliers: int = 8              # ... number of inlier sync pulses ...
    detect_min_frac_reliable: float = 0.03   # ... fraction of reliable pixels (pure noise gives mask_percentile/100) ...
    detect_min_timing_conf: float = 0.5      # ... timing confidence
    carrier_block_s: float = 0.5             # carrier-presence (wideband quieting) block length
    carrier_min_quiet_db: float = 1.0        # minimum quieting (dB below the noise floor) to call the carrier present
    carrier_min_run_s: float = 1.5           # minimum duration of a carrier run
    def to_json(self, path):
        with open(path, "w") as fh: json.dump(asdict(self), fh, indent=1)
    @classmethod
    def from_json(cls, path):
        with open(path) as fh: d = json.load(fh)
        p = cls()
        for k, v in d.items():
            if hasattr(p, k): setattr(p, k, tuple(v) if isinstance(getattr(p, k), tuple) else v)
        return p
    @classmethod
    def default(cls):
        frozen = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_frozen.json")
        return cls.from_json(frozen) if os.path.exists(frozen) else cls()
