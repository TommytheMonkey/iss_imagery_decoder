"""Robot 36 image reconstruction from preprocessed audio and a Timing.

Port of the verified take-2 'combo' pipeline (claude-recovery/take2/combo/combo_lib.py, 04_display.py) and of
scripts/robot36.py, 10_snr_parity.py and 06_timing.py:
    per-pixel Hann spectra (tones.tone_power) -> optional whitening by the file's own receiver-noise shape
    -> soft (independent pixels) or chain (Gaussian Markov chain along the line, forward-backward) estimator
    -> luminance, chroma (soft estimator, line-pair averaging), JPEG YCrCb colour
    -> reliability mask from the (3 lines x 15 px)-smoothed posterior sd, reliability-gated 3-line inverse-variance fusion
    -> per-line audio SNR (2400 Hz reference), separator-tone parity, wideband (8-20 kHz) quieting.
The mask / fusion thresholds and the SNR floor are calibrated on a 'control' decode of the file's own no-signal audio with
the identical pipeline (so that pure noise keeps params.mask_percentile % of its pixels by construction); when the file has
no such stretch, fixed fallback values from Params are used.  No reference image is read anywhere in this module.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.ndimage import uniform_filter
from .params import Params
from .timing import Timing
from .modes import ROBOT36
from . import tones

LN2 = math.log(2.0)
H, W = ROBOT36.height, ROBOT36.width
# line blocks used for REPORTING only (the blocks of the ISS analysis); they never influence the decoding.
BLOCKS = {'top_0_20': (0, 20), 'mid_20_62': (20, 62), 'band_62_108': (62, 108), 'low_108_240': (108, 240), 'all': (0, 240)}


@dataclass
class DecodeResult:
    Y: np.ndarray                # (240,320) float 0..255 luminance, 1-line estimator
    C: np.ndarray                # (240,320) float chroma levels (R-Y even lines, B-Y odd)
    rgb: np.ndarray              # (240,320,3) uint8
    post_sd_Y: np.ndarray        # (240,320) posterior sd in Hz
    post_sd_C: np.ndarray
    mask: np.ndarray             # (240,320) bool reliable
    Y_fused: np.ndarray          # reliability-gated 3-line fusion (display)
    rgb_fused_masked: np.ndarray # (240,320,3) uint8, unreliable pixels grey
    line_snr_db: np.ndarray      # (240,) audio SNR estimate (2400 Hz ref) in the Y segment
    line_quieting_db: np.ndarray # (240,) wideband 8-20 kHz level vs file noise (needs raw audio; NaN if unavailable)
    line_sep_parity: np.ndarray  # (240,) +1 if the separator tone matches the expected even/odd tone, -1 if opposite, 0 unknown
    sep_parity_agreement: float  # fraction of lines agreeing (0.5 = random)
    frac_reliable: float
    reliable_runs: list = field(default_factory=list)
    noise_shape: np.ndarray | None = None
    estimator: str = ""          # 'chain' | 'soft' (the estimator actually used for the luminance)
    # --- additional outputs (all optional, defaults keep the contract above intact)
    estimator_detail: str = ""               # settings and the regime decision, e.g. 'chain T=8 sigma=80 Hz/px whitened'
    snr_regime: str = ""                     # 'fm-threshold' (established chain prior) | 'clean' (above threshold: soft estimator)
    snr_median_db: float = float('nan')      # median per-line SNR over the lines inside the audio (the regime statistic)
    f_Y: np.ndarray | None = None            # (240,320) posterior-mean frequency (Hz) of the luminance estimator
    Y_fused_masked: np.ndarray | None = None # Y_fused with unreliable pixels set to 128
    post_sd_Y_fused: np.ndarray | None = None
    line_frac_reliable: np.ndarray | None = None   # (240,) fraction of reliable pixels per line
    line_in_range: np.ndarray | None = None        # (240,) bool: the whole line lies inside the audio
    sep_parity_agreement_reliable: float = float('nan')  # agreement over the lines that count as reliable
    mask_threshold_hz: float = float('nan')
    fuse_threshold_hz: float = float('nan')
    snr_floor_db: float = float('nan')       # 99th percentile of the control decode's per-line SNR (NaN without a control)
    quieting_ref_db: float = float('nan')    # 10log10 of the wideband reference power used for line_quieting_db
    control: dict = field(default_factory=dict)    # control-decode summary (t0, n_lines, rowcorr, frac_reliable, ...)
    noise_regions: list = field(default_factory=list)
    calibration: str = ""                    # 'control' or 'fallback'
    freqs: np.ndarray | None = None
    whitening: str = ""                      # 'noise regions' | 'whole-file tone-excluded median' | 'skipped (...)' | 'off'


# ----------------------------------------------------------------------------------------------- geometry
def frequency_grid(params: Params) -> np.ndarray:
    n = int(round((params.f_grid_hi - params.f_grid_lo) / params.f_grid_step)) + 1
    return params.f_grid_lo + params.f_grid_step * np.arange(n, dtype=np.float64)


def pixel_times(t_line0: float, period: float, params: Params | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Absolute centre times of Y pixels and chroma pixels, shape (240,320) each (see modes.ROBOT36)."""
    m = ROBOT36
    ls = float(t_line0) + np.arange(H, dtype=np.float64)[:, None] * float(period)
    px = np.arange(W, dtype=np.float64)[None, :] + 0.5
    clock_scale = float(period) / 0.150
    ty = ls + clock_scale * (m.y_start_ms + px * m.y_ms / W) / 1000.0
    tc = ls + clock_scale * (m.c_start_ms + px * m.c_ms / W) / 1000.0
    return ty, tc


def line_start_times(t_line0: float, period: float, n_lines: int = H) -> np.ndarray:
    return float(t_line0) + np.arange(n_lines, dtype=np.float64) * float(period)


# ----------------------------------------------------------------------------------------------- colour
def ycrcb_to_rgb(Y: np.ndarray, Cr: np.ndarray, Cb: np.ndarray) -> np.ndarray:
    """JPEG YCrCb -> RGB uint8."""
    Y = np.asarray(Y, np.float64); Cr = np.asarray(Cr, np.float64) - 128.0; Cb = np.asarray(Cb, np.float64) - 128.0
    R = Y + 1.402 * Cr
    G = Y - 0.714136 * Cr - 0.344136 * Cb
    B = Y + 1.772 * Cb
    return np.clip(np.round(np.stack([R, G, B], -1)), 0, 255).astype(np.uint8)


def _line_average(A: np.ndarray, k: int) -> np.ndarray:
    """edge-replicated boxcar over k adjacent rows (k odd)."""
    A = np.asarray(A, np.float64)
    if k <= 1 or A.shape[0] == 0: return A
    r = k // 2; n = A.shape[0]; out = np.zeros_like(A)
    for d in range(-r, r + 1):
        out += A[np.clip(np.arange(n) + d, 0, n - 1)]
    return out / k


def colour_image(Y: np.ndarray, C: np.ndarray, pair_avg: int = 3) -> np.ndarray:
    """Y: (L,W) 0..255; C: (L,W) chroma levels 0..255 (even lines R-Y, odd lines B-Y). pair_avg: average the chroma over
    this many line pairs (odd). Returns (L,W,3) uint8 (robot36.colour_image)."""
    L = Y.shape[0]
    Cr = _line_average(C[0::2], pair_avg); Cb = _line_average(C[1::2], pair_avg)
    Crf = np.repeat(Cr, 2, axis=0)[:L]; Cbf = np.repeat(Cb, 2, axis=0)[:L]
    if Crf.shape[0] < L: Crf = np.vstack([Crf, np.full((L - Crf.shape[0], W), 128.0)])
    if Cbf.shape[0] < L: Cbf = np.vstack([Cbf, np.full((L - Cbf.shape[0], W), 128.0)])
    return ycrcb_to_rgb(Y, Crf, Cbf)


# ----------------------------------------------------------------------------------------------- fusion / mask / scores
def gated_line_fusion(mu: np.ndarray, sd: np.ndarray, k: int, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-variance fusion of lines l-r..l+r in the frequency domain; a neighbour contributes only where its posterior
    sd is below thr (combo_lib.gated_lines, 'weighted/M6'). Returns (mu_fused, sd_fused)."""
    mu = np.asarray(mu, np.float64); sd = np.maximum(np.asarray(sd, np.float64), 1e-6)
    n = mu.shape[0]; w = 1.0 / sd ** 2
    num = mu * w; den = w.copy(); r = k // 2; rows = np.arange(n)
    for d in range(-r, r + 1):
        if d == 0: continue
        idx = rows + d
        valid = (idx >= 0) & (idx < n)
        idx = np.clip(idx, 0, n - 1)
        ok = (sd[idx] < thr) & valid[:, None]
        num = num + np.where(ok, mu[idx] * w[idx], 0.0)
        den = den + np.where(ok, w[idx], 0.0)
    return num / den, np.sqrt(1.0 / den)


def smooth_sd(sd: np.ndarray, lines: int = 3, px: int = 15) -> np.ndarray:
    return uniform_filter(np.asarray(sd, np.float64), size=(max(1, int(lines)), max(1, int(px))), mode='nearest')


def rowcorr(A: np.ndarray, a: int = 0, b: int | None = None, lag: int = 1, quantise: bool = True) -> float:
    """mean adjacent-row Pearson correlation (row means removed) of the 8-bit-quantised image rows a..b (take2_score rule)."""
    A = np.asarray(A, np.float64)
    if quantise: A = np.round(np.clip(A, 0, 255))
    R = A[a:b]
    if R.shape[0] <= lag: return float('nan')
    R = R - R.mean(1, keepdims=True)
    num = (R[:-lag] * R[lag:]).sum(1)
    den = np.sqrt((R[:-lag] ** 2).sum(1) * (R[lag:] ** 2).sum(1)) + 1e-9
    return float(np.mean(num / den))


def line_runs(flags: np.ndarray) -> list:
    """contiguous runs of True as [(first, last), ...]."""
    flags = np.asarray(flags, bool); runs = []; i = 0; n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]: j += 1
            runs.append((int(i), int(j))); i = j + 1
        else:
            i += 1
    return runs


# ----------------------------------------------------------------------------------------------- per-line metrics
def line_snr_db(PY: np.ndarray, win_ms: float, px_step: int = 4, ref_bw_hz: float = 2400.0) -> np.ndarray:
    """Per-line audio SNR (dB) referred to a 2400 Hz bandwidth (scripts/10_snr_parity.py): every px_step-th pixel spectrum,
    noise per bin = median/ln2 (exponential statistics), tone = peak - noise, ENBW of the Hann window = 1.5/T."""
    P = np.asarray(PY[:, ::px_step], np.float64)
    med = np.median(P, -1); pk = P.max(-1); nb = med / LN2
    tone = np.maximum(pk - nb, 0.0); enbw = 1.5 / (win_ms / 1000.0)
    snr = tone / np.maximum(nb / enbw * ref_bw_hz, 1e-30)
    return 10.0 * np.log10(np.mean(snr, axis=1) + 1e-12)


def separator_parity(x: np.ndarray, fs: int, x_t0: float, t_line0: float, period: float, n_lines: int = H) -> np.ndarray:
    """+1 where the 100-104.5 ms separator tone matches the expected tone (1500 Hz even lines, 2300 Hz odd lines),
    -1 where it is the opposite, 0 unknown."""
    m = ROBOT36
    tsep = line_start_times(t_line0, period, n_lines) + (m.sep_start_ms + m.sep_ms / 2.0) / 1000.0
    Ps = np.asarray(tones.tone_power(x, fs, x_t0, tsep, m.sep_ms, np.array([m.f_black, m.f_white])), np.float64)
    d = Ps[:, 1] - Ps[:, 0]                       # > 0 -> 2300 Hz (odd-line separator)
    vote = np.sign(d)
    expected = np.where(np.arange(n_lines) % 2 == 0, -1.0, 1.0)
    return (vote * expected).astype(np.int8)


def band_power_at(raw: np.ndarray, fs: int, raw_t0: float, centres: np.ndarray, dur_s: float,
                  band: tuple = (8000.0, 20000.0)) -> np.ndarray:
    """Band power (linear) of Hann-windowed blocks of dur_s centred at `centres` (abs s); NaN where the block is outside raw."""
    n = int(round(dur_s * fs)); win = np.hanning(n)
    f = np.fft.rfftfreq(n, 1.0 / fs); sel = (f >= band[0]) & (f < band[1])
    centres = np.atleast_1d(np.asarray(centres, np.float64))
    out = np.full(len(centres), np.nan)
    i0 = np.round((centres - raw_t0) * fs - n / 2.0).astype(np.int64)
    for k, s0 in enumerate(i0):
        if s0 < 0 or s0 + n > len(raw): continue
        seg = raw[s0:s0 + n].astype(np.float64); seg -= seg.mean()
        S = np.fft.rfft(seg * win)
        out[k] = float((np.abs(S[sel]) ** 2).sum())
    return out


def block_band_power(raw: np.ndarray, fs: int, block_s: float, band: tuple = (8000.0, 20000.0), raw_t0: float = 0.0,
                     chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Band power of consecutive non-overlapping Hann-windowed blocks over the whole raw signal; returns (t_centre, power)."""
    n = int(round(block_s * fs)); nb = len(raw) // n
    if nb == 0: return np.zeros(0), np.zeros(0)
    win = np.hanning(n); f = np.fft.rfftfreq(n, 1.0 / fs); sel = (f >= band[0]) & (f < band[1])
    out = np.empty(nb)
    for s in range(0, nb, chunk):
        e = min(nb, s + chunk)
        blk = raw[s * n:e * n].astype(np.float64).reshape(e - s, n)
        blk -= blk.mean(1, keepdims=True)
        S = np.fft.rfft(blk * win, axis=1)
        out[s:e] = (np.abs(S[:, sel]) ** 2).sum(1)
    t = raw_t0 + (np.arange(nb) + 0.5) * n / fs
    return t, out


def line_quieting_db(raw: np.ndarray, fs: int, raw_t0: float, t_line0: float, period: float,
                     band: tuple = (8000.0, 20000.0), n_lines: int = H, ref_percentile: float = 80.0):
    """Per-line wideband level (dB) relative to the file's no-carrier noise floor (the ref_percentile-th percentile of the
    band power over all period-long blocks of the raw audio; a below-threshold FM discriminator is noisiest WITHOUT carrier,
    so the upper part of the distribution is the no-carrier state).  Negative values = quieting = carrier present."""
    centres = line_start_times(t_line0, period, n_lines) + period / 2.0
    p_line = band_power_at(raw, fs, raw_t0, centres, period, band)
    _, p_ref = block_band_power(raw, fs, period, band, raw_t0)
    if len(p_ref) < 4 or not np.isfinite(p_ref).any():
        return np.full(n_lines, np.nan), float('nan')
    ref = float(np.nanpercentile(p_ref, ref_percentile))
    if ref <= 0: return np.full(n_lines, np.nan), float('nan')
    with np.errstate(divide='ignore', invalid='ignore'):
        q = 10.0 * np.log10(p_line / ref)
    return q, 10.0 * math.log10(ref)


# ----------------------------------------------------------------------------------------------- estimators
def _norm_shape(shape) -> np.ndarray | None:
    if shape is None: return None
    s = np.asarray(shape, np.float64).ravel()
    if s.size == 0 or not np.isfinite(s).all() or s.mean() <= 0: return None
    return s / s.mean()


def snr_regime(snr_db: np.ndarray, params: Params) -> tuple[str, float]:
    """('clean' | 'fm-threshold', median line SNR). The established chain prior (T = 8, sigma = 80 Hz/px; T = 16, sigma = 40 without
    the gate) was verified at CNR -2.5 dB, where the FM-threshold click noise makes the per-window likelihood overconfident. Above the
    FM threshold (median per-line SNR >= params.clean_snr_db, 8 dB re 2400 Hz; the gate's own signature test stops firing there) that
    prior only blurs sharp content (synthetic card at +6 dB: chain 0.94, soft 0.97 truth correlation), so the soft estimator
    (posterior mean, T = params.soft_T) is used instead. The decision is one per decode and applies to the control decode too."""
    v = np.asarray(snr_db, np.float64); v = v[np.isfinite(v)]
    med = float(np.median(v)) if v.size else float('nan')
    thr = float(getattr(params, 'clean_snr_db', 8.0))
    return ('clean' if (np.isfinite(med) and med >= thr) else 'fm-threshold'), med


def _estimate_Y(P: np.ndarray, freqs: np.ndarray, shape, params: Params, gate_applied: bool, regime: str = 'fm-threshold'):
    """returns (name, detail, f_est, post_sd) for the luminance estimator selected by params.estimator and the SNR regime."""
    est = (params.estimator or 'chain').lower()
    wh = " whitened" if shape is not None else ""
    if est == 'chain' and regime == 'clean':
        f, sd = tones.soft_estimate(P, freqs, shape, T=float(params.soft_T))[:2]
        name = 'soft'
        detail = (f"soft T={float(params.soft_T):g}{wh} (clean regime: median line SNR >= {float(getattr(params, 'clean_snr_db', 8.0)):g} dB, "
                  f"the chain prior is not used above the FM threshold)")
    elif est == 'chain':
        if gate_applied: T, sig = float(params.chain_T), float(params.chain_sigma_hz)
        else: T, sig = float(params.chain_T_nogate), float(params.chain_sigma_hz_nogate)
        f, sd = tones.chain_estimate(P, freqs, shape, T=T, sigma_hz=sig)[:2]
        name = 'chain'
        detail = f"chain T={T:g} sigma={sig:g} Hz/px{wh}" + ("" if gate_applied else " (no-gate fallback)")
    elif est == 'soft':
        f, sd = tones.soft_estimate(P, freqs, shape, T=float(params.soft_T))[:2]
        name = 'soft'
        detail = f"soft T={float(params.soft_T):g}{wh}"
    else:
        raise ValueError(f"unknown estimator {params.estimator!r} (chain|soft)")
    return name, detail, np.asarray(f, np.float64), np.asarray(sd, np.float64)


# ----------------------------------------------------------------------------------------------- noise regions / control
def noise_regions(x_t0: float, duration_s: float, t_line0: float, period: float, pre_margin_s: float = 3.0,
                  post_margin_s: float = 2.0, edge_s: float = 0.5, min_len_s: float = 3.0) -> list:
    """Stretches of the available audio outside the frame (and outside the VIS header before it) as [(a, b), ...] (abs s)."""
    regs = []
    a, b = x_t0 + edge_s, t_line0 - pre_margin_s
    if b - a >= min_len_s: regs.append((float(a), float(b)))
    a, b = t_line0 + H * period + post_margin_s, x_t0 + duration_s - edge_s
    if b - a >= min_len_s: regs.append((float(a), float(b)))
    return regs


def _control_placement(regions: list, t_line0: float, period: float, min_lines: int):
    """Longest no-signal region; a control frame of up to H lines is placed at its end farthest from the image frame."""
    best = None
    for (a, b) in regions:
        n = int(math.floor((b - a) / period))
        if n >= min_lines and (best is None or n > best[0]): best = (n, a, b)
    if best is None: return None
    n, a, b = best; n = min(n, H)
    ctrl_t0 = a if b <= t_line0 else b - n * period
    return float(ctrl_t0), int(n)


def _decode_lines(x, fs, x_t0, t0, period, n_lines, params, freqs, shape, gate_applied, regime='fm-threshold'):
    """luminance decode of n_lines lines starting at t0 with the pipeline's estimator (used for the control; the regime decision
    of the image decode is imposed so that the calibration uses the identical estimator)."""
    ty, _ = pixel_times(t0, period, params)
    P = tones.tone_power(x, fs, x_t0, ty[:n_lines], params.win_ms, freqs)
    name, detail, f, sd = _estimate_Y(P, freqs, shape, params, gate_applied, regime)
    return dict(t0=float(t0), n_lines=int(n_lines), f=f, sd=sd, Y=tones.to_level(f, ROBOT36.f_black, ROBOT36.f_white),
                snr_db=line_snr_db(P, params.win_ms), estimator=detail)


# ----------------------------------------------------------------------------------------------- main decoder
def decode_robot36(x: np.ndarray, fs: int, x_t0: float, timing: Timing, params: Params,
                   raw_audio: np.ndarray | None = None, gate_applied: bool = True,
                   quieting_band_hz: tuple = (8000.0, 20000.0), raw_t0: float | None = None,
                   calibrate_on_noise: bool = True) -> DecodeResult:
    """tone_power at Y and chroma pixel centres (params.win_ms / win_ms_chroma, grid params.f_grid_*);
    noise shape from estimate_noise_shape (regions: outside the frame if the file is longer than the frame, else percentile);
    estimator per params.estimator ('chain' with chain_T/chain_sigma_hz, or the no-gate fallback values when the gate was
    not applied; 'soft'); chroma with the soft estimator and chroma_pair_avg; RGB via JPEG YCrCb; reliability mask from the
    smoothed posterior sd with a threshold = params.mask_percentile-th percentile of the posterior sd over no-signal windows
    (if available) else params.mask_sd_hz_fallback; fused display; per-line SNR (peak - noise)/(N0*2400) from the Y spectra;
    separator parity from 4.5 ms windows at 100-104.5 ms (1500 vs 2300 Hz power); quieting from raw_audio per line
    (8-20 kHz band power vs the file's lowest-percentile reference).

    raw_audio (optional) is the pre-gate wideband audio in the same units/fs, starting at raw_t0 (default x_t0)."""
    m = ROBOT36
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    fs = int(fs)
    t0 = float(timing.t_line0); period = float(timing.period)
    if not (math.isfinite(t0) and math.isfinite(period) and period > 0):
        raise ValueError(f"unusable timing: t_line0={t0}, period={period}")
    freqs = frequency_grid(params)
    duration = len(x) / fs
    ty, tc = pixel_times(t0, period, params)

    # ---- luminance spectra and estimator
    PY = tones.tone_power(x, fs, x_t0, ty, params.win_ms, freqs)
    regions = noise_regions(x_t0, duration, t0, period)
    shape = shape_c = None
    whitening = 'off'
    if params.whiten:
        shape = _norm_shape(tones.estimate_noise_shape(x, fs, freqs, params.win_ms, regions if regions else None, x_t0))
        if params.win_ms_chroma == params.win_ms: shape_c = shape
        else: shape_c = _norm_shape(tones.estimate_noise_shape(x, fs, freqs, params.win_ms_chroma, regions if regions else None, x_t0))
        whitening = 'noise regions' if regions else 'whole-file tone-excluded median'
        # without a noise-only stretch the shape is estimated on the signal itself; on a clean recording the tone-excluded
        # median is the leakage of the picture's own tones (a 15 dB 'shape' at CNR +12 dB that biases every pixel), whereas
        # the receiver-noise shape of an FM discriminator spans only a few dB over 1350-2450 Hz: such a shape is not used
        max_range = float(getattr(params, 'whiten_max_range_db', 8.0))
        if not regions and shape is not None:
            rng_db = float(10.0 * np.log10(shape.max() / max(shape.min(), 1e-30)))
            if rng_db > max_range:
                shape = shape_c = None
                whitening = f'skipped (no noise-only stretch; the whole-file shape spans {rng_db:.1f} dB > {max_range:g} dB, i.e. tone leakage)'
    # ---- which lines lie entirely inside the available audio
    half = max(params.win_ms, params.win_ms_chroma) / 2000.0
    in_range = (ty[:, 0] - half >= x_t0) & (tc[:, -1] + half <= x_t0 + duration)

    # ---- SNR regime (one decision per decode, imposed on the control decode as well) and the luminance estimator
    snr = line_snr_db(PY, params.win_ms)
    regime, snr_med = snr_regime(snr[in_range] if in_range.any() else snr, params)
    est_name, est_detail, fY, sdY = _estimate_Y(PY, freqs, shape, params, gate_applied, regime)
    if whitening.startswith('skipped'): est_detail += ' [whitening ' + whitening + ']'
    Y = tones.to_level(fY, m.f_black, m.f_white)

    # ---- chroma (soft estimator at the chroma pixel centres)
    PC = tones.tone_power(x, fs, x_t0, tc, params.win_ms_chroma, freqs)
    fC, sdC = tones.soft_estimate(PC, freqs, shape_c, T=float(params.soft_T))[:2]
    fC = np.asarray(fC, np.float64); sdC = np.asarray(sdC, np.float64)
    C = tones.to_level(fC, m.f_black, m.f_white)

    # ---- calibration on the file's own no-signal audio (identical pipeline)
    fuse_pct = float(getattr(params, 'fuse_percentile', 1.0))
    fuse_fallback = float(getattr(params, 'fuse_sd_hz_fallback', 67.0))
    min_ctrl = int(getattr(params, 'control_min_lines', 40))
    ctrl = None; control_info = {}
    if calibrate_on_noise:
        pl = _control_placement(regions, t0, period, min_ctrl)
        if pl is not None:
            ctrl = _decode_lines(x, fs, x_t0, pl[0], period, pl[1], params, freqs, shape, gate_applied, regime)
    if ctrl is not None:
        thr_mask = float(np.percentile(smooth_sd(ctrl['sd'], params.mask_smooth_lines, params.mask_smooth_px), params.mask_percentile))
        thr_fuse = float(np.percentile(ctrl['sd'], fuse_pct))
        snr_floor = float(np.percentile(ctrl['snr_db'], 99.0))
        calibration = 'control'
    else:
        thr_mask = float(params.mask_sd_hz_fallback); thr_fuse = fuse_fallback; snr_floor = float('nan'); calibration = 'fallback'

    # ---- mask, fusion, colour
    ssd = smooth_sd(sdY, params.mask_smooth_lines, params.mask_smooth_px)
    mask = (ssd < thr_mask) & in_range[:, None]
    mu_f, sd_f = gated_line_fusion(fY, sdY, 3, thr_fuse)
    Y_fused = tones.to_level(mu_f, m.f_black, m.f_white)
    rgb = colour_image(Y, C, int(params.chroma_pair_avg))
    Y_fused_masked = np.where(mask, Y_fused, 128.0)
    rgb_fm = colour_image(Y_fused_masked, np.where(mask, C, 128.0), int(params.chroma_pair_avg))
    rgb_fm[~mask] = 128

    # ---- per-line metrics
    parity = separator_parity(x, fs, x_t0, t0, period)
    parity[~in_range] = 0
    line_frac = mask.mean(1)
    min_frac = float(getattr(params, 'mask_min_line_frac', 0.3))
    rel_lines = line_frac > min_frac
    agree = float(np.mean(parity[parity != 0] > 0)) if np.any(parity != 0) else 0.5
    sel = rel_lines & (parity != 0)
    agree_rel = float(np.mean(parity[sel] > 0)) if np.any(sel) else float('nan')
    if raw_audio is not None:
        quiet, qref = line_quieting_db(np.asarray(raw_audio), fs, x_t0 if raw_t0 is None else float(raw_t0), t0, period, quieting_band_hz)
    else:
        quiet, qref = np.full(H, np.nan), float('nan')

    if ctrl is not None:
        c_ssd = smooth_sd(ctrl['sd'], params.mask_smooth_lines, params.mask_smooth_px)
        control_info = dict(t0=ctrl['t0'], n_lines=ctrl['n_lines'], estimator=ctrl['estimator'],
                            rowcorr_Y=rowcorr(ctrl['Y']), frac_reliable=float((c_ssd < thr_mask).mean()),
                            mean_level=float(ctrl['Y'].mean()), std_level=float(ctrl['Y'].std()),
                            post_sd_median_hz=float(np.median(ctrl['sd'])), snr_db_median=float(np.median(ctrl['snr_db'])),
                            snr_db_p99=snr_floor)

    return DecodeResult(
        Y=Y, C=C, rgb=rgb, post_sd_Y=sdY, post_sd_C=sdC, mask=mask, Y_fused=Y_fused, rgb_fused_masked=rgb_fm,
        line_snr_db=snr, line_quieting_db=quiet, line_sep_parity=parity, sep_parity_agreement=agree,
        frac_reliable=float(mask.mean()), reliable_runs=line_runs(rel_lines), noise_shape=shape, estimator=est_name,
        estimator_detail=est_detail, snr_regime=regime, snr_median_db=snr_med,
        f_Y=fY, Y_fused_masked=Y_fused_masked, post_sd_Y_fused=sd_f, line_frac_reliable=line_frac, line_in_range=in_range,
        sep_parity_agreement_reliable=agree_rel, mask_threshold_hz=thr_mask, fuse_threshold_hz=thr_fuse,
        snr_floor_db=snr_floor, quieting_ref_db=qref, control=control_info, noise_regions=list(regions),
        calibration=calibration, freqs=freqs, whitening=whitening)


# ----------------------------------------------------------------------------------------------- quick decode for timing refinement
def quick_luma(x: np.ndarray, fs: int, x_t0: float, t_line0: float, period: float, params: Params,
               lines: tuple[int, int] | None = None, win_ms: float = 12.0, px_step: int = 2, f_step_hz: float = 20.0):
    """Cheap luminance decode (soft estimator, optional line subset) for coherence refinement. Returns (Y, sync_contrast).
    Y has shape (n_lines, 320 // px_step) on a coarse 1400-2400 Hz grid; sync_contrast = mean over the lines of the
    1130-1270 Hz share of the 900-2600 Hz energy in a 9 ms window on the sync pulse (0-9 ms) minus the same share
    averaged over five windows inside the Y segment (12-100 ms)."""
    m = ROBOT36
    a, b = (0, H) if lines is None else (max(0, int(lines[0])), min(H, int(lines[1])))
    if b <= a: return np.zeros((0, W // max(1, px_step))), 0.0
    x = np.asarray(x, np.float32)
    ty, _ = pixel_times(t_line0, period, params)
    ty = ty[a:b, ::max(1, int(px_step))]
    freqs = np.arange(1400.0, 2400.0 + 0.5 * f_step_hz, f_step_hz)
    P = tones.tone_power(x, fs, x_t0, ty, win_ms, freqs)
    f_est = np.asarray(tones.soft_estimate(P, freqs, None, T=1.0)[0], np.float64)
    Y = tones.to_level(f_est, m.f_black, m.f_white)
    # sync-column contrast
    ls = line_start_times(t_line0, period)[a:b]
    f_all = np.arange(900.0, 2600.0 + 1.0, 20.0)
    band = (f_all >= 1130.0) & (f_all <= 1270.0)
    t_sync = ls + m.sync_ms / 2000.0
    t_y = ls[:, None] + (np.array([20.0, 40.0, 60.0, 80.0, 95.0]) / 1000.0)[None, :]
    Ps = np.asarray(tones.tone_power(x, fs, x_t0, t_sync, m.sync_ms, f_all), np.float64)
    Pyy = np.asarray(tones.tone_power(x, fs, x_t0, t_y, m.sync_ms, f_all), np.float64)
    share_s = Ps[:, band].sum(-1) / (Ps.sum(-1) + 1e-30)
    share_y = Pyy[..., band].sum(-1) / (Pyy.sum(-1) + 1e-30)
    contrast = float(np.mean(share_s) - np.mean(share_y))
    return Y, contrast
