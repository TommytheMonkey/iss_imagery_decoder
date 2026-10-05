"""DC/Doppler removal, FM-threshold gate, band-pass. All functions are pure (return new arrays).

Units: `x` is the demodulator output as float32 instantaneous frequency deviation in Hz (rtl_fm WAV value x 48000).
Ported (logic copied, nothing imported) from the verified take-2 limiter pipeline:
  * `remove_dc`      <- take2/limiter/limiter_lib.remove_dc_robust  (running trimmed mean, |x - m| < 0.2 FS = 9600 Hz, 3 iterations)
  * `threshold_gate` <- take2/limiter/limiter_lib.preprocess_env    (det='in', M = 3, q = 2, L = 0.08 FS = 3840 Hz; 'wb_envin_M3_q2_L0.08')
  * `bandpass`       <- scripts/common.bandpass                      (6th-order Butterworth, sosfiltfilt)
The 'auto' decision of the gate is new; see `threshold_gate` for what is tested and why.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from .params import Params
from .tones import tone_power

# defaults of the tunables that are read from Params with getattr (so that an older Params/params_frozen.json still works)
_GATE_DEFAULTS = dict(gate_q=2.0, gate_block_s=2.0, gate_auto_gain_db=1.0, gate_core_max_ratio=2.0,
                      gate_test_win_ms=12.0, gate_test_hop_ms=6.0)
_TEST_F_LO, _TEST_F_HI, _TEST_F_STEP = 1100.0, 2400.0, 10.0     # tone grid of the contrast test (VIS/sync/black..white)
_DC_TRIM_HZ = 9600.0          # 0.2 FS at 48 kHz full scale (limiter_lib.remove_dc_robust L=0.2)
_DC_ITERS = 3


@dataclass
class GateInfo:
    mode_requested: str
    applied: bool
    level_hz: float
    M: int
    frac_above_level: float          # fraction of |x| > level before gating
    frac_zeroed: float               # fraction of samples driven to (near) zero (gate weight < 0.1, i.e. > 20 dB attenuation)
    excess_kurtosis_wideband: float
    two_population_score: float      # >1 means the FM-threshold mixture signature was detected (= contrast_gain_db / gain threshold)
    note: str = ""
    # --- additional statistics (all with defaults; added by the dsp implementation) ---
    q: float = 2.0                               # gate exponent: w = exp(-(e/L^2)^q)
    core_sigma_hz: float = float("nan")          # robust width 1.4826*MAD of x (whole signal)
    frac_above_level_gauss: float = float("nan") # fraction beyond level_hz expected for a Gaussian of width core_sigma_hz
    amp_tail_ratio: float = float("nan")         # frac_above_level / frac_above_level_gauss (the amplitude-only statistic; ~1 on the ISS file)
    scale_ok: bool = True                        # core_sigma_hz <= gate_core_max_ratio * level_hz
    contrast_gain_db: float = float("nan")       # decisive gate effect: P90 in-band tone contrast gated minus ungated (max of 3-block medians)
    contrast_gain_threshold_db: float = float("nan")
    contrast_ungated_db: float = float("nan")    # P90 contrast (peak/median, dB) of the ungated signal in the decisive block
    contrast_gated_db: float = float("nan")
    test_win_ms: float = float("nan")
    test_hop_ms: float = float("nan")
    block_s: float = float("nan")
    n_blocks: int = 0
    n_blocks_signature: int = 0                  # blocks whose smoothed gain exceeds the threshold
    frac_blocks_signature: float = 0.0
    mean_weight: float = 1.0                     # mean gate weight (1 = nothing attenuated)
    block_gains_db: list = field(default_factory=list)      # per-block contrast gain (rounded)
    block_contrast_db: list = field(default_factory=list)   # per-block ungated P90 contrast (rounded)


# ----------------------------------------------------------------------------------------------------- helpers
_RM_CHUNK = 1 << 21           # samples per chunk of the running mean (bounds scipy's double line buffers to ~16 MB)


def _running_mean(v: np.ndarray, n: int) -> np.ndarray:
    """centred boxcar mean over n samples (n odd), O(N), double accumulation, float32 out, edges reflected.
    Computed in chunks with n//2 samples of real context on each side (identical to the whole-array filter except that
    the reflection applies only at the true ends, where the chunk includes them): uniform_filter1d allocates two
    float64 line buffers of the array length, i.e. 0.5 GB per call on a 660 s recording."""
    v = np.asarray(v, np.float32)
    n = int(n); N = v.shape[0]
    if N <= _RM_CHUNK + n:
        return uniform_filter1d(v, size=n, mode="reflect", output=np.float32)
    out = np.empty(N, np.float32)
    h = n // 2
    for a in range(0, N, _RM_CHUNK):
        b = min(N, a + _RM_CHUNK)
        lo, hi = max(0, a - h), min(N, b + h)
        y = uniform_filter1d(v[lo:hi], size=n, mode="reflect", output=np.float32)
        out[a:b] = y[a - lo:b - lo]
    return out


def _gauss_two_sided_tail(k: float) -> float:
    """P(|z| > k) for a standard normal."""
    return math.erfc(k / math.sqrt(2.0))


def _excess_kurtosis(x: np.ndarray, chunk: int = 1 << 20) -> float:
    """excess kurtosis of x in float64 without a full float64 copy (chunked two-pass)."""
    n = x.size
    if n < 4:
        return float("nan")
    s = 0.0
    for i in range(0, n, chunk):
        s += float(np.sum(x[i:i + chunk], dtype=np.float64))
    mean = s / n
    m2 = 0.0
    m4 = 0.0
    for i in range(0, n, chunk):
        d = x[i:i + chunk].astype(np.float64) - mean
        d2 = d * d
        m2 += float(d2.sum())
        m4 += float((d2 * d2).sum())
    m2 /= n
    m4 /= n
    return float(m4 / (m2 * m2) - 3.0) if m2 > 0 else float("nan")


def _robust_sigma(v: np.ndarray) -> tuple[float, float]:
    """(median, 1.4826*MAD) of a 1-D array."""
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, 1.4826 * mad


def _running_median3(v: np.ndarray) -> np.ndarray:
    if v.size < 3:
        return v.copy()
    out = v.copy()
    out[1:-1] = np.median(np.stack([v[:-2], v[1:-1], v[2:]], axis=0), axis=0)
    return out


def _gate_weights(x: np.ndarray, L: float, M: int, q: float) -> np.ndarray:
    """w = exp(-(e/L^2)^q), e = centred moving average of x^2 over M samples (M forced odd), float32."""
    Mo = M + (M % 2 == 0) if M > 1 else 1
    e = np.multiply(x, x, dtype=np.float32)
    if Mo > 1:
        e = _running_mean(e, Mo)                     # the squared signal is released here
    np.divide(e, np.float32(L * L), out=e)
    np.power(e, np.float32(q), out=e)
    np.negative(e, out=e)
    np.exp(e, out=e)
    return e


def _frame_contrast_db(x: np.ndarray, fs: int, win_ms: float, hop_ms: float) -> np.ndarray:
    """10log10(peak/median) of the Hann-window power spectrum of every frame on the SSTV tone grid (1100-2400 Hz, 10 Hz).
    Scale-free; frames at hop_ms, fully inside the signal."""
    freqs = np.arange(_TEST_F_LO, _TEST_F_HI + 0.5 * _TEST_F_STEP, _TEST_F_STEP)
    w = int(round(win_ms * fs / 1000.0))
    w += (w % 2 == 0)
    hop = max(int(round(hop_ms * fs / 1000.0)), 1)
    if x.size <= w:
        centres = np.array([(x.size - 1) / 2.0 / fs])
    else:
        centres = (np.arange(0, x.size - w + 1, hop) + w // 2) / fs
    P = tone_power(x, fs, 0.0, centres, win_ms, freqs)
    pk = P.max(axis=-1).astype(np.float64)
    med = np.median(P, axis=-1).astype(np.float64)
    return 10.0 * np.log10(pk / (med + 1e-30) + 1e-30)


def _contrast_gain(x: np.ndarray, y: np.ndarray, fs: int, block_s: float, win_ms: float, hop_ms: float) -> dict:
    """Gate-effect test. Per block of block_s: the 90th percentile of the per-frame in-band tone contrast of the gated (y) and the
    ungated (x) signal; gain = gated - ungated. Decisive gain = max over blocks of the 3-block running median (so a 36 s frame inside
    a 10 min file is found and a single odd block cannot decide). Returns the decisive values and the per-block arrays."""
    q0 = _frame_contrast_db(x, fs, win_ms, hop_ms)
    q1 = _frame_contrast_db(y, fs, win_ms, hop_ms)
    fpb = max(int(round(block_s * 1000.0 / hop_ms)), 1)
    nb = q0.size // fpb
    if nb >= 1:
        Q0 = q0[:nb * fpb].reshape(nb, fpb)
        Q1 = q1[:nb * fpb].reshape(nb, fpb)
    else:
        Q0, Q1, nb = q0[None, :], q1[None, :], 1
    p0 = np.percentile(Q0, 90, axis=1)
    p1 = np.percentile(Q1, 90, axis=1)
    g = p1 - p0
    smooth = _running_median3(g) if nb >= 3 else np.full(nb, float(np.median(g)))
    i = int(np.argmax(smooth))
    return dict(gain=float(smooth[i]), p0=float(p0[i]), p1=float(p1[i]), gains=g, p0_all=p0, smooth=smooth, n_blocks=nb)


# ----------------------------------------------------------------------------------------------------- public API
def remove_dc(x: np.ndarray, fs: int, window_s: float = 0.5, trim_hz: float = _DC_TRIM_HZ, iters: int = _DC_ITERS) -> np.ndarray:
    """Subtract a running trimmed mean (e.g. 20 % trimmed, computed on decimated blocks and interpolated) so the
    Doppler-driven DC of an FM discriminator output is removed without affecting the 1.1-2.3 kHz tones.

    Implementation (ported from take2/limiter remove_dc_robust, verified on the ISS recording): the trimmed mean is a
    running mean over `window_s` of the samples within +-trim of the current estimate, iterated `iters` times, so it
    tracks the centre of the carrier-captured hump of a below-threshold discriminator output and ignores the broad
    +-fs/2 noise floor (whose centre is 0 Hz, not the Doppler offset). trim = max(trim_hz, 1.4826*MAD(x - running mean)),
    i.e. for a signal whose spread is larger than trim_hz (generic audio WAV scaling) the mask is all-ones and this is a
    plain running mean. The running means are exact per-sample boxcars (O(N), double accumulation), no decimation needed.
    Returns float32 of the same length. Pure: `x` is not modified."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.copy()
    n = int(round(window_s * fs))
    n += (n % 2 == 0)
    n = max(n, 3)
    if n >= x.size:                       # shorter than one window: a global trimmed mean
        n = x.size - (1 - x.size % 2)
        n = max(n, 1)
    # memory: every full-length temporary is float32 and reused (d, inside, num); a 660 s recording costs 4 copies, not 8
    m = _running_mean(x, n)
    d = np.subtract(x, m, dtype=np.float32)
    _, s_rob = _robust_sigma(d)
    trim = float(max(trim_hz, s_rob))
    inside = np.empty_like(x)
    for _ in range(int(iters)):
        np.subtract(x, m, out=d)
        np.abs(d, out=d)
        inside[:] = (d < np.float32(trim))
        np.multiply(x, inside, out=d)
        num = _running_mean(d, n)
        den = _running_mean(inside, n)
        np.maximum(den, np.float32(1e-6), out=inside)        # inside is free now: reuse it for the denominator
        np.divide(num, inside, out=num)
        np.copyto(m, num, where=(den > 0.05))
    np.subtract(x, m, out=d)
    return d


def threshold_gate(x: np.ndarray, fs: int, params: Params, mode: str | None = None) -> tuple[np.ndarray, GateInfo]:
    """FM-threshold gate (from take2/limiter, verified): e = moving average of x^2 over M samples;
    w = exp(-(e / L^2)^2) with L = params.gate_level_hz; return x*w.
    mode 'on': always apply; 'off': never; 'auto': apply only when the two-population signature is present:
    compare the fraction of samples with |x| > L against the fraction expected from a Gaussian with the robust core
    width (e.g. 1.4826*MAD of x); if the observed fraction exceeds the expected by a large factor (signature of the
    uniform +-fs/2 noise floor of a below-threshold discriminator), apply. Record everything in GateInfo.

    What 'auto' actually tests, and why (measured on the ISS recording, see .scratch/dsp/notes.md):
      The amplitude-only statistic above is REPORTED (`frac_above_level`, `frac_above_level_gauss`, `amp_tail_ratio`) but
      cannot decide: at CNR -2 dB the de-emphasised and the pre-de-emphasis amplitude distributions of the recording are
      Gaussian-shaped to within 10 % (amp_tail_ratio 0.94-1.01, excess kurtosis -0.2..+0.3), because the captured-carrier
      hump (sigma ~7.5 kHz pre-de-emphasis) is only ~2x narrower than the +-fs/2 floor. The two populations of a
      below-threshold discriminator are *tone-bearing* samples (carrier captured) and *tone-free* noise bursts, so the
      signature is tested where it lives: gating the bursts raises the in-band tone contrast.
      * w = gate weights; per 2 s block (params.gate_block_s) the 90th percentile of the per-frame contrast
        10log10(peak/median) of 12 ms Hann spectra on the 1100-2400 Hz tone grid (scale-free), for x*w and for x;
        gain = gated - ungated; decisive gain = max over the file of the 3-block running median.
      * two_population_score = decisive gain / params.gate_auto_gain_db (default 1.0 dB); signature if score > 1
        (ISS recording: +3.5 dB in the frame, rtl_fm simulation +2..+6 dB for CNR -6..+2 dB; Gaussian noise, no-carrier
        floor, clean or noisy generic audio: -49..+0.2 dB, the null extreme over 330 blocks is ~+0.3 dB).
      * scale veto: 1.4826*MAD(x) must not exceed params.gate_core_max_ratio x L (default 2 x 3840 = 7680 Hz): the gate is
        calibrated for the de-emphasised rtl_fm scale (floor rms 0.105 FS = 5000 Hz); a generic audio WAV at 0.5 FS or a
        non-de-emphasised discriminator output (robust width 15 kHz) is refused with an explanatory note.
      The decision is per file: when the signature is found anywhere the whole signal is gated (as in the verified pipeline,
      where controls and no-signal stretches were gated identically). `mode` overrides params.gate_mode. Pure."""
    x = np.asarray(x, dtype=np.float32)
    L = float(params.gate_level_hz)
    M = int(params.gate_M)
    g = {k: getattr(params, k, v) for k, v in _GATE_DEFAULTS.items()}
    q = float(g["gate_q"])
    block_s = float(g["gate_block_s"])
    gain_min = float(g["gate_auto_gain_db"])
    core_max = float(g["gate_core_max_ratio"])
    t_win = float(g["gate_test_win_ms"])
    t_hop = float(g["gate_test_hop_ms"])
    mode = (mode if mode is not None else params.gate_mode).lower()
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"gate mode must be auto|on|off, got {mode!r}")

    info = GateInfo(mode_requested=mode, applied=False, level_hz=L, M=M, frac_above_level=float("nan"), frac_zeroed=0.0,
                    excess_kurtosis_wideband=float("nan"), two_population_score=float("nan"), q=q,
                    contrast_gain_threshold_db=gain_min, test_win_ms=t_win, test_hop_ms=t_hop, block_s=block_s)
    if x.size == 0:
        info.note = "empty input"
        return x.copy(), info

    # ---- amplitude statistics (reported in every mode)
    info.frac_above_level = float(np.mean(np.abs(x) > L))
    info.excess_kurtosis_wideband = _excess_kurtosis(x)
    _, sig = _robust_sigma(x)
    info.core_sigma_hz = float(sig)
    if sig > 0:
        info.frac_above_level_gauss = _gauss_two_sided_tail(L / sig)
        info.amp_tail_ratio = info.frac_above_level / max(info.frac_above_level_gauss, 1e-12)
    info.scale_ok = bool(sig <= core_max * L)

    # ---- the gate weights (limiter_lib.preprocess_env, det='in') and the gate-effect test
    w = _gate_weights(x, L, M, q)
    y = np.multiply(x, w, dtype=np.float32)
    ct = _contrast_gain(x, y, fs, block_s, t_win, t_hop)
    info.contrast_gain_db = ct["gain"]
    info.contrast_ungated_db = ct["p0"]
    info.contrast_gated_db = ct["p1"]
    info.n_blocks = int(ct["n_blocks"])
    sel = ct["smooth"] > gain_min
    info.n_blocks_signature = int(sel.sum())
    info.frac_blocks_signature = float(sel.mean())
    info.block_gains_db = [float(round(float(v), 2)) for v in ct["gains"]]
    info.block_contrast_db = [float(round(float(v), 2)) for v in ct["p0_all"]]
    info.two_population_score = info.contrast_gain_db / gain_min if gain_min > 0 else float("inf")
    signature = bool(np.isfinite(info.two_population_score) and info.two_population_score > 1.0)

    # ---- decision
    if mode == "on":
        apply = True
        info.note = f"forced on (contrast gain {info.contrast_gain_db:+.2f} dB, scale_ok={info.scale_ok})"
    elif mode == "off":
        apply = False
        info.note = f"forced off (contrast gain {info.contrast_gain_db:+.2f} dB, scale_ok={info.scale_ok})"
    else:
        apply = signature and info.scale_ok
        if signature and not info.scale_ok:
            info.note = (f"gate would raise the tone contrast by {info.contrast_gain_db:+.2f} dB but the robust width "
                         f"{info.core_sigma_hz:.0f} Hz exceeds {core_max:g} x L = {core_max * L:.0f} Hz: the input is not "
                         f"de-emphasised FM-discriminator audio at the expected scale, gate not applied")
        elif signature:
            info.note = (f"FM-threshold signature: gating raises the in-band tone contrast by {info.contrast_gain_db:+.2f} dB "
                         f"(threshold {gain_min:g} dB) in {info.n_blocks_signature}/{info.n_blocks} blocks: gate applied")
        else:
            info.note = (f"no FM-threshold signature: gating changes the in-band tone contrast by {info.contrast_gain_db:+.2f} dB "
                         f"(threshold {gain_min:g} dB): gate not applied")

    if not apply:
        info.applied = False
        info.frac_zeroed = 0.0
        info.mean_weight = 1.0
        del y, w
        return x.copy(), info
    info.applied = True
    info.frac_zeroed = float(np.mean(w < 0.1))
    info.mean_weight = float(w.mean(dtype=np.float64))
    del w
    return y, info


def bandpass(x: np.ndarray, fs: int, lo: float, hi: float, order: int = 6, chunk_s: float = 60.0, overlap_s: float = 2.0) -> np.ndarray:
    """Zero-phase Butterworth band-pass (sosfiltfilt), float32 out.

    Long signals are filtered in chunks of chunk_s with overlap_s of context on each side (the impulse response of the
    6th-order 400-3200 Hz Butterworth decays within ~20 ms, so the result equals the whole-signal filter to float precision)
    because sosfiltfilt works in float64 and holds several copies of its input (3 GB for a 660 s recording)."""
    x = np.asarray(x)
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    n = int(x.shape[0]); nc = int(chunk_s * fs); ov = int(overlap_s * fs)
    if n <= nc + 2 * ov:
        return sosfiltfilt(sos, x).astype(np.float32)
    out = np.empty(n, np.float32)
    for a in range(0, n, nc):
        b = min(n, a + nc)
        lo_i, hi_i = max(0, a - ov), min(n, b + ov)
        y = sosfiltfilt(sos, x[lo_i:hi_i])
        out[a:b] = y[a - lo_i:b - lo_i]
    return out
