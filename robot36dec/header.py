"""VIS header detection and decoding (normal path).

Two-stage search ported from the verified analysis (scripts/04_header_vis.py, scripts/06_timing.py and the adversarial
re-test take2/skepticA/s1_vis_header.py):

  stage 1  a code-independent template of the whole 910 ms VIS header (leader 1900 Hz 300 ms, break 1200 Hz 10 ms,
           leader 1900 Hz 300 ms, start bit 1200 Hz 30 ms, eight data/parity bits in the 1100-or-1300 Hz band, stop bit
           1200 Hz 30 ms) is correlated with robust-z-scored log band-energy ratios framed at params.vis_hop_ms hop over
           the whole signal (unit-variance normalisation: sum of z over template frames / sqrt(N)); the z traces are
           zero-padded ("no information") so that a header touching either end of the signal is still found;
  stage 2  for each peak (>= 1 s apart): joint search over all 128 VIS codes x alignment with the full VIS template at
           1 ms hop, bit decoding by sampling 24 ms windows at the bit centres while scanning the sampling offset over
           +-params.vis_bit_offset_scan_ms (stability plateau), parity (with the joint matched filter as arbiter when the
           plateau code fails parity), mode name, bit tone frequencies, false-alarm statistics (z and noise maximum of the
           stage-1 score excluding +-2 s around the candidate) and a calibrated confidence in [0, 1].

Only the audio decides: no reference images, no randomness, no file access.
"""
from dataclasses import dataclass, field
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks
from .params import Params
from .modes import VIS_TABLE, VIS_LEADER, VIS_BIT_MS, VIS_TOTAL_MS

F_ONE, F_ZERO, F_SYNC, F_LEADER = 1100.0, 1300.0, 1200.0, 1900.0
TONE_BANDS = (F_ONE, F_SYNC, F_ZERO, F_LEADER)
LOW = "low"                                   # 1100 Hz OR 1300 Hz: code-independent data-bit evidence
CONTRAST = "contrast"                         # per-frame spectral contrast (dB), carried in the band-trace dict, not a band
BAND_HALF_HZ = 50.0
TOTAL_BAND = (900.0, 2600.0)
VIS_START_MS = float(sum(ms for _, ms in VIS_LEADER[:3]))      # 610 ms: start bit begins here
N_ELEMENTS = 10                               # start bit, 7 data bits, parity bit, stop bit
BIT_WIN_MS = 24.0                             # sampling window for the bit decoder
BIT_SCAN_STEP_MS = 2.0
EXCLUDE_S = 2.0                               # +-2 s around a candidate excluded from the false-alarm statistics
MF_ADOPT_MARGIN = 0.5                         # joint-MF code margin needed to override a parity-failed plateau code
PARITY_FAIL_CAP = 0.5                         # confidence cap when the reported bits fail parity
NOISE_REF_MIN_FRAMES = 200                    # band ratios are z-scored against the no-tone frames when at least this many exist
Z_FLOOR = -2.0                                # a frame can count at most this much AGAINST a template element (bounded influence)


@dataclass
class VISCandidate:
    t_leader: float            # absolute start of first 1900 Hz leader (s)
    t_line0: float             # absolute end of stop bit = image line 0 start (s)
    score: float               # matched-filter score at the peak
    z: float                   # (score - median)/robust sd over the rest of the file
    noise_max: float           # maximum score elsewhere (> 2 s away)
    bits: list = field(default_factory=list)      # 8 decoded bits LSB first (7 data + parity)
    vis_code: int | None = None
    parity_ok: bool | None = None
    mode_name: str = "unknown"
    bit_freqs_hz: list = field(default_factory=list)
    plateau_ms: float = 0.0    # width of the bit-sampling-offset range over which the decoded code is stable
    confidence: float = 0.0    # 0..1
    # ---- extra diagnostics (all optional, defaults keep the contract above intact) ----
    detection_confidence: float = 0.0    # header-exists evidence alone (independent of the code), 0..1
    code_reliable: bool = False          # confidence >= 0.6 (the timing resolver's credibility threshold)
    code_available: bool = False         # all ten bit windows lie inside the signal
    code_source: str = "none"            # 'plateau' | 'matched-filter' | 'none'
    decoded_code: int | None = None      # reported code even when vis_code is suppressed as unreliable
    plateau_bits: list = field(default_factory=list)      # bits measured at the plateau centre (before any arbitration)
    mf_code: int | None = None           # code preferred by the joint 128-code matched filter at the refined alignment
    mf_margin: float = 0.0               # full-template score gap (1 ms hop, unit-variance units) to the 2nd best code
    score_full: float = 0.0              # refined full-VIS matched-filter score (1 ms hop, decoded code)
    z_full: float = 0.0                  # z of the decoded-code full template over the whole file (stage-1 hop), excl. +-2 s
    t_leader_coarse: float = 0.0         # stage-1 peak time before refinement (s)
    bit_llr_db: list = field(default_factory=list)       # 8 x 10log10(E1100/E1300), averaged over the plateau
    bit_freqs_nominal_hz: list = field(default_factory=list)   # nominal tone of each of the 10 elements for the reported bits
    n_freq_ok: int = 0                   # elements with |f_measured - f_nominal| < 60 Hz (of 10)
    plateau_offset_ms: float = 0.0       # sampling offset (ms) at the plateau centre relative to the refined alignment
    notes: str = ""


# ----------------------------------------------------------------------------------------------- helpers
def vis_bits(code: int) -> list[int]:
    """7 data bits LSB first + even parity bit."""
    b = [(int(code) >> i) & 1 for i in range(7)]
    return b + [sum(b) % 2]


def vis_mode_name(code: int | None) -> str:
    if code is None:
        return "unknown"
    return VIS_TABLE.get(int(code), f"VIS {int(code)} (unassigned)")


def _vis_elements(code: int | None):
    """[(band, start_ms, end_ms)] of the 910 ms header. code None -> data bits as one code-independent LOW-band element."""
    els, pos = [], 0.0
    for fr, ms in VIS_LEADER:
        els.append((float(fr), pos, pos + ms)); pos += ms
    data_start = pos                                   # 610 ms
    if code is None:
        els.append((LOW, data_start, data_start + 8 * VIS_BIT_MS))
    else:
        for i, b in enumerate(vis_bits(code)):
            els.append((F_ONE if b else F_ZERO, data_start + i * VIS_BIT_MS, data_start + (i + 1) * VIS_BIT_MS))
    stop_start = data_start + 8 * VIS_BIT_MS          # 880 ms
    els.append((F_SYNC, stop_start, stop_start + VIS_BIT_MS))
    assert abs(stop_start + VIS_BIT_MS - VIS_TOTAL_MS) < 1e-9
    return els


def _band_traces(x: np.ndarray, fs: int, win_ms: float, hop_ms: float, nfft: int = 4096, chunk: int = 1024):
    """Hann-framed band energies. Returns (t_rel centre times (s, relative to x[0]), {band: E}, E_total 900-2600 Hz).
    E also carries the per-frame spectral contrast under the key CONTRAST (10 log10 of the peak-to-median bin power over
    900-2600 Hz): a frame without a tone has a contrast of ~5-8 dB, a tone frame > 12 dB. The chunk is 1024 frames so
    that the complex128 FFT buffer stays below ~35 MB on long recordings."""
    w = int(round(win_ms * fs / 1000.0)); h = max(1, int(round(hop_ms * fs / 1000.0)))
    n = 1 + (len(x) - w) // h if len(x) >= w else 0
    f = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = (f >= TOTAL_BAND[0]) & (f <= TOTAL_BAND[1]); fsel = f[sel]
    masks = {fc: np.abs(fsel - fc) <= BAND_HALF_HZ for fc in TONE_BANDS}
    E = {fc: np.empty(n, np.float32) for fc in TONE_BANDS}; Etot = np.empty(n, np.float32)
    con = np.empty(n, np.float32)
    if n > 0:
        win = np.hanning(w).astype(np.float32)
        xf = np.ascontiguousarray(x, dtype=np.float32)
        frames = sliding_window_view(xf, w)[::h][:n]
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            S = np.fft.rfft(frames[s0:s1] * win, n=nfft, axis=1)[:, sel]
            P = (S.real ** 2 + S.imag ** 2)
            Etot[s0:s1] = P.sum(1)
            con[s0:s1] = 10.0 * np.log10(P.max(1) / (np.median(P, axis=1) + 1e-30) + 1e-30)
            for fc in TONE_BANDS:
                E[fc][s0:s1] = P[:, masks[fc]].sum(1)
    E[LOW] = E[F_ONE] + E[F_ZERO]
    E[CONTRAST] = con
    t = (np.arange(n) * h + (w - 1) / 2.0) / fs
    return t, E, Etot


def _robust_z(r: np.ndarray):
    med = float(np.median(r)); mad = 1.4826 * float(np.median(np.abs(r - med))) + 1e-9
    return ((r - med) / mad).astype(np.float32), med, mad


def _ratio_z(E: dict, Etot: np.ndarray, noise_contrast_db: float = 12.0) -> dict:
    """r_f = 10 log10(E_f / E_tot), robust z-scored. Frames without energy (digital silence, gated) get z = 0.

    Reference of the z-score: the NO-TONE frames (spectral contrast below noise_contrast_db) when at least
    NOISE_REF_MIN_FRAMES of them exist, else all frames. On a recording that is mostly receiver noise (the reference
    recording) both references coincide; on a clean recording that is mostly picture the file median of a band that the
    picture never visits (1100-1300 Hz) sits ~20 dB below the noise level, so that every carrier-off stretch used to score
    as 'tone present' in all four bands at once and inflated the false-alarm statistics (header z fell from 19 at CNR 0 dB
    to 6 at +20 dB). z is floored at Z_FLOOR so that picture frames cannot count unboundedly against a template element."""
    med_tot = float(np.median(Etot)) if len(Etot) else 0.0
    floor = 1e-6 * med_tot if med_tot > 0 else 1e-30
    valid = Etot > floor
    con = E.get(CONTRAST)
    noise_like = (valid & (np.asarray(con) < noise_contrast_db)) if con is not None else np.zeros(len(Etot), bool)
    use_noise = int(noise_like.sum()) >= NOISE_REF_MIN_FRAMES
    Z = {}
    for band, e in E.items():
        if band == CONTRAST:
            continue
        r = 10.0 * np.log10((e.astype(np.float64) + 1e-3 * floor) / (Etot.astype(np.float64) + 1e-3 * floor))
        if valid.any() and not valid.all():
            r[~valid] = np.median(r[valid])
        ref = r[noise_like] if use_noise else r
        med = float(np.median(ref)); mad = 1.4826 * float(np.median(np.abs(ref - med))) + 1e-9
        Z[band] = np.maximum((r - med) / mad, Z_FLOOR).astype(np.float32)
    Z["_noise_reference"] = use_noise
    return Z


def _pad_z(Z: dict, pad: int) -> dict:
    """Zero-pad ('no information') every z trace by `pad` frames on both sides."""
    zp = np.zeros(pad, np.float32)
    return {b: np.concatenate([zp, z, zp]) for b, z in Z.items() if isinstance(z, np.ndarray)}


def _template_frames(elements, hop_ms: float, win_ms: float):
    """Frame index ranges [ja, jb) per element (frame j is centred (j+0.5)*hop_ms after the leader start). Only frames whose
    window lies mostly inside the element (margin win/4) are used; elements shorter than the window are skipped."""
    margin = 0.25 * win_ms
    out = []
    for band, a, b in elements:
        if b - a < win_ms - 1e-9:
            continue
        lo, hi = a + margin, b - margin
        ja = int(np.ceil(lo / hop_ms - 0.5 - 1e-9)); jb = int(np.floor(hi / hop_ms - 0.5 + 1e-9)) + 1
        if jb > ja:
            out.append((band, ja, jb))
    return out


def _template_len(hop_ms: float, win_ms: float) -> int:
    """Number of hop frames spanned by the full header template (all 128 codes have the same span)."""
    return max(jb for _, _, jb in _template_frames(_vis_elements(None), hop_ms, win_ms))


def _cumsums(Z: dict) -> dict:
    return {b: np.concatenate([[0.0], np.cumsum(z.astype(np.float64))]) for b, z in Z.items() if isinstance(z, np.ndarray)}


def _element_sums(C: dict, band, ja: int, jb: int, npos: int) -> np.ndarray:
    """sum_{j=ja}^{jb-1} z_band[k + j] for template start k = 0..npos-1."""
    c = C[band]
    return c[jb:jb + npos] - c[ja:ja + npos]


def _mf_score(Z: dict, tf, n_frames: int):
    """Unit-variance matched filter: S(k) = sum over template frames of z_band[k+j] / sqrt(N). Returns (S, N); S has one
    entry per template start frame k for which the whole template fits inside the (padded) trace."""
    L = max(jb for _, _, jb in tf); npos = n_frames - L + 1
    if npos <= 0:
        return np.zeros(0), 0
    C = _cumsums(Z); acc = np.zeros(npos, np.float64); N = 0
    for band, ja, jb in tf:
        acc += _element_sums(C, band, ja, jb, npos); N += jb - ja
    return acc / np.sqrt(max(N, 1)), N


def _joint_code_scores(Z: dict, hop_ms: float, win_ms: float, n_frames: int):
    """Full-VIS matched filter for all 128 codes at once: (128, npos) array, unit-variance units, and npos."""
    fixed = _vis_elements(None)[:4] + _vis_elements(None)[-1:]           # leaders, break, start, stop
    tf_fixed = _template_frames(fixed, hop_ms, win_ms)
    bit_els = [(None, VIS_START_MS + VIS_BIT_MS * (1 + i), VIS_START_MS + VIS_BIT_MS * (2 + i)) for i in range(8)]
    tf_bits = [_template_frames([(F_ZERO, a, b)], hop_ms, win_ms) for _, a, b in bit_els]
    L = max([jb for _, _, jb in tf_fixed] + [jb for tf in tf_bits for _, _, jb in tf])
    npos = n_frames - L + 1
    if npos <= 0:
        return np.zeros((128, 0)), 0
    C = _cumsums(Z); N = 0
    base = np.zeros(npos, np.float64)
    for band, ja, jb in tf_fixed:
        base += _element_sums(C, band, ja, jb, npos); N += jb - ja
    A = np.zeros((8, npos), np.float64)                                   # 1100 minus 1300 per bit slot
    for i, tf in enumerate(tf_bits):
        for _, ja, jb in tf:
            base += _element_sums(C, F_ZERO, ja, jb, npos)
            A[i] += _element_sums(C, F_ONE, ja, jb, npos) - _element_sums(C, F_ZERO, ja, jb, npos)
            N += jb - ja
    bits = np.array([vis_bits(c) for c in range(128)], np.float64)        # (128, 8)
    S = (base[None, :] + bits @ A) / np.sqrt(max(N, 1))
    return S, npos


def _element_spectra(x: np.ndarray, fs: int, x_t0: float, centres: np.ndarray, win_ms: float, nfft: int = 16384):
    """Power spectra (900-2600 Hz) of Hann windows centred at absolute times `centres`; samples outside x count as zero.
    Returns (f, P[len(centres), nb])."""
    w = int(round(win_ms * fs / 1000.0))
    xp = np.concatenate([np.zeros(w, np.float32), np.asarray(x, np.float32), np.zeros(w, np.float32)])
    idx = np.round((centres.ravel() - x_t0) * fs).astype(np.int64)[:, None] - w // 2 + np.arange(w)[None, :] + w
    idx = np.clip(idx, 0, len(xp) - 1)
    fr = xp[idx] * np.hanning(w).astype(np.float32)[None, :]
    f = np.fft.rfftfreq(nfft, 1.0 / fs); sel = (f >= TOTAL_BAND[0]) & (f <= TOTAL_BAND[1])
    S = np.fft.rfft(fr, n=nfft, axis=1)[:, sel]
    return f[sel], (S.real ** 2 + S.imag ** 2).astype(np.float64)


def _peak_freq(f: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Dominant frequency per spectrum with parabolic interpolation (on log power)."""
    L = np.log(P + 1e-30); k = np.argmax(L, axis=1); kc = np.clip(k, 1, L.shape[1] - 2); rows = np.arange(L.shape[0])
    a, b, c = L[rows, kc - 1], L[rows, kc], L[rows, kc + 1]
    den = a - 2 * b + c; den = np.where(np.abs(den) < 1e-12, -1e-12, den)
    delta = np.clip(0.5 * (a - c) / den, -1, 1)
    return f[kc] + delta * (f[1] - f[0])


def _runs_of_equal(v: np.ndarray):
    """[(start, stop_exclusive)] of maximal runs of equal values."""
    runs, i = [], 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[j + 1] == v[i]:
            j += 1
        runs.append((i, j + 1)); i = j + 1
    return runs


def _decode_bits(x: np.ndarray, fs: int, x_t0: float, t_leader: float, scan_ms: float):
    """Sampling-offset scan of the 10 VIS elements (start, 7 data, parity, stop) with 24 ms windows; stability plateau.
    Returns a dict with the plateau bits, code, parity, plateau width, LLRs and dominant tone frequencies (spectra averaged
    over the plateau offsets)."""
    offsets = np.arange(-scan_ms, scan_ms + 1e-9, BIT_SCAN_STEP_MS) / 1000.0
    t_start = t_leader + VIS_START_MS / 1000.0
    cen = t_start + offsets[:, None] + (VIS_BIT_MS / 2000.0) + (VIS_BIT_MS / 1000.0) * np.arange(N_ELEMENTS)[None, :]
    f, P = _element_spectra(x, fs, x_t0, cen, BIT_WIN_MS)
    P = P.reshape(len(offsets), N_ELEMENTS, -1)
    m11 = (f >= F_ONE - BAND_HALF_HZ) & (f < F_ONE + BAND_HALF_HZ)
    m13 = (f >= F_ZERO - BAND_HALF_HZ) & (f < F_ZERO + BAND_HALF_HZ)
    llr = 10.0 * np.log10((P[:, 1:9, m11].sum(-1) + 1e-30) / (P[:, 1:9, m13].sum(-1) + 1e-30))   # (n_off, 8)
    bits = (llr > 0).astype(int)
    pattern = (bits * (1 << np.arange(8))[None, :]).sum(1)                                        # all 8 bits as int
    runs = _runs_of_equal(pattern)
    # longest run; ties -> the run with the larger mean |LLR| (more decisive bits)
    best = max(runs, key=lambda r: (r[1] - r[0], float(np.abs(llr[r[0]:r[1]]).mean())))
    i0, i1 = best; ic = (i0 + i1) // 2
    sel = slice(i0, i1)
    bits_c = bits[ic].tolist(); code = int(sum(b << i for i, b in enumerate(bits_c[:7])))
    parity_ok = bool((sum(bits_c[:7]) % 2) == bits_c[7])
    return dict(bits=bits_c, code=code, parity_ok=parity_ok, plateau_ms=float((i1 - i0) * BIT_SCAN_STEP_MS),
                plateau_offset_ms=float(offsets[ic] * 1000.0), llr=[float(v) for v in llr[sel].mean(0)],
                freqs=[float(v) for v in _peak_freq(f, P[sel].mean(0))])


def _nominal_and_agreement(bits: list, freqs: list):
    nominal = [F_SYNC] + [F_ONE if b else F_ZERO for b in bits] + [F_SYNC]
    n_ok = int(np.sum(np.abs(np.array(freqs) - np.array(nominal)) < 60.0)) if len(freqs) == len(nominal) else 0
    return [float(v) for v in nominal], n_ok


def _false_alarm_stats(t_pos: np.ndarray, S: np.ndarray, t_cand: float, exclude_s: float = EXCLUDE_S):
    """(median, robust sd, max) of the score trace excluding +-exclude_s around the candidate."""
    far = np.abs(t_pos - t_cand) > exclude_s
    if far.sum() < 10:
        far = np.ones_like(far, bool)
    s = S[far]
    med = float(np.median(s)); sd = 1.4826 * float(np.median(np.abs(s - med))) + 1e-9
    return med, sd, float(s.max())


def _sigmoid(v: float) -> float:
    return float(1.0 / (1.0 + np.exp(-v)))


def _confidence(z: float, z_full: float, vis_min_z: float, code_available: bool, parity_ok: bool, plateau_ms: float,
                n_freq_ok: int, mf_agree: bool, mf_margin: float) -> tuple[float, float]:
    """Calibrated combination of detection evidence and code evidence; returns (confidence, detection_confidence).
    Detection: z_det = mean of the code-independent z and the decoded-code z (both vs the rest of the file), mapped by a
    logistic centred 1.5 sd above params.vis_min_z (with the default 6 the 50 % point is z_det = 7.5; pure-noise maxima
    measured on the reference recording's 36 s noise windows are 2.6-5.1, the true header gives 15-23).
    Code: parity (weak: correlated noisy bit decisions satisfy it > 50 % of the time), stability plateau of the offset scan
    (noise 4-12 ms, real header 16-26 ms), dominant-tone agreement of the 10 elements with the decoded bits (noise <= 5/10,
    real 10/10) and the joint matched filter's code margin (noise <= 0.5, real 1.8-2.7) when it agrees with the reported
    code. confidence = c_det * (0.25 + 0.75 * c_code): a strong detection with no usable code never exceeds 0.25, and a code
    that fails parity is capped at PARITY_FAIL_CAP (below the 0.6 credibility threshold)."""
    z_det = 0.5 * (float(z) + float(z_full))
    c_det = _sigmoid((z_det - (vis_min_z + 1.5)) / 1.0)
    if not code_available:
        return float(np.clip(0.25 * c_det, 0.0, 1.0)), c_det
    c_par = 1.0 if parity_ok else 0.0
    c_plat = float(np.clip((plateau_ms - 8.0) / 8.0, 0.0, 1.0))          # >= 16 ms -> 1
    c_freq = float(np.clip((n_freq_ok - 5.0) / 4.0, 0.0, 1.0))           # 9 of 10 -> 1, 8 of 10 -> 0.75
    c_mf = float(np.clip((mf_margin - 0.3) / 1.2, 0.0, 1.0)) if mf_agree else 0.0
    c_code = 0.15 * c_par + 0.25 * c_plat + 0.35 * c_freq + 0.25 * c_mf
    conf = c_det * (0.25 + 0.75 * c_code)
    if not parity_ok:
        conf = min(conf, PARITY_FAIL_CAP)
    return float(np.clip(conf, 0.0, 1.0)), c_det


def vis_score_trace(x: np.ndarray, fs: int, x_t0: float, params: Params, code: int | None = None):
    """Diagnostic helper (for plots): the stage-1 matched-filter score trace over the whole signal.
    Returns (t_leader_positions_abs_s, score) for every template position whose span lies inside the signal.
    code None -> the code-independent detection template, otherwise the full template of that VIS code.
    Same framing as find_vis (params.vis_frame_ms / params.vis_hop_ms)."""
    hop_ms = float(params.vis_hop_ms); win_ms = float(params.vis_frame_ms)
    t_rel, E, Etot = _band_traces(np.asarray(x), int(fs), win_ms, hop_ms)
    Z = _ratio_z(E, Etot, float(getattr(params, "vis_noise_contrast_db", 12.0)))
    S, _ = _mf_score(Z, _template_frames(_vis_elements(code), hop_ms, win_ms), len(t_rel))
    return x_t0 + t_rel[:len(S)] - 0.5 * hop_ms / 1000.0, S


# ----------------------------------------------------------------------------------------------- main entry
def find_vis(x: np.ndarray, fs: int, x_t0: float, params: Params, max_candidates: int = 5,
             refine_win_ms: float = 10.0, refine_span_ms: float = 150.0, local_pad_s: float = 3.0) -> list[VISCandidate]:
    """1. Frame the band-passed audio (params.vis_frame_ms windows, params.vis_hop_ms hop); band energies at
       1100/1200/1300/1900 Hz (+-50 Hz) and total 900-2600 Hz; r_f = 10log10(E_f/E_tot) robust-z-scored over the file.
    2. Template of the leader/break/leader/start sequence (1900:300 ms, 1200:10, 1900:300, 1200:30) correlated with the
       z-scored ratios (fftconvolve); peaks separated by >= 1 s; keep top max_candidates.
    3. For each peak: refine at 1 ms hop over +-100 ms with the FULL VIS template for the best-scoring bit pattern
       (try all 128 codes or decode bits first), then decode the 8 bits by sampling 24 ms windows at bit centres while
       scanning the sampling offset over +-params.vis_bit_offset_scan_ms in 2 ms steps; the decoded code must be stable
       over a plateau >= 10 ms; record bit frequencies at the plateau centre; parity check; mode name from VIS_TABLE.
    4. z and noise_max computed from the score trace excluding +-2 s around the peak; confidence combines z (>= vis_min_z),
       parity, plateau width and bit-frequency agreement (|f - nominal| < 60 Hz).
    Return candidates sorted by confidence. Must run on a 660 s file in < 60 s.

    Implementation notes: the stage-1 template additionally contains the code-independent parts of the data section
    (the stop bit and the 1100-or-1300 Hz band over the eight bit slots) and uses the unit-variance normalisation
    sum(z)/sqrt(N) instead of per-tone means, which gave a much larger separation from noise on the reference recording;
    the correlation is evaluated exactly with cumulative sums (equivalent to fftconvolve with indicator kernels) on
    zero-padded z traces so that headers at the very start/end of the signal are found (partial headers score partially).
    The bit frequencies are measured on the spectra averaged over the plateau offsets (more robust than one window).
    When the plateau code fails parity and the joint 128-code matched filter prefers a code within 2 bits with a margin
    >= MF_ADOPT_MARGIN, that code is reported (code_source 'matched-filter'); a reported code that fails parity is capped
    at confidence PARITY_FAIL_CAP. vis_code is None when confidence < 0.2 or the bit windows fall outside the signal.
    Extra keyword arguments (defaults validated on the reference recording): refine_win_ms = analysis window of the
    1 ms-hop refinement, refine_span_ms = alignment search span around the stage-1 peak, local_pad_s = audio around the
    candidate used for the local z-scoring."""
    x = np.asarray(x)
    fs = int(fs); hop_ms = float(params.vis_hop_ms); win_ms = float(params.vis_frame_ms)
    hop_s = hop_ms / 1000.0; dur = len(x) / fs
    if dur < 0.5 * VIS_TOTAL_MS / 1000.0:
        return []
    # ---- stage 1: code-independent template over the whole signal (zero-padded by one template length each side)
    t_rel, E, Etot = _band_traces(x, fs, win_ms, hop_ms)
    if len(t_rel) == 0:
        return []
    noise_con = float(getattr(params, "vis_noise_contrast_db", 12.0))
    Z = _ratio_z(E, Etot, noise_con)
    noise_ref = bool(Z.get("_noise_reference", False))
    tf1 = _template_frames(_vis_elements(None), hop_ms, win_ms)
    L1 = max(jb for _, _, jb in tf1); pad = L1
    S1, _ = _mf_score(_pad_z(Z, pad), tf1, len(t_rel) + 2 * pad)
    if len(S1) == 0:
        return []
    kk1 = np.arange(len(S1))
    t_pos = x_t0 + t_rel[0] + (kk1 - pad) * hop_s - 0.5 * hop_s          # leader start for each template position
    full = (kk1 >= pad) & (kk1 <= len(t_rel) - L1 + pad)                   # template entirely inside the signal
    if full.sum() < 10:
        full = np.ones(len(S1), bool)
    dist = max(1, int(round(1.0 / hop_s)))
    pk, _ = find_peaks(S1, distance=dist)
    if len(pk) == 0:
        pk = np.array([int(np.argmax(S1))])
    top = pk[np.argsort(S1[pk])[::-1][:max(1, int(max_candidates))]]
    full_cache: dict[int, np.ndarray] = {}
    out: list[VISCandidate] = []
    for k in top:
        t1 = float(t_pos[k]); score1 = float(S1[k])
        med, sd, nmax = _false_alarm_stats(t_pos[full], S1[full], t1)
        z = (score1 - med) / sd
        notes = []
        if t1 < x_t0 - 1e-6 or t1 + VIS_TOTAL_MS / 1000.0 > x_t0 + dur + 1e-6:
            notes.append("header partly outside the signal")
        # ---- stage 2a: local 1 ms-hop traces and the joint code x alignment search (padded likewise)
        a = max(0.0, t1 - x_t0 - local_pad_s); b = min(dur, t1 - x_t0 + VIS_TOTAL_MS / 1000.0 + local_pad_s)
        s0 = int(round(a * fs)); s1 = int(round(b * fs))
        xs = x[s0:s1]; xs_t0 = x_t0 + s0 / fs
        tr, Er, Etr = _band_traces(xs, fs, refine_win_ms, 1.0)
        if len(tr) == 0:
            out.append(VISCandidate(t_leader=t1, t_line0=t1 + VIS_TOTAL_MS / 1000.0, score=score1, z=float(z),
                                    noise_max=nmax, t_leader_coarse=t1, notes="too short for refinement"))
            continue
        Lr = _template_len(1.0, refine_win_ms); padr = Lr
        Sj, npos = _joint_code_scores(_pad_z(_ratio_z(Er, Etr, noise_con), padr), 1.0, refine_win_ms, len(tr) + 2 * padr)
        tpos_r = xs_t0 + tr[0] + (np.arange(npos) - padr) * 1e-3 - 0.5e-3
        win_mask = np.abs(tpos_r - t1) <= refine_span_ms / 1000.0
        if not win_mask.any():
            win_mask = np.ones(npos, bool)
        Sj_w = np.where(win_mask[None, :], Sj, -np.inf)
        mf_code, kk = np.unravel_index(int(np.argmax(Sj_w)), Sj_w.shape)
        mf_code = int(mf_code); kk = int(kk)
        col = np.sort(Sj[:, kk])[::-1]; mf_margin = float(col[0] - col[1])
        trace = Sj[mf_code]
        delta = 0.0
        if 0 < kk < npos - 1:
            den = trace[kk - 1] - 2 * trace[kk] + trace[kk + 1]
            if den < 0:
                delta = float(np.clip(0.5 * (trace[kk - 1] - trace[kk + 1]) / den, -0.5, 0.5))
        t_leader = float(tpos_r[kk] + delta * 1e-3)
        score_full = float(trace[kk])
        # ---- stage 2b: bit decoding with the sampling-offset scan (only if the 8 data/parity bit windows at their nominal
        #      positions lie inside the signal; the code-independent start/stop bits may be clipped)
        half_w = BIT_WIN_MS / 2000.0
        code_available = (t_leader + (VIS_START_MS + VIS_BIT_MS) / 1000.0 - half_w >= x_t0 - 1e-6 and
                          t_leader + (VIS_START_MS + 9 * VIS_BIT_MS) / 1000.0 + half_w <= x_t0 + dur + 1e-6)
        if code_available:
            dec = _decode_bits(xs, fs, xs_t0, t_leader, float(params.vis_bit_offset_scan_ms))
            bits, code, parity_ok, code_source = dec["bits"], dec["code"], dec["parity_ok"], "plateau"
            if not parity_ok and mf_margin >= MF_ADOPT_MARGIN:
                hd = int(sum(int(u != v) for u, v in zip(bits, vis_bits(mf_code))))
                if hd <= 2:
                    notes.append(f"plateau code {code} failed parity; joint matched-filter code {mf_code} "
                                 f"(margin {mf_margin:.2f}, {hd} bit(s) apart) adopted")
                    bits, code, parity_ok, code_source = vis_bits(mf_code), mf_code, True, "matched-filter"
            nominal, n_ok = _nominal_and_agreement(bits, dec["freqs"])
            # decoded-code full template on the same padded position grid as stage 1 (same span -> same t_pos)
            if code not in full_cache:
                full_cache[code], _ = _mf_score(_pad_z(Z, pad), _template_frames(_vis_elements(code), hop_ms, win_ms),
                                                len(t_rel) + 2 * pad)
            Sf = full_cache[code]
            near = np.abs(t_pos[:len(Sf)] - t_leader) <= 1.5 * hop_s
            if len(Sf) == len(S1) and near.any():
                medf, sdf, _ = _false_alarm_stats(t_pos[full], Sf[full], t_leader)
                z_full = (float(Sf[near].max()) - medf) / sdf
            else:
                z_full = float(z)
            mf_agree = (mf_code == code)
            if not mf_agree:
                notes.append(f"joint matched filter prefers VIS {mf_code} ({vis_mode_name(mf_code)})")
            if dec["plateau_ms"] < 10.0:
                notes.append(f"plateau {dec['plateau_ms']:.0f} ms < 10 ms")
            if not parity_ok:
                notes.append("parity failed")
        else:
            dec = dict(bits=[], plateau_ms=0.0, plateau_offset_ms=0.0, llr=[], freqs=[])
            bits, code, parity_ok, code_source = [], None, None, "none"
            nominal, n_ok = [], 0; z_full = float(z); mf_agree = False
            notes.append("bit windows outside the signal: code not decodable")
        conf, c_det = _confidence(float(z), float(z_full), float(params.vis_min_z), code_available, bool(parity_ok),
                                  dec["plateau_ms"], n_ok, mf_agree, mf_margin)
        if z < params.vis_min_z:
            notes.append(f"z {z:.1f} below vis_min_z {params.vis_min_z:g}")
        notes.append("band ratios z-scored against the no-tone frames" if noise_ref else "band ratios z-scored against all frames (too few no-tone frames)")
        reliable = conf >= 0.6
        vis_code = code if (code is not None and conf >= 0.2) else None
        if code is not None and vis_code is None:
            notes.append("code unreliable (confidence < 0.2): vis_code suppressed")
        out.append(VISCandidate(
            t_leader=t_leader, t_line0=t_leader + VIS_TOTAL_MS / 1000.0, score=score1, z=float(z), noise_max=nmax,
            bits=list(bits), vis_code=vis_code, parity_ok=parity_ok,
            mode_name=vis_mode_name(vis_code) if vis_code is not None else "unknown",
            bit_freqs_hz=dec["freqs"], plateau_ms=dec["plateau_ms"], confidence=conf,
            detection_confidence=float(c_det), code_reliable=reliable, code_available=code_available,
            code_source=code_source, decoded_code=code, plateau_bits=list(dec["bits"]), mf_code=mf_code,
            mf_margin=mf_margin, score_full=score_full, z_full=float(z_full), t_leader_coarse=t1,
            bit_llr_db=dec["llr"], bit_freqs_nominal_hz=nominal, n_freq_ok=n_ok,
            plateau_offset_ms=dec["plateau_offset_ms"], notes="; ".join(notes)))
    out.sort(key=lambda c: (c.confidence, c.z), reverse=True)
    return out
