"""Line synchronisation without a header: sync likelihood, ACF comb period, phase fold, per-line pulse tracking.

Ported from the verified analysis scripts (05_sync_period.py: ACF comb and period fold; 06_timing.py: per-line sync
tracking with a robust iterative linear fit) and hardened for use without any header:
  * sync_likelihood   s(t) = share of the 900-2600 Hz energy that falls in the 1200 Hz sync band, 9 ms Hann / 0.5 ms hop
  * estimate_period_acf  comb statistic of the ACF at k*150 ms (minus half-period lags), period refinement by a weighted
                      least-squares fit of the comb peak lags k*P over long lags, fold-scan cross-check, period scan 100-1100 ms
  * fold_phase        folded profile and matched-filter pulse phase
  * track_lines       per-line peak search around the predicted pulse centre, mode-seeking offset initialisation, robust
                      iterative linear fit with shrinking tolerance (12 -> 4 ms), inlier runs
  * scan_comb_windows (helper) sliding-window comb significance to locate the frame in a long recording
All functions are deterministic and use only the audio-derived sync likelihood."""
from dataclasses import dataclass, field
import numpy as np
from .params import Params
from .modes import ROBOT36

_TINY = 1e-30


@dataclass
class PeriodEstimate:
    period: float | None         # best period (s) consistent with Robot 36 (None if not found)
    comb_z: float                # comb significance of the Robot 36 fundamental (ACF at k*150 ms minus half-period lags, in ACF-sd units)
    acf_150: float
    best_period_any: float       # best period over the full search range (s), for reporting other modes
    best_z_any: float
    candidates: list = field(default_factory=list)   # [(period_s, z), ...] top 5
    # --- extras (defaults; diagnostics and redundancy) ---
    period_parabolic: float | None = None    # parabolic interpolation of the ACF peak near 150 ms
    period_combfit: float | None = None      # weighted LS fit of comb peak lags k*P over long lags
    period_fold: float | None = None         # fine fold-scan (epoch folding) maximum near 150 ms
    n_comb_peaks: int = 0                    # peaks used by the comb fit
    combfit_resid_ms: float = float("nan")   # rms residual of the comb fit (ms)
    combfit_se: float = float("nan")         # formal standard error of the comb-fit period (s); window-to-window scatter is ~3x larger
    comb_on: float = float("nan")            # mean ACF at 150..900 ms
    comb_off: float = float("nan")           # mean ACF at 75..825 ms
    acf_sd: float = float("nan")             # robust sd (1.4826 MAD) of the ACF over 50-1200 ms (the comb_z denominator)
    comb_z_plain: float = 0.0                # comb statistic with the plain sd denominator (analysis-script definition)
    comb_z_half_rate: float = float("nan")   # [fix] comb statistic at TWICE the nominal period (peaks at k*300 ms minus troughs at 150+k*300 ms):
                                             # ~0 for a Robot 36 pulse train (both lag sets are peaks), large for a 300 ms train (Robot 72) aliased onto 150 ms
    duration_s: float = 0.0                  # length of the analysed sync-likelihood trace
    acf_lags_s: np.ndarray | None = None     # ACF trace for plots (0 .. 1.2 s)
    acf: np.ndarray | None = None
    scan_periods_s: np.ndarray | None = None # period scan for plots
    scan_z: np.ndarray | None = None
    found: bool = False                      # comb_z >= comb_z_min and period within tolerance


@dataclass
class LineSync:
    t_pulse: np.ndarray          # per-line detected pulse centre time (s)
    strength: np.ndarray         # per-line peak sync likelihood
    inlier: np.ndarray           # bool per line
    t_line0: float               # robust-fit line 0 start (s) = fitted pulse centre - sync_ms/2
    period: float                # robust-fit line period (s)
    resid_ms_sd: float
    n_inliers: int
    threshold: float             # noise 99.5th percentile of the sync likelihood
    runs: list = field(default_factory=list)   # contiguous inlier runs [(first, last), ...]
    # --- extras (defaults) ---
    line_index: np.ndarray | None = None     # line index of each entry (may start below 0 when line_offset < 0)
    t_expected: np.ndarray | None = None     # fitted pulse centre per line
    resid_ms: np.ndarray | None = None       # (t_pulse - t_expected) in ms
    valid: np.ndarray | None = None          # the search window had data
    noise_median: float = float("nan")       # median of the noise reference of the sync likelihood
    noise_sd: float = float("nan")           # robust sd (1.4826 MAD) of the noise reference
    noise_ref: str = ""                      # how the noise reference was obtained
    n_candidates: int = 0                    # lines whose peak exceeds the threshold (before the residual test)
    fit_iterations: int = 0
    converged: bool = False                  # >= 5 inliers in the final fit
    shape: np.ndarray | None = None          # pulse-shape ratio per line (1 = clean pulse, ~0 = plateau/tone, nan = no data)
    n_shape_rejected: int = 0                # above-threshold, in-tolerance peaks rejected by the pulse-shape test
    period_se: float = float("nan")          # standard error of the fitted period (s), from the inlier residuals
    t_pivot_se: float = float("nan")         # standard error of the fitted pulse time at the pivot line (s)
    pivot_line: float = float("nan")         # mean inlier line index (the fit pivot: changing the period rotates about it)
    threshold_strict: float = float("nan")   # strict threshold (97th pct of the per-line maxima): ~0.6 % false inliers per line
    strong: np.ndarray | None = None         # inlier & strength > threshold_strict
    n_strong: int = 0
    expected_false_inliers: float = float("nan")   # from the density of above-threshold peaks outside the tolerance
    inlier_excess_z: float = float("nan")    # (n_inliers - expected)/sqrt(expected + 1); pure noise reaches ~3 (the fit adapts to it)

    @property
    def credible(self) -> bool:
        """>= 8 inliers AND clearly more than the uniform background would give: n >= 2.5*expected + 4.
        On the reference recording's pure-noise controls the default threshold yields 6-13 'inliers' per 240 lines
        (expected 4-6), so a raw count is not a detection statistic; this excess test rejects every control."""
        exp = self.expected_false_inliers if np.isfinite(self.expected_false_inliers) else 0.0
        return bool(self.n_inliers >= 8 and self.n_inliers >= 2.5*exp + 4.0)

    @property
    def strength_z(self):
        """peak strength in noise-sd units (nan where the window had no data)."""
        sd = self.noise_sd if np.isfinite(self.noise_sd) and self.noise_sd > 0 else 1.0
        return (self.strength - self.noise_median)/sd


def _p(params, name, default):
    return getattr(params, name, default) if params is not None else default


# ------------------------------------------------------------------------------------------------- sync likelihood
def sync_likelihood(x: np.ndarray, fs: int, x_t0: float, params: Params, chunk: int = 20000,
                    nfft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """s(t) = E(1130-1270 Hz) / E(900-2600 Hz) per params.sync_win_ms Hann window at params.sync_hop_ms hop.
    Returns (t_abs, s). Vectorised framing in chunks.

    The band energies are evaluated on the nfft-point DFT grid (fs/nfft = 11.72 Hz at 48 kHz) restricted to 900-2600 Hz,
    exactly as the analysis scripts did with a zero-padded FFT, but computed as one real matrix product per chunk
    (frames @ [cos | sin] with the Hann window folded into the matrix), which is cheaper than the padded FFT.
    t_abs is the absolute time of each frame centre (x[0] is at x_t0)."""
    x = np.asarray(x, dtype=np.float32)
    win = int(round(params.sync_win_ms*fs/1000.0)); hop = max(1, int(round(params.sync_hop_ms*fs/1000.0)))
    if len(x) < win or win < 2:
        return np.zeros(0), np.zeros(0, np.float32)
    n = 1 + (len(x) - win)//hop
    f = np.arange(nfft//2 + 1)*fs/nfft
    sel_tot = (f >= 900.0) & (f < 2600.0)
    f_tot = f[sel_tot]
    sel_sync = (f_tot >= params.sync_band_lo) & (f_tot < params.sync_band_hi)
    if sel_sync.sum() == 0:                       # degenerate configuration: take the nearest bin
        sel_sync = np.zeros(len(f_tot), bool); sel_sync[np.argmin(np.abs(f_tot - 0.5*(params.sync_band_lo + params.sync_band_hi)))] = True
    nf = len(f_tot)
    nn = np.arange(win, dtype=np.float64)
    arg = 2.0*np.pi*np.outer(nn, f_tot)/fs
    w = np.hanning(win)
    M = np.concatenate([np.cos(arg), -np.sin(arg)], axis=1)*w[:, None]
    M = M.astype(np.float32)                        # (win, 2 nf)
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop]
    s = np.empty(n, np.float32)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        F = np.ascontiguousarray(frames[a:b])
        Z = F @ M
        P = Z[:, :nf]**2 + Z[:, nf:]**2
        e_tot = P.sum(axis=1); e_sync = P[:, sel_sync].sum(axis=1)
        s[a:b] = e_sync/np.maximum(e_tot, _TINY)
    t = x_t0 + (np.arange(n)*hop + (win - 1)/2.0)/fs
    return t, s


# ------------------------------------------------------------------------------------------------- helpers
def _hop_of(t):
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2: return 0.0005
    return float(np.median(np.diff(t[:min(len(t), 100000)])))


def _acf(s, max_lag_n):
    """unbiased, normalised autocorrelation of s (mean removed) for lags 0..max_lag_n-1 via FFT."""
    s0 = np.asarray(s, dtype=np.float64); s0 = s0 - s0.mean()
    n = len(s0)
    nfft = 1 << int(np.ceil(np.log2(max(2*n, 2))))
    S = np.fft.rfft(s0, nfft)
    r = np.fft.irfft(S*np.conj(S), nfft)[:min(n, max_lag_n)]
    r = r/np.maximum(n - np.arange(len(r)), 1)
    return r/max(r[0], _TINY)


def _parabolic_peak(y, k):
    """sub-sample peak position and value from the parabola through y[k-1], y[k], y[k+1]."""
    if k <= 0 or k >= len(y) - 1: return float(k), float(y[k])
    a, b, c = y[k-1], y[k], y[k+1]
    den = a - 2*b + c
    if den >= 0: return float(k), float(b)
    d = 0.5*(a - c)/den
    d = float(np.clip(d, -1, 1))
    return k + d, float(b - 0.25*(a - c)*d)


def _comb_z(acf, hop, P, L_max, sd_ref):
    """(mean ACF at k*P) - (mean ACF at (k-1/2)*P) for k*P <= L_max, in units of sd_ref (nan if no peak fits)."""
    K = int(np.floor(L_max/P + 1e-9))
    if K < 1 or sd_ref <= 0: return np.nan, np.nan, np.nan
    kk = np.arange(1, K + 1)
    lags = np.arange(len(acf))*hop
    on = float(np.mean(np.interp(kk*P, lags, acf))); off = float(np.mean(np.interp((kk - 0.5)*P, lags, acf)))
    return (on - off)/sd_ref, on, off


def _fold_mf_peak(ph_frac, s0, nb, tmpl_f):
    """peak of the circular matched filter (template spectrum tmpl_f) of the folded, mean-removed profile."""
    b = np.minimum((ph_frac*nb).astype(np.int64), nb - 1)
    cnt = np.bincount(b, minlength=nb); prof = np.bincount(b, weights=s0, minlength=nb)/np.maximum(cnt, 1)
    pm = prof - prof.mean()
    return float(np.fft.irfft(np.fft.rfft(pm)*tmpl_f, nb).max())


def _robust_sd(v):
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 3: return float("nan")
    med = np.median(v)
    return float(np.median(np.abs(v - med))*1.4826)


# ------------------------------------------------------------------------------------------------- ACF period
def estimate_period_acf(t: np.ndarray, s: np.ndarray, params: Params, refine_max_lag_s: float = 40.0,
                        scan_step_ms: float = 0.25, keep_traces: bool = True) -> PeriodEstimate:
    """ACF of s (mean removed) via FFT; comb statistic for Robot 36: mean ACF at lags 150,300,...,900 ms minus mean at
    75,225,...,825 ms, divided by the sd of the ACF over 50-1200 ms; refine the period by parabolic interpolation of the
    ACF peak near 150 ms and by a least-squares fit of the comb peak lags k*P; also scan P in params.period_search_ms
    for the best comb anywhere (reporting only).

    Refinement detail: the comb fit uses every ACF peak at k*P up to min(refine_max_lag_s, half the trace) in three stages
    (k <= 10, <= 40, all) so that the search window never has to absorb the accumulated drift k*dP; peaks are weighted by
    their local significance and the fit is through the origin (lag_k = k*P). A fine fold scan (epoch folding, +-300 ppm,
    2 ppm steps) is reported as period_fold for cross-checking. `period` is the comb-fit period when >= 5 peaks were usable,
    else the parabolic one, and None when comb_z < comb_z_min (Params field, default 4) or outside period_tolerance_ppm."""
    t = np.asarray(t, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    P_nom = params.period_nominal_ms/1000.0
    tol = params.period_tolerance_ppm*1e-6
    z_min = float(_p(params, "comb_z_min", 4.0))
    n = len(s)
    hop = _hop_of(t)
    dur = n*hop
    empty = PeriodEstimate(None, 0.0, 0.0, P_nom, 0.0, [], duration_s=dur)
    if n < 10 or dur < 0.4:
        return empty
    L_plot = min(1.2, 0.9*dur)
    max_lag_n = int(min(n, np.ceil((min(refine_max_lag_s, 0.5*dur) + 0.1)/hop) + 1, n))
    max_lag_n = max(max_lag_n, int(np.ceil(L_plot/hop)) + 2)
    acf = _acf(s, max_lag_n)
    lags = np.arange(len(acf))*hop
    # ---- Robot 36 comb statistic (k = 1..6, lags 150..900 ms; troughs 75..825 ms)
    # sd of the ACF over 50-1200 ms: robust (1.4826 MAD) so that the comb peaks themselves do not inflate it;
    # the plain-sd version (the analysis scripts' definition, which saturates near z ~ 5 for a strong comb) is kept as comb_z_plain.
    i_lo = int(round(0.05/hop)); i_hi = int(min(len(acf), round(1.2/hop)))
    base = acf[i_lo:i_hi]
    sd_plain = float(base.std()) if len(base) > 10 else float("nan")
    sd_ref = _robust_sd(base) if len(base) > 10 else float("nan")
    if not (np.isfinite(sd_ref) and sd_ref > 0): sd_ref = sd_plain
    L_comb = min(6*P_nom + 1e-6, L_plot)
    comb_z, on, off = _comb_z(acf, hop, P_nom, L_comb, sd_ref) if np.isfinite(sd_ref) and sd_ref > 0 else (np.nan, np.nan, np.nan)
    if not np.isfinite(comb_z): comb_z = 0.0
    # comb at twice the nominal period: a Robot 72 pulse train (300 ms) aliased onto the 150 ms grid gives ACF peaks only at
    # multiples of 300 ms and troughs at 150 + k*300 ms, so this statistic is large; a genuine Robot 36 train has peaks at every
    # multiple of 150 ms and gives ~0 (both lag sets are peaks)
    comb_z_half = _comb_z(acf, hop, 2.0*P_nom, L_plot, sd_ref)[0] if np.isfinite(sd_ref) and sd_ref > 0 else np.nan
    comb_z_plain = (on - off)/sd_plain if (np.isfinite(sd_plain) and sd_plain > 0 and np.isfinite(on)) else 0.0
    acf_150 = float(np.interp(P_nom, lags, acf)) if lags[-1] >= P_nom else 0.0
    # ---- scan P over the search range (reporting other modes)
    lo_ms, hi_ms = params.period_search_ms
    hi_ms = min(hi_ms, L_plot*1000.0 - 1.0)
    cands = []; best_any = P_nom; best_z_any = 0.0; scan_P = None; scan_zv = None
    if hi_ms > lo_ms and np.isfinite(sd_ref) and sd_ref > 0:
        scan_P = np.arange(lo_ms, hi_ms + 1e-9, scan_step_ms)/1000.0
        scan_zv = np.empty(len(scan_P))
        for i, P in enumerate(scan_P):
            scan_zv[i] = _comb_z(acf, hop, P, L_plot, sd_ref)[0]
        scan_zv = np.nan_to_num(scan_zv, nan=0.0)
        order = np.argsort(scan_zv)[::-1]
        shown = []
        for i in order:
            if any(abs(scan_P[i] - p) < 0.003 for p, _ in shown): continue
            shown.append((float(scan_P[i]), float(scan_zv[i])))
            if len(shown) >= 5: break
        cands = shown
        if cands: best_any, best_z_any = cands[0]
    # ---- refinement 1: parabolic interpolation of the ACF peak near 150 ms
    P_par = None
    w_par = int(round(0.010/hop))
    k_nom = int(round(P_nom/hop))
    if k_nom + w_par + 1 < len(acf):
        seg = acf[k_nom - w_par:k_nom + w_par + 1]
        k = int(np.argmax(seg)) + k_nom - w_par
        kp, _ = _parabolic_peak(acf, k)
        P_par = float(kp*hop)
    # ---- refinement 2: weighted LS fit of comb peak lags k*P over long lags (3 stages)
    P_fit = None; n_pk = 0; resid_ms = float("nan"); fit_se = float("nan")
    if P_par is not None and comb_z >= 0.5*z_min:
        P_cur = P_par
        L_ref = lags[-1]
        for K_stage in (10, 40, 100000):
            K = int(min(K_stage, np.floor(L_ref/P_cur)))
            if K < 1: break
            kk = np.arange(1, K + 1)
            half_w = max(3, int(round(0.004/hop)))          # +-4 ms search around k*P_cur
            centres = np.round(kk*P_cur/hop).astype(int)
            ok = (centres - half_w > 0) & (centres + half_w + 1 < len(acf))
            kk, centres = kk[ok], centres[ok]
            if len(kk) < 3: break
            idx = centres[:, None] + np.arange(-half_w, half_w + 1)[None, :]
            seg = acf[idx]
            j = np.argmax(seg, axis=1)
            pk_pos = np.empty(len(kk)); pk_val = np.empty(len(kk))
            for i in range(len(kk)):
                kp, v = _parabolic_peak(acf, int(idx[i, j[i]]))
                pk_pos[i] = kp*hop; pk_val[i] = v
            # local baseline: median of the ACF between (k-1/2)P and (k+1/2)P excluding +-9 ms around the peak
            bw = int(round(0.5*P_cur/hop)); ex = int(round(0.009/hop))
            sig = np.empty(len(kk))
            for i in range(len(kk)):
                c = centres[i]; lo_i = max(1, c - bw); hi_i = min(len(acf), c + bw)
                reg = np.r_[acf[lo_i:max(lo_i, c - ex)], acf[min(hi_i, c + ex + 1):hi_i]]
                sig[i] = (pk_val[i] - (np.median(reg) if len(reg) else 0.0))/max(sd_ref, _TINY)
            use = sig > 2.5
            if use.sum() < 3: break
            wgt = np.clip(sig[use], 0, 10.0)
            k_u = kk[use].astype(np.float64); l_u = pk_pos[use]
            for _ in range(3):                               # robust re-weighting against mis-located peaks
                P_new = float(np.sum(wgt*k_u*l_u)/max(np.sum(wgt*k_u*k_u), _TINY))
                r = l_u - k_u*P_new
                keep = np.abs(r) < 0.002
                if keep.sum() < 3 or keep.all(): break
                k_u, l_u, wgt = k_u[keep], l_u[keep], wgt[keep]
            P_cur = P_new; n_pk = int(len(k_u)); resid_ms = float(np.sqrt(np.mean((l_u - k_u*P_cur)**2))*1000.0)
            rr = l_u - k_u*P_cur
            fit_se = float(np.sqrt((np.sum(wgt*rr*rr)/max(n_pk - 1, 1))/max(np.sum(wgt*k_u*k_u), _TINY))) if n_pk > 1 else float("nan")
            if K >= np.floor(L_ref/P_cur): break
        if n_pk >= 5: P_fit = P_cur
    # ---- refinement 3 (reporting only): fold scan (epoch folding) around the best estimate, +-300 ppm in 5 ppm steps,
    #      metric = matched-filter peak of the folded profile (triangle of half-width 9 ms), parabolic maximum
    P_fold = None
    P_ref = P_fit if P_fit is not None else (P_par if P_par is not None else P_nom)
    if comb_z >= 0.5*z_min and dur >= 3*P_nom:
        s0 = s - np.median(s)
        rel = np.arange(-300.0, 300.0 + 1e-9, 5.0)*1e-6
        nb = int(round(P_nom*1000*2))
        hw = ROBOT36.sync_ms/1000.0/P_ref*nb
        kk = np.arange(nb); dd = np.minimum(kk, nb - kk).astype(np.float64)
        tmpl = np.clip(1.0 - dd/hw, 0.0, None); tmpl -= tmpl.mean()
        tmpl_f = np.conj(np.fft.rfft(tmpl))
        vals = np.array([_fold_mf_peak(np.mod(t, P_ref*(1 + r_))/(P_ref*(1 + r_)), s0, nb, tmpl_f) for r_ in rel])
        kbest = int(np.argmax(vals))
        kp, _ = _parabolic_peak(vals, kbest)
        P_fold = float(P_ref*(1 + np.interp(kp, np.arange(len(rel)), rel)))
    # ---- decision
    P_best = P_fit if P_fit is not None else P_par
    found = bool(P_best is not None and comb_z >= z_min and abs(P_best/P_nom - 1.0) <= tol)
    period = float(P_best) if found else None
    return PeriodEstimate(period=period, comb_z=float(comb_z), acf_150=acf_150, best_period_any=float(best_any),
                          best_z_any=float(best_z_any), candidates=cands, period_parabolic=P_par, period_combfit=P_fit,
                          period_fold=P_fold, n_comb_peaks=n_pk, combfit_resid_ms=resid_ms, combfit_se=(fit_se if P_fit is not None else float("nan")),
                          comb_on=float(on), comb_off=float(off),
                          acf_sd=sd_ref, comb_z_plain=float(comb_z_plain), comb_z_half_rate=float(comb_z_half), duration_s=dur,
                          acf_lags_s=(lags[:int(round(L_plot/hop))].copy() if keep_traces else None),
                          acf=(acf[:int(round(L_plot/hop))].copy() if keep_traces else None),
                          scan_periods_s=(scan_P if keep_traces else None), scan_z=(scan_zv if keep_traces else None), found=found)


def scan_comb_windows(t: np.ndarray, s: np.ndarray, params: Params, win_s: float = 36.0, hop_s: float = 6.0):
    """Sliding-window Robot 36 comb significance: returns (t_centre, comb_z) per window and the best window (t_start, t_end).
    Used to locate the frame in a long recording before estimate_period_acf / fold_phase / track_lines are run on it
    (the comb z of a 36 s frame inside a 660 s trace is diluted by ~sqrt(660/36))."""
    t = np.asarray(t, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    if len(t) < 10: return np.zeros(0), np.zeros(0), (float(t[0]) if len(t) else 0.0, float(t[-1]) if len(t) else 0.0)
    hop = _hop_of(t); nwin = int(round(win_s/hop)); nhop = max(1, int(round(hop_s/hop)))
    if nwin >= len(s): nwin = len(s); nhop = len(s)
    starts = np.arange(0, len(s) - nwin + 1, nhop)
    zc = np.empty(len(starts)); tc = np.empty(len(starts))
    P_nom = params.period_nominal_ms/1000.0
    for i, a in enumerate(starts):
        seg = s[a:a + nwin]
        acf = _acf(seg, int(round(1.25/hop)) + 2)
        i_lo = int(round(0.05/hop)); i_hi = int(min(len(acf), round(1.2/hop)))
        sd_ref = _robust_sd(acf[i_lo:i_hi]) if i_hi - i_lo > 10 else np.nan
        z = _comb_z(acf, hop, P_nom, min(6*P_nom + 1e-6, (len(acf) - 1)*hop), sd_ref)[0] if np.isfinite(sd_ref) and sd_ref > 0 else np.nan
        zc[i] = 0.0 if not np.isfinite(z) else z
        tc[i] = t[a] + 0.5*nwin*hop
    ib = int(np.argmax(zc)) if len(zc) else 0
    best = (float(t[starts[ib]]), float(t[min(len(t) - 1, starts[ib] + nwin - 1)])) if len(zc) else (float(t[0]), float(t[-1]))
    return tc, zc, best


# ------------------------------------------------------------------------------------------------- phase fold
def fold_phase(t: np.ndarray, s: np.ndarray, period: float, bins_per_ms: float = 2.0,
               pulse_ms: float = ROBOT36.sync_ms) -> tuple[np.ndarray, float, float]:
    """Fold s at `period`; return (profile, pulse_phase_s, contrast) where pulse_phase_s is the phase (s) of the
    sync pulse centre (profile peak, with the 9 ms pulse width taken into account) and contrast = (peak-median)/sd.

    The profile is the mean of s per phase bin (bin width 1/bins_per_ms ms). The pulse centre is located by circular
    matched filtering of the mean-removed profile with the expected response of a `pulse_ms` pulse seen through a
    `pulse_ms` window (a triangle of half-width pulse_ms), with parabolic sub-bin interpolation. The phase is absolute:
    pulses are centred at pulse_phase_s + k*period, so line starts are at pulse_phase_s - sync_ms/2 + k*period.
    contrast = (peak - median)/robust sd (1.4826 MAD) of the matched-filtered profile."""
    t = np.asarray(t, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    nb = max(8, int(round(period*1000.0*bins_per_ms)))
    if len(s) == 0 or period <= 0:
        return np.zeros(nb), 0.0, 0.0
    ph = np.mod(t, period)/period
    b = np.minimum((ph*nb).astype(np.int64), nb - 1)
    cnt = np.bincount(b, minlength=nb)
    prof = np.bincount(b, weights=s, minlength=nb)/np.maximum(cnt, 1)
    if (cnt == 0).any():                                     # fill empty bins with the median so they do not bias the filter
        prof[cnt == 0] = np.median(prof[cnt > 0]) if (cnt > 0).any() else 0.0
    hw = max(1.0, pulse_ms/1000.0/period*nb)                 # half-width of the pulse response in bins
    k = np.arange(nb); d = np.minimum(k, nb - k).astype(np.float64)
    tmpl = np.clip(1.0 - d/hw, 0.0, None); tmpl -= tmpl.mean()
    pm = prof - prof.mean()
    mf = np.fft.irfft(np.fft.rfft(pm)*np.conj(np.fft.rfft(tmpl)), nb)      # circular correlation, peak at the pulse centre bin
    kmax = int(np.argmax(mf))
    a_, b_, c_ = mf[(kmax - 1) % nb], mf[kmax], mf[(kmax + 1) % nb]
    den = a_ - 2*b_ + c_
    delta = float(np.clip(0.5*(a_ - c_)/den, -1, 1)) if den < 0 else 0.0
    phase_bins = (kmax + delta + 0.5) % nb
    pulse_phase_s = float(phase_bins/nb*period)
    med = np.median(mf); mad = np.median(np.abs(mf - med))*1.4826
    contrast = float((mf.max() - med)/max(mad, _TINY))
    return prof, pulse_phase_s, contrast


# ------------------------------------------------------------------------------------------------- noise threshold
def _noise_reference(t, s, t_line0, period, line_index, regions=None, pct=99.5, hop=None, line_pct=80.0, strict_pct=97.0):
    """Noise reference of the sync likelihood -> (threshold, median, robust sd, label, strict threshold).
    regions given: the pct-th percentile of the samples inside the (absolute) regions (strict: pct + (100-pct)*0.8).
    otherwise 'inline-max': per tracked line, the maximum of s over phases 40-95 ms (the Y segment, which carries no 1200 Hz
    tone by the Robot 36 structure, with the 20 ms search slack and the 9 ms window half-width kept clear of the pulse), over
    the NOISIER HALF of the lines (by their pulse-free median; tone-carrying lines have a lower 1200 Hz share and cannot
    produce false pulses); threshold = line_pct-th percentile of these maxima. Calibrated on the reference recording's seven
    pure-noise controls (gate on and off): the 99.5th percentile of the noise samples equals the 79-80th percentile of the
    per-line maxima of the noisier half, so line_pct = 80 reproduces the established threshold on pure noise and does not
    drop when most lines carry strong tones (a pooled-sample percentile does). The strict threshold (strict_pct = 97, about
    3 % false candidates per line, ~0.6 % false inliers) marks 'strong' pulses for the recovery frame placement.
    median/sd are those of the pulse-free samples of the noisier half.
    fallback 'lowest10': the pct-th percentile of the 10 % of 1 s blocks with the lowest mean s."""
    s = np.asarray(s, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    hop = _hop_of(t) if hop is None else hop
    if len(s) == 0 or len(t) == 0:                       # empty trace: nothing can be a pulse
        return float("inf"), float("nan"), float("nan"), "empty trace", float("inf")
    vals = None; label = ""
    if regions:
        m = np.zeros(len(s), bool)
        for (a, b) in regions:
            m |= (t >= a) & (t <= b)
        if m.sum() >= 200: vals = s[m]; label = "regions"
    if vals is None and period > 0 and len(line_index):
        ph0 = int(round(0.040/hop)); ph1 = int(round(0.095/hop))
        if ph1 > ph0 + 5:
            offs = np.arange(ph0, ph1)
            starts = np.round((t_line0 + period*np.asarray(line_index, dtype=np.float64) - t[0])/hop).astype(np.int64)
            idx = starts[:, None] + offs[None, :]
            okl = (idx[:, 0] >= 0) & (idx[:, -1] < len(s))
            if okl.sum() >= 8:
                seg = s[idx[okl]]
                mx = seg.max(axis=1); md = np.median(seg, axis=1)
                sel = md >= np.median(md)
                if sel.sum() < 4: sel = np.ones(len(mx), bool)
                thr = float(np.percentile(mx[sel], line_pct)); thr_strict = float(np.percentile(mx[sel], strict_pct))
                noisy = seg[sel].ravel()
                med = float(np.median(noisy)); sd = float(np.median(np.abs(noisy - med))*1.4826)
                return thr, med, sd, f"inline-max (per-line max over phases 40-95 ms, noisier half, {line_pct:.0f}th pct)", thr_strict
    if vals is None:
        nblk = max(1, int(round(1.0/hop)))
        nb = len(s)//nblk
        if nb >= 3:
            blk = s[:nb*nblk].reshape(nb, nblk)
            means = blk.mean(axis=1)
            k = max(1, int(np.ceil(0.1*nb)))
            sel = np.argsort(means)[:k]
            vals = blk[sel].ravel(); label = "lowest 10 % of 1 s blocks"
        else:
            vals = s; label = "all samples"
    thr = float(np.percentile(vals, pct)); thr_strict = float(np.percentile(vals, pct + (100.0 - pct)*0.8))
    med = float(np.median(vals)); sd = float(np.median(np.abs(vals - med))*1.4826)
    return thr, med, sd, label, thr_strict


# ------------------------------------------------------------------------------------------------- per-line tracking
def _smooth(s, hop, width_ms):
    """zero-phase Hann smoothing of s over width_ms (returns s itself when width_ms <= hop)."""
    n = int(round(width_ms/1000.0/hop))
    if n < 3: return s
    n += (n % 2 == 0)
    k = np.hanning(n + 2)[1:-1]; k /= k.sum()
    pad = n//2
    sp = np.pad(s, pad, mode="edge")
    return np.convolve(sp, k, mode="valid")


def track_lines(t: np.ndarray, s: np.ndarray, t_line0: float, period: float, params: Params, n_lines: int | None = None,
                line_offset: int = 0, noise_regions: list | None = None, threshold: float | None = None,
                fit_period: bool = True, tolerances_ms: tuple = (12.0, 8.0, 6.0, 5.0, 4.0, 4.0),
                smooth_ms: float = 0.0, subsample: bool = False, weighted: bool = False,
                fit_tol_ms: float | None = None, shape_test: bool = True, shape_side_ms: float = 15.0,
                shape_min_ratio: float = 0.5, refit: bool = True) -> LineSync:
    """For each line n: search s within +-params.line_search_ms of the expected pulse centre t_line0 + sync_ms/2 + n*period;
    record the peak; inliers = strength > noise 99.5th percentile (from the lowest-s 10 % of the time axis or a
    user-supplied noise stretch) and |residual| < params.line_inlier_ms after a robust iterative linear fit
    (refit 6 times, shrinking tolerance 12 -> 4 ms). Returns LineSync with the fitted t_line0/period.

    Additions over the analysis script: (a) the noise reference defaults to the pulse-free phases (40-95 ms) of the
    tracked lines themselves, which stays valid for short recordings that are signal throughout; noise_regions (absolute s)
    overrides it and the lowest-s 10 % of 1 s blocks is the fallback; (b) before the first fit the common offset of the
    above-threshold peaks is found as the mode of their residual histogram (1 ms bins, +-2 ms smoothing), so a start error
    up to the full search range (+-20 ms) is recovered even though the first tolerance is 12 ms; (c) n_lines/line_offset
    allow tracking an extended range (e.g. lines -60..299) for frame placement; (d) fit_period=False fits the offset only;
    (e) precision options: smooth_ms (Hann-smooth s before the peak search), subsample (parabolic sub-hop peak time),
    weighted (least squares weighted by the strength above the noise median), fit_tol_ms (final fit tolerance when
    tighter than line_inlier_ms; the inlier flag always uses line_inlier_ms); (f) shape_test: a peak counts as a pulse
    only if s falls back towards the noise level on both sides of it: with side = the median of s over
    [shape_side_ms-2, shape_side_ms+6] ms before and after the peak, (peak - max(sides)) >= shape_min_ratio * (peak - noise
    median). A 9 ms sync pulse seen through the 9 ms window is ~+-9 ms wide, so 13-21 ms is baseline for a real pulse, whereas the 30 ms VIS bits (1100/1200/1300 Hz, all leaking into the sync band) form plateaus
    that fail the test; this keeps header tones from being counted as sync pulses 1-2 lines before line 0.
    (g) refit=False evaluates the pulses on the GIVEN grid (t_line0, period) without the offset search and without the
    iterative fit, so that the per-line detections are indexed on a final timing (diagnostics of the resolved frame)."""
    t = np.asarray(t, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    N = int(params.n_lines if n_lines is None else n_lines)
    half_sync = ROBOT36.sync_ms/2000.0
    n_idx = np.arange(line_offset, line_offset + N)
    nf = n_idx.astype(np.float64)
    a = float(t_line0) + half_sync; b = float(period)
    hop = _hop_of(t)
    W = max(1, int(round(params.line_search_ms/1000.0/hop)))
    offs = np.arange(-W, W + 1)
    ns = len(s)
    s_pk = _smooth(s, hop, smooth_ms) if (smooth_ms > 0 and ns > 3) else s

    def gather(a, b):
        tc = a + b*nf
        if ns == 0:
            return tc, np.full(N, np.nan), tc.copy(), np.zeros(N, bool), np.zeros(N, np.int64)
        ic = np.round((tc - t[0])/hop).astype(np.int64)
        idx = ic[:, None] + offs[None, :]
        valid = (idx >= 0) & (idx < ns)
        vals = np.where(valid, s_pk[np.clip(idx, 0, ns - 1)], -np.inf)
        j = np.argmax(vals, axis=1)
        rows = np.arange(N)
        pk = vals[rows, j]
        ok = np.isfinite(pk)
        ipk = np.clip(idx[rows, j], 0, ns - 1)
        tpk = t[ipk]
        if subsample:
            im = np.clip(ipk - 1, 0, ns - 1); ip = np.clip(ipk + 1, 0, ns - 1)
            ya, yb, yc = s_pk[im], s_pk[ipk], s_pk[ip]
            den = ya - 2*yb + yc
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.where(den < 0, 0.5*(ya - yc)/den, 0.0)
            d = np.clip(np.nan_to_num(d), -1.0, 1.0)
            d[(ipk == 0) | (ipk == ns - 1)] = 0.0
            tpk = tpk + d*hop
        tpk = np.where(ok, tpk, tc)
        pk = np.where(ok, pk, np.nan)
        return tc, pk, tpk, ok, ipk

    if threshold is None:
        thr, nmed, nsd, label, thr_strict = _noise_reference(t, s, a - half_sync, b, n_idx, regions=noise_regions, hop=hop)
    else:
        thr = float(threshold); nmed, nsd, label = float("nan"), float("nan"), "user threshold"; thr_strict = thr
        if ns:
            _, nmed, nsd, _, thr_auto_strict = _noise_reference(t, s, a - half_sync, b, n_idx, regions=noise_regions, hop=hop)
            thr_strict = max(thr, thr_auto_strict)
    # side windows for the shape test: [side-2 ms, side+6 ms] on both sides of the peak; their MEDIANS are compared with the
    # peak (a median is stable against noise spikes for a real pulse and stays high for a tone plateau)
    side_lo = max(1, int(round((shape_side_ms - 2.0)/1000.0/hop))); side_hi = max(side_lo + 1, int(round((shape_side_ms + 6.0)/1000.0/hop)))
    side_offs = np.arange(side_lo, side_hi + 1)
    nref = nmed if np.isfinite(nmed) else float(np.median(s)) if ns else 0.0

    def shape_ok(pk, ipk, ok):
        """pulse-shape test (see docstring); lines without data pass as False, test disabled -> all True."""
        if not shape_test or ns == 0: return np.ones(N, bool), np.full(N, np.nan)
        im = np.clip(ipk[:, None] - side_offs[None, :], 0, ns - 1); ip = np.clip(ipk[:, None] + side_offs[None, :], 0, ns - 1)
        sides = np.maximum(np.median(s[im], axis=1), np.median(s[ip], axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = (pk - sides)/np.maximum(pk - nref, _TINY)
        ratio = np.where(ok & (pk > nref), ratio, np.nan)
        return ok & (np.nan_to_num(ratio, nan=-1.0) >= shape_min_ratio), ratio
    # ---- iteration 0: common offset = mode of the residuals of the above-threshold peaks
    tc, pk, tpk, ok, ipk = gather(a, b)
    cand = ok & (pk > thr) & shape_ok(pk, ipk, ok)[0]
    n_cand0 = int(cand.sum())
    if refit and n_cand0 >= 3:
        r_ms = (tpk - tc)[cand]*1000.0
        span = W*hop*1000.0
        edges = np.arange(-span - 0.5, span + 1.5, 1.0)
        h, _ = np.histogram(r_ms, bins=edges)
        hs = np.convolve(h, np.ones(5), mode="same")
        km = int(np.argmax(hs)); mode_ms = 0.5*(edges[km] + edges[km + 1])
        near = r_ms[np.abs(r_ms - mode_ms) <= 2.5]
        if len(near) >= 2 and hs[km] >= 3:
            a += float(np.median(near))/1000.0
    # ---- robust iterative linear fit with shrinking tolerance
    n_it = 0; converged = False
    tols = list(tolerances_ms)
    if fit_tol_ms is not None and fit_tol_ms < tols[-1]:
        tols = tols + [float(fit_tol_ms)]*2
    scale = (thr - nmed) if (np.isfinite(nmed) and thr > nmed) else None

    def fit_weights(pk_in):
        if not weighted or scale is None: return None
        return np.sqrt(np.clip((pk_in - nmed)/scale, 0.5, 4.0))

    for tol in (tols if refit else []):
        tc, pk, tpk, ok, ipk = gather(a, b)
        resid = tpk - tc
        inl = ok & (pk > thr) & (np.abs(resid) < tol/1000.0) & shape_ok(pk, ipk, ok)[0]
        n_it += 1
        if inl.sum() >= 5:
            if fit_period:
                db, da = np.polyfit(nf[inl], resid[inl], 1, w=fit_weights(pk[inl]))
                a += float(da); b += float(db)
            else:
                a += float(np.median(resid[inl]))
            converged = True
        elif inl.sum() >= 2 and not fit_period:
            a += float(np.median(resid[inl]))
    # ---- final evaluation
    tc, pk, tpk, ok, ipk = gather(a, b)
    resid = tpk - tc
    shp_ok, shp = shape_ok(pk, ipk, ok)
    tol_s = params.line_inlier_ms/1000.0
    inl = ok & (pk > thr) & (np.abs(resid) < tol_s) & shp_ok
    n_inl = int(inl.sum())
    # expected false inliers from the uniform background of above-threshold peaks OUTSIDE the tolerance (per ms of window)
    cand_all = ok & (pk > thr)
    n_out = int(np.sum(cand_all & (np.abs(resid) >= tol_s)))
    span_out_ms = max(2.0*W*hop*1000.0 - 2.0*params.line_inlier_ms, 1e-6)
    exp_false = float(n_out/span_out_ms*2.0*params.line_inlier_ms)
    excess_z = float((n_inl - exp_false)/np.sqrt(exp_false + 1.0))
    strong = inl & (pk > thr_strict)
    runs = []
    i = 0
    while i < N:
        if inl[i]:
            j = i
            while j + 1 < N and inl[j + 1]: j += 1
            runs.append((int(n_idx[i]), int(n_idx[j]))); i = j + 1
        else:
            i += 1
    sd_ms = float(np.std(resid[inl])*1000.0) if n_inl >= 2 else float("nan")
    p_se = t_se = piv = float("nan")
    if n_inl >= 3:
        r = resid[inl]; nn = nf[inl]; piv = float(nn.mean())
        s2 = float(np.sum(r*r)/(n_inl - 2)) if n_inl > 2 else float(np.var(r))
        sxx = float(np.sum((nn - piv)**2))
        p_se = float(np.sqrt(s2/sxx)) if sxx > 0 else float("nan")
        t_se = float(np.sqrt(s2/n_inl))
    return LineSync(t_pulse=tpk, strength=np.where(ok, pk, 0.0), inlier=inl, t_line0=float(a - half_sync), period=float(b),
                    resid_ms_sd=sd_ms, n_inliers=n_inl, threshold=float(thr), runs=runs, line_index=n_idx, t_expected=tc,
                    resid_ms=resid*1000.0, valid=ok, noise_median=nmed, noise_sd=nsd, noise_ref=label,
                    n_candidates=int(np.sum(ok & (pk > thr))), fit_iterations=n_it, converged=converged,
                    shape=shp, n_shape_rejected=int(np.sum(ok & (pk > thr) & (np.abs(resid) < tol_s) & ~shp_ok)),
                    period_se=p_se, t_pivot_se=t_se, pivot_line=piv, threshold_strict=float(thr_strict), strong=strong,
                    n_strong=int(strong.sum()), expected_false_inliers=exp_false, inlier_excess_z=excess_z)
