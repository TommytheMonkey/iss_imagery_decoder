"""Combine redundant timing estimates (VIS end, sync fit, ACF+fold, coherence) into one Timing with confidence.

Design (SPEC section 4, `timing.py`):
  * normal path   line 0 from the VIS stop-bit end, refined to the sync-fit phase when the sync fit is credible and agrees
                  within 3 ms ('vis+sync'); period from the sync fit if >= 8 inliers and within tolerance, else the ACF comb,
                  else nominal; agreement['vis_vs_sync_ms'] = (sync-fit t_line0 - VIS t_line0) reduced modulo the period.
  * recovery path period from the sync fit / ACF comb (agreement reported in ppm); phase from the sync fit (which was
                  initialised from fold_phase); line 0 placed on the pulse grid so that a 240-line frame holds every inlier
                  pulse and lies inside signal_span (carrier onset/offset, +-frame_slack_s): anchor priority = header-end
                  signature (stop bit before / picture after line 0) > a Robot 36 VIS candidate with confidence >=
                  vis_anchor_min_conf, valid parity and agreeing bit tones (mode auto only) > the first inlier pulse that is
                  followed by enough further inliers within frame_start_window_lines lines ("dense start": 'enough' is the
                  count a uniform false-inlier background at the calibrated rate reaches with probability <=
                  frame_start_false_alarm) > the first supported pulse > the midpoint of the feasible range; pulses that
                  cannot share a 240-line frame with the last core pulse, and pulses outside the 240-line window holding the
                  most pulses, are ignored; the start line(s) are checked for picture content (per-line tone SNR step from
                  the pre-frame noise to the picture, `line_snr_fn`) so that a false pulse adjacent to the frame cannot start
                  it; the integer-line ambiguity (feasible range and the expected number of undetected leading lines) is
                  reported and enters the confidence.
  * structure     a pulse train on every other line only (300 ms) or an ACF comb that is stronger at twice the nominal
                  period flags a non-Robot-36 line structure (Robot 72 shares header and sync pulse with Robot 36); the
                  confidence is cut and agreement['robot36_structure_ok'] is False.
  * coherence     coordinate search over the coh_* grid of the score returned by coherence_score(decode_quick, ...);
                  the refined values are adopted only when the gain >= coh_min_gain and the sync fit is weak (< 20 inliers)
                  or absent (the sync fit is more precise than the 0.5 ms grid); the gain is always reported.
  * confidence    weighted combination of the comb significance, inlier count, VIS confidence, estimator agreement and
                  coherence gain (weights depend on the path), scaled by the placement ambiguity and by the period's
                  standard error; never raises for weak signals.
Optional extras over the skeleton signature (all keyword, all default None): sync_signal=(t, s) lets the recovery path
re-track an extended line range for the frame placement and compute a frame-local comb z; parity_fn(t0, P) -> fraction of
lines whose separator tone matches the even/odd pattern (0.5 = random) fixes the parity of the integer-line placement;
line_snr_fn(t0, P, n_lines) -> per-line tone SNR (dB) of n_lines lines starting at t0 gives the picture-presence evidence;
audio_span=(t_first, t_last) of the analysed audio keeps an unconstrained midpoint placement inside the audio."""
from dataclasses import dataclass, field
import math
import numpy as np
from .params import Params
from .header import VISCandidate
from .sync import PeriodEstimate, LineSync, track_lines, estimate_period_acf
from .modes import ROBOT36, VIS_TOTAL_MS


@dataclass
class Timing:
    t_line0: float
    period: float
    source: str                        # e.g. 'vis+sync', 'sync-fit', 'acf-fold', 'override'
    confidence: float                  # 0..1
    agreement: dict = field(default_factory=dict)   # e.g. {'vis_vs_sync_ms': 0.5, 'acf_vs_sync_ppm': 3, 'coherence_gain': ...}
    alternatives: list = field(default_factory=list) # other (t_line0, period, source, score)
    notes: str = ""


def _p(params, name, default):
    return getattr(params, name, default) if params is not None else default


def _wrap(d, P):
    return (d + P/2.0) % P - P/2.0


def _clip01(x):
    return float(min(1.0, max(0.0, x)))


# ------------------------------------------------------------------------------------------------- coherence
def coherence_score(decode_quick, t_line0: float, period: float, block_lines: int = 40) -> float:
    """decode_quick(t_line0, period) -> (Y, sync_contrast); score = adjacent-row correlation of Y (row means removed)
    in the best 40-line block + sync-column contrast. Used only for small refinements and tie-breaks."""
    Y, sc = decode_quick(t_line0, period)
    sc = float(sc) if sc is not None and np.isfinite(sc) else 0.0
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or Y.shape[0] < 2 or Y.shape[1] < 2:
        return sc
    R = Y - Y.mean(axis=1, keepdims=True)
    num = (R[:-1]*R[1:]).sum(axis=1)
    den = np.sqrt((R[:-1]**2).sum(axis=1)*(R[1:]**2).sum(axis=1)) + 1e-9
    rc = num/den                                                   # one value per adjacent row pair
    k = max(1, min(block_lines - 1, len(rc)))
    if k >= len(rc):
        best = float(rc.mean())
    else:
        c = np.concatenate([[0.0], np.cumsum(rc)])
        best = float(np.max((c[k:] - c[:-k])/k))
    return best + sc


def _coherence_refine(decode_quick, t0, P, params):
    """coordinate search (t0 pass, period pass, repeated once) over the coh_* grid centred on (t0, P).
    Returns (t0_best, P_best, score_at_start, best_score, n_evaluations)."""
    span = params.coh_t0_span_ms/1000.0; step = max(params.coh_t0_step_ms/1000.0, 1e-6)
    ppm_span = params.coh_ppm_span; ppm_step = max(params.coh_ppm_step, 1e-3)
    ts = t0 + np.arange(-span, span + step/2, step)
    Ps = P*(1.0 + np.arange(-ppm_span, ppm_span + ppm_step/2, ppm_step)*1e-6)
    cache = {}

    def sc(tt, pp):
        key = (round(float(tt), 7), round(float(pp), 11))
        if key not in cache:
            cache[key] = coherence_score(decode_quick, float(tt), float(pp))
        return cache[key]

    s0 = sc(t0, P)
    bt, bP, bs = float(t0), float(P), s0
    for _ in range(2):
        changed = False
        vals = np.array([sc(tt, bP) for tt in ts]); i = int(np.argmax(vals))
        if vals[i] > bs: bt, bs, changed = float(ts[i]), float(vals[i]), True
        vals = np.array([sc(bt, pp) for pp in Ps]); i = int(np.argmax(vals))
        if vals[i] > bs: bP, bs, changed = float(Ps[i]), float(vals[i]), True
        if not changed: break
    return bt, bP, float(s0), float(bs), len(cache)


# ------------------------------------------------------------------------------------------------- start rule helpers
def _binom_tail(n: int, k: int, p: float) -> float:
    """P(Bin(n, p) >= k)."""
    if k <= 0: return 1.0
    if k > n: return 0.0
    return float(sum(math.comb(n, j)*p**j*(1.0 - p)**(n - j) for j in range(k, n + 1)))


def start_rule_min_count(p0: float, window_lines: int, false_alarm: float) -> int:
    """Smallest m (>= 2) such that an inlier pulse followed by >= m-1 further inliers within the next window_lines-1 lines
    is improbable under a uniform false-inlier background of rate p0 per line: P(Bin(window_lines-1, p0) >= m-1) <=
    false_alarm. With p0 = 0.02 and a 12-line window, m = 4; with p0 = 0.04, m = 5. A genuine frame start at 60 %
    per-line detection reaches m = 4 in 99 % of the cases, at 30 % detection in 69 %."""
    n = max(1, int(window_lines) - 1)
    p0 = float(min(max(p0, 1e-4), 0.5))
    for m in range(2, n + 2):
        if _binom_tail(n, m - 1, p0) <= false_alarm:
            return m
    return n + 1


def first_dense_start(I, window_lines: int, m_min: int, window2_lines: int | None = None, m2_min: int | None = None):
    """First index s in the inlier line indices I with >= m_min inliers in [s, s + window_lines) (or, when a second window
    is given, >= m2_min inliers in [s, s + window2_lines): the longer window catches a sparsely detected start); None when
    there is none."""
    I = np.sort(np.asarray(I, dtype=np.int64))
    for s in I:
        if int(np.sum((I >= s) & (I < s + window_lines))) >= m_min:
            return int(s)
        if window2_lines is not None and int(np.sum((I >= s) & (I < s + window2_lines))) >= m2_min:
            return int(s)
    return None


# per-line tone SNR (decode.line_snr_db, dB re 2400 Hz) of pure receiver noise: the control decodes of the reference
# recording and of the synthetic rtl_fm noise give a median of -8.9..-9.2 dB and a 99th percentile of -7.2..-7.4 dB
NOISE_SNR_PRIOR_DB = -8.8
NOISE_SNR_MAX_DB = -7.0          # a line below this carries no detectable tone (noise-like)
HEADER_END_MIN_SHARE = 0.15      # minimum 1200 Hz share in the 24 ms before a line start for a stop-bit signature (noise ~0.07)


def _picture_profile(line_snr_fn, t_grid0, P, k_from, k_to):
    """per-line tone SNR (dB) for pulse-grid lines k_from..k_to-1 -> dict {k: snr} (nan entries dropped)."""
    n = int(k_to - k_from)
    if n <= 0: return {}
    v = np.asarray(line_snr_fn(float(t_grid0 + k_from*P), float(P), n), dtype=np.float64)
    return {int(k_from + i): float(v[i]) for i in range(min(n, len(v))) if np.isfinite(v[i])}


def _picture_levels(prof, I_set, L0, k_lo, K=6):
    """(noise_level, picture_level) of the per-line tone SNR around a candidate start L0: noise = median of the lines in
    [k_lo-K, L0) that are below NOISE_SNR_MAX_DB (>= 3 of them, else the prior), picture = median of the inlier lines in
    (L0, L0+20) at or above NOISE_SNR_MAX_DB (>= 4 of them, else None: the picture level is not measurable)."""
    pre = [prof[k] for k in range(k_lo - K, L0) if k in prof and prof[k] < NOISE_SNR_MAX_DB]
    noise_level = float(np.median(pre)) if len(pre) >= 3 else NOISE_SNR_PRIOR_DB
    post = [prof[k] for k in range(L0 + 1, L0 + 20) if k in prof and k in I_set and prof[k] >= NOISE_SNR_MAX_DB]
    picture_level = float(np.median(post)) if len(post) >= 4 else None
    return noise_level, picture_level


def _noise_line(prof, k, post_db):
    """line k has no detectable picture content (below the noise ceiling and well below the picture level)."""
    return (k in prof) and prof[k] < NOISE_SNR_MAX_DB and prof[k] < post_db - 2.0


# ------------------------------------------------------------------------------------------------- frame placement
def _longest_run(runs):
    if not runs: return None
    best = None
    for (a, b) in runs:
        if best is None or (b - a) > (best[1] - best[0]): best = (a, b)
    return best


def _header_end_scores(t, s, t_grid0, P, ks, N):
    """Header-end (stop-bit) signature per candidate line index k: mean s in the 30 ms BEFORE the line start (the VIS stop
    bit, 1200 Hz, when the header survived) minus mean s in the Y segment AFTER the sync pulse (18-60 ms). Only the true
    line 0 is preceded by 1200-like content and followed by picture tones (a header bit line is 1200-like on both sides,
    an image line on neither). Returns (z per k, median, robust sd) with z relative to the image lines k+1..k+N/4."""
    t = np.asarray(t, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    hop = float(np.median(np.diff(t[:min(len(t), 100000)]))) if len(t) > 1 else 0.0005
    pre = np.arange(int(round(-0.027/hop)), int(round(-0.003/hop)))
    post = np.arange(int(round(0.018/hop)), int(round(0.060/hop)))

    def score(kk):
        i0 = np.round((t_grid0 + kk*P - t[0])/hop).astype(np.int64)
        ia = i0[:, None] + pre[None, :]; ib = i0[:, None] + post[None, :]
        ok = (ia.min(axis=1) >= 0) & (ib.max(axis=1) < len(s))
        out = np.full(len(kk), np.nan)
        if ok.any():
            out[ok] = s[ia[ok]].mean(axis=1) - s[ib[ok]].mean(axis=1)
        return out

    def before(kk):
        i0 = np.round((t_grid0 + kk*P - t[0])/hop).astype(np.int64)
        ia = i0[:, None] + pre[None, :]
        ok = (ia.min(axis=1) >= 0) & (ia.max(axis=1) < len(s))
        out = np.full(len(kk), np.nan)
        if ok.any(): out[ok] = s[ia[ok]].mean(axis=1)
        return out

    ks = np.asarray(ks, dtype=np.int64)
    if len(ks) == 0: return np.zeros(0), float("nan"), float("nan")
    sc = score(ks)
    ref = score(np.arange(ks.max() + 1, ks.max() + 1 + max(20, N//4)))
    ref = ref[np.isfinite(ref)]
    if len(ref) < 10: return np.full(len(ks), np.nan), float("nan"), float("nan")
    med = float(np.median(ref)); sd = float(np.median(np.abs(ref - med))*1.4826)
    # a stop bit is a 1200 Hz tone: its share must exceed the image-line level by a real margin (0.08) even when the
    # reference scatter is tiny (noise-free synthetic signals), so the z denominator is floored at 0.08/6
    sd = max(sd, 0.08/6.0)
    z = (sc - med)/sd
    # and the share before the line start must be tone-like in absolute terms: noise before a header-less frame start
    # followed by picture gives the same sign of difference (noise 0.07-0.13 vs picture 0.03) but never a 1200 Hz share of a
    # captured stop bit (0.4 on the reference recording at CNR -3 dB)
    b = before(ks)
    floor_share = max(HEADER_END_MIN_SHARE, 2.5*float(np.median(s[:min(len(s), 2000000)])))
    z = np.where(np.isfinite(b) & (b < floor_share), np.minimum(z, 0.0), z)
    return z, med, sd


def _frame_core(I, N):
    """Split the inlier line indices into the ones inside the N-line window that holds the most of them (the earliest such
    window) and the ones outside it. Isolated false pulses far from the frame (noise gives ~0.6 % strong false inliers per
    line) must not widen or shift the feasible range, so only the core defines it."""
    I = np.sort(np.asarray(I, dtype=np.int64))
    if len(I) == 0:
        return I, I
    ends = np.searchsorted(I, I + N, side="left")          # number of inliers below I[j] + N
    counts = ends - np.arange(len(I))                      # inliers in [I[j], I[j] + N)
    j = int(np.argmax(counts))
    return I[j:ends[j]], np.concatenate([I[:j], I[ends[j]:]])


def _vis_anchor_ok(c, min_conf):
    """A VIS candidate may anchor the recovery placement only when it is a Robot 36 code read with valid parity, at least
    6 of its 10 bit tones agree with the code and its confidence is >= min_conf (a false image-band candidate that reads
    code 8 by chance has random parity and tone agreement)."""
    return (c.vis_code == ROBOT36.vis and c.confidence >= min_conf and bool(c.parity_ok)
            and int(getattr(c, "n_freq_ok", 10)) >= 6)


def _place_frame(ls: LineSync, N: int, signal_span, vis, params, sync_signal=None, use_header_end=True,
                 line_snr_fn=None, audio_span=None):
    """Choose the integer line index L0 (on the pulse grid of ls: line n starts at ls.t_line0 + n*ls.period) that is
    line 0 of the frame. Returns (L0, info) where info holds the feasible range, the anchor used, the alternatives and the
    placement ambiguity (expected number of lines by which the start may be off).

    Feasible range: every pulse of the core (the N-line window holding the most strong inliers) must lie inside the frame,
    intersected with signal_span (carrier onset/offset +- frame_slack_s) when that is consistent. The upper bound is the
    'dense start': the first inlier followed by enough inliers within frame_start_window_lines lines (calibrated against
    the false-inlier rate, see start_rule_min_count); isolated earlier pulses are pre-frame noise. Anchor priority:
    header-end signature > credible Robot 36 VIS candidate (mode auto only; the caller passes [] in mode recovery) >
    dense start > first supported inlier (another one within 10 lines) > first inlier > midpoint of the feasible range.
    When line_snr_fn is given, a pulse-based start whose own line has noise-level tone SNR while the following lines show
    picture content is moved to the first picture line (at most 3 lines)."""
    P = ls.period
    t_grid0 = ls.t_line0
    slack = float(_p(params, "frame_slack_s", 0.5))
    run_len = max(1, int(_p(params, "frame_run_lines", 5))); run_min = max(1, int(_p(params, "frame_run_min", 5)))
    min_vconf = float(_p(params, "vis_anchor_min_conf", 0.2))
    Wd = max(3, int(_p(params, "frame_start_window_lines", 12)))
    alpha = float(_p(params, "frame_start_false_alarm", 1e-3))
    step_min = float(_p(params, "picture_step_min_db", 3.0))
    idx = np.asarray(ls.line_index if ls.line_index is not None else np.arange(len(ls.inlier)), dtype=np.int64)
    inl_b = np.asarray(ls.inlier, bool)
    # calibrated false-inlier rate per line (expected false inliers over the tracked lines)
    exp_false = float(getattr(ls, "expected_false_inliers", float("nan")))
    n_tracked = max(1, len(inl_b))
    p0 = exp_false/n_tracked if np.isfinite(exp_false) and exp_false >= 0 else 0.02
    p0 = float(min(max(p0, 0.002), 0.2))
    m_min = start_rule_min_count(p0, Wd, alpha)
    m2_min = start_rule_min_count(p0, 2*Wd, alpha)
    # the feasible range is defined by the STRONG inliers (strict threshold, ~0.6 % false rate per line) when there are
    # enough of them, because a single false inlier far from the frame would widen the range by up to 240 lines
    strong = getattr(ls, "strong", None)
    all_inl = idx[inl_b]
    if strong is not None and int(np.sum(strong)) >= 3:
        I_s = idx[np.asarray(strong, bool)]; basis = "strong inliers"
    else:
        I_s = all_inl; basis = "all inliers"
    core_s, outside = _frame_core(I_s, N)
    # pulses considered for the frame START: every inlier (strong or not) from 20 lines before the first strong core pulse
    # to the last one; the density test below must not be broken by a genuine but weak pulse
    if len(core_s):
        I = all_inl[(all_inl >= int(core_s.min()) - 20) & (all_inl <= int(core_s.max()))]
        if len(I) == 0: I = core_s
    else:
        I = core_s
    info = dict(n_inliers=int(len(I)), n_strong_core=int(len(core_s)), false_inlier_rate_per_line=p0, start_rule_min_count=int(m_min),
                start_rule_window_lines=int(Wd),
                notes=[f"feasible range from {len(core_s)} {basis} ({len(I)} pulses considered for the start; start rule: >= {m_min} inliers "
                       f"within {Wd} lines or >= {m2_min} within {2*Wd}, false-inlier rate {p0:.3f}/line)"], alternatives={})
    if len(outside):
        info["notes"].append(f"{len(outside)} pulse(s) outside the {N}-line window holding the most pulses ignored for the placement "
                             f"(lines {outside[:8].tolist()}{'...' if len(outside) > 8 else ''})")
        info["n_outside_core"] = int(len(outside))
    anchor_kind = None
    if len(I) == 0:
        lo_p, hi_p = -10**6, 10**6
        i_anchor = None
        info["notes"].append("no inlier pulses: frame placed on the provisional grid")
    else:
        i_last = int(core_s.max()) if len(core_s) else int(I.max())
        lo_p = i_last - (N - 1)
        # a pulse that cannot share a frame with the last core pulse is noise (this is what produced lo > hi before)
        dropped = I[I < lo_p]
        if len(dropped):
            info["notes"].append(f"{len(dropped)} pulse(s) more than {N - 1} lines before the last core pulse ignored (lines {dropped[:8].tolist()})")
            I = I[I >= lo_p]
            if len(I) == 0: I = core_s
        I = np.sort(I)
        s_dense = first_dense_start(I, Wd, m_min, 2*Wd, m2_min)
        sup = [int(i) for i in I if np.any((I > i) & (I <= i + 10))]
        if s_dense is not None:
            i_anchor, anchor_kind = s_dense, "dense"
        elif sup:
            i_anchor, anchor_kind = sup[0], "supported"
        else:
            i_anchor, anchor_kind = int(I.min()), "first"
        hi_p = int(i_anchor)
        pre = I[I < i_anchor]
        if len(pre):
            info["notes"].append(f"{len(pre)} isolated pulse(s) before the first dense start (>= {m_min} inliers within {Wd} lines) treated as "
                                 f"pre-frame noise (lines {pre[:8].tolist()}{'...' if len(pre) > 8 else ''})")
            info["n_pre_frame_pulses"] = int(len(pre))
    lo, hi = lo_p, hi_p
    if signal_span is not None:
        t_on, t_off = float(signal_span[0]), float(signal_span[1])
        lo_s = int(math.ceil((t_on - slack - t_grid0)/P))
        hi_s = int(math.floor((t_off + slack - N*P - t_grid0)/P))
        info["span_range"] = (lo_s, hi_s)
        lo2, hi2 = max(lo, lo_s), min(hi, hi_s)
        if lo2 <= hi2:
            lo, hi = lo2, hi2
        else:
            info["notes"].append(f"signal_span range {lo_s}..{hi_s} conflicts with the pulse range {lo_p}..{hi_p}; pulses win")
    if lo < -10**5 and hi > 10**5:                       # nothing constrains the placement at all
        lo = hi = 0
    elif lo < -10**5:
        lo = hi
    elif hi > 10**5:
        hi = lo
    if hi < lo:                                          # cannot happen after the drop above, kept as a hard guard
        info["notes"].append(f"feasible range inverted ({lo}..{hi}); collapsed to {lo}")
        hi = lo
    info["feasible"] = (int(lo), int(hi))
    clip = lambda k: int(np.clip(int(k), lo, hi))
    alts = {}
    supported = False
    if len(I):
        i_first = int(I.min())
        supported = bool(np.any((I > i_first) & (I <= i_first + 10)))
        alts["first-pulse"] = clip(i_first)
        if anchor_kind == "dense": alts["first-dense-start"] = clip(i_anchor)
        # the previous rule (first pulse that starts a run of >= run_min inliers within run_len lines) is reported for reference
        starts = [int(i) for i in I if int(np.sum((I >= i) & (I < i + run_len))) >= run_min]
        if starts: alts["first-run-start"] = clip(starts[0])
        sup = [int(i) for i in I if np.any((I > i) & (I <= i + 10))]
        if sup: alts["first-supported-pulse"] = clip(sup[0])
        run = _longest_run(ls.runs)
        if run is not None: alts["longest-run-start"] = clip(run[0])
    mid = int(math.floor((lo + hi)/2.0 + 0.5))
    if audio_span is not None and len(I) == 0:
        # unconstrained placement (no pulses): keep the frame inside the analysed audio when the feasible range allows it
        lo_a = int(math.ceil((float(audio_span[0]) - t_grid0)/P)); hi_a = int(math.floor((float(audio_span[1]) - N*P - t_grid0)/P))
        lo3, hi3 = max(lo, lo_a), min(hi, hi_a)
        if lo3 <= hi3: mid = int(math.floor((lo3 + hi3)/2.0 + 0.5))
    alts["midpoint"] = clip(mid)
    if signal_span is not None:
        alts["onset+header"] = clip(int(round((t_on + VIS_TOTAL_MS/1000.0 - t_grid0)/P)))
        alts["onset"] = clip(int(round((t_on - t_grid0)/P)))
        alts["offset-240"] = clip(int(round((t_off - N*P - t_grid0)/P)))
    # only a Robot 36 header candidate of some credibility may anchor (an image-band false detection or a candidate whose
    # code could not be read never does)
    vis_ok = [c for c in (vis or []) if _vis_anchor_ok(c, min_vconf)]
    vis_anchor = None
    for c in sorted(vis_ok, key=lambda c: -c.confidence):
        k = int(round((c.t_line0 - t_grid0)/P))
        if lo - 1 <= k <= hi + 1:
            vis_anchor = (clip(k), c); break
    header_end = None
    # the VIS header occupies the 6 lines before line 0 and its 1100-1300 Hz bits can pass as sync pulses: an anchor that
    # places line 0 up to `max_hdr` lines AFTER the first inlier is accepted when at most 3 inliers precede it within 7 lines
    def header_override(k):
        if not len(I): return True
        before = I[I < k]
        return len(before) <= 3 and (len(before) == 0 or k - int(before.min()) <= 7)
    if use_header_end and sync_signal is not None and hi - lo <= 400:
        try:
            ks = np.arange(lo - 2, hi + 8)
            z, _, _ = _header_end_scores(sync_signal[0], sync_signal[1], t_grid0, P, ks, N)
            if len(z) and np.isfinite(z).any():
                order = np.argsort(np.nan_to_num(z, nan=-np.inf))[::-1]
                zb = float(z[order[0]]); z2 = float(z[order[1]]) if len(order) > 1 and np.isfinite(z[order[1]]) else -np.inf
                kb = int(ks[order[0]])
                info["header_end_z"] = zb; info["header_end_k"] = kb; info["header_end_runner_up_z"] = z2
                if zb >= 6.0 and zb - z2 >= 3.0:
                    if lo - 1 <= kb <= hi + 1:
                        header_end = clip(kb)
                    elif kb > hi and header_override(kb):
                        header_end = kb
                        info["notes"].append(f"{int(np.sum(I < kb))} inlier pulse(s) up to {kb - int(I.min())} line(s) before the header-end line 0 treated as header tones")
        except Exception as e:  # noqa
            info["notes"].append(f"header-end anchor failed: {e!r}")
    if vis_anchor is None and vis_ok:
        for c in sorted(vis_ok, key=lambda c: -c.confidence):
            k = int(round((c.t_line0 - t_grid0)/P))
            if hi < k <= hi + 7 and header_override(k):
                vis_anchor = (k, c)
                info["notes"].append(f"{int(np.sum(I < k))} inlier pulse(s) before the VIS-anchored line 0 treated as header tones"); break
    amb = 0.0
    if header_end is not None:
        alts["header-end"] = header_end
        L0 = header_end; anchor = f"header-end signature (stop bit before / picture after, z {info['header_end_z']:.1f})"; kind = "header-end"
    elif vis_anchor is not None:
        alts["vis-anchor"] = vis_anchor[0]
        L0 = vis_anchor[0]; anchor = f"vis-anchor (VIS candidate conf {vis_anchor[1].confidence:.2f}, code {vis_anchor[1].vis_code})"; kind = "vis"
    elif "first-dense-start" in alts:
        L0 = alts["first-dense-start"]; anchor = f"first dense start (inlier pulse with >= {m_min} inliers within {Wd} lines)"; kind = "dense"
    elif "first-supported-pulse" in alts:
        L0 = alts["first-supported-pulse"]; anchor = "first supported inlier pulse"; kind = "supported"
    elif "first-pulse" in alts and supported:
        L0 = alts["first-pulse"]; anchor = "first inlier pulse"; kind = "first"
    else:
        L0 = alts["midpoint"]; anchor = "midpoint of the feasible range"; kind = "midpoint"
    # picture presence of the start line(s): when the picture level is clearly resolvable above the pre-frame noise level
    # (>= picture_step_min_db + 1 dB), leading pulses whose own line sits in the noise half of that step are false pulses
    # (noise lines adjacent to the frame) and the start moves to the first pulse on a picture line. When the step is not
    # resolvable (fade-in, very low CNR, a header before the frame) the pulses alone decide: a genuine weak line keeps its pulse.
    onset = None
    if kind in ("dense", "supported", "first") and line_snr_fn is not None and len(I):
        try:
            K = 6
            k_lo = max(int(lo), int(L0) - 30)
            prof = _picture_profile(line_snr_fn, t_grid0, P, k_lo - K, int(L0) + 36)
            I_set = set(int(v) for v in I)
            noise_level, picture_level = _picture_levels(prof, I_set, int(L0), k_lo, K)
            info["picture_levels_db"] = dict(noise=noise_level, picture=picture_level)
            if picture_level is not None and picture_level - noise_level >= step_min + 1.0:
                step = picture_level - noise_level
                thr_noise = noise_level + 0.35*step
                k = int(L0); moved = []
                while (k in prof) and prof[k] < thr_noise and (k - int(L0)) <= 24:
                    nxt = [int(v) for v in I if v > k]
                    if not nxt: break
                    moved.append(k); k = nxt[0]
                if moved and (k in prof) and prof[k] >= thr_noise:
                    onset = dict(noise_db=noise_level, picture_db=picture_level, step_db=step, moved=moved)
                    info["notes"].append(f"start line(s) {moved[:8]} have noise-level tone SNR ({', '.join(f'{prof[q]:+.1f}' for q in moved[:8])} dB; pre-frame noise "
                                         f"{noise_level:+.1f}, picture {picture_level:+.1f} dB): their pulses are noise, line 0 moved to grid line {k}")
                    L0 = k; anchor += f" + picture-presence check (+{k - moved[0]} line)"; kind = "onset"
            # fade-in at the start: when the tone SNR keeps rising over the first lines the transmission may have begun in a
            # fade and the first lines (undetectable: noise-level SNR, no pulses) would be lost; reported as extra ambiguity
            first = [prof[k] for k in range(L0, L0 + 5) if k in prof]; later = [prof[k] for k in range(L0 + 6, L0 + 16) if k in prof]
            if len(first) >= 3 and len(later) >= 5:
                rise = float(np.median(later) - np.median(first))
                info["fade_in_db"] = rise
                if rise > 1.5:
                    info["fade_in_ambiguity_lines"] = (rise - 1.5)/2.0
                    info["notes"].append(f"tone SNR rises by {rise:.1f} dB over the first lines (fade-in): lines before the first detected pulse may be lost")
        except Exception as e:  # noqa
            info["notes"].append(f"picture-presence check failed: {e!r}")
    # pulses before the chosen start in excess of the false-inlier expectation: they may be genuine (sparse detection at the
    # true start) and widen the start uncertainty towards the first of them
    if len(I):
        pre_all = I[(I < L0) & (I >= lo)]
        n_pre = int(len(pre_all)); exp_pre = p0*max(0, int(L0) - int(lo))
        info["pre_start_pulses"] = dict(n=n_pre, expected_false=exp_pre)
        excess_frac = max(0.0, (n_pre - exp_pre)/max(n_pre, 1)) if n_pre > exp_pre + 2.0*math.sqrt(exp_pre + 1.0) else 0.0
        info["pre_start_excess_lines"] = excess_frac*float(int(L0) - int(pre_all.min())) if n_pre else 0.0
        if info["pre_start_excess_lines"] > 0.5:
            info["notes"].append(f"{n_pre} pulses before the chosen start where {exp_pre:.1f} false ones are expected: the frame may start up to "
                                 f"{int(L0) - int(pre_all.min())} lines earlier (sparse detection at the start)")
    # placement ambiguity: expected number of undetected leading lines (geometric, from the local detection density) for a
    # pulse-based start, half the feasible range for the midpoint, 0 for an evidence anchor; bounded by the feasible range
    if kind in ("header-end", "vis"):
        amb = 0.0
    elif kind == "onset":
        amb = 0.3
    elif kind == "dense":
        p1 = float(np.sum((I >= L0) & (I < L0 + Wd)))/Wd
        amb = min((1.0 - p1)/max(p1, 0.05), float(max(0, L0 - lo)))
    elif kind in ("supported", "first"):
        amb = float(min(max(0, L0 - lo), 6))
    else:
        amb = 0.5*float(hi - lo)
    if kind in ("dense", "supported", "first", "onset"):
        amb += float(info.get("fade_in_ambiguity_lines", 0.0))
        amb = max(amb, min(10.0, float(info.get("pre_start_excess_lines", 0.0))))
    # a weakly anchored frame (no dense start: isolated pulses only, or nothing) is kept inside the analysed audio when the
    # feasible range allows it; its pulses are mostly noise and a line 0 before the audio start is meaningless
    if kind in ("supported", "first", "midpoint") and audio_span is not None:
        lo_a = int(math.ceil((float(audio_span[0]) - t_grid0)/P))
        if L0 < lo_a <= hi:
            info["notes"].append(f"weakly anchored start moved from grid line {L0} to the first line inside the audio ({lo_a})")
            L0 = lo_a
    info["anchor"] = anchor; info["anchor_kind"] = kind; info["alternatives"] = alts; info["L0"] = int(L0)
    info["placement_ambiguity_lines"] = float(amb)
    return int(L0), info


def _pulse_pattern(ls: LineSync, L0: int, N: int) -> dict:
    """Even/odd distribution of the inlier pulses inside the frame. A Robot 72 transmission (300 ms lines) tracked on the
    150 ms grid puts every pulse on the same line parity; Robot 36 gives ~50/50. 'alternating' when >= 12 pulses, >= 90 %
    on one parity and the binomial z >= 4."""
    idx = np.asarray(ls.line_index if ls.line_index is not None else np.arange(len(ls.inlier)), dtype=np.int64)
    inl = np.asarray(ls.inlier, bool)
    sel = inl & (idx >= L0) & (idx < L0 + N)
    k = idx[sel] - L0
    n_even = int(np.sum(k % 2 == 0)); n_odd = int(len(k) - n_even); n = n_even + n_odd
    frac = max(n_even, n_odd)/n if n else 0.5
    z = abs(n_even - n_odd)/math.sqrt(n) if n else 0.0
    return dict(n_even=n_even, n_odd=n_odd, one_parity_fraction=float(frac), z=float(z),
                alternating=bool(n >= 12 and frac >= 0.9 and z >= 4.0))


# ------------------------------------------------------------------------------------------------- main
def resolve_timing(vis: list[VISCandidate], period_est: PeriodEstimate, line_sync: LineSync | None,
                   decode_quick, params: Params, mode: str = "auto", t0_override: float | None = None,
                   period_override: float | None = None, signal_span: tuple[float, float] | None = None,
                   sync_signal: tuple[np.ndarray, np.ndarray] | None = None, parity_fn=None,
                   use_header_end: bool = True, line_snr_fn=None, audio_span: tuple[float, float] | None = None) -> Timing:
    """Normal path (mode 'normal' or 'auto' with a credible VIS, confidence >= 0.6 and vis_code == 8): t_line0 from the VIS
    stop-bit end, period from the sync fit if its inliers >= 8 and |period - nominal| within tolerance, else ACF, else nominal;
    agreement['vis_vs_sync_ms'] = (sync-fit t_line0 - VIS t_line0) reduced modulo period.
    Recovery path (mode 'recovery', or 'auto' without a credible VIS, or 'normal' without a Robot 36 candidate of confidence
    >= 0.2): period from the sync fit / ACF comb (must agree within params.period_tolerance_ppm or report the discrepancy);
    phase from fold_phase; line 0 placed by _place_frame (dense start of the detected pulses, header-end signature, VIS,
    picture presence, separator parity) so that the frame holds 240 lines inside signal_span, with the ambiguity
    (integer-line uncertainty) reported in notes and in the confidence; then coherence refinement over params.coh_* grid;
    confidence from comb_z, n_inliers, VIS confidence, the coherence gain, the placement ambiguity and the period's
    standard error. Overrides always win (source 'override'). Never raise for weak signals: return low confidence and a
    usable Timing."""
    P_nom = params.period_nominal_ms/1000.0
    tol = params.period_tolerance_ppm*1e-6
    N = int(params.n_lines)
    half_sync = ROBOT36.sync_ms/2000.0
    notes = []; agreement = {}; alternatives = []
    vis = list(vis) if vis else []
    # the candidate that defines the normal path is the best Robot 36 one when a credible one exists (the CLI uses the same
    # rule), else the best overall (reported, never used for timing unless mode 'normal' and it reads code 8 with conf >= 0.2)
    r36 = [c for c in vis if c.vis_code == ROBOT36.vis and c.confidence >= 0.6]
    vis_best = max(r36, key=lambda c: c.confidence) if r36 else (max(vis, key=lambda c: c.confidence) if vis else None)
    vis_credible = vis_best is not None and vis_best.confidence >= 0.6 and vis_best.vis_code == ROBOT36.vis
    if mode == "normal":
        use_normal = vis_best is not None and vis_best.vis_code == ROBOT36.vis and vis_best.confidence >= 0.2
        if vis_best is None: notes.append("mode 'normal' requested but no VIS candidate: using the recovery path")
        elif not use_normal: notes.append(f"mode 'normal' requested but the best VIS candidate is not a usable Robot 36 header "
                                           f"(code {vis_best.vis_code}, conf {vis_best.confidence:.2f}): using the recovery path")
        elif not vis_credible: notes.append(f"mode 'normal': VIS candidate of low confidence ({vis_best.confidence:.2f}) used as requested")
    elif mode == "recovery":
        use_normal = False
    else:
        use_normal = vis_credible
        if vis_best is not None and not vis_credible:
            notes.append(f"VIS candidate not credible (conf {vis_best.confidence:.2f}, code {vis_best.vis_code}): recovery path")
    agreement["mode"] = "normal" if use_normal else "recovery"
    agreement["vis_confidence"] = float(vis_best.confidence) if vis_best is not None else 0.0

    def sync_credible(ls):
        """>= 8 inliers, significantly above the false-inlier background (LineSync.credible), period within tolerance."""
        if ls is None or abs(ls.period/P_nom - 1.0) > tol: return False
        cred = getattr(ls, "credible", None)
        return bool(cred) if cred is not None else ls.n_inliers >= 8

    acf_ok = period_est is not None and period_est.period is not None
    comb_z = float(period_est.comb_z) if period_est is not None else 0.0
    agreement["comb_z"] = comb_z
    ls = line_sync
    n_inl = int(ls.n_inliers) if ls is not None else 0
    agreement["n_inliers"] = n_inl
    if ls is not None:
        agreement["n_strong_inliers"] = int(getattr(ls, "n_strong", 0))
        agreement["expected_false_inliers"] = float(getattr(ls, "expected_false_inliers", float("nan")))
        agreement["sync_fit_credible"] = bool(sync_credible(ls))

    acf_in_tol = acf_ok and abs(period_est.period/P_nom - 1.0) <= tol

    def pick_period(ls):
        """Period and its source. When both the sync fit (>= 8 inliers, in tolerance) and the ACF comb fit exist they are
        combined by inverse variance: the sync fit's standard error comes from its inlier residuals and lever arm, the comb
        fit's from its peak residuals inflated 3x (window-to-window scatter), floored at 2 / 5 ppm. Returns
        (period, source, sync_se, acf_se, combined_se)."""
        if sync_credible(ls):
            P_s = float(ls.period)
            se_s = ls.period_se if (ls.period_se is not None and np.isfinite(ls.period_se)) else 50e-6*P_nom
            se_s = max(se_s, 2e-6*P_nom)
            if acf_in_tol:
                P_a = float(period_est.period)
                se_a = period_est.combfit_se if (period_est.combfit_se is not None and np.isfinite(period_est.combfit_se)) else 30e-6*P_nom
                se_a = max(3.0*se_a, 5e-6*P_nom)
                if period_est.period_combfit is None: se_a = max(se_a, 100e-6*P_nom)      # parabolic only: coarse
                w_s, w_a = 1.0/se_s**2, 1.0/se_a**2
                return (w_s*P_s + w_a*P_a)/(w_s + w_a), "sync+acf", se_s, se_a, 1.0/math.sqrt(w_s + w_a)
            return P_s, "sync-fit", se_s, None, se_s
        if acf_in_tol:
            se_a = period_est.combfit_se if (period_est.combfit_se is not None and np.isfinite(period_est.combfit_se)) else 30e-6*P_nom
            se_a = max(3.0*se_a, 5e-6*P_nom)
            return float(period_est.period), "acf", None, se_a, se_a
        if ls is not None and ls.n_inliers >= 8:
            notes.append(f"sync-fit period {(ls.period/P_nom-1)*1e6:+.0f} ppm outside tolerance {params.period_tolerance_ppm:.0f} ppm; ignored")
        return P_nom, "nominal", None, None, None

    def phase_at_period(ls, P_new):
        """line-0 time of the sync fit re-expressed with period P_new, rotating about the fit pivot (mean inlier index)
        so that the detected pulses stay in place."""
        piv = ls.pivot_line if (ls.pivot_line is not None and np.isfinite(ls.pivot_line)) else 0.0
        return float(ls.t_line0 + (ls.period - P_new)*piv)

    def record_period(ls, P, p_src, se_s, se_a, se_c):
        agreement["period_source"] = p_src
        if ls is not None and acf_ok and ls.n_inliers >= 8:
            agreement["acf_vs_sync_ppm"] = float((period_est.period/ls.period - 1.0)*1e6)
        if acf_ok: agreement["period_acf"] = float(period_est.period)
        if ls is not None:
            agreement["period_sync_fit"] = float(ls.period)
            if se_s is not None: agreement["period_sync_fit_se_ppm"] = float(se_s/P_nom*1e6)
        if se_a is not None: agreement["period_acf_se_ppm"] = float(se_a/P_nom*1e6)
        if se_c is not None: agreement["period_se_ppm"] = float(se_c/P_nom*1e6)

    P, p_src, se_s, se_a, se_c = pick_period(ls)
    record_period(ls, P, p_src, se_s, se_a, se_c)
    if "acf_vs_sync_ppm" in agreement and abs(agreement["acf_vs_sync_ppm"]) > params.period_tolerance_ppm:
        notes.append(f"ACF period and sync-fit period disagree by {agreement['acf_vs_sync_ppm']:+.0f} ppm")

    source = "nominal"
    t0 = None
    place_info = None
    L0 = None
    if use_normal:
        t0_vis = float(vis_best.t_line0)
        agreement["t_line0_vis"] = t0_vis
        if ls is not None and ls.n_inliers >= 8:
            t0_sync = phase_at_period(ls, P)
            d = _wrap(t0_sync - t0_vis, P)
            agreement["vis_vs_sync_ms"] = float(d*1000.0)
            k = int(round((t0_sync - t0_vis)/P))
            if abs(d) <= 0.003:
                t0 = float(t0_sync - k*P); source = "vis+sync"
                if k != 0: notes.append(f"sync-fit phase used; sync track was offset by {k} line(s) from the VIS line 0")
            else:
                t0 = t0_vis; source = "vis"
                notes.append(f"sync-fit phase disagrees with the VIS by {d*1000:+.1f} ms; VIS stop-bit end used")
        else:
            t0 = t0_vis; source = "vis"
        alternatives.append((t0_vis, P, "vis-stop-bit", None))
        if ls is not None: alternatives.append((float(ls.t_line0), float(ls.period), "sync-fit", None))
    else:
        if ls is not None and ls.n_inliers >= 3:
            ls_fit = ls                                   # the fit that defines period credibility and the phase
            if sync_signal is not None:
                # detection pass over an extended line range AT THE CHOSEN PERIOD (phase-only fit): it finds pulses before /
                # after the provisional frame for the placement without letting the no-carrier tail re-fit the period
                margin = int(_p(params, "recovery_margin_lines", 60))
                t_s, s_s = sync_signal
                ls2 = track_lines(t_s, s_s, phase_at_period(ls_fit, P), P, params, n_lines=N + 2*margin,
                                  line_offset=-margin, threshold=ls_fit.threshold, fit_period=False)
                if ls2.n_inliers >= 3:
                    ls = ls2; agreement["n_inliers_extended"] = int(ls2.n_inliers)
                    agreement["n_strong_inliers_extended"] = int(getattr(ls2, "n_strong", 0))
                else:
                    notes.append("extended detection pass found too few pulses; the supplied line sync is used")
            # in mode 'recovery' the VIS decoder's output is ignored entirely; in 'auto' a non-credible candidate may still
            # anchor the integer placement
            L0, place_info = _place_frame(ls, N, signal_span, vis if mode != "recovery" else [], params,
                                          sync_signal=sync_signal, use_header_end=use_header_end, line_snr_fn=line_snr_fn,
                                          audio_span=audio_span)
            # pulse grid re-expressed with the chosen (possibly combined) period, rotating about the fit pivot
            t0 = float(phase_at_period(ls, P) + L0*P)
            source = "sync-fit" if sync_credible(ls_fit) else ("acf-fold" if acf_ok else "sync-weak")
            lo, hi = place_info["feasible"]
            notes.append(f"recovery line 0 = pulse-grid line {L0} ({place_info['anchor']}); feasible {lo}..{hi} "
                         f"= integer-line ambiguity of {hi - lo} line(s); expected start uncertainty {place_info['placement_ambiguity_lines']:.1f} line(s)")
            for nm, k in place_info["alternatives"].items():
                alternatives.append((float(phase_at_period(ls, P) + k*P), P, f"placement:{nm}", None))
            notes += place_info["notes"]
            agreement["placement_anchor"] = place_info["anchor_kind"]
            agreement["placement_ambiguity_lines"] = float(place_info["placement_ambiguity_lines"])
            agreement["placement_feasible_lines"] = int(hi - lo)
            if "header_end_z" in place_info:
                agreement["header_end_z"] = float(place_info["header_end_z"])
                agreement["header_end_line_offset"] = int(place_info["header_end_k"]) - int(L0)
            if place_info.get("picture_levels_db") is not None:
                agreement["picture_levels_db"] = place_info["picture_levels_db"]
            if place_info.get("fade_in_db") is not None:
                agreement["fade_in_db"] = float(place_info["fade_in_db"])
            if place_info.get("pre_start_pulses") is not None:
                agreement["pre_start_pulses"] = place_info["pre_start_pulses"]
        else:
            # no usable pulses: nominal timing at the best available anchor
            if ls is not None: t0 = float(ls.t_line0)
            elif signal_span is not None: t0 = float(signal_span[0]) + VIS_TOTAL_MS/1000.0
            else: t0 = 0.0
            source = "nominal"
            notes.append("no credible sync pulses: nominal timing, low confidence")

    # ---- separator parity (optional) fixes the even/odd ambiguity of the integer placement
    if parity_fn is not None and not use_normal and ls is not None and ls.n_inliers >= 3:
        try:
            agr = float(parity_fn(t0, P))
            agreement["sep_parity_agreement"] = agr
            if agr < 0.35:
                lo, hi = place_info["feasible"] if place_info else (0, 0)
                L0c = place_info["L0"] if place_info else 0
                # both odd neighbours are tested (a false pulse one line before the frame makes +1 the right move) and the
                # one with the better separator agreement wins; the frame may start one line after the first inlier
                cands = [k for k in (L0c - 1, L0c + 1) if lo - 1 <= k <= hi + 1] or [L0c - 1]
                scores = {k: float(parity_fn(float(t0 + (k - L0c)*P), P)) for k in cands}
                k = cands[0]; why = "earlier line preferred"
                if len(cands) == 2:
                    if scores[cands[1]] >= scores[cands[0]] + 0.15:
                        k = cands[1]; why = "clearly better separator agreement"
                    elif scores[cands[0]] >= scores[cands[1]] + 0.15:
                        why = "clearly better separator agreement"
                    elif line_snr_fn is not None:
                        # tie: a missing line-0 pulse (header remains spoil its shape test) is more common than a false pulse
                        # exactly one line before the frame, unless the earlier line shows no picture content
                        try:
                            t_grid = float(t0 + (cands[0] - L0c)*P)       # start of the earlier candidate line
                            prof = np.asarray(line_snr_fn(t_grid, P, 20), dtype=np.float64)
                            first = prof[0]; later = prof[2:20]; later = later[np.isfinite(later)]
                            if len(later) >= 4 and np.isfinite(first):
                                sref = float(np.median(later))
                                if first < NOISE_SNR_MAX_DB and first < sref - 2.0:
                                    k = cands[1]; why = f"the earlier line has noise-level tone SNR ({first:+.1f} dB vs picture {sref:+.1f})"
                                else:
                                    why = f"earlier line preferred (its tone SNR {first:+.1f} dB is picture-like)"
                        except Exception as e:  # noqa
                            why = f"earlier line preferred (picture check failed: {e!r})"
                t0 = float(t0 + (k - L0c)*P)
                notes.append(f"separator parity {agr:.2f} < 0.35: line 0 shifted by {k - L0c:+d} line to fix the even/odd pattern "
                             f"(agreement {', '.join(f'{q - L0c:+d}: {scores[q]:.2f}' for q in cands)}; {why})")
                agreement["sep_parity_agreement_after"] = float(scores[k])
                if place_info is not None: place_info["L0"] = int(k)
                L0 = int(k)
        except Exception as e:  # noqa
            notes.append(f"parity_fn failed: {e!r}")

    # ---- Robot 36 line structure: pulses on every other line only, or a comb that is stronger at twice the period
    structure_ok = True
    if ls is not None and L0 is not None:
        try:
            pat = _pulse_pattern(ls, int(L0), N)
            agreement["pulse_pattern"] = pat
            if pat["alternating"]:
                structure_ok = False
                notes.append(f"sync pulses only on every other line ({pat['n_even']} even / {pat['n_odd']} odd): a 300 ms pulse train "
                             f"(Robot 72?) aliased onto the 150 ms grid, not a Robot 36 line structure")
        except Exception as e:  # noqa
            notes.append(f"pulse pattern test failed: {e!r}")
    cz2 = float(getattr(period_est, "comb_z_half_rate", float("nan"))) if period_est is not None else float("nan")
    if np.isfinite(cz2):
        agreement["comb_z_half_rate"] = cz2
        if cz2 > max(4.0, 0.5*comb_z):
            structure_ok = False
            notes.append(f"the ACF comb is stronger at twice the line period (z {cz2:.1f} at 300 ms vs {comb_z:.1f} at 150 ms): "
                         f"300 ms line period (Robot 72?), not Robot 36")
    agreement["robot36_structure_ok"] = bool(structure_ok)

    # ---- coherence refinement
    coh_gain = None
    if decode_quick is not None and t0 is not None:
        try:
            bt, bP, s_start, s_best, n_eval = _coherence_refine(decode_quick, t0, P, params)
            coh_gain = s_best - s_start
            agreement["coherence_score"] = float(s_best); agreement["coherence_score_start"] = float(s_start)
            agreement["coherence_gain"] = float(coh_gain); agreement["coherence_dt0_ms"] = float((bt - t0)*1000.0)
            agreement["coherence_dppm"] = float((bP/P - 1.0)*1e6); agreement["coherence_evaluations"] = int(n_eval)
            min_gain = float(_p(params, "coh_min_gain", 0.02))
            weak_fit = n_inl < 20
            if coh_gain >= min_gain and (weak_fit or source in ("acf-fold", "nominal", "sync-weak", "vis")):
                t0, P = bt, bP; source = source + "+coh"
                notes.append(f"coherence refinement applied (gain {coh_gain:+.3f}, dt0 {agreement['coherence_dt0_ms']:+.1f} ms, {agreement['coherence_dppm']:+.0f} ppm)")
            else:
                notes.append(f"coherence refinement reported only (gain {coh_gain:+.3f}); sync fit kept")
        except Exception as e:  # noqa
            notes.append(f"coherence refinement failed: {e!r}")

    # ---- frame-local comb significance (robust against dilution when period_est came from a long trace)
    if sync_signal is not None and t0 is not None:
        try:
            t_s, s_s = sync_signal
            m = (t_s >= t0 - 1.0) & (t_s <= t0 + N*P + 1.0)
            if m.sum() > 2000:
                pe_f = estimate_period_acf(t_s[m], s_s[m], params, keep_traces=False)
                agreement["comb_z_frame"] = float(pe_f.comb_z)
                comb_z = max(comb_z, float(pe_f.comb_z))
        except Exception as e:  # noqa
            notes.append(f"frame-local comb failed: {e!r}")

    # ---- overrides
    if t0_override is not None:
        alternatives.append((float(t0) if t0 is not None else float("nan"), float(P), source, None))
        t0 = float(t0_override); source = "override"; notes.append("t_line0 overridden by the user")
    if period_override is not None:
        if source != "override": alternatives.append((float(t0), float(P), source, None))
        P = float(period_override); source = "override"; notes.append("period overridden by the user")

    # ---- confidence (inlier term uses the excess over the false-inlier background, so pure noise scores ~0)
    c_comb = _clip01((comb_z - 2.0)/8.0)
    exp_false = float(getattr(ls, "expected_false_inliers", 0.0)) if ls is not None else 0.0
    if not np.isfinite(exp_false): exp_false = 0.0
    c_inl = _clip01((n_inl - 2.5*exp_false - 5.0)/40.0)
    c_coh = _clip01(coh_gain/0.1) if coh_gain is not None else 0.0
    if use_normal:
        c_vis = _clip01(vis_best.confidence)
        c_agree = _clip01(1.0 - abs(agreement["vis_vs_sync_ms"])/10.0) if "vis_vs_sync_ms" in agreement else 0.0
        conf = 0.4*c_vis + 0.25*c_inl + 0.15*c_comb + 0.15*c_agree + 0.05*c_coh
        c_place = 1.0
    else:
        c_agree = _clip01(1.0 - abs(agreement["acf_vs_sync_ppm"])/150.0) if "acf_vs_sync_ppm" in agreement else 0.0
        conf = 0.35*c_comb + 0.35*c_inl + 0.2*c_agree + 0.1*c_coh
        # integer-line placement ambiguity: 1 for an evidence anchor, 1/(1 + amb/2) for amb expected uncertain lines
        amb = float(agreement.get("placement_ambiguity_lines", 0.0))
        c_place = 1.0/(1.0 + amb/2.0)
    # the confidence that a Robot 36 signal with this period and phase exists, before the placement and period-precision
    # factors (used by the exit-code decision: an uncertain integer-line placement is not an absent signal)
    conf_signal = conf
    if not use_normal:
        conf *= 0.5 + 0.5*c_place
    # period precision: a short lever arm (few lines) gives a period standard error of 50-100 ppm, which the confidence
    # must reflect (1 at <= 20 ppm, 0 at >= 100 ppm; weight 0.3)
    se_ppm = agreement.get("period_se_ppm")
    c_per = _clip01(1.0 - (float(se_ppm) - 20.0)/80.0) if se_ppm is not None else (1.0 if source == "override" else 0.5)
    conf *= 0.7 + 0.3*c_per
    if source.startswith("nominal"): conf = min(conf, 0.1)
    if abs(P/P_nom - 1.0) > tol:
        conf *= 0.5; notes.append(f"period {(P/P_nom-1)*1e6:+.0f} ppm from nominal exceeds the tolerance")
    if not structure_ok:
        conf *= 0.3; conf_signal *= 0.3; notes.append("confidence reduced: the pulse train is not a Robot 36 line structure")
    if source.startswith("nominal"): conf_signal = min(conf_signal, 0.1)
    if abs(P/P_nom - 1.0) > tol: conf_signal *= 0.5
    if source == "override": conf = max(conf, 0.5); conf_signal = max(conf_signal, 0.5)
    agreement["confidence_signal"] = _clip01(conf_signal)
    agreement["confidence_components"] = dict(comb=c_comb, inliers=c_inl, agree=c_agree, coherence=c_coh,
                                              vis=(_clip01(vis_best.confidence) if (use_normal and vis_best) else 0.0),
                                              placement=c_place, period_precision=c_per, structure=(1.0 if structure_ok else 0.3))
    return Timing(t_line0=float(t0), period=float(P), source=source, confidence=_clip01(conf), agreement=agreement,
                  alternatives=alternatives, notes="; ".join(notes))
