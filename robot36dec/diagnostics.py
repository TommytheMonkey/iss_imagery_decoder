"""Outputs: PNGs, JSON, CSV, optional plots. Writes only inside outdir."""
import os, json, csv, math
from dataclasses import is_dataclass, asdict
import numpy as np
from . import __version__
from .modes import ROBOT36, VIS_LEADER, VIS_BIT_MS, VIS_TOTAL_MS, VIS_TABLE
from .decode import BLOCKS, rowcorr, line_start_times, H, W

# audio_for_plots convention (set by cli.py): dict with 'x' (band-passed preprocessed audio, Hz), 'fs', 'x_t0' (abs s of x[0]),
# optional 'sync_t'/'sync_s' (sync.sync_likelihood output), optional 'raw' (pre-gate audio aligned with x).


# ----------------------------------------------------------------------------------------------- helpers
def to_plain(o):
    """Recursively convert numpy / dataclass / tuple objects to JSON-serialisable plain Python (NaN/inf -> None)."""
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (int, np.integer)): return int(o)
    if isinstance(o, (float, np.floating)):
        v = float(o); return v if math.isfinite(v) else None
    if o is None or isinstance(o, str): return o
    if isinstance(o, dict): return {str(k): to_plain(v) for k, v in o.items()}
    if isinstance(o, np.ndarray): return to_plain(o.tolist())
    if isinstance(o, (list, tuple, set)): return [to_plain(v) for v in o]
    if is_dataclass(o) and not isinstance(o, type):
        try: return to_plain(asdict(o))
        except Exception: return to_plain(vars(o))
    if hasattr(o, '__dict__'): return to_plain(vars(o))
    return str(o)


def trim_long_lists(o, max_len: int = 64):
    """Replace lists longer than max_len (e.g. ACF traces carried by dataclasses) by a compact summary (recursively)."""
    if isinstance(o, dict): return {k: trim_long_lists(v, max_len) for k, v in o.items()}
    if isinstance(o, list):
        if len(o) > max_len:
            nums = [v for v in o if isinstance(v, (int, float)) and v is not None]
            if len(nums) == len(o):
                return dict(_summary=f'{len(o)} values omitted', n=len(o), min=min(nums), max=max(nums), first=o[:4], last=o[-4:])
            return dict(_summary=f'{len(o)} items omitted', n=len(o))
        return [trim_long_lists(v, max_len) for v in o]
    return o


def _u8(img) -> np.ndarray:
    return np.round(np.clip(np.asarray(img, np.float64), 0, 255)).astype(np.uint8)


def save_gray_png(img, path: str, scale: int = 1):
    from PIL import Image
    a = _u8(img); im = Image.fromarray(a, mode='L')
    if scale > 1: im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(path)


def save_rgb_png(img, path: str, scale: int = 1):
    from PIL import Image
    a = np.asarray(img, np.uint8); im = Image.fromarray(a, mode='RGB')
    if scale > 1: im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    im.save(path)


def _f(v, nd=6):
    """CSV float formatting with empty cells for NaN/None."""
    if v is None: return ''
    try:
        v = float(v)
    except Exception:
        return str(v)
    return '' if not math.isfinite(v) else f'{v:.{nd}f}'


def _get(o, name, default=None):
    return default if o is None else getattr(o, name, default)


def _vis_tone_sequence(code: int | None):
    """(freq, ms) list of the VIS header for a code (bits LSB first + even parity), 8 if unknown."""
    code = 8 if code is None else int(code) & 0x7F
    bits = [(code >> i) & 1 for i in range(7)]; bits.append(sum(bits) % 2)
    return list(VIS_LEADER) + [((1100.0 if b else 1300.0), VIS_BIT_MS) for b in bits] + [(1200.0, VIS_BIT_MS)]


def _frame_spectra(x, fs, win_ms, hop_ms, nfft, fmin, fmax, chunk_frames=1000):
    """Hann-windowed power spectra at hop_ms (t = frame centres rel. x[0]); band-limited to [fmin, fmax] (common.frame_spectra)."""
    w = int(round(win_ms * fs / 1000)); h = max(1, int(round(hop_ms * fs / 1000)))
    if len(x) < w: return np.zeros(0), np.zeros(0), np.zeros((0, 0), np.float32)
    win = np.hanning(w).astype(np.float32)
    n = 1 + (len(x) - w) // h
    f = np.fft.rfftfreq(nfft, 1.0 / fs); sel = (f >= fmin) & (f <= fmax)
    out = np.empty((n, int(sel.sum())), np.float32)
    for s0 in range(0, n, chunk_frames):
        s1 = min(n, s0 + chunk_frames)
        idx = np.arange(w)[None, :] + h * np.arange(s0, s1)[:, None]
        S = np.fft.rfft(x[idx] * win, n=nfft, axis=1)
        out[s0:s1] = (np.abs(S[:, sel]) ** 2).astype(np.float32)
    t = (np.arange(n) * h + w / 2.0) / fs
    return t, f[sel], out


def _peak_freq(P, f):
    """dominant frequency per frame (parabolic interpolation on dB) and peak-to-median ratio (dB)."""
    L = 10 * np.log10(P + 1e-20)
    k = np.argmax(L, axis=1); kc = np.clip(k, 1, len(f) - 2); rows = np.arange(len(L))
    a = L[rows, kc - 1]; b = L[rows, kc]; c = L[rows, kc + 1]
    den = a - 2 * b + c; den = np.where(den == 0, 1e-9, den)
    delta = np.clip(0.5 * (a - c) / den, -1, 1)
    return f[kc] + delta * (f[1] - f[0]), b - np.median(L, axis=1)


def _block_stats(result, timing, line_sync, blocks: dict) -> dict:
    out = {}
    n = result.Y.shape[0]
    for name, (a, b) in blocks.items():
        a = max(0, int(a)); b = min(n, int(b))
        if b <= a: continue
        snr = result.line_snr_db[a:b]; q = result.line_quieting_db[a:b]; par = result.line_sep_parity[a:b]
        d = dict(lines=[a, b], n_lines=b - a,
                 snr_db_median=float(np.nanmedian(snr)) if np.isfinite(snr).any() else None,
                 snr_db_max=float(np.nanmax(snr)) if np.isfinite(snr).any() else None,
                 quieting_db_mean=float(np.nanmean(q)) if np.isfinite(q).any() else None,
                 frac_reliable=float(result.mask[a:b].mean()),
                 n_lines_reliable=int(np.sum(result.line_frac_reliable[a:b] > 0.3)) if result.line_frac_reliable is not None else None,
                 sep_parity_agreement=float(np.mean(par[par != 0] > 0)) if np.any(par != 0) else None,
                 rowcorr_Y=rowcorr(result.Y, a, b), rowcorr_Y_fused=rowcorr(result.Y_fused, a, b))
        if math.isfinite(result.snr_floor_db):
            d['n_lines_above_snr_floor_3db'] = int(np.sum(snr > result.snr_floor_db + 3.0))
        if line_sync is not None and getattr(line_sync, 'inlier', None) is not None:
            inl = np.asarray(line_sync.inlier, bool)
            d['n_sync_inliers'] = int(inl[a:min(b, len(inl))].sum())
        out[name] = d
    return out


# ----------------------------------------------------------------------------------------------- main writer
def write_outputs(result, timing, vis_candidates, period_est, line_sync, gate_info, params, outdir: str, basename: str,
                  plots: bool = False, extra: dict | None = None, audio_for_plots=None) -> dict:
    """Write: <base>_image.png (RGB), <base>_luma.png, <base>_luma_x3.png (nearest 3x), <base>_luma_fused.png,
    <base>_image_fused_masked.png, <base>_mask.png, <base>_diagnostics.json (timing, vis list, period estimate, line-sync
    summary, gate info, params, per-block SNR/quieting, confidence metrics, software version), <base>_line_sync.csv
    (line, t_expected, t_detected, residual_ms, strength, inlier), <base>_line_metrics.csv (line, t_start, snr_db,
    quieting_db, sep_parity, frac_reliable). With plots=True also <base>_plots/: header spectrogram + VIS alignment,
    sync ACF + comb, folded sync raster, per-line metrics, dominant-frequency raster. Returns dict of written paths.
    JSON must be plain Python types (convert numpy)."""
    outdir = os.path.abspath(outdir); os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, basename)
    paths = {}
    m = ROBOT36
    t0 = float(timing.t_line0); period = float(timing.period)
    vis_candidates = list(vis_candidates or [])

    # ---------------- images
    save_rgb_png(result.rgb, base + '_image.png'); paths['image'] = base + '_image.png'
    save_gray_png(result.Y, base + '_luma.png'); paths['luma'] = base + '_luma.png'
    save_gray_png(result.Y, base + '_luma_x3.png', scale=3); paths['luma_x3'] = base + '_luma_x3.png'
    save_gray_png(result.Y_fused, base + '_luma_fused.png'); paths['luma_fused'] = base + '_luma_fused.png'
    save_rgb_png(result.rgb_fused_masked, base + '_image_fused_masked.png'); paths['image_fused_masked'] = base + '_image_fused_masked.png'
    save_gray_png(np.where(result.mask, 255.0, 0.0), base + '_mask.png'); paths['mask'] = base + '_mask.png'
    # extra display versions (cheap, useful)
    save_rgb_png(result.rgb, base + '_image_x3.png', scale=3); paths['image_x3'] = base + '_image_x3.png'
    if result.Y_fused_masked is not None:
        save_gray_png(result.Y_fused_masked, base + '_luma_fused_masked.png'); paths['luma_fused_masked'] = base + '_luma_fused_masked.png'
        save_gray_png(result.Y_fused_masked, base + '_luma_fused_masked_x3.png', scale=3); paths['luma_fused_masked_x3'] = base + '_luma_fused_masked_x3.png'

    # ---------------- line-sync CSV
    n_ls = len(line_sync.t_pulse) if (line_sync is not None and getattr(line_sync, 't_pulse', None) is not None) else H
    with open(base + '_line_sync.csv', 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['line', 't_expected', 't_detected', 'residual_ms', 'strength', 'inlier'])
        for n in range(n_ls):
            t_exp = t0 + m.sync_ms / 2000.0 + n * period
            if line_sync is not None and getattr(line_sync, 't_pulse', None) is not None:
                t_det = float(line_sync.t_pulse[n]); st = float(line_sync.strength[n]); inl = int(bool(line_sync.inlier[n]))
                res = (t_det - t_exp) * 1000.0 if math.isfinite(t_det) else float('nan')
            else:
                t_det, st, inl, res = float('nan'), float('nan'), 0, float('nan')
            wr.writerow([n, _f(t_exp, 6), _f(t_det, 6), _f(res, 3), _f(st, 5), inl])
    paths['line_sync_csv'] = base + '_line_sync.csv'

    # ---------------- line-metrics CSV
    ls_t = line_start_times(t0, period, result.Y.shape[0])
    with open(base + '_line_metrics.csv', 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['line', 't_start', 'snr_db', 'quieting_db', 'sep_parity', 'frac_reliable'])
        for n in range(result.Y.shape[0]):
            fr = result.line_frac_reliable[n] if result.line_frac_reliable is not None else result.mask[n].mean()
            wr.writerow([n, _f(ls_t[n], 6), _f(result.line_snr_db[n], 2), _f(result.line_quieting_db[n], 2),
                         int(result.line_sep_parity[n]), _f(fr, 4)])
    paths['line_metrics_csv'] = base + '_line_metrics.csv'

    # ---------------- JSON
    nominal = m.line_ms / 1000.0
    timing_d = to_plain(timing)
    timing_d['period_ms'] = period * 1000.0
    timing_d['period_ppm_vs_nominal'] = (period / nominal - 1.0) * 1e6
    timing_d['t_frame_end'] = t0 + H * period
    vis_list = [to_plain(v) for v in vis_candidates]
    best = vis_list[0] if vis_list else None
    ls_d = None
    if line_sync is not None:
        inl = np.asarray(_get(line_sync, 'inlier', np.zeros(0)), bool)
        tp = np.asarray(_get(line_sync, 't_pulse', np.zeros(0)), np.float64)
        st = np.asarray(_get(line_sync, 'strength', np.zeros(0)), np.float64)
        ls_d = dict(t_line0=_get(line_sync, 't_line0'), period=_get(line_sync, 'period'),
                    period_ms=None if _get(line_sync, 'period') is None else float(line_sync.period) * 1000.0,
                    resid_ms_sd=_get(line_sync, 'resid_ms_sd'), n_inliers=_get(line_sync, 'n_inliers'),
                    threshold=_get(line_sync, 'threshold'), runs=_get(line_sync, 'runs', []), n_lines_tracked=int(len(tp)),
                    inlier_lines=np.where(inl)[0].tolist(), t_pulse_inliers=tp[inl].tolist() if len(tp) == len(inl) else [],
                    strength_inliers=st[inl].tolist() if len(st) == len(inl) else [],
                    residual_ms_inliers=((tp[inl] - (t0 + m.sync_ms / 2000.0 + np.where(inl)[0] * period)) * 1000.0).tolist() if len(tp) == len(inl) else [])
    blocks20 = {f'lines_{a:03d}_{a + 20:03d}': (a, a + 20) for a in range(0, H, 20)}
    decode_d = dict(estimator=result.estimator, estimator_detail=getattr(result, 'estimator_detail', ''), snr_regime=getattr(result, 'snr_regime', ''),
                    whitening=getattr(result, 'whitening', ''),
                    snr_median_db=getattr(result, 'snr_median_db', None), calibration=result.calibration, frac_reliable=result.frac_reliable,
                    reliable_runs=result.reliable_runs, n_reliable_lines=int(np.sum(result.line_frac_reliable > 0.3)) if result.line_frac_reliable is not None else None,
                    sep_parity_agreement=result.sep_parity_agreement, sep_parity_agreement_reliable_lines=result.sep_parity_agreement_reliable,
                    mask_threshold_hz=result.mask_threshold_hz, fuse_threshold_hz=result.fuse_threshold_hz,
                    snr_floor_db=result.snr_floor_db, quieting_ref_db=result.quieting_ref_db,
                    quieting_sign_convention='line level minus no-carrier floor (dB); negative = quieting = carrier present',
                    noise_regions=result.noise_regions, control=result.control,
                    rowcorr_Y_all=rowcorr(result.Y), rowcorr_Y_fused_all=rowcorr(result.Y_fused),
                    luma_mean=float(result.Y.mean()), luma_std=float(result.Y.std()),
                    saturation_frac=float(((result.Y <= 0.5) | (result.Y >= 254.5)).mean()),
                    chroma_median=float(np.median(result.C)), chroma_std_reliable=float(result.C[result.mask].std()) if result.mask.any() else None,
                    post_sd_Y_median_hz=float(np.median(result.post_sd_Y)),
                    post_sd_Y_median_hz_reliable=float(np.median(result.post_sd_Y[result.mask])) if result.mask.any() else None,
                    n_lines_in_range=int(result.line_in_range.sum()) if result.line_in_range is not None else None,
                    noise_shape=None if result.noise_shape is None else dict(freqs=result.freqs, shape=result.noise_shape,
                                                                            tilt_db=float(10 * np.log10(result.noise_shape[-1] / result.noise_shape[0])) if result.noise_shape is not None and result.noise_shape[0] > 0 else None))
    vis_used = bool((extra or {}).get('vis_used_for_timing'))
    conf = dict(timing_confidence=float(timing.confidence), timing_source=timing.source,
                vis_used_for_timing=vis_used,
                vis_confidence=(best or {}).get('confidence'), vis_code=(best or {}).get('vis_code'), vis_z=(best or {}).get('z'),
                vis_parity_ok=(best or {}).get('parity_ok'),
                vis_note=('best VIS candidate used for the timing' if vis_used else
                          'best VIS candidate reported only: the timing comes from the sync pulses (header ignored or not credible)'),
                robot36_structure_ok=(getattr(timing, 'agreement', {}) or {}).get('robot36_structure_ok', True),
                comb_z=_get(period_est, 'comb_z'), acf_150=_get(period_est, 'acf_150'),
                n_sync_inliers=_get(line_sync, 'n_inliers'), sync_resid_ms_sd=_get(line_sync, 'resid_ms_sd'),
                sep_parity_agreement=result.sep_parity_agreement, frac_reliable=result.frac_reliable,
                agreement=to_plain(getattr(timing, 'agreement', {})))
    if extra:
        for k in ('signal_detected', 'signal_score', 'signal_criteria', 'exit_code'):
            if k in extra: conf[k] = to_plain(extra[k])
    diag = dict(software=dict(name='robot36dec', version=__version__), run=to_plain(extra or {}),
                timing=timing_d, vis=dict(n_candidates=len(vis_list), best=best, candidates=vis_list, used_for_timing=(extra or {}).get('vis_used_for_timing')),
                period_estimate=trim_long_lists(to_plain(period_est)), line_sync=ls_d, gate=trim_long_lists(to_plain(gate_info), 2048), decode=decode_d,
                blocks=_block_stats(result, timing, line_sync, BLOCKS), blocks_20=_block_stats(result, timing, line_sync, blocks20),
                confidence=conf, params=to_plain(asdict(params)) if is_dataclass(params) else to_plain(params),
                outputs={k: os.path.basename(v) for k, v in paths.items()})
    with open(base + '_diagnostics.json', 'w') as fh:
        json.dump(to_plain(diag), fh, indent=1, allow_nan=False)
    paths['diagnostics_json'] = base + '_diagnostics.json'

    # ---------------- plots
    if plots:
        plotdir = base + '_plots'; os.makedirs(plotdir, exist_ok=True)
        pp = _write_plots(result, timing, vis_candidates, period_est, line_sync, params, plotdir, basename, audio_for_plots)
        paths.update(pp)
        # record the plot list in the JSON too
        try:
            diag['outputs'].update({k: os.path.relpath(v, outdir) for k, v in pp.items() if isinstance(v, str)})
            with open(base + '_diagnostics.json', 'w') as fh: json.dump(to_plain(diag), fh, indent=1, allow_nan=False)
        except Exception:
            pass
    return paths


# ----------------------------------------------------------------------------------------------- plots
def _write_plots(result, timing, vis, period_est, line_sync, params, plotdir, basename, audio) -> dict:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib is optional
        return {'plot_errors': [f'matplotlib unavailable: {e}']}
    out = {}; errors = []
    m = ROBOT36
    t0 = float(timing.t_line0); period = float(timing.period)
    x = fs = x_t0 = None
    if isinstance(audio, dict) and audio.get('x') is not None:
        x = np.asarray(audio['x'], np.float32); fs = int(audio['fs']); x_t0 = float(audio.get('x_t0', 0.0))
    st = audio.get('sync_t') if isinstance(audio, dict) else None
    ss = audio.get('sync_s') if isinstance(audio, dict) else None

    def _seg(ta, tb):
        i0 = int(max(0, round((ta - x_t0) * fs))); i1 = int(min(len(x), round((tb - x_t0) * fs)))
        return x[i0:i1], x_t0 + i0 / fs

    # ---- 1. header spectrogram + VIS alignment
    try:
        if x is None: raise RuntimeError('no audio')
        from scipy.signal import spectrogram
        best = vis[0] if vis else None
        t_lead = float(best.t_leader) if best is not None else t0 - VIS_TOTAL_MS / 1000.0
        seg, seg_t0 = _seg(t_lead - 0.6, t_lead + 1.6)
        fig, ax = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        if len(seg) > fs // 10:
            nper = int(0.015 * fs)
            f, tt, S = spectrogram(seg.astype(np.float64), fs, nperseg=nper, noverlap=nper - nper // 8, nfft=4096, window='hann', mode='psd')
            sel = (f >= 900) & (f <= 2600); SdB = 10 * np.log10(S[sel] + 1e-20)
            ax[0].pcolormesh(tt + seg_t0, f[sel], SdB, vmin=np.percentile(SdB, 5), vmax=np.percentile(SdB, 99.8), cmap='magma', shading='auto')
            tf, ff, P = _frame_spectra(seg, fs, 8.0, 1.0, 4096, 900, 2600)
            if len(tf):
                fpk, q = _peak_freq(P, ff)
                sc = ax[1].scatter(tf + seg_t0, fpk, c=np.clip(q, 0, 25), s=5, cmap='viridis')
                plt.colorbar(sc, ax=ax[1], pad=0.01, label='peak/median dB')
        code = best.vis_code if best is not None else None
        tt_seq = []; ff_seq = []; pos = t_lead
        for fr, ms in _vis_tone_sequence(code):
            tt_seq += [pos, pos + ms / 1000.0]; ff_seq += [fr, fr]; pos += ms / 1000.0
        for a_ in ax:
            a_.plot(tt_seq, ff_seq, 'r-', lw=1, alpha=0.8, label=f'ideal VIS {code if code is not None else "8 (assumed)"} sequence')
            a_.axvline(t0, color='c', ls='--', lw=1, label=f'line 0 start (timing) {t0:.4f} s')
            if best is not None: a_.axvline(float(best.t_line0), color='w', ls=':', lw=1, label=f'VIS stop-bit end {float(best.t_line0):.4f} s')
            a_.set_ylim(900, 2600); a_.set_ylabel('Hz')
        ax[0].legend(loc='upper left', fontsize=8)
        ttl = 'no credible VIS header; window placed 0.91 s before the timing line 0' if best is None else \
            f'VIS candidate: code {best.vis_code} ({best.mode_name}), z {best.z:.1f}, parity {"OK" if best.parity_ok else "BAD"}, plateau {best.plateau_ms:.0f} ms, confidence {best.confidence:.2f}'
        ax[0].set_title(f'{basename}: header spectrogram (15 ms) — {ttl}')
        ax[1].set_title('dominant tone (8 ms frames, 1 ms hop) vs ideal VIS tone sequence'); ax[1].set_xlabel('time into input (s)')
        plt.tight_layout(); p = os.path.join(plotdir, 'header_vis.png'); plt.savefig(p, dpi=100); plt.close(fig); out['plot_header_vis'] = p
    except Exception as e:
        errors.append(f'header_vis: {e}'); plt.close('all')

    # ---- 2. sync ACF + comb
    try:
        if st is None or ss is None: raise RuntimeError('no sync likelihood')
        st = np.asarray(st, np.float64); ss = np.asarray(ss, np.float64)
        wa, wb = t0 - 2.0, t0 + H * period + 2.0
        sel = (st >= wa) & (st <= wb)
        if sel.sum() < 1000: sel = np.ones(len(st), bool)
        s = ss[sel] - ss[sel].mean(); hop = float(np.median(np.diff(st[sel]))) if sel.sum() > 1 else 0.0005
        n = len(s); nf = 1 << int(np.ceil(np.log2(2 * n)))
        acf = np.fft.irfft(np.abs(np.fft.rfft(s, nf)) ** 2, nf)[:int(1.25 / hop)]
        acf = acf / (acf[0] + 1e-30); lags = np.arange(len(acf)) * hop * 1000.0
        fig, ax = plt.subplots(2, 1, figsize=(16, 9))
        ax[0].plot(lags, acf, lw=0.8); ax[0].set_xlim(0, 1200); lo = int(0.05 / hop)
        ax[0].set_ylim(np.percentile(acf[lo:], 0.5) - 0.02, max(np.percentile(acf[lo:], 99.9) * 1.5, 0.05))
        for k in range(1, 7):
            ax[0].axvline(k * period * 1000, color='r', ls='-', lw=0.6, alpha=0.6)
            ax[0].axvline((k - 0.5) * period * 1000, color='b', ls=':', lw=0.6, alpha=0.6)
        ax[0].set_xlabel('lag (ms)'); ax[0].set_ylabel('normalised ACF of the sync likelihood')
        pe = period_est
        ax[0].set_title(f'ACF of the 1200 Hz sync likelihood ({wa:.1f}-{wb:.1f} s); red = k x period {period * 1000:.4f} ms, blue dotted = half-period lags; '
                        f'comb z = {_get(pe, "comb_z", float("nan")):.1f}, ACF(150 ms) = {_get(pe, "acf_150", float("nan")):.3f}')
        ax[1].plot(lags, acf, lw=0.8); ax[1].set_xlim(100, 200)
        ax[1].axvline(period * 1000, color='r', lw=0.8, label=f'timing period {period * 1000:.4f} ms')
        if _get(pe, 'period') is not None: ax[1].axvline(float(pe.period) * 1000, color='g', ls='--', lw=0.8, label=f'ACF period {float(pe.period) * 1000:.4f} ms')
        if _get(pe, 'best_period_any') is not None:
            ax[1].set_title(f'zoom; best period anywhere in {params.period_search_ms[0]:.0f}-{params.period_search_ms[1]:.0f} ms: '
                            f'{float(pe.best_period_any) * 1000:.3f} ms (z {_get(pe, "best_z_any", float("nan")):.1f}); candidates: '
                            + ', '.join(f'{float(c[0]) * 1000:.2f} ms (z {float(c[1]):.1f})' for c in (_get(pe, 'candidates', []) or [])[:5]))
        ax[1].legend(fontsize=8); ax[1].set_xlabel('lag (ms)')
        plt.tight_layout(); p = os.path.join(plotdir, 'sync_acf.png'); plt.savefig(p, dpi=100); plt.close(fig); out['plot_sync_acf'] = p
    except Exception as e:
        errors.append(f'sync_acf: {e}'); plt.close('all')

    # ---- 3. folded sync raster
    try:
        if st is None or ss is None: raise RuntimeError('no sync likelihood')
        st = np.asarray(st, np.float64); ss = np.asarray(ss, np.float64)
        nb = int(round(m.line_ms * 2)); ras = np.full((H, nb), np.nan)
        offs = np.arange(nb) * 0.0005
        for n_ in range(H):
            tq = t0 + n_ * period + offs
            ok = (tq >= st[0]) & (tq <= st[-1])
            if ok.any(): ras[n_, ok] = np.interp(tq[ok], st, ss)
        fig, ax = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw=dict(width_ratios=[3, 1]))
        vmax = np.nanpercentile(ras, 99.5) if np.isfinite(ras).any() else 1.0
        im = ax[0].imshow(ras, aspect='auto', cmap='magma', extent=[0, m.line_ms, H, 0], vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax[0], label='1200 Hz share of 900-2600 Hz energy (9 ms window)')
        ax[0].axvline(m.sync_ms / 2, color='c', ls='--', lw=0.8); ax[0].axvline(m.sync_ms, color='c', ls=':', lw=0.8)
        if line_sync is not None and getattr(line_sync, 'inlier', None) is not None:
            inl = np.where(np.asarray(line_sync.inlier, bool))[0]
            inl = inl[inl < H]
            tp = np.asarray(line_sync.t_pulse, np.float64)
            ax[0].plot((tp[inl] - (t0 + inl * period)) * 1000.0, inl + 0.5, 'c.', ms=3, label='detected sync pulses (inliers)')
            ax[0].legend(loc='upper right', fontsize=8)
        ax[0].set_xlabel('ms within line'); ax[0].set_ylabel('line')
        ax[0].set_title(f'sync-likelihood raster folded at line 0 = {t0:.4f} s, period {period * 1000:.4f} ms (pulse expected at 0-9 ms)')
        prof = np.nanmean(ras, axis=0)
        ax[1].plot(prof, np.arange(nb) * 0.5); ax[1].invert_yaxis(); ax[1].set_ylabel('ms within line'); ax[1].set_xlabel('mean over lines')
        ax[1].set_title('folded profile'); ax[1].axhline(m.sync_ms / 2, color='c', ls='--', lw=0.8)
        plt.tight_layout(); p = os.path.join(plotdir, 'sync_raster.png'); plt.savefig(p, dpi=100); plt.close(fig); out['plot_sync_raster'] = p
    except Exception as e:
        errors.append(f'sync_raster: {e}'); plt.close('all')

    # ---- 4. per-line metrics
    try:
        fig, ax = plt.subplots(4, 1, figsize=(16, 13), sharex=True)
        lines = np.arange(H)
        if line_sync is not None and getattr(line_sync, 'strength', None) is not None:
            n_ls = min(H, len(line_sync.strength))
            stg = np.asarray(line_sync.strength[:n_ls], np.float64); inl = np.asarray(line_sync.inlier[:n_ls], bool)
            ax[0].plot(lines[:n_ls], stg, '.-', lw=0.6, ms=3, label='peak sync likelihood near expected pulse')
            ax[0].plot(lines[:n_ls][inl], stg[inl], 'o', ms=4, color='g', label=f'inliers ({int(inl.sum())})')
            if _get(line_sync, 'threshold') is not None: ax[0].axhline(float(line_sync.threshold), color='r', ls='--', label='noise 99.5th percentile')
            tp = np.asarray(line_sync.t_pulse[:n_ls], np.float64)
            resid = (tp - (t0 + m.sync_ms / 2000.0 + lines[:n_ls] * period)) * 1000.0
            ax[1].plot(lines[:n_ls][inl], resid[inl], 'o', ms=3, color='g', label='inliers')
            ax[1].plot(lines[:n_ls][~inl], resid[~inl], 'x', ms=3, color='gray', alpha=0.5, label='others')
            ax[1].set_ylim(-params.line_search_ms, params.line_search_ms)
        ax[0].set_ylabel('sync strength'); ax[0].legend(fontsize=8)
        ax[1].set_ylabel('pulse residual vs timing (ms)'); ax[1].axhline(0, color='k', lw=0.5); ax[1].legend(fontsize=8)
        ax[2].plot(lines, result.line_snr_db, label='audio SNR in Y segment (dB re 2400 Hz)')
        if math.isfinite(result.snr_floor_db): ax[2].axhline(result.snr_floor_db, color='r', ls='--', label=f'control 99th percentile (floor) {result.snr_floor_db:.1f} dB')
        ax2 = ax[2].twinx(); fr = result.line_frac_reliable if result.line_frac_reliable is not None else result.mask.mean(1)
        ax2.fill_between(lines, 0, fr, color='g', alpha=0.25, step='mid', label='fraction of reliable pixels'); ax2.set_ylim(0, 1); ax2.set_ylabel('reliable fraction')
        ax[2].set_ylabel('dB'); ax[2].legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
        if np.isfinite(result.line_quieting_db).any():
            ax[3].plot(lines, result.line_quieting_db, label='wideband level vs no-carrier floor (dB; negative = quieting)')
        par = result.line_sep_parity.astype(float)
        ax3 = ax[3].twinx(); ax3.plot(lines, par, '.', ms=3, color='m', alpha=0.6, label='separator parity (+1 match / -1 opposite)'); ax3.set_ylim(-1.5, 1.5)
        ax[3].set_ylabel('dB'); ax[3].set_xlabel('Robot 36 line'); ax[3].legend(loc='upper left', fontsize=8); ax3.legend(loc='upper right', fontsize=8)
        for a_ in ax:
            for r in result.reliable_runs: a_.axvspan(r[0] - 0.5, r[1] + 0.5, color='g', alpha=0.08)
        ax[0].set_title(f'{basename}: per-line diagnostics (green bands = lines with > 30 % reliable pixels; {getattr(result, "estimator_detail", "") or result.estimator}; '
                        f'separator parity agreement {result.sep_parity_agreement:.2f} all / {result.sep_parity_agreement_reliable:.2f} reliable lines)')
        plt.tight_layout(); p = os.path.join(plotdir, 'line_metrics.png'); plt.savefig(p, dpi=100); plt.close(fig); out['plot_line_metrics'] = p
    except Exception as e:
        errors.append(f'line_metrics: {e}'); plt.close('all')

    # ---- 5. dominant-frequency raster
    try:
        if x is None: raise RuntimeError('no audio')
        seg, seg_t0 = _seg(t0 - 0.05, t0 + H * period + 0.05)
        tf, ff, P = _frame_spectra(seg, fs, 8.0, 1.0, 2048, 900, 2600)
        if len(tf) == 0: raise RuntimeError('frame too short')
        fpk, q = _peak_freq(P, ff); tf = tf + seg_t0
        nb = int(round(m.line_ms)); rasf = np.full((H, nb), np.nan); rass = np.full((H, nb), np.nan)
        offs = (np.arange(nb) + 0.5) / 1000.0
        for n_ in range(H):
            tq = t0 + n_ * period + offs
            ok = (tq >= tf[0]) & (tq <= tf[-1])
            if ok.any():
                idx = np.clip(np.searchsorted(tf, tq[ok]), 0, len(tf) - 1)
                rasf[n_, ok] = fpk[idx]; rass[n_, ok] = q[idx]
        fig, ax = plt.subplots(1, 3, figsize=(24, 10), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
        im = ax[0].imshow(rasf, aspect='auto', cmap='jet', extent=[0, m.line_ms, H, 0], vmin=1100, vmax=2400)
        plt.colorbar(im, ax=ax[0], label='dominant Hz (8 ms window)')
        ax[0].set_title('dominant-frequency raster: expect sync 1200 @ 0-9, Y 12-100, sep 1500/2300 @ 100-104.5, porch 1900, chroma 106-150')
        im = ax[1].imshow(rass, aspect='auto', cmap='viridis', extent=[0, m.line_ms, H, 0], vmin=0, vmax=25)
        plt.colorbar(im, ax=ax[1], label='peak/median dB'); ax[1].set_title('tone strength raster')
        for a_ in ax[:2]:
            for xx in (m.sync_ms, m.y_start_ms, m.sep_start_ms, m.c_start_ms): a_.axvline(xx, color='w', lw=0.5, alpha=0.6)
            a_.set_xlabel('ms within line'); a_.set_ylabel('line')
        ax[2].imshow(_u8(result.Y), cmap='gray', vmin=0, vmax=255, aspect='auto', extent=[0, W, H, 0]); ax[2].set_title('decoded luminance')
        ax[2].set_xlabel('pixel')
        plt.tight_layout(); p = os.path.join(plotdir, 'freq_raster.png'); plt.savefig(p, dpi=90); plt.close(fig); out['plot_freq_raster'] = p
    except Exception as e:
        errors.append(f'freq_raster: {e}'); plt.close('all')

    if errors: out['plot_errors'] = errors
    return out
