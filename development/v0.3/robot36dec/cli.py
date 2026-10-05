"""Command line: python -m robot36dec decode INPUT --out DIR [options]

Pipeline (SPEC.md section 3):
  load (WAV or IQ->demodulate) -> window (--start/--end) -> remove_dc -> threshold_gate -> bandpass
  -> carrier presence (wideband quieting of the pre-gate audio) -> header.find_vis -> sync likelihood -> frame locator
  (VIS, carrier runs or a scanning ACF comb) -> estimate_period_acf -> fold_phase -> track_lines -> timing.resolve_timing
  -> decode.decode_robot36 -> diagnostics.write_outputs.
The program writes only inside --out.  Exit code 0 = decoded with a credible signal, 2 = no credible signal (diagnostics are
still written), 1 = error.
"""
import argparse, os, sys, time, math, json, traceback, functools
from dataclasses import asdict
import numpy as np

from . import __version__
from .params import Params
from .modes import ROBOT36, VIS_TABLE, VIS_TOTAL_MS
from .timing import start_rule_min_count, first_dense_start


def build_parser():
    p = argparse.ArgumentParser(prog="robot36dec", description="Robot 36 SSTV decoder with header-free timing recovery")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode", help="decode a WAV (FM-demodulated audio) or an RTL-SDR u8 IQ file")
    d.add_argument("input")
    d.add_argument("--out", required=True, help="output directory (created; the ONLY place the program writes)")
    d.add_argument("--basename", default=None, help="output file prefix (default: input file stem)")
    d.add_argument("--iq", action="store_true", help="input is unsigned 8-bit interleaved IQ")
    d.add_argument("--iq-rate", type=float, default=None, help="IQ sample rate (S/s), required with --iq")
    d.add_argument("--center-freq", type=float, default=None, help="tuner centre frequency (Hz), required with --iq")
    d.add_argument("--signal-freq", type=float, default=437.550e6, help="SSTV carrier frequency (Hz)")
    d.add_argument("--wav-fullscale-hz", type=float, default=None, help="deviation (Hz) at WAV full scale (default rtl_fm: 48000)")
    d.add_argument("--mode", choices=["auto", "normal", "recovery"], default="auto")
    d.add_argument("--t0", type=float, default=None, help="override line-0 start time (s into input)")
    d.add_argument("--period", type=float, default=None, help="override line period (s)")
    d.add_argument("--start", type=float, default=0.0, help="only analyse from this time (s)")
    d.add_argument("--end", type=float, default=None, help="only analyse up to this time (s)")
    d.add_argument("--gate", choices=["auto", "on", "off"], default=None)
    d.add_argument("--estimator", choices=["chain", "soft"], default=None)
    d.add_argument("--window-ms", type=float, default=None)
    d.add_argument("--params", default=None, help="JSON file of Params overriding params_frozen.json / defaults")
    d.add_argument("--plots", action="store_true")
    d.add_argument("--save-audio", action="store_true", help="also write the demodulated/preprocessed audio as WAV into --out")
    d.add_argument("--quiet", action="store_true")
    d.add_argument("--iq-search-hz", type=float, default=30000.)
    d.add_argument("--receiver-ppm", type=float, default=0.)
    d.add_argument("--uncertainty-ppm", type=float, default=20.)
    d.add_argument("--channel-bandwidths", default="11000,15000,25000")

    return p


# ----------------------------------------------------------------------------------------------- helpers
def _load_params(args) -> Params:
    params = Params.default()
    if args.params:
        with open(args.params) as fh: d = json.load(fh)
        for k, v in d.items():
            if hasattr(params, k):
                setattr(params, k, tuple(v) if isinstance(getattr(params, k), tuple) else v)
    if args.gate: params.gate_mode = args.gate
    if args.estimator: params.estimator = args.estimator
    if args.window_ms: params.win_ms = float(args.window_ms)
    if args.wav_fullscale_hz: params.wav_fullscale_hz = float(args.wav_fullscale_hz)
    return params


def _window(t, s, a, b):
    sel = (t >= a) & (t <= b)
    return t[sel], s[sel]


def _runs_from_flags(flags, t, min_len_s):
    """[(t_start, t_end), ...] of runs of True in a per-block flag array (t = block centres)."""
    runs = []; i = 0; n = len(flags)
    if n == 0: return runs
    dt = float(np.median(np.diff(t))) if n > 1 else 0.0
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]: j += 1
            ta, tb = float(t[i] - dt / 2), float(t[j] + dt / 2)
            if tb - ta >= min_len_s: runs.append((ta, tb))
            i = j + 1
        else:
            i += 1
    return runs


def carrier_presence(raw, fs, raw_t0, params, band=(8000.0, 20000.0)) -> dict:
    """Carrier presence from wideband quieting of the pre-gate discriminator output (scripts/09_carrier_cadence.py):
    per-block band level in dB relative to its robust median, 2 s smoothing, present where the smoothed level is below
    -max(carrier_min_quiet_db, 4 sigma/sqrt(n_smooth))."""
    from .decode import block_band_power
    block_s = float(getattr(params, 'carrier_block_s', 0.5))
    t, p = block_band_power(np.asarray(raw), fs, block_s, band, raw_t0)
    if len(p) < 8:
        return dict(t=t, level_db=np.zeros(len(p)), present=np.zeros(len(p), bool), runs=[], sigma_db=float('nan'), threshold_db=float('nan'))
    lev = 10 * np.log10(np.maximum(p, 1e-30)); lev -= np.median(lev)
    n_sm = max(1, int(round(2.0 / block_s)))
    sm = np.convolve(lev, np.ones(n_sm) / n_sm, mode='same')
    sig = 1.4826 * float(np.median(np.abs(lev - np.median(lev)))) + 1e-9
    thr = -max(float(getattr(params, 'carrier_min_quiet_db', 1.0)), 4.0 * sig / math.sqrt(n_sm))
    present = sm < thr
    runs = _runs_from_flags(present, t, float(getattr(params, 'carrier_min_run_s', 1.5)))
    return dict(t=t, level_db=lev, smoothed_db=sm, present=present, runs=runs, sigma_db=sig, threshold_db=thr, band_hz=list(band))


def _pulse_grid_start(phase, period, t_from, t_s, s, sync_ms):
    """first sync-pulse centre >= t_from on the folded pulse grid; the fold phase may be referenced to absolute time or to
    t_s[0] (sync.fold_phase convention), so both hypotheses are tested against the data and the better one is used."""
    best = None
    for ref in (0.0, float(t_s[0])):
        g0 = ref + float(phase)
        k = np.arange(math.ceil((t_from - g0) / period), math.ceil((t_s[-1] - g0) / period))
        grid = g0 + k * period
        if len(grid) == 0: continue
        score = float(np.mean(np.interp(grid, t_s, s)))
        if best is None or score > best[0]: best = (score, grid[0])
    if best is None: return float(t_from)
    return float(best[1]) - sync_ms / 2000.0


def _merge_runs(runs, gap_s=10.0):
    """merge carrier runs separated by less than gap_s into hulls."""
    out = []
    for a, b in sorted(runs):
        if out and a - out[-1][1] < gap_s: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out


def _track(sync, t, s, t0, P, params, n_lines):
    """sync.track_lines over n_lines lines (keyword if supported, else a Params copy)."""
    try:
        return sync.track_lines(t, s, t0, P, params, n_lines=int(n_lines))
    except TypeError:
        import dataclasses
        return sync.track_lines(t, s, t0, P, dataclasses.replace(params, n_lines=int(n_lines)))


def _leader_share(xb, fs, x_t0, t_cand, n_win=14):
    """mean share of 1900 +-60 Hz in the 900-2600 Hz energy (20 ms windows) over [t_cand - 0.29, t_cand - 0.03] s:
    ~0.07 for noise or SSTV data, > 0.4 when the second VIS leader occupies that interval."""
    from . import tones
    centres = t_cand - 0.29 + np.linspace(0.0, 0.26, n_win)
    f_all = np.arange(900.0, 2600.0 + 1.0, 20.0)
    P = np.asarray(tones.tone_power(np.asarray(xb, np.float32), fs, x_t0, centres, 20.0, f_all), np.float64)
    band = np.abs(f_all - 1900.0) <= 60.0
    return float(np.mean(P[:, band].sum(-1) / (P.sum(-1) + 1e-30)))


def _frame_start_from_inliers(inl, n_frame=240, p0=0.02, window_lines=12, false_alarm=1e-3, frac=0.9):
    """index of the first 'dense start' (an inlier followed by enough inliers within window_lines lines: the count a uniform
    false-inlier background of rate p0 per line reaches with probability <= false_alarm, see timing.start_rule_min_count)
    whose n_frame-line window holds >= frac of the maximum inlier count; returns (k0, count, k_last_inlier_in_window, m_min).
    This is only the provisional grid: timing.resolve_timing re-places the frame with the same rule plus the header-end /
    VIS / picture-presence / parity evidence."""
    inl = np.asarray(inl, bool); n = len(inl)
    if n == 0 or not inl.any(): return 0, 0, 0, 0
    c = np.concatenate([[0], np.cumsum(inl)])
    counts = np.array([c[min(n, k + n_frame)] - c[k] for k in range(n)])
    best = int(counts.max())
    m_min = start_rule_min_count(p0, window_lines, false_alarm)
    k0 = int(np.argmax(counts))
    for k in range(n):
        if inl[k] and (c[min(n, k + window_lines)] - c[k]) >= m_min and counts[k] >= frac * best:
            k0 = k; break
    idx = np.where(inl[k0:min(n, k0 + n_frame)])[0]
    k_last = int(k0 + idx[-1]) if len(idx) else k0
    return k0, int(counts[k0]), k_last, m_min


def _parity_fraction(xb, fs, x_t0, t0, period):
    """fraction of lines whose separator tone matches the even/odd pattern for line 0 at t0 (0.5 = random)."""
    from .decode import separator_parity
    par = separator_parity(xb, fs, x_t0, float(t0), float(period))
    nz = par[par != 0]
    return float(np.mean(nz > 0)) if len(nz) else 0.5


def _line_snr(xb, fs, x_t0, params, t0, period, n_lines):
    """per-line tone SNR (dB re 2400 Hz, decode.line_snr_db) of n_lines lines starting at t0: the picture-presence
    evidence of timing._place_frame (a line of pre-frame noise has no tone; a picture line has one in 1500-2300 Hz)."""
    from .decode import pixel_times, frequency_grid, line_snr_db
    from . import tones
    n = int(n_lines)
    if n <= 0: return np.zeros(0)
    ty, _ = pixel_times(float(t0), float(period), params)
    ty = ty[:n, ::4]                                      # 80 windows per line
    freqs = frequency_grid(params)
    P = tones.tone_power(xb, fs, x_t0, ty, params.win_ms, freqs)
    return line_snr_db(P, params.win_ms, px_step=1)


def _signal_decision(timing, vis_best, period_est, line_sync, result, params, mode):
    """Measured protocol structure is mandatory; pixel reliability is not identity."""
    agr=getattr(timing,'agreement',{}) or {}
    evidence=agr.get('protocol_evidence',{})
    structure=bool(evidence.get('valid',False)) and bool(agr.get('robot36_structure_ok',True))
    vis_ok=bool(vis_best is not None and vis_best.vis_code==8 and vis_best.parity_ok and vis_best.confidence>=.6)
    confidence=float(agr.get('confidence_signal',timing.confidence))
    detected=structure
    from .protocol import classify_evidence
    verdict=classify_evidence(evidence) if detected else 'NO EVIDENCE'
    return detected, confidence if detected else 0.,dict(robot36_structure_ok=structure,
            protocol_evidence=evidence,vis_credible=vis_ok,signal_confidence=confidence,
            reason=evidence.get('reason','no measured protocol evidence'),
            verdict=verdict)


def _fmt_runs(runs):
    return ', '.join(f'{a}-{b}' if b > a else f'{a}' for a, b in runs) if runs else '-'


def _write_error_json(outdir, basename, msg, stage, t_start, extra=None):
    try:
        os.makedirs(outdir, exist_ok=True)
        d = dict(software=dict(name='robot36dec', version=__version__), error=msg, stage=stage, runtime_s=time.time() - t_start)
        if extra: d.update(extra)
        with open(os.path.join(outdir, f'{basename}_diagnostics.json'), 'w') as fh: json.dump(d, fh, indent=1)
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    """Pipeline per SPEC.md section 3. Returns 0 on success, 2 if no credible signal (diagnostics still written), 1 on error.
    Prints a short human-readable summary (timing, VIS, confidence, output paths) unless --quiet."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:                     # argparse exits 2 on a usage error; 2 is this program's 'no signal' code
        code = e.code if isinstance(e.code, int) else 1
        return 0 if code == 0 else 1            # --help -> 0, usage error -> 1
    if args.cmd != 'decode':
        parser.print_help(); return 1
    t_start = time.time()
    outdir = os.path.abspath(args.out)
    basename = args.basename or os.path.splitext(os.path.basename(args.input))[0]
    quiet = bool(args.quiet)
    log = (lambda *a, **k: None) if quiet else (lambda *a, **k: print(*a, **k, flush=True))
    stage = 'setup'
    try:
        if args.iq:
            if args.iq_rate is None or args.center_freq is None:
                raise ValueError('--iq requires --iq-rate and --center-freq')
            if args.t0 is not None or args.period is not None or args.start or args.end is not None:
                raise ValueError('candidate-preserving IQ analysis scans the whole input; timing overrides/windows are WAV-only')
            from .candidates import analyze
            from .io import IQSpec
            params=_load_params(args);params.iq_search_hz=args.iq_search_hz
            last=[0.]
            def progress(stage,fraction):
                now=time.time()
                if now-last[0]>=20:log(f'{stage}: {fraction:.0%}');last[0]=now
            report=analyze(args.input,outdir,IQSpec(args.iq_rate,args.center_freq,args.signal_freq),params,
                           tuple(float(v) for v in args.channel_bandwidths.split(',')),args.receiver_ppm,
                           args.uncertainty_ppm,progress=progress)
            log(f"{report['verdict']}: {len(report['candidates'])} RF hypotheses; {outdir}/candidates.json")
            return 0 if report['verdict'] in ('PROBABLE','CONFIRMED ROBOT36') else 2
        return _run(args, outdir, basename, log, t_start)
    except KeyboardInterrupt:
        return 1
    except Exception as e:
        stage = getattr(e, '_stage', stage)
        print(f'robot36dec: error during {stage}: {e}', file=sys.stderr)
        if not quiet: traceback.print_exc()
        _write_error_json(outdir, basename, f'{type(e).__name__}: {e}', stage, t_start, extra=dict(input=os.path.abspath(args.input)))
        return 1


class _StageError(Exception):
    def __init__(self, stage, exc):
        super().__init__(f'{exc}'); self._stage = stage; self.__cause__ = exc


def _run(args, outdir, basename, log, t_start) -> int:
    from . import io as r36io, iqdemod, preprocess, header, sync, timing as timing_mod, decode as decode_mod, diagnostics
    m = ROBOT36
    nominal = m.line_ms / 1000.0
    warnings = []
    params = _load_params(args)
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f'input not found: {args.input}')
    os.makedirs(outdir, exist_ok=True)
    log(f'robot36dec {__version__}: {args.input} -> {outdir}/{basename}_*  (mode {args.mode}, estimator {params.estimator}, gate {params.gate_mode})')

    # ------------------------------------------------------------------ audio stage
    iq_diag = None
    try:
        if args.iq:
            if args.iq_rate is None or args.center_freq is None:
                raise ValueError('--iq requires --iq-rate and --center-freq')
            spec = r36io.IQSpec(fs=float(args.iq_rate), center_freq=float(args.center_freq), signal_freq=float(args.signal_freq))
            audio, iq_diag = iqdemod.demodulate(args.input, spec, params)
            input_kind = f'IQ u8 {args.iq_rate / 1e6:.4g} MS/s, centre {args.center_freq / 1e6:.6g} MHz, signal {args.signal_freq / 1e6:.6g} MHz'
        else:
            audio = r36io.load_wav(args.input, params.wav_fullscale_hz, params.fs)
            orig_fs = audio.meta.get('original_fs', audio.fs)
            input_kind = (f'WAV {orig_fs} Hz, {audio.meta.get("channels", 1)} ch, full scale = {params.wav_fullscale_hz:g} Hz'
                          + (f', resampled {orig_fs} -> {audio.fs} Hz ({audio.meta.get("resample_up")}/{audio.meta.get("resample_down")})' if audio.meta.get('resampled') else ''))
    except Exception as e:
        raise _StageError('audio input', e) from e
    fs = int(audio.fs); x_full = np.asarray(audio.samples, np.float32); t_file0 = float(audio.t0)
    audio.samples = None                        # the only reference to the full-length audio is x_full from here on
    total_dur = len(x_full) / fs
    log(f'audio: {input_kind}; {total_dur:.3f} s at {fs} Hz')
    if iq_diag is not None:
        nb = int(iq_diag.get('n_blocks', 0) or 0); nl = int(iq_diag.get('n_blocks_locked', 0) or 0)
        nnb = int(iq_diag.get('n_blocks_narrowband_rejected', 0) or 0)
        f_exp = float(args.signal_freq) - float(args.center_freq)
        if nb and nl == 0:
            msg = (f'IQ: the carrier was never detected: no block locked within +-{float(iq_diag.get("search_hz", 0)) / 1e3:.1f} kHz of the expected '
                   f'offset {f_exp / 1e3:+.1f} kHz (signal {args.signal_freq / 1e6:.6f} MHz - centre {args.center_freq / 1e6:.6f} MHz). The demodulated audio is '
                   f'receiver noise. Check --center-freq / --signal-freq / --iq-rate'
                   + (f'; {nnb} block(s) held only a narrowband (CW) carrier, which is not an FM signal' if nnb else ''))
            warnings.append(msg); log('warning: ' + msg)
        else:
            log(f'IQ: carrier locked in {nl}/{nb} blocks ({100.0 * nl / max(nb, 1):.0f} %), first {iq_diag.get("first_lock_s")} s, last {iq_diag.get("last_lock_s")} s, '
                f'mean offset {float(iq_diag.get("f_offset_mean_hz", 0.0)):+.0f} Hz from the expected {f_exp / 1e3:+.1f} kHz, carrier SNR mean '
                f'{iq_diag.get("carrier_snr_mean_db")} dB in {float(iq_diag.get("channel_bw_hz", 0)) / 1e3:.0f} kHz'
                + (f'; {nnb} block(s) with a narrowband interferer rejected' if nnb else ''))
            if nnb and nnb >= 0.5 * nb:
                msg = f'IQ: a narrowband (CW) carrier at {iq_diag.get("narrowband_hz_median")} Hz was rejected in {nnb}/{nb} blocks; the SSTV carrier was tracked next to it'
                warnings.append(msg); log('warning: ' + msg)

    # ------------------------------------------------------------------ window
    start = max(float(args.start), t_file0); end = t_file0 + total_dur if args.end is None else min(float(args.end), t_file0 + total_dur)
    i0 = int(round((start - t_file0) * fs)); i1 = int(round((end - t_file0) * fs))
    if i1 - i0 < 2 * fs:
        raise _StageError('window', ValueError(f'analysis window too short ({(i1 - i0) / fs:.2f} s); need >= 2 s'))
    if (i0, i1) == (0, len(x_full)):
        x = x_full                              # whole file: no copy
    else:
        x = x_full[i0:i1].copy()                # window: copy it and release the rest
    x_t0 = t_file0 + i0 / fs; duration = len(x) / fs
    del x_full
    if (i0, i1) != (0, int(round(total_dur * fs))): log(f'window: {x_t0:.3f}-{x_t0 + duration:.3f} s ({duration:.1f} s)')

    # ------------------------------------------------------------------ preprocessing
    try:
        t1 = time.time()
        x_dc = preprocess.remove_dc(x, fs, params.dc_window_s)
        del x                                   # the raw audio is not needed again (x_dc is the pre-gate reference)
        x_g, gate_info = preprocess.threshold_gate(x_dc, fs, params)
        xb = preprocess.bandpass(x_g, fs, params.bp_lo_hz, params.bp_hi_hz)
        del x_g
        log(f'preprocess: DC removed ({params.dc_window_s:g} s trimmed mean); gate {params.gate_mode} -> {"APPLIED" if gate_info.applied else "not applied"} '
            f'(L {gate_info.level_hz:g} Hz, M {gate_info.M}, frac |x|>L {gate_info.frac_above_level:.4f}, two-population score {gate_info.two_population_score:.2f}'
            f'{"; " + gate_info.note if gate_info.note else ""}); band-pass {params.bp_lo_hz:g}-{params.bp_hi_hz:g} Hz  [{time.time() - t1:.1f} s]')
    except Exception as e:
        raise _StageError('preprocessing', e) from e

    # ------------------------------------------------------------------ carrier presence (wideband quieting of the pre-gate audio)
    if args.iq:
        hb = float(params.iq_channel_bw_hz) / 2.0
        qband = (min(8000.0, 0.5 * hb), min(20000.0, 0.9 * hb))
    else:
        qband = (8000.0, 20000.0)
    try:
        cp = carrier_presence(x_dc, fs, x_t0, params, qband)
        if cp['runs']:
            log(f'carrier: quieting runs ({qband[0]:.0f}-{qband[1]:.0f} Hz, threshold {cp["threshold_db"]:.2f} dB): '
                + ', '.join(f'{a:.1f}-{b:.1f} s' for a, b in cp['runs'][:8]) + (' ...' if len(cp['runs']) > 8 else ''))
        else:
            log(f'carrier: no quieting run found (sigma {cp["sigma_db"]:.2f} dB)')
    except Exception as e:
        warnings.append(f'carrier presence failed: {e}'); cp = dict(runs=[], t=np.zeros(0), level_db=np.zeros(0), threshold_db=float('nan'), sigma_db=float('nan'))

    # ------------------------------------------------------------------ header (normal path; informational in recovery mode)
    vis = []
    try:
        t1 = time.time()
        vis = list(header.find_vis(xb, fs, x_t0, params) or [])
        if vis:
            b = vis[0]
            log(f'VIS: best candidate code {b.vis_code} ({b.mode_name}), leader {b.t_leader:.4f} s, line 0 {b.t_line0:.4f} s, z {b.z:.1f} '
                f'(noise max {b.noise_max:.2f}), parity {"OK" if b.parity_ok else "BAD"}, plateau {b.plateau_ms:.0f} ms, confidence {b.confidence:.2f}'
                f'{" [ignored: recovery mode]" if args.mode == "recovery" else ""}  [{time.time() - t1:.1f} s]')
        else:
            log(f'VIS: no candidate  [{time.time() - t1:.1f} s]')
    except Exception as e:
        warnings.append(f'find_vis failed: {e}'); log(f'warning: find_vis failed: {e}')
    # the candidate used for timing is the first credible Robot 36 one (the list is sorted by confidence; a credible header of
    # another mode or an image-band false detection may precede it); vis[0] is still what the report calls 'best'
    vis_r36 = [c for c in vis if c.vis_code == m.vis and c.confidence >= 0.6]
    vis_best = vis_r36[0] if vis_r36 else (vis[0] if vis else None)
    vis_credible = bool(vis_r36)
    use_vis = (args.mode != 'recovery') and vis_credible
    if vis and vis_best is not vis[0]:
        log(f'VIS: Robot 36 candidate used for timing: leader {vis_best.t_leader:.4f} s, confidence {vis_best.confidence:.2f}')

    # ------------------------------------------------------------------ sync likelihood, frame locator, period, phase, line tracking
    period_est = None; line_sync = None; span = None; locator = 'none'; t_s = s = None; fold = None
    try:
        t1 = time.time()
        t_s, s = sync.sync_likelihood(xb, fs, x_t0, params)
        t_s = np.asarray(t_s, np.float64); s = np.asarray(s, np.float64)
        log(f'sync: likelihood computed ({len(s)} frames at {params.sync_hop_ms:g} ms)  [{time.time() - t1:.1f} s]')
    except Exception as e:
        warnings.append(f'sync_likelihood failed: {e}'); log(f'warning: sync_likelihood failed: {e}')
    frame_len = m.height * nominal
    locator_note = ''; pulse_span = None; header_blank = None; s_trk = s
    if t_s is not None and len(t_s) > 10:
        try:
            hulls = _merge_runs(cp['runs'], gap_s=10.0)
            # choose the analysis window for the ACF / fold
            if use_vis:
                win = (vis_best.t_line0 - 1.0, vis_best.t_line0 + frame_len + 1.0); locator = 'vis'
                for a, b in hulls:
                    if a - 2.0 <= vis_best.t_line0 <= b: span = (a, b); break
            else:
                cands = [(a, b) for a, b in hulls if b - a >= 10.0]
                locator = 'carrier-runs' if cands else 'comb-scan'
                if not cands:
                    step = 20.0; wlen = frame_len + 4.0
                    starts = np.arange(t_s[0], max(t_s[0] + 1e-3, t_s[-1] - wlen + step), step)
                    cands = [(float(a), float(min(t_s[-1], a + wlen))) for a in starts] or [(float(t_s[0]), float(t_s[-1]))]
                best = None
                for a, b in cands:
                    tw, sw = _window(t_s, s, a - 1.0, b + 1.0)
                    if len(sw) < int(3.0 / (params.sync_hop_ms / 1000.0)): continue
                    try:
                        pe = sync.estimate_period_acf(tw, sw, params); z = float(pe.comb_z)
                    except Exception as e:
                        warnings.append(f'estimate_period_acf failed on {a:.1f}-{b:.1f}: {e}'); continue
                    if best is None or z > best[0]: best = (z, a, b)
                if best is None:
                    win = (float(t_s[0]), float(t_s[-1]))
                else:
                    z, a, b = best
                    if locator == 'carrier-runs': span = (a, b)
                    win = (a - 2.0, max(b + 2.0, a + frame_len + 2.0))
                    log(f'frame locator ({locator}): window {a:.1f}-{b:.1f} s, comb z {z:.1f}')
            tw, sw = _window(t_s, s, win[0], win[1])
            if len(sw) < 100: tw, sw = t_s, s
            period_est = sync.estimate_period_acf(tw, sw, params)
            P0 = float(period_est.period) if (period_est.period is not None and np.isfinite(period_est.period) and period_est.period > 0) else nominal
            if abs(P0 / nominal - 1.0) * 1e6 > params.period_tolerance_ppm: P0 = nominal
            log(f'period (ACF comb on {tw[0]:.1f}-{tw[-1]:.1f} s): Robot 36 period {"-" if period_est.period is None else f"{period_est.period * 1000:.4f} ms"} '
                f'(comb z {period_est.comb_z:.1f}, ACF(150 ms) {period_est.acf_150:.3f}); best anywhere {period_est.best_period_any * 1000:.3f} ms (z {period_est.best_z_any:.1f})')
            try:
                profile, phase, contrast = sync.fold_phase(tw, sw, P0)
                fold = dict(pulse_phase_s=float(phase), contrast=float(contrast))
                log(f'fold: pulse phase {phase:.4f} s, contrast {contrast:.1f}')
            except Exception as e:
                warnings.append(f'fold_phase failed: {e}'); phase = None
            if use_vis:
                t_init = float(vis_best.t_line0)
                # blank the leader/break/start/data bits; the stop bit (last 30 ms) is kept because it is the header-end
                # signature the recovery placement uses to anchor line 0 (line 0's own pulse fails the shape test anyway)
                header_blank = (float(vis_best.t_leader) - 0.02, float(vis_best.t_line0) - 0.035)
            else:
                # header-free: track the pulse train over a wide window first, then start the 240-line frame at the first
                # run of pulses whose 240-line window holds (nearly) the most inliers (redundant with the carrier onset)
                t_from_w = max(float(t_s[0]), win[0] - 20.0); t_to_w = min(float(t_s[-1]), win[1] + 20.0)
                if phase is not None: t_init_w = _pulse_grid_start(phase, P0, t_from_w, t_s, s, m.sync_ms)
                else: t_init_w = t_from_w
                n_wide = int((t_to_w - t_init_w) / P0)
                t_init = None
                if n_wide >= m.height + 10:
                    try:
                        ls_w = _track(sync, t_s, s, t_init_w, P0, params, n_wide)
                        inl_w = np.asarray(ls_w.inlier, bool)
                        ef_w = float(getattr(ls_w, 'expected_false_inliers', float('nan')))
                        p0_w = min(max(ef_w / max(1, len(inl_w)), 0.002), 0.2) if np.isfinite(ef_w) else 0.02
                        k0, cnt, k_last, m_min_w = _frame_start_from_inliers(inl_w, m.height, p0=p0_w,
                                                                             window_lines=int(getattr(params, 'frame_start_window_lines', 12)),
                                                                             false_alarm=float(getattr(params, 'frame_start_false_alarm', 1e-3)))
                        P_w = float(ls_w.period) if (np.isfinite(ls_w.period) and abs(ls_w.period / nominal - 1) * 1e6 <= params.period_tolerance_ppm and ls_w.n_inliers >= 8) else P0
                        t0_w = float(ls_w.t_line0) if (np.isfinite(ls_w.t_line0) and ls_w.n_inliers >= 5) else t_init_w
                        t_init = t0_w + k0 * P_w
                        locator_note = (f'pulse-train start at wide-track line {k0} of {n_wide} ({cnt} inliers inside the 240-line frame, {ls_w.n_inliers} in the wide window; '
                                        f'dense start = >= {m_min_w} inliers within {int(getattr(params, "frame_start_window_lines", 12))} lines at a false-inlier rate of {p0_w:.3f}/line)')
                        # the VIS start bit (1200 Hz, 30 ms) lies exactly 2 line periods before line 0 and passes as the first sync
                        # pulse; it is preceded by the 300 ms 1900 Hz leader, whereas line 0 is preceded by the VIS data bits
                        lead_share = _leader_share(xb, fs, x_t0, t_init)
                        if lead_share > 0.25:
                            t_init += 2.0 * P_w; k0 += 2
                            header_blank = (t_init - (VIS_TOTAL_MS + 40.0) / 1000.0, t_init - 0.035)
                            locator_note += f'; 1900 Hz leader found before the first pulse (share {lead_share:.2f}): it is the VIS start bit, line 0 moved +2 lines'
                        pulse_span = (t_init, t0_w + (k_last + 1) * P_w)
                        log(f'frame start (pulse train): {t_init:.4f} s; {locator_note}')
                    except Exception as e:
                        warnings.append(f'wide pulse tracking failed: {e}')
                if t_init is None:
                    t_from = (span[0] - 0.3) if span else (win[0] + 1.0)
                    t_init = _pulse_grid_start(phase, P0, max(t_from, float(tw[0])), tw, sw, m.sync_ms) if phase is not None else float(t_from)
            # the VIS header's 1200 Hz elements (break, start bit, stop bit) fall inside the search windows of lines -2..0 and
            # steal their pulses; once the header is located it is blanked (set to the median) in the copy of the sync
            # likelihood used for tracking and timing (the diagnostics keep the original)
            if header_blank is not None:
                s_trk = s.copy(); sel = (t_s >= header_blank[0]) & (t_s <= header_blank[1]); s_trk[sel] = float(np.median(s))
                log(f'sync: VIS header {header_blank[0]:.3f}-{header_blank[1]:.3f} s excluded from pulse tracking')
            line_sync = sync.track_lines(t_s, s_trk, t_init, P0, params)
            log(f'sync fit: {line_sync.n_inliers} inlier pulses, line 0 {line_sync.t_line0:.4f} s, period {line_sync.period * 1000:.4f} ms '
                f'({(line_sync.period / nominal - 1) * 1e6:+.0f} ppm), residual sd {line_sync.resid_ms_sd:.2f} ms, runs {_fmt_runs(line_sync.runs)}')
        except Exception as e:
            warnings.append(f'sync stage failed: {e}'); log(f'warning: sync stage failed: {e}')
            if not args.quiet: traceback.print_exc()
    # only the carrier hull (independent evidence) constrains the frame placement; the pulse-train span is redundant with the
    # inlier pulses themselves and, being cut from a 240-line window, can truncate the frame end by a few lines (reported only)
    span_source = 'carrier' if span is not None else None
    if period_est is None:
        period_est = sync.PeriodEstimate(period=None, comb_z=0.0, acf_150=0.0, best_period_any=nominal, best_z_any=0.0, candidates=[])

    # ------------------------------------------------------------------ timing resolution
    decode_quick = functools.partial(decode_mod.quick_luma, xb, fs, x_t0, params=params)
    parity_fn = functools.partial(_parity_fraction, xb, fs, x_t0)
    line_snr_fn = functools.partial(_line_snr, xb, fs, x_t0, params)
    rt_args = (vis if args.mode != 'recovery' else [], period_est, line_sync, decode_quick, params)
    rt_kw = dict(mode=args.mode, t0_override=args.t0, period_override=args.period, signal_span=span)
    try:
        try:
            tmg = timing_mod.resolve_timing(*rt_args, **rt_kw, sync_signal=(t_s, s_trk) if t_s is not None else None, parity_fn=parity_fn,
                                            line_snr_fn=line_snr_fn, audio_span=(x_t0, x_t0 + duration))
        except TypeError:
            tmg = timing_mod.resolve_timing(*rt_args, **rt_kw)
    except Exception as e:
        warnings.append(f'resolve_timing failed: {e}'); log(f'warning: resolve_timing failed: {e}')
        if not args.quiet: traceback.print_exc()
        tmg = None
    if tmg is None or not (np.isfinite(tmg.t_line0) and np.isfinite(tmg.period) and tmg.period > 0):
        # last-resort fallback so that diagnostics are still produced
        if args.t0 is not None: t0f, src = float(args.t0), 'override'
        elif use_vis: t0f, src = float(vis_best.t_line0), 'vis (fallback)'
        elif line_sync is not None and np.isfinite(line_sync.t_line0): t0f, src = float(line_sync.t_line0), 'sync-fit (fallback)'
        else: t0f, src = (span[0] if span else x_t0 + 1.0), 'guess (fallback)'
        pf = float(args.period) if args.period else (float(line_sync.period) if (line_sync is not None and np.isfinite(line_sync.period) and abs(line_sync.period / nominal - 1) * 1e6 < params.period_tolerance_ppm) else nominal)
        tmg = timing_mod.Timing(t_line0=t0f, period=pf, source=src, confidence=0.0, notes='resolve_timing unavailable; fallback timing')
    log(f'timing: line 0 = {tmg.t_line0:.4f} s, period {tmg.period * 1000:.5f} ms ({(tmg.period / nominal - 1) * 1e6:+.0f} ppm), source {tmg.source}, '
        f'confidence {tmg.confidence:.2f}' + (f'; agreement {json.dumps(diagnostics.to_plain(tmg.agreement))}' if tmg.agreement else '') + (f'; {tmg.notes}' if tmg.notes else ''))
    if tmg.t_line0 + m.height * tmg.period > x_t0 + duration + 1.0 or tmg.t_line0 < x_t0 - 1.0:
        warnings.append('frame extends beyond the analysed audio; lines outside are unreliable')
    # the per-line sync detections reported in the CSV/JSON must be indexed on the FINAL timing (the recovery placement
    # and the parity fix may move line 0 by whole lines relative to the pulse grid the fit was made on): evaluate the pulses
    # on the resolved grid without refitting; the fit's LineSync keeps the period / credibility statistics
    line_sync_fit = line_sync
    if line_sync is not None and t_s is not None and len(t_s) > 10 and args.t0 is None:
        try:
            ls_final = sync.track_lines(t_s, s_trk, float(tmg.t_line0), float(tmg.period), params, threshold=line_sync.threshold, refit=False)
            if abs(ls_final.n_inliers - line_sync.n_inliers) > max(4, 0.25 * line_sync.n_inliers):
                log(f'sync: {ls_final.n_inliers} pulses lie on the resolved frame grid ({line_sync.n_inliers} on the fitted pulse grid)')
            line_sync = ls_final
        except Exception as e:
            warnings.append(f'final line-sync evaluation failed: {e}')

    # Independent structure measurement precedes rendering. Overrides select a grid only.
    from .protocol import measure_geometry, inspect_vis
    measured=measure_geometry(xb,fs,tmg.t_line0-x_t0,tmg.period)
    measured['window_start_s']=x_t0
    measured['vis']=inspect_vis(xb,fs,x_t0,params)
    tmg.agreement['protocol_evidence']=measured
    if not measured['valid']:
        extra=dict(software=dict(version=__version__),input=args.input,detected=False,
                   verdict='NO EVIDENCE',reason=measured['reason'],protocol_evidence=measured,
                   timing=diagnostics.to_plain(tmg),images=[])
        with open(os.path.join(outdir,f'{basename}_diagnostics.json'),'w') as fh:
            json.dump(diagnostics.to_plain(extra),fh,indent=2,allow_nan=False)
        log('NO EVIDENCE: '+measured['reason'])
        return 2
    # ------------------------------------------------------------------ decode
    try:
        t1 = time.time()
        result = decode_mod.decode_robot36(xb, fs, x_t0, tmg, params, raw_audio=x_dc, gate_applied=bool(gate_info.applied), quieting_band_hz=qband)
        log(f'decode: {result.estimator_detail or result.estimator}; SNR regime {result.snr_regime} (median line SNR {result.snr_median_db:+.1f} dB); '
            f'calibration {result.calibration} (mask threshold {result.mask_threshold_hz:.1f} Hz, fusion gate {result.fuse_threshold_hz:.1f} Hz'
            + (f', control at {result.control["t0"]:.1f} s x {result.control["n_lines"]} lines: row-corr {result.control["rowcorr_Y"]:+.3f}, reliable {result.control["frac_reliable"] * 100:.2f} %' if result.control else '')
            + f'); reliable pixels {result.frac_reliable * 100:.1f} % in lines {_fmt_runs(result.reliable_runs)}; separator parity agreement {result.sep_parity_agreement:.2f}'
            + (f' ({result.sep_parity_agreement_reliable:.2f} on reliable lines)' if np.isfinite(result.sep_parity_agreement_reliable) else '')
            + f'; row-corr all {decode_mod.rowcorr(result.Y):+.3f}  [{time.time() - t1:.1f} s]')
    except Exception as e:
        raise _StageError('decode', e) from e

    # ------------------------------------------------------------------ decision, outputs
    detected, score, crit = _signal_decision(tmg, vis_best, period_est, line_sync, result, params, args.mode)
    exit_code = 0 if detected else 2
    extra = dict(input=os.path.abspath(args.input), input_kind=input_kind, duration_s=total_dur, analysis_window_s=[x_t0, x_t0 + duration],
                 mode=args.mode, vis_used_for_timing=use_vis, frame_locator=locator, frame_locator_note=locator_note, signal_span_s=list(span) if span else None,
                 signal_span_source=span_source, pulse_train_span_s=list(pulse_span) if pulse_span else None,
                 vis_header_excluded_from_sync_s=list(header_blank) if header_blank else None,
                 carrier_runs_s=[list(r) for r in cp.get('runs', [])], carrier_threshold_db=cp.get('threshold_db'), carrier_sigma_db=cp.get('sigma_db'),
                 quieting_band_hz=list(qband), fold=fold, overrides=dict(t0=args.t0, period=args.period),
                 signal_detected=detected, signal_score=score, signal_criteria=crit, signal_decision_reason=crit.get('reason', ''),
                 exit_code=exit_code, warnings=warnings,
                 line_sync_fit=dict(t_line0=float(line_sync_fit.t_line0), period=float(line_sync_fit.period), n_inliers=int(line_sync_fit.n_inliers),
                                    resid_ms_sd=float(line_sync_fit.resid_ms_sd), runs=list(line_sync_fit.runs)) if line_sync_fit is not None else None,
                 iq=diagnostics.to_plain(iq_diag) if iq_diag is not None else None, runtime_s=time.time() - t_start,
                 argv=sys.argv[1:] if len(sys.argv) > 1 else None)
    audio_for_plots = dict(x=xb, fs=fs, x_t0=x_t0, sync_t=t_s, sync_s=s, raw=x_dc)
    try:
        paths = diagnostics.write_outputs(result, tmg, vis, period_est, line_sync, gate_info, params, outdir, basename, plots=bool(args.plots),
                                          extra=extra, audio_for_plots=audio_for_plots)
    except Exception as e:
        raise _StageError('write_outputs', e) from e
    if args.save_audio:
        try:
            import soundfile as sf
            scale = float(params.wav_fullscale_hz) if params.wav_fullscale_hz else 48000.0
            p1 = os.path.join(outdir, f'{basename}_audio_demod.wav'); sf.write(p1, (x_dc / scale).astype(np.float32), fs, subtype='FLOAT'); paths['audio_demod'] = p1
            p2 = os.path.join(outdir, f'{basename}_audio_preprocessed.wav'); sf.write(p2, (xb / scale).astype(np.float32), fs, subtype='FLOAT'); paths['audio_preprocessed'] = p2
        except Exception as e:
            warnings.append(f'save-audio failed: {e}'); log(f'warning: save-audio failed: {e}')

    # ------------------------------------------------------------------ summary
    log('')
    log(f'== {basename}: {"SIGNAL DECODED" if detected else "NO CREDIBLE ROBOT 36 SIGNAL (exit 2)"}  score {score:.2f}  '
        f'[VIS {"yes" if crit["vis_credible"] else "no"}, comb z {crit["comb_z"]:.1f}, sync inliers {crit["n_sync_inliers"]}, reliable {crit["frac_reliable"] * 100:.1f} %, timing conf {crit["timing_confidence"]:.2f}'
        f'{"" if crit["robot36_structure_ok"] else ", NOT a Robot 36 line structure"}]')
    if crit.get('reason'): log(f'   reason: {crit["reason"]}')
    if vis_best is not None:
        log(f'   VIS: code {vis_best.vis_code} = {VIS_TABLE.get(vis_best.vis_code, vis_best.mode_name)}, parity {"OK" if vis_best.parity_ok else "BAD"}, z {vis_best.z:.1f}, confidence {vis_best.confidence:.2f}, leader {vis_best.t_leader:.4f} s'
            f'{" (used for the timing)" if use_vis else " (reported only; timing from the sync pulses)"}')
    log(f'   timing: line 0 {tmg.t_line0:.4f} s, period {tmg.period * 1000:.5f} ms ({(tmg.period / nominal - 1) * 1e6:+.0f} ppm), source {tmg.source}, confidence {tmg.confidence:.2f}')
    if line_sync is not None:
        log(f'   line sync: {line_sync.n_inliers} detected pulses (runs {_fmt_runs(line_sync.runs)}), fit period {line_sync.period * 1000:.4f} ms, residual sd {line_sync.resid_ms_sd:.2f} ms')
    for name, (a, b) in decode_mod.BLOCKS.items():
        snr = result.line_snr_db[a:b]
        log(f'   lines {a:3d}-{b:3d}: SNR median {np.nanmedian(snr):+5.1f} dB, reliable {result.mask[a:b].mean() * 100:5.1f} %, row-corr Y {decode_mod.rowcorr(result.Y, a, b):+.3f}')
    if warnings: log('   warnings: ' + ' | '.join(warnings))
    log(f'   outputs ({len(paths)}): {outdir}/{basename}_image.png, _luma.png, _luma_fused.png, _image_fused_masked.png, _mask.png, _diagnostics.json, _line_sync.csv, _line_metrics.csv'
        + (f', _plots/ ({sum(1 for k in paths if k.startswith("plot_"))} plots)' if args.plots else '') + f'   [{time.time() - t_start:.1f} s]')
    return exit_code
