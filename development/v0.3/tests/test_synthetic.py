"""SPEC.md section 5 step 1: synthetic validation.

Part A checks the synthesiser itself with an independent reference decoder (passes without the package decoder).
Parts B-D exercise the public API and the command line on synthetic Robot 36 transmissions through the rtl_fm-style
FM channel: both timing paths, VIS detection, truth correlation, pure noise -> exit 2, IQ u8 round trips, and the
guarantee that input files are never modified. Tolerances: SPEC.md (timing < 1 ms, period < 20 ppm, correlation
>= 0.95 at +6 dB and >= 0.85 at 0 dB); the lower-CNR correlation floors (0.70 at -2 dB, 0.50 at -4 dB) are this
suite's conservative additions (the gate + chain estimator reached 0.93-0.98 at -4..0 dB on this simulator).
"""
import csv
import json
import math

import numpy as np
import pytest

import helpers
from helpers import FS, FREQS, pearson, line_offset, ppm_error
from robot36dec import synth
from robot36dec.modes import ROBOT36, VIS_TOTAL_MS

T0_TOL_S = 1.0e-3            # timing error tolerance (WAV path)
T0_TOL_IQ_S = 2.0e-3         # IQ path: decimation/channel FIR group delays are allowed to cost up to 1 ms more
PERIOD_TOL_PPM = 20.0
VIS_MIN_CONF = 0.6           # a VIS "claim" = candidate with confidence >= 0.6 (resolve_timing's credibility threshold)


def _timing_errors(run, case):
    tm = run.timing
    assert tm is not None, "no timing record in the diagnostics JSON\n" + run.describe()
    k, resid = line_offset(float(tm["t_line0"]), case.t_line0, case.period)
    return tm, k, resid, ppm_error(float(tm["period"]), case.period)


def period_tol_ppm(n_lines: int) -> float:
    """20 ppm for a full 240-line frame (SPEC). A partial frame of n lines gives the sync fit a shorter lever arm: at a
    fixed per-pulse jitter the slope error scales as n^-1.5 (120 lines -> 57 ppm, 60 lines -> 160 ppm)."""
    return PERIOD_TOL_PPM * (240.0 / max(1, int(n_lines))) ** 1.5


def _assert_timing(run, case, tol_s=T0_TOL_S, allow_line_offset=False):
    tm, k, resid, ppm = _timing_errors(run, case)
    tol_ppm = period_tol_ppm(case.n_lines)
    msg = (f"timing: t_line0 {tm['t_line0']:.5f} vs truth {case.t_line0:.5f} (k={k} lines, residual "
           f"{resid*1e3:+.3f} ms, tolerance {tol_s*1e3:.1f} ms), period {tm['period']:.8f} ({ppm:+.1f} ppm, tolerance "
           f"{tol_ppm:.0f} ppm for {case.n_lines} lines), source {tm['source']!r}, confidence {tm['confidence']}\n"
           + run.describe())
    if allow_line_offset:
        assert abs(resid) < tol_s, msg
    else:
        assert k == 0 and abs(resid) < tol_s, msg
    assert abs(ppm) < tol_ppm, msg
    return tm


def _reference_corr(case):
    """Truth correlation of the simplest possible estimator (12 ms argmax, true timing) on the same audio: what the
    decoder must at least match. Computed only for failure messages."""
    audio = case.audio_hz
    if audio is None and case.kind == "iq":
        from conftest import iq_offset_fn
        audio = helpers.local_iq_demod(case.path, case.iq_rate, iq_offset_fn)      # knows the true offset
    if audio is None:
        return float("nan")
    Y = helpers.local_luma_argmax(audio, FS, 0.0, case.t_line0, case.period, case.n_lines)
    return pearson(Y, case.truth_Y)


def _assert_corr(run, case, threshold):
    luma = run.luma()
    assert luma.shape == (240, 320), luma.shape
    c = pearson(luma[:case.n_lines], case.truth_Y)
    if c < threshold:
        dec = helpers.find_record(run.diag, ("estimator",)) if run.diag else None
        est = dec.get("estimator") if dec else "?"
        pytest.fail(f"truth correlation {c:.4f} < {threshold} (first {case.n_lines} lines, estimator {est!r}); "
                    f"independent 12 ms argmax with the true timing reaches {_reference_corr(case):.4f} on this audio\n"
                    + run.describe())
    return c


# ============================================================================================ A. synthesiser
class TestSynthModule:
    def test_card_is_deterministic_and_well_formed(self):
        a = synth.test_card(); b = synth.test_card()
        assert a.shape == (240, 320, 3) and a.dtype == np.uint8
        assert np.array_equal(a, b)
        Y = helpers.luma_truth(a)
        assert Y.min() == 0.0 and Y.max() == 255.0                      # black bar and white rectangle
        assert np.all(a[0:20] == 0) is np.False_                         # title bar carries text-like blocks
        assert np.all(a[2, :, 0] == 0)                                   # ... but the bar's top rows are black
        assert synth.test_card(160, 120).shape == (120, 160, 3)

    def test_encoder_layout_and_vis_bits(self):
        card = synth.test_card()
        track, t0 = synth.encode_robot36(card, FS, with_vis=True, pre_silence_s=1.0, post_silence_s=1.0)
        assert track.dtype == np.float32
        assert abs(t0 - (1.0 + VIS_TOTAL_MS / 1000.0)) < 1e-12
        assert abs(len(track) / FS - (1.0 + 0.910 + 240 * 0.150 + 1.0)) < 1.0 / FS
        # silence before/after, leader at 1.0 s, break at 1.3 s, start bit at 1.61 s
        assert helpers.tone_at(track, FS, 0.5) == 0.0 and track[-1] == 0.0
        assert helpers.tone_at(track, FS, 1.15) == 1900.0
        assert helpers.tone_at(track, FS, 1.305) == 1200.0
        assert helpers.tone_at(track, FS, 1.625) == 1200.0
        # VIS 8 bits LSB first + even parity: 0,0,0,1,0,0,0,1  (1300 Hz = 0, 1100 Hz = 1), 30 ms each from 1.64 s
        bits = [helpers.tone_at(track, FS, 1.64 + 0.015 + 0.03 * k) for k in range(8)]
        assert bits == [1300, 1300, 1300, 1100, 1300, 1300, 1300, 1100]
        assert synth.vis_bits(8) == [0, 0, 0, 1, 0, 0, 0, 1]
        assert synth.vis_bits(44) == [0, 0, 1, 1, 0, 1, 0, 1]           # Martin 1: three ones -> parity 1
        assert helpers.tone_at(track, FS, 1.64 + 0.24 + 0.015) == 1200.0  # stop bit
        # line layout for several lines: sync 1200 (0-9 ms), porch 1500 (9-12), separator 1500/2300 (100-104.5),
        # porch 1900 (104.5-106)
        m = ROBOT36
        for n in (0, 1, 2, 3, 100, 239):
            ls = t0 + n * 0.150
            assert helpers.tone_at(track, FS, ls + 0.0045) == m.f_sync
            assert helpers.tone_at(track, FS, ls + 0.0105) == m.f_black
            assert helpers.tone_at(track, FS, ls + 0.10225) == (m.f_black if n % 2 == 0 else m.f_white)
            assert helpers.tone_at(track, FS, ls + 0.10525) == 1900.0
        # luminance pixels: black bar -> 1500 Hz, white rectangle (line 180, px 50) -> 2300 Hz, ramp mid grey
        assert helpers.tone_at(track, FS, t0 + 2 * 0.150 + 0.012 + 0.5 * 0.088 / 320) == 1500.0
        assert helpers.tone_at(track, FS, t0 + 180 * 0.150 + 0.012 + 50.5 * 0.088 / 320) == 2300.0
        f_mid = helpers.tone_at(track, FS, t0 + 40 * 0.150 + 0.012 + 160.5 * 0.088 / 320)
        assert 1850.0 < f_mid < 1950.0
        # every active sample carries a legal SSTV tone
        act = track[int(1.0 * FS):int((len(track) / FS - 1.0) * FS) - 1]
        assert act.min() >= 1100.0 and act.max() <= 2300.0

    def test_encoder_partial_frame_and_period_scaling(self):
        card = synth.test_card()
        tr, t0 = synth.encode_robot36(card, FS, with_vis=False, n_lines=60, pre_silence_s=0.5, post_silence_s=0.25)
        assert abs(t0 - 0.5) < 1e-12
        assert abs(len(tr) / FS - (0.5 + 60 * 0.150 + 0.25)) < 1.0 / FS
        assert helpers.tone_at(tr, FS, 0.5 + 59 * 0.150 + 0.0045) == 1200.0
        p = 0.14999640                                                     # the recording's -24 ppm period
        tr2, t02 = synth.encode_robot36(card, FS, with_vis=True, line_period_s=p, n_lines=240)
        for n in (0, 120, 239):
            assert helpers.tone_at(tr2, FS, t02 + n * p + 0.0045) == 1200.0
            assert helpers.tone_at(tr2, FS, t02 + n * p + 0.0105) == 1500.0
        # the last line ends exactly 240 periods after line 0
        end = t02 + 240 * p
        assert tr2[int(round(end * FS)) + 2] == 0.0 and tr2[int(round(end * FS)) - 3] != 0.0
        with pytest.raises(ValueError):
            synth.encode_robot36(np.zeros((240, 320)), FS)

    @pytest.mark.parametrize("oversample", [8, 1])
    def test_channel_at_high_cnr_decodes_with_local_argmax(self, oversample):
        """Encode + simulate at +10 dB, independent 12 ms argmax luminance decode: correlation with the card > 0.95."""
        card = synth.test_card()
        Y = helpers.luma_truth(card)
        track, t0 = synth.encode_robot36(card, FS, with_vis=True)
        audio = synth.simulate_fm_channel(track, 10.0, FS, seed=1, oversample=oversample)
        assert audio.dtype == np.float32 and audio.shape == track.shape
        Yd = helpers.local_luma_argmax(audio, FS, 0.0, t0, 0.150, 240)
        c = pearson(Yd, Y)
        assert c > 0.95, c
        # leader tone amplitude after de-emphasis (alpha = 1/4): 3000 Hz deviation * |H(1900 Hz)| = 0.758 -> ~2270 Hz
        seg = audio[int(1.05 * FS):int(1.25 * FS)]
        n = np.arange(len(seg)); w = np.hanning(len(seg))
        amp = 2 * np.abs(np.sum(seg * w * np.exp(-2j * np.pi * 1900 * n / FS))) / np.sum(w)
        assert 2150 < amp < 2400, amp
        assert synth.deemphasis_alpha(48000, 75.0) == 0.25
        assert abs(synth.deemphasis_alpha(48000, 75.0, rtlfm_integer=False) - 0.2425) < 1e-3
        with pytest.raises(ValueError):
            synth.simulate_fm_channel(track[:FS], 0.0, FS, oversample=6)

    def test_channel_noise_only_statistics(self):
        """Carrier off: rtl_fm's below-threshold output (bounded +-24 kHz, RMS ~0.109 FS, ~half the samples beyond 0.08 FS)."""
        track = np.zeros(10 * FS, np.float32)
        a = synth.simulate_fm_channel(track, 0.0, FS, seed=5, carrier_on=False)
        b = synth.simulate_fm_channel(track, 0.0, FS, seed=5, carrier_on=False)
        c = synth.simulate_fm_channel(track, 0.0, FS, seed=6, carrier_on=False)
        assert np.array_equal(a, b) and not np.array_equal(a, c)         # deterministic per seed
        rms = np.sqrt(np.mean(a.astype(np.float64) ** 2)) / 48000.0
        assert 0.095 < rms < 0.125, rms                                     # recording: 0.1046 FS
        assert abs(a.mean()) < 150.0
        assert np.abs(a).max() <= 24000.0 + 1.0
        frac = np.mean(np.abs(a) > 3840.0)                                  # the gate level 0.08 FS
        assert 0.40 < frac < 0.60, frac
        # no tone: the 1.1-2.3 kHz band is featureless (peak/median of a long spectrum below ~15 dB)
        S = np.abs(np.fft.rfft(a[:1 << 18] * np.hanning(1 << 18))) ** 2
        f = np.fft.rfftfreq(1 << 18, 1 / FS); m = (f > 1000) & (f < 2600)
        assert 10 * np.log10(S[m].max() / np.median(S[m])) < 16.0

    def test_channel_doppler_fading_arrays_and_carrier_gate(self):
        card = synth.test_card()
        track, t0 = synth.encode_robot36(card, FS, with_vis=True, n_lines=8)
        hi = synth.simulate_fm_channel(track, 30.0, FS, seed=2, doppler_hz=-1000.0)
        # Doppler offset appears as a DC of the discriminator output during the carrier-on part
        dc = hi[int(1.05 * FS):int(1.6 * FS)].mean()
        assert abs(dc + 1000.0) < 60.0, dc
        assert abs(hi[int(0.1 * FS):int(0.9 * FS)].mean()) < 300.0        # carrier off: no Doppler, noise only
        dop = np.linspace(0.0, -2000.0, len(track))
        cnr = -2.0 + 2.5 * np.sin(2 * np.pi * np.arange(len(track)) / FS / 4.0)
        out = synth.simulate_fm_channel(track, cnr, FS, seed=3, doppler_hz=dop)
        assert out.shape == track.shape and np.isfinite(out).all()
        # forcing the carrier on during the "silence" produces a quiet (captured) output there
        on = synth.simulate_fm_channel(track, 20.0, FS, seed=4, carrier_on=True)
        assert np.sqrt(np.mean(on[int(0.2 * FS):int(0.8 * FS)] ** 2)) < 2000.0
        with pytest.raises(ValueError):
            synth.simulate_fm_channel(track, 0.0, FS, carrier_on=np.ones(5, bool))

    def test_write_iq_u8_roundtrip_with_local_demod(self, tmp_path):
        """The u8 IQ writer: size, interleaving, level, and a local demod (true offset known) recovers the picture."""
        card = synth.test_card()
        Y = helpers.luma_truth(card)
        track, t0 = synth.encode_robot36(card, FS, with_vis=True, n_lines=40)
        fs_iq = 250000.0
        path = tmp_path / "rt.u8"
        off = lambda t: 10000.0 + 1500.0 - 100.0 * np.asarray(t, np.float64)
        n = synth.write_iq_u8(track, FS, fs_iq, off, 10.0, str(path), seed=4)
        raw = np.fromfile(path, np.uint8)
        assert n == int(round(len(track) / FS * fs_iq)) and raw.size == 2 * n
        assert 125.0 < raw.mean() < 130.0 and 25.0 < raw.std() < 50.0
        assert np.mean((raw == 0) | (raw == 255)) < 0.01
        n2 = synth.write_iq_u8(track, FS, fs_iq, off, 10.0, str(tmp_path / "rt2.u8"), seed=4)
        assert n2 == n and np.array_equal(raw, np.fromfile(tmp_path / "rt2.u8", np.uint8))   # deterministic
        audio = helpers.local_iq_demod(path, fs_iq, off)
        Yd = helpers.local_luma_argmax(audio, FS, 0.0, t0, 0.150, 40)
        assert pearson(Yd, Y[:40]) > 0.93


# ============================================================================================ B. public API
class TestPublicAPI:
    def test_load_wav_units_and_meta(self, case_p6_vis):
        from robot36dec.io import load_wav, AudioSignal
        sig = load_wav(str(case_p6_vis.path), fullscale_hz=48000.0)
        assert isinstance(sig, AudioSignal)
        assert sig.fs == FS and sig.t0 == 0.0 and sig.samples.dtype == np.float32
        assert len(sig.samples) == len(case_p6_vis.audio_hz)
        err = sig.samples.astype(np.float64) - case_p6_vis.audio_hz
        assert np.abs(err).max() < 1.6                                      # int16 step = 48000/32768 = 1.46 Hz
        for key in ("duration_s", "fullscale_hz", "channels"):
            assert key in sig.meta, sig.meta
        assert abs(sig.meta["duration_s"] - case_p6_vis.duration_s) < 1e-3
        half = load_wav(str(case_p6_vis.path), fullscale_hz=24000.0)
        assert np.allclose(half.samples[:FS], 0.5 * sig.samples[:FS], atol=1e-3)
        assert case_p6_vis.input_unchanged()

    def test_preprocess_gate_modes(self, noise_wav, case_p6_vis, make_wav):
        from robot36dec.params import Params
        from robot36dec.preprocess import remove_dc, threshold_gate, bandpass, GateInfo
        p = Params()
        x = noise_wav.audio_hz[:10 * FS]
        # 'on' / 'off' are unconditional
        for mode, expect in (("on", True), ("off", False)):
            p.gate_mode = mode
            y, info = threshold_gate(x, FS, p)
            assert isinstance(info, GateInfo) and info.applied is expect and info.mode_requested == mode, info
            assert y.shape == x.shape
            if expect:
                rms_in = np.sqrt(np.mean(x.astype(np.float64) ** 2)); rms_out = np.sqrt(np.mean(y.astype(np.float64) ** 2))
                assert rms_out < 0.35 * rms_in, (rms_in, rms_out)        # the +-fs/2 noise floor is suppressed
                assert 0.4 < info.frac_above_level < 0.6 and info.frac_zeroed > 0.3, info
            else:
                assert np.array_equal(y, x)
        # 'auto' on a below-threshold transmission (-4 dB, the recording's regime): the mixture of the narrow
        # captured-carrier core and the uniform +-fs/2 noise floor is the two-population signature the gate must detect
        p.gate_mode = "auto"
        low = make_wav(-4.0, False, 240)
        _, info = threshold_gate(low.audio_hz, FS, p)
        assert info.applied and info.two_population_score > 1.0, info
        # at +6 dB only the ~5 % carrier-off samples sit on the floor: either decision is acceptable, but it is recorded
        _, info6 = threshold_gate(case_p6_vis.audio_hz, FS, p)
        assert info6.mode_requested == "auto" and isinstance(info6.applied, bool), info6
        # pure noise alone is a single population: auto is not required to fire, but the decision must be recorded
        _, info_n = threshold_gate(x, FS, p)
        assert info_n.mode_requested == "auto" and isinstance(info_n.applied, bool), info_n
        assert 0.4 < info_n.frac_above_level < 0.6, info_n
        # a clean tone of unknown scale (generic sound-card audio): no signature -> not applied, signal untouched
        t = np.arange(10 * FS) / FS
        tone = (1500.0 * np.sin(2 * np.pi * 1900.0 * t)).astype(np.float32)
        y, info = threshold_gate(tone, FS, p)
        assert not info.applied and np.allclose(y, tone), info
        # DC removal and band-pass keep the tone and remove the offset
        xd = remove_dc(tone + 20000.0, FS, 0.5)
        assert abs(xd[FS:-FS].mean()) < 50.0 and np.allclose(xd[FS:-FS], tone[FS:-FS], atol=60.0)
        xb = bandpass(tone, FS, 400.0, 3200.0)
        assert xb.dtype == np.float32 and abs(np.std(xb[FS:-FS]) / np.std(tone[FS:-FS]) - 1.0) < 0.05

    def _preprocessed(self, case):
        from robot36dec.params import Params
        from robot36dec.preprocess import remove_dc, threshold_gate, bandpass
        p = Params()
        x = remove_dc(case.audio_hz, FS, p.dc_window_s)
        x, info = threshold_gate(x, FS, p)
        x = bandpass(x, FS, p.bp_lo_hz, p.bp_hi_hz)
        return x, p, info

    def test_find_vis_on_synthetic(self, case_p6_vis, case_p6_novis):
        from robot36dec.header import find_vis, VISCandidate
        x, p, _ = self._preprocessed(case_p6_vis)
        cands = find_vis(x, FS, 0.0, p)
        assert cands and isinstance(cands[0], VISCandidate)
        best = cands[0]
        assert best.vis_code == 8 and best.parity_ok and best.mode_name == "Robot 36", best
        assert best.confidence >= VIS_MIN_CONF and best.z >= p.vis_min_z, best
        assert abs(best.t_leader - case_p6_vis.t_leader) < T0_TOL_S, best
        assert abs(best.t_line0 - case_p6_vis.t_line0) < T0_TOL_S, best
        assert len(best.bits) == 8 and best.bits == [0, 0, 0, 1, 0, 0, 0, 1], best.bits
        assert best.plateau_ms >= 10.0
        assert all(c.confidence <= best.confidence for c in cands)       # sorted
        # no header -> nothing credible claiming Robot 36
        x2, p2, _ = self._preprocessed(case_p6_novis)
        cands2 = find_vis(x2, FS, 0.0, p2)
        assert not [c for c in cands2 if c.vis_code == 8 and c.confidence >= VIS_MIN_CONF], cands2

    def test_sync_period_fold_and_tracking(self, case_p6_novis, noise_wav):
        from robot36dec.sync import sync_likelihood, estimate_period_acf, fold_phase, track_lines, PeriodEstimate, LineSync
        case = case_p6_novis
        x, p, _ = self._preprocessed(case)
        t, s = sync_likelihood(x, FS, 0.0, p)
        assert t.shape == s.shape and len(t) > 1000
        hop = np.diff(t)
        assert np.allclose(hop, p.sync_hop_ms / 1000.0, atol=1e-6) and np.isfinite(s).all()
        pe = estimate_period_acf(t, s, p)
        assert isinstance(pe, PeriodEstimate) and pe.period is not None
        assert abs(ppm_error(pe.period, case.period)) < PERIOD_TOL_PPM, pe
        # comb significance: the statistic saturates (~5) for a perfectly periodic train because the ACF's own comb
        # structure sets the sd; require the decoder's detection level and a clear margin over pure noise
        assert pe.comb_z >= 3.0, pe
        xn, pn, _ = self._preprocessed(noise_wav)
        tn, sn = sync_likelihood(xn, FS, 0.0, pn)
        pe_n = estimate_period_acf(tn, sn, pn)
        assert pe.comb_z > pe_n.comb_z + 2.0, (pe.comb_z, pe_n.comb_z)
        assert pe_n.comb_z < 3.0, pe_n
        assert pe.acf_150 > 0.5 > pe_n.acf_150, (pe.acf_150, pe_n.acf_150)
        profile, phase, contrast = fold_phase(t, s, pe.period)
        assert contrast > 3.0 and len(profile) > 100, (contrast, len(profile))
        # the pulse centre phase must agree with truth (line start + 4.5 ms) modulo the period
        k, resid = line_offset(phase, case.t_line0 + ROBOT36.sync_ms / 2000.0, pe.period)
        assert abs(resid) < 1.5e-3, (phase, resid)
        # tracking from a deliberately wrong initial guess (+3 ms, +50 ppm) must converge on the true grid
        ls = track_lines(t, s, case.t_line0 + 0.003, case.period * (1 + 50e-6), p)
        assert isinstance(ls, LineSync) and ls.t_pulse.shape == (p.n_lines,) and ls.inlier.shape == (p.n_lines,)
        assert ls.n_inliers >= 0.8 * p.n_lines, ls.n_inliers
        assert abs(ls.t_line0 - case.t_line0) < T0_TOL_S, ls.t_line0
        assert abs(ppm_error(ls.period, case.period)) < PERIOD_TOL_PPM, ls.period
        assert ls.resid_ms_sd < 1.0 and ls.runs, (ls.resid_ms_sd, ls.runs)

    def test_resolve_timing_override_and_recovery(self, case_p6_novis):
        from robot36dec.timing import resolve_timing, Timing
        from robot36dec.sync import PeriodEstimate
        stub = lambda t0, period, **kw: (np.zeros((240, 320)), 0.0)
        pe = PeriodEstimate(period=None, comb_z=0.0, acf_150=0.0, best_period_any=0.15, best_z_any=0.0)
        tm = resolve_timing([], pe, None, stub, __import__("robot36dec.params", fromlist=["Params"]).Params(),
                            mode="auto", t0_override=12.345, period_override=0.1499)
        assert isinstance(tm, Timing) and tm.source == "override"
        assert tm.t_line0 == 12.345 and tm.period == 0.1499
        # weak-signal inputs must not raise: a usable low-confidence Timing comes back
        tm2 = resolve_timing([], pe, None, stub, __import__("robot36dec.params", fromlist=["Params"]).Params(),
                             mode="recovery", signal_span=(0.0, 40.0))
        assert isinstance(tm2, Timing) and tm2.confidence <= 0.3 and math.isfinite(tm2.t_line0)
        assert 0.1 < tm2.period < 0.2

    def test_decode_robot36_with_true_timing(self, case_p6_vis):
        from robot36dec.decode import decode_robot36, DecodeResult, pixel_times
        from robot36dec.timing import Timing
        case = case_p6_vis
        x, p, info = self._preprocessed(case)
        ty, tc = pixel_times(case.t_line0, case.period, p)
        assert ty.shape == (240, 320) and tc.shape == (240, 320)
        assert abs(ty[0, 0] - (case.t_line0 + 0.012 + 0.5 * 0.088 / 320)) < 1e-9
        assert abs(tc[1, 319] - (case.t_line0 + case.period + 0.106 + 319.5 * 0.044 / 320)) < 1e-9
        tm = Timing(t_line0=case.t_line0, period=case.period, source="truth", confidence=1.0)
        res = decode_robot36(x, FS, 0.0, tm, p, raw_audio=case.audio_hz, gate_applied=info.applied)
        assert isinstance(res, DecodeResult)
        assert res.Y.shape == (240, 320) and res.C.shape == (240, 320)
        assert res.rgb.shape == (240, 320, 3) and res.rgb.dtype == np.uint8
        assert res.mask.dtype == np.bool_ and res.mask.shape == (240, 320)
        c = pearson(res.Y, case.truth_Y)
        assert c >= 0.95, (f"decode_robot36 corr {c:.4f} with estimator {res.estimator!r}; independent 12 ms argmax "
                           f"reaches {pearson(helpers.local_luma_argmax(x, FS, 0.0, case.t_line0, case.period, 240, bandpass=False), case.truth_Y):.4f}")
        assert pearson(res.Y_fused, case.truth_Y) >= 0.85      # 3-line fusion blurs thin horizontal features by design
        assert res.frac_reliable >= 0.8 and res.mask.mean() >= 0.8
        assert res.sep_parity_agreement >= 0.9
        assert np.mean(res.line_sep_parity == 1) >= 0.9
        assert res.line_snr_db.shape == (240,) and np.median(res.line_snr_db) > 0.0
        assert res.line_quieting_db.shape == (240,) and np.isfinite(res.line_quieting_db).all()
        assert res.estimator in ("chain", "soft")
        # colour: the red rectangle (lines 205-230, px 20-100) must come out red, the blue disc blue
        red = res.rgb[210:225, 30:90].reshape(-1, 3).mean(0)
        blue = res.rgb[180:200, 235:265].reshape(-1, 3).mean(0)
        assert red[0] > red[1] + 60 and red[0] > red[2] + 60, red
        assert blue[2] > blue[0] + 60 and blue[2] > blue[1] + 30, blue

    def test_cli_main_in_process_noise_returns_2(self, noise_wav, tmp_path):
        from robot36dec.cli import main, build_parser
        ns = build_parser().parse_args(["decode", str(noise_wav.path), "--out", str(tmp_path)])
        assert ns.cmd == "decode" and ns.mode == "auto"
        rc = main(["decode", str(noise_wav.path), "--out", str(tmp_path), "--quiet"])
        assert rc == 2
        diag_path = tmp_path / f"{noise_wav.path.stem}_diagnostics.json"
        assert diag_path.exists()
        diag = json.loads(diag_path.read_text())
        assert not helpers.credible_claims(helpers.get_vis_candidates(diag), VIS_MIN_CONF)
        assert noise_wav.input_unchanged()


# ============================================================================================ C. command line
CLI_CASES = [
    # id,                    cnr,  vis,   mode,       lines, corr, expect_vis_used
    ("p6_vis_auto_full",      6.0,  True,  "auto",     240,   0.95, True),
    ("p6_vis_recovery_full",  6.0,  True,  "recovery", 240,   0.95, False),
    ("p6_novis_auto_full",    6.0,  False, "auto",     240,   0.95, False),
    ("0_vis_auto_120",        0.0,  True,  "auto",     120,   0.85, True),
    ("0_novis_auto_120",      0.0,  False, "auto",     120,   0.85, False),
    ("m2_vis_normal_120",    -2.0,  True,  "normal",   120,   0.70, True),
    ("m2_novis_recovery_120", -2.0, False, "recovery", 120,   0.70, False),
    ("m4_novis_recovery_full", -4.0, False, "recovery", 240,  0.50, False),
]


@pytest.mark.cli
class TestCLI:
    @pytest.mark.parametrize("cid,cnr,vis,mode,n_lines,corr_min,vis_used", CLI_CASES, ids=[c[0] for c in CLI_CASES])
    def test_decode_matrix(self, cid, cnr, vis, mode, n_lines, corr_min, vis_used, make_wav, decode_cli, record_property):
        case = make_wav(cnr, vis, n_lines)
        run = decode_cli(case, mode, extra=("--plots",) if cid == "p6_vis_auto_full" else ())
        assert run.returncode == 0, run.describe()
        assert run.diag is not None, "diagnostics JSON missing\n" + run.describe()
        tm = _assert_timing(run, case)
        record_property("elapsed_s", round(run.elapsed_s, 1))
        record_property("timing", {k: tm[k] for k in ("t_line0", "period", "source", "confidence")})
        conf_min = 0.6 if cnr >= 0 else 0.35
        assert float(tm["confidence"]) >= conf_min, run.describe()
        cands = run.vis
        claims = helpers.vis8_claims(cands, VIS_MIN_CONF)
        if vis_used:
            assert claims, "VIS 8 present but not credibly detected\n" + run.describe()
            best = max(claims, key=lambda c: c.get("confidence", 0))
            assert best.get("parity_ok") is True, best
            assert abs(float(best["t_leader"]) - case.t_leader) < T0_TOL_S, best
            assert abs(float(best["t_line0"]) - case.t_line0) < T0_TOL_S, best
            assert helpers.uses_vis(tm["source"]), tm
        else:
            assert not helpers.uses_vis(tm["source"]), tm            # recovery / header-less: timing from sync only
            if not vis:
                assert not claims, "VIS claimed on a header-less transmission\n" + run.describe()
        c = _assert_corr(run, case, corr_min)
        record_property("truth_corr", round(c, 4))

    def test_outputs_complete_and_well_formed(self, run_p6_vis_auto, case_p6_vis):
        run = run_p6_vis_auto
        assert run.returncode == 0, run.describe()
        for suffix in helpers.OUTPUT_SUFFIXES:
            assert run.path(suffix).exists(), f"missing output {run.path(suffix).name}\n" + run.describe()
        assert helpers.png_size(run.path("_image.png"))[:2] == (320, 240)
        assert helpers.png_size(run.path("_image.png"))[2] in ("RGB", "RGBA")
        assert helpers.png_size(run.path("_luma.png"))[:2] == (320, 240)
        assert helpers.png_size(run.path("_luma_x3.png"))[:2] == (960, 720)
        assert helpers.png_size(run.path("_luma_fused.png"))[:2] == (320, 240)
        assert helpers.png_size(run.path("_image_fused_masked.png"))[:2] == (320, 240)
        assert helpers.png_size(run.path("_mask.png"))[:2] == (320, 240)
        assert pearson(helpers.load_gray_png(run.path("_luma_fused.png")), case_p6_vis.truth_Y) >= 0.85
        mask = helpers.load_gray_png(run.path("_mask.png"))
        assert (mask > 127).mean() >= 0.8                                    # clean +6 dB: nearly all reliable
        for name, n_rows in (("_line_sync.csv", 240), ("_line_metrics.csv", 240)):
            with open(run.path(name), newline="") as fh:
                rows = list(csv.reader(fh))
            assert len(rows) == n_rows + 1, (name, len(rows))
            assert "line" in [h.strip().lower() for h in rows[0]], rows[0]
        with open(run.path("_line_sync.csv"), newline="") as fh:
            hdr = [h.strip().lower() for h in next(csv.reader(fh))]
        for col in ("t_expected", "t_detected", "residual_ms", "strength", "inlier"):
            assert col in hdr, hdr
        diag = run.diag
        txt = json.dumps(diag)                                               # plain JSON types only
        assert "NaN" not in txt                   # NaN allowed by json, but must load
        assert helpers.get_timing(diag) and helpers.get_period_estimate(diag) is not None
        assert helpers.get_line_sync(diag) is not None and helpers.get_gate(diag) is not None
        assert helpers.get_vis_candidates(diag)
        ls = helpers.get_line_sync(diag)
        assert int(ls["n_inliers"]) >= 0.8 * 240, ls
        gate = helpers.get_gate(diag)
        assert isinstance(gate["applied"], bool) and gate["mode_requested"] == "auto", gate
        # software version and parameters are recorded
        assert helpers.find_record(diag, ("win_ms", "estimator", "gate_mode")) is not None
        assert any(k for k in diag if "version" in k.lower()) or helpers.find_record(diag, ("version",)) is not None
        plots = run.outdir / f"{run.basename}_plots"
        assert plots.is_dir() and len(list(plots.glob("*.png"))) >= 3, list(run.outdir.iterdir())

    def test_pure_noise_exit_2_and_no_vis_claim(self, noise_wav, decode_cli):
        run = decode_cli(noise_wav, "auto")
        assert run.returncode == 2, run.describe()
        assert run.diag is not None, "diagnostics must still be written for no-signal inputs\n" + run.describe()
        tm = run.timing
        assert tm is not None and float(tm["confidence"]) <= 0.3, tm
        assert not helpers.uses_vis(tm["source"]), tm
        assert not helpers.credible_claims(run.vis, VIS_MIN_CONF), run.vis
        assert run.path("_luma.png").exists() and run.path("_diagnostics.json").exists()
        assert noise_wav.input_unchanged()

    def test_input_file_is_never_modified(self, case_p6_vis, run_p6_vis_auto, decode_cli):
        """Hash/size/mtime before (taken when the file was written) and after decodes in both modes."""
        decode_cli(case_p6_vis, "recovery")
        assert run_p6_vis_auto.returncode == 0, run_p6_vis_auto.describe()
        assert helpers.sha256_file(case_p6_vis.path) == case_p6_vis.sha256
        assert case_p6_vis.input_unchanged(), "input file or its directory changed"
        # and nothing was written outside --out
        top = {p.name for p in run_p6_vis_auto.outdir.iterdir()}
        assert top and all(n.startswith(run_p6_vis_auto.basename) for n in top), top

    def test_overrides_win(self, make_wav, decode_cli):
        case = make_wav(0.0, False, 120)
        run = decode_cli(case, "auto", extra=("--t0", f"{case.t_line0:.6f}", "--period", f"{case.period:.8f}"), tag="override")
        assert run.returncode == 0, run.describe()
        tm = run.timing
        assert tm["source"] == "override", tm
        assert abs(float(tm["t_line0"]) - case.t_line0) < 1e-6 and abs(float(tm["period"]) - case.period) < 1e-9
        _assert_corr(run, case, 0.85)

    def test_cli_rejects_bad_arguments(self, noise_wav, tmp_path):
        import subprocess
        from conftest import PYTHON, ROOT
        proc = subprocess.run([PYTHON, "-m", "robot36dec", "decode", str(tmp_path / "missing.wav"), "--out",
                               str(tmp_path / "o")], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
        proc = subprocess.run([PYTHON, "-m", "robot36dec", "decode", str(noise_wav.path), "--iq", "--out",
                               str(tmp_path / "o2")], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0, "--iq without --iq-rate/--center-freq must fail"
        proc = subprocess.run([PYTHON, "-m", "robot36dec", "nonsense"], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
        assert proc.returncode != 0


# ============================================================================================ D. IQ u8 round trip
@pytest.mark.iq
@pytest.mark.cli
class TestIQ:
    @pytest.mark.parametrize("fs_iq", [250000.0, 1024000.0], ids=["250kSps", "1024kSps"])
    def test_iq_u8_roundtrip(self, fs_iq, make_iq, decode_cli, record_property):
        """u8 IQ at fs_iq, carrier 10 kHz + 1.5 kHz off the nominal position with a -100 Hz/s Doppler ramp, CNR +6 dB
        (48 kHz reference), 120-line frame: channel selection, tracking, demodulation and the normal path must work."""
        case = make_iq(fs_iq, cnr_db=6.0, n_lines=120)
        assert case.path.stat().st_size == 2 * int(round(case.duration_s * fs_iq))
        run = decode_cli(case, "auto")
        assert run.returncode == 0, run.describe()
        tm = _assert_timing(run, case, tol_s=T0_TOL_IQ_S)
        record_property("elapsed_s", round(run.elapsed_s, 1))
        claims = helpers.vis8_claims(run.vis, VIS_MIN_CONF)
        assert claims, "VIS 8 not detected on the IQ path\n" + run.describe()
        assert helpers.uses_vis(tm["source"]), tm
        c = _assert_corr(run, case, 0.90)
        record_property("truth_corr", round(c, 4))
        # the carrier track (if reported) must have found the carrier near its true initial offset
        track = helpers.find_record(run.diag, ("f_offset",))
        if track is not None and isinstance(track.get("f_offset"), list) and track["f_offset"]:
            f0 = float(np.nanmedian(np.asarray(track["f_offset"][:4], float)))
            assert abs(f0 - 1500.0) < 400.0 or abs(f0 - 11500.0) < 400.0, f0
        assert case.input_unchanged()
