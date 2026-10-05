"""Synthetic Robot 36 signals for tests: encoder, rtl_fm-style FM channel, IQ u8 writer.

Ported (logic copied, nothing imported) from
  claude-recovery/scripts/13_fm_threshold_sim.py      - Robot 36 encoder, simple 48 kS/s NBFM channel + discriminator
  claude-recovery/take2/limiter/synth_lib.py          - rtl_fm-faithful chain: oversampled complex noise, binomial
                                                        half-band decimation passes, cic_9_tables[5] droop FIR,
                                                        atan2 discriminator, rtl_fm de-emphasis
This module is test support only: nothing in the decoding path imports it.

Units: all "audio" returned here is instantaneous frequency deviation in Hz at the demodulator output rate
(1.0 FS of an rtl_fm int16 WAV = 48000 Hz; multiply by 1/48000 to write an rtl_fm-compatible WAV).
"""
import numpy as np
from scipy.signal import lfilter

from .modes import ROBOT36, VIS_LEADER, VIS_BIT_MS

# rtl_fm fifth_order() half-band kernel [1,5,10,10,5,1] (rtl_fm shifts >>4 i.e. DC gain 2; we normalise to DC gain 1,
# which does not change the discriminator output) and the 9-tap droop-compensation FIR cic_9_tables[5] (5 passes).
_HALFBAND = np.array([1, 5, 10, 10, 5, 1], np.float32) / 32.0
_DROOP_COMP = np.array([-120, -602, 6015, -26269, 77757, -26269, 6015, -602, -120], np.float32) / 65536.0
_CARRIER_CHUNK = 1 << 21          # complex samples per chunk when synthesising the carrier (bounds memory)

__all__ = ["test_card", "encode_robot36", "simulate_fm_channel", "write_iq_u8", "rgb_to_ycrcb", "vis_bits",
           "deemphasis_alpha"]


# ----------------------------------------------------------------------------------------------- test image
def test_card(width: int = 320, height: int = 240, seed: int = 0) -> np.ndarray:
    """Deterministic RGB test image: black title bar, grey ramp, coloured discs, white/red rectangles, text-like blocks."""
    from PIL import Image, ImageDraw
    sx, sy = width / 320.0, height / 240.0

    def X(v):
        return int(round(v * sx))

    def Yc(v):
        return int(round(v * sy))

    img = Image.new("RGB", (width, height), (90, 90, 90))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, width, Yc(20)], fill=(0, 0, 0))                                  # title bar
    for i in range(width):                                                                # grey ramp
        d.line([i, Yc(20), i, Yc(60)], fill=(int(255 * i / width),) * 3)
    d.ellipse([X(30), Yc(70), X(110), Yc(150)], fill=(230, 180, 150))                     # beige disc
    d.ellipse([X(130), Yc(80), X(190), Yc(140)], fill=(30, 30, 30))                       # dark disc
    d.ellipse([X(250), Yc(70), X(310), Yc(130)], fill=(240, 240, 240))                    # white disc
    d.ellipse([X(200), Yc(150), X(300), Yc(230)], fill=(40, 80, 200))                     # blue disc
    d.rectangle([X(20), Yc(170), X(160), Yc(200)], fill=(255, 255, 255))                  # white rectangle
    d.rectangle([X(20), Yc(205), X(100), Yc(230)], fill=(200, 40, 40))                    # red rectangle
    # text-like blocks: a row of bright "words" in the title bar and a row of dark "words" below the discs.
    # Word widths 20-40 px (5.5-11 ms) so that a 12 ms analysis window can still resolve them (4-14 px "letters"
    # would be unresolvable by design and would cap the achievable truth correlation at ~0.94 even without noise).
    rng = np.random.default_rng(seed)
    for (y0, y1, x_end, colour) in ((Yc(5), Yc(15), X(300), (235, 235, 235)),
                                    (Yc(153), Yc(163), X(190), (25, 25, 25))):
        x = X(10)
        while x < x_end:
            w = X(int(rng.integers(20, 41)))
            d.rectangle([x, y0, min(x + w, x_end), y1], fill=colour)
            x += w + X(int(rng.integers(12, 25)))
    return np.asarray(img, dtype=np.uint8).copy()


def rgb_to_ycrcb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """JPEG YCrCb (Y, Cr, Cb) as floats 0..255 from an (H, W, 3) RGB array (the encoder's colour model)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    Y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    Cr = 128.0 + 0.713 * (rgb[..., 0] - Y)
    Cb = 128.0 + 0.564 * (rgb[..., 2] - Y)
    return Y, Cr, Cb


def vis_bits(code: int) -> list[int]:
    """8 VIS bits LSB first: 7 data bits + even parity bit (Robot 36 = 8 -> [0,0,0,1,0,0,0,1])."""
    bits = [(int(code) >> i) & 1 for i in range(7)]
    return bits + [sum(bits) % 2]


# ----------------------------------------------------------------------------------------------- encoder
def _level_to_freq(v):
    return ROBOT36.f_black + (ROBOT36.f_white - ROBOT36.f_black) * np.clip(np.asarray(v, np.float64), 0, 255) / 255.0


def encode_robot36(rgb: np.ndarray, fs: int = 48000, with_vis: bool = True, pre_silence_s: float = 1.0,
                   post_silence_s: float = 1.0, line_period_s: float = 0.150, n_lines: int | None = None,
                   vis_code: int = 8) -> tuple[np.ndarray, float]:
    """Phase-continuous tone sequence (instantaneous frequency in Hz, float32) of VIS (if with_vis) + 240 lines;
    returns (freq_track_hz, t_line0) where t_line0 is the absolute start of line 0 in the returned array.
    Silence is represented by frequency 0 Hz (no tone) and the channel simulator turns the carrier off there.

    Extra keyword arguments (defaults reproduce the contract): `n_lines` transmits only the first n_lines lines of the
    image (partial frame for fast tests; rows wrap if n_lines > image height); `vis_code` selects the VIS code (8).
    A non-nominal `line_period_s` scales every segment of the line proportionally (a transmitter clock error).
    Segment boundaries are sample-accurate: sample i (time i/fs) carries the frequency of the segment containing i/fs,
    exactly as in scripts/13_fm_threshold_sim.py."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape (H, W, 3)")
    H, W = rgb.shape[0], rgb.shape[1]
    n_lines = H if n_lines is None else int(n_lines)
    Y, Cr, Cb = rgb_to_ycrcb(rgb)
    m = ROBOT36
    scale = float(line_period_s) / (m.line_ms / 1000.0)              # within-line stretch (1.0 at 150.000 ms)

    seg_f, seg_d = [], []                                             # frequencies (Hz), durations (ms)
    if with_vis:
        for f, dur in VIS_LEADER:
            seg_f.append(f); seg_d.append(dur)
        for b in vis_bits(vis_code):
            seg_f.append(1100.0 if b else 1300.0); seg_d.append(VIS_BIT_MS)
        seg_f.append(1200.0); seg_d.append(VIS_BIT_MS)                # stop bit
    t_line0 = float(pre_silence_s) + sum(seg_d) / 1000.0
    px_y = np.full(W, m.y_ms / W); px_c = np.full(W, m.c_ms / W)
    for ln in range(n_lines):
        row = ln % H
        chroma = Cr[row] if ln % 2 == 0 else Cb[row]
        f_line = np.concatenate([[m.f_sync, m.f_black], _level_to_freq(Y[row]),
                                 [m.f_black if ln % 2 == 0 else m.f_white, 1900.0], _level_to_freq(chroma)])
        d_line = np.concatenate([[m.sync_ms, m.porch_ms], px_y, [m.sep_ms, m.porch2_ms], px_c]) * scale
        seg_f.append(f_line); seg_d.append(d_line)
    freqs = np.concatenate([np.atleast_1d(np.asarray(v, np.float64)) for v in seg_f]) if seg_f else np.zeros(0)
    durs = np.concatenate([np.atleast_1d(np.asarray(v, np.float64)) for v in seg_d]) / 1000.0 if seg_d else np.zeros(0)
    edges = float(pre_silence_s) + np.concatenate([[0.0], np.cumsum(durs)])
    t_end = edges[-1]
    N = int(round((t_end + float(post_silence_s)) * fs))
    t = np.arange(N) / float(fs)
    track = np.zeros(N, np.float32)
    if len(freqs):
        active = (t >= edges[0]) & (t < t_end)
        k = np.searchsorted(edges, t[active], side="right") - 1
        track[active] = freqs[np.clip(k, 0, len(freqs) - 1)]
    return track, t_line0


# ----------------------------------------------------------------------------------------------- FM channel
def deemphasis_alpha(fs: int, deemphasis_us: float, rtlfm_integer: bool = True) -> float:
    """One-pole de-emphasis coefficient. rtl_fm: deemph_a = round(1/(1-exp(-1/(fs*tau)))) and y += (x-y)/deemph_a,
    i.e. alpha = 1/4 exactly at 48 kHz / 75 us (verified on the recording: inverse bounded by +-0.5 FS).
    rtlfm_integer=False gives the exact analogue value 1-exp(-1/(fs*tau)) (= 0.2425 at 48 kHz)."""
    a = 1.0 - np.exp(-1.0 / (fs * deemphasis_us * 1e-6))
    return 1.0 / max(1, int(round(1.0 / a))) if rtlfm_integer else float(a)


def _tone_audio(freq_track_hz: np.ndarray, fs: float) -> np.ndarray:
    """Unit-amplitude SSTV audio sin(2 pi cumsum(f)/fs) (float64 phase; silence = constant level, carrier off there)."""
    f = np.asarray(freq_track_hz, np.float64)
    return np.sin(2.0 * np.pi * np.cumsum(f) / fs)


def _carrier_on(freq_track_hz: np.ndarray, carrier_on) -> np.ndarray:
    f = np.asarray(freq_track_hz)
    if carrier_on is None:
        return f != 0
    if np.isscalar(carrier_on) or np.ndim(carrier_on) == 0:
        return np.full(f.shape, bool(carrier_on))
    on = np.asarray(carrier_on, bool)
    if on.shape != f.shape:
        raise ValueError("carrier_on must be a scalar or match freq_track_hz")
    return on


def _discriminate(z: np.ndarray, fs: float) -> np.ndarray:
    """Polar discriminator in Hz: f[n] = fs/(2 pi) angle(z[n] conj z[n-1]); f[0] = f[1] (rtl_fm polar_discriminant)."""
    d = np.angle(z[1:] * np.conj(z[:-1])).astype(np.float64) * (fs / (2.0 * np.pi))
    return np.concatenate([d[:1], d]) if len(d) else np.zeros(len(z))


def _deemphasise(f_hz: np.ndarray, fs: float, deemphasis_us: float, rtlfm_integer: bool) -> np.ndarray:
    if not deemphasis_us or deemphasis_us <= 0:
        return f_hz
    a = deemphasis_alpha(fs, deemphasis_us, rtlfm_integer)
    return lfilter([a], [1.0, -(1.0 - a)], f_hz)


def simulate_fm_channel(freq_track_hz: np.ndarray, cnr_db, fs_demod: int = 48000, deviation_hz: float = 3000.0,
                        doppler_hz=None, deemphasis_us: float = 75.0, seed: int = 0, carrier_on=None,
                        oversample: int = 8, droop_compensation: bool = True,
                        rtlfm_integer_deemph: bool = True) -> np.ndarray:
    """NBFM at complex baseband fs_demod: carrier A*exp(j*2*pi*cumsum(deviation*audio/ fs)) (audio = sin of the tone phase),
    plus complex Gaussian noise with CNR (scalar or per-sample array, dB, in fs_demod bandwidth), optional Doppler offset
    (scalar or array), rtl_fm-style discriminator (fs/2pi * angle(z[n]*conj z[n-1])), optional de-emphasis; returns Hz.

    Channel model (take2/limiter/synth_lib.py, rtl_fm-faithful): the carrier and complex white Gaussian noise are
    generated at fs_demod*oversample; the signal is then decimated by `oversample` with log2(oversample) passes of
    rtl_fm's binomial half-band filter [1,5,10,10,5,1] and (if droop_compensation) rtl_fm's 9-tap cic_9 droop FIR, so
    that the discriminator sees rtl_fm's coloured, droop-shaped noise. CNR is defined as carrier power over the noise
    power of an ideal fs_demod-wide (brick-wall) band: A^2 = 10^(cnr/10) * 2/oversample with unit-variance noise per
    quadrature at the oversampled rate (identical to synth_lib for oversample=32). oversample must be a power of two;
    oversample=1 is the simple model of scripts/13_fm_threshold_sim.py (white noise straight at fs_demod, no FIR).
    The default oversample=8 (3 passes) matches the 5-pass chain to within a few hundredths of a dB (the two
    highest-rate passes have negligible in-band droop and alias leakage) at a quarter of the cost; cic_9_tables[5] is
    used for any pass count (the tables differ by < 1 %).
    De-emphasis uses rtl_fm's integer-rounded coefficient (alpha = 1/4 at 48 kHz / 75 us) unless rtlfm_integer_deemph
    is False. The carrier is off (A = 0) where carrier_on is False (default: where freq_track_hz == 0).
    Deterministic for a given seed. Delay of the output relative to the frequency track: about 0.08 ms (half-sample
    discriminator, de-emphasis group delay 3 samples, half-band filter half-sample shifts)."""
    f = np.asarray(freq_track_hz, np.float64)
    N = f.size
    fs = float(fs_demod)
    UP = int(oversample)
    if UP < 1 or (UP & (UP - 1)) != 0:
        raise ValueError("oversample must be a power of two (1, 2, 4, 8, 16, 32, ...)")
    fsc = fs * UP
    on = _carrier_on(f, carrier_on)
    inst = (float(deviation_hz) * _tone_audio(f, fs))
    if doppler_hz is not None:
        inst = inst + np.asarray(doppler_hz, np.float64)
    cnr = np.broadcast_to(np.asarray(cnr_db, np.float64), (N,))
    amp = np.sqrt(10.0 ** (cnr / 10.0) * 2.0 / UP) * on                  # per-sample carrier amplitude at fsc

    rng = np.random.default_rng(seed)
    n_c = N * UP
    z = np.empty(n_c, np.complex64)
    z.real = rng.standard_normal(n_c, dtype=np.float32)
    z.imag = rng.standard_normal(n_c, dtype=np.float32)
    if np.any(on):
        acc = 0.0                                                       # phase in cycles (float64 carry)
        step = max(UP, (_CARRIER_CHUNK // UP) * UP)
        for a0 in range(0, n_c, step):
            a1 = min(n_c, a0 + step)
            i0, i1 = a0 // UP, (a1 + UP - 1) // UP
            inst_c = np.repeat(inst[i0:i1], UP)[: a1 - a0]
            amp_c = np.repeat(amp[i0:i1], UP)[: a1 - a0]
            ph = acc + np.cumsum(inst_c / fsc)
            acc = float(ph[-1])
            ph = np.mod(ph, 1.0).astype(np.float32)
            z[a0:a1] += (amp_c.astype(np.float32) * np.exp((2j * np.pi) * ph)).astype(np.complex64)
    passes = int(round(np.log2(UP)))
    for _ in range(passes):
        z = np.convolve(z, _HALFBAND, mode="same")[::2]
    if passes and droop_compensation:
        z = np.convolve(z, _DROOP_COMP, mode="same")
    z = np.ascontiguousarray(z[:N])
    out = _discriminate(z, fs)
    out = _deemphasise(out, fs, deemphasis_us, rtlfm_integer_deemph)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------------------------- IQ writer
def write_iq_u8(freq_track_hz: np.ndarray, fs_audio: int, fs_iq: float, f_offset_hz, cnr_db: float, path: str,
                deviation_hz: float = 3000.0, seed: int = 0, amplitude: float = 0.3, chunk_s: float = 1.0) -> int:
    """Upsample the NBFM baseband to fs_iq, shift by f_offset_hz (scalar or callable t->Hz for Doppler), add noise at
    cnr_db referenced to a 48 kHz bandwidth, quantise to unsigned 8-bit interleaved I/Q and write to path.

    The instantaneous carrier frequency deviation*audio (audio = sin of the tone phase at fs_audio) is linearly
    interpolated to fs_iq and integrated at fs_iq, so no band-limited resampling of the complex carrier is needed;
    the carrier is off where freq_track_hz == 0. Noise: unit variance per quadrature at fs_iq, so the carrier
    amplitude is A = sqrt(10^(cnr/10) * 2 * 48000/fs_iq). The per-quadrature RMS of the written signal is
    `amplitude` of full scale (127.5 counts), peaks beyond full scale are clipped (realistic RTL-SDR behaviour).
    Written in chunks of chunk_s seconds (never holds the whole file); deterministic for a given seed.
    Returns the number of complex samples written."""
    f = np.asarray(freq_track_hz, np.float64)
    N = f.size
    fs_a = float(fs_audio); fs_iq = float(fs_iq)
    t_audio = np.arange(N) / fs_a
    inst_audio = float(deviation_hz) * _tone_audio(f, fs_a)
    on_audio = (f != 0).astype(np.float64)
    n_iq = int(round(N / fs_a * fs_iq))
    A = float(np.sqrt(10.0 ** (float(cnr_db) / 10.0) * 2.0 * 48000.0 / fs_iq))
    scale = float(amplitude) / np.sqrt(1.0 + A * A / 2.0)
    rng = np.random.default_rng(seed)
    chunk = max(1, int(round(chunk_s * fs_iq)))
    acc = 0.0
    written = 0
    with open(path, "wb") as fh:
        for a0 in range(0, n_iq, chunk):
            a1 = min(n_iq, a0 + chunk)
            t = np.arange(a0, a1) / fs_iq
            inst = np.interp(t, t_audio, inst_audio)
            on = np.interp(t, t_audio, on_audio) > 0.5
            off = f_offset_hz(t) if callable(f_offset_hz) else float(f_offset_hz)
            ph = acc + np.cumsum((inst + off) / fs_iq)
            acc = float(ph[-1])
            carrier = np.exp((2j * np.pi) * np.mod(ph, 1.0)) * (A * on)
            zi = carrier.real + rng.standard_normal(a1 - a0)
            zq = carrier.imag + rng.standard_normal(a1 - a0)
            out = np.empty(2 * (a1 - a0), np.uint8)
            out[0::2] = np.clip(np.round(127.5 + 127.5 * scale * zi), 0, 255).astype(np.uint8)
            out[1::2] = np.clip(np.round(127.5 + 127.5 * scale * zq), 0, 255).astype(np.uint8)
            fh.write(out.tobytes())
            written += a1 - a0
    return written
