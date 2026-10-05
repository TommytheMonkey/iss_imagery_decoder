"""IQ front end: channel selection, carrier/Doppler tracking, FM demodulation, resampling to 48 kHz. Chunked, read-only.

Design (all stages keep state across chunks, so the result is identical to processing the whole file at once):

  u8 IQ (memmap) -> complex64 -> block-mean DC removal -> fixed mix by -(signal_freq - center_freq)
     -> polyphase FIR decimation by D1 to f_int >= 96 kS/s                                   [raw-rate work]
     pass 1: Welch power spectra per block (params.iq_block_s) -> carrier search (channel-matched box filter peak,
             local noise floor from the flanks, noise-subtracted power centroid) -> alpha-beta tracker with slew limit,
             gating, hold through unlocked blocks and re-acquisition -> local-linear smoother -> CarrierTrack
     pass 2: fine mix by the smoothed track (phase-continuous) -> rational polyphase resampling f_int -> 48 kHz
             -> channel low-pass (iq_channel_bw_hz total, FFT overlap-save) -> polar discriminator (Hz)
             -> optional rtl_fm-style one-pole de-emphasis -> group-delay compensated float32 Hz at 48 kHz.

The raw file is read twice (or once with the intermediate stream cached when it is small enough); it is never loaded
whole and no stage is O(N^2). Nothing is written anywhere.
"""
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.signal import firwin, kaiserord, lfilter
from scipy.signal.windows import hann
from scipy.ndimage import uniform_filter1d

from .io import AudioSignal, IQSpec, iq_num_samples, iter_iq_u8, rational_ratio
from .params import Params


@dataclass
class CarrierTrack:
    t: np.ndarray            # block centre times (s)
    f_offset: np.ndarray     # tracked carrier offset from (signal_freq - center_freq) baseband position, Hz
    snr_db: np.ndarray       # block carrier-to-noise estimate (dB) in the channel bandwidth
    locked: np.ndarray       # bool per block

    def to_dict(self) -> dict:
        return {"t_s": self.t.tolist(), "f_offset_hz": self.f_offset.tolist(),
                "snr_db": [None if not np.isfinite(v) else float(v) for v in self.snr_db],
                "locked": self.locked.tolist()}


# ----------------------------------------------------------------------------------------------------------------------
# constants (not tunables): rates, filter specs, detector thresholds that are tied to the block statistics
_INTERMEDIATE_FS_MIN = 96000.0        # intermediate rate after the first decimation (>= 2 x (search + bw/2 + margin))
_STOPBAND_DB = 60.0                   # Kaiser design attenuation for all FIR stages
_SEARCH_MARGIN_HZ = 2000.0            # extra passband beyond search +- bw/2 in the first stage
_STAGE2_MARGIN_HZ = 500.0             # stage-2 passband beyond bw/2 so the channel edge is set by the channel filter
_CHANNEL_TRANSITION_HZ = 1500.0       # channel low-pass transition width at 48 kHz
_SPECTRUM_BIN_HZ = 25.0               # target bin width of the carrier-search spectra
_MIN_PEAK_Z = 8.0                     # z-score of the channel-box peak over the local noise floor to count a detection
_TRACK_ALPHA = 0.35                   # alpha-beta tracker gains (critically damped: beta = alpha^2 / (2 - alpha))
_TRACK_BETA = _TRACK_ALPHA ** 2 / (2.0 - _TRACK_ALPHA)
_GATE_BASE_HZ = 400.0                 # measurement gate when locked; grows by max_slew*dt per missed block
_REACQ_AGREE_HZ = 300.0               # two consecutive off-track detections closer than this re-initialise the track
_DEFAULT_LOCK_SNR_DB = -6.0           # fallback if Params has no iq_lock_snr_db
_DEFAULT_TRACK_SMOOTH_S = 5.0         # fallback if Params has no iq_track_smooth_s (half width of the smoother)
_NARROWBAND_FRAC = 0.7                # a detection whose in-band excess power is this concentrated in 5 bins (~125 Hz) is a CW, not FM
_CACHE_AUTO_LIMIT_BYTES = 768 * 1024 * 1024


# ----------------------------------------------------------------------------------------------------------------------
# filter design helpers

def _design_lowpass(fs: float, f_pass: float, f_stop: float, atten_db: float = _STOPBAND_DB) -> np.ndarray:
    """Odd-length Kaiser-windowed FIR low-pass (symmetric -> integer group delay (L-1)/2)."""
    if not (0 < f_pass < f_stop < fs / 2):
        raise ValueError(f"bad low-pass spec: pass {f_pass} stop {f_stop} at fs {fs}")
    width = (f_stop - f_pass) / (fs / 2.0)
    numtaps, beta = kaiserord(atten_db, width)
    numtaps = int(numtaps)
    if numtaps % 2 == 0:
        numtaps += 1
    numtaps = max(numtaps, 3)
    return firwin(numtaps, (f_pass + f_stop) / 2.0, window=("kaiser", beta), fs=fs).astype(np.float64)


# ----------------------------------------------------------------------------------------------------------------------
# stateful streaming DSP blocks

class _Decimator:
    """Stateful polyphase FIR decimator for complex64 data with real taps, integer factor D.
    y[m] = sum_k h[k] x[(m+1)*D - 1 - k]  (x indexed from the start of the stream, zeros before it), i.e. output m
    carries the input at time m*D delayed by ((L-1)/2 - (D-1)) input samples. Each polyphase row is one contiguous
    (M, D) x (D,) product evaluated with einsum (fixed summation order, so the output is bit-identical for any
    chunking of the input; BLAS GEMV is ~40 % faster but its rounding depends on M); ~L/D complex MACs per *input*
    sample."""

    def __init__(self, h: np.ndarray, D: int):
        L = len(h)
        self.D = int(D)
        self.Q = int(math.ceil(L / self.D))
        hp = np.zeros(self.Q * self.D, dtype=np.float64)
        hp[:L] = h
        # H[q, p] = h[q*D + (D-1-p)]
        self.H = np.ascontiguousarray(hp.reshape(self.Q, self.D)[:, ::-1]).astype(np.complex64)
        self.delay_in_samples = (L - 1) / 2.0 - (self.D - 1)
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.Q - 1, self.D), dtype=np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        if n % self.D:
            raise ValueError("chunk length must be a multiple of the decimation factor")
        M = n // self.D
        X = np.concatenate([self.hist, x.reshape(M, self.D)], axis=0)       # (M + Q - 1, D)
        y = np.zeros(M, dtype=np.complex64)
        Q = self.Q
        for q in range(Q):
            y += np.einsum("md,d->m", X[Q - 1 - q:Q - 1 - q + M], self.H[q])
        if Q > 1:
            self.hist = X[-(Q - 1):].copy()
        return y


class _Resampler:
    """Stateful polyphase rational resampler (up U, down D) for complex64 data; taps designed at rate f_in*U.
    y[k] = sum_j h[k*D - j*U] x[j]: output k sits at upsampled index k*D, delayed by (L-1)/2 upsampled samples.
    Implemented as a gather of Q = ceil(L/U) input samples per output; outputs are emitted as soon as all the
    input they depend on has arrived, so chunk boundaries do not matter."""

    def __init__(self, h: np.ndarray, U: int, D: int):
        L = len(h)
        self.U, self.D = int(U), int(D)
        self.Q = int(math.ceil(L / self.U))
        hp = np.zeros(self.Q * self.U, dtype=np.float64)
        hp[:L] = h * self.U                                   # interpolation gain
        self.Hp = np.ascontiguousarray(hp.reshape(self.Q, self.U).T).astype(np.float32)   # Hp[p, q] = h[p + q*U]
        self.qidx = np.arange(self.Q, dtype=np.int64)
        self.delay_up_samples = (L - 1) / 2.0
        self.reset()

    def reset(self):
        self.hist = np.zeros(self.Q - 1, dtype=np.complex64)
        self.j0 = 0          # global index of the next input sample
        self.k_next = 0      # global index of the next output sample

    def process(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        J0, J1 = self.j0, self.j0 + n
        U, D, Q = self.U, self.D, self.Q
        k_hi = -((-J1 * U) // D)                              # ceil(J1*U/D): first output needing input >= J1
        ks = np.arange(self.k_next, k_hi, dtype=np.int64)
        xe = np.concatenate([self.hist, x]) if Q > 1 else x
        base = J0 - (Q - 1)                                   # global index of xe[0]
        if ks.size:
            kd = ks * D
            jtop = kd // U
            p = kd - jtop * U
            lt = jtop - base
            idx = lt[:, None] - self.qidx[None, :]            # (K, Q), all inside xe by construction
            y = (xe[idx] * self.Hp[p]).sum(axis=1).astype(np.complex64)
        else:
            y = np.zeros(0, dtype=np.complex64)
        if Q > 1:
            self.hist = xe[-(Q - 1):].copy()
        self.j0 = J1
        self.k_next = k_hi
        return y


class _FIRStream:
    """Stateful causal FIR (real taps, complex64 data) by fixed-block FFT overlap-save. The block grid is independent
    of how the input is chunked, so the output is bit-identical for any chunking; outputs are emitted once a block is
    complete (the caller flushes with zeros at the end). Exact steady-state causal FIR output."""

    def __init__(self, h: np.ndarray, n_fft: int = 8192):
        self.L = len(h)
        while n_fft < 4 * self.L:
            n_fft *= 2
        self.n_fft = int(n_fft)
        self.step = self.n_fft - (self.L - 1)
        self.H = np.fft.fft(h.astype(np.float64), self.n_fft)
        self.delay_samples = (self.L - 1) / 2.0
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.L - 1, dtype=np.complex64)     # (L-1) history followed by not yet processed input

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0]:
            self.buf = np.concatenate([self.buf, x])
        n_blocks = (self.buf.shape[0] - (self.L - 1)) // self.step
        if n_blocks <= 0:
            return np.zeros(0, dtype=np.complex64)
        blocks = np.lib.stride_tricks.sliding_window_view(self.buf, self.n_fft)[::self.step][:n_blocks]
        Y = np.fft.ifft(np.fft.fft(blocks.astype(np.complex128), axis=1) * self.H[None, :], axis=1)
        y = Y[:, self.L - 1:].reshape(-1).astype(np.complex64)
        self.buf = self.buf[n_blocks * self.step:].copy()
        return y


class _Discriminator:
    """Polar discriminator f[n] = fs/(2 pi) * angle(z[n] conj z[n-1]) with the previous sample carried over."""

    def __init__(self, fs: float):
        self.scale = fs / (2.0 * np.pi)
        self.prev = None

    def process(self, z: np.ndarray) -> np.ndarray:
        if z.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        prev = z[:1] if self.prev is None else self.prev
        zz = np.concatenate([prev, z])
        d = zz[1:] * np.conj(zz[:-1])
        self.prev = z[-1:].copy()
        return (np.angle(d) * self.scale).astype(np.float32)


class _Deemphasis:
    """rtl_fm-style one-pole de-emphasis y[n] = y[n-1] + a (x[n] - y[n-1]), a = 1 - exp(-1/(fs tau)).
    (rtl_fm itself rounds 1/a to an integer, 4 at 48 kHz / 75 us, i.e. a = 0.25 instead of 0.2425; the
    difference is a 3 % gain change, irrelevant for frequency estimation.)"""

    def __init__(self, fs: float, tau_s: float):
        self.alpha = 1.0 - math.exp(-1.0 / (fs * tau_s))
        self.b = np.array([self.alpha])
        self.a = np.array([1.0, -(1.0 - self.alpha)])
        self.zi = np.zeros(1)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 0:
            return x
        y, self.zi = lfilter(self.b, self.a, x.astype(np.float64), zi=self.zi)
        return y.astype(np.float32)


_LO_K = 2048


def _lo_fixed(frac_cycles_per_sample: float, n0: int, N: int) -> np.ndarray:
    """exp(-2j pi f/fs n) for n = n0..n0+N-1 as complex64. Built on an absolute grid n = r*K + k as the outer product
    b[r] a[k] (O(N/K + K) transcendental evaluations, phase reduced modulo one cycle in float64), so the value for a
    given absolute sample index does not depend on the chunking."""
    K = _LO_K
    frac = float(frac_cycles_per_sample)
    r0 = n0 // K
    r1 = (n0 + N - 1) // K
    k = np.arange(K, dtype=np.float64)
    r = np.arange(r0, r1 + 1, dtype=np.float64)
    a = np.exp(-2j * np.pi * ((frac * k) % 1.0)).astype(np.complex64)
    b = np.exp(-2j * np.pi * ((frac * K * r) % 1.0)).astype(np.complex64)
    s = n0 - r0 * K
    return (b[:, None] * a[None, :]).ravel()[s:s + N]


_DC_BLOCK = 1 << 20


class _FrontEnd:
    """Raw-rate stages shared by both passes: block-mean DC removal, fixed coarse mix, decimation to f_int.
    The DC is estimated per block of _DC_BLOCK raw samples on a grid anchored at the file start (independent of how
    the caller chunks the input); a block is processed once it is complete, so the output lags the input by up to one
    block and `finish()` processes the trailing partial block. The caller streams contiguous chunks."""

    def __init__(self, fs_iq: float, f_coarse_hz: float, h1, D1: int):
        self.fs_iq = fs_iq
        self.frac = f_coarse_hz / fs_iq
        self.D1 = int(D1)
        self.n_dc = max(self.D1, (_DC_BLOCK // self.D1) * self.D1)
        self.dec = _Decimator(h1, D1) if (D1 > 1 and h1 is not None) else None
        self.dc_sum = 0.0 + 0.0j
        self.dc_n = 0
        self.dc_max_abs = 0.0
        self.reset()

    def reset(self):
        if self.dec is not None:
            self.dec.reset()
        self.pending = np.zeros(0, dtype=np.complex64)
        self.n_next = 0                  # absolute raw index of pending[0]

    def _block(self, blk: np.ndarray, n0: int, measure_dc: bool) -> np.ndarray:
        dc = complex(blk.mean()) if blk.shape[0] else 0j
        if measure_dc and blk.shape[0]:
            self.dc_sum += dc * blk.shape[0]
            self.dc_n += blk.shape[0]
            self.dc_max_abs = max(self.dc_max_abs, abs(dc))
        z = blk - np.complex64(dc)
        pad = (-z.shape[0]) % self.D1
        if pad:
            z = np.concatenate([z, np.zeros(pad, dtype=np.complex64)])
        if self.frac != 0.0:
            z *= _lo_fixed(self.frac, n0, z.shape[0])
        return self.dec.process(z) if self.dec is not None else z

    def process(self, z: np.ndarray, measure_dc: bool = True) -> np.ndarray:
        buf = np.concatenate([self.pending, z]) if self.pending.shape[0] else z
        n_full = buf.shape[0] // self.n_dc
        outs = []
        for i in range(n_full):
            outs.append(self._block(buf[i * self.n_dc:(i + 1) * self.n_dc], self.n_next + i * self.n_dc, measure_dc))
        self.pending = buf[n_full * self.n_dc:].copy()
        self.n_next += n_full * self.n_dc
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.complex64)

    def finish(self, measure_dc: bool = True) -> np.ndarray:
        """Process the trailing partial DC block (mean over its real samples only; zero-padded to a multiple of D1)."""
        if self.pending.shape[0] == 0:
            return np.zeros(0, dtype=np.complex64)
        n = self.pending.shape[0]
        y = self._block(self.pending, self.n_next, measure_dc)
        self.n_next += n + ((-n) % self.D1)
        self.pending = np.zeros(0, dtype=np.complex64)
        return y


# ----------------------------------------------------------------------------------------------------------------------
# carrier search and tracking

class _BlockSpectra:
    """Welch-averaged power spectra (Hann, 50 % overlap) of the intermediate stream, accumulated per block."""

    def __init__(self, f_int: float, n_fft: int, block_s: float, n_blocks: int):
        self.f_int, self.n_fft, self.hop = float(f_int), int(n_fft), int(n_fft) // 2
        self.win = hann(self.n_fft, sym=False).astype(np.float32)
        self.win_power = float(np.sum(self.win ** 2))
        self.block_len = float(block_s) * float(f_int)
        self.n_blocks = int(n_blocks)
        self.sum = np.zeros((self.n_blocks, self.n_fft), dtype=np.float64)
        self.count = np.zeros(self.n_blocks, dtype=np.int64)
        self.carry = np.zeros(0, dtype=np.complex64)
        self.carry_m0 = 0

    def add(self, y: np.ndarray, m0: int):
        if self.carry.shape[0]:
            buf = np.concatenate([self.carry, y])
            b0 = self.carry_m0
        else:
            buf, b0 = y, m0
        n = buf.shape[0]
        if n < self.n_fft:
            self.carry, self.carry_m0 = buf.copy(), b0
            return
        n_frames = (n - self.n_fft) // self.hop + 1
        frames = np.lib.stride_tricks.sliding_window_view(buf, self.n_fft)[::self.hop][:n_frames]
        F = np.fft.fft(frames * self.win, axis=1)
        P = (F.real.astype(np.float64) ** 2 + F.imag.astype(np.float64) ** 2) / self.win_power
        P = np.fft.fftshift(P, axes=1)
        centres = b0 + self.hop * np.arange(n_frames, dtype=np.float64) + self.n_fft / 2.0
        ids = np.clip((centres / self.block_len).astype(np.int64), 0, self.n_blocks - 1)
        for b in np.unique(ids):
            sel = ids == b
            self.sum[b] += P[sel].sum(axis=0)
            self.count[b] += int(sel.sum())
        consumed = n_frames * self.hop
        self.carry, self.carry_m0 = buf[consumed:].copy(), b0 + consumed

    def spectra(self) -> np.ndarray:
        cnt = np.maximum(self.count, 1)[:, None].astype(np.float64)
        return self.sum / cnt

    def freqs(self) -> np.ndarray:
        return np.fft.fftshift(np.fft.fftfreq(self.n_fft, d=1.0 / self.f_int))


_MEASURE_BATCH = 256


def _measure_blocks(P: np.ndarray, freqs: np.ndarray, counts: np.ndarray, bw_hz: float, search_hz: float,
                    pass_hz: float, lock_snr_db: float, min_z: float = _MIN_PEAK_Z) -> dict:
    """`_measure_blocks_impl` applied in batches of _MEASURE_BATCH blocks (all operations are row-wise, so the result
    is identical; this bounds the (n_blocks x n_fft) temporaries for long recordings)."""
    B = P.shape[0]
    if B <= _MEASURE_BATCH:
        return _measure_blocks_impl(P, freqs, counts, bw_hz, search_hz, pass_hz, lock_snr_db, min_z)
    parts = [_measure_blocks_impl(P[i:i + _MEASURE_BATCH], freqs, counts[i:i + _MEASURE_BATCH], bw_hz, search_hz,
                                  pass_hz, lock_snr_db, min_z) for i in range(0, B, _MEASURE_BATCH)]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "n_box"}
    out["n_box"] = parts[0]["n_box"]
    return out


def _measure_blocks_impl(P: np.ndarray, freqs: np.ndarray, counts: np.ndarray, bw_hz: float, search_hz: float,
                         pass_hz: float, lock_snr_db: float, min_z: float = _MIN_PEAK_Z) -> dict:
    """Per-block carrier measurement from averaged power spectra.
    - channel-matched detector: moving average of P over the channel bandwidth; its peak within +-search_hz;
    - local noise floor: mean of the medians of the two flanks (bw/2 .. 1.25 bw from the peak), which cancels a
      linear tilt of the receiver noise floor;
    - SNR in the channel bandwidth and a z-score of the peak against the chi-square scatter of the box average;
    - carrier estimate: power centroid of (P - floor)+ over the channel band, iterated so the band is centred
      (FM spectra are symmetric about the carrier, so the centroid is the carrier even when a sideband is the
      strongest line); the centroid is confined to the declared +-search_hz;
    - narrowband rejection: a detection whose in-band excess power is concentrated (> _NARROWBAND_FRAC) in the 5 bins
      around the spectral maximum is a CW carrier, not a +-3 kHz FM signal; its +-bw/2 neighbourhood is masked and the
      search is repeated once, so a strong unmodulated neighbour cannot capture the tracker."""
    B, nf = P.shape
    df = float(freqs[1] - freqs[0])
    half = max(1, int(round(bw_hz / 2.0 / df)))
    n_box = 2 * half + 1
    kk = np.arange(nf)
    valid = np.abs(freqs) <= pass_hz
    search = valid & (np.abs(freqs) <= search_hz)
    if not search.any():
        search = valid
    S = uniform_filter1d(P, n_box, axis=1, mode="nearest")
    allowed = np.broadcast_to(search[None, :], (B, nf)).copy()
    narrow_any = np.zeros(B, dtype=bool)
    f_cw = np.full(B, np.nan)
    rows = np.arange(B)
    n_eff = np.maximum(counts.astype(np.float64) * 0.9, 1.0)
    for attempt in range(2):
        S_search = np.where(allowed, S, -np.inf)
        k_peak = np.argmax(S_search, axis=1)
        dist = np.abs(kk[None, :] - k_peak[:, None])
        band = dist <= half
        flank_lo = int(round(bw_hz / 2.0 / df))
        flank_hi = int(round(1.25 * bw_hz / df))
        left = valid[None, :] & (kk[None, :] < k_peak[:, None]) & (dist > flank_lo) & (dist <= flank_hi)
        right = valid[None, :] & (kk[None, :] > k_peak[:, None]) & (dist > flank_lo) & (dist <= flank_hi)
        with np.errstate(all="ignore"):
            fl = np.nanmedian(np.where(left, P, np.nan), axis=1)
            fr = np.nanmedian(np.where(right, P, np.nan), axis=1)
            glob = np.nanmedian(np.where(valid[None, :] & ~band, P, np.nan), axis=1)
        both = np.isfinite(fl) & np.isfinite(fr)
        floor = np.where(both, 0.5 * (fl + fr), np.where(np.isfinite(fl), fl, np.where(np.isfinite(fr), fr, glob)))
        floor = np.where(np.isfinite(floor) & (floor > 0), floor, np.nan)
        with np.errstate(all="ignore"):
            sigma_S = floor / np.sqrt(n_eff * n_box)
            S_pk = S[rows, k_peak]
            z = (S_pk - floor) / sigma_S
            excess = np.maximum(P - floor[:, None], 0.0)
            sig = (excess * band).sum(axis=1)
            snr_lin = sig / (floor * n_box)
            snr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-6))
            # narrowband test: share of the in-band excess within +-2 bins of the in-band spectral maximum
            k_max = np.argmax(np.where(band, excess, -1.0), axis=1)
            near = np.abs(kk[None, :] - k_max[:, None]) <= 2
            frac_nb = (excess * (band & near)).sum(axis=1) / np.maximum(sig, 1e-30)
            # centroid refinement (confined to the declared search range)
            f_c = freqs[k_peak].astype(np.float64)
            for _ in range(3):
                w = excess * ((np.abs(freqs[None, :] - f_c[:, None]) <= bw_hz / 2.0) & valid[None, :])
                ws = w.sum(axis=1)
                f_new = np.where(ws > 0, (w * freqs[None, :]).sum(axis=1) / np.where(ws > 0, ws, 1.0), f_c)
                f_c = np.clip(f_new, -search_hz, search_hz)
        ok_block = (counts >= 2) & np.isfinite(floor) & np.isfinite(z)
        strong = ok_block & (snr_db >= lock_snr_db) & (z >= min_z)
        narrow = strong & (frac_nb > _NARROWBAND_FRAC)
        if attempt == 0:
            if not narrow.any():
                break
            narrow_any = narrow.copy()
            f_cw[narrow] = freqs[k_max[narrow]]
            # every channel box that still contains the CW must be excluded, i.e. box centres within a full channel width
            allowed[narrow] &= ~(np.abs(kk[None, :] - k_max[narrow][:, None]) <= 2 * half)
            allowed[narrow & ~np.any(allowed, axis=1)] = search[None, :]       # nothing left to search: keep the original
        else:
            # a second narrowband detection in the same block is also rejected (not an FM signal)
            narrow_any |= narrow
    detected = ok_block & (snr_db >= lock_snr_db) & (z >= min_z) & ~narrow
    return {"f_meas": np.where(ok_block, f_c, np.nan), "snr_db": np.where(ok_block, snr_db, np.nan),
            "z": np.where(ok_block, z, np.nan), "detected": detected, "floor": floor, "k_peak": k_peak,
            "ok_block": ok_block, "n_box": n_box, "narrowband": narrow_any, "f_narrowband": f_cw,
            "frac_narrowband": np.where(ok_block, frac_nb, np.nan)}


def _alpha_beta_track(f_meas: np.ndarray, detected: np.ndarray, dt: float, max_slew: float,
                      alpha: float = _TRACK_ALPHA, beta: float = _TRACK_BETA) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal alpha-beta (g-h) tracker on (offset, slew). Gated updates on detected blocks; prediction held through
    undetected/outlier blocks with a gate that widens by max_slew*dt per miss; re-acquisition when two consecutive
    detections agree with each other but not with the prediction. Returns (f_pred, locked, inlier)."""
    B = len(f_meas)
    f_pred = np.full(B, np.nan)
    locked = np.zeros(B, dtype=bool)
    inlier = np.zeros(B, dtype=bool)
    f = None
    fdot = 0.0
    misses = 0
    pending = None          # (index, value) of a detection that did not fit the track
    for i in range(B):
        det = bool(detected[i]) and np.isfinite(f_meas[i])
        fm = float(f_meas[i]) if det else None
        if f is None:
            if det:
                f, fdot, misses = fm, 0.0, 0
                locked[i] = inlier[i] = True
            f_pred[i] = f if f is not None else np.nan
            continue
        fp = f + fdot * dt
        gate = _GATE_BASE_HZ + max_slew * dt * (misses + 1)
        if det and abs(fm - fp) <= gate:
            r = fm - fp
            f = fp + alpha * r
            fdot = float(np.clip(fdot + beta * r / dt, -max_slew, max_slew))
            misses = 0
            locked[i] = inlier[i] = True
            pending = None
        else:
            f = fp
            misses += 1
            if det:
                if pending is not None and pending[0] == i - 1 and abs(fm - pending[1]) <= _REACQ_AGREE_HZ:
                    f = fm
                    fdot = float(np.clip((fm - pending[1]) / dt, -max_slew, max_slew))
                    misses = 0
                    locked[i] = inlier[i] = True
                    inlier[i - 1] = locked[i - 1] = True
                    f_pred[i - 1] = pending[1]
                    pending = None
                else:
                    pending = (i, fm)
            else:
                pending = None
        f_pred[i] = f
    # back-fill the blocks before the first lock with the first value (hold)
    first = np.flatnonzero(np.isfinite(f_pred))
    if first.size:
        f_pred[:first[0]] = f_pred[first[0]]
    else:
        f_pred[:] = 0.0
    return f_pred, locked, inlier


def _smooth_track(t: np.ndarray, f_meas: np.ndarray, inlier: np.ndarray, snr_db: np.ndarray, half_width_s: float,
                  max_slew: float, dt: float) -> np.ndarray:
    """Two-sided smoother: SNR-weighted local linear regression (tricube kernel, half width half_width_s) through the
    inlier measurements, linear interpolation across gaps, constant hold beyond the ends, slew limited."""
    B = len(t)
    idx = np.flatnonzero(inlier & np.isfinite(f_meas))
    if idx.size == 0:
        return np.zeros(B)
    w_snr = np.clip(10.0 ** (np.nan_to_num(snr_db, nan=-30.0) / 10.0), 0.05, 50.0)
    out = np.full(B, np.nan)
    ti, fi, wi = t[idx], f_meas[idx], w_snr[idx]
    for b in range(B):
        sel = np.abs(ti - t[b]) <= half_width_s
        if not sel.any():
            continue
        u = (ti[sel] - t[b]) / max(half_width_s, 1e-9)
        w = wi[sel] * (1.0 - np.abs(u) ** 3) ** 3 + 1e-12
        x = ti[sel] - t[b]
        y = fi[sel]
        sw = w.sum()
        if sel.sum() >= 3 and (x.max() - x.min()) > dt:
            xm = (w * x).sum() / sw
            ym = (w * y).sum() / sw
            sxx = (w * (x - xm) ** 2).sum()
            slope = (w * (x - xm) * (y - ym)).sum() / sxx if sxx > 0 else 0.0
            slope = float(np.clip(slope, -max_slew, max_slew))
            out[b] = ym - slope * xm
        else:
            out[b] = (w * y).sum() / sw
    good = np.isfinite(out)
    if good.sum() == 0:
        return np.zeros(B)
    out = np.interp(t, t[good], out[good])                 # gaps: linear; ends: constant hold
    step = max_slew * dt
    for i in range(1, B):                                  # slew limit (forward)
        out[i] = min(max(out[i], out[i - 1] - step), out[i - 1] + step)
    for i in range(B - 2, -1, -1):                         # and backward, so the limit is symmetric
        out[i] = min(max(out[i], out[i + 1] - step), out[i + 1] + step)
    return out


# ----------------------------------------------------------------------------------------------------------------------

def _plan(spec: IQSpec, params: Params) -> dict:
    """Rates, filters and band limits for the chain."""
    fs_iq = float(spec.fs)
    if not (fs_iq > 0):
        raise ValueError("IQSpec.fs must be positive")
    fs_out = int(params.fs)
    bw = float(params.iq_channel_bw_hz)
    if not (1000.0 <= bw < fs_out - 2 * _STAGE2_MARGIN_HZ - 1000.0):
        raise ValueError(f"iq_channel_bw_hz={bw} must lie in [1000, {fs_out - 2 * _STAGE2_MARGIN_HZ - 1000})")
    f_coarse = float(spec.signal_freq) - float(spec.center_freq)
    D1 = max(1, int(fs_iq // _INTERMEDIATE_FS_MIN))
    f_int = fs_iq / D1
    search = float(params.iq_search_hz)
    # passband of the first stage must fit the raw band around the coarse offset and the intermediate Nyquist zone
    f_pass1 = search + bw / 2.0 + _SEARCH_MARGIN_HZ
    limit_raw = fs_iq / 2.0 - abs(f_coarse) - 0.02 * fs_iq
    limit_int = 0.4 * f_int
    f_pass1_eff = min(f_pass1, limit_raw, limit_int)
    if f_pass1_eff < bw / 2.0 + 1000.0:
        raise ValueError(f"signal_freq - center_freq = {f_coarse:.0f} Hz leaves no room for a {bw:.0f} Hz channel "
                         f"inside the +-{fs_iq / 2:.0f} Hz capture")
    search_eff = max(0.0, f_pass1_eff - bw / 2.0 - _SEARCH_MARGIN_HZ) if f_pass1_eff < f_pass1 else search
    if D1 > 1:
        h1 = _design_lowpass(fs_iq, f_pass1_eff, f_int - f_pass1_eff)
    else:
        h1 = None
    U, Dn = rational_ratio(fs_out * D1, fs_iq)              # U/Dn == fs_out / f_int
    f_up = f_int * U
    f_pass2 = bw / 2.0 + _STAGE2_MARGIN_HZ
    h2 = _design_lowpass(f_up, f_pass2, fs_out - f_pass2)
    h3 = _design_lowpass(float(fs_out), bw / 2.0, bw / 2.0 + _CHANNEL_TRANSITION_HZ)
    n_fft = int(2 ** round(math.log2(max(f_int / _SPECTRUM_BIN_HZ, 1.0))))
    n_fft = int(min(max(n_fft, 1024), 16384))
    return dict(fs_iq=fs_iq, fs_out=fs_out, bw=bw, f_coarse=f_coarse, D1=D1, f_int=f_int, U=U, Dn=Dn, f_up=f_up,
                h1=h1, h2=h2, h3=h3, f_pass1=f_pass1_eff, search=search_eff, search_requested=search, n_fft=n_fft)


def demodulate(path: str, spec: IQSpec, params: Params, *, chunk_samples: int = 1 << 20,
               cache_intermediate: bool | None = None, return_spectra: bool = False,
               progress=None) -> tuple[AudioSignal, dict]:
    """Full IQ chain, processed in chunks with overlap-save style continuity:
    1. remove DC (block mean) from the u8-derived complex samples;
    2. coarse channel: mix by -(signal_freq - center_freq), low-pass + decimate to an intermediate rate (>= 96 kS/s, integer factors);
    3. carrier search/tracking: per block of params.iq_block_s, power spectrum peak within +-params.iq_search_hz above the median
       noise floor; a tracker (alpha-beta / Kalman) with slew limit params.iq_max_slew_hz_s that holds the last prediction
       through no-carrier blocks and flags locked/unlocked; the predicted Doppler trend (approximately linear, up to ~ -180 Hz/s
       at TCA for 437 MHz) is used for prediction;
    4. fine mix by the smoothed track so the carrier sits at 0 Hz, channel low-pass of params.iq_channel_bw_hz total width,
       decimate to 48 kHz;
    5. polar discriminator: f[n] = fs/(2*pi) * angle(z[n]*conj(z[n-1])) in Hz; optional one-pole de-emphasis
       (params.iq_deemphasis_us; 0 disables) applied as in rtl_fm (a = 1-exp(-1/(fs*tau)));
    6. return AudioSignal(samples=f (float32 Hz), fs=48000, t0=0.0, source=path, meta={...}) and diag with the CarrierTrack,
       the decimation factors, the channel bandwidth and the mean/max carrier SNR.
    Must process a 1.024 MS/s, 11-minute file without loading it whole and without O(N^2) behaviour.

    Extra keyword arguments (all optional): chunk_samples = raw complex samples per chunk; cache_intermediate = keep the
    decimated intermediate stream of pass 1 in memory for pass 2 (None: automatic, when it is below ~768 MB; the output is
    bit-identical either way); return_spectra = include the per-block search spectra in diag; progress = callable
    (stage: str, fraction: float) for progress reporting. Output sample i is aligned to input time i / 48000 (all filter
    group delays compensated); the audio length is floor(n_iq * 48000 / fs_iq)."""
    fmt = str(spec.fmt).lower()
    if fmt not in ("u8", "cu8", "uint8"):
        raise ValueError(f"unsupported IQ format {spec.fmt!r}: only unsigned 8-bit interleaved I/Q is supported")
    t_start = time.perf_counter()
    plan = _plan(spec, params)
    fs_iq, fs_out, bw = plan["fs_iq"], plan["fs_out"], plan["bw"]
    D1, f_int, U, Dn = plan["D1"], plan["f_int"], plan["U"], plan["Dn"]
    n_iq = iq_num_samples(path)
    if n_iq < 2:
        raise ValueError(f"{path}: no IQ samples")
    duration_s = n_iq / fs_iq
    block_s = float(params.iq_block_s)
    max_slew = float(params.iq_max_slew_hz_s)
    lock_snr_db = float(getattr(params, "iq_lock_snr_db", _DEFAULT_LOCK_SNR_DB))
    smooth_s = float(getattr(params, "iq_track_smooth_s", _DEFAULT_TRACK_SMOOTH_S))
    chunk = max(D1, (int(chunk_samples) // D1) * D1)
    n_int_total = -((-n_iq) // D1)
    if cache_intermediate is None:
        cache_intermediate = n_int_total * 8 <= _CACHE_AUTO_LIMIT_BYTES
    fe = _FrontEnd(fs_iq, plan["f_coarse"], plan["h1"], D1)

    # ---------------- pass 1: intermediate stream -> block spectra -> carrier measurements ----------------
    n_blocks = max(1, int(math.ceil(duration_s / block_s)))
    spectra = _BlockSpectra(f_int, plan["n_fft"], block_s, n_blocks)
    cache = [] if cache_intermediate else None
    n0 = 0
    m0 = 0

    def take_pass1(y: np.ndarray):
        nonlocal m0
        if y.shape[0] == 0:
            return
        spectra.add(y, m0)
        if cache is not None:
            cache.append(y)
        m0 += y.shape[0]

    for z in iter_iq_u8(path, chunk_samples=chunk):
        take_pass1(fe.process(z))
        n0 += z.shape[0]
        if progress is not None:
            progress("search", min(1.0, n0 / n_iq))
    take_pass1(fe.finish())
    t_pass1 = time.perf_counter()
    P = spectra.spectra()
    freqs = spectra.freqs()
    meas = _measure_blocks(P, freqs, spectra.count, bw, plan["search"], plan["f_pass1"], lock_snr_db)
    t_blocks = np.minimum((np.arange(n_blocks) + 0.5) * block_s, duration_s)
    f_pred, locked_causal, inlier = _alpha_beta_track(meas["f_meas"], meas["detected"], block_s, max_slew)
    f_offset = _smooth_track(t_blocks, meas["f_meas"], inlier, meas["snr_db"], smooth_s, max_slew, block_s)
    track = CarrierTrack(t=t_blocks.astype(np.float64), f_offset=f_offset.astype(np.float64),
                         snr_db=meas["snr_db"].astype(np.float64), locked=inlier.astype(bool))

    # ---------------- pass 2: fine mix, resample, channel filter, discriminate ----------------
    if cache is None:
        fe.reset()          # the raw file is re-read from the start; with the cache the front end keeps its
                            # end-of-file state so the final flush continues the stream correctly
    res = _Resampler(plan["h2"], U, Dn)
    chan = _FIRStream(plan["h3"])
    disc = _Discriminator(fs_out)
    deemph_us = float(params.iq_deemphasis_us)
    deemph = _Deemphasis(fs_out, deemph_us * 1e-6) if deemph_us > 0 else None
    tau1 = fe.dec.delay_in_samples / fs_iq if fe.dec is not None else 0.0
    tau2 = res.delay_up_samples / plan["f_up"]
    tau3 = chan.delay_samples / fs_out
    tau4 = 0.5 / fs_out
    group_delay_s = tau1 + tau2 + tau3 + tau4
    n_delay = int(round(group_delay_s * fs_out))
    n_out = int((n_iq * fs_out) // int(round(fs_iq))) if abs(fs_iq - round(fs_iq)) < 1e-6 else int(n_iq * fs_out / fs_iq)
    out = np.zeros(n_out, dtype=np.float32)
    produced = 0        # demodulated samples produced so far (before delay trimming)
    phase_cycles = 0.0  # fine-mix phase carry (cycles)

    def emit(f_chunk: np.ndarray):
        nonlocal produced
        if f_chunk.shape[0] == 0:
            return
        a = produced
        b = produced + f_chunk.shape[0]
        produced = b
        lo = max(a, n_delay)
        hi = min(b, n_delay + n_out)
        if hi > lo:
            out[lo - n_delay:hi - n_delay] = f_chunk[lo - a:hi - a]

    def run_chain(y: np.ndarray, m_start: int):
        nonlocal phase_cycles
        M = y.shape[0]
        if M == 0:
            return
        tt = (m_start + np.arange(M, dtype=np.float64)) / f_int
        f_tr = np.interp(tt, t_blocks, f_offset)
        inc = np.cumsum(f_tr) / f_int
        ph = phase_cycles + np.concatenate([[0.0], inc[:-1]])
        phase_cycles = (phase_cycles + inc[-1]) % 1.0
        lo = np.exp(-2j * np.pi * (ph % 1.0)).astype(np.complex64)
        z48 = res.process(y * lo)
        z48 = chan.process(z48)
        f = disc.process(z48)
        if deemph is not None:
            f = deemph.process(f)
        emit(f)

    m_start = 0
    if cache is not None:
        n_chunks = len(cache)
        for i in range(n_chunks):
            y = cache[i]
            cache[i] = None
            run_chain(y, m_start)
            m_start += y.shape[0]
            if progress is not None:
                progress("demodulate", min(1.0, (i + 1) / max(n_chunks, 1)))
        del cache
    else:
        n0 = 0
        for z in iter_iq_u8(path, chunk_samples=chunk):
            y = fe.process(z, measure_dc=False)
            run_chain(y, m_start)
            n0 += z.shape[0]
            m_start += y.shape[0]
            if progress is not None:
                progress("demodulate", min(1.0, n0 / n_iq))
        y = fe.finish(measure_dc=False)
        run_chain(y, m_start)
        m_start += y.shape[0]
    # flush: push zeros through the chain until the delayed output covers n_out samples (the channel filter emits
    # whole blocks, so the flush also covers one FFT block)
    n_flush_raw = int(math.ceil((group_delay_s + (chan.n_fft + chan.L) / fs_out + 0.05) * fs_iq))
    n_flush_raw += (-n_flush_raw) % D1
    y = fe.process(np.zeros(n_flush_raw, dtype=np.complex64), measure_dc=False)
    run_chain(y, m_start)
    m_start += y.shape[0]
    y = fe.finish(measure_dc=False)
    run_chain(y, m_start)
    t_end = time.perf_counter()

    # ---------------- diagnostics ----------------
    lk = track.locked
    snr_locked = track.snr_db[lk] if lk.any() else np.zeros(0)
    cnr_offset_db = 10.0 * math.log10(fs_out / bw)
    dc_mean = fe.dc_sum / max(fe.dc_n, 1)
    elapsed = t_end - t_start
    meta = {
        "input": "iq_u8",
        "fs_iq": fs_iq,
        "center_freq_hz": float(spec.center_freq),
        "signal_freq_hz": float(spec.signal_freq),
        "f_coarse_hz": plan["f_coarse"],
        "n_iq_samples": int(n_iq),
        "duration_s": float(duration_s),
        "intermediate_fs": float(f_int),
        "decimation": {"D1": int(D1), "up": int(U), "down": int(Dn)},
        "channel_bw_hz": bw,
        "search_hz": float(plan["search"]),
        "search_hz_requested": float(plan["search_requested"]),
        "block_s": block_s,
        "n_fft": int(plan["n_fft"]),
        "deemphasis_us": deemph_us,
        "deemph_alpha": float(deemph.alpha) if deemph is not None else 0.0,
        "filter_taps": {"stage1": int(len(plan["h1"])) if plan["h1"] is not None else 0, "stage2": int(len(plan["h2"])),
                        "channel": int(len(plan["h3"]))},
        "group_delay_s": float(group_delay_s),
        "n_delay_samples": int(n_delay),
        "n_samples": int(n_out),
        "dc_mean_re": float(dc_mean.real), "dc_mean_im": float(dc_mean.imag), "dc_max_abs": float(fe.dc_max_abs),
        "n_blocks": int(n_blocks),
        "n_blocks_locked": int(lk.sum()),
        "n_blocks_detected": int(np.sum(meas["detected"])),
        "n_blocks_narrowband_rejected": int(np.sum(meas.get("narrowband", np.zeros(0, bool)))),
        "narrowband_hz_median": (float(np.nanmedian(meas["f_narrowband"])) if np.isfinite(meas.get("f_narrowband", np.full(1, np.nan))).any() else None),
        "frac_locked": float(lk.mean()) if n_blocks else 0.0,
        "first_lock_s": float(track.t[lk][0]) if lk.any() else None,
        "last_lock_s": float(track.t[lk][-1]) if lk.any() else None,
        "carrier_snr_mean_db": float(np.mean(snr_locked)) if snr_locked.size else None,
        "carrier_snr_max_db": float(np.max(snr_locked)) if snr_locked.size else None,
        "cnr_48k_offset_db": float(cnr_offset_db),      # CNR in 48 kHz = snr_db(channel) - this
        "f_offset_mean_hz": float(np.mean(f_offset[lk])) if lk.any() else 0.0,
        "f_offset_slew_hz_s": float(np.median(np.diff(f_offset)) / block_s) if n_blocks > 1 else 0.0,
        "cache_intermediate": bool(cache_intermediate),
        "elapsed_s": float(elapsed),
        "elapsed_pass1_s": float(t_pass1 - t_start),
        "realtime_factor": float(duration_s / elapsed) if elapsed > 0 else None,
        "units": "Hz (instantaneous frequency deviation)",
    }
    diag = {
        "carrier_track": track,
        "f_meas_hz": meas["f_meas"],
        "detected": meas["detected"],
        "narrowband": meas.get("narrowband"),
        "frac_narrowband": meas.get("frac_narrowband"),
        "z": meas["z"],
        "noise_floor": meas["floor"],
        "f_pred_causal_hz": f_pred,
        "locked_causal": locked_causal,
        "plan": {k: v for k, v in plan.items() if not k.startswith("h")},
        **{k: v for k, v in meta.items() if k not in ("decimation",)},
        "decimation": meta["decimation"],
    }
    if return_spectra:
        diag["block_spectra"] = {"freqs_hz": freqs, "power": P, "counts": spectra.count.copy()}
    audio = AudioSignal(samples=out, fs=fs_out, t0=0.0, source=str(path), meta=meta)
    return audio, diag
