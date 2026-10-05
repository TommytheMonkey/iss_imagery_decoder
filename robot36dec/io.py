"""Input handling. Inputs are opened read-only and never modified.

Audio units inside the package are float32 *instantaneous frequency deviation in Hz* at 48 kHz. An rtl_fm WAV
(int16) maps full scale to fs_out = 48000 Hz of deviation, so Hz = sample_in_[-1,1) * 48000 (see SPEC.md section 1).
"""
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator
import os

import numpy as np


@dataclass
class AudioSignal:
    samples: np.ndarray          # float32, instantaneous frequency deviation in Hz
    fs: int                      # 48000 after resampling
    t0: float = 0.0              # absolute time (s) of samples[0] relative to the start of the input file
    source: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(len(self.samples)) / float(self.fs)

    def time_axis(self) -> np.ndarray:
        """Absolute time (s) of every sample."""
        return self.t0 + np.arange(len(self.samples), dtype=np.float64) / float(self.fs)


@dataclass
class IQSpec:
    fs: float                    # IQ sample rate (S/s)
    center_freq: float           # tuner centre frequency (Hz)
    signal_freq: float = 437.550e6
    fmt: str = "u8"              # only unsigned 8-bit interleaved I/Q is required


# unsigned 8-bit -> float32 in (-1, 1): (u8 - 127.5) / 127.5 (RTL-SDR convention, zero at 127.5)
_U8_LUT = ((np.arange(256, dtype=np.float64) - 127.5) / 127.5).astype(np.float32)


def rational_ratio(target_fs: float, fs: float, max_den: int = 1 << 20) -> tuple[int, int]:
    """(up, down) with up/down ~= target_fs/fs. Exact when both rates are integer valued (the usual case);
    otherwise the best rational approximation with a bounded denominator."""
    if abs(target_fs - round(target_fs)) < 1e-6 and abs(fs - round(fs)) < 1e-6:
        fr = Fraction(int(round(target_fs)), int(round(fs)))
    else:
        fr = Fraction(float(target_fs) / float(fs)).limit_denominator(max_den)
    return fr.numerator, fr.denominator


def load_wav(path: str, fullscale_hz: float = 48000.0, target_fs: int = 48000) -> AudioSignal:
    """Read a mono (or first-channel) PCM WAV read-only with soundfile, convert to float32 Hz units
    (sample_value_in_[-1,1] * fullscale_hz), resample with scipy.signal.resample_poly if fs != target_fs.
    meta must include: original fs, channels, duration_s, fullscale_hz."""
    import soundfile as sf

    info = sf.info(path)
    # soundfile opens the file in read mode only; dtype='float32' scales integer PCM to [-1, 1) (int16 / 32768).
    data, fs = sf.read(path, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(data[:, 0])
    del data
    n_orig = int(x.shape[0])
    duration_s = n_orig / float(fs)
    resampled = False
    up = down = 1
    if int(fs) != int(target_fs):
        from scipy.signal import resample_poly
        up, down = rational_ratio(target_fs, fs)
        x = resample_poly(x.astype(np.float64), up, down).astype(np.float32)
        resampled = True
    if fullscale_hz != 1.0:
        x = (x * np.float32(fullscale_hz)).astype(np.float32, copy=False)
    meta = {
        "original_fs": int(fs),
        "channels": int(info.channels),
        "duration_s": float(duration_s),
        "fullscale_hz": float(fullscale_hz),
        "subtype": str(info.subtype),
        "format": str(info.format),
        "n_samples_original": n_orig,
        "n_samples": int(x.shape[0]),
        "resampled": resampled,
        "resample_up": int(up),
        "resample_down": int(down),
        "units": "Hz (instantaneous frequency deviation)",
    }
    return AudioSignal(samples=x, fs=int(target_fs), t0=0.0, source=str(path), meta=meta)


def iq_num_samples(path: str) -> int:
    """Number of complex samples in an unsigned 8-bit interleaved IQ file (two bytes per sample; a trailing odd
    byte is ignored)."""
    return int(os.path.getsize(path) // 2)


def u8_to_complex64(raw: np.ndarray) -> np.ndarray:
    """Convert an interleaved I,Q uint8 array (even length) to complex64 with (u8 - 127.5)/127.5 per component.
    Uses a 256-entry lookup and a dtype view (no per-sample arithmetic in Python)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if raw.size % 2:
        raw = raw[:-1]
    return _U8_LUT[raw].view(np.complex64)


def iter_iq_u8(path: str, chunk_samples: int = 1 << 20, overlap: int = 0,
               lookahead: int = 0) -> Iterator[np.ndarray]:
    """Yield complex64 chunks from an unsigned 8-bit interleaved IQ file opened read-only (buffered reads of exactly the
    bytes needed; a memory map was used before but its resident pages counted up to the whole file against the process);
    (u8 - 127.5)/127.5. Optional overlap (in complex samples) for filter continuity. Never loads the whole file.

    Chunk layout (``n_new`` = number of samples first appearing in this chunk, == chunk_samples except for the
    last chunk):
      * overlap == 0 and lookahead == 0 (default): consecutive, non-overlapping chunks that concatenate exactly
        to the file.
      * overlap > 0: every chunk is prefixed with the ``overlap`` samples preceding its first new sample (zeros
        for the first chunk), i.e. classic overlap-save history.
      * lookahead > 0: every chunk is suffixed with the ``lookahead`` samples following its last new sample
        (zeros beyond the end of the file).
    So len(chunk) == overlap + n_new + lookahead and the new samples are chunk[overlap:overlap + n_new].
    The file is opened read-only and the mapping is released when the iterator is exhausted or closed."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if overlap < 0 or lookahead < 0:
        raise ValueError("overlap and lookahead must be non-negative")
    n_total = iq_num_samples(path)
    if n_total == 0:
        return
    with open(path, "rb") as fh:
        start = 0
        while start < n_total:
            stop = min(n_total, start + chunk_samples)
            lo = start - overlap
            hi = stop + lookahead
            lo_c = max(lo, 0)
            hi_c = min(hi, n_total)
            fh.seek(2 * lo_c)
            raw = np.frombuffer(fh.read(2 * (hi_c - lo_c)), dtype=np.uint8)
            body = u8_to_complex64(raw)
            if lo_c != lo or hi_c != hi:
                out = np.zeros(hi - lo, dtype=np.complex64)
                out[lo_c - lo:lo_c - lo + body.shape[0]] = body
                body = out
            yield body
            start = stop
