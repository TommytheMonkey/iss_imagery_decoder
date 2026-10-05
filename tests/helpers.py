"""Shared helpers for the robot36dec test-suite. Test code only: the package never imports this module.

Also holds the established facts about the reference recording (SPEC.md section 2). They are test oracles and are
never fed to the decoder."""
import hashlib
import os
import pathlib

import numpy as np

TESTS_DIR = pathlib.Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent                                     # robot36-decoder/
PROJECT_DIR = ROOT.parents[1]                               # iss-image-recovery/
REAL_WAV = PROJECT_DIR / "audio" / "ISS_2026-10-04_10-23-00.wav"
VENV_PY = ROOT.parent / ".venv" / "bin" / "python"

FS = 48000
FREQS = np.arange(1350.0, 2451.0, 10.0)
WIDTH, HEIGHT = 320, 240

# --- established facts about the reference recording (oracles only) ---------------------------------------------
REAL_DURATION_S = 659.373
REAL_T_LEADER = 322.7945          # first 1900 Hz leader start (s)
REAL_T_LINE0 = 323.7050           # line 0 start (s)
REAL_PERIOD = 0.14999640          # line period (s), -24 ppm
REAL_VIS = 8
REAL_N_SYNC_LINES = 62            # lines with an individually detectable sync pulse
REAL_NOISE_STARTS = (430.0, 470.0, 510.0, 550.0, 590.0, 10.0, 60.0)   # any 36 s from these is pure noise
REAL_VIS_REGION = (322.0, 323.75)  # replaced by noise in the header-less copy test

OUTPUT_SUFFIXES = ("_image.png", "_luma.png", "_luma_x3.png", "_luma_fused.png", "_image_fused_masked.png",
                   "_mask.png", "_diagnostics.json", "_line_sync.csv", "_line_metrics.csv")


# --- files ---------------------------------------------------------------------------------------------------------
def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_signature(path) -> tuple:
    """(sha256, size, mtime_ns): everything a decoder run must leave untouched."""
    st = os.stat(path)
    return (sha256_file(path), st.st_size, st.st_mtime_ns)


def dir_listing(path) -> dict:
    """{name: size} of a directory (to prove nothing was written next to an input)."""
    p = pathlib.Path(path)
    return {q.name: q.stat().st_size for q in p.iterdir()}


# --- diagnostics JSON access (tolerant to the exact nesting chosen by diagnostics.write_outputs) --------------------
TIMING_KEYS = ("t_line0", "period", "source", "confidence")            # timing.Timing
VIS_KEYS = ("vis_code", "t_leader", "t_line0")                        # header.VISCandidate
PERIOD_KEYS = ("comb_z", "best_period_any")                           # sync.PeriodEstimate
LINESYNC_KEYS = ("n_inliers", "resid_ms_sd")                          # sync.LineSync
GATE_KEYS = ("applied", "two_population_score")                       # preprocess.GateInfo


def find_record(obj, required):
    """Depth-first search of a JSON tree for the first dict holding all `required` keys (None if absent)."""
    if isinstance(obj, dict):
        if all(k in obj for k in required):
            return obj
        for v in obj.values():
            r = find_record(v, required)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_record(v, required)
            if r is not None:
                return r
    return None


def find_records(obj, required, out=None):
    """All dicts in a JSON tree holding all `required` keys (document order)."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        if all(k in obj for k in required):
            out.append(obj)
        else:
            for v in obj.values():
                find_records(v, required, out)
    elif isinstance(obj, list):
        for v in obj:
            find_records(v, required, out)
    return out


def _preferred(diag, key, required):
    v = diag.get(key) if isinstance(diag, dict) else None
    if isinstance(v, dict) and all(k in v for k in required):
        return v
    return find_record(diag, required)


def get_timing(diag) -> dict | None:
    return _preferred(diag, "timing", TIMING_KEYS)


def get_vis_candidates(diag) -> list:
    """VIS candidate dicts: diag['vis'] as a list, or diag['vis']['candidates'] (+ 'best'), else a tree search."""
    vis = diag.get("vis") if isinstance(diag, dict) else None
    if isinstance(vis, list):
        return [c for c in vis if isinstance(c, dict) and "vis_code" in c]
    if isinstance(vis, dict):
        if isinstance(vis.get("candidates"), list):
            cands = [c for c in vis["candidates"] if isinstance(c, dict) and "vis_code" in c]
            best = vis.get("best")
            if isinstance(best, dict) and "vis_code" in best and best not in cands:
                cands.insert(0, best)
            return cands
        if "vis_code" in vis:
            return [vis]
    return find_records(diag, VIS_KEYS)


def get_period_estimate(diag) -> dict | None:
    return _preferred(diag, "period_estimate", PERIOD_KEYS)


def get_line_sync(diag) -> dict | None:
    return _preferred(diag, "line_sync", LINESYNC_KEYS)


def get_gate(diag) -> dict | None:
    return _preferred(diag, "gate", GATE_KEYS)


def vis8_claims(cands, min_conf: float = 0.6) -> list:
    """VIS candidates that claim Robot 36 (code 8) with a credible confidence."""
    return [c for c in cands if c.get("vis_code") == 8 and float(c.get("confidence") or 0.0) >= min_conf]


def credible_claims(cands, min_conf: float = 0.6) -> list:
    return [c for c in cands if c.get("vis_code") is not None and float(c.get("confidence") or 0.0) >= min_conf]


def uses_vis(source) -> bool:
    return "vis" in str(source).lower()


# --- timing arithmetic ---------------------------------------------------------------------------------------------
def line_offset(t_est: float, t_true: float, period: float) -> tuple[int, float]:
    """(k, residual_s): t_est = t_true + k*period + residual, |residual| <= period/2."""
    k = int(round((t_est - t_true) / period))
    return k, float(t_est - t_true - k * period)


def ppm_error(p_est: float, p_true: float) -> float:
    return float((p_est / p_true - 1.0) * 1e6)


# --- images --------------------------------------------------------------------------------------------------------
def luma_truth(rgb) -> np.ndarray:
    """The encoder's luminance (0.299 R + 0.587 G + 0.114 B), float 0..255."""
    rgb = np.asarray(rgb, np.float64)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def load_gray_png(path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def png_size(path) -> tuple[int, int, str]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size[0], im.size[1], im.mode


def pearson(a, b) -> float:
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def tone_at(track, fs, t) -> float:
    return float(track[int(round(t * fs))])


# --- independent reference decoder (NOT the package's): 12 ms Hann DFT + argmax at every Y pixel centre -------------
def local_luma_argmax(x, fs, x_t0, t_line0, period, n_lines, win_ms: float = 12.0, bandpass: bool = True):
    """Luminance (n_lines, 320) levels 0..255 from audio in Hz by the simplest possible estimator; used to validate the
    synthesiser without depending on the decoder under test (robot36.py 'argmax' logic, 10 Hz grid, no interpolation)."""
    x = np.asarray(x, np.float64)
    if bandpass:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(6, [400, 3200], btype="bandpass", fs=fs, output="sos")
        x = sosfiltfilt(sos, x - x.mean())
    x = x.astype(np.float32)
    y_start_ms, y_ms = 12.0, 88.0
    n = np.arange(n_lines)[:, None]
    ty = t_line0 + n * period + (y_start_ms + (np.arange(WIDTH)[None, :] + 0.5) * y_ms / WIDTH) / 1000.0
    w = int(round(win_ms * fs / 1000)); w += (w % 2 == 0)
    half = w // 2; nn = np.arange(-half, half + 1)
    win = np.hanning(w).astype(np.float32)
    M = np.exp(-2j * np.pi * np.outer(nn, FREQS) / fs).astype(np.complex64)
    c = ty.ravel()
    out = np.empty((len(c), len(FREQS)), np.float32)
    for s in range(0, len(c), 20000):
        cc = c[s:s + 20000]
        idx = np.clip(np.round((cc - x_t0) * fs).astype(np.int64)[:, None] + nn[None, :], 0, len(x) - 1)
        out[s:s + 20000] = np.abs((x[idx] * win[None, :]) @ M) ** 2
    f = FREQS[np.argmax(out, axis=1)].reshape(ty.shape)
    return np.clip((f - 1500.0) / 800.0 * 255.0, 0, 255)


def local_iq_demod(path, fs_iq, f_offset_fn, fs_out=48000):
    """Minimal IQ reader/demodulator for validating write_iq_u8 (knows the true offset; not a decoder)."""
    from scipy.signal import resample_poly
    from fractions import Fraction
    raw = np.fromfile(path, np.uint8)
    z = ((raw[0::2].astype(np.float32) - 127.5) + 1j * (raw[1::2].astype(np.float32) - 127.5)) / 127.5
    t = np.arange(len(z)) / fs_iq
    off = f_offset_fn(t) if callable(f_offset_fn) else np.full(len(z), float(f_offset_fn))
    z = z * np.exp(-2j * np.pi * np.cumsum(off) / fs_iq)
    fr = Fraction(int(fs_out), int(round(fs_iq)))
    zb = resample_poly(z, fr.numerator, fr.denominator)
    f = np.angle(zb[1:] * np.conj(zb[:-1])) * fs_out / (2 * np.pi)
    return np.concatenate([f[:1], f]).astype(np.float32)
