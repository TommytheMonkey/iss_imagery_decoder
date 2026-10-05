"""pytest configuration for robot36dec: paths, skip logic, cached synthetic inputs, cached CLI runs.

Run with the claude-recovery venv:  ../.venv/bin/python -m pytest tests -q
Synthetic inputs are synthesised once per session into a pytest temp directory (never next to real recordings) and
their sha256/size/mtime are recorded immediately so that tests can prove the decoder never touches its inputs.
"""
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest
import soundfile as sf

TESTS_DIR = pathlib.Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent
for _p in (str(ROOT), str(TESTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import helpers  # noqa: E402
from robot36dec import synth  # noqa: E402

PYTHON = str(helpers.VENV_PY) if helpers.VENV_PY.exists() else sys.executable
CLI_TIMEOUT_S = 1800
FS = helpers.FS
IQ_CENTER_FREQ = 437.540e6
IQ_SIGNAL_FREQ = 437.550e6              # nominal baseband offset +10 kHz
IQ_DOPPLER_START_HZ = 1500.0            # initial offset from the nominal position
IQ_DOPPLER_RATE_HZ_S = -100.0           # Doppler ramp


def pytest_configure(config):
    config.addinivalue_line("markers", "real: needs the reference ISS recording (skipped when it is absent)")
    config.addinivalue_line("markers", "iq: IQ u8 round-trip tests (write multi-MB files)")
    config.addinivalue_line("markers", "cli: runs the decoder command line in a subprocess")


def iq_offset_fn(t):
    """True carrier offset (Hz) from the IQ centre: nominal 10 kHz + 1.5 kHz initial error + Doppler ramp."""
    return (IQ_SIGNAL_FREQ - IQ_CENTER_FREQ) + IQ_DOPPLER_START_HZ + IQ_DOPPLER_RATE_HZ_S * np.asarray(t, np.float64)


@dataclasses.dataclass
class SynthCase:
    path: pathlib.Path
    kind: str                       # 'wav' | 'iq'
    t_line0: float                  # true start of line 0 (s into the file)
    period: float                   # true line period (s)
    n_lines: int                    # lines transmitted
    cnr_db: float
    with_vis: bool
    truth_Y: np.ndarray             # (n_lines, 320) transmitted luminance
    seed: int
    duration_s: float
    pre_silence_s: float
    sha256: str
    size: int
    mtime_ns: int
    dir_listing: dict
    track: np.ndarray | None = None       # frequency track (Hz) that was transmitted
    audio_hz: np.ndarray | None = None    # simulated demodulated audio (Hz) before int16 quantisation (wav only)
    iq_rate: float | None = None
    center_freq: float | None = None
    signal_freq: float | None = None

    @property
    def t_leader(self) -> float:
        return self.pre_silence_s

    def input_unchanged(self) -> bool:
        return (helpers.file_signature(self.path) == (self.sha256, self.size, self.mtime_ns)
                and helpers.dir_listing(self.path.parent) == self.dir_listing)


@dataclasses.dataclass
class DecodeRun:
    cmd: list
    returncode: int
    stdout: str
    stderr: str
    outdir: pathlib.Path
    basename: str
    diag: dict | None
    elapsed_s: float

    def path(self, suffix: str) -> pathlib.Path:
        return self.outdir / f"{self.basename}{suffix}"

    def describe(self) -> str:
        return (f"exit {self.returncode} after {self.elapsed_s:.1f} s\ncmd: {' '.join(map(str, self.cmd))}\n"
                f"--- stdout (tail) ---\n{self.stdout[-4000:]}\n--- stderr (tail) ---\n{self.stderr[-4000:]}")

    @property
    def timing(self):
        return helpers.get_timing(self.diag) if self.diag else None

    @property
    def vis(self):
        return helpers.get_vis_candidates(self.diag) if self.diag else []

    def luma(self) -> np.ndarray:
        return helpers.load_gray_png(self.path("_luma.png"))


# --------------------------------------------------------------------------------------------- synthetic inputs
@pytest.fixture(scope="session")
def card():
    return synth.test_card()


@pytest.fixture(scope="session")
def truth_Y(card):
    return helpers.luma_truth(card)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("synth_inputs")


def _seed_for(cnr_db, with_vis, n_lines, tag=""):
    return 1000 + (int(round(cnr_db * 10)) % 997) + (7 if with_vis else 0) + int(n_lines) + sum(map(ord, tag))


@pytest.fixture(scope="session")
def make_wav(synth_dir, card, truth_Y):
    """Factory: synthesise (once per parameter set) an rtl_fm-style WAV of the test card through the FM channel."""
    cache = {}

    def make(cnr_db: float, with_vis: bool, n_lines: int = 240, period: float = 0.150, pre_silence: float = 1.0,
             post_silence: float = 1.0, doppler_hz=None, seed: int | None = None, tag: str = "") -> SynthCase:
        key = (round(float(cnr_db), 3), bool(with_vis), int(n_lines), float(period), pre_silence, post_silence,
               None if doppler_hz is None else float(np.mean(doppler_hz)), seed, tag)
        if key in cache:
            return cache[key]
        seed = _seed_for(cnr_db, with_vis, n_lines, tag) if seed is None else seed
        track, t0 = synth.encode_robot36(card, FS, with_vis=with_vis, pre_silence_s=pre_silence,
                                         post_silence_s=post_silence, line_period_s=period, n_lines=n_lines)
        audio = synth.simulate_fm_channel(track, cnr_db, FS, seed=seed, doppler_hz=doppler_hz)
        sign = "m" if cnr_db < 0 else "p"
        name = f"synth_cnr{sign}{abs(cnr_db):g}_{'vis' if with_vis else 'novis'}_{n_lines}l{tag}.wav"
        case_dir = synth_dir / name[:-4]                     # one directory per input: proves nothing is written beside it
        case_dir.mkdir(exist_ok=True)
        path = case_dir / name
        sf.write(str(path), np.clip(audio / 48000.0, -1.0, 1.0), FS, subtype="PCM_16")
        sha, size, mtime = helpers.file_signature(path)
        case = SynthCase(path=path, kind="wav", t_line0=t0, period=period, n_lines=n_lines, cnr_db=cnr_db,
                         with_vis=with_vis, truth_Y=truth_Y[np.arange(n_lines) % truth_Y.shape[0]], seed=seed,
                         duration_s=len(track) / FS, pre_silence_s=pre_silence, sha256=sha, size=size,
                         mtime_ns=mtime, dir_listing=helpers.dir_listing(case_dir), track=track, audio_hz=audio)
        cache[key] = case
        return case

    return make


@pytest.fixture(scope="session")
def noise_wav(synth_dir) -> SynthCase:
    """40 s of carrier-off rtl_fm noise (what the receiver outputs when nothing is transmitted)."""
    dur = 40.0
    track = np.zeros(int(dur * FS), np.float32)
    audio = synth.simulate_fm_channel(track, 0.0, FS, seed=999, carrier_on=False)
    case_dir = synth_dir / "synth_noise_only_40s"
    case_dir.mkdir(exist_ok=True)
    path = case_dir / "synth_noise_only_40s.wav"
    sf.write(str(path), np.clip(audio / 48000.0, -1.0, 1.0), FS, subtype="PCM_16")
    sha, size, mtime = helpers.file_signature(path)
    return SynthCase(path=path, kind="wav", t_line0=float("nan"), period=0.150, n_lines=0, cnr_db=float("-inf"),
                     with_vis=False, truth_Y=np.zeros((0, 320)), seed=999, duration_s=dur, pre_silence_s=dur,
                     sha256=sha, size=size, mtime_ns=mtime, dir_listing=helpers.dir_listing(case_dir), track=track,
                     audio_hz=audio)


@pytest.fixture(scope="session")
def make_iq(synth_dir, card, truth_Y):
    """Factory: u8 IQ file of a (partial) frame at fs_iq with a 10 kHz nominal offset plus a Doppler ramp."""
    cache = {}

    def make(fs_iq: float, cnr_db: float = 6.0, n_lines: int = 60, with_vis: bool = True,
             seed: int | None = None) -> SynthCase:
        key = (float(fs_iq), float(cnr_db), int(n_lines), bool(with_vis), seed)
        if key in cache:
            return cache[key]
        seed = _seed_for(cnr_db, with_vis, n_lines, f"iq{int(fs_iq)}") if seed is None else seed
        track, t0 = synth.encode_robot36(card, FS, with_vis=with_vis, n_lines=n_lines)
        stem = f"synth_iq_{int(fs_iq)}sps_cnr{cnr_db:g}_{n_lines}l"
        case_dir = synth_dir / stem
        case_dir.mkdir(exist_ok=True)
        path = case_dir / f"{stem}.u8"
        synth.write_iq_u8(track, FS, fs_iq, iq_offset_fn, cnr_db, str(path), seed=seed)
        sha, size, mtime = helpers.file_signature(path)
        case = SynthCase(path=path, kind="iq", t_line0=t0, period=0.150, n_lines=n_lines, cnr_db=cnr_db,
                         with_vis=with_vis, truth_Y=truth_Y[np.arange(n_lines) % truth_Y.shape[0]], seed=seed,
                         duration_s=len(track) / FS, pre_silence_s=1.0, sha256=sha, size=size, mtime_ns=mtime,
                         dir_listing=helpers.dir_listing(case_dir), track=track, iq_rate=float(fs_iq),
                         center_freq=IQ_CENTER_FREQ, signal_freq=IQ_SIGNAL_FREQ)
        cache[key] = case
        return case

    return make


# --------------------------------------------------------------------------------------------- CLI runner
@pytest.fixture(scope="session")
def runs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("decoder_runs")


def run_cli(path, outdir, mode="auto", extra=(), iq: SynthCase | None = None) -> DecodeRun:
    """Run `python -m robot36dec decode` in a subprocess; parse the diagnostics JSON if it was written."""
    path = pathlib.Path(path)
    cmd = [PYTHON, "-m", "robot36dec", "decode", str(path), "--out", str(outdir), "--mode", mode, *extra]
    if iq is not None and iq.kind == "iq":
        cmd += ["--iq", "--iq-rate", f"{iq.iq_rate:g}", "--center-freq", f"{iq.center_freq:.1f}",
                "--signal-freq", f"{iq.signal_freq:.1f}"]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MPLBACKEND", "Agg")
    t = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=CLI_TIMEOUT_S)
    elapsed = time.time() - t
    basename = path.stem
    diag_path = pathlib.Path(outdir) / f"{basename}_diagnostics.json"
    diag = None
    if diag_path.exists():
        with open(diag_path) as fh:
            diag = json.load(fh)
    return DecodeRun(cmd=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr,
                     outdir=pathlib.Path(outdir), basename=basename, diag=diag, elapsed_s=elapsed)


@pytest.fixture(scope="session")
def decode_cli(runs_dir):
    """Factory: decode a SynthCase (or a path) once per (input, mode, extra args); results are cached."""
    cache = {}

    def run(case, mode: str = "auto", extra=(), tag: str = "") -> DecodeRun:
        path = case.path if isinstance(case, SynthCase) else pathlib.Path(case)
        key = (str(path), mode, tuple(extra), tag)
        if key in cache:
            return cache[key]
        outdir = runs_dir / f"{path.stem}__{mode}{('_' + tag) if tag else ''}_{len(cache):02d}"
        res = run_cli(path, outdir, mode, extra, iq=case if isinstance(case, SynthCase) else None)
        cache[key] = res
        return res

    return run


# --------------------------------------------------------------------------------------------- common cases
@pytest.fixture(scope="session")
def case_p6_vis(make_wav) -> SynthCase:
    return make_wav(6.0, True, 240)


@pytest.fixture(scope="session")
def case_p6_novis(make_wav) -> SynthCase:
    return make_wav(6.0, False, 240)


@pytest.fixture(scope="session")
def run_p6_vis_auto(decode_cli, case_p6_vis) -> DecodeRun:
    return decode_cli(case_p6_vis, "auto", extra=("--plots",))


# --------------------------------------------------------------------------------------------- real recording
@pytest.fixture(scope="session")
def real_wav() -> pathlib.Path:
    if not helpers.REAL_WAV.exists():
        pytest.skip(f"reference recording not found: {helpers.REAL_WAV}")
    return helpers.REAL_WAV


@pytest.fixture(scope="session")
def real_signature(real_wav):
    """Signature of the recording and of its directory, taken BEFORE any decoder run of this session."""
    return helpers.file_signature(real_wav), helpers.dir_listing(real_wav.parent)


@pytest.fixture(scope="session")
def real_run_normal(real_wav, real_signature, decode_cli) -> DecodeRun:
    return decode_cli(real_wav, "normal", tag="real")


@pytest.fixture(scope="session")
def real_run_recovery(real_wav, real_signature, decode_cli) -> DecodeRun:
    return decode_cli(real_wav, "recovery", tag="real")


@pytest.fixture(scope="session")
def real_vis_masked_wav(real_wav, real_signature, synth_dir) -> pathlib.Path:
    """Bit-exact copy of the recording with the VIS region (322.0-323.75 s) replaced by the file's own pure noise
    (from 430.0 s), same length, so that absolute times are preserved. Written in the temp directory only."""
    data, fs = sf.read(str(real_wav), dtype="int16", always_2d=False)
    assert fs == FS
    a, b = (int(round(t * fs)) for t in helpers.REAL_VIS_REGION)
    n0 = int(round(helpers.REAL_NOISE_STARTS[0] * fs))
    data = np.array(data, copy=True)
    data[a:b] = data[n0:n0 + (b - a)]
    path = synth_dir / "real_vis_region_replaced_by_noise.wav"
    sf.write(str(path), data, fs, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def real_noise_cut_wav(real_wav, real_signature, synth_dir) -> pathlib.Path:
    """36 s of pure receiver noise cut from the recording at 430 s (a control with the real noise statistics)."""
    start = int(round(helpers.REAL_NOISE_STARTS[0] * FS))
    data, fs = sf.read(str(real_wav), start=start, stop=start + 36 * FS, dtype="int16", always_2d=False)
    path = synth_dir / "real_noise_430s_36s.wav"
    sf.write(str(path), data, fs, subtype="PCM_16")
    return path


@pytest.fixture(scope="session")
def real_run_vis_masked(real_vis_masked_wav, decode_cli) -> DecodeRun:
    return decode_cli(real_vis_masked_wav, "auto", tag="real")


@pytest.fixture(scope="session")
def real_run_noise(real_noise_cut_wav, decode_cli) -> DecodeRun:
    return decode_cli(real_noise_cut_wav, "auto", tag="real")
