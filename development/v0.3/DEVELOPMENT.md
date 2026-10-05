# Robot36dec v0.3 development copy

The parent repository is the untouched v0.2.0 implementation. `V02_BASELINE.json`
records all copied source/test/tool/document hashes; `python -m tools.verify_v02`
checks them. The copied `FROZEN.md` and `VALIDATION.md` are historical v0.2 records,
not claims that v0.3 has the same behavior or validation.

Run from this directory. Install with `pip install -e '.[test,plots]'` in a normal
network-enabled environment. In this restricted workspace, Python dependencies
are shared read-only from the existing environments. `tools/run-python` sets the
necessary paths and prevents bytecode writes.

## Raw IQ analysis

```
python -m robot36dec decode INPUT.iq --iq --iq-rate 1024000 \
  --center-freq 437650000 --signal-freq 437550000 --out OUTPUT
```

Raw IQ always uses the candidate-preserving pipeline. The old public
`iqdemod.demodulate()` single-carrier API now raises an explicit migration error;
use `robot36dec.candidates.analyze(path, out, spec)` instead. It scans the entire input;
`--start`, `--end`, `--t0`, and `--period` are WAV-only and rejected for IQ. Each RF
trajectory is demodulated independently at 11, 15, and 25 kHz total bandwidth,
with 0/75 us de-emphasis and gate off/auto. Search defaults to +/-30 kHz; the
minimum required range includes +/-11.25 kHz Doppler, declared oscillator
uncertainty, and margin. Intermediate decimation adapts to that range. Unsupported
ranges fail explicitly rather than silently truncate.

`--receiver-ppm` describes an **observed positive frequency shift** in units of
nominal RF frequency, which is subtracted from the residual searched for Doppler.
It is not the rtl_sdr driver's `-p` sign convention. `--uncertainty-ppm` describes
remaining uncertainty (default 20). Apparent RF Hz and residual Hz are reported
separately; without ephemeris/calibration, residual motion is not pure Doppler.

RF screening uses signed noise subtraction, a robust spatial noise model,
Gamma-median bias correction, and conservative Hann/Welch covariance. A spectral
z-score is a model-dependent screening statistic, never a carrier lock or ISS
identification. Narrow peaks remain hypotheses and are never excluded merely
because a low modulation index leaves a strong carrier line. Sideband aliases
and fragmented trajectories are deliberately retained; candidate count is not
transmitter count. No candidate-count cap is applied.

Cadence is acquired using sync measurements on alternating **line pairs**.
Presence is selected from training sync only (at least 12 four-line groups);
disjoint pairs validate sync contrast, off-pulse geometry, measured pulse FWHM,
1500 Hz porch, alternating 1500/2300 Hz separators, and parity contrast.
The short 1900 Hz second porch is also measured: strong contradiction vetoes
acceptance, while positive evidence is required for CONFIRMED status. Weak
second-porch evidence can support only PROBABLE status. These
checks do not use image pixels, image smoothness, or tone-estimator confidence.
A valid measured VIS additionally requires header geometry, decisive measured
bits, code 8, even parity, and agreement with the acquired sync grid. Templates
cannot repair measured parity. A header alone cannot establish an image.

Verdicts:
- CONFIRMED ROBOT36: required held-out geometry and aligned valid measured VIS.
- PROBABLE: required held-out geometry without a credible aligned VIS.
- INCONCLUSIVE: some structural checks pass, but required checks fail.
- NO EVIDENCE: the tested hypotheses lack enough structural support.

Images are rendered only for the first two verdicts. No negative verdict proves
that an image cannot be recovered. Thresholds are engineering rules, not
calibrated posterior probabilities; the report records all variants and searches.
Demodulation uses overlapping 40 s windows to bound memory for persistent RF.
Auto-gate checks that leave audio unchanged reuse the identical off-gate evidence,
explicitly marked in the report. No candidate or bandwidth is skipped. The frontend FIR latency is reported;
absolute audio timing retains this small latency (well below the 0.5 ms grid).

## Tests

```
python -m unittest tests.test_v03 -v
python -m tools.validate_v03 --out reports/validation --cases all
python -m tools.verify_v02
```

The independent encoder and RF simulator import no decoder code. CNR is referred
to 48 kHz: complex RF noise variance is `A^2 * fs_iq / (48000 * 10^(CNR/10))`.
IQ is quantized unsigned 8-bit I,Q at 1.024 MS/s with -100 kHz capture offset,
frequency drift, fades, and 6 kHz oscillator shift. The full sensitivity sweep
uses 240 lines at +6, 0, -3, -6, -9, -12 dB. ADC clipping is recorded.

The copied v0.2 pytest modules are historical tests for the original API/output
contract. They are not the v0.3 test suite: raw IQ now produces a candidate report,
and rejected WAVs produce diagnostics without a forced image. The new unittest
suite can also run under pytest; expensive RF validation is an explicit command.
Real IQ analysis is performed only after these independent validations complete,
with code/config hashes frozen beforehand. No parameter is selected using those
recordings or a previous decoder classification.

## Parallel execution of the frozen candidate evaluations

`tools.parallel_candidates` is a computational resume tool for the two full-recording analyses. It extracts the exact candidate loop from the frozen production function, removes only progress/checkpoint writing, and evaluates independent remaining candidates with four workers. Completed v0.3 measurements are retained; no v0.2 classification enters it. Candidate ordering in the final report remains unchanged. Scheduling by duration changes computation order only.

Before use, `--equivalence` regenerates the independent low-deviation RF fixture and requires exact equality of all 26 candidates, all variants, all measurements and all decisions against the completed serial RF regression. The proof is `reports/validation/parallel-equivalence.json`. Production source/configuration remains at the pre-real freeze. The final real reports explicitly record parallel resume and its provenance.

`tools.fast_execution` additionally skips the unused gate diagnostics in the off branch (which returns the unmodified input). The automatic-gate decision and both bandpasses remain unchanged. Its full low-deviation RF report is exactly equal to the serial report; tone, Gaussian-noise and impulsive-control audio arrays also match bit for bit (`fast-execution-equivalence.json`, `fast-audio-equivalence.json`). These are execution-equivalence checks, not independent DSP truth tests. The independent encoder/negative controls provide the latter. Real-report execution metadata records this shortcut. Production source and statistical thresholds remain unchanged.
