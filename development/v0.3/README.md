# Robot36dec v0.3 development copy

This isolated copy preserves the parent frozen v0.2.0 implementation. It replaces single-carrier IQ selection with candidate-preserving RF analysis and requires independently measured Robot 36 protocol structure before accepting an image.

Read [DEVELOPMENT.md](DEVELOPMENT.md) for usage, metadata, verdicts, tests and limitations; [implementation details](reports/IMPLEMENTATION.md) describe the changes. The completed validation and recording analysis are documented in [FINAL_REPORT.md](reports/FINAL_REPORT.md).

Run from this directory using `tools/run-python` in this workspace:

```
./tools/run-python -m unittest tests.test_v03 -v
./tools/run-python -m robot36dec decode INPUT.iq --iq --iq-rate 1024000 \
  --center-freq 437650000 --signal-freq 437550000 --out OUTPUT
```

The old README is retained as [README_V02.md](README_V02.md). Its acceptance and validation statements describe v0.2, not this development version. A negative v0.3 verdict does not establish that no image is recoverable.

## Reproducible baseline

Use absolute input/output paths with `tools/analyze-iq`; the wrapper selects this
isolated v0.3 copy from any working directory. Exact commands are saved in
[CAPTURE_COMMANDS.md](CAPTURE_COMMANDS.md). Source hashes are in
`SOURCE_SHA256SUMS`; final validation and environment details are in `BASELINE.json`.

Required metadata: **complex samples/second**, unsigned 8-bit interleaved **I,Q**
(no file header), actual receiver center frequency, and expected nominal carrier
frequency, all in Hz. Do not infer the actual center from the filename. For the
two reference recordings these are 1,024,000, 437,650,000 and 437,550,000,
respectively. Supply measured receiver frequency correction and uncertainty when
known; otherwise the documented search assumptions apply.

| Verdict | Meaning |
| --- | --- |
| CONFIRMED ROBOT36 | Held-out Robot 36 line geometry, positive short second-porch evidence, and aligned valid code-8 VIS with even parity. |
| PROBABLE | Required line structure passes, but evidence needed for confirmation is missing. |
| INCONCLUSIVE | Partial structure is present; required checks fail. |
| NO EVIDENCE | Tested hypotheses lack sufficient structure. This does not prove absence. |

The independent synthetic sweep confirmed +6, 0 and −3 dB, classified −6 dB as
PROBABLE, −9 dB as INCONCLUSIVE and −12 dB as NO EVIDENCE. These are single-seed
results with CNR referenced to **48 kHz**, before imposed fades, with Doppler,
6 kHz oscillator error and −100 kHz capture offset. They are not calibrated
real-world sensitivity thresholds or detection probabilities.

Remaining limits include finite search/clock-uncertainty bounds, fragmented or
aliased RF tracks, an approximate colored-noise model, FM threshold effects,
limited false-alarm sampling, and unqualified pixel/color fidelity. The real
captures have approximately one ADC-code RMS variation; that quantization regime
is not adequately represented by the sensitivity sweep. Frequency motion alone
does not identify ISS or establish pure Doppler. Missing VIS can leave frame
origin uncertain. **Do not use a negative verdict to rule out a recoverable image.**

The full [16:54 report](reports/real/16-54/candidates.json) and
[18:30 report](reports/real/18-30/candidates.json), their complete RF tracks and
supporting diagnostics are preserved as reference results. Both were
INCONCLUSIVE, with no accepted image. `reports/reference/REFERENCE_SHA256SUMS`
protects these report artifacts against accidental changes.
