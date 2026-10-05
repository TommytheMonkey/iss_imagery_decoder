# Analyze a new RTL-SDR recording

This command works from any directory. Replace the absolute input and output
paths. Set the numbers to the **actual metadata of the new recording**; the
example assumes the same capture setup as the two reference recordings.

```sh
/home/tommy/iss/robot36-decoder/development/v0.3/tools/analyze-iq /ABSOLUTE/PATH/NEXT.iq --iq-rate 1024000 --center-freq 437650000 --signal-freq 437550000 --out /ABSOLUTE/PATH/NEXT-analysis
```

Input is headerless unsigned 8-bit interleaved I,Q. Rate is complex samples per
second, frequencies are Hz. The filename is not metadata. Use a fresh output
directory. The full recording is scanned; no time-region hint is supplied.
All retained RF candidates are evaluated at 11/15/25 kHz total channel bandwidth,
0/75 us de-emphasis, and gate off/auto. Reports retain candidate trajectories,
structural checks, VIS evidence, acceptance/rejection reasons and any accepted images.

Default residual search is ±30 kHz and remaining receiver uncertainty is 20 ppm;
required coverage also accounts for Doppler and margin. To declare different
uncertainty or a measured correction, append `--uncertainty-ppm VALUE` and/or
`--receiver-ppm VALUE`. The latter means an observed positive RF shift, not the
rtl_sdr driver's `-p` convention. Consult `tools/analyze-iq --help` and
[DEVELOPMENT.md](DEVELOPMENT.md) before changing those assumptions.

# Reproduce baseline tests

Run from `/home/tommy/iss/robot36-decoder/development/v0.3`:

```sh
./tools/run-python -m unittest tests.test_v03 -v
./tools/run-python -m tools.validate_v03 --out reports/validation/NEW-RUN --cases all --workers 4
./tools/run-python -m tools.verify_v02
sha256sum -c SOURCE_SHA256SUMS
sha256sum -c reports/reference/REFERENCE_SHA256SUMS
```

The RF suite is computationally expensive. The copied v0.2 pytest modules are
historical API tests; the current complete v0.3 suite is the 25 unit tests plus
14 explicit RF cases. The final run is stored in `reports/validation/baseline-final`.
