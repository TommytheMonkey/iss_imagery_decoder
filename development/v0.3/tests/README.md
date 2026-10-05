The v0.3 unit regression suite is `python -m unittest tests.test_v03 -v`.
The independent end-to-end RF validation is `python -m tools.validate_v03 --out reports/validation/release-final --cases all --workers 4`.
In this workspace use `tools/run-python` instead of `python` to access the existing dependencies without modifying them.

`rf_reference.py` imports no decoder code. It generates literal Robot 36 waveforms, unrelated/null controls and quantized 1.024 MS/s unsigned I,Q with explicit CNR units, oscillator error, Doppler, fades and the −100 kHz capture offset.

The copied `test_synthetic.py`, `test_real_recording.py`, `helpers.py` and `conftest.py` are historical v0.2 tests. They exercise the old single-carrier API/output contract, which v0.3 deliberately replaces. They are retained for review and are not the v0.3 test entry point. The parent implementation and its tests remain unchanged.
