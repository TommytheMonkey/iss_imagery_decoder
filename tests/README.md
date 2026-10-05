Tests use the claude-recovery venv: `../.venv/bin/python -m pytest tests -q` (install pytest into the venv if missing: `uv pip install --python ../.venv/bin/python pytest`).
`test_real_recording.py` is skipped unless `../../audio/ISS_2026-10-04_10-23-00.wav` exists (relative to robot36-decoder/).
