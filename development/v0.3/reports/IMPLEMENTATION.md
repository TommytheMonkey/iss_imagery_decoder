# v0.3 development implementation

This is an isolated copy under `development/v0.3`; the workspace has no Git repository. The parent v0.2.0 files remain unchanged. `V02_BASELINE.json` and `tools/verify_v02.py` verify all 25 original copied files. `v02-to-v03.patch` provides the exact source/test differences. The pre-real implementation and configuration are recorded in `V03_SCIENCE_FREEZE.json`, and each analysis embeds its actual production source hashes.

## Code changes

1. `protocol.py` adds waveform measurements independent of reconstructed pixels: cadence acquisition, held-out sync and pulse geometry, both sync parities, 1500 Hz porch, alternating chroma separators, and the 1900 Hz second porch. A valid measured VIS must have code 8 and even parity and align with the recovered line grid and chroma parity. Bad parity is never repaired. Measured second-porch contradiction vetoes acceptance; weak second-porch evidence permits only PROBABLE.
2. `cli.py` requires measured protocol structure before accepting WAV or rendering an image. Image smoothness/reliability and a header alone cannot establish Robot 36. IQ uses the new full-candidate pipeline and rejects timing/range overrides that would bias a whole-recording analysis.
3. `timing.py` removes override confidence floors. Confidence uses the independently measured timing source and period; a requested time/period cannot raise it.
4. `rf.py` replaces clipped per-bin excess significance with signed integration against a robust spatial background, with Gamma-median bias and Hann/Welch covariance correction. CNR reports its actual integration bandwidth and a common 48 kHz equivalent. Spectral significance remains an engineering screening statistic, not a calibrated false-alarm probability.
5. Narrow peaks remain candidates. No concentration-based CW veto discards legitimate low-deviation FM. Each candidate must independently pass waveform evidence, so actual CW can remain a reported RF hypothesis without an SSTV acceptance.
6. Frequency search includes declared receiver uncertainty and a conservative Doppler bound. The intermediate rate/filter adapt to the requested range and fail explicitly if it cannot be covered. Signed -100 kHz coarse translation is independently tested.
7. `candidates.py` preserves every retained spectral track, including fragments and sideband placements, and independently demodulates each using 11/15/25 kHz total RF bandwidth, 0/75 us de-emphasis and gate off/auto. All choices and rejections are reported. Overlapping 40 s windows bound RAM; no single winning RF carrier replaces the list.
8. `iqdemod.py` retains the tested DSP primitives but removes the public single-carrier interface with an explicit migration error. The old private trackers are unused by v0.3.
9. `header.py` removes matched-template parity repair. `decode.py` scales pixel timing with the recovered line period. `params.py` uses the v0.3 search default even when reading the copied frozen v0.2 parameter file; that file remains unchanged.
10. New independent tests and a decoder-free RF simulator provide protocol/null controls and the requested six-point CNR sweep. The mixed FM/CW fixture was corrected during validation to actually include FM modulation. A new half-rate pulse control exposed and fixed a pair-averaging alias bug.

## Interpretation limits

The candidate list is complete for this screening algorithm within its declared search interval, not a census of all transmitters. Spatial merging, thresholding, greedy trajectory association, sideband centroids, fades and unresolved overlap can still miss or fragment a signal. Frequency tracks are apparent RF motion relative to the stated nominal; oscillator error and Doppler cannot be separated without calibration/ephemeris. Robot 36 evidence does not identify the transmitter as ISS.

The synthetic sweep uses one independent encoder, seed and propagation profile; it does not establish detection probability or an operating false-alarm rate. The strict short-porch and sync-parity tests trade sensitivity for rejection. A negative result cannot establish that no image is recoverable. The retained pixel decoder still uses a long tone-analysis window and is not a weak-signal maximum-likelihood image decoder.
