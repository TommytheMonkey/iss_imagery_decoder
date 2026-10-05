# Preserved real-recording reference results

The original full-recording outputs remain in `reports/real/16-54` and
`reports/real/18-30`; finalization does not rerun or overwrite them. The SHA-256
manifest here covers every file under `reports/real` except the two large
`intermediate.c64` execution caches. Paths are relative to the v0.3 root.
It includes the input-file hashes and metadata, candidate inventories, complete
frequency tracks, diagnostics, execution provenance and checkpoint reports.

Both recordings were INCONCLUSIVE, with no PROBABLE/CONFIRMED candidate or
recovered image. These are reference outputs, not ground-truth negative labels.
See `reports/FINAL_REPORT.md` for interpretation and sensitivity limits.

Verify from the v0.3 root:

```sh
sha256sum -c reports/reference/REFERENCE_SHA256SUMS
```

The caches remain available for follow-up, but are not source or reference-report
artifacts. The input recordings are external to the baseline; their recorded
SHA-256 values allow verification before reanalysis. Algorithms/configuration
were frozen before inspecting either target recording (`V03_SCIENCE_FREEZE.json`).
