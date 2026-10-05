"""Per-pixel tone estimation: windowed-DFT power, noise-shape whitening, soft (posterior-mean) and chain (HMM) estimators.

Ported (logic copied, nothing imported) from the verified analysis code:
  * `tone_power`          <- scripts/robot36.tone_power (bit-identical arithmetic: odd Hann window, complex64 DFT matrix)
  * `soft_estimate`       <- scripts/robot36.estimate(method='soft') + take2/combo/combo_lib.est_soft (whitening, posterior sd)
  * `chain_estimate`      <- take2/viterbi/hmm_lib.emission + trans_logmat('gauss', sym=True) + forward_backward,
                             settings of take2/combo ('gate_w_hmm12_T8_g80': T = 8, sigma = 80 Hz/px)
  * `estimate_noise_shape`<- take2/combo/combo_lib.noise_shape (regions) / robust percentile over the whole signal
All estimators are scale-free (the evidence is normalised by the per-window median power) and deterministic.
"""
import warnings
import numpy as np

_LN2 = float(np.log(2.0))


def tone_power(x: np.ndarray, fs: int, x_t0: float, centres: np.ndarray, win_ms: float, freqs: np.ndarray,
               chunk: int = 20000) -> np.ndarray:
    """|DFT|^2 of Hann-windowed frames centred at `centres` (absolute seconds; x[0] is at x_t0) evaluated at `freqs`.
    Returns centres.shape + (len(freqs),), float32. Vectorised: frames @ exp(-2j*pi*n*f/fs).
    Window length = round(win_ms*fs/1000) forced odd (577 samples at 12 ms / 48 kHz); frames that reach beyond the signal
    are edge-clamped (sample indices clipped), as in the verified implementation."""
    centres = np.asarray(centres, dtype=np.float64)
    shp = centres.shape
    c = centres.ravel()
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float32)
    w = int(round(win_ms * fs / 1000.0))
    w += (w % 2 == 0)
    half = w // 2
    n = np.arange(-half, half + 1)
    win = np.hanning(w).astype(np.float32)
    Mx = np.exp(-2j * np.pi * np.outer(n, freqs) / fs).astype(np.complex64)
    out = np.empty((len(c), len(freqs)), np.float32)
    if len(c) == 0 or x.size == 0:
        out[:] = 0.0
        return out.reshape(shp + (len(freqs),))
    chunk = max(int(chunk), 1)
    for s in range(0, len(c), chunk):
        cc = c[s:s + chunk]
        idx = np.round((cc - x_t0) * fs).astype(np.int64)[:, None] + n[None, :]
        np.clip(idx, 0, x.size - 1, out=idx)
        fr = x[idx] * win[None, :]
        out[s:s + chunk] = np.abs(fr @ Mx) ** 2
    return out.reshape(shp + (len(freqs),))


def estimate_noise_shape(x: np.ndarray, fs: int, freqs: np.ndarray, win_ms: float, regions: list[tuple[float, float]] | None,
                         x_t0: float = 0.0, hop_ms: float | None = None, percentile: float = 20.0,
                         max_windows: int = 50000, method: str = "tone_excluded_median", exclude_hz: float = 350.0) -> np.ndarray:
    """Relative noise power per frequency (mean 1). If regions (absolute s) are given, use windows from them;
    otherwise use the 20th-percentile power per frequency over windows spread across the whole signal (robust to tones).

    regions: median power per frequency over windows at hop_ms (default win_ms/2) inside the regions, i.e. the receiver-noise
    spectral shape of no-tone audio that went through the same preprocessing (reproduces take2/combo noise_shape_gate_w12.npy
    to 0.1 dB rms).
    no regions (signal-only file): windows spread evenly over the signal (up to max_windows). method 'tone_excluded_median'
    (default): per window the strongest grid bin is located and every bin within +-exclude_hz of it is ignored for that
    window (the Hann main lobe is +-167 Hz at 12 ms, the near sidelobes are < -58 dB beyond 350 Hz); the shape is the median
    of the remaining values per bin, falling back to the (median-scaled) 20th percentile over all windows where fewer than 32
    windows survive. This removes the
    tone bias of a plain low percentile (a 20th percentile is inflated by 1/(1-c) where c is the fraction of windows whose
    tone leaks into the bin, which distorts the shape by up to ~3 dB on a continuous SSTV signal). method 'percentile':
    the plain `percentile`-th percentile per bin (the docstring's original estimator, kept for comparison).
    All estimates are exponential-distribution quantiles of the same noise PSD, so after the mean-1 normalisation they
    estimate the same shape. Returns float64 (len(freqs),), strictly positive; all-ones if nothing can be measured."""
    x = np.asarray(x, dtype=np.float32)
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    K = len(freqs)
    w_s = win_ms / 1000.0
    dur = x.size / fs
    t_lo, t_hi = x_t0 + w_s / 2, x_t0 + dur - w_s / 2
    if x.size == 0 or t_hi <= t_lo:
        return np.ones(K)
    hop = (hop_ms if hop_ms is not None else win_ms / 2.0) / 1000.0
    centres = None
    use_regions = False
    if regions:
        parts = []
        for a, b in regions:
            a2, b2 = max(float(a), t_lo), min(float(b), t_hi)
            if b2 > a2:
                parts.append(np.arange(a2, b2 + 1e-9, hop))
        if parts:
            centres = np.concatenate(parts)
            use_regions = True
    if centres is None or centres.size < 8:
        n_win = int(min(max_windows, max(8, np.floor((t_hi - t_lo) / hop) + 1)))
        centres = np.linspace(t_lo, t_hi, n_win)
        use_regions = False
    if centres.size > max_windows:
        centres = centres[np.linspace(0, centres.size - 1, max_windows).round().astype(np.int64)]
    P = tone_power(x, fs, x_t0, centres, win_ms, freqs).astype(np.float64)
    if use_regions:
        shape = np.median(P, axis=0)
    elif method == "percentile":
        shape = np.percentile(P, percentile, axis=0)
    else:
        f_peak = freqs[np.argmax(P, axis=-1)]                                   # (n,)
        contaminated = np.abs(freqs[None, :] - f_peak[:, None]) <= exclude_hz    # (n, K)
        count = (~contaminated).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN bins are handled by the fallback below
            shape = np.nanmedian(np.where(contaminated, np.nan, P), axis=0)
        # fallback for bins where (almost) every window has its peak nearby (e.g. a constant tone for most of the file):
        # the 20th percentile over all windows, scaled to the median of an exponential distribution (x ln2/ln1.25), so that
        # it sits on the same scale as the other bins; its tone bias is 1/(1-c) for a contaminated fraction c, like any quantile.
        fallback = np.percentile(P, 20.0, axis=0) * (np.log(2.0) / np.log(1.25))
        bad = (count < 32) | ~np.isfinite(shape)
        shape = np.where(bad, fallback, shape)
    shape = np.asarray(shape, dtype=np.float64)
    if not np.all(np.isfinite(shape)) or shape.max() <= 0:
        return np.ones(K)
    shape = np.maximum(shape, 1e-6 * shape.max())      # never zero (whitening divides by it)
    return shape / shape.mean()


def _whiten_and_evidence(P: np.ndarray, noise_shape: np.ndarray | None, T: float):
    """returns (P_w float64, evidence = P_w/(sigma2*T), quality_db) with sigma2 = median over freqs / ln 2 per window."""
    Pw = np.asarray(P, dtype=np.float64)
    if noise_shape is not None:
        Pw = Pw / np.asarray(noise_shape, dtype=np.float64)
    med = np.median(Pw, axis=-1, keepdims=True)
    pk = Pw.max(axis=-1, keepdims=True)
    quality_db = (10.0 * np.log10(pk / (med + 1e-20) + 1e-20))[..., 0]
    sig2 = med / _LN2
    ll = Pw / (sig2 * T + 1e-30)
    return Pw, ll, quality_db


def _posterior_stats(post: np.ndarray, freqs: np.ndarray):
    mu = (post * freqs).sum(-1)
    var = (post * freqs ** 2).sum(-1) - mu ** 2
    return mu, np.sqrt(np.maximum(var, 0.0))


def soft_estimate(P: np.ndarray, freqs: np.ndarray, noise_shape: np.ndarray | None = None, T: float = 1.0):
    """Bayesian posterior-mean frequency per window. Whiten P by noise_shape; sigma2 = median over freqs / ln 2 per window;
    posterior ∝ exp(P_w / (sigma2*T)); return (f_est, post_sd, quality_db) where quality = 10log10(peak/median).
    Works on any shape (..., n_freq); quality_db is computed on the whitened power."""
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    _, ll, quality_db = _whiten_and_evidence(P, noise_shape, float(T))
    ll = ll - ll.max(axis=-1, keepdims=True)
    post = np.exp(ll)
    post /= post.sum(axis=-1, keepdims=True)
    mu, sd = _posterior_stats(post, freqs)
    return mu, sd, quality_db


def gaussian_transition(freqs: np.ndarray, sigma_hz: float) -> np.ndarray:
    """Symmetric Gaussian chain potential exp(-0.5*(df/sigma)^2) over the frequency grid, scaled by one global constant
    (the largest row sum) so that it is a sub-stochastic symmetric kernel. NOT row-normalised on purpose: the row-normalised
    (directed HMM) version makes the band-edge states sticky and produces edge streaks in noise (take2/viterbi finding);
    a global constant leaves the forward-backward posterior unchanged."""
    f = np.asarray(freqs, dtype=np.float64).ravel()
    d = f[None, :] - f[:, None]
    Tm = np.exp(-0.5 * (d / float(sigma_hz)) ** 2)
    return Tm / Tm.sum(axis=1).max()


def forward_backward(e: np.ndarray, Tm: np.ndarray) -> np.ndarray:
    """Posterior state marginals (L, N, K) of a chain with scaled emissions e (L, N, K) in (0, 1] and symmetric kernel Tm (K, K),
    scaled-probability recursions (every step renormalised, no underflow), vectorised over the L chains."""
    L, N, K = e.shape
    TmT = Tm.T.copy()
    alpha = np.empty((L, N, K))
    a = e[:, 0, :] / K
    a = a / (a.sum(-1, keepdims=True) + 1e-300)
    alpha[:, 0] = a
    for i in range(1, N):
        a = (a @ Tm) * e[:, i, :]
        a /= a.sum(-1, keepdims=True) + 1e-300
        alpha[:, i] = a
    post = np.empty_like(alpha)
    post[:, -1] = alpha[:, -1]
    b = np.ones((L, K))
    for i in range(N - 2, -1, -1):
        b = (e[:, i + 1, :] * b) @ TmT
        b /= b.sum(-1, keepdims=True) + 1e-300
        p = alpha[:, i] * b
        post[:, i] = p / (p.sum(-1, keepdims=True) + 1e-300)
    return post


def chain_estimate(P: np.ndarray, freqs: np.ndarray, noise_shape: np.ndarray | None, T: float = 8.0, sigma_hz: float = 80.0):
    """Forward-backward posterior mean along axis -2 (pixels of one line) of a Gaussian Markov chain over frequency states:
    emission log-likelihood = P_w/(sigma2*T) (sigma2 as in soft_estimate), symmetric Gaussian transition with std sigma_hz
    (normalised over the grid), scaled-probability forward/backward passes (no underflow), returns (f_est, post_sd).
    Input P has shape (n_lines, n_px, n_freq); vectorise over lines. A 2-D input (n_px, n_freq) is treated as one line."""
    P = np.asarray(P)
    if P.ndim < 2:
        raise ValueError("chain_estimate expects (..., n_px, n_freq)")
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    lead = P.shape[:-2]
    P3 = P.reshape((-1,) + P.shape[-2:])
    _, ll, _ = _whiten_and_evidence(P3, noise_shape, float(T))
    e = np.exp(ll - ll.max(axis=-1, keepdims=True))
    Tm = gaussian_transition(freqs, sigma_hz)
    post = forward_backward(e, Tm)
    mu, sd = _posterior_stats(post, freqs)
    return mu.reshape(lead + (P.shape[-2],)), sd.reshape(lead + (P.shape[-2],))


def to_level(f: np.ndarray, f_black: float = 1500.0, f_white: float = 2300.0) -> np.ndarray:
    return np.clip((f - f_black) / (f_white - f_black) * 255.0, 0, 255)
