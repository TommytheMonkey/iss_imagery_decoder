#!/usr/bin/env python
"""BLIND VALIDATION ONLY. Compare finished decoder outputs (<base>_luma.png) with reference copies of the Series 33 slides.

This script is never imported by the decoder and must be run only after robot36dec/params_frozen.json and FROZEN.md exist
(SPEC.md section 5 step 4). It reads images only; the decoder has already run and nothing here can influence a decode.
It also records the provenance of the comparison: the modification times of FROZEN.md, params_frozen.json and the decoded
images, the sha256 of params_frozen.json against the value quoted in FROZEN.md, and the time of this run.

Usage:
    validate_against_references.py <decoded_luma.png> <reference_dir_or_files ...> [--out VALIDATION.md]
        [--label NAME] [--fused] [--extra NAME=PATH[:fused] ...] [--control NAME=PATH ...]
        [--max-vshift 3] [--max-hshift 4] [--sigma-px 9] [--frozen FROZEN.md] [--params-frozen params_frozen.json]
        [--prior "LABEL=v(0-20),v(20-62),v(62-108),v(108-240)[,v(all)]" ...] [--aux-dir DIR] [--no-figure]
        [--exclude REGEX ...] [--primary-ref SUBSTRING] [--report-ref SUBSTRING ...] [--preamble FILE.md]

Method (ported from scripts/take2_match_event_images.py and take2_identify_figure.py; all numbers are Pearson
correlations = normalised cross-correlation (NCC) of mean-removed blocks):
  * every reference image is converted to luminance, resampled to the Robot 36 frame (320 x 240) and passed through the
    decoder's channel blur: a horizontal Gaussian of sigma 9 px (the 12 ms Hann analysis window spans ~44 px), plus a
    3-line vertical average ONLY when the decoded image being compared is a 3-line fused display (--fused / ':fused');
    the decoded image itself is never filtered;
  * per line block (0-20, 20-62, 62-108, 108-240, all) the NCC of the decoded block with the same block of a reference is
    computed at zero shift and at the best small shift (|dy| <= max-vshift lines, |dx| <= max-hshift px: the registration
    tolerance of a timing solution). References are ranked by the whole-frame ('all') NCC at zero shift; the best-matching
    slide is the slide label (parsed from the file name, 'slideNN') of the top-ranked reference with a known slide label
    (images labelled 'unknown' or 'other' are ranked and listed but cannot name the slide), and the margin is the distance
    to the best reference of any OTHER slide (whole frame and band 62-108). The per-block tables use the primary
    reference: --primary-ref (chosen a priori, e.g. the clean copy used in the earlier analysis) or else that top-ranked
    reference; --report-ref adds further named copies to a per-block table; --exclude drops files whose name matches a
    regex (e.g. decodes produced by this same pipeline from other stations' audio, which are not independent references);
  * null distribution ("what does an unrelated placement give?") per block: (a) wrong slide = every reference of another
    known slide, zero shift; (b) wrong position = the decoded block against the best reference (and, pooled, against every
    reference) at vertical positions that do not overlap the true one (|shift| >= block height, 2-line steps, also mirrored
    horizontally; for the whole frame and blocks taller than half of it, cyclic shifts of 20..220 lines, also mirrored); (c) control = the decoder's own
    pure-noise decodes (--control) against the best reference at zero shift and at their best small shift. The report
    gives mean, sd, 95th percentile and maximum of each null and the z-score of the match against the wrong-position null;
  * per-line NCC curve of each decoded image against the best reference (CSV + figure), the fraction of lines above the
    null's 95th percentile of a single line (pooled wrong positions), and the runs of lines whose 5-line-smoothed NCC
    exceeds 0.5.
Writes <out> (Markdown), <aux-dir>/<out-stem>_lines.csv, <aux-dir>/<out-stem>_ranking.csv and <aux-dir>/<out-stem>_figure.png.
Exit code 0 (1 when no reference image is found).
"""
import argparse
import csv
import datetime as _dt
import hashlib
import os
import re
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PKG_ROOT)
try:
    from robot36dec.decode import BLOCKS, H, W            # block definitions of the decoder's own reports
except Exception:                                          # the tool must also work on its own
    H, W = 240, 320
    BLOCKS = {'top_0_20': (0, 20), 'mid_20_62': (20, 62), 'band_62_108': (62, 108), 'low_108_240': (108, 240), 'all': (0, 240)}

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff', '.webp')
CHANNEL_SIGMA_PX = 9.0          # 12 ms Hann window ~ 44 px wide -> horizontal Gaussian sigma ~ 9 px (take2 match scripts)
BAND = 'band_62_108'
SMOOTH_LINES = 5                # 5-line smoothing of the per-line curve (take2_identify_figure.py)
RUN_THRESHOLD = 0.5             # "lines that survive": smoothed per-line NCC above this


# ----------------------------------------------------------------------------------------------------------------------
# images
def load_luma(path, size=(W, H)):
    """Luminance as float64 (H, W); anything that is not 320 x 240 is resampled (full frame, LANCZOS). Returns (array, original size)."""
    from PIL import Image
    with Image.open(path) as im:
        orig = im.size
        g = im.convert('L')
        if g.size != size:
            g = g.resize(size, Image.LANCZOS)
        return np.asarray(g, dtype=np.float64), orig


def channel_blur(ref, sigma_px=CHANNEL_SIGMA_PX, lines=1):
    """The decoder's effective blur applied to a clean reference: horizontal Gaussian (analysis window) and, for a fused
    (3-line) decoded image only, the matching vertical average."""
    out = gaussian_filter1d(ref, sigma=sigma_px, axis=1, mode='nearest') if sigma_px > 0 else ref.copy()
    if lines and lines > 1:
        out = uniform_filter1d(out, size=lines, axis=0, mode='nearest')
    return out


def ncc(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    a = a - a.mean(); b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _pair(dec, ref, a, b, dy=0, dx=0):
    """Decoded rows a..b against reference rows a+dy..b+dy (clipped), horizontally shifted by dx (decoded col c <-> ref col c-dx)."""
    rows = [r for r in range(a, b) if 0 <= r + dy < ref.shape[0]]
    if len(rows) < 4:
        return None, None
    d = dec[rows]
    r = ref[[q + dy for q in rows]]
    if dx > 0:
        d, r = d[:, dx:], r[:, :-dx]
    elif dx < 0:
        d, r = d[:, :dx], r[:, -dx:]
    return d, r


def block_ncc(dec, ref, a, b, dy=0, dx=0):
    d, r = _pair(dec, ref, a, b, dy, dx)
    return float('nan') if d is None else ncc(d, r)


def best_shift(dec, ref, a, b, max_dy, max_dx):
    best = (float('-inf'), 0, 0)
    for dy in range(-max_dy, max_dy + 1):
        for dx in range(-max_dx, max_dx + 1):
            v = block_ncc(dec, ref, a, b, dy, dx)
            if np.isfinite(v) and v > best[0]:
                best = (v, dy, dx)
    return best


def wrong_position_values(dec, ref, a, b):
    """NCC of decoded rows a..b against the reference at positions that cannot be the true one. Blocks shorter than the
    frame: reference rows s..s+n with |s - a| >= n, 2-line steps (take2 null_dist), plus the mirrored reference; the whole
    frame, and any block taller than half the frame: cyclic shifts of 20..220 lines, plus mirrored."""
    n = b - a
    Hf = ref.shape[0]
    vals = []
    if n > Hf - n:          # the block is taller than half the frame (or the whole frame): no non-overlapping position exists
        for dy in range(20, Hf - 20 + 1, 10):
            rolled = np.roll(ref, -dy, axis=0)
            vals.append(ncc(dec[a:b], rolled[a:b]))
            vals.append(ncc(dec[a:b], rolled[a:b, ::-1]))
    else:
        for s in range(0, Hf - n + 1, 2):
            if abs(s - a) < n:
                continue
            vals.append(ncc(dec[a:b], ref[s:s + n]))
            vals.append(ncc(dec[a:b], ref[s:s + n, ::-1]))
    return np.array([v for v in vals if np.isfinite(v)])


def per_line_ncc(dec, ref, dy=0, dx=0):
    """Vectorised row-wise NCC of decoded row r with reference row r+dy (shifted by dx). NaN where no reference row exists."""
    out = np.full(dec.shape[0], np.nan)
    rows = np.array([r for r in range(dec.shape[0]) if 0 <= r + dy < ref.shape[0]])
    if len(rows) == 0:
        return out
    d = dec[rows]
    r = ref[rows + dy]
    if dx > 0:
        d, r = d[:, dx:], r[:, :-dx]
    elif dx < 0:
        d, r = d[:, :dx], r[:, -dx:]
    d = d - d.mean(axis=1, keepdims=True)
    r = r - r.mean(axis=1, keepdims=True)
    den = np.sqrt((d * d).sum(axis=1) * (r * r).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        v = (d * r).sum(axis=1) / den
    v[~np.isfinite(v)] = 0.0
    out[rows] = v
    return out


def per_line_null(dec, refs):
    """Pooled single-line null: every decoded row against reference rows at least 10 lines away (cyclic shifts, 10-line steps)."""
    vals = []
    for ref in refs:
        for dy in range(10, ref.shape[0] - 10 + 1, 10):
            vals.append(per_line_ncc(dec, np.roll(ref, -dy, axis=0)))
    if not vals:
        return np.zeros(0)
    v = np.concatenate(vals)
    return v[np.isfinite(v)]


def smooth(v, n=SMOOTH_LINES):
    return np.convolve(np.nan_to_num(v), np.ones(n) / n, mode='same')


def runs_above(v, thr):
    """[(start, end_inclusive), ...] of consecutive indices where v > thr."""
    idx = np.where(v > thr)[0]
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            out.append((int(start), int(prev))); start = i
        prev = i
    out.append((int(start), int(prev)))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# references and provenance
def slide_label(basename):
    m = re.search(r'slide\s*(\d{1,2})', basename, re.IGNORECASE)
    if m:
        return f'slide{int(m.group(1)):02d}'
    if re.search(r'_other_', basename):
        return 'other'
    return 'unknown'


def collect_references(items):
    paths = []
    for it in items:
        if os.path.isdir(it):
            for name in sorted(os.listdir(it)):
                if name.lower().endswith(IMAGE_EXT):
                    paths.append(os.path.join(it, name))
        elif os.path.isfile(it):
            paths.append(it)
    return paths


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mtime_iso(path):
    return _dt.datetime.fromtimestamp(os.path.getmtime(path), _dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def quoted_params_hash(frozen_md):
    """The sha256 FROZEN.md quotes for params_frozen.json (None if not found)."""
    try:
        with open(frozen_md) as fh:
            for line in fh:
                if 'params_frozen.json' in line:
                    m = re.search(r'`([0-9a-f]{64})`', line)
                    if m:
                        return m.group(1)
    except OSError:
        pass
    return None


def fmt(v, nd=3, sign=True):
    if v is None or not np.isfinite(v):
        return 'n/a'
    return f'{v:+.{nd}f}' if sign else f'{v:.{nd}f}'


def stats(arr):
    arr = np.asarray(arr, np.float64)
    if arr.size == 0:
        return dict(n=0, mean=float('nan'), sd=float('nan'), p95=float('nan'), max=float('nan'))
    return dict(n=int(arr.size), mean=float(arr.mean()), sd=float(arr.std()), p95=float(np.percentile(arr, 95)), max=float(arr.max()))


def parse_named(items, allow_fused=False):
    """'NAME=PATH[:fused]' -> [(name, path, fused)]."""
    out = []
    for it in items or []:
        if '=' not in it:
            raise SystemExit(f'expected NAME=PATH, got {it!r}')
        name, path = it.split('=', 1)
        fused = False
        if allow_fused and path.endswith(':fused'):
            path, fused = path[:-len(':fused')], True
        out.append((name, path, fused))
    return out


def parse_prior(items):
    out = []
    for it in items or []:
        label, vals = it.split('=', 1)
        nums = [float(x) for x in vals.split(',')]
        out.append((label, nums))
    return out


# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('decoded', help='<base>_luma.png written by the decoder (grayscale 320x240)')
    ap.add_argument('references', nargs='+', help='reference slide images or directories holding them')
    ap.add_argument('--out', default='VALIDATION.md')
    ap.add_argument('--label', default=None, help='name of the primary decoded image in the report (default: its file name)')
    ap.add_argument('--fused', action='store_true', help='the primary decoded image is a 3-line fused display (3-line average on the references)')
    ap.add_argument('--extra', action='append', default=[], metavar='NAME=PATH[:fused]', help='further decoded images reported in the same tables')
    ap.add_argument('--control', action='append', default=[], metavar='NAME=PATH', help="the decoder's pure-noise control decodes (no-picture null)")
    ap.add_argument('--control-luma', default=None, help='(compatibility) one control decode, named "control"')
    ap.add_argument('--max-vshift', type=int, default=3)
    ap.add_argument('--max-hshift', type=int, default=4)
    ap.add_argument('--sigma-px', type=float, default=CHANNEL_SIGMA_PX, help='horizontal channel blur of the references, Gaussian sigma in px (0 = none)')
    ap.add_argument('--frozen', default=os.path.join(PKG_ROOT, 'FROZEN.md'))
    ap.add_argument('--params-frozen', default=os.path.join(PKG_ROOT, 'robot36dec', 'params_frozen.json'))
    ap.add_argument('--prior', action='append', default=[], metavar='LABEL=v0_20,v20_62,v62_108,v108_240[,vall]',
                    help='earlier per-block numbers to list next to the new ones (comparison rows only)')
    ap.add_argument('--aux-dir', default=None, help='directory for the CSV files and the figure (default: next to --out)')
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--exclude', action='append', default=[], metavar='REGEX', help='drop reference files whose base name matches (re.search)')
    ap.add_argument('--primary-ref', default=None, metavar='SUBSTRING', help='reference used for the per-block tables and the per-line curve (default: top-ranked known slide)')
    ap.add_argument('--report-ref', action='append', default=[], metavar='SUBSTRING', help='further reference copies listed in a per-block table')
    ap.add_argument('--preamble', default=None, metavar='FILE.md', help='Markdown inserted after the title (a hand-written summary); the numbers below it are always regenerated')
    args = ap.parse_args(argv)
    run_started = _dt.datetime.now(_dt.timezone.utc)
    t_start = time.time()

    aux_dir = args.aux_dir or (os.path.dirname(os.path.abspath(args.out)) or '.')
    os.makedirs(aux_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.out))[0]
    csv_lines_path = os.path.join(aux_dir, stem + '_lines.csv')
    csv_rank_path = os.path.join(aux_dir, stem + '_ranking.csv')
    fig_path = os.path.join(aux_dir, stem + '_figure.png')

    # ---- decoded images (primary + extras) and controls
    decoded = []   # (label, path, fused, array)
    arr, _ = load_luma(args.decoded)
    decoded.append((args.label or os.path.basename(args.decoded), args.decoded, args.fused, arr))
    for name, path, fused in parse_named(args.extra, allow_fused=True):
        decoded.append((name, path, fused, load_luma(path)[0]))
    controls = []
    ctrl_items = parse_named(args.control)
    if args.control_luma:
        ctrl_items.append(('control', args.control_luma, False))
    for name, path, _ in ctrl_items:
        controls.append((name, path, load_luma(path)[0]))

    # ---- references
    ref_paths = collect_references(args.references)
    excluded = []
    if args.exclude:
        keep = []
        for p in ref_paths:
            if any(re.search(pat, os.path.basename(p)) for pat in args.exclude):
                excluded.append(os.path.basename(p))
            else:
                keep.append(p)
        ref_paths = keep
    if not ref_paths:
        print('no reference images found', file=sys.stderr)
        return 1
    refs = []
    for p in ref_paths:
        raw, orig = load_luma(p)
        refs.append(dict(path=p, base=os.path.basename(p), group=slide_label(os.path.basename(p)), orig=orig, raw=raw,
                         blur1=channel_blur(raw, args.sigma_px, 1), blur3=channel_blur(raw, args.sigma_px, 3)))
    known_slides = sorted({r['group'] for r in refs if r['group'].startswith('slide')})

    def find_ref(sub):
        hits = [i for i, r in enumerate(refs) if sub in r['base']]
        if len(hits) != 1:
            raise SystemExit(f'--primary-ref/--report-ref {sub!r} matches {len(hits)} references (need exactly 1)')
        return hits[0]
    i_primary_fixed = find_ref(args.primary_ref) if args.primary_ref else None
    report_idx = [find_ref(sub) for sub in args.report_ref]

    def ref_img(r, fused):
        return r['blur3'] if fused else r['blur1']

    # ---- provenance / freeze check
    prov = []
    frozen_ok = os.path.isfile(args.frozen)
    frozen_mtime = os.path.getmtime(args.frozen) if frozen_ok else None
    params_ok = os.path.isfile(args.params_frozen)
    params_sha = sha256_of(args.params_frozen) if params_ok else None
    quoted = quoted_params_hash(args.frozen) if frozen_ok else None
    prov.append(f'* this run started {run_started.strftime("%Y-%m-%dT%H:%M:%SZ")} (UTC)')
    if frozen_ok:
        prov.append(f'* `{os.path.relpath(args.frozen, PKG_ROOT)}` last modified {mtime_iso(args.frozen)} -> '
                    + ('**predates this run**' if frozen_mtime < run_started.timestamp() else '**NEWER than this run (freeze violated?)**'))
    else:
        prov.append(f'* FROZEN.md not found at `{args.frozen}` -> **the decoder is not frozen; this comparison is not a valid step 4**')
    if params_ok:
        match = (quoted is not None and quoted == params_sha)
        prov.append(f'* `robot36dec/params_frozen.json` last modified {mtime_iso(args.params_frozen)}, sha256 `{params_sha}` -> '
                    + ('matches the hash quoted in FROZEN.md' if match else ('**does not match** the hash quoted in FROZEN.md' if quoted else 'FROZEN.md quotes no hash')))
    else:
        prov.append('* `robot36dec/params_frozen.json` not found')
    for label, path, fused, _ in decoded:
        rel = os.path.relpath(path, PKG_ROOT)
        when = mtime_iso(path)
        note = ''
        if frozen_ok:
            note = ' (written before FROZEN.md; parameters identical to the frozen set by hash)' if os.path.getmtime(path) < frozen_mtime else ' (written after the freeze)'
        prov.append(f'* decoded `{label}` = `{rel}`{" [fused: 3-line average on the references]" if fused else ""}, last modified {when}{note}, sha256 `{sha256_of(path)[:16]}...`')
    for name, path, _ in controls:
        prov.append(f'* control `{name}` = `{os.path.relpath(path, PKG_ROOT)}`, last modified {mtime_iso(path)}')
    prov.append(f'* {len(refs)} reference images from {", ".join("`" + os.path.relpath(x, PKG_ROOT) + "`" for x in args.references)}; '
                f'slide labels parsed from the file names: {len(known_slides)} known slides '
                f'({", ".join(known_slides)}), {sum(r["group"] == "unknown" for r in refs)} received images of unknown slide, '
                f'{sum(r["group"] == "other" for r in refs)} non-slide images (waterfalls, composites, graphics) kept as distractors')
    if excluded:
        prov.append(f'* {len(excluded)} files excluded by --exclude {args.exclude}: ' + ', '.join(f'`{x}`' for x in excluded))
    if i_primary_fixed is not None:
        prov.append(f'* primary reference fixed a priori: `{refs[i_primary_fixed]["base"]}` (per-block tables and per-line curves); the ranking is unaffected')

    # ---- ranking and per-block statistics, per decoded image
    results = {}
    rank_rows = []
    for label, path, fused, dec in decoded:
        all_scores = np.array([block_ncc(dec, ref_img(r, fused), 0, H) for r in refs])
        band_scores = np.array([block_ncc(dec, ref_img(r, fused), *BLOCKS[BAND]) for r in refs])
        order = np.argsort(-all_scores)
        i_top = int(order[0])
        known_order = [int(i) for i in order if refs[i]['group'].startswith('slide')]
        i_best_known = known_order[0] if known_order else i_top
        best_group = refs[i_best_known]['group']
        i_best = i_primary_fixed if i_primary_fixed is not None else i_best_known
        # best per slide group
        group_best = {}
        for i in order:
            g = refs[i]['group']
            if g not in group_best:
                group_best[g] = int(i)
        other_known = [i for g, i in group_best.items() if g.startswith('slide') and g != best_group]
        runner_all = max(other_known, key=lambda i: all_scores[i]) if other_known else None
        runner_band = max(other_known, key=lambda i: band_scores[i]) if other_known else None
        # per block
        blocks = {}
        same_slide_idx = [i for i, r in enumerate(refs) if r['group'] == best_group]
        wrong_slide_idx = [i for i, r in enumerate(refs) if r['group'].startswith('slide') and r['group'] != best_group]
        slide_idx = [i for i, r in enumerate(refs) if r['group'].startswith('slide')]
        for name, (a, b) in BLOCKS.items():
            rb = ref_img(refs[i_best], fused)
            v0 = block_ncc(dec, rb, a, b)
            vb, dy, dx = best_shift(dec, rb, a, b, args.max_vshift, args.max_hshift)
            ws = np.array([block_ncc(dec, ref_img(refs[i], fused), a, b) for i in wrong_slide_idx])
            ws_best_i = wrong_slide_idx[int(np.argmax(ws))] if len(ws) else None
            wp_self = wrong_position_values(dec, rb, a, b)
            wp_pool = np.concatenate([wrong_position_values(dec, ref_img(refs[i], fused), a, b) for i in slide_idx]) if slide_idx else np.zeros(0)
            same = np.array([block_ncc(dec, ref_img(refs[i], fused), a, b) for i in same_slide_idx])
            s_self, s_pool, s_ws = stats(wp_self), stats(wp_pool), stats(ws)
            ctrl_vals = []
            for cname, cpath, carr in controls:
                c0 = block_ncc(carr, rb, a, b)
                cb = best_shift(carr, rb, a, b, args.max_vshift, args.max_hshift)[0]
                ctrl_vals.append((cname, c0, cb))
            blocks[name] = dict(a=a, b=b, ncc0=v0, ncc_best=vb, dy=dy, dx=dx,
                                wrong_slide=s_ws, wrong_slide_best=(refs[ws_best_i]['base'] if ws_best_i is not None else None),
                                wp_self=s_self, wp_pool=s_pool,
                                z_self=((v0 - s_self['mean']) / s_self['sd']) if s_self['n'] and s_self['sd'] > 0 else float('nan'),
                                z_pool=((v0 - s_pool['mean']) / s_pool['sd']) if s_pool['n'] and s_pool['sd'] > 0 else float('nan'),
                                same_slide=stats(same), same_slide_min=(float(same.min()) if len(same) else float('nan')),
                                controls=ctrl_vals)
        # per-line curve at the whole-frame best shift
        vb_all, dy_all, dx_all = best_shift(dec, ref_img(refs[i_best], fused), 0, H, args.max_vshift, args.max_hshift)
        curve = per_line_ncc(dec, ref_img(refs[i_best], fused), dy_all, dx_all)
        null_lines = per_line_null(dec, [ref_img(refs[i], fused) for i in slide_idx])
        p95_line = float(np.percentile(null_lines, 95)) if len(null_lines) else float('nan')
        ctrl_curves = [(cname, per_line_ncc(carr, ref_img(refs[i_best], fused))) for cname, cpath, carr in controls]
        report_rows = []
        for i in report_idx:
            report_rows.append((refs[i]['base'], refs[i]['group'], refs[i]['orig'],
                                {n: block_ncc(dec, ref_img(refs[i], fused), a, b) for n, (a, b) in BLOCKS.items()}))
        results[label] = dict(path=path, fused=fused, dec=dec, all_scores=all_scores, band_scores=band_scores, order=order, i_best=i_best,
                              i_top=i_top, i_best_known=i_best_known, report_rows=report_rows,
                              best_group=best_group, group_best=group_best, runner_all=runner_all, runner_band=runner_band,
                              blocks=blocks, curve=curve, dy_all=dy_all, dx_all=dx_all, p95_line=p95_line, null_lines=null_lines,
                              ctrl_curves=ctrl_curves)
        for rank, i in enumerate(order):
            rank_rows.append([label, rank + 1, refs[i]['base'], refs[i]['group'], f'{all_scores[i]:.4f}', f'{band_scores[i]:.4f}'])

    # ---- report
    L = []
    L.append(f'# Blind validation of the frozen robot36dec decode against the Series 33 reference slides\n')
    L.append('The decoder ran without any access to these images (SPEC.md rule 2; FROZEN.md); this comparison was made afterwards '
             '(SPEC.md section 5 step 4) and scores the finished decode only. Nothing in this report was available to the decoder, '
             'and no number here feeds back into it.\n')
    if args.preamble:
        with open(args.preamble) as fh:
            L.append(fh.read().rstrip() + '\n')
    L.append('## Provenance and freeze check\n')
    L.extend(prov)
    L.append('')
    L.append('## Method\n')
    L.append(f'* References: luminance, resampled to the Robot 36 frame ({W} x {H}, full frame, LANCZOS) and passed through the decoder\'s '
             f'channel blur, a horizontal Gaussian of sigma {args.sigma_px:g} px (the 12 ms analysis window spans ~44 px); a 3-line '
             'vertical average is added only when the decoded image is a 3-line fused display. The decoded images are used as written.')
    L.append('* Statistic: Pearson correlation (NCC of mean-removed blocks) per line block '
             + ', '.join(f'{n} = lines {a}-{b}' for n, (a, b) in BLOCKS.items())
             + f', at zero shift and at the best small shift (|dy| <= {args.max_vshift} lines, |dx| <= {args.max_hshift} px).')
    L.append('* Ranking: references sorted by the whole-frame NCC at zero shift; the best-matching slide is the slide label of the top '
             'reference; the margin is the gap to the best reference of any other known slide.')
    L.append('* Nulls: (a) wrong slide = every reference of another known slide at zero shift; (b) wrong position = the decoded block '
             'against the best reference, and pooled against every slide reference, at non-overlapping vertical positions (|shift| >= '
             'block height, 2-line steps, plus the horizontally mirrored reference; whole frame and blocks taller than half the frame: cyclic shifts 20-220 lines, plus mirrored); '
             '(c) control = the decoder\'s own pure-noise decodes against the best reference. z = (NCC at zero shift - null mean) / null sd.')
    L.append(f'* Per-line curve: NCC of each decoded line with the best reference at the whole-frame best shift; single-line null = pooled '
             f'wrong positions (cyclic shifts >= 10 lines) of all slide references; "surviving lines" = {SMOOTH_LINES}-line smoothed NCC > {RUN_THRESHOLD}.')
    L.append('')

    for label, R in results.items():
        dec = R['dec']; i_best = R['i_best']; rb = refs[i_best]
        L.append(f'## Decoded image `{label}`' + (' (3-line fused; references 3-line averaged)' if R['fused'] else '') + '\n')
        L.append(f'`{os.path.relpath(R["path"], PKG_ROOT)}`\n')
        # ranking
        L.append('### Reference ranking by whole-frame NCC (zero shift), top 15 and the best copy of every other slide\n')
        L.append('| rank | reference | slide | NCC all | NCC band 62-108 |\n|---|---|---|---|---|')
        shown = set()
        for rank, i in enumerate(R['order'][:15]):
            shown.add(int(i))
            L.append(f'| {rank + 1} | `{refs[i]["base"]}` | {refs[i]["group"]} | {fmt(R["all_scores"][i])} | {fmt(R["band_scores"][i])} |'
                     + ('  <- best' if i == i_best else ''))
        others = [(g, i) for g, i in R['group_best'].items() if int(i) not in shown]
        for g, i in sorted(others, key=lambda gi: -R['all_scores'][gi[1]]):
            rank = int(np.where(R['order'] == i)[0][0]) + 1
            L.append(f'| {rank} | `{refs[i]["base"]}` | {g} | {fmt(R["all_scores"][i])} | {fmt(R["band_scores"][i])} |')
        L.append('')
        n_same = sum(1 for r in refs if r['group'] == R['best_group'])
        top_groups = [refs[i]['group'] for i in R['order'][:n_same]]
        n_top_same = sum(1 for g in top_groups if g == R['best_group'])
        rk = refs[R['i_best_known']]
        L.append(f'**Best-matching slide: {R["best_group"]}** (top-ranked known slide: `{rk["base"]}`, whole-frame NCC {fmt(R["all_scores"][R["i_best_known"]])}'
                 + (f'; the overall top rank is `{refs[R["i_top"]]["base"]}` ({refs[R["i_top"]]["group"]})' if R['i_top'] != R['i_best_known'] else '')
                 + f'; {n_top_same} of the top {n_same} ranks are copies of {R["best_group"]}, out of {n_same} copies in the set).')
        L.append(f'Primary reference for the block and line statistics below: `{rb["base"]}` (original size {rb["orig"][0]}x{rb["orig"][1]}'
                 + (', fixed a priori)' if i_primary_fixed is not None else ', the top-ranked known slide)') + '.')
        if R['runner_all'] is not None:
            ra, rbnd = R['runner_all'], R['runner_band']
            L.append(f'Margin over the best copy of any other slide: whole frame {fmt(R["all_scores"][i_best])} vs {fmt(R["all_scores"][ra])} '
                     f'(`{refs[ra]["base"]}`, {refs[ra]["group"]}) = **{fmt(R["all_scores"][i_best] - R["all_scores"][ra])}**; '
                     f'band 62-108 {fmt(R["band_scores"][i_best])} vs {fmt(R["band_scores"][rbnd])} (`{refs[rbnd]["base"]}`, {refs[rbnd]["group"]}) '
                     f'= **{fmt(R["band_scores"][i_best] - R["band_scores"][rbnd])}**.')
        L.append('')
        # per block
        L.append(f'### NCC per line block with the best reference `{rb["base"]}` and the null distributions\n')
        hdr = '| block | lines | NCC zero shift | best NCC (dy, dx) | same-slide copies min / mean (n) | wrong slide mean / sd / max (n) | wrong position, best ref: mean / sd / 95th / max (n) | z (self) | wrong position, pooled: mean / sd / 95th / max (n) | z (pooled) |'
        sep = '|---|---|---|---|---|---|---|---|---|---|'
        for cname, _, _ in controls:
            hdr += f' control `{cname}` zero / best shift |'; sep += '---|'
        L.append(hdr); L.append(sep)
        for name, B in R['blocks'].items():
            ws, s1, s2, ss = B['wrong_slide'], B['wp_self'], B['wp_pool'], B['same_slide']
            row = (f'| {name} | {B["a"]}-{B["b"]} | {fmt(B["ncc0"])} | {fmt(B["ncc_best"])} ({B["dy"]:+d}, {B["dx"]:+d}) | '
                   f'{fmt(B["same_slide_min"])} / {fmt(ss["mean"])} ({ss["n"]}) | '
                   f'{fmt(ws["mean"])} / {fmt(ws["sd"], sign=False)} / {fmt(ws["max"])} ({ws["n"]}) | '
                   f'{fmt(s1["mean"])} / {fmt(s1["sd"], sign=False)} / {fmt(s1["p95"])} / {fmt(s1["max"])} ({s1["n"]}) | {fmt(B["z_self"], 1)} | '
                   f'{fmt(s2["mean"])} / {fmt(s2["sd"], sign=False)} / {fmt(s2["p95"])} / {fmt(s2["max"])} ({s2["n"]}) | {fmt(B["z_pool"], 1)} |')
            for cname, c0, cb in B['controls']:
                row += f' {fmt(c0)} / {fmt(cb)} |'
            L.append(row)
        L.append('')
        wsb = {n: B['wrong_slide_best'] for n, B in R['blocks'].items()}
        L.append('Strongest wrong-slide reference per block: ' + '; '.join(f'{n}: `{v}`' for n, v in wsb.items() if v) + '.\n')
        if R['report_rows']:
            L.append('### NCC per line block (zero shift) with the named reference copies\n')
            L.append('| reference | slide | original size | ' + ' | '.join(f'{BLOCKS[n][0]}-{BLOCKS[n][1]}' for n in BLOCKS) + ' |')
            L.append('|---|---|---|' + '---|' * len(BLOCKS))
            for base, group, orig, vals in R['report_rows']:
                L.append(f'| `{base}` | {group} | {orig[0]}x{orig[1]} | ' + ' | '.join(fmt(vals[n]) for n in BLOCKS) + ' |')
            L.append('')
        # per line
        curve, p95 = R['curve'], R['p95_line']
        L.append(f'### Per-line NCC against `{rb["base"]}` (whole-frame best shift dy {R["dy_all"]:+d}, dx {R["dx_all"]:+d}; '
                 f'single-line null 95th percentile {fmt(p95)}, n = {len(R["null_lines"])})\n')
        L.append('| block | lines | mean line NCC | median | fraction of lines above the null 95th |'
                 + ''.join(f' control `{c}` mean |' for c, _ in R['ctrl_curves']))
        L.append('|---|---|---|---|---|' + '---|' * len(R['ctrl_curves']))
        for name, (a, b) in BLOCKS.items():
            seg = curve[a:b]; seg = seg[np.isfinite(seg)]
            frac = float(np.mean(seg > p95)) if len(seg) else float('nan')
            row = f'| {name} | {a}-{b} | {fmt(seg.mean() if len(seg) else float("nan"))} | {fmt(np.median(seg) if len(seg) else float("nan"))} | {fmt(frac, 2, sign=False)} |'
            for c, cc in R['ctrl_curves']:
                cs = cc[a:b]; cs = cs[np.isfinite(cs)]
                row += f' {fmt(cs.mean() if len(cs) else float("nan"))} |'
            L.append(row)
        sm = smooth(curve)
        rr = runs_above(sm, RUN_THRESHOLD)
        n_lines = sum(e - s + 1 for s, e in rr)
        L.append('')
        L.append(f'Lines whose {SMOOTH_LINES}-line-smoothed NCC exceeds {RUN_THRESHOLD}: ' + (', '.join(f'{s}-{e}' for s, e in rr) if rr else 'none')
                 + f' ({n_lines} lines).')
        for c, cc in R['ctrl_curves']:
            rc = runs_above(smooth(cc), RUN_THRESHOLD)
            L.append(f'Same statistic for control `{c}`: ' + (', '.join(f'{s}-{e}' for s, e in rc) if rc else 'none') + f' ({sum(e - s + 1 for s, e in rc)} lines).')
        L.append('')

    # ---- comparison with earlier numbers
    priors = parse_prior(args.prior)
    if priors or len(results) > 1:
        L.append('## Comparison across decodes' + (' and with the earlier (take2) numbers' if priors else '') + '\n')
        names = list(BLOCKS.keys())
        L.append('| decode | ' + ' | '.join(f'{BLOCKS[n][0]}-{BLOCKS[n][1]}' for n in names) + ' |')
        L.append('|---|' + '---|' * len(names))
        for label, R in results.items():
            L.append(f'| `{label}` (zero shift vs `{refs[R["i_best"]]["base"]}`) | ' + ' | '.join(fmt(R['blocks'][n]['ncc0']) for n in names) + ' |')
        for label, nums in priors:
            cells = [f'{v:+.2f}' for v in nums] + ['n/a'] * (len(names) - len(nums))
            L.append(f'| {label} | ' + ' | '.join(cells[:len(names)]) + ' |')
        L.append('')
        if priors:
            L.append('The earlier rows are quoted from the analysis report (same statistic: zero shift, reference `109_*BI8SWK*` with the '
                     'horizontal channel blur, 3-line average only for fused images); they are listed for orientation and were not used anywhere.\n')

    L.append('## Interpretation\n')
    L.append('A block whose NCC exceeds the wrong-position null by several sigma and the wrong-slide maximum carries the reference '
             'picture; a block within the null is noise or an unrelated placement. The pure-noise control decodes show what the '
             'identical pipeline produces with no picture at all. Nothing in this report was available to the decoder.\n')
    L.append('## Files\n')
    L.append(f'* per-line curves: `{os.path.relpath(csv_lines_path, PKG_ROOT)}`')
    L.append(f'* full reference ranking: `{os.path.relpath(csv_rank_path, PKG_ROOT)}`')
    if not args.no_figure:
        L.append(f'* figure: `{os.path.relpath(fig_path, PKG_ROOT)}`')
    L.append('')
    L.append('## Command\n')
    L.append('```\n' + ' '.join(['python', os.path.relpath(os.path.abspath(__file__), PKG_ROOT)] + [a if ' ' not in a else repr(a) for a in (argv if argv is not None else sys.argv[1:])]) + '\n```\n')

    # ---- CSV
    with open(csv_lines_path, 'w', newline='') as fh:
        wr = csv.writer(fh)
        head = ['line']
        for label, R in results.items():
            head.append(f'ncc_{label}')
            head.append(f'ncc_{label}_smoothed{SMOOTH_LINES}')
        for label, R in results.items():
            for c, _ in R['ctrl_curves']:
                head.append(f'ncc_control_{c}_vs_bestref_of_{label}')
        head += [f'null_p95_line_{label}' for label in results]
        wr.writerow(head)
        for r in range(H):
            row = [r]
            for label, R in results.items():
                v = R['curve'][r]; row.append('' if not np.isfinite(v) else f'{v:.4f}')
                row.append(f'{smooth(R["curve"])[r]:.4f}')
            for label, R in results.items():
                for c, cc in R['ctrl_curves']:
                    v = cc[r]; row.append('' if not np.isfinite(v) else f'{v:.4f}')
            row += [f'{R["p95_line"]:.4f}' for R in results.values()]
            wr.writerow(row)
    with open(csv_rank_path, 'w', newline='') as fh:
        wr = csv.writer(fh); wr.writerow(['decoded', 'rank', 'reference', 'slide', 'ncc_all_zero_shift', 'ncc_band_62_108_zero_shift'])
        wr.writerows(rank_rows)

    # ---- figure
    if not args.no_figure:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            prim = next(iter(results.values()))
            rb = refs[prim['i_best']]
            n_dec = len(results)
            ncol = 2 + n_dec
            fig = plt.figure(figsize=(4.2 * ncol, 13))
            gs = fig.add_gridspec(3, ncol, height_ratios=[1.0, 0.7, 0.8])
            ax = fig.add_subplot(gs[0, 0]); ax.imshow(rb['raw'], cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'best reference ({rb["group"]})\n{rb["base"][:38]}', fontsize=9)
            ax = fig.add_subplot(gs[0, 1]); ax.imshow(ref_img(rb, prim['fused']), cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'reference through the channel blur\n(sigma {args.sigma_px:g} px horizontal' + (', 3-line avg)' if prim['fused'] else ')'), fontsize=9)
            for k, (label, R) in enumerate(results.items()):
                ax = fig.add_subplot(gs[0, 2 + k]); ax.imshow(R['dec'], cmap='gray', vmin=0, vmax=255)
                B = R['blocks']
                ax.set_title(f'{label}\nNCC band {B[BAND]["ncc0"]:.2f}, all {B["all"]["ncc0"]:.2f}', fontsize=9)
            a, b = BLOCKS[BAND]
            lo, hi = max(0, a - 6), min(H, b + 4)
            crops = [(f'reference (blurred), lines {lo}-{hi}', ref_img(rb, prim['fused'])[lo:hi])]
            crops += [(f'{label}, lines {lo}-{hi}', R['dec'][lo:hi]) for label, R in results.items()]
            if controls:
                crops.append((f'control {controls[0][0]}, lines {lo}-{hi}', controls[0][2][lo:hi]))
            for k, (title, img) in enumerate(crops[:ncol]):
                ax = fig.add_subplot(gs[1, k]); ax.imshow(img, cmap='gray', vmin=0, vmax=255, aspect='auto'); ax.set_title(title, fontsize=9)
            for a_ in fig.axes:
                a_.set_xticks([]); a_.set_yticks([])
            ax = fig.add_subplot(gs[2, :])
            for label, R in results.items():
                ax.plot(smooth(R['curve']), label=f'{label} ({SMOOTH_LINES}-line smoothed)', lw=1.4)
            if prim['ctrl_curves']:
                c, cc = prim['ctrl_curves'][0]
                ax.plot(smooth(cc), label=f'control {c}', color='gray', alpha=0.7, lw=1.0)
            ax.axhline(prim['p95_line'], color='r', ls='--', lw=0.8, label=f'single-line null 95th ({prim["p95_line"]:+.2f})')
            ax.axhline(0, color='k', lw=0.5)
            for name, (a, b) in BLOCKS.items():
                if name != 'all':
                    ax.axvline(a, color='k', lw=0.4, alpha=0.4)
            ax.set_xlim(0, H - 1); ax.set_ylim(-0.6, 1.0)
            ax.set_xlabel('Robot 36 line'); ax.set_ylabel('per-line NCC with the best reference')
            ax.set_title('Which lines of the slide are in the decode (blind comparison, after the freeze)', fontsize=10)
            ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
            plt.tight_layout(); plt.savefig(fig_path, dpi=90); plt.close(fig)
        except Exception as ex:  # the figure is optional
            L.append(f'(figure not written: {ex})\n')

    L.append(f'_Runtime {time.time() - t_start:.1f} s._\n')
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(L))
    print('\n'.join(L))
    print(f'\nwritten: {args.out}, {csv_lines_path}, {csv_rank_path}' + ('' if args.no_figure else f', {fig_path}'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
