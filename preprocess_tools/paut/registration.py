"""PAUT <-> XCT in-plane registration from the centres of the through-holes.

Same idea and the same stored convention as the registrations already in the database
(``preprocess_tools.register.register_ut_xct_monoelement``: hole centres in UT and XCT, rigid
body transformation), adapted to the processed PAUT volumes:

* UT holes: A-scans without entrance echo inside the part. The processed PAUT has the
  entrance echo at a known depth (``pre`` samples into the TIFF), so the map of the echo
  amplitude there shows the part bright and the through-holes dark. Holes are the dark
  regions enclosed by the part, with sub-pixel centroids, in mm.
* XCT holes: air columns through the thickness. Mean over the middle of the thickness,
  Otsu, enclosed dark regions, centroids in mm.
* Matching: every assignment of the XCT holes to UT holes, rotation + translation fitted by
  least squares (Kabsch), with and without a mirror; the smallest RMS residual wins. A mirror
  is only used when it fits clearly better (the PAUT index/scan axes and the XCT axes can
  have opposite handedness).

Stored matrix (``registrations.registration_matrix``), as in the database: a 3x3 matrix ``M``
used with ``scipy.ndimage.affine_transform`` on each XCT (Y, X) slice to produce the XCT in
the UT footprint at XCT resolution:

    [row_xct_px, col_xct_px, 1]^T = M @ [row_ut_mm / xct_voxel, col_ut_mm / xct_voxel, 1]^T

with ``row_ut_mm = index * index_spacing`` and ``col_ut_mm = scan * scan_spacing`` of the
processed PAUT TIFF (Y = index, X = scan) and pixel indices of the XCT TIFF (Z, Y, X).
The reference measurement is the UT, the registered measurement the XCT (as in the database).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.signal import hilbert
from skimage.filters import threshold_otsu

REGISTRATION_VERSION = 1
REGISTRATION_TYPE = (
    "PAUT hole-centre registration 2026 v1, UTvsXCT-preprocessing, file preprocess_tools/paut/registration.py "
    "function register. Extract the centres of the through-holes from the PAUT entrance-echo amplitude map and "
    "from the XCT material projection, and register them using a rigid body transformation (rotation and "
    "translation; a mirror only when the axes have opposite handedness)")
CONVENTION = (
    "registration_matrix M is used with scipy.ndimage.affine_transform on each XCT (Y, X) slice: "
    "[row_xct_px, col_xct_px, 1] = M @ [row_ut_mm / xct_voxel_mm, col_ut_mm / xct_voxel_mm, 1], "
    "row_ut_mm = index * index_spacing_mm, col_ut_mm = scan * scan_spacing_mm of the processed PAUT TIFF "
    "(ZYX, Y = index, X = scan), XCT pixel indices of the XCT TIFF (ZYX). Reference = UT, registered = XCT.")


# ----------------------------------------------------------------------------- holes

def _holes_from_bright_part(img, spacing_rc, min_area_mm2, smooth_px=0.0):
    """Dark regions enclosed by the bright part in a 2D map. Returns (holes, part_mask, enclosed).

    holes: list of dicts with centre (row_mm, col_mm), area_mm2, equivalent diameter, contrast.
    Centroids are weighted by darkness, so they are sub-pixel.
    """
    a = np.asarray(img, np.float64)
    if smooth_px:
        a = ndimage.gaussian_filter(a, smooth_px)
    part = a > threshold_otsu(a)
    lab, n = ndimage.label(part)
    if n == 0:
        return [], part, np.zeros_like(part)
    sizes = ndimage.sum(np.ones_like(a), lab, range(1, n + 1))
    part = lab == (1 + int(np.argmax(sizes)))
    filled = ndimage.binary_fill_holes(part)
    enclosed = filled & ~part
    lab, n = ndimage.label(enclosed)
    px_area = spacing_rc[0] * spacing_rc[1]
    level = float(np.median(a[part]))
    holes = []
    for k in range(1, n + 1):
        m = lab == k
        area = m.sum() * px_area
        if area < min_area_mm2:
            continue
        w = np.clip(level - a, 0, None) * m
        if w.sum() <= 0:
            w = m.astype(float)
        r, c = ndimage.center_of_mass(w)
        holes.append({'row_mm': float(r * spacing_rc[0]), 'col_mm': float(c * spacing_rc[1]),
                      'area_mm2': float(area), 'diameter_mm': float(2 * np.sqrt(area / np.pi)),
                      'contrast': float(1 - a[m].mean() / level) if level else 0.0})
    holes.sort(key=lambda h: -h['area_mm2'])
    return holes, part, enclosed


@dataclass
class UTHoles:
    holes: list
    echo_map: np.ndarray            # (index, scan) entrance-echo amplitude
    part: np.ndarray
    spacing_rc: tuple               # (index, scan) mm
    echo_index: int                 # sample of the entrance echo in the TIFF
    depth_spacing: float
    shape_zyx: tuple
    cscan: np.ndarray = None        # amplitude below the echo, for display


def ut_holes(tif, record=None, window_mm=(-0.3, 0.3), below_mm=(0.5, 3.0), min_area_mm2=4.0):
    """Hole centres (mm) in a processed PAUT TIFF from its entrance-echo amplitude map."""
    tif = Path(tif)
    record = record or json.loads(tif.with_name(tif.stem + '_processing.json').read_text())
    sp_i, sp_s, dz = record['output']['spacing_mm_index_scan_depth']
    lo = record['crop']['source_start_mm']['depth']           # depth of the first stored sample (mm, echo = 0)
    k0 = int(round(-lo / dz))                                  # stored sample of the entrance echo
    vol = tifffile.imread(tif).astype(np.float32)              # (Z, index, scan)
    env = np.abs(hilbert(vol - vol.mean(0, keepdims=True), axis=0))
    a, b = (max(0, k0 + int(round(w / dz))) for w in window_mm)
    echo = env[a:b + 1].max(0)
    c0, c1 = (min(env.shape[0] - 1, k0 + int(round(w / dz))) for w in below_mm)
    cscan = env[c0:c1 + 1].max(0)
    holes, part, _ = _holes_from_bright_part(echo, (sp_i, sp_s), min_area_mm2, smooth_px=0.5)
    return UTHoles(holes, echo, part, (sp_i, sp_s), k0, dz, vol.shape, cscan)


@dataclass
class XCTHoles:
    holes: list
    projection: np.ndarray          # mean over the middle of the thickness, at ``step`` px
    part: np.ndarray
    voxel_mm: float
    step: int
    shape_zyx: tuple
    top_surface_slice: float
    slices: tuple
    bottom_surface_slice: float = None


def xct_holes(tif, voxel_mm=0.025, step=4, min_area_mm2=4.0, middle=(0.3, 0.7)):
    """Hole centres (mm) in an XCT TIFF (Z, Y, X): air columns through the thickness."""
    try:
        vol = tifffile.memmap(str(tif), mode='r')
    except Exception:
        vol = tifffile.imread(str(tif))
    nz = vol.shape[0]
    # material depth range: slices whose mean (on a sparse grid) is above halfway between air and material
    prof = np.array([np.asarray(vol[k, ::16, ::16], np.float32).mean() for k in range(nz)])
    thr = 0.5 * (prof.max() + prof.min())
    inside = np.flatnonzero(prof > thr)
    z0, z1 = (int(inside[0]), int(inside[-1])) if inside.size else (0, nz - 1)
    a, b = int(z0 + middle[0] * (z1 - z0)), int(z0 + middle[1] * (z1 - z0))
    acc = None
    for k in range(a, b + 1):
        s = np.asarray(vol[k], np.float32)
        acc = s if acc is None else acc + s
    proj_full = acc / (b - a + 1)
    proj = proj_full[::step, ::step]
    holes, part, _ = _holes_from_bright_part(proj_full, (voxel_mm, voxel_mm), min_area_mm2, smooth_px=2.0)
    return XCTHoles(holes, proj, part[::step, ::step], voxel_mm, step, tuple(vol.shape), float(z0), (a, b),
                    float(z1))


# ----------------------------------------------------------------------------- fit

def _kabsch(src, dst, reflect):
    """Least-squares orthogonal R and t with dst ≈ R @ src + t, det(R) = -1 when ``reflect`` else +1."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    want = -1.0 if reflect else 1.0
    d = want * np.sign(np.linalg.det(Vt.T @ U.T)) or want
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, cd - R @ cs


@dataclass
class Fit:
    R: np.ndarray
    t: np.ndarray
    mirror: bool
    pairs: list                      # (ut_index, xct_index)
    residuals_mm: list
    rms_mm: float
    rotation_deg: float
    alternatives: list = field(default_factory=list)   # best fits of other assignments / handedness


def fit_holes(ut_pts, xct_pts, handedness='auto', prefer_proper_margin=0.5):
    """Best rigid map UT mm -> XCT mm over all assignments of the XCT holes to UT holes.

    ``handedness``: 'proper' (rotation only), 'mirror' (rotation + mirror) or 'auto' (both; a
    mirrored fit is chosen only if its RMS is below ``prefer_proper_margin`` × the best proper
    one). ``alternatives`` holds the next best fits that were considered."""
    ut_pts, xct_pts = np.asarray(ut_pts, float), np.asarray(xct_pts, float)
    n = len(xct_pts)
    if n < 3 or len(ut_pts) < n:
        raise ValueError(f"need at least 3 XCT holes and as many UT holes (XCT {n}, UT {len(ut_pts)})")
    fits = []
    for perm in itertools.permutations(range(len(ut_pts)), n):
        src = ut_pts[list(perm)]
        for reflect in {'auto': (False, True), 'proper': (False,), 'mirror': (True,)}[handedness]:
            R, t = _kabsch(src, xct_pts, reflect)
            res = np.linalg.norm(src @ R.T + t - xct_pts, axis=1)
            fits.append((float(np.sqrt(np.mean(res ** 2))), reflect, perm, R, t, res))
    fits.sort(key=lambda f: f[0])
    best_proper = next((f for f in fits if not f[1]), None)
    best_mirror = next((f for f in fits if f[1]), None)
    best = best_proper or best_mirror
    if best_proper is not None and best_mirror is not None and best_mirror[0] < prefer_proper_margin * best_proper[0]:
        best = best_mirror
    rms, reflect, perm, R, t, res = best
    alts = [{'rms_mm': f[0], 'mirror': f[1], 'pairs': [(int(u), j) for j, u in enumerate(f[2])]}
            for f in fits if f is not best][:5]
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return Fit(R, t, bool(reflect), [(int(u), j) for j, u in enumerate(perm)],
               [float(r) for r in res], rms, rot, alts)


def db_matrix(fit: Fit, xct_voxel_mm):
    """The 3x3 matrix in the database convention (see the module docstring)."""
    M = np.eye(3)
    M[:2, :2] = fit.R
    M[:2, 2] = fit.t / xct_voxel_mm
    return M


# ----------------------------------------------------------------------------- one sample

@dataclass
class Result:
    ut: UTHoles
    xct: XCTHoles
    fit: Fit | None
    error: str | None = None

    def summary(self):
        if self.fit is None:
            return {'error': self.error, 'ut_holes': len(self.ut.holes), 'xct_holes': len(self.xct.holes)}
        f = self.fit
        return {'rms_mm': round(f.rms_mm, 3), 'max_residual_mm': round(max(f.residuals_mm), 3),
                'rotation_deg': round(f.rotation_deg, 3), 'mirror': f.mirror,
                'ut_holes': len(self.ut.holes), 'xct_holes': len(self.xct.holes),
                'next_best_rms_mm': round(f.alternatives[0]['rms_mm'], 3) if f.alternatives else None,
                'next_best_mirror': f.alternatives[0]['mirror'] if f.alternatives else None}


def register(ut_tif, xct_tif, xct_voxel_mm=0.025, handedness='auto', ut_record=None, xct_cache=None,
             ignore_ut=(), ignore_xct=(), min_area_mm2=4.0):
    """Detect the holes in both and fit the rigid map.

    ``handedness``: 'auto' (mirror only when it fits clearly better), 'proper' (rotation only) or
    'mirror'. ``ignore_ut`` / ``ignore_xct``: hole numbers (as shown in the QC figure) to leave out.
    ``xct_cache``: an XCTHoles of the same file to reuse (reading the XCT is the slow part)."""
    ut = ut_holes(ut_tif, ut_record, min_area_mm2=min_area_mm2)
    xct = xct_cache or xct_holes(xct_tif, xct_voxel_mm, min_area_mm2=min_area_mm2)
    try:
        ui = [i for i in range(len(ut.holes)) if i not in set(ignore_ut)]
        xi = [i for i in range(len(xct.holes)) if i not in set(ignore_xct)]
        if len(xi) > len(ui) >= 3:
            xi = xi[:len(ui)]                                  # largest holes first
        up = [(ut.holes[i]['row_mm'], ut.holes[i]['col_mm']) for i in ui]
        xp = [(xct.holes[i]['row_mm'], xct.holes[i]['col_mm']) for i in xi]
        fit = fit_holes(up, xp, handedness)
        fit.pairs = [(ui[u], xi[x]) for u, x in fit.pairs]     # back to the hole numbers of the figure
        for a in fit.alternatives:
            a['pairs'] = [(ui[u], xi[x]) for u, x in a['pairs']]
        return Result(ut, xct, fit)
    except ValueError as e:
        return Result(ut, xct, None, str(e))


# ----------------------------------------------------------------------------- QC

def warp_xct_projection(res: Result, M, out_mm=0.25):
    """XCT projection resampled into the UT footprint with the stored matrix ``M`` (database
    convention), on a ``out_mm`` grid of UT (index, scan) millimetres. Uses ``M`` exactly as
    ``preprocess_tools.register.apply_transform_parameters`` does, so it checks the stored value."""
    U, X = res.ut, res.xct
    n_r = int(np.ceil(U.echo_map.shape[0] * U.spacing_rc[0] / out_mm))
    n_c = int(np.ceil(U.echo_map.shape[1] * U.spacing_rc[1] / out_mm))
    v, s = X.voxel_mm, X.step
    # out index -> UT mm -> UT mm / voxel (M's input) -> XCT px -> projection px (/ step)
    A = np.asarray(M, float)[:2, :2] * (out_mm / (v * s))
    b = np.asarray(M, float)[:2, 2] / s
    warped = ndimage.affine_transform(X.projection, A, offset=b, output_shape=(n_r, n_c), order=1, cval=np.nan)
    inside = ndimage.affine_transform(X.part.astype(np.float32), A, offset=b, output_shape=(n_r, n_c), order=0,
                                      cval=0.0) > 0.5
    return warped, inside


def _upsample_ut(img, spacing_rc, out_mm=0.25):
    zr, zc = spacing_rc[0] / out_mm, spacing_rc[1] / out_mm
    return ndimage.zoom(np.asarray(img, float), (zr, zc), order=1)


def plot_registration(res: Result, title='', fig=None):
    """Confirmation figure: holes in both, the XCT resampled into the UT with the stored matrix
    (blend and checkerboard)."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(16, 10))
    fig.clear()
    ax = fig.subplot_mosaic([['ut', 'ut', 'xct'], ['blend', 'blend', 'xct'], ['check', 'check', 'info']])
    U, X, F = res.ut, res.xct, res.fit
    sr, sc = U.spacing_rc
    ext_u = [-sc / 2, U.echo_map.shape[1] * sc - sc / 2, U.echo_map.shape[0] * sr - sr / 2, -sr / 2]
    ax['ut'].imshow(U.echo_map, cmap='gray', extent=ext_u, aspect='equal')
    for i, h in enumerate(U.holes):
        used = F is not None and any(u == i for u, _ in F.pairs)
        ax['ut'].plot(h['col_mm'], h['row_mm'], '+', color='r' if used else '0.6', ms=13, mew=1.5)
        ax['ut'].annotate(f"U{i}", (h['col_mm'], h['row_mm']), xytext=(6, -6), textcoords='offset points',
                          color='r' if used else '0.6', fontsize=9)
    v = X.voxel_mm * X.step
    ax['xct'].imshow(X.projection, cmap='gray', extent=[0, X.projection.shape[1] * v, X.projection.shape[0] * v, 0])
    for i, h in enumerate(X.holes):
        ax['xct'].plot(h['col_mm'], h['row_mm'], 'o', mfc='none', mec='c', ms=13, mew=1.5)
        ax['xct'].annotate(f"X{i}", (h['col_mm'], h['row_mm']), xytext=(8, 4), textcoords='offset points', color='c')
    ax['xct'].set_title('XCT: holes ○, UT holes mapped ×')
    ax['xct'].set_xlabel('X (mm)')
    ax['xct'].set_ylabel('Y (mm)')
    ax['ut'].set_title('UT entrance-echo amplitude: holes +, XCT holes mapped ○')
    if F is None:
        for k in ('blend', 'check', 'info'):
            ax[k].axis('off')
        ax['info'].text(0, 1, f"NO FIT: {res.error}", va='top', color='r', fontsize=11, transform=ax['info'].transAxes)
        fig.suptitle(title)
        return fig
    Ri = np.linalg.inv(F.R)
    for u, x in F.pairs:
        hu, hx = U.holes[u], X.holes[x]
        p = F.R @ [hu['row_mm'], hu['col_mm']] + F.t
        ax['xct'].plot(p[1], p[0], 'rx', ms=10, mew=1.5)
        q = Ri @ (np.array([hx['row_mm'], hx['col_mm']]) - F.t)
        ax['ut'].plot(q[1], q[0], 'o', mfc='none', mec='c', ms=13, mew=1.5)
    M = db_matrix(F, X.voxel_mm)
    w, inside = warp_xct_projection(res, M)
    ut_up = _upsample_ut(U.cscan, U.spacing_rc)[:w.shape[0], :w.shape[1]]
    ext = [0, w.shape[1] * 0.25, w.shape[0] * 0.25, 0]
    ax['blend'].imshow(ut_up, cmap='gray', extent=ext, aspect='equal')
    ax['blend'].imshow(np.where(inside, w, np.nan), cmap='magma', alpha=0.55, extent=ext, aspect='equal')
    ax['blend'].set_title('UT C-scan below the echo (grey) + XCT resampled by the stored matrix (colour)')
    norm = lambda a: np.clip((a - np.nanpercentile(a, 0.5)) / (np.nanpercentile(a, 99.5) - np.nanpercentile(a, 0.5) + 1e-9), 0, 1)
    e_up = norm(_upsample_ut(U.echo_map, U.spacing_rc)[:w.shape[0], :w.shape[1]])
    wn = norm(np.where(inside, w, np.nan))
    rr, cc = np.indices(w.shape)
    tiles = ((rr // 20 + cc // 20) % 2).astype(bool)                       # 5 mm tiles
    chk = np.where(tiles & inside, wn, e_up)
    ax['check'].imshow(chk, cmap='gray', extent=ext, aspect='equal', vmin=0, vmax=1)
    ax['check'].set_title('Checkerboard: UT entrance echo / XCT (5 mm tiles inside the XCT)')
    for k in ('ut', 'blend', 'check'):
        ax[k].set_xlabel('scan (mm)')
        ax[k].set_ylabel('index (mm)')
    s = res.summary()
    lines = [f"RMS residual {s['rms_mm']:.3f} mm   (max {s['max_residual_mm']:.3f} mm)",
             f"rotation {s['rotation_deg']:+.2f}°   mirror: {'YES' if F.mirror else 'no'}",
             f"holes used {len(F.pairs)}   (UT found {s['ut_holes']}, XCT found {s['xct_holes']})",
             "pairs: " + ", ".join(f"U{u}→X{x} {r:.2f}" for (u, x), r in zip(F.pairs, F.residuals_mm))]
    if F.alternatives:
        a = F.alternatives[0]
        ratio = a['rms_mm'] / max(F.rms_mm, 1e-6)
        lines.append(f"next best: {a['rms_mm']:.3f} mm ({'mirror' if a['mirror'] else 'no mirror'}), {ratio:.1f}×")
        if ratio < 3:
            lines.append("WARNING: next best fit is close — check the hole pairs")
    if s['rms_mm'] > 0.5:
        lines.append("WARNING: residual above 0.5 mm")
    ax['info'].axis('off')
    ax['info'].text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=10, transform=ax['info'].transAxes,
                    color='k' if not any(l.startswith('WARNING') for l in lines) else '#b26a00')
    fig.suptitle(title)
    fig.tight_layout()
    return fig
