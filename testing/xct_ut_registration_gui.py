#!/usr/bin/env python
"""
Interactive XCT <-> UT rigid registration (SimpleITK + PyQt6).

Extracted from ``notebooks/jorge/2_stage_xy_rz_reg_stepwise.ipynb`` with these fixes:

* US smoothing sigma is applied in *voxels* (the notebook passed voxels to a
  filter that expects millimetres, blurring ~70 depth samples).
* Locked degrees of freedom use ``SetOptimizerWeights`` instead of zeroing
  parameters in an iteration callback.
* XCT is anti-aliased (block average + residual Gaussian) before downsampling.
* tz is initialised from the UT front-wall depth vs the XCT top surface and then
  kept fixed (the notebook forced tz = 0).
* Every stage is re-scored with the same deterministic (full-sampling) metric
  and costs are reported as costs (lower is better). A stage is only accepted
  if it improves on the best result so far; otherwise the next stage starts
  from that best result. Stage 3 grid points with low mask overlap are ignored.
* Defaults use Mattes MI with full sampling. On both a synthetic phantom and
  JI_7, Joint-Histogram MI (the notebook default) and 15 % random sampling on
  the tiny coarse grid drove Stage 1 out of the overlap region.
* The Stage 2 XCT grid is 0.25 mm in-plane by default: with XCT resampled to the
  UT 1 mm grid, MI shows interpolation minima at whole-pixel shifts.

Conventions
-----------
* Every volume is viewed as (Z, Y, X) with Z = depth (probe side at Z = 0).
  Use the axis order / flip controls of the Load tab to get there.
* Spacings are entered as (dz, dy, dx) in mm.
* ``Execute(fixed=UT, moving=XCT)``: the resulting transform maps UT physical
  points to XCT physical points, i.e. it is the transform used to resample the
  XCT into the UT grid ("ct_to_us" in file names, as in the notebooks).
* Euler3D parameters are [rx, ry, rz, tx, ty, tz] in SimpleITK (x, y, z) order,
  so rz is the in-plane rotation and tz the depth translation.

Usage
-----
    python xct_ut_registration_gui.py [--xct PATH] [--ut PATH] ...
    python xct_ut_registration_gui.py --xct ct.tif --ut ut.tif --run-headless OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
import zipfile
from dataclasses import asdict, dataclass, field
from itertools import permutations
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tifffile
from scipy import ndimage
from scipy.signal import hilbert
from skimage.filters import threshold_otsu

try:
    import h5py
except ImportError:  # h5 support is optional
    h5py = None


# =============================================================================
# I/O
# =============================================================================

VOLUME_FILTER = "Volumes (*.npy *.npz *.tif *.tiff *.h5 *.hdf5 *.he5);;All files (*)"
H5_EXTS = ('.h5', '.hdf5', '.he5')


def _npz_shapes(path):
    """{key: shape} of the arrays inside an .npz, read from the headers only."""
    shapes = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith('.npy'):
                continue
            key = name[:-4]
            try:
                with zf.open(name) as fh:
                    version = np.lib.format.read_magic(fh)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shapes[key] = reader(fh)[0]
            except Exception:
                shapes[key] = None
    return shapes


def list_volume_keys(path):
    """Names of the >=3D arrays inside a .npz / .h5 file ([] for single-array formats)."""
    path = str(path)
    ext = Path(path).suffix.lower()
    if ext == '.npz':
        return [k for k, s in _npz_shapes(path).items() if s is None or len(s) >= 3]
    if ext in H5_EXTS:
        if h5py is None:
            raise ImportError("h5py is required to read .h5 files")
        keys = []
        with h5py.File(path, 'r') as f:
            f.visititems(lambda name, obj: keys.append(name)
                         if isinstance(obj, h5py.Dataset) and obj.ndim >= 3 else None)
        return keys
    return []


def _squeeze_to_3d(arr, path):
    if arr.ndim == 3:
        return arr
    if arr.ndim > 3 and sum(s > 1 for s in arr.shape) <= 3 and isinstance(arr, np.ndarray):
        arr = arr.reshape([s for s in arr.shape if s > 1] or [1])
        while arr.ndim < 3:
            arr = arr[np.newaxis]
        return arr
    raise ValueError(f"{Path(path).name}: expected a 3D volume, got shape {arr.shape}")


def load_raw(path, key=None):
    """Open a 3D volume lazily when the format allows it.

    Returns an array-like with ``shape``, ``dtype`` and basic slicing:
    a memmap for .npy / memmappable .tif, an ``h5py.Dataset`` for .h5,
    and an in-memory array otherwise (.npz, compressed .tif).
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    ext = Path(path).suffix.lower()
    if ext == '.npy':
        arr = np.load(path, mmap_mode='r')
    elif ext == '.npz':
        with np.load(path) as z:
            key = key or z.files[0]
            arr = z[key]
    elif ext in ('.tif', '.tiff'):
        try:
            arr = tifffile.memmap(path, mode='r')
        except Exception:
            arr = tifffile.imread(path)
    elif ext in H5_EXTS:
        if h5py is None:
            raise ImportError("h5py is required to read .h5 files")
        keys = list_volume_keys(path)
        if not keys:
            raise ValueError(f"{Path(path).name}: no 3D dataset found")
        f = h5py.File(path, 'r')  # kept open by the returned dataset
        arr = f[key or keys[0]]
    else:
        img = sitk.ReadImage(path)  # .mha, .nrrd, .nii, ...
        arr = sitk.GetArrayFromImage(img)
    return _squeeze_to_3d(arr, path)


AXES = ('Z', 'Y', 'X')
PERMUTATIONS = list(permutations(range(3)))


def rot90_orientation(perm, flips, spacing, k):
    """Compose an in-plane rotation by ``k`` quarter turns (``np.rot90(view, k, axes=(1, 2))``)
    into a (perm, flips, spacing) orientation. Spacing follows its axis."""
    perm, flips, spacing = list(perm), list(flips), list(spacing)
    for _ in range(int(k) % 4):
        # rot90 once: new Y = old X reversed, new X = old Y
        perm = [perm[0], perm[2], perm[1]]
        flips = [flips[0], not flips[2], flips[1]]
        spacing = [spacing[0], spacing[2], spacing[1]]
    return tuple(perm), tuple(flips), tuple(spacing)


def perm_label(p):
    return '  '.join(f"{AXES[i]}←raw{p[i]}" for i in range(3))


class VolumeView:
    """Lazy (Z, Y, X) view of a raw 3D array.

    View axis ``i`` reads raw axis ``perm[i]``, reversed when ``flips[i]``.
    Works for numpy arrays, memmaps and h5py datasets (which support neither
    transposes nor negative strides), so changing the orientation is free.
    """

    def __init__(self, raw, perm=(0, 1, 2), flips=(False, False, False),
                 spacing=(1.0, 1.0, 1.0), name='', meta=None):
        self.raw = raw
        self.perm = tuple(int(p) for p in perm)
        self.flips = tuple(bool(f) for f in flips)
        self.spacing = tuple(float(s) for s in spacing)
        self.name = name
        self.meta = meta or {}

    @property
    def shape(self):
        return tuple(int(self.raw.shape[p]) for p in self.perm)

    @property
    def dtype(self):
        return self.raw.dtype

    @property
    def extent_mm(self):
        return tuple(n * s for n, s in zip(self.shape, self.spacing))

    def read(self, sel):
        """Read ``sel`` (one int or step-1 slice per view axis) as a numpy array in view order."""
        raw_sel = [slice(None)] * 3
        kept = []
        for i, s in enumerate(sel):
            n, ax = self.shape[i], self.perm[i]
            if isinstance(s, (int, np.integer)):
                k = int(s) % n
                raw_sel[ax] = n - 1 - k if self.flips[i] else k
            else:
                a, b, _ = s.indices(n)
                raw_sel[ax] = slice(n - b, n - a) if self.flips[i] else slice(a, b)
                kept.append(i)
        out = np.asarray(self.raw[tuple(raw_sel)])
        kept_raw = sorted(self.perm[i] for i in kept)
        out = np.transpose(out, [kept_raw.index(self.perm[i]) for i in kept])
        flip_axes = [j for j, i in enumerate(kept) if self.flips[i]]
        return np.flip(out, axis=flip_axes) if flip_axes else out

    def slice(self, axis, index):
        sel = [slice(None)] * 3
        sel[axis] = int(index)
        return self.read(sel)

    def to_numpy(self):
        return np.ascontiguousarray(self.read([slice(None)] * 3))

    def preview(self, max_voxels=4_000_000):
        """Strided subsample of the whole view, in view order. Returns (array, step)."""
        n = float(np.prod(self.raw.shape))
        step = max(1, int(np.ceil((n / max_voxels) ** (1 / 3))))
        arr = np.asarray(self.raw[::step, ::step, ::step])
        arr = np.transpose(arr, self.perm)
        flip_axes = [i for i in range(3) if self.flips[i]]
        return (np.flip(arr, axis=flip_axes) if flip_axes else arr), step

    def describe(self):
        z, y, x = self.shape
        ez, ey, ex = self.extent_mm
        return (f"raw {tuple(self.raw.shape)} {self.dtype}  →  Z×Y×X = {z}×{y}×{x}   "
                f"extent {ez:.2f} × {ey:.2f} × {ex:.2f} mm")


def envelope_z(vol):
    """Hilbert envelope along axis 0 (depth). Float-safe, unlike ``signal.envelope``."""
    v = np.asarray(vol, dtype=np.float32)
    v = v - v.mean(axis=0, keepdims=True)
    return np.abs(hilbert(v, axis=0)).astype(np.float32)


# =============================================================================
# Image utilities
# =============================================================================

def to_sitk(arr, spacing_zyx):
    arr = np.asarray(arr)
    if arr.dtype == bool or arr.dtype == np.float16:
        arr = arr.astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing([float(s) for s in spacing_zyx[::-1]])
    return img


def resample(img, ref, transform=None, interp=sitk.sitkLinear, default=0.0):
    return sitk.Resample(img, ref, transform or sitk.Transform(), interp, float(default),
                         img.GetPixelID())


def smooth_voxels(img, sigma_vox):
    """Gaussian smoothing with sigma given in voxels per (x, y, z) axis (0 = none)."""
    sigma_vox = np.broadcast_to(np.asarray(sigma_vox, float), (3,))
    if not np.any(sigma_vox > 0):
        return img
    var_mm2 = [float((s * sp) ** 2) for s, sp in zip(sigma_vox, img.GetSpacing())]
    return sitk.DiscreteGaussian(sitk.Cast(img, sitk.sitkFloat32), var_mm2, 64, 0.01, True)


def downsample_antialiased(img, new_spacing_xyz):
    """Resample to ``new_spacing_xyz`` (mm) keeping the physical extent.

    Axes that shrink are block-averaged first (BinShrink), then smoothed for the
    remaining factor, then linearly resampled; axes that grow are only resampled.
    """
    sp = np.array(img.GetSpacing())
    new = np.array(new_spacing_xyz, float)
    factors = np.maximum(1, np.floor(new / sp + 1e-6)).astype(int)
    if np.any(factors > 1):
        img = sitk.BinShrink(img, factors.tolist())
    img = sitk.Cast(img, sitk.sitkFloat32)
    sp2 = np.array(img.GetSpacing())
    sigma_mm = 0.5 * np.sqrt(np.maximum(new ** 2 - sp2 ** 2, 0.0))
    if np.any(sigma_mm > 0.1 * sp2):
        img = sitk.DiscreteGaussian(img, (sigma_mm ** 2).tolist(), 64, 0.01, True)
    size = np.array(img.GetSize())
    new_size = np.maximum(1, np.round(size * sp2 / new)).astype(int)
    origin = np.array(img.GetOrigin()) - 0.5 * sp2 + 0.5 * new  # keep the first voxel edge
    return sitk.Resample(img, new_size.tolist(), sitk.Transform(), sitk.sitkLinear,
                         origin.tolist(), new.tolist(), img.GetDirection(), 0.0, sitk.sitkFloat32)


def normalize(img, mask=None, percentiles=(2, 98), sigma_vox=0.0, log_compress=False):
    """Optional smoothing (voxels) → percentile clip (inside ``mask``) → [0, 1] → optional log1p."""
    img = smooth_voxels(sitk.Cast(img, sitk.sitkFloat32), sigma_vox)
    arr = sitk.GetArrayFromImage(img)
    vals = arr
    if mask is not None:
        m = sitk.GetArrayViewFromImage(mask) > 0
        if m.any():
            vals = arr[m]
    vals = vals.ravel()
    if vals.size > 5_000_000:
        vals = vals[:: vals.size // 5_000_000]
    lo, hi = np.percentile(vals, percentiles)
    out = np.clip((arr - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(arr)
    if log_compress:
        out = np.log1p(2.0 * out) / np.log1p(2.0)
    res = sitk.GetImageFromArray(out.astype(np.float32))
    res.CopyInformation(img)
    return res


def _largest_component(binary):
    lab, n = ndimage.label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def ct_material_mask(ct_img):
    """Otsu → largest 3D component → fill internal voids (3D and per Z slice)."""
    arr = sitk.GetArrayFromImage(ct_img)
    sample = arr.ravel()[:: max(1, arr.size // 5_000_000)]
    m = _largest_component(arr > threshold_otsu(sample))
    m = ndimage.binary_fill_holes(m)
    for z in range(m.shape[0]):
        m[z] = ndimage.binary_fill_holes(m[z])
    out = sitk.GetImageFromArray(m.astype(np.uint8))
    out.CopyInformation(ct_img)
    return out


def us_footprint_mask(us_norm, min_depth=0):
    """Otsu on the max-amplitude C-scan (below ``min_depth``), largest component, holes filled,
    extruded through depth. Returns (3D mask image, 2D footprint)."""
    arr = sitk.GetArrayViewFromImage(us_norm)
    cscan = arr[min(min_depth, arr.shape[0] - 1):].max(axis=0)
    fp = ndimage.binary_fill_holes(_largest_component(cscan > threshold_otsu(cscan)))
    m3 = np.repeat(fp[np.newaxis].astype(np.uint8), arr.shape[0], axis=0)
    out = sitk.GetImageFromArray(m3)
    out.CopyInformation(us_norm)
    return out, fp


def resample_mask(mask, ref, transform=None):
    """Partial-volume aware mask resampling (linear, then >= 0.5)."""
    m = sitk.Resample(sitk.Cast(mask, sitk.sitkFloat32), ref, transform or sitk.Transform(),
                      sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return sitk.Cast(m >= 0.5, sitk.sitkUInt8)


def ut_front_wall(us_norm, footprint, min_depth=0):
    """Median depth (index, mm) of the per-A-scan amplitude maximum below ``min_depth``."""
    arr = sitk.GetArrayViewFromImage(us_norm)
    md = min(min_depth, arr.shape[0] - 1)
    idx = arr[md:].argmax(axis=0) + md
    k = float(np.median(idx[footprint])) if footprint.any() else float(np.median(idx))
    return k, us_norm.GetOrigin()[2] + k * us_norm.GetSpacing()[2]


def ct_top_surface(ct_mask):
    """Median depth (index, mm) of the first material voxel along Z."""
    m = sitk.GetArrayViewFromImage(ct_mask) > 0
    has = m.any(axis=0)
    if not has.any():
        return 0.0, ct_mask.GetOrigin()[2]
    k = float(np.median(m.argmax(axis=0)[has]))
    return k, ct_mask.GetOrigin()[2] + k * ct_mask.GetSpacing()[2]


def dice(a, b):
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 0.0


# =============================================================================
# Registration
# =============================================================================

METRICS = ('JOINT_HISTOGRAM_MI', 'MATTES_MI', 'ANTS_NEIGHBORHOOD_CORRELATION')
DOF = ('rx', 'ry', 'rz', 'tx', 'ty', 'tz')


@dataclass
class StageConfig:
    metric: str = 'MATTES_MI'
    bins: int = 32
    sampling: float = 1.0           # fraction of voxels; >= 1 means all (deterministic)
    jh_variance: float = 1.5        # JOINT_HISTOGRAM_MI only
    ants_radius: int = 4            # ANTS_NEIGHBORHOOD_CORRELATION only
    shrink: list = field(default_factory=lambda: [3, 2, 1])
    sigmas_mm: list = field(default_factory=lambda: [3.0, 1.5, 0.0])
    max_iters: int = 300
    learning_rate: float = 2.0
    min_step: float = 1e-5
    relaxation: float = 0.5
    free_dof: list = field(default_factory=lambda: [False, False, True, True, True, False])


@dataclass
class Stage3Config:
    enabled: bool = True
    range_mm: float = 1.0           # ± around the best result so far, in tx and ty
    step_mm: float = 0.25
    range_deg: float = 0.5          # ± in rz (only if rz is free in Stage 2)
    step_deg: float = 0.1
    min_overlap: float = 0.9        # ignore grid points with less mask overlap than this × the max


@dataclass
class PipelineConfig:
    ct_clip: list = field(default_factory=lambda: [2.0, 98.0])    # percentiles inside the material mask
    us_clip: list = field(default_factory=lambda: [0.0, 100.0])
    us_sigma_vox: float = 0.5
    us_log: bool = True
    us_gate_mm: float = 0.5         # ignore the first mm of each A-scan (front-wall detection, UT mask)
    stage1_spacing_mm: float = 0.8
    stage2_inplane_mm: float = 0.25  # XCT in-plane spacing for Stage 2 (0 = UT spacing). Keep it finer than
    #                                 the UT pixel: equal grids give MI minima at whole-pixel shifts.
    tz_from_surfaces: bool = True
    search_rot90: bool = True       # Stage 0: try XCT rotated by 0/90/180/270° in-plane, keep the best footprint fit
    moving_mask: bool = True        # also restrict the metric with the XCT mask (overlap then varies with the transform)
    seed: int = 78
    threads: int = 0                # ITK threads; 0 = all cores. Multithreaded sums are not bit-reproducible
    #                                 across runs (Stage 2 moved ~0.1 mm on JI_7); 1 = reproducible, ~3x slower.
    stage1: StageConfig = field(default_factory=StageConfig)
    stage2: StageConfig = field(default_factory=lambda: StageConfig(
        bins=64, ants_radius=5, shrink=[2, 1], sigmas_mm=[1.0, 0.0],
        max_iters=2500, learning_rate=1.0, min_step=1e-8))
    stage3: Stage3Config = field(default_factory=Stage3Config)

    @classmethod
    def from_dict(cls, d):
        known = {f for f in cls.__dataclass_fields__}
        d = {k: v for k, v in d.items() if k in known}  # drop keys from older versions (e.g. us_min_depth)
        for k, sub in (('stage1', StageConfig), ('stage2', StageConfig), ('stage3', Stage3Config)):
            if isinstance(d.get(k), dict):
                d[k] = sub(**d[k])
        return cls(**d)


class Cancelled(Exception):
    pass


@dataclass
class StageResult:
    name: str
    transform: sitk.Euler3DTransform
    cost: float                 # deterministic cost of the Stage 2 metric (lower is better)
    overlap: int                # UT-mask voxels that land inside the XCT mask
    footprint_dice: float       # 2D in-plane footprint Dice (masks; not independent of the metric)
    iterations: int = 0
    stop: str = ''
    accepted: bool = True
    note: str = ''

    def params(self):
        p = self.transform.GetParameters()
        return {'rx_deg': np.rad2deg(p[0]), 'ry_deg': np.rad2deg(p[1]), 'rz_deg': np.rad2deg(p[2]),
                'tx_mm': p[3], 'ty_mm': p[4], 'tz_mm': p[5]}


@dataclass
class PipelineResult:
    config: PipelineConfig
    stages: list
    final: str
    cost_log: list              # (stage, level, iteration, cost)
    grid: dict | None
    z_info: dict
    us_fine: sitk.Image         # normalized UT, native grid (reference space)
    us_mask: sitk.Image
    us_footprint: np.ndarray
    ct_fine_norm: sitk.Image
    ct_fine_raw: sitk.Image
    ct_mask: sitk.Image
    warnings: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)

    def stage(self, name):
        return next(s for s in self.stages if s.name == name)

    def warp_ct(self, name, raw=False):
        key = (name, raw)
        if key not in self._cache:
            src = self.ct_fine_raw if raw else self.ct_fine_norm
            img = resample(src, self.us_fine, self.stage(name).transform)
            self._cache[key] = sitk.GetArrayFromImage(img)
        return self._cache[key]

    def warp_ct_mask(self, name):
        key = (name, 'mask')
        if key not in self._cache:
            m = resample(self.ct_mask, self.us_fine, self.stage(name).transform, sitk.sitkNearestNeighbor)
            self._cache[key] = sitk.GetArrayFromImage(m)
        return self._cache[key]


def configure_metric(reg, metric, bins, sampling, jh_variance, ants_radius, seed):
    if metric == 'MATTES_MI':
        reg.SetMetricAsMattesMutualInformation(int(bins))
    elif metric == 'JOINT_HISTOGRAM_MI':
        reg.SetMetricAsJointHistogramMutualInformation(int(bins), float(jh_variance))
    elif metric == 'ANTS_NEIGHBORHOOD_CORRELATION':
        reg.SetMetricAsANTSNeighborhoodCorrelation(int(ants_radius))
    else:
        raise ValueError(f"Unknown metric {metric!r}; choose from {METRICS}")
    if metric != 'ANTS_NEIGHBORHOOD_CORRELATION' and sampling < 1.0:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(float(sampling), int(seed))
    else:
        reg.SetMetricSamplingStrategy(reg.NONE)


class RegistrationPipeline:
    """Stage 0 (init) → Stage 1 (coarse) → Stage 2 (fine) → Stage 3 (local grid).

    ``log(str)`` receives progress text, ``on_cost(stage, level, iteration, cost)``
    every optimizer iteration. Both may be called from a worker thread.
    """

    def __init__(self, ct, ct_spacing, us, us_spacing, cfg: PipelineConfig,
                 log=print, on_cost=None):
        self.ct, self.ct_spacing = ct, tuple(ct_spacing)
        self.us, self.us_spacing = us, tuple(us_spacing)
        self.cfg = cfg
        self.log = log
        self.on_cost = on_cost or (lambda *a: None)
        self._cancel = False
        self.cost_log = []
        self.warnings = []

    def warn(self, msg):
        self.warnings.append(msg)
        self.log(f"  WARNING: {msg}")

    def cancel(self):
        self._cancel = True

    def _check(self):
        if self._cancel:
            raise Cancelled()

    # ------------------------------------------------------------------ prep
    def _prepare(self):
        cfg = self.cfg
        t0 = time.time()
        self.log("Building images…")
        ct_raw = to_sitk(self.ct, self.ct_spacing)
        us_raw = sitk.Cast(to_sitk(self.us, self.us_spacing), sitk.sitkFloat32)
        self.ct = self.us = None  # release the numpy copies
        usx, usy, _ = us_raw.GetSpacing()
        inplane = cfg.stage2_inplane_mm or None
        fine_sp = (inplane or usx, inplane or usy, ct_raw.GetSpacing()[2])
        iso = [cfg.stage1_spacing_mm] * 3

        self.log(f"  XCT fine grid {fine_sp} mm (x, y, z), coarse {cfg.stage1_spacing_mm} mm iso")
        self.ct_fine_raw = downsample_antialiased(ct_raw, fine_sp)
        ct_coarse_raw = downsample_antialiased(ct_raw, iso)
        del ct_raw
        us_coarse_raw = downsample_antialiased(us_raw, iso)
        self._check()

        self.log("Masks and normalization…")
        self.ct_mask = ct_material_mask(self.ct_fine_raw)
        ct_mask_coarse = resample_mask(self.ct_mask, ct_coarse_raw)
        self.us_fine = normalize(us_raw, None, cfg.us_clip, cfg.us_sigma_vox, cfg.us_log)
        nz, dz = self.us_fine.GetSize()[2], self.us_fine.GetSpacing()[2]
        self.gate = int(round(cfg.us_gate_mm / dz))
        if self.gate > nz // 2:
            self.warn(f"UT gate {cfg.us_gate_mm} mm = {self.gate} samples is more than half of the "
                      f"{nz} samples ({nz * dz:.2f} mm); using no gate")
            self.gate = 0
        self.us_mask, self.us_fp = us_footprint_mask(self.us_fine, self.gate)
        us_coarse = normalize(us_coarse_raw, None, cfg.us_clip, cfg.us_sigma_vox, cfg.us_log)
        us_mask_coarse = resample_mask(self.us_mask, us_coarse)
        self.ct_fine_norm = normalize(self.ct_fine_raw, self.ct_mask, cfg.ct_clip)
        ct_coarse = normalize(ct_coarse_raw, ct_mask_coarse, cfg.ct_clip)
        self.coarse = (us_coarse, ct_coarse, us_mask_coarse, ct_mask_coarse)
        for name, m in (('XCT', self.ct_mask), ('UT', self.us_mask)):
            a = sitk.GetArrayViewFromImage(m)
            self.log(f"  {name} mask: {int(a.sum()):,} voxels ({100 * a.mean():.1f}%) on {m.GetSize()}")
        self.log(f"  UT footprint: {int(self.us_fp.sum())} A-scans of {self.us_fp.size}")
        self.log(f"  prepared in {time.time() - t0:.1f} s")

    # --------------------------------------------------------------- stages
    def _eval_reg(self, mmask=None):
        s2 = self.cfg.stage2
        reg = sitk.ImageRegistrationMethod()
        configure_metric(reg, s2.metric, s2.bins, 1.0, s2.jh_variance, s2.ants_radius, self.cfg.seed)
        reg.SetMetricFixedMask(self.us_mask)
        if self.cfg.moving_mask:
            reg.SetMetricMovingMask(self.ct_mask if mmask is None else mmask)
        reg.SetInterpolator(sitk.sitkLinear)
        return reg

    def _cost(self, tx, moving=None, mmask=None):
        """Evaluation-metric cost of ``tx`` (inf when the images do not overlap)."""
        reg = self._eval_reg(mmask)
        reg.SetInitialTransform(sitk.Euler3DTransform(tx), inPlace=True)
        try:
            return reg.MetricEvaluate(self.us_fine, self.ct_fine_norm if moving is None else moving)
        except RuntimeError:
            return np.inf

    def _evaluate(self, name, tx, **kw):
        """Deterministic score of ``tx`` on the fine images (same metric for every stage)."""
        reg = self._eval_reg()
        reg.SetInitialTransform(sitk.Euler3DTransform(tx), inPlace=True)
        cost = reg.MetricEvaluate(self.us_fine, self.ct_fine_norm)
        warped = sitk.GetArrayViewFromImage(
            resample(self.ct_mask, self.us_fine, tx, sitk.sitkNearestNeighbor)) > 0
        overlap = int(np.logical_and(warped, sitk.GetArrayViewFromImage(self.us_mask) > 0).sum())
        if overlap == 0:
            cost = np.inf  # SimpleITK returns DBL_MAX when nothing overlaps
        res = StageResult(name, sitk.Euler3DTransform(tx), float(cost), overlap,
                          dice(warped.any(axis=0), self.us_fp), **kw)
        p = res.params()
        self.log(f"  [{name}] cost={res.cost:.6f}  overlap={overlap:,}  footprint Dice={res.footprint_dice:.3f}\n"
                 f"      rz={p['rz_deg']:+.3f}°  tx={p['tx_mm']:+.3f}  ty={p['ty_mm']:+.3f}  "
                 f"tz={p['tz_mm']:+.3f} mm  (rx={p['rx_deg']:+.3f}°, ry={p['ry_deg']:+.3f}°)")
        return res

    def _initial_transform(self):
        us_c, ct_c, usm_c, ctm_c = self.coarse
        try:
            tx = sitk.CenteredTransformInitializer(
                usm_c, ctm_c, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
            how = 'MOMENTS on masks'
        except Exception:
            tx = sitk.CenteredTransformInitializer(
                us_c, ct_c, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
            how = 'GEOMETRY'
        tx = sitk.Euler3DTransform(tx)
        k_us, z_us = ut_front_wall(self.us_fine, self.us_fp, self.gate)
        k_ct, z_ct = ct_top_surface(self.ct_mask)
        t = list(tx.GetTranslation())
        self.z_info = {'ut_front_wall_index': k_us, 'ut_front_wall_mm': z_us,
                       'ct_top_surface_index': k_ct, 'ct_top_surface_mm': z_ct,
                       'tz_moments_mm': t[2], 'tz_surfaces_mm': z_ct - z_us}
        self.log(f"Stage 0: {how}.  UT front wall at sample {k_us:.0f} ({z_us:.3f} mm), "
                 f"XCT top surface at slice {k_ct:.0f} ({z_ct:.3f} mm)")
        if self.cfg.tz_from_surfaces:
            t[2] = z_ct - z_us
            self.log(f"  tz from surfaces = {t[2]:+.3f} mm (moments gave {self.z_info['tz_moments_mm']:+.3f})")
            if abs(t[2] - self.z_info['tz_moments_mm']) > 0.5 * self.us_fine.GetSize()[2] * self.us_fine.GetSpacing()[2]:
                self.warn("surface and moments tz disagree by more than half the UT depth — check the "
                          "UT gate and that Z points into the part from the probe side for both")
        tx.SetTranslation(t)
        return self._search_rot90(tx) if self.cfg.search_rot90 else tx

    def _footprint_dice_2d(self, tx):
        """Dice of the UT footprint vs the XCT footprint moved in-plane by (rz, tx, ty) of ``tx`` (ignores tz)."""
        if not hasattr(self, '_fp2d'):
            def img2d(fp, ref):
                im = sitk.GetImageFromArray(fp.astype(np.uint8))
                im.SetSpacing(ref.GetSpacing()[:2])
                im.SetOrigin(ref.GetOrigin()[:2])
                return im
            ct_fp = sitk.GetArrayViewFromImage(self.ct_mask).any(axis=0)
            self._fp2d = (img2d(self.us_fp, self.us_fine), img2d(ct_fp, self.ct_mask))
        us2d, ct2d = self._fp2d
        p = tx.GetParameters()
        t2 = sitk.Euler2DTransform(tx.GetCenter()[:2], p[2], p[3:5])
        return dice(sitk.GetArrayViewFromImage(sitk.Resample(ct2d, us2d, t2, sitk.sitkNearestNeighbor, 0)),
                    self.us_fp)

    def _search_rot90(self, tx):
        """Try the XCT at 0/90/180/270° in-plane and keep the best fit.

        Candidates are ranked by 2D footprint Dice (independent of tz); near-ties, e.g. 0°/180° for a
        rectangular coupon, are broken by the evaluation-metric cost. The same four rotations are also
        scored on the XCT mirrored in X: a clearly better mirrored cost means the inputs have opposite
        handedness, which no rigid transform can fix.
        """
        rots = []
        for k in range(4):
            t = sitk.Euler3DTransform(tx)
            t.SetRotation(0.0, 0.0, k * np.pi / 2)
            rots.append(t)
        dices = [self._footprint_dice_2d(t) for t in rots]
        costs = [self._cost(t) for t in rots]
        best_d = max(dices)
        k_best = min((k for k in range(4) if dices[k] >= best_d - 0.03), key=lambda k: costs[k])
        self.log("  90° search:  " + "   ".join(f"{90 * k}°: Dice {dices[k]:.3f}, cost {costs[k]:.4f}"
                                                 for k in range(4)) + f"   → {90 * k_best}°")
        # mirror check: flip the XCT in X about its image centre and move the moments centroid accordingly
        def mirror_x(im):  # sitk.Flip also flips the direction cosines (no physical change): reset them
            out = sitk.Flip(im, [True, False, False])
            out.CopyInformation(im)
            return out
        mirrored, mmask = mirror_x(self.ct_fine_norm), mirror_x(self.ct_mask)
        sx, nx, ox = self.ct_mask.GetSpacing()[0], self.ct_mask.GetSize()[0], self.ct_mask.GetOrigin()[0]
        xc = ox + 0.5 * (nx - 1) * sx
        m_costs = []
        for t in rots:
            tm = sitk.Euler3DTransform(t)
            tr = list(tm.GetTranslation())
            tr[0] = 2 * xc - (tm.GetCenter()[0] + tr[0]) - tm.GetCenter()[0]  # mirrored moving centroid
            tm.SetTranslation(tr)
            m_costs.append(self._cost(tm, mirrored, mmask))
        self.log("  mirrored XCT:  " + "   ".join(f"{90 * k}°: cost {m_costs[k]:.4f}" for k in range(4)))
        self.z_info['rot90_search'] = {'deg': [0, 90, 180, 270], 'footprint_dice': dices, 'cost': costs,
                                       'mirrored_cost': m_costs, 'chosen_deg': 90 * k_best}
        bu, bm = min(costs), min(m_costs)
        if np.isfinite(bm) and bm < bu and (bm - bu) < 0.25 * bu:
            self.warn(f"the XCT fits clearly better MIRRORED (best cost {bm:.4f} vs {bu:.4f}). A rigid transform "
                      f"cannot undo a mirror image: enable 'flip X' (or 'flip Y') for the XCT on the Load tab "
                      f"and run again")
        if best_d < 0.5:
            self.warn(f"best footprint Dice after the 90° search is only {best_d:.3f}; check spacings "
                      f"and orientation")
        return rots[k_best]

    def _run_stage(self, name, sc: StageConfig, fixed, moving, fmask, mmask, init):
        if len(sc.shrink) != len(sc.sigmas_mm):
            raise ValueError(f"{name}: shrink factors and smoothing sigmas need the same length")
        reg = sitk.ImageRegistrationMethod()
        configure_metric(reg, sc.metric, sc.bins, sc.sampling, sc.jh_variance, sc.ants_radius, self.cfg.seed)
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=sc.learning_rate, minStep=sc.min_step, numberOfIterations=int(sc.max_iters),
            relaxationFactor=sc.relaxation, gradientMagnitudeTolerance=1e-6,
            estimateLearningRate=reg.EachIteration, maximumStepSizeInPhysicalUnits=sc.learning_rate)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetOptimizerWeights([1.0 if f else 0.0 for f in sc.free_dof])
        reg.SetShrinkFactorsPerLevel([int(s) for s in sc.shrink])
        reg.SetSmoothingSigmasPerLevel([float(s) for s in sc.sigmas_mm])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetMetricFixedMask(fmask)
        if self.cfg.moving_mask:
            reg.SetMetricMovingMask(mmask)
        tx = sitk.Euler3DTransform(init)
        reg.SetInitialTransform(tx, inPlace=True)
        count = [0]

        def on_iter():
            count[0] += 1
            entry = (name, reg.GetCurrentLevel(), reg.GetOptimizerIteration(), reg.GetMetricValue())
            self.cost_log.append(entry)
            self.on_cost(*entry)
            if self._cancel:
                reg.StopRegistration()

        reg.AddCommand(sitk.sitkIterationEvent, on_iter)
        reg.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: self.log(
            f"  {name} level {reg.GetCurrentLevel() + 1}/{len(sc.shrink)}"))
        free = ','.join(d for d, f in zip(DOF, sc.free_dof) if f)
        self.log(f"{name}: {sc.metric}, free DOF [{free}], fixed {fixed.GetSize()}, moving {moving.GetSize()}")
        t0 = time.time()
        reg.Execute(fixed, moving)
        self._check()
        stop = reg.GetOptimizerStopConditionDescription()
        self.log(f"  {count[0]} iterations in {time.time() - t0:.1f} s — {stop}")
        return tx, count[0], stop

    def _run_stage3(self, s2):
        """Local grid around ``s2`` (the best result so far) with the evaluation metric."""
        c = self.cfg.stage3
        nt = int(round(c.range_mm / c.step_mm))
        nr = int(round(c.range_deg / c.step_deg)) if self.cfg.stage2.free_dof[2] else 0
        reg = self._eval_reg()
        reg.SetOptimizerAsExhaustive([0, 0, nr, nt, nt, 0], stepLength=1.0)
        reg.SetOptimizerScales([1.0, 1.0, np.deg2rad(c.step_deg), c.step_mm, c.step_mm, 1.0])
        tx = sitk.Euler3DTransform(s2.transform)
        reg.SetInitialTransform(tx, inPlace=True)
        total = (2 * nt + 1) ** 2 * (2 * nr + 1)
        rec = []

        def on_iter():
            p = reg.GetOptimizerPosition()
            rec.append((p[2], p[3], p[4], reg.GetMetricValue(), reg.GetMetricNumberOfValidPoints()))
            self.cost_log.append(('Stage 3', 0, len(rec), rec[-1][3]))
            if len(rec) % max(1, total // 20) == 0:
                self.log(f"  Stage 3: {len(rec)}/{total}")
            if self._cancel:
                reg.StopRegistration()

        reg.AddCommand(sitk.sitkIterationEvent, on_iter)
        self.log(f"Stage 3: grid ±{c.range_mm} mm @ {c.step_mm} mm, ±{c.range_deg if nr else 0}° "
                 f"@ {c.step_deg}° → {total} evaluations")
        t0 = time.time()
        reg.Execute(self.us_fine, self.ct_fine_norm)
        self._check()
        self.log(f"  done in {time.time() - t0:.1f} s")

        r = np.array(rec, float)
        valid = r[:, 4]
        ok = valid >= c.min_overlap * valid.max()
        best = int(np.argmin(np.where(ok, r[:, 3], np.inf)))
        p0 = s2.transform.GetParameters()
        shape = (2 * nr + 1, 2 * nt + 1, 2 * nt + 1)
        cost = np.full(shape, np.nan)
        vcube = np.zeros(shape)
        ir = np.round((r[:, 0] - p0[2]) / np.deg2rad(c.step_deg)).astype(int) + nr if nr else np.zeros(len(r), int)
        iy = np.round((r[:, 2] - p0[4]) / c.step_mm).astype(int) + nt
        ix = np.round((r[:, 1] - p0[3]) / c.step_mm).astype(int) + nt
        cost[ir, iy, ix] = np.where(ok, r[:, 3], np.nan)
        vcube[ir, iy, ix] = valid
        grid = {'rz_deg': np.rad2deg(p0[2]) + np.arange(-nr, nr + 1) * c.step_deg,
                'tx_mm': p0[3] + np.arange(-nt, nt + 1) * c.step_mm,
                'ty_mm': p0[4] + np.arange(-nt, nt + 1) * c.step_mm,
                'cost': cost, 'valid': vcube, 'best_index': (int(ir[best]), int(iy[best]), int(ix[best])),
                'excluded_low_overlap': int((~ok).sum())}
        self.log(f"  {grid['excluded_low_overlap']} grid points ignored for overlap < "
                 f"{c.min_overlap:.0%} of max")
        edges = [n for n, i, r_ in (('rz', ir[best], nr), ('ty', iy[best], nt), ('tx', ix[best], nt))
                 if r_ and i in (0, 2 * r_)]
        if edges:
            self.warn(f"Stage 3 optimum on the {'/'.join(edges)} border of the grid — the true optimum may "
                      f"lie outside the search range; consider enlarging it")
        grid['on_border'] = edges
        params = list(p0)
        params[2], params[3], params[4] = r[best, 0], r[best, 1], r[best, 2]
        best_tx = sitk.Euler3DTransform(s2.transform)
        best_tx.SetParameters(params)
        return self._evaluate('Stage 3', best_tx, iterations=len(rec), stop='exhaustive grid'), grid

    # ------------------------------------------------------------------ run
    def run(self) -> PipelineResult:
        prev = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        if self.cfg.threads:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(int(self.cfg.threads))
        try:
            return self._run()
        finally:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(prev)

    def _run(self) -> PipelineResult:
        cfg = self.cfg
        self._prepare()
        self._check()
        stages = [self._evaluate('Init', self._initial_transform(), stop='initializer')]
        best = stages[0]

        def accept(res):
            nonlocal best
            stages.append(res)
            d = res.cost - best.cost
            if d < 0:
                res.note = f"accepted (Δcost {d:+.6f} vs {best.name})"
                best = res
            else:
                res.accepted = False
                res.note = f"rejected: not better than {best.name} (Δcost {d:+.6f})"
            self.log(f"  {res.name} {res.note}")

        def failed(name, err):
            msg = str(err).strip().splitlines()[-1]
            if 'overlap' in msg or 'outside' in msg:
                msg += (" — the images barely overlap at the start of this stage: check orientation "
                        "(axis order / flips / rotation), spacings and the UT gate")
            self.warn(f"{name} failed: {msg}")
            res = StageResult(name, sitk.Euler3DTransform(best.transform), np.inf, 0, 0.0,
                              stop='error', accepted=False, note=f"failed: {msg}")
            stages.append(res)

        us_c, ct_c, usm_c, ctm_c = self.coarse
        try:
            tx1, n1, stop1 = self._run_stage('Stage 1', cfg.stage1, us_c, ct_c, usm_c, ctm_c, best.transform)
            accept(self._evaluate('Stage 1', tx1, iterations=n1, stop=stop1))
        except RuntimeError as e:
            failed('Stage 1', e)

        try:
            tx2, n2, stop2 = self._run_stage('Stage 2', cfg.stage2, self.us_fine, self.ct_fine_norm,
                                             self.us_mask, self.ct_mask, best.transform)
            accept(self._evaluate('Stage 2', tx2, iterations=n2, stop=stop2))
        except RuntimeError as e:
            failed('Stage 2', e)

        grid = None
        if cfg.stage3.enabled:
            try:
                s3, grid = self._run_stage3(best)
                accept(s3)
            except (RuntimeError, ValueError) as e:
                failed('Stage 3', e)

        self.log(f"Final transform: {best.name}")
        return PipelineResult(cfg, stages, best.name, self.cost_log, grid, self.z_info,
                              self.us_fine, self.us_mask, self.us_fp,
                              self.ct_fine_norm, self.ct_fine_raw, self.ct_mask, self.warnings)


def _tif(path, arr, spacing_zyx):
    dz, dy, dx = spacing_zyx
    tifffile.imwrite(path, np.ascontiguousarray(arr, dtype=np.float32), imagej=True,
                     resolution=(1 / dx, 1 / dy), metadata={'axes': 'ZYX', 'spacing': dz, 'unit': 'mm'})


def save_results(res: PipelineResult, out_dir, inputs=None, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for st in res.stages:
        sitk.WriteTransform(st.transform, str(out / f"{st.name.lower().replace(' ', '')}_ct_to_us.tfm"))
    final = res.stage(res.final)
    sitk.WriteTransform(final.transform, str(out / "final_ct_to_us.tfm"))
    sitk.WriteTransform(final.transform.GetInverse(), str(out / "final_us_to_ct.tfm"))
    sp = res.us_fine.GetSpacing()[::-1]
    _tif(out / "ut_normalized.tif", sitk.GetArrayViewFromImage(res.us_fine), sp)
    _tif(out / "ct_in_ut_normalized.tif", res.warp_ct(res.final), sp)
    _tif(out / "ct_in_ut_raw.tif", res.warp_ct(res.final, raw=True), sp)
    _tif(out / "ct_mask_in_ut.tif", res.warp_ct_mask(res.final), sp)
    _tif(out / "ut_mask.tif", sitk.GetArrayViewFromImage(res.us_mask), sp)
    if res.grid is not None:
        np.savez(out / "stage3_grid.npz", **{k: np.asarray(v) for k, v in res.grid.items()})
    report = {
        'inputs': inputs or {},
        'config': asdict(res.config),
        'final': res.final,
        'transform_convention': 'Execute(fixed=UT, moving=XCT): maps UT points to XCT points '
                                '(use it to resample XCT into the UT grid).',
        'z_init': res.z_info,
        'warnings': res.warnings,
        'stages': [{'name': s.name, **{k: float(v) for k, v in s.params().items()},
                    'center_xyz_mm': list(s.transform.GetCenter()),
                    'cost': s.cost, 'overlap_voxels': s.overlap, 'footprint_dice': s.footprint_dice,
                    'iterations': s.iterations, 'stop': s.stop, 'accepted': s.accepted, 'note': s.note}
                   for s in res.stages],
        'ut_grid': {'size_xyz': list(res.us_fine.GetSize()), 'spacing_xyz_mm': list(res.us_fine.GetSpacing())},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log(f"Saved results to {out}")
    return out


# =============================================================================
# GUI
# =============================================================================

from PyQt6 import QtCore, QtWidgets  # noqa: E402
from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


class MplCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None, figsize=(8, 6), toolbar=True):
        super().__init__(parent)
        self.fig = Figure(figsize=figsize, layout='constrained')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumSize(320, 220)  # constrained layout breaks on near-zero canvases (hidden tabs)
        self.toolbar = NavigationToolbar2QT(self.canvas, self) if toolbar else None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if self.toolbar:
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def navigating(self):
        return bool(self.toolbar and self.toolbar.mode)

    def draw(self):
        self.canvas.draw_idle()


class IndexSlider(QtWidgets.QWidget):
    """Label + slider + spin box for an integer index."""
    valueChanged = pyqtSignal(int)

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.spin = QtWidgets.QSpinBox()
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QtWidgets.QLabel(label))
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.valueChanged.emit)

    def set_range(self, n, value=None):
        for w in (self.slider, self.spin):
            w.blockSignals(True)
            w.setRange(0, max(0, n - 1))
            w.setValue(n // 2 if value is None else int(np.clip(value, 0, max(0, n - 1))))
            w.blockSignals(False)

    def value(self):
        return self.slider.value()

    def setValue(self, v):
        self.slider.setValue(int(v))


def _dspin(value, lo, hi, decimals=4, step=None):
    s = QtWidgets.QDoubleSpinBox()
    s.setDecimals(decimals)
    s.setRange(lo, hi)
    s.setValue(value)
    if step:
        s.setSingleStep(step)
    return s


def _ispin(value, lo, hi):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


# ----------------------------------------------------------------- Load tab

class VolumeLoaderBox(QtWidgets.QGroupBox):
    viewChanged = pyqtSignal(object)

    def __init__(self, title, default_spacing, allow_envelope=False, parent=None):
        super().__init__(title, parent)
        self.raw = None
        self.path = None
        self._env_cache = {}
        self.path_edit = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse…")
        self.key_combo = QtWidgets.QComboBox()
        self.key_combo.setEnabled(False)
        self.load_btn = QtWidgets.QPushButton("Load")
        self.spacing = [_dspin(s, 1e-6, 1e3, 5, 0.001) for s in default_spacing]
        self.perm_combo = QtWidgets.QComboBox()
        for p in PERMUTATIONS:
            self.perm_combo.addItem(perm_label(p), p)
        self.flips = [QtWidgets.QCheckBox(f"flip {a}") for a in AXES]
        self.rot = QtWidgets.QComboBox()
        for k in range(4):
            self.rot.addItem(f"rotate {90 * k}°", k)
        self.rot.setToolTip("In-plane rotation (counter-clockwise in the top view), applied after axis order and flips")
        self.envelope = QtWidgets.QCheckBox("Hilbert envelope along Z (RF input)") if allow_envelope else None
        self.info = QtWidgets.QLabel("No volume loaded")
        self.info.setWordWrap(True)

        form = QtWidgets.QGridLayout(self)
        form.addWidget(QtWidgets.QLabel("File"), 0, 0)
        form.addWidget(self.path_edit, 0, 1, 1, 4)
        form.addWidget(browse, 0, 5)
        form.addWidget(QtWidgets.QLabel("Dataset / key"), 1, 0)
        form.addWidget(self.key_combo, 1, 1, 1, 4)
        form.addWidget(self.load_btn, 1, 5)
        form.addWidget(QtWidgets.QLabel("Spacing dz, dy, dx (mm, before rotation)"), 2, 0)
        for i, s in enumerate(self.spacing):
            form.addWidget(s, 2, 1 + i)
        form.addWidget(QtWidgets.QLabel("Axis order"), 3, 0)
        form.addWidget(self.perm_combo, 3, 1, 1, 2)
        for i, f in enumerate(self.flips):
            form.addWidget(f, 3, 3 + i)
        form.addWidget(QtWidgets.QLabel("In-plane rotation"), 4, 0)
        form.addWidget(self.rot, 4, 1, 1, 2)
        row = 5
        if self.envelope:
            form.addWidget(self.envelope, row, 1, 1, 4)
            row += 1
        form.addWidget(self.info, row, 0, 1, 6)

        browse.clicked.connect(self._browse)
        self.path_edit.editingFinished.connect(self._refresh_keys)
        self.load_btn.clicked.connect(self.load)
        self.perm_combo.currentIndexChanged.connect(self._emit)
        self.rot.currentIndexChanged.connect(self._emit)
        for w in self.flips + ([self.envelope] if self.envelope else []):
            w.toggled.connect(self._emit)
        for s in self.spacing:
            s.editingFinished.connect(self._emit)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.title(), self.path_edit.text(), VOLUME_FILTER)
        if path:
            self.path_edit.setText(path)
            self._refresh_keys()

    def _refresh_keys(self):
        self.key_combo.clear()
        path = self.path_edit.text().strip()
        try:
            keys = list_volume_keys(path) if path and Path(path).exists() else []
        except Exception as e:
            keys = []
            self.info.setText(f"Cannot list datasets: {e}")
        self.key_combo.addItems(keys)
        self.key_combo.setEnabled(bool(keys))

    def set_orientation(self, perm=None, flips=None, spacing=None, rot90=None):
        if rot90 is not None:
            self.rot.setCurrentIndex(int(rot90) % 4)
        if perm is not None:
            self.perm_combo.setCurrentIndex(PERMUTATIONS.index(tuple(perm)))
        if flips is not None:
            for cb, f in zip(self.flips, flips):
                cb.setChecked(bool(f))
        if spacing is not None:
            for s, v in zip(self.spacing, spacing):
                s.setValue(float(v))

    def load(self, path=None, key=None):
        if path:
            self.path_edit.setText(str(path))
            self._refresh_keys()
            if key:
                self.key_combo.setCurrentText(key)
        path = self.path_edit.text().strip()
        if not path:
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.raw = load_raw(path, self.key_combo.currentText() or None)
            self.path = path
            self._env_cache.clear()
        except Exception as e:
            self.raw = None
            QtWidgets.QMessageBox.critical(self, "Load failed", f"{path}\n\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._emit()

    def settings(self):
        return {'path': self.path, 'key': self.key_combo.currentText() or None,
                'axis_order': list(self.perm_combo.currentData()),
                'flips_zyx': [f.isChecked() for f in self.flips],
                'spacing_zyx_mm': [s.value() for s in self.spacing],
                'rot90': self.rot.currentData(),
                'envelope': bool(self.envelope and self.envelope.isChecked())}

    def view(self):
        if self.raw is None:
            return None
        st = self.settings()
        meta = st
        perm, flips, spacing = rot90_orientation(st['axis_order'], st['flips_zyx'], st['spacing_zyx_mm'], st['rot90'])
        v = VolumeView(self.raw, perm, flips, spacing, self.title(), meta)
        if st['envelope']:
            k = (v.perm, v.flips)
            if k not in self._env_cache:
                QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    self._env_cache = {k: envelope_z(v.to_numpy())}
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
            v = VolumeView(self._env_cache[k], spacing=v.spacing, name=v.name, meta=meta)
        return v

    def _emit(self):
        v = self.view()
        self.info.setText(v.describe() if v else "No volume loaded")
        self.viewChanged.emit(v)


class OrientationPreview(MplCanvas):
    """Top view (max over Z) and side view (max over Y) to check orientation at a glance."""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(6, 3), toolbar=False)
        self.setMinimumHeight(220)

    def set_view(self, v):
        self.fig.clear()
        if v is None:
            self.draw()
            return
        arr, _ = v.preview(2_000_000)
        arr = arr.astype(np.float32)
        lo, hi = np.percentile(arr, (1, 99.5))
        nz, ny, nx = v.shape  # label axes in full-resolution indices, not subsampled ones
        a1, a2 = self.fig.subplots(1, 2, width_ratios=[1, 1.4])
        a1.imshow(arr.max(axis=0), cmap='gray', vmin=lo, vmax=hi, extent=(0, nx, ny, 0),
                  aspect=v.spacing[1] / v.spacing[2], interpolation='nearest')
        a1.set_title("top: max over Z", fontsize=9)
        a1.set_xlabel("X")
        a1.set_ylabel("Y")
        a2.imshow(arr.max(axis=1), cmap='gray', vmin=lo, vmax=hi, extent=(0, nx, nz, 0),
                  aspect='auto', interpolation='nearest')
        a2.set_title("side: max over Y  (probe side should be at Z = 0, top)", fontsize=9)
        a2.set_xlabel("X")
        a2.set_ylabel("Z")
        self.draw()


class LoadTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.xct = VolumeLoaderBox("XCT", (0.025, 0.025, 0.025))
        self.ut = VolumeLoaderBox("UT", (0.016, 1.0, 1.0), allow_envelope=True)
        self.xct_prev = OrientationPreview()
        self.ut_prev = OrientationPreview()
        self.xct.viewChanged.connect(self.xct_prev.set_view)
        self.ut.viewChanged.connect(self.ut_prev.set_view)
        lay = QtWidgets.QGridLayout(self)
        lay.addWidget(self.xct, 0, 0)
        lay.addWidget(self.ut, 0, 1)
        lay.addWidget(self.xct_prev, 1, 0)
        lay.addWidget(self.ut_prev, 1, 1)
        hint = QtWidgets.QLabel(
            "Bring both volumes to (Z = depth, Y, X) with the same handedness: Z points into the part "
            "from the probe side, and the top views should show the part with the same in-plane "
            "orientation (rotation by a few degrees and a shift are solved by the registration; "
            "transposes and mirror images are not).")
        hint.setWordWrap(True)
        lay.addWidget(hint, 2, 0, 1, 2)
        lay.setRowStretch(1, 1)


# --------------------------------------------------------- XCT inspector

class XCTInspector(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.v = None
        self.sz, self.sy, self.sx = IndexSlider("Z"), IndexSlider("Y"), IndexSlider("X")
        self.lo = _dspin(1.0, 0, 100, 1, 0.5)
        self.hi = _dspin(99.5, 0, 100, 1, 0.5)
        self.slab = _ispin(0, 0, 500)
        self.phys = QtWidgets.QCheckBox("physical aspect in side views")
        self.mpl = MplCanvas(figsize=(10, 7))
        ctrl = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("contrast pct"), self.lo, self.hi,
                  QtWidgets.QLabel("XY slab ±slices (MIP)"), self.slab, self.phys):
            ctrl.addWidget(w)
        ctrl.addStretch()
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(ctrl)
        for s in (self.sz, self.sy, self.sx):
            lay.addWidget(s)
        lay.addWidget(self.mpl, 1)
        for s in (self.sz, self.sy, self.sx):
            s.valueChanged.connect(self.update_slices)
        self.slab.valueChanged.connect(self.update_slices)
        self.lo.valueChanged.connect(self._contrast)
        self.hi.valueChanged.connect(self._contrast)
        self.phys.toggled.connect(self._aspects)
        self.mpl.canvas.mpl_connect('button_press_event', self._click)

    def set_view(self, v):
        self.v = v
        self.mpl.fig.clear()
        if v is None:
            self.mpl.draw()
            return
        z, y, x = v.shape
        for s, n in zip((self.sz, self.sy, self.sx), v.shape):
            s.set_range(n)
        self.sample = v.preview(3_000_000)[0].ravel().astype(np.float32)
        gs = self.mpl.fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
        self.ax_xy = self.mpl.fig.add_subplot(gs[:, 0])
        self.ax_xz = self.mpl.fig.add_subplot(gs[0, 1])
        self.ax_yz = self.mpl.fig.add_subplot(gs[1, 1])
        self.im = {}
        for name, ax, shape in (('xy', self.ax_xy, (y, x)), ('xz', self.ax_xz, (z, x)), ('yz', self.ax_yz, (z, y))):
            self.im[name] = ax.imshow(np.zeros(shape, np.float32), cmap='gray', interpolation='nearest')
        self.ax_xy.set(title="XY (C-plane)", xlabel="X", ylabel="Y")
        self.ax_xz.set(title="XZ", xlabel="X", ylabel="Z")
        self.ax_yz.set(title="YZ", xlabel="Y", ylabel="Z")
        kw = dict(color='yellow', lw=0.6, alpha=0.7)
        self.cross = {'xy': (self.ax_xy.axhline(0, **kw), self.ax_xy.axvline(0, **kw)),
                      'xz': (self.ax_xz.axhline(0, **kw), self.ax_xz.axvline(0, **kw)),
                      'yz': (self.ax_yz.axhline(0, **kw), self.ax_yz.axvline(0, **kw))}
        self.ax_hist = self.ax_xy.inset_axes([0.0, -0.22, 1.0, 0.12])
        self._aspects()
        self._contrast()
        self.update_slices()

    def _aspects(self):
        if self.v is None:
            return
        dz, dy, dx = self.v.spacing
        self.ax_xy.set_aspect(dy / dx)
        self.ax_xz.set_aspect(dz / dx if self.phys.isChecked() else 'auto')
        self.ax_yz.set_aspect(dz / dy if self.phys.isChecked() else 'auto')
        self.mpl.draw()

    def _contrast(self):
        if self.v is None:
            return
        vmin, vmax = np.percentile(self.sample, (self.lo.value(), self.hi.value()))
        for im in self.im.values():
            im.set_clim(vmin, vmax)
        h = self.ax_hist
        h.clear()
        h.hist(self.sample, bins=128, color='0.5', log=True)
        h.axvline(vmin, color='r', lw=0.8)
        h.axvline(vmax, color='r', lw=0.8)
        h.set_yticks([])
        h.tick_params(labelsize=7)
        self.mpl.draw()

    def update_slices(self):
        if self.v is None:
            return
        z, y, x = self.sz.value(), self.sy.value(), self.sx.value()
        n = self.slab.value()
        if n:
            nz = self.v.shape[0]
            xy = self.v.read([slice(max(0, z - n), min(nz, z + n + 1)), slice(None), slice(None)]).max(axis=0)
        else:
            xy = self.v.slice(0, z)
        self.im['xy'].set_data(xy)
        self.im['xz'].set_data(self.v.slice(1, y))
        self.im['yz'].set_data(self.v.slice(2, x))
        for name, (h, vline), (hv, vv) in (('xy', self.cross['xy'], (y, x)),
                                           ('xz', self.cross['xz'], (z, x)),
                                           ('yz', self.cross['yz'], (z, y))):
            h.set_ydata([hv, hv])
            vline.set_xdata([vv, vv])
        self.ax_xy.set_title(f"XY at Z={z}" + (f" (MIP ±{n})" if n else ""))
        self.ax_xz.set_title(f"XZ at Y={y}")
        self.ax_yz.set_title(f"YZ at X={x}")
        self.mpl.draw()

    def _click(self, ev):
        if self.v is None or ev.inaxes is None or ev.xdata is None or self.mpl.navigating():
            return
        c, r = int(round(ev.xdata)), int(round(ev.ydata))
        if ev.inaxes is self.ax_xy:
            self.sx.setValue(c), self.sy.setValue(r)
        elif ev.inaxes is self.ax_xz:
            self.sx.setValue(c), self.sz.setValue(r)
        elif ev.inaxes is self.ax_yz:
            self.sy.setValue(c), self.sz.setValue(r)


# ---------------------------------------------------------- UT inspector

class UTInspector(QtWidgets.QWidget):
    MODES = ("Single depth", "Gate: max amplitude", "Gate: depth of max (TOF)")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.v = None
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(self.MODES)
        self.mode.setCurrentIndex(1)
        self.sz = IndexSlider("depth Z")
        self.g0, self.g1 = IndexSlider("gate start"), IndexSlider("gate end")
        self.y, self.x = _ispin(0, 0, 0), _ispin(0, 0, 0)
        self.lo = _dspin(1.0, 0, 100, 1, 0.5)
        self.hi = _dspin(99.5, 0, 100, 1, 0.5)
        self.mpl = MplCanvas(figsize=(11, 7))
        ctrl = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("C-scan"), self.mode, QtWidgets.QLabel("A-scan Y"), self.y,
                  QtWidgets.QLabel("X"), self.x, QtWidgets.QLabel("contrast pct"), self.lo, self.hi):
            ctrl.addWidget(w)
        ctrl.addStretch()
        hint = QtWidgets.QLabel("Click the C-scan to pick an A-scan; click a B-scan or the A-scan to move the depth cursor.")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(ctrl)
        for s in (self.sz, self.g0, self.g1):
            lay.addWidget(s)
        lay.addWidget(hint)
        lay.addWidget(self.mpl, 1)
        self.mode.currentIndexChanged.connect(self.redraw)
        for s in (self.sz, self.g0, self.g1):
            s.valueChanged.connect(self.redraw)
        for w in (self.y, self.x):
            w.valueChanged.connect(self.redraw)
        self.lo.valueChanged.connect(self._clims)
        self.hi.valueChanged.connect(self._clims)
        self.mpl.canvas.mpl_connect('button_press_event', self._click)

    def set_view(self, v):
        self.mpl.fig.clear()
        self.v = None
        if v is None:
            self.mpl.draw()
            return
        if np.prod(v.shape) * 4 < 2e9:  # UT volumes are small: keep them in memory
            v = VolumeView(v.to_numpy().astype(np.float32), spacing=v.spacing, name=v.name, meta=v.meta)
        nz, ny, nx = v.shape
        self.sample = v.preview(3_000_000)[0].ravel().astype(np.float32)
        self.sz.set_range(nz)
        self.g0.set_range(nz, 0)
        self.g1.set_range(nz, nz - 1)
        for w, n in ((self.y, ny), (self.x, nx)):
            w.blockSignals(True)
            w.setRange(0, n - 1)
            w.setValue(n // 2)
            w.blockSignals(False)
        f = self.mpl.fig
        gs = f.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
        self.ax_c = f.add_subplot(gs[:, 0])
        self.ax_a = f.add_subplot(gs[0, 1:])
        self.ax_bx = f.add_subplot(gs[1, 1])
        self.ax_by = f.add_subplot(gs[1, 2])
        dz = v.spacing[0]
        self.ax_a.secondary_xaxis('top', functions=(lambda s: s * dz, lambda m: m / dz)).set_xlabel("depth (mm)")
        self.im_c = self.ax_c.imshow(np.zeros((ny, nx)), cmap='viridis', interpolation='nearest',
                                     aspect=v.spacing[1] / v.spacing[2])
        self.cbar = f.colorbar(self.im_c, ax=self.ax_c, shrink=0.6)
        self.mark = self.ax_c.plot([0], [0], '+', color='r', ms=12, mew=1.5)[0]
        self.im_bx = self.ax_bx.imshow(np.zeros((nz, nx)), cmap='gray', aspect='auto', interpolation='nearest')
        self.im_by = self.ax_by.imshow(np.zeros((nz, ny)), cmap='gray', aspect='auto', interpolation='nearest')
        self.line_a = self.ax_a.plot(np.arange(nz), np.zeros(nz), lw=0.9)[0]
        self.ax_a.set(xlabel="depth (samples)", ylabel="amplitude", xlim=(0, nz - 1))
        self.ax_bx.set(xlabel="X", ylabel="Z")
        self.ax_by.set(xlabel="Y", ylabel="Z")
        self.vlines = {}
        for ax in (self.ax_a,):
            self.vlines['a'] = [ax.axvline(0, color=c, lw=0.8, ls=ls) for c, ls in (('r', '-'), ('orange', '--'), ('orange', '--'))]
        for key, ax in (('bx', self.ax_bx), ('by', self.ax_by)):
            self.vlines[key] = [ax.axhline(0, color=c, lw=0.8, ls=ls) for c, ls in (('r', '-'), ('orange', '--'), ('orange', '--'))]
            self.vlines[key + '_lat'] = ax.axvline(0, color='c', lw=0.6)
        self.v = v
        self._clims()
        self.redraw()

    def _clims(self):
        if self.v is None:
            return
        vmin, vmax = np.percentile(self.sample, (self.lo.value(), self.hi.value()))
        for im in (self.im_bx, self.im_by):
            im.set_clim(vmin, vmax)
        self.redraw()

    def _cscan(self):
        z, g0, g1 = self.sz.value(), self.g0.value(), self.g1.value()
        g0, g1 = min(g0, g1), max(g0, g1) + 1
        mode = self.mode.currentIndex()
        if mode == 0:
            return self.v.slice(0, z), f"C-scan at Z={z} ({z * self.v.spacing[0]:.3f} mm)", 'amplitude'
        gated = self.v.read([slice(g0, g1), slice(None), slice(None)])
        if mode == 1:
            return gated.max(axis=0), f"max amplitude in gate [{g0}, {g1 - 1}]", 'amplitude'
        return (gated.argmax(axis=0) + g0) * self.v.spacing[0], f"depth of max in gate [{g0}, {g1 - 1}]", 'depth (mm)'

    def redraw(self):
        if self.v is None:
            return
        y, x, z = self.y.value(), self.x.value(), self.sz.value()
        g0, g1 = self.g0.value(), self.g1.value()
        c, title, label = self._cscan()
        self.im_c.set_data(c)
        if self.mode.currentIndex() == 2:
            self.im_c.set_clim(*np.percentile(c, (1, 99)))
        else:
            self.im_c.set_clim(*np.percentile(c, (self.lo.value(), self.hi.value())))
        self.cbar.set_label(label)
        self.ax_c.set_title(title, fontsize=10)
        self.mark.set_data([x], [y])
        a = self.v.read([slice(None), y, x])
        self.line_a.set_ydata(a)
        self.ax_a.set_ylim(float(a.min()), float(a.max()) * 1.05 + 1e-9)
        self.ax_a.set_title(f"A-scan at Y={y}, X={x}", fontsize=10)
        for ln, val in zip(self.vlines['a'], (z, g0, g1)):
            ln.set_xdata([val, val])
        self.im_bx.set_data(self.v.slice(1, y))
        self.im_by.set_data(self.v.slice(2, x))
        self.ax_bx.set_title(f"B-scan along X at Y={y}", fontsize=10)
        self.ax_by.set_title(f"B-scan along Y at X={x}", fontsize=10)
        for key, lat in (('bx', x), ('by', y)):
            for ln, val in zip(self.vlines[key], (z, g0, g1)):
                ln.set_ydata([val, val])
            self.vlines[key + '_lat'].set_xdata([lat, lat])
        self.mpl.draw()

    def _click(self, ev):
        if self.v is None or ev.inaxes is None or ev.xdata is None or self.mpl.navigating():
            return
        c, r = int(round(ev.xdata)), int(round(ev.ydata))
        if ev.inaxes is self.ax_c:
            self.y.setValue(r), self.x.setValue(c)
        elif ev.inaxes is self.ax_bx:
            self.x.setValue(c), self.sz.setValue(r)
        elif ev.inaxes is self.ax_by:
            self.y.setValue(c), self.sz.setValue(r)
        elif ev.inaxes is self.ax_a:
            self.sz.setValue(c)


# ------------------------------------------------------- Registration tab

class StageForm(QtWidgets.QGroupBox):
    def __init__(self, title, sc: StageConfig, parent=None):
        super().__init__(title, parent)
        self.metric = QtWidgets.QComboBox()
        self.metric.addItems(METRICS)
        self.bins = _ispin(32, 2, 4096)
        self.sampling = _dspin(15, 0.1, 100, 1, 5)
        self.jh_var = _dspin(1.5, 0.01, 100, 2, 0.1)
        self.radius = _ispin(4, 1, 50)
        self.shrink = QtWidgets.QLineEdit()
        self.sigmas = QtWidgets.QLineEdit()
        self.iters = _ispin(300, 1, 10_000_000)
        self.lr = _dspin(1.0, 1e-6, 1e3, 4)
        self.min_step = QtWidgets.QLineEdit()
        self.relax = _dspin(0.5, 0.01, 0.99, 2, 0.05)
        self.dof = [QtWidgets.QCheckBox(d) for d in DOF]
        f = QtWidgets.QFormLayout(self)
        f.addRow("metric", self.metric)
        f.addRow("histogram bins", self.bins)
        f.addRow("sampling % (100 = all)", self.sampling)
        f.addRow("JH-MI variance", self.jh_var)
        f.addRow("ANTS NC radius", self.radius)
        f.addRow("shrink factors", self.shrink)
        f.addRow("smoothing σ (mm)", self.sigmas)
        f.addRow("max iterations / level", self.iters)
        f.addRow("learning rate (max step mm)", self.lr)
        f.addRow("min step", self.min_step)
        f.addRow("relaxation", self.relax)
        row = QtWidgets.QHBoxLayout()
        for cb in self.dof:
            row.addWidget(cb)
        f.addRow("free DOF", row)
        self.set(sc)

    def set(self, sc: StageConfig):
        self.metric.setCurrentText(sc.metric)
        self.bins.setValue(sc.bins)
        self.sampling.setValue(min(100.0, sc.sampling * 100))
        self.jh_var.setValue(sc.jh_variance)
        self.radius.setValue(sc.ants_radius)
        self.shrink.setText(', '.join(str(s) for s in sc.shrink))
        self.sigmas.setText(', '.join(str(s) for s in sc.sigmas_mm))
        self.iters.setValue(sc.max_iters)
        self.lr.setValue(sc.learning_rate)
        self.min_step.setText(f"{sc.min_step:g}")
        self.relax.setValue(sc.relaxation)
        for cb, v in zip(self.dof, sc.free_dof):
            cb.setChecked(bool(v))

    def get(self) -> StageConfig:
        return StageConfig(
            metric=self.metric.currentText(), bins=self.bins.value(), sampling=self.sampling.value() / 100,
            jh_variance=self.jh_var.value(), ants_radius=self.radius.value(),
            shrink=[int(s) for s in self.shrink.text().replace(',', ' ').split()],
            sigmas_mm=[float(s) for s in self.sigmas.text().replace(',', ' ').split()],
            max_iters=self.iters.value(), learning_rate=self.lr.value(), min_step=float(self.min_step.text()),
            relaxation=self.relax.value(), free_dof=[cb.isChecked() for cb in self.dof])


class ConfigForm(QtWidgets.QWidget):
    def __init__(self, cfg: PipelineConfig, parent=None):
        super().__init__(parent)
        self.ct_lo, self.ct_hi = _dspin(2, 0, 100, 1), _dspin(98, 0, 100, 1)
        self.us_lo, self.us_hi = _dspin(0, 0, 100, 1), _dspin(100, 0, 100, 1)
        self.us_sigma = _dspin(0.5, 0, 50, 2, 0.25)
        self.us_log = QtWidgets.QCheckBox("log compression")
        self.gate = _dspin(0.5, 0, 1000, 3, 0.1)
        self.s1_sp = _dspin(0.8, 0.001, 100, 3, 0.1)
        self.s2_inplane = _dspin(0.25, 0.0, 100, 3, 0.05)
        self.tz = QtWidgets.QCheckBox("initialise tz from UT front wall vs XCT top surface")
        self.rot90 = QtWidgets.QCheckBox("try XCT rotated 0/90/180/270° in-plane and keep the best fit")
        self.mmask = QtWidgets.QCheckBox("use XCT mask in the metric too (overlap varies with the transform)")
        self.seed = _ispin(78, 0, 2**31 - 1)
        self.threads = _ispin(0, 0, 1024)
        self.s3_en = QtWidgets.QCheckBox("enabled")
        self.s3_rmm, self.s3_smm = _dspin(1, 0, 100, 3, 0.25), _dspin(0.25, 0.001, 100, 3, 0.05)
        self.s3_rdeg, self.s3_sdeg = _dspin(0.5, 0, 45, 3, 0.1), _dspin(0.1, 0.001, 45, 3, 0.05)
        self.s3_ovl = _dspin(0.9, 0, 1, 2, 0.05)
        self.s3_count = QtWidgets.QLabel()

        pre = QtWidgets.QGroupBox("Preprocessing & Stage 0")
        f = QtWidgets.QFormLayout(pre)
        f.addRow("XCT clip percentiles (inside mask)", self._pair(self.ct_lo, self.ct_hi))
        f.addRow("UT clip percentiles", self._pair(self.us_lo, self.us_hi))
        f.addRow("UT smoothing σ (voxels)", self.us_sigma)
        f.addRow("", self.us_log)
        f.addRow("UT gate: skip first … mm (front wall, mask)", self.gate)
        f.addRow("Stage 1 isotropic spacing (mm)", self.s1_sp)
        f.addRow("Stage 2 XCT in-plane spacing (mm, 0 = UT)", self.s2_inplane)
        f.addRow("", self.tz)
        f.addRow("", self.rot90)
        f.addRow("", self.mmask)
        f.addRow("random seed", self.seed)
        f.addRow("ITK threads (0 = all, 1 = reproducible)", self.threads)
        self.s1 = StageForm("Stage 1 — coarse", cfg.stage1)
        self.s2 = StageForm("Stage 2 — fine (UT native grid); also the evaluation metric", cfg.stage2)
        s3 = QtWidgets.QGroupBox("Stage 3 — local exhaustive grid around the best result")
        f3 = QtWidgets.QFormLayout(s3)
        f3.addRow("", self.s3_en)
        f3.addRow("tx/ty range ± / step (mm)", self._pair(self.s3_rmm, self.s3_smm))
        f3.addRow("rz range ± / step (deg)", self._pair(self.s3_rdeg, self.s3_sdeg))
        f3.addRow("min overlap (fraction of max)", self.s3_ovl)
        f3.addRow("", self.s3_count)
        for w in (self.s3_rmm, self.s3_smm, self.s3_rdeg, self.s3_sdeg):
            w.valueChanged.connect(self._count)

        lay = QtWidgets.QGridLayout(self)
        lay.addWidget(pre, 0, 0)
        lay.addWidget(s3, 1, 0)
        lay.addWidget(self.s1, 0, 1, 2, 1)
        lay.addWidget(self.s2, 0, 2, 2, 1)
        self.set(cfg)

    @staticmethod
    def _pair(a, b):
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a)
        h.addWidget(b)
        return w

    def _count(self):
        nt = int(round(self.s3_rmm.value() / self.s3_smm.value()))
        nr = int(round(self.s3_rdeg.value() / self.s3_sdeg.value()))
        self.s3_count.setText(f"{(2 * nt + 1) ** 2 * (2 * nr + 1):,} metric evaluations")

    def set(self, c: PipelineConfig):
        self.ct_lo.setValue(c.ct_clip[0]), self.ct_hi.setValue(c.ct_clip[1])
        self.us_lo.setValue(c.us_clip[0]), self.us_hi.setValue(c.us_clip[1])
        self.us_sigma.setValue(c.us_sigma_vox)
        self.us_log.setChecked(c.us_log)
        self.gate.setValue(c.us_gate_mm)
        self.s1_sp.setValue(c.stage1_spacing_mm)
        self.s2_inplane.setValue(c.stage2_inplane_mm)
        self.tz.setChecked(c.tz_from_surfaces)
        self.rot90.setChecked(c.search_rot90)
        self.mmask.setChecked(c.moving_mask)
        self.seed.setValue(c.seed)
        self.threads.setValue(c.threads)
        self.s1.set(c.stage1)
        self.s2.set(c.stage2)
        s3 = c.stage3
        self.s3_en.setChecked(s3.enabled)
        self.s3_rmm.setValue(s3.range_mm), self.s3_smm.setValue(s3.step_mm)
        self.s3_rdeg.setValue(s3.range_deg), self.s3_sdeg.setValue(s3.step_deg)
        self.s3_ovl.setValue(s3.min_overlap)
        self._count()

    def get(self) -> PipelineConfig:
        return PipelineConfig(
            ct_clip=[self.ct_lo.value(), self.ct_hi.value()], us_clip=[self.us_lo.value(), self.us_hi.value()],
            us_sigma_vox=self.us_sigma.value(), us_log=self.us_log.isChecked(),
            us_gate_mm=self.gate.value(), stage1_spacing_mm=self.s1_sp.value(),
            stage2_inplane_mm=self.s2_inplane.value(), tz_from_surfaces=self.tz.isChecked(), search_rot90=self.rot90.isChecked(),
            moving_mask=self.mmask.isChecked(),
            seed=self.seed.value(), threads=self.threads.value(), stage1=self.s1.get(), stage2=self.s2.get(),
            stage3=Stage3Config(self.s3_en.isChecked(), self.s3_rmm.value(), self.s3_smm.value(),
                                self.s3_rdeg.value(), self.s3_sdeg.value(), self.s3_ovl.value()))


def plot_cost_log(fig, cost_log):
    fig.clear()
    stages = [s for s in ('Stage 1', 'Stage 2') if any(e[0] == s for e in cost_log)]
    if not stages:
        return
    axes = np.atleast_1d(fig.subplots(1, len(stages)))
    for ax, st in zip(axes, stages):
        entries = [e for e in cost_log if e[0] == st]
        for lvl in sorted({e[1] for e in entries}):
            c = [e[3] for e in entries if e[1] == lvl]
            ax.plot(np.arange(len(c)), c, lw=1, label=f"level {lvl + 1}")
        ax.set(title=f"{st} cost (lower is better)", xlabel="iteration")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)


class PipelineWorker(QtCore.QObject):
    log = pyqtSignal(str)
    cost = pyqtSignal(str, int, int, float)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, ct_view, ut_view, cfg):
        super().__init__()
        self.ct_view, self.ut_view, self.cfg = ct_view, ut_view, cfg
        self.pipeline = None
        self._cancel = False

    def run(self):
        try:
            self.log.emit("Reading XCT into memory…")
            ct = self.ct_view.to_numpy()
            ut = self.ut_view.to_numpy()
            if self._cancel:
                raise Cancelled()
            self.pipeline = RegistrationPipeline(ct, self.ct_view.spacing, ut, self.ut_view.spacing, self.cfg,
                                                 log=self.log.emit, on_cost=self.cost.emit)
            del ct, ut
            self.finished.emit(self.pipeline.run())
        except Cancelled:
            self.failed.emit("Cancelled.")
        except Exception:
            self.failed.emit(traceback.format_exc())

    def cancel(self):
        self._cancel = True
        if self.pipeline:
            self.pipeline.cancel()


class RegistrationTab(QtWidgets.QWidget):
    resultReady = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.views = {'XCT': None, 'UT': None}
        self.thread = self.worker = None
        self.form = ConfigForm(PipelineConfig())
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)
        self.run_btn = QtWidgets.QPushButton("Run registration")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        defaults = QtWidgets.QPushButton("Defaults")
        save_cfg, load_cfg = QtWidgets.QPushButton("Save config…"), QtWidgets.QPushButton("Load config…")
        self.status = QtWidgets.QLabel()
        self.logbox = QtWidgets.QPlainTextEdit()
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(20000)
        self.mpl = MplCanvas(figsize=(8, 3))
        bar = QtWidgets.QHBoxLayout()
        for w in (self.run_btn, self.cancel_btn, defaults, save_cfg, load_cfg, self.status):
            bar.addWidget(w)
        bar.addStretch()
        bottom = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        bottom.addWidget(self.logbox)
        bottom.addWidget(self.mpl)
        split = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        split.addWidget(scroll)
        split.addWidget(bottom)
        split.setSizes([450, 350])
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(split, 1)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn.clicked.connect(self.cancel)
        defaults.clicked.connect(lambda: self.form.set(PipelineConfig()))
        save_cfg.clicked.connect(self._save_cfg)
        load_cfg.clicked.connect(self._load_cfg)
        self.cost_log = []
        self._dirty = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._replot)
        self.timer.start(500)

    def set_view(self, which, v):
        self.views[which] = v

    def _save_cfg(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save config", "registration_config.json", "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(asdict(self.form.get()), indent=2))

    def _load_cfg(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load config", "", "JSON (*.json)")
        if path:
            d = json.loads(Path(path).read_text())
            self.form.set(PipelineConfig.from_dict(d.get('config', d)))

    def append_log(self, text):
        self.logbox.appendPlainText(text)

    def _on_cost(self, stage, level, it, cost):
        if stage != 'Stage 3':
            self.cost_log.append((stage, level, it, cost))
            self._dirty = True
            self.status.setText(f"{stage} level {level + 1} iter {it}: cost {cost:.6f}")

    def _replot(self):
        if self._dirty:
            self._dirty = False
            plot_cost_log(self.mpl.fig, self.cost_log)
            self.mpl.draw()

    def run(self):
        missing = [k for k, v in self.views.items() if v is None]
        if missing:
            QtWidgets.QMessageBox.warning(self, "Missing input", f"Load the {' and '.join(missing)} volume first.")
            return
        try:
            cfg = self.form.get()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Invalid parameter", str(e))
            return
        self.cost_log = []
        self.logbox.clear()
        self.append_log(f"XCT: {self.views['XCT'].describe()}\nUT:  {self.views['UT'].describe()}")
        self.thread = QtCore.QThread(self)
        self.worker = PipelineWorker(self.views['XCT'], self.views['UT'], cfg)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append_log)
        self.worker.cost.connect(self._on_cost)
        self.worker.finished.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("running…")
        self.thread.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("cancelling…")

    def _finish(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _done(self, res):
        self._finish()
        self.status.setText(f"done — final: {res.final}" + (f"  ({len(res.warnings)} warnings)" if res.warnings else ""))
        self.resultReady.emit(res)
        if res.warnings:
            QtWidgets.QMessageBox.warning(self, "Registration warnings", "\n\n".join(res.warnings))

    def _failed(self, msg):
        self._finish()
        self.status.setText("stopped")
        self.append_log(msg)


# ------------------------------------------------------------ Results tab

def blend(us, ct, mode, alpha=0.5, checker=8):
    us, ct = np.clip(us, 0, 1), np.clip(ct, 0, 1)
    if mode == "Magenta / green":
        return np.dstack([us, ct, us])
    if mode == "Alpha blend":
        g = (1 - alpha) * us + alpha * ct
        return np.dstack([g, g, g])
    if mode == "Checkerboard":
        yy, xx = np.indices(us.shape)
        sel = ((yy // checker + xx // checker) % 2).astype(bool)
        g = np.where(sel, ct, us)
        return np.dstack([g, g, g])
    g = us if mode == "UT only" else ct
    return np.dstack([g, g, g])


class ResultsTab(QtWidgets.QWidget):
    BLENDS = ("Magenta / green", "Alpha blend", "Checkerboard", "UT only", "XCT only")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.res = None
        self.inputs = {}
        self.stage = QtWidgets.QComboBox()
        self.blend = QtWidgets.QComboBox()
        self.blend.addItems(self.BLENDS)
        self.alpha = _dspin(0.5, 0, 1, 2, 0.1)
        self.checker = _ispin(8, 1, 500)
        self.contours = QtWidgets.QCheckBox("footprint contours")
        self.contours.setChecked(True)
        self.project = QtWidgets.QCheckBox("XY: project part interior")
        self.project.setToolTip("UT: max amplitude, XCT: mean, over the central 70 % of the warped XCT "
                                "thickness (front/back-wall echoes excluded)")
        self.save_btn = QtWidgets.QPushButton("Save results…")
        self.sz, self.sy, self.sx = IndexSlider("Z"), IndexSlider("Y"), IndexSlider("X")
        self.table = QtWidgets.QTableWidget()
        self.table.setMaximumHeight(170)
        self.mpl = MplCanvas(figsize=(11, 6))
        self.grid_mpl = MplCanvas(figsize=(10, 5))
        self.conv_mpl = MplCanvas(figsize=(10, 4))
        ctrl = QtWidgets.QHBoxLayout()
        for w in (QtWidgets.QLabel("stage"), self.stage, QtWidgets.QLabel("blend"), self.blend,
                  QtWidgets.QLabel("alpha"), self.alpha, QtWidgets.QLabel("checker px"), self.checker,
                  self.contours, self.project, self.save_btn):
            ctrl.addWidget(w)
        ctrl.addStretch()
        overlay = QtWidgets.QWidget()
        ol = QtWidgets.QVBoxLayout(overlay)
        ol.addLayout(ctrl)
        for s in (self.sz, self.sy, self.sx):
            ol.addWidget(s)
        ol.addWidget(self.mpl, 1)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(overlay, "Overlay (UT space)")
        tabs.addTab(self.grid_mpl, "Stage 3 cost surface")
        tabs.addTab(self.conv_mpl, "Convergence")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(tabs, 1)
        lay.addWidget(self.table)
        for w in (self.sz, self.sy, self.sx):
            w.valueChanged.connect(self.redraw)
        self.stage.currentIndexChanged.connect(self.redraw)
        self.blend.currentIndexChanged.connect(self.redraw)
        self.alpha.valueChanged.connect(self.redraw)
        self.checker.valueChanged.connect(self.redraw)
        self.contours.toggled.connect(self.redraw)
        self.project.toggled.connect(self.redraw)
        self.save_btn.clicked.connect(self._save)
        self.mpl.canvas.mpl_connect('button_press_event', self._click)

    def set_result(self, res: PipelineResult, inputs=None):
        self.res = res
        self.inputs = inputs or {}
        self.stage.blockSignals(True)
        self.stage.clear()
        for s in res.stages:
            self.stage.addItem(s.name + (" (final)" if s.name == res.final else ""), s.name)
        self.stage.setCurrentIndex([s.name for s in res.stages].index(res.final))
        self.stage.blockSignals(False)
        self.us = sitk.GetArrayFromImage(res.us_fine)
        nz, ny, nx = self.us.shape
        cover = res.warp_ct_mask(res.final).sum(axis=(1, 2))
        self.sz.set_range(nz, int(cover.argmax()) if cover.any() else nz // 2)
        self.sy.set_range(ny)
        self.sx.set_range(nx)
        f = self.mpl.fig
        f.clear()
        gs = f.add_gridspec(2, 2, width_ratios=[1, 1.3])
        self.ax_xy = f.add_subplot(gs[:, 0])
        self.ax_xz = f.add_subplot(gs[0, 1])
        self.ax_yz = f.add_subplot(gs[1, 1])
        self._fill_table()
        self._plot_grid()
        plot_cost_log(self.conv_mpl.fig, res.cost_log)
        self.conv_mpl.draw()
        self.redraw()

    def _fill_table(self):
        cols = ['stage', 'rz °', 'tx mm', 'ty mm', 'tz mm', 'rx °', 'ry °', 'cost', 'overlap',
                'footprint Dice', 'iters', 'note / stop']
        rows = self.res.stages
        self.table.setColumnCount(len(cols))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(cols)
        for r, s in enumerate(rows):
            p = s.params()
            vals = [s.name + (" ★" if s.name == self.res.final else ""),
                    f"{p['rz_deg']:+.3f}", f"{p['tx_mm']:+.3f}", f"{p['ty_mm']:+.3f}", f"{p['tz_mm']:+.3f}",
                    f"{p['rx_deg']:+.3f}", f"{p['ry_deg']:+.3f}", f"{s.cost:.6f}", f"{s.overlap:,}",
                    f"{s.footprint_dice:.3f}", str(s.iterations), s.note or s.stop]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

    def _plot_grid(self):
        f = self.grid_mpl.fig
        f.clear()
        g = self.res.grid
        if g is None:
            f.text(0.5, 0.5, "Stage 3 was disabled", ha='center')
            self.grid_mpl.draw()
            return
        a1, a2 = f.subplots(1, 2, width_ratios=[1.3, 1])
        ir, iy, ix = g['best_index']
        tx, ty = g['tx_mm'], g['ty_mm']
        st = (tx[1] - tx[0]) / 2 if len(tx) > 1 else 0.5
        im = a1.imshow(g['cost'][ir], origin='lower', cmap='viridis_r',
                       extent=[tx[0] - st, tx[-1] + st, ty[0] - st, ty[-1] + st])
        f.colorbar(im, ax=a1, label="cost (lower is better)")
        c = len(tx) // 2
        a1.plot(tx[c], ty[c], 'wx', ms=10, label='grid centre (best before Stage 3)')
        a1.plot(tx[ix], ty[iy], 'r+', ms=12, mew=2, label='grid best')
        a1.set(title=f"cost over tx, ty at rz = {g['rz_deg'][ir]:+.2f}°  (blank = low overlap)",
               xlabel="tx (mm)", ylabel="ty (mm)")
        a1.legend(fontsize=8)
        with np.errstate(all='ignore'):
            best_per_rz = np.nanmin(g['cost'].reshape(len(g['rz_deg']), -1), axis=1)
        a2.plot(g['rz_deg'], best_per_rz, 'o-')
        a2.axvline(g['rz_deg'][len(g['rz_deg']) // 2], color='k', ls='--', lw=0.8, label='grid centre')
        a2.set(title="best cost per rz", xlabel="rz (deg)", ylabel="cost")
        a2.legend(fontsize=8)
        a2.grid(alpha=0.3)
        self.grid_mpl.draw()

    def redraw(self):
        if self.res is None:
            return
        name = self.stage.currentData()
        ct = self.res.warp_ct(name)
        ctm = self.res.warp_ct_mask(name)
        z, y, x = self.sz.value(), self.sy.value(), self.sx.value()
        mode, a, k = self.blend.currentText(), self.alpha.value(), self.checker.value()
        sp = self.res.us_fine.GetSpacing()  # x, y, z
        us_xy, ct_xy, xy_title = self.us[z], ct[z], f"XY at Z={z} ({z * sp[2]:.3f} mm)"
        cover = ctm.sum(axis=(1, 2))
        if self.project.isChecked() and cover.any():
            rows = np.flatnonzero(cover > 0.5 * cover.max())
            trim = int(0.15 * len(rows))
            z0, z1 = rows[trim], rows[-1 - trim] + 1
            us_xy, ct_xy = self.us[z0:z1].max(axis=0), ct[z0:z1].mean(axis=0)
            us_xy = np.clip(us_xy / max(np.percentile(us_xy, 99.5), 1e-9), 0, 1)
            lo, hi = np.percentile(ct_xy[ctm[z0:z1].any(axis=0)], (1, 99)) if ctm.any() else (0, 1)
            ct_xy = np.clip((ct_xy - lo) / max(hi - lo, 1e-9), 0, 1)
            xy_title = f"XY projection over Z {z0}–{z1 - 1} (UT max, XCT mean)"
        views = ((self.ax_xy, us_xy, ct_xy, xy_title, sp[1] / sp[0], (y, x)),
                 (self.ax_xz, self.us[:, y], ct[:, y], f"XZ at Y={y}", 'auto', (z, x)),
                 (self.ax_yz, self.us[:, :, x], ct[:, :, x], f"YZ at X={x}", 'auto', (z, y)))
        for ax, u, c, title, aspect, (hv, vv) in views:
            ax.clear()
            ax.imshow(blend(u, c, mode, a, k), aspect=aspect, interpolation='nearest')
            ax.axhline(hv, color='y', lw=0.5, alpha=0.6)
            ax.axvline(vv, color='y', lw=0.5, alpha=0.6)
            ax.set_title(title, fontsize=10)
        if self.contours.isChecked():
            self.ax_xy.contour(self.res.us_footprint.astype(float), [0.5], colors='cyan', linewidths=0.8)
            if ctm.any():
                self.ax_xy.contour(ctm.any(axis=0).astype(float), [0.5], colors='orange', linewidths=0.8)
        legend = {"Magenta / green": "magenta = UT, green = XCT", "Checkerboard": "tiles alternate UT / XCT"}
        self.ax_xy.set_xlabel(legend.get(mode, '') + ("   cyan = UT footprint, orange = XCT footprint"
                                                       if self.contours.isChecked() else ''), fontsize=8)
        self.mpl.draw()

    def _click(self, ev):
        if self.res is None or ev.inaxes is None or ev.xdata is None or self.mpl.navigating():
            return
        c, r = int(round(ev.xdata)), int(round(ev.ydata))
        if ev.inaxes is self.ax_xy:
            self.sx.setValue(c), self.sy.setValue(r)
        elif ev.inaxes is self.ax_xz:
            self.sx.setValue(c), self.sz.setValue(r)
        elif ev.inaxes is self.ax_yz:
            self.sy.setValue(c), self.sz.setValue(r)

    def _save(self):
        if self.res is None:
            return
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory")
        if d:
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                save_results(self.res, d, self.inputs)
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.information(self, "Saved", f"Results written to\n{d}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XCT ↔ UT registration")
        self.resize(1500, 950)
        self.load_tab = LoadTab()
        self.xct_tab = XCTInspector()
        self.ut_tab = UTInspector()
        self.reg_tab = RegistrationTab()
        self.res_tab = ResultsTab()
        self.tabs = QtWidgets.QTabWidget()
        for w, name in ((self.load_tab, "1 · Load"), (self.xct_tab, "2 · XCT inspector"),
                        (self.ut_tab, "3 · UT inspector"), (self.reg_tab, "4 · Registration"),
                        (self.res_tab, "5 · Results")):
            self.tabs.addTab(w, name)
        self.setCentralWidget(self.tabs)
        self.load_tab.xct.viewChanged.connect(self.xct_tab.set_view)
        self.load_tab.ut.viewChanged.connect(self.ut_tab.set_view)
        self.load_tab.xct.viewChanged.connect(lambda v: self.reg_tab.set_view('XCT', v))
        self.load_tab.ut.viewChanged.connect(lambda v: self.reg_tab.set_view('UT', v))
        self.reg_tab.resultReady.connect(self._result)

    def _result(self, res):
        inputs = {'xct': self.load_tab.xct.settings(), 'ut': self.load_tab.ut.settings()}
        self.res_tab.set_result(res, inputs)
        self.tabs.setCurrentWidget(self.res_tab)


# =============================================================================
# Entry point
# =============================================================================

def _parse_flips(s):
    s = (s or '').upper()
    return [a in s for a in AXES]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for m, sp in (('xct', (0.025, 0.025, 0.025)), ('ut', (0.016, 1.0, 1.0))):
        ap.add_argument(f'--{m}', help=f"{m.upper()} volume (.npy .npz .tif .tiff .h5)")
        ap.add_argument(f'--{m}-key', help="dataset / array name inside .npz or .h5")
        ap.add_argument(f'--{m}-spacing', type=float, nargs=3, metavar=('DZ', 'DY', 'DX'), default=sp)
        ap.add_argument(f'--{m}-axes', default='0,1,2', help="raw axes giving (Z,Y,X), e.g. 2,1,0")
        ap.add_argument(f'--{m}-flip', default='', help="axes to flip after reordering, e.g. ZX")
        ap.add_argument(f'--{m}-rot90', type=int, default=0, choices=range(4),
                        help="in-plane quarter turns (counter-clockwise) applied after axes/flips")
    ap.add_argument('--ut-envelope', action='store_true', help="UT is RF: take the Hilbert envelope along Z")
    ap.add_argument('--config', help="JSON config (as saved from the GUI or report.json)")
    ap.add_argument('--run-headless', metavar='OUT_DIR', help="run the pipeline without GUI and save results")
    a = ap.parse_args(argv)
    axes = {m: tuple(int(i) for i in getattr(a, f'{m}_axes').split(',')) for m in ('xct', 'ut')}
    cfg = PipelineConfig()
    if a.config:
        d = json.loads(Path(a.config).read_text())
        cfg = PipelineConfig.from_dict(d.get('config', d))

    if a.run_headless:
        views = {}
        for m in ('xct', 'ut'):
            raw = load_raw(getattr(a, m), getattr(a, f'{m}_key'))
            o = rot90_orientation(axes[m], _parse_flips(getattr(a, f'{m}_flip')), getattr(a, f'{m}_spacing'),
                                  getattr(a, f'{m}_rot90'))
            views[m] = VolumeView(raw, *o, name=m)
            print(f"{m.upper()}: {views[m].describe()}")
        ut = views['ut'].to_numpy()
        if a.ut_envelope:
            ut = envelope_z(ut)
        pipe = RegistrationPipeline(views['xct'].to_numpy(), views['xct'].spacing, ut, views['ut'].spacing, cfg)
        res = pipe.run()
        inputs = {m: {'path': getattr(a, m), 'key': getattr(a, f'{m}_key'), 'axis_order': list(axes[m]),
                      'flips_zyx': _parse_flips(getattr(a, f'{m}_flip')),
                      'spacing_zyx_mm': list(getattr(a, f'{m}_spacing')),
                      'rot90': getattr(a, f'{m}_rot90')} for m in ('xct', 'ut')}
        save_results(res, a.run_headless, inputs)
        return 0

    app = QtWidgets.QApplication(sys.argv[:1])
    win = MainWindow()
    win.reg_tab.form.set(cfg)
    for m, box in (('xct', win.load_tab.xct), ('ut', win.load_tab.ut)):
        box.set_orientation(axes[m], _parse_flips(getattr(a, f'{m}_flip')), getattr(a, f'{m}_spacing'),
                            getattr(a, f'{m}_rot90'))
        if m == 'ut' and a.ut_envelope:
            box.envelope.setChecked(True)
        if getattr(a, m):
            box.load(getattr(a, m), getattr(a, f'{m}_key'))
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
