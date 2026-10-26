"""PAUT <-> XCT registration with ``testing/xct_ut_registration_gui.py``: from its result to the database.

The registration itself is done in the GUI (``RegistrationPipeline``: Stage 0 init, Stage 1 coarse,
Stage 2 fine, Stage 3 grid), with whatever orientation and parameters were set there. This module
converts the final Euler3D transform (UT view physical mm -> XCT view physical mm) to the 3x3 matrix
in the database convention of :mod:`.registration` (:data:`registration.CONVENTION`), i.e. relative
to the UT and XCT TIFFs as stored, whatever axis order / flips / rotation the views used. It also
gives the optional start from the through-hole fit and the hole check of a result.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import registration as holes_rg

GUI_PATH = Path(__file__).resolve().parents[2] / 'testing' / 'xct_ut_registration_gui.py'

PIPELINE_OVERRIDES = {'us_gate_mm': 3.5,      # front-wall gate: just before the echo of the processed PAUT
                      'rz_center_deg': 0.0, 'rz_range_deg': 20.0}   # the coupons are within a few degrees of 0°
DEFAULTS = {'xct_axes': [0, 2, 1], 'xct_flip': ''}   # XCT view: the PAUT index/scan axes are the XCT X/Y swapped

REGISTRATION_TYPE = (
    "PAUT-XCT SimpleITK rigid registration 2026 v1, UTvsXCT-preprocessing, testing/xct_ut_registration_gui.py "
    "class RegistrationPipeline (Stage 0 init, Stage 1 coarse, Stage 2 fine, Stage 3 grid; free rz, tx, ty; tz from "
    "the UT entrance echo and the XCT top surface), reviewed in the GUI and converted to the database matrix by "
    "preprocess_tools/paut/registration_sitk.py")


def gui():
    """The registration GUI module (imported once, from its file)."""
    name = 'xct_ut_registration_gui'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, GUI_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


# ----------------------------------------------------------------------------- frames

def view_frame(settings, raw_shape):
    """2D affine of a GUI view: raw [axis 1 mm, axis 2 mm] = G @ [x_view_mm, y_view_mm] + g.

    ``settings``: ``VolumeLoaderBox.settings()`` of the GUI (axis order, flips, spacing, rot90).
    The view must keep depth on raw axis 0; view Y reads raw axis perm[1] and view X raw axis
    perm[2], reversed when flipped (as ``VolumeView``)."""
    perm, flips, spacing = gui().rot90_orientation(settings['axis_order'], settings['flips_zyx'],
                                                   settings['spacing_zyx_mm'], settings['rot90'])
    if perm[0] != 0 or sorted(perm[1:]) != [1, 2]:
        raise ValueError(f"the view must keep depth on raw axis 0 (axes {perm}): the database matrix is in-plane")
    G, g = np.zeros((2, 2)), np.zeros(2)
    for view_ax, comp in ((2, 0), (1, 1)):          # view X -> component 0 (x), view Y -> component 1 (y)
        raw_ax = perm[view_ax]
        r = raw_ax - 1                               # raw axis 1 (row) -> 0, raw axis 2 (col) -> 1
        if flips[view_ax]:
            G[r, comp] = -1.0
            g[r] = (raw_shape[raw_ax] - 1) * spacing[view_ax]
        else:
            G[r, comp] = 1.0
    return G, g, {'axes': list(perm), 'flips_zyx': list(flips), 'spacing_zyx_mm': list(spacing)}


def euler_to_db(tx, ut_frame, xct_frame, voxel_mm, voxel_used_mm=None):
    """Euler3D UT view physical -> XCT view physical, to (R, t) in raw [row, col] mm
    (UT TIFF [row, col] mm -> XCT TIFF [row, col] mm) and the 3x3 database matrix.

    ``voxel_mm``: XCT voxel size of the database (the convention divides the UT mm by it);
    ``voxel_used_mm``: the actual XCT voxel the registration used (default the same). When they
    differ, M = [[R · voxel_mm / voxel_used_mm, t / voxel_used_mm], [0, 1]] still gives the right XCT
    pixel for every UT point in the database convention (M is then a scaled rotation)."""
    (Gu, gu), (Gx, gx) = ut_frame, xct_frame
    R3 = np.array(tx.GetMatrix()).reshape(3, 3)
    c, t = np.array(tx.GetCenter()), np.array(tx.GetTranslation())
    off = c + t - R3 @ c
    A, b = R3[:2, :2], off[:2]
    Gi = np.linalg.inv(Gu)
    R = Gx @ A @ Gi
    tt = Gx @ (b - A @ Gi @ gu) + gx
    v = voxel_used_mm or voxel_mm
    M = np.eye(3)
    M[:2, :2] = R * (voxel_mm / v)
    M[:2, 2] = tt / v
    return R, tt, M


def hole_fit_to_init(fit, ut_frame, xct_frame, center_xy):
    """Euler3D parameters/centre equivalent to a hole fit, or None when the handedness disagrees with the views."""
    (Gu, gu), (Gx, gx) = ut_frame, xct_frame
    Gxi = np.linalg.inv(Gx)
    A = Gxi @ fit.R @ Gu
    if np.linalg.det(A) < 0:
        return None
    b = Gxi @ (fit.R @ gu + fit.t - gx)
    rz = float(np.arctan2(A[1, 0], A[0, 0]))
    c = np.array(center_xy, float)
    T = A @ c + b - c
    return [0.0, 0.0, rz, float(T[0]), float(T[1]), 0.0], [float(c[0]), float(c[1]), 0.0]


def hole_check(hres, R, t):
    """Distances between the XCT hole centres and the matched UT hole centres moved by (R, t)."""
    if hres is None or hres.fit is None:
        return None
    U, X = hres.ut.holes, hres.xct.holes
    d = [float(np.linalg.norm(R @ [U[u]['row_mm'], U[u]['col_mm']] + t - [X[x]['row_mm'], X[x]['col_mm']]))
         for u, x in hres.fit.pairs]
    return {'pairs': hres.fit.pairs, 'distance_mm': d, 'rms_mm': float(np.sqrt(np.mean(np.square(d)))),
            'hole_fit_rms_mm': hres.fit.rms_mm}


# ----------------------------------------------------------------------------- one result

@dataclass
class Outcome:
    result: object                   # PipelineResult of the GUI module
    R: np.ndarray                    # UT [row, col] mm -> XCT [row, col] mm
    t: np.ndarray
    M: np.ndarray                    # database matrix
    holes: holes_rg.Result | None
    hole_check: dict | None          # final transform applied to the matched hole centres
    info: dict = field(default_factory=dict)

    @property
    def final(self):
        return self.result.stage(self.result.final)


def from_gui(res, inputs, xct_voxel_mm, ut_record=None, holes=None, init_from_holes=False):
    """Outcome of a GUI result. ``inputs``: {'ut'|'xct': {'settings': VolumeLoaderBox.settings(),
    'raw_shape': raw shape}} as they were when the registration was run."""
    Gu, gu, ut_view = view_frame(inputs['ut']['settings'], inputs['ut']['raw_shape'])
    Gx, gx, xct_view = view_frame(inputs['xct']['settings'], inputs['xct']['raw_shape'])
    sx = inputs['xct']['settings']['spacing_zyx_mm']
    if not np.isclose(sx[1], sx[2], rtol=1e-6):
        raise ValueError(f"XCT in-plane spacing {sx[1:]} is not isotropic: the database convention has one XCT voxel")
    v_used = float(sx[1])
    final = res.stage(res.final).transform
    R, t, M = euler_to_db(final, (Gu, gu), (Gx, gx), xct_voxel_mm, v_used)
    p = final.GetParameters()
    warnings = []
    if abs(p[0]) > 1e-6 or abs(p[1]) > 1e-6:
        warnings.append(f"the final transform tilts (rx {np.rad2deg(p[0]):+.3f}°, ry {np.rad2deg(p[1]):+.3f}°); "
                        f"the database matrix is in-plane and drops the tilt")
    if not np.isclose(v_used, xct_voxel_mm, rtol=1e-3):
        warnings.append(f"XCT voxel {v_used} mm used instead of the database voxel size {xct_voxel_mm} mm: the matrix "
                        f"includes the scale {xct_voxel_mm / v_used:.5f} (database convention still holds)")
    cfg = res.config
    init = 'pipeline' if cfg.init_params is None else 'holes' if init_from_holes else 'given'
    dz = inputs['ut']['settings']['spacing_zyx_mm'][0]
    k0 = None
    if ut_record is not None:
        k0 = int(round(-ut_record['crop']['source_start_mm']['depth'] / dz))
    info = {'options': {'gui_inputs': inputs}, 'init': init, 'ut_front_wall_sample': k0,
            'ut_spacing_zyx': ut_view['spacing_zyx_mm'], 'ut_view': ut_view, 'xct_view': xct_view,
            'mirror': bool(np.linalg.det(R) < 0), 'rotation_deg': float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            'conversion_warnings': warnings, 'xct_voxel_used_mm': v_used}
    return Outcome(res, R, t, M, holes, hole_check(holes, R, t), info)
