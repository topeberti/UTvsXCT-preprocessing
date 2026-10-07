"""
Phased-array ultrasound (PAUT) preprocessing of FocusData exports.

Input: the files written by ``export_fpd_folder.py`` for one acquisition,
``<name>_ch1_b1_g1_dg1.npy`` (RF, float32 in percent, axes index × scan × depth)
and ``<name>_metadata.json`` (equipment metadata including the axes in mm).

Steps
-----
1. :func:`load_export`       RF volume and its axes in mm.
2. :func:`echo_start`        shift every A-scan so the entrance echo is at 0 mm
                             (removes the pendulum offset of the inspection).
3. :func:`detect_footprint`  "auto detect piece": a gate just below the entrance
                             echo only sees the part (water and material velocities
                             differ, so the tank echo falls outside it).
4. :func:`crop`              lateral bounds + depth range, axes rebased to 0.
5. :func:`mirror`            optional flips, decided per sample from the QC.
6. :func:`to_zyx`            (depth, index, scan) for storage, Z = depth.

:func:`process` chains them and returns a JSON-serialisable record of every
parameter and result, so a run can be reproduced from its sidecar.

Reproducing the colleague's fpd-batch tool (processing_version 5)
----------------------------------------------------------------
Echo start uses the colleague's own functions verbatim (detection, k-NN gap
interpolation, shift application). By default the echo is detected on the Hilbert
envelope (:func:`detection_signal`); ``signal='abs'`` with depth [0, 6] reproduces his files.
With the parameters stored in his ``.h5`` files
the 49 ``Na_Panels`` samples of ``_raw_3p5MHz_16elem`` are reproduced bit-identically
(every A-scan) with identical crop bounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

PROCESSING_VERSION = 3


@dataclass
class PAUTVolume:
    data: np.ndarray            # (index, scan, depth)
    index_mm: np.ndarray
    scan_mm: np.ndarray
    depth_mm: np.ndarray
    metadata: dict = field(default_factory=dict, repr=False)

    @property
    def spacing_mm(self):
        """(index, scan, depth) spacing in mm."""
        return tuple(float(np.median(np.diff(a))) if a.size > 1 else 0.0
                     for a in (self.index_mm, self.scan_mm, self.depth_mm))

    @property
    def shape(self):
        return self.data.shape


def metadata_path_for(npy_path):
    """``<name>_metadata.json`` next to ``<name>_ch…_dg….npy``."""
    npy_path = Path(npy_path)
    return npy_path.with_name(f"{npy_path.name.rsplit('_ch', 1)[0]}_metadata.json")


def load_export(npy_path, metadata_path=None, definition=(1, 1, 1, 1)) -> PAUTVolume:
    """Load one exported A-scan volume and its axes (mm) from the export metadata."""
    npy_path = Path(npy_path)
    metadata_path = Path(metadata_path) if metadata_path else metadata_path_for(npy_path)
    md = json.loads(metadata_path.read_text())
    data = np.load(npy_path)
    exports = [e for e in md.get('volume_exports', []) if list(e.get('definition', [])) == list(definition)]
    if exports and 'axes' in exports[0]:
        ax = exports[0]['axes']
        index_mm, scan_mm, depth_mm = (np.asarray(ax[k]['values'], np.float64)
                                       for k in ('index_mm', 'scan_mm', 'depth_mm'))
    else:  # older exports: rebuild from the data group and gate
        ch, beam, gate, dg = definition
        g = md['channels'][ch - 1]['beams'][beam - 1]['gates'][gate - 1]
        grp = g['data_groups'][dg - 1]
        index_mm = np.arange(data.shape[0]) * grp['index_resolution_m'] * 1e3
        scan_mm = np.arange(data.shape[1]) * grp['scan_resolution_m'] * 1e3
        depth_mm = g['start_mm'] + np.arange(data.shape[2]) * g['width_mm'] / data.shape[2]
    if (index_mm.size, scan_mm.size, depth_mm.size) != data.shape:
        raise ValueError(f"{npy_path.name}: axes {index_mm.size, scan_mm.size, depth_mm.size} "
                         f"do not match data {data.shape}")
    return PAUTVolume(data, index_mm, scan_mm, depth_mm, md)


def _gate_indices(depth_mm, gate_mm):
    lo, hi = gate_mm
    sel = np.flatnonzero((depth_mm >= lo - 1e-6) & (depth_mm <= hi + 1e-6))
    if sel.size == 0:
        raise ValueError(f"gate {gate_mm} mm is outside the depth axis "
                         f"[{depth_mm[0]:.3f}, {depth_mm[-1]:.3f}] mm")
    return int(sel[0]), int(sel[-1]) + 1


# ----------------------------------------------------------------------------- echo start
# detect_echo_start_block, interpolate_missing_echo_shifts and apply_echo_start_shifts are
# copied verbatim from the colleague's fpd tool (EchoStartWorker) so the signals are
# treated exactly the same; validated bit-identical on the 49 Na_Panels .h5 outputs.

def detect_echo_start_block(
    values: np.ndarray,
    gate_start: int,
    gate_stop: int,
    threshold: float,
    gain: float,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the hit mask and selected depth sample for each trace."""
    block = np.asarray(values, dtype=np.float32)
    if block.ndim != 3 or block.shape[2] == 0:
        raise ValueError("Echo start requires a non-empty 3-D volume block.")
    start = max(0, min(int(gate_start), block.shape[2] - 1))
    stop = max(start + 1, min(int(gate_stop), block.shape[2]))
    if method not in {"first", "maximum"}:
        raise ValueError(f"Unknown Echo start detection method: {method}")
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Echo start threshold must be a non-negative finite value.")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("Echo start gain must be a positive finite value.")

    gated = block[:, :, start:stop]
    detected_amplitude = np.abs(gated) * float(gain)
    valid = np.isfinite(detected_amplitude) & (detected_amplitude >= threshold)
    hits = np.any(valid, axis=2)
    if method == "first":
        selected = np.argmax(valid, axis=2)
    else:
        candidates = np.where(valid, detected_amplitude, -np.inf)
        selected = np.argmax(candidates, axis=2)
    selected = selected.astype(np.int64, copy=False) + start

    return hits, selected


def interpolate_missing_echo_shifts(
    hits: np.ndarray,
    selected: np.ndarray,
    index_axis: np.ndarray,
    scan_axis: np.ndarray,
    *,
    neighbors: int = 12,
    cancelled=lambda: False,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer missing shifts from nearby detections in physical X/Y space."""
    hit_map = np.asarray(hits, dtype=bool)
    selected_map = np.asarray(selected, dtype=np.float64)
    index_values = np.asarray(index_axis, dtype=np.float64)
    scan_values = np.asarray(scan_axis, dtype=np.float64)
    expected = (index_values.size, scan_values.size)
    if hit_map.shape != expected or selected_map.shape != expected:
        raise ValueError("Echo start detection maps do not match the spatial axes.")
    if neighbors < 1:
        raise ValueError("Echo start interpolation needs at least one neighbor.")

    shifts = np.where(hit_map, selected_map, np.nan)
    interpolated = np.zeros(expected, dtype=bool)
    valid_flat = np.flatnonzero(hit_map)
    missing_flat = np.flatnonzero(~hit_map)
    if valid_flat.size == 0 or missing_flat.size == 0:
        return shifts.astype(np.float32), interpolated

    valid_index, valid_scan = np.divmod(valid_flat, scan_values.size)
    donor_coordinates = np.column_stack(
        (index_values[valid_index], scan_values[valid_scan])
    )
    donor_shifts = selected_map.reshape(-1)[valid_flat]
    tree = cKDTree(donor_coordinates)
    neighbor_count = min(int(neighbors), valid_flat.size)
    spacing = np.concatenate(
        (
            np.abs(np.diff(index_values)),
            np.abs(np.diff(scan_values)),
        )
    )
    spacing = spacing[np.isfinite(spacing) & (spacing > 0.0)]
    spatial_scale = float(np.median(spacing)) if spacing.size else 1.0

    chunk_size = 50_000
    flat_shifts = shifts.reshape(-1)
    flat_interpolated = interpolated.reshape(-1)
    for offset in range(0, missing_flat.size, chunk_size):
        if cancelled():
            raise InterruptedError("Echo start cancelled.")
        query_flat = missing_flat[offset: offset + chunk_size]
        query_index, query_scan = np.divmod(query_flat, scan_values.size)
        query_coordinates = np.column_stack(
            (index_values[query_index], scan_values[query_scan])
        )
        distances, donor_indices = tree.query(
            query_coordinates, k=neighbor_count
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            donor_indices = donor_indices[:, None]
        values = donor_shifts[donor_indices]

        nearest = distances[:, 0]
        radius = np.maximum(nearest * 3.0, spatial_scale * 2.5)
        local = distances <= radius[:, None]
        center = np.nanmedian(np.where(local, values, np.nan), axis=1)
        deviations = np.abs(values - center[:, None])
        mad = np.nanmedian(np.where(local, deviations, np.nan), axis=1)
        tolerance = np.maximum(1.0, 3.0 * 1.4826 * mad)
        reliable = local & (deviations <= tolerance[:, None])
        reliable_count = np.count_nonzero(reliable, axis=1)
        reliable[reliable_count == 0] = local[reliable_count == 0]

        exact = reliable & (distances <= 1e-12)
        exact_count = np.count_nonzero(exact, axis=1)
        weights = np.where(
            reliable,
            1.0 / np.maximum(distances, spatial_scale * 1e-6) ** 2,
            0.0,
        )
        estimates = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
        if np.any(exact_count):
            estimates[exact_count > 0] = (
                np.sum(np.where(exact, values, 0.0), axis=1)[exact_count > 0]
                / exact_count[exact_count > 0]
            )
        flat_shifts[query_flat] = estimates
        flat_interpolated[query_flat] = True
        if progress is not None:
            progress(min(missing_flat.size, offset + query_flat.size), missing_flat.size)

    return shifts.astype(np.float32), interpolated


def apply_echo_start_shifts(
    values: np.ndarray,
    shifts: np.ndarray,
    apply_mask: np.ndarray,
) -> np.ndarray:
    """Shift traces left without wrapping, interpolating fractional shifts."""
    block = np.asarray(values, dtype=np.float32)
    shift_map = np.asarray(shifts, dtype=np.float64)
    mask = np.asarray(apply_mask, dtype=bool)
    if block.ndim != 3 or block.shape[2] == 0:
        raise ValueError("Echo start requires a non-empty 3-D volume block.")
    if shift_map.shape != block.shape[:2] or mask.shape != block.shape[:2]:
        raise ValueError("Echo start shift maps must match the spatial volume shape.")

    flat_source = block.reshape(-1, block.shape[2])
    flat_target = flat_source.copy()
    flat_mask = mask.reshape(-1) & np.isfinite(shift_map.reshape(-1))
    flat_shifts = np.clip(
        shift_map.reshape(-1), 0.0, float(block.shape[2] - 1)
    )
    flat_target[flat_mask] = 0.0
    bases = np.zeros(flat_shifts.shape, dtype=np.int64)
    bases[flat_mask] = np.floor(flat_shifts[flat_mask]).astype(np.int64)
    fractions = flat_shifts - bases
    for base in np.unique(bases[flat_mask]):
        matching = flat_mask & (bases == base)
        rows = np.flatnonzero(matching)
        row_fractions = fractions[rows]
        integer_rows = rows[np.isclose(row_fractions, 0.0, atol=1e-12)]
        remaining = block.shape[2] - int(base)
        if integer_rows.size:
            flat_target[integer_rows, :remaining] = flat_source[
                integer_rows, int(base):
            ]
        fractional_rows = rows[~np.isclose(row_fractions, 0.0, atol=1e-12)]
        if fractional_rows.size and remaining > 1:
            weights = fractions[fractional_rows, None]
            flat_target[fractional_rows, : remaining - 1] = (
                flat_source[fractional_rows, int(base):-1] * (1.0 - weights)
                + flat_source[fractional_rows, int(base) + 1:] * weights
            )
    return flat_target.reshape(block.shape)


def detection_signal(data, signal='abs_peakhold_ma', hold_samples=5, average_samples=3):
    """Signal on which the entrance echo is detected (the shift is still applied to the RF).

    'envelope': Hilbert envelope |hilbert(RF)| along depth (per-A-scan mean removed first).
    'abs': |RF|.  'abs_peakhold_ma': |RF| -> peak hold (causal running maximum over
    ``hold_samples``: each sample takes the largest |a| of itself and the preceding
    hold_samples - 1, so a peak is held without moving the echo earlier) -> centred moving
    average over ``average_samples``.
    """
    if signal == 'envelope':
        from scipy.signal import hilbert
        rf = np.asarray(data, dtype=np.float32)
        rf = rf - rf.mean(axis=-1, keepdims=True)
        return np.abs(hilbert(rf, axis=-1)).astype(np.float32)
    det = np.abs(np.asarray(data, dtype=np.float32))
    if signal == 'abs':
        return det
    if signal != 'abs_peakhold_ma':
        raise ValueError(f"unknown detection signal {signal!r} (envelope, abs, abs_peakhold_ma)")
    h, m = int(hold_samples), int(average_samples)
    if h > 1:
        padded = np.concatenate([np.zeros(det.shape[:-1] + (h - 1,), np.float32), det], axis=-1)
        det = np.lib.stride_tricks.sliding_window_view(padded, h, axis=-1).max(axis=-1)
    if m > 1:
        det = uniform_filter1d(det, size=m, axis=-1, mode='nearest')
    return np.ascontiguousarray(det, dtype=np.float32)


def echo_start(vol: PAUTVolume, gate_mm=(-5.0, 5.0), threshold_pct=30.0, method='first', gain=1.0,
               interpolate_missing=True, neighbors=12, pre_echo_mm=0.0,
               signal='abs_peakhold_ma', hold_samples=5, average_samples=3):
    """Align the entrance echo of every A-scan to 0 mm (the colleague's algorithm).

    Detection: first (``method='first'``) or strongest (``'maximum'``) sample inside
    ``gate_mm`` with ``d * gain >= threshold_pct``, where ``d`` is :func:`detection_signal`
    (|RF| after a short peak hold and moving average by default; ``signal='abs'`` = plain |RF|). A-scans without a hit get a shift
    from :func:`interpolate_missing_echo_shifts` (robust k-NN inverse-distance weighting
    of the hits in mm) when ``interpolate_missing``; otherwise they are left unshifted.
    The shifts are applied to the original RF with :func:`apply_echo_start_shifts`
    (samples past the end become 0).

    ``pre_echo_mm`` keeps that much signal before the echo (shift stops short of it),
    so the aligned depth axis runs from ``-pre_echo_mm`` with the echo at 0 mm; with 0
    this is exactly the colleague's behaviour.

    Returns ``(aligned_volume, info, qc)``.
    """
    dz = vol.spacing_mm[2]
    ga, gb = _gate_indices(vol.depth_mm, gate_mm)
    det = detection_signal(vol.data, signal, hold_samples, average_samples)
    hits, selected = detect_echo_start_block(det, ga, gb, threshold_pct, gain, method)
    if interpolate_missing:
        shift_map, interpolated = interpolate_missing_echo_shifts(
            hits, selected, vol.index_mm.astype(np.float32), vol.scan_mm.astype(np.float32),
            neighbors=int(neighbors))
    else:
        shift_map = np.where(hits, selected, np.nan).astype(np.float32)
        interpolated = np.zeros_like(hits)
    applied = hits | interpolated
    pre = int(round(max(0.0, float(pre_echo_mm)) / dz))
    # A-scans whose echo is closer than ``pre`` samples to the start of the recording: pad the
    # recording with zeros in front so they stay aligned (no padding when pre = 0).
    low = float(np.nanmin(np.where(applied, shift_map, np.inf))) if applied.any() else float(pre)
    pad = max(0, int(np.ceil(pre - low)))
    clipped = applied & (shift_map < pre)
    data = vol.data
    if pad:
        data = np.concatenate([np.zeros(data.shape[:2] + (pad,), data.dtype), data], axis=-1)
    aligned = apply_echo_start_shifts(data, shift_map + pad - pre, applied)

    echo_mm = vol.depth_mm[0] + shift_map.astype(np.float64) * dz
    zero = aligned[:, :, min(pre, aligned.shape[2] - 1)].copy()
    zero[~applied] = np.nan
    info = {
        'gate_mm': [float(g) for g in gate_mm], 'gate_samples': [ga, gb],
        'threshold_pct': float(threshold_pct), 'method': method, 'gain': float(gain),
        'detection_signal': signal, 'hold_samples': int(hold_samples), 'average_samples': int(average_samples),
        'interpolate_missing': bool(interpolate_missing), 'neighbors': int(neighbors),
        'hit_count': int(hits.sum()), 'interpolated_count': int(interpolated.sum()),
        'unresolved_count': int((~applied).sum()), 'trace_count': int(hits.size),
        'pre_echo_mm': float(pre * dz), 'pre_echo_samples': pre, 'clipped_pre_echo': int(clipped.sum()),
        'zero_padded_samples': pad,
        'echo_mm': {'min': float(np.nanmin(echo_mm)), 'max': float(np.nanmax(echo_mm)),
                    'median': float(np.nanmedian(echo_mm))},
    }
    qc = {'hit': hits, 'interpolated': interpolated, 'applied': applied,
          'shift_samples': shift_map, 'echo_mm': echo_mm, 'cscan_zero': zero}
    out = replace(vol, data=aligned, depth_mm=(np.arange(aligned.shape[2]) - pre) * dz)
    return out, info, qc


def detect_footprint(vol: PAUTVolume, gate_mm=(0.0, 6.0), threshold_pct=15.0, margin_mm=0.5):
    """"Auto detect piece": A-scans whose max |a| inside ``gate_mm`` (just below the
    aligned entrance echo) reaches ``threshold_pct``; their bounding box is widened
    by ``margin_mm`` per axis (in mm, so a 0.5 mm margin adds no 0.75 mm pixel).

    Returns ``({'index': [i0, i1], 'scan': [j0, j1]}, info, qc)``, bounds half-open.
    """
    ga, gb = _gate_indices(vol.depth_mm, gate_mm)
    peak = np.abs(vol.data[..., ga:gb]).max(-1)
    mask = peak >= threshold_pct
    if not mask.any():
        raise ValueError(f"no A-scan reaches {threshold_pct}% inside gate {gate_mm} mm")
    bounds = {}
    for name, axis_mm, rows in (('index', vol.index_mm, mask.any(1)), ('scan', vol.scan_mm, mask.any(0))):
        hits = np.flatnonzero(rows)
        lo_mm, hi_mm = axis_mm[hits[0]] - margin_mm, axis_mm[hits[-1]] + margin_mm
        sel = np.flatnonzero((axis_mm >= lo_mm - 1e-6) & (axis_mm <= hi_mm + 1e-6))
        bounds[name] = [int(sel[0]), int(sel[-1]) + 1]
    info = {'gate_mm': [float(g) for g in gate_mm], 'threshold_pct': float(threshold_pct),
            'margin_mm': float(margin_mm), 'selected_traces': int(mask.sum()), 'bounds': bounds}
    return bounds, info, {'mask': mask, 'peak': peak}


def crop(vol: PAUTVolume, bounds=None, depth_range_mm=(0.0, 6.0), depth_samples=None):
    """Crop laterally (``bounds`` from :func:`detect_footprint`, None keeps all) and in depth,
    then rebase all axes to start at 0.

    Depth: with ``depth_samples`` = N, keep exactly N samples starting at the first sample at or
    after ``depth_range_mm[0]`` (zeros appended if the recording is shorter); otherwise keep the
    inclusive mm range ``depth_range_mm`` (None keeps all).
    """
    i0, i1 = bounds['index'] if bounds else (0, vol.shape[0])
    j0, j1 = bounds['scan'] if bounds else (0, vol.shape[1])
    end_pad = 0
    if depth_samples:
        lo = depth_range_mm[0] if depth_range_mm is not None else vol.depth_mm[0]
        k0 = int(np.flatnonzero(vol.depth_mm >= lo - 1e-6)[0])
        k1 = k0 + int(depth_samples)
        end_pad = max(0, k1 - vol.shape[2])
        if end_pad:
            dz = vol.spacing_mm[2]
            vol = replace(vol, data=np.concatenate([vol.data, np.zeros(vol.shape[:2] + (end_pad,), vol.data.dtype)], -1),
                          depth_mm=np.concatenate([vol.depth_mm, vol.depth_mm[-1] + dz * np.arange(1, end_pad + 1)]))
    else:
        k0, k1 = (0, vol.shape[2]) if depth_range_mm is None else _gate_indices(vol.depth_mm, depth_range_mm)
    axes = {'index': vol.index_mm[i0:i1], 'scan': vol.scan_mm[j0:j1], 'depth': vol.depth_mm[k0:k1]}
    info = {'bounds': {'index': [int(i0), int(i1)], 'scan': [int(j0), int(j1)], 'depth': [int(k0), int(k1)]},
            'depth_range_mm': list(depth_range_mm) if depth_range_mm is not None else None,
            'depth_mode': 'samples' if depth_samples else 'mm', 'depth_samples': int(k1 - k0),
            'zero_padded_end_samples': int(end_pad),
            'source_start_mm': {k: float(v[0]) for k, v in axes.items()}}
    out = replace(vol, data=np.ascontiguousarray(vol.data[i0:i1, j0:j1, k0:k1]),
                  index_mm=axes['index'] - axes['index'][0], scan_mm=axes['scan'] - axes['scan'][0],
                  depth_mm=axes['depth'] - axes['depth'][0])
    return out, info


def mirror(vol: PAUTVolume, index=False, scan=False):
    """Flip the index and/or scan axis (the colleague's 'horizontal' / 'vertical')."""
    data = vol.data
    if index:
        data = data[::-1]
    if scan:
        data = data[:, ::-1]
    return replace(vol, data=np.ascontiguousarray(data)), {'index': bool(index), 'scan': bool(scan)}


def to_zyx(data):
    """(index, scan, depth) → (depth, index, scan): Z = depth, Y = index, X = scan."""
    return np.ascontiguousarray(np.transpose(data, (2, 0, 1)))


# Values stored by the colleague's tool in the _raw_3p5MHz Na_Panels .h5 files.
DEFAULT_PARAMS = {
    'echo_start': {'enabled': True, 'gate_mm': [-5.0, 5.0], 'threshold_pct': 30.0,
                   'method': 'maximum', 'gain': 1.0, 'interpolate_missing': True, 'neighbors': 12,
                   # detection signal: 'envelope' (Hilbert), 'abs_peakhold_ma' (|RF| + peak hold + moving
                   # average, sizes below) or 'abs' (plain |RF|, the colleague's files)
                   'signal': 'envelope', 'hold_samples': 5, 'average_samples': 3},
    'footprint': {'enabled': True, 'gate_mm': [0.0, 6.0], 'threshold_pct': 15.0, 'margin_mm': 0.5},
    'depth_range_mm': [-4.0, 6.5],  # relative to the entrance echo (keeps 4 mm before it); [0, 6] = colleague's files
    # depth crop mode: 'mm' keeps depth_range_mm; 'samples' keeps depth_samples samples from depth_range_mm[0]
    'depth_mode': 'samples', 'depth_samples': 128,
    'mirror': {'index': False, 'scan': False},
}


def merge_params(*overrides):
    """DEFAULT_PARAMS updated (one level deep) by each override in turn."""
    p = json.loads(json.dumps(DEFAULT_PARAMS))
    for o in overrides:
        for k, v in (o or {}).items():
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                p[k] = {**p[k], **v}
            else:
                p[k] = v
    return p


def process(vol: PAUTVolume, params=None):
    """Echo start → auto detect piece → crop → mirror.

    ``params`` follows :data:`DEFAULT_PARAMS` (missing keys take the defaults).
    Returns ``(processed_volume, record, qc)``: ``record`` is JSON-serialisable
    with every parameter and result, ``qc`` holds arrays for plots.
    """
    p = merge_params(params)
    record = {'processing_version': PROCESSING_VERSION, 'params': p,
              'input': {'shape_index_scan_depth': list(vol.shape),
                        'spacing_mm_index_scan_depth': list(vol.spacing_mm),
                        'depth_start_mm': float(vol.depth_mm[0])}}
    qc = {}
    v = vol
    if p['echo_start']['enabled']:
        es = p['echo_start']
        lo = p['depth_range_mm'][0] if p['depth_range_mm'] is not None else 0.0
        v, record['echo_start'], qc['echo_start'] = echo_start(
            v, es['gate_mm'], es['threshold_pct'], es['method'], es.get('gain', 1.0),
            es.get('interpolate_missing', True), es.get('neighbors', 12), pre_echo_mm=max(0.0, -lo),
            signal=es.get('signal', 'abs'), hold_samples=es.get('hold_samples', 1),
            average_samples=es.get('average_samples', 1))
    else:
        record['echo_start'] = {'skipped': 'disabled (e.g. echo start applied by the equipment)'}
    qc['aligned'] = v
    bounds = None
    if p['footprint']['enabled']:
        fp = p['footprint']
        bounds, record['footprint'], qc['footprint'] = detect_footprint(
            v, fp['gate_mm'], fp['threshold_pct'], fp['margin_mm'])
    v, record['crop'] = crop(v, bounds, p['depth_range_mm'],
                             p.get('depth_samples') if p.get('depth_mode') == 'samples' else None)
    v, record['mirror'] = mirror(v, p['mirror'].get('index', False), p['mirror'].get('scan', False))
    record['output'] = {'shape_index_scan_depth': list(v.shape),
                        'spacing_mm_index_scan_depth': list(v.spacing_mm),
                        'stored_axis_order': ['depth', 'index', 'scan'], 'stored_axes': ['z', 'y', 'x']}
    return v, record, qc
