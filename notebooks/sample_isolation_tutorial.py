#!/usr/bin/env python3
"""
XCT Sample Isolation — annotated run script with interactive GUIs.

Two interactive windows appear before processing begins:
  1. Z-range selector  — browse slices, set start/end crop bounds.
  2. Reslice selector  — preview all orientations, pick one.

Processing figures are saved to SAVING_FOLDER/visualization/.
A log file is written to SAVING_FOLDER/sample_isolation.log.
"""

import logging
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess_tools import io, reslicer

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VOLUME_PATH   = Path('/home/jorgecabrejas/Dev/Data/equalized/19_1+2-24_1+2-25_1+2_25um_eq_8b')
SAVING_FOLDER = Path('/home/jorgecabrejas/Desktop/isolated_samples_19_1+2-24_1+2-25_1+2_25um_eq_8b')
N_SAMPLES     = 6
SAMPLE_NAMES  = [f"19_{i}" for i in (1, 2)] + [f"24_{i}" for i in range(1, 3)] + [f"25_{i}" for i in range(1, 3)]
# ──────────────────────────────────────────────────────────────────────────────

RESLICE_OPTIONS = ['None', 'Top', 'Left', 'Right', 'Bottom']
_BG    = '#1e1e1e'
_PANEL = '#2a2a2a'
_GREEN = '#00cc55'
_RED   = '#ff4444'
_BLUE  = '#4499ee'


# ─── Logging & figure helpers ─────────────────────────────────────────────────

def setup_logging(saving_folder):
    viz_dir = saving_folder / 'visualization'
    viz_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(saving_folder / 'sample_isolation.log', mode='w'),
        ],
    )
    return logging.getLogger('isolate'), viz_dir


def save_fig(viz_dir, log, name, title=''):
    path = viz_dir / f'{name}.png'
    if title:
        plt.suptitle(title, fontsize=12, weight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close('all')
    log.info(f'  figure → {path.name}')


def ortho_views(vol, viz_dir, log, name, title):
    mz, my, mx = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(vol[mz],       cmap='gray'); axes[0].set_title(f'Axial  Z={mz}')
    axes[1].imshow(vol[:, my, :], cmap='gray'); axes[1].set_title(f'Coronal  Y={my}')
    axes[2].imshow(vol[:, :, mx], cmap='gray'); axes[2].set_title(f'Sagittal  X={mx}')
    for ax in axes:
        ax.axis('off')
    save_fig(viz_dir, log, name, title)


def slice_montage(vol, axis, viz_dir, log, name, title, max_tiles=64):
    """Grid of evenly-sampled slices along the given axis (0=Z, 1=Y, 2=X)."""
    n = vol.shape[axis]
    step = max(1, n // max_tiles)
    indices = list(range(0, n, step))[:max_tiles]
    cols = min(16, len(indices))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes_flat = np.array(axes).flatten()
    for ax in axes_flat:
        ax.axis('off')
    for k, idx in enumerate(indices):
        axes_flat[k].imshow(np.take(vol, idx, axis=axis), cmap='gray')
        axes_flat[k].set_title(str(idx), fontsize=5)
    save_fig(viz_dir, log, name, title)


# ─── Interactive GUI: Z-range selector ───────────────────────────────────────

def gui_select_z_range(volume, log):
    """
    Show an interactive window with:
      - Left : current Z slice (updates with the 'View Z' slider)
      - Top-right : Y-vs-Z spatial overview with coloured markers
      - Bottom-right : per-slice mean intensity profile with coloured markers
      - Three sliders: View Z / Start Z / End Z
      - Confirm button

    Returns (z_start, z_end).
    """
    n = volume.shape[0]
    log.info(f'  Launching Z-range GUI  ({n} slices total)')

    # Per-slice mean — subsample so this stays fast
    step_p = max(1, n // 500)
    z_prof = np.arange(0, n, step_p)
    log.info(f'  Computing per-slice means ({len(z_prof)} samples) …')
    means = np.array([float(volume[i].mean()) for i in z_prof])

    # Spatial overview: rows = Y (subsampled), cols = Z (subsampled)
    mid_x   = volume.shape[2] // 2
    step_z  = max(1, n // 400)
    step_y  = max(1, volume.shape[1] // 300)
    # overview.T → shape (Y', Z') so cols run along Z — correct for a horizontal Z axis
    overview = volume[::step_z, ::step_y, mid_x].T

    state = {'z': n // 2, 'start': 0, 'end': n - 1}

    fig = plt.figure(figsize=(15, 8), facecolor=_BG)
    fig.suptitle(
        'Step 1b — Select Z-slice range\n'
        'Green = Start   |   Red = End   |   White dashed = current view   |   Click Confirm when done',
        color='white', fontsize=10, weight='bold',
    )

    # Layout: left column spans both rows (slice display); right column = overview + profile
    # Reserve bottom 0.28 for sliders / button
    gs = fig.add_gridspec(
        2, 2,
        left=0.04, right=0.97, top=0.90, bottom=0.28,
        height_ratios=[1.2, 1], hspace=0.40, wspace=0.22,
    )
    ax_img  = fig.add_subplot(gs[:, 0])   # slice image (full left column)
    ax_ov   = fig.add_subplot(gs[0, 1])   # Y-vs-Z overview
    ax_prof = fig.add_subplot(gs[1, 1])   # per-slice profile

    for ax in (ax_img, ax_ov, ax_prof):
        ax.set_facecolor(_PANEL)

    # ── Slice image ───────────────────────────────────────────────────────────
    im_slice   = ax_img.imshow(volume[state['z']], cmap='gray', aspect='auto')
    title_img  = ax_img.set_title(f'Z = {state["z"]}', color='white', fontsize=10)
    ax_img.axis('off')

    # ── Spatial overview ──────────────────────────────────────────────────────
    ax_ov.imshow(overview, cmap='gray', aspect='auto')
    # X ticks: column index → actual Z value
    z_ticks = np.linspace(0, overview.shape[1] - 1, 6).astype(int)
    ax_ov.set_xticks(z_ticks)
    ax_ov.set_xticklabels([int(t * step_z) for t in z_ticks], fontsize=6, color='white')
    ax_ov.set_xlabel('Z slice', color='white', fontsize=8)
    ax_ov.set_ylabel('Y', color='white', fontsize=8)
    ax_ov.set_title('Y-vs-Z overview (X = mid)', color='white', fontsize=9)
    ax_ov.tick_params(axis='y', colors='white', labelsize=6)
    for sp in ax_ov.spines.values():
        sp.set_edgecolor('#555')

    # Markers in column-index space (Z / step_z)
    ov_vz    = ax_ov.axvline(state['z'] / step_z,     color='white',  lw=1.2, ls='--', alpha=0.8)
    ov_start = ax_ov.axvline(state['start'] / step_z, color=_GREEN, lw=2)
    ov_end   = ax_ov.axvline(state['end'] / step_z,   color=_RED,   lw=2)

    # ── Per-slice mean profile ────────────────────────────────────────────────
    ax_prof.plot(z_prof, means, color=_BLUE, lw=0.9)
    ax_prof.set_xlabel('Z slice', color='white', fontsize=8)
    ax_prof.set_ylabel('Mean intensity', color='white', fontsize=8)
    ax_prof.set_title('Per-slice mean intensity', color='white', fontsize=9)
    ax_prof.tick_params(colors='white', labelsize=6)
    for sp in ax_prof.spines.values():
        sp.set_edgecolor('#555')

    pr_vz    = ax_prof.axvline(state['z'],     color='white',  lw=1.2, ls='--', alpha=0.8)
    pr_start = ax_prof.axvline(state['start'], color=_GREEN, lw=2, label='Start')
    pr_end   = ax_prof.axvline(state['end'],   color=_RED,   lw=2, label='End')
    ax_prof.legend(fontsize=7, facecolor='#333', edgecolor='#555', labelcolor='white')

    # ── Sliders ───────────────────────────────────────────────────────────────
    ax_sl_z = fig.add_axes([0.04, 0.195, 0.91, 0.025], facecolor='#333')
    ax_sl_s = fig.add_axes([0.04, 0.155, 0.91, 0.025], facecolor='#333')
    ax_sl_e = fig.add_axes([0.04, 0.115, 0.91, 0.025], facecolor='#333')

    sl_z = Slider(ax_sl_z, 'View Z ', 0, n - 1, valinit=state['z'],     valstep=1, color='#666')
    sl_s = Slider(ax_sl_s, 'Start Z', 0, n - 1, valinit=state['start'], valstep=1, color=_GREEN)
    sl_e = Slider(ax_sl_e, 'End Z  ', 0, n - 1, valinit=state['end'],   valstep=1, color=_RED)

    for sl in (sl_z, sl_s, sl_e):
        sl.label.set_color('white')
        sl.valtext.set_color('white')

    info = fig.text(
        0.50, 0.076,
        f'Range: [{state["start"]} — {state["end"]}]  '
        f'({state["end"] - state["start"] + 1} slices)',
        ha='center', color='#aaa', fontsize=9,
    )

    ax_btn = fig.add_axes([0.40, 0.025, 0.20, 0.045])
    btn = Button(ax_btn, 'Confirm', color='#0055cc', hovercolor='#0077ff')
    btn.label.set_color('white')
    btn.label.set_fontsize(10)

    def _refresh_info():
        info.set_text(
            f'Range: [{state["start"]} — {state["end"]}]  '
            f'({state["end"] - state["start"] + 1} slices)'
        )

    def on_z(val):
        z = int(sl_z.val)
        state['z'] = z
        im_slice.set_data(volume[z])
        title_img.set_text(f'Z = {z}')
        ov_vz.set_xdata([z / step_z, z / step_z])
        pr_vz.set_xdata([z, z])
        fig.canvas.draw_idle()

    def on_start(val):
        state['start'] = int(sl_s.val)
        ov_start.set_xdata([state['start'] / step_z, state['start'] / step_z])
        pr_start.set_xdata([state['start'], state['start']])
        _refresh_info()
        fig.canvas.draw_idle()

    def on_end(val):
        state['end'] = int(sl_e.val)
        ov_end.set_xdata([state['end'] / step_z, state['end'] / step_z])
        pr_end.set_xdata([state['end'], state['end']])
        _refresh_info()
        fig.canvas.draw_idle()

    def on_confirm(_event):
        plt.close(fig)

    sl_z.on_changed(on_z)
    sl_s.on_changed(on_start)
    sl_e.on_changed(on_end)
    btn.on_clicked(on_confirm)

    plt.show(block=True)
    log.info(f'  Selected: start={state["start"]}  end={state["end"]}')
    return state['start'], state['end']


# ─── Interactive GUI: reslice selector ───────────────────────────────────────

def gui_select_reslice(volume, log):
    """
    Show an interactive window with:
      - Top strip : mid-axial thumbnail for each of None/Top/Left/Right/Bottom
      - Bottom    : three orthogonal mid-plane previews for the selected option
      - Confirm button

    Click a thumbnail to select. Returns the chosen name as a string.
    """
    log.info('  Launching reslice GUI')
    state = {'choice': 'None', 'idx': 0}

    def _mid_slices(v):
        mz, my, mx = v.shape[0] // 2, v.shape[1] // 2, v.shape[2] // 2
        return v[mz], v[:, my, :], v[:, :, mx]

    # Precompute previews — reslice returns views so no extra memory is allocated
    previews = {}
    for name in RESLICE_OPTIONS:
        v = volume if name == 'None' else reslicer.reslice(volume, name)
        previews[name] = _mid_slices(v)

    fig = plt.figure(figsize=(16, 8), facecolor=_BG)
    fig.suptitle(
        'Step 1c — Select reslice direction\n'
        'Click a thumbnail to select and preview   |   Click Confirm to apply',
        color='white', fontsize=10, weight='bold',
    )

    # Top strip: thumbnails
    gs_top = fig.add_gridspec(1, 5, left=0.02, right=0.98, top=0.88, bottom=0.52, wspace=0.08)
    ax_thumbs = [fig.add_subplot(gs_top[0, i]) for i in range(5)]

    # Bottom: large 3-panel preview
    gs_bot = fig.add_gridspec(1, 3, left=0.04, right=0.96, top=0.47, bottom=0.10, wspace=0.12)
    ax_prev = [fig.add_subplot(gs_bot[0, i]) for i in range(3)]

    for ax in ax_thumbs + ax_prev:
        ax.set_facecolor(_PANEL)
        ax.axis('off')

    # Render thumbnails
    for i, (ax, name) in enumerate(zip(ax_thumbs, RESLICE_OPTIONS)):
        ax.imshow(previews[name][0], cmap='gray', aspect='auto')
        ax.set_title(name, color='white', fontsize=10, pad=4, weight='bold')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(2)
            sp.set_edgecolor('#444')

    # Render large preview (default: None)
    view_labels = ['Axial (mid Z)', 'Coronal (mid Y)', 'Sagittal (mid X)']
    prev_ims = []
    for j, (ax, lbl) in enumerate(zip(ax_prev, view_labels)):
        im = ax.imshow(previews['None'][j], cmap='gray', aspect='auto')
        ax.set_title(lbl, color='white', fontsize=9)
        prev_ims.append(im)

    # Highlight default selection (None = index 0)
    for sp in ax_thumbs[0].spines.values():
        sp.set_edgecolor(_BLUE)
        sp.set_linewidth(3)

    sel_text = fig.text(0.5, 0.05, 'Selected: None', ha='center', color='#aaa', fontsize=11)

    def _select(name, idx):
        state['choice'] = name
        state['idx']    = idx
        for j, ax in enumerate(ax_thumbs):
            col, lw = (_BLUE, 3) if j == idx else ('#444', 2)
            for sp in ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(lw)
        for j, im in enumerate(prev_ims):
            im.set_data(previews[name][j])
            im.autoscale()
        sel_text.set_text(f'Selected: {name}')
        fig.canvas.draw_idle()

    # Connect clicks on thumbnail axes
    def _make_handler(name, idx):
        def _handler(event):
            if event.inaxes is ax_thumbs[idx]:
                _select(name, idx)
        return _handler

    for i, name in enumerate(RESLICE_OPTIONS):
        fig.canvas.mpl_connect('button_press_event', _make_handler(name, i))

    ax_btn = fig.add_axes([0.41, 0.01, 0.18, 0.05])
    btn = Button(ax_btn, 'Confirm', color='#0055cc', hovercolor='#0077ff')
    btn.label.set_color('white')
    btn.label.set_fontsize(10)
    btn.on_clicked(lambda _e: plt.close(fig))

    plt.show(block=True)
    log.info(f'  Reslice choice: {state["choice"]}')
    return state['choice']



# ─── Main pipeline ────────────────────────────────────────────────────────────

def main():
    log, viz_dir = setup_logging(SAVING_FOLDER)
    names = SAMPLE_NAMES if SAMPLE_NAMES is not None \
            else [f'sample_{i:02d}' for i in range(N_SAMPLES)]

    # ── 1. Load ───────────────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 1 — Load volume')
    log.info(f'  path : {VOLUME_PATH}')
    volume = io.load_tif(VOLUME_PATH)
    if volume.ndim != 3:
        volume = volume.squeeze()
        log.info(f'  squeezed to 3D: {volume.shape}')
    log.info(f'  shape  : {volume.shape}  (Z, Y, X)')
    log.info(f'  dtype  : {volume.dtype}')
    log.info(f'  range  : [{volume.min()}, {volume.max()}]')
    log.info(f'  memory : {volume.nbytes / 1e9:.2f} GB')

    ortho_views(volume, viz_dir, log, '01_raw_volume', 'Raw volume — orthogonal mid-planes')

    # ── 1b. Z-crop GUI ────────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 1b — Interactive Z-slice range selection')
    z_start, z_end = gui_select_z_range(volume, log)
    volume = volume[z_start:z_end + 1].copy()
    log.info(f'  Cropped to Z=[{z_start}:{z_end}]  new shape={volume.shape}')
    ortho_views(volume, viz_dir, log, '01b_cropped', f'Cropped volume  Z=[{z_start}:{z_end}]')

    # ── 1c. Reslice GUI ───────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 1c — Interactive reslice selection')
    reslice_name = gui_select_reslice(volume, log)
    if reslice_name != 'None':
        volume = reslicer.reslice(volume, reslice_name).copy()
        log.info(f'  Applied reslice "{reslice_name}"  new shape={volume.shape}')
        ortho_views(volume, viz_dir, log, '01c_resliced', f'After reslice "{reslice_name}"')
    else:
        log.info('  No reslice applied')

    # ── 2. Otsu threshold ─────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 2 — Otsu threshold')
    step = max(1, volume.size // 2_000_000)
    sample_vals = volume.ravel()[::step].astype(np.float32)
    thresh = threshold_otsu(sample_vals)
    binary = volume > thresh
    log.info(f'  threshold  : {thresh:.4g}')
    log.info(f'  foreground : {binary.mean() * 100:.1f}% of voxels')

    mz = volume.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(sample_vals, bins=256, log=True, color='steelblue')
    axes[0].axvline(thresh, color='red', linewidth=2, label=f'Otsu = {thresh:.4g}')
    axes[0].set_xlabel('Intensity'); axes[0].set_ylabel('Count (log scale)')
    axes[0].legend(); axes[0].set_title('Intensity histogram')
    axes[1].imshow(volume[mz], cmap='gray', interpolation='nearest')
    axes[1].set_title(f'Raw mid-axial slice  (Z={mz})'); axes[1].axis('off')
    axes[2].imshow(binary[mz], cmap='gray', interpolation='nearest')
    axes[2].set_title('Binary mask  (white = foreground)'); axes[2].axis('off')
    save_fig(viz_dir, log, '02_otsu_threshold', 'Step 2 — Otsu thresholding')

    # ── 3. Hole filling ───────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 3 — Fill holes slice-by-slice (Z axis)')
    filled = np.zeros_like(binary)
    for i in range(binary.shape[0]):
        filled[i] = binary_fill_holes(binary[i])
        if (i + 1) % 500 == 0 or (i + 1) == binary.shape[0]:
            log.info(f'  {i + 1}/{binary.shape[0]} slices done')
    added = int(filled.sum()) - int(binary.sum())
    log.info(f'  voxels added by fill : {added:,}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(binary[mz], cmap='gray'); axes[0].set_title('Before fill')
    axes[1].imshow(filled[mz], cmap='gray'); axes[1].set_title('After fill')
    diff = filled[mz].astype(np.int8) - binary[mz].astype(np.int8)
    axes[2].imshow(diff, cmap='hot'); axes[2].set_title('Added voxels (bright = filled)')
    for ax in axes:
        ax.axis('off')
    save_fig(viz_dir, log, '03_hole_filling', 'Step 3 — Hole filling per 2D slice')
    del binary

    # ── 4. Connected-component labeling ───────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 4 — 3D connected-component labeling')
    label_image = label(filled).astype(np.int32)
    del filled
    n_comp = int(label_image.max())
    log.info(f'  components found : {n_comp}')

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(label_image[mz] % 20, cmap='tab20', interpolation='nearest')
    ax.set_title(f'Mid-axial slice — {n_comp} components (colours mod 20)'); ax.axis('off')
    save_fig(viz_dir, log, '04_labels', 'Step 4 — Connected-component labels')

    # ── 5. Select N largest ────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info(f'STEP 5 — Select {N_SAMPLES} largest components')
    props = regionprops(label_image)
    props.sort(key=lambda r: r.area, reverse=True)

    log.info(f'  {"Rank":<6} {"Label":<8} {"Area (voxels)":>16}  Bounding box')
    log.info('  ' + '-' * 68)
    for rank, p in enumerate(props[:max(N_SAMPLES + 5, 12)]):
        tag = '  ← SELECTED' if rank < N_SAMPLES else ''
        log.info(f'  {rank:<6} {p.label:<8} {p.area:>16,}  {p.bbox}{tag}')

    top_n = min(N_SAMPLES + 5, len(props))
    fig, ax = plt.subplots(figsize=(10, 4))
    areas  = [p.area for p in props[:top_n]]
    colors = ['steelblue' if i < N_SAMPLES else 'lightgray' for i in range(top_n)]
    ax.bar(range(top_n), areas, color=colors)
    ax.set_yscale('log')
    ax.set_xticks(range(top_n))
    ax.set_xlabel('Rank'); ax.set_ylabel('Area (voxels, log scale)')
    ax.set_title(f'Region sizes — blue bars = selected top {N_SAMPLES}')
    save_fig(viz_dir, log, '05_region_areas', 'Step 5 — Region size ranking')

    # ── 6. Crop & mask each sample ─────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 6 — Crop and mask each sample')
    minimum_value = volume[volume.shape[0] // 2, volume.shape[1] // 2].min()
    log.info(f'  background fill value : {minimum_value}')

    raw_pairs = []
    for i in range(N_SAMPLES):
        p  = props[i]
        bb = p.bbox
        log.info(f'  rank {i}  label={p.label}  bbox={bb}')
        cropped = volume[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]].copy()
        cropped[label_image[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]] != p.label] = minimum_value
        raw_pairs.append((bb, cropped))

    del label_image

    # ── 7. Spatial sort ────────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 7 — Sort samples by position along X axis (bbox[2])')
    raw_pairs.sort(key=lambda pair: pair[0][2])
    for i, (bb, vol) in enumerate(raw_pairs):
        log.info(f'  sample {i} ({names[i]})  shape={vol.shape}  min-X={bb[2]}')

    # ── 8. Per-sample visualisation ────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 8 — Visualise each extracted sample')
    for i, (bb, vol) in enumerate(raw_pairs):
        name = names[i]
        log.info(f'  {name}  shape={vol.shape}')

        ortho_views(vol, viz_dir, log,
                    f'06_sample_{i:02d}_{name}_ortho',
                    f'{name} — orthogonal mid-planes')

        slice_montage(vol, axis=0, viz_dir=viz_dir, log=log,
                      name=f'06_sample_{i:02d}_{name}_axial',
                      title=f'{name} — axial slices (Z, max 64 shown)',
                      max_tiles=64)

        slice_montage(vol, axis=1, viz_dir=viz_dir, log=log,
                      name=f'06_sample_{i:02d}_{name}_coronal',
                      title=f'{name} — coronal slices (Y, max 64 shown)',
                      max_tiles=64)

        slice_montage(vol, axis=2, viz_dir=viz_dir, log=log,
                      name=f'06_sample_{i:02d}_{name}_sagittal',
                      title=f'{name} — sagittal slices (X, max 64 shown)',
                      max_tiles=64)

    # ── 9. Save ────────────────────────────────────────────────────────────────
    log.info('─' * 60)
    log.info('STEP 9 — Save isolated volumes')
    for i, (bb, vol) in enumerate(raw_pairs):
        name = names[i]
        out_path = SAVING_FOLDER / name
        out_path.mkdir(parents=True, exist_ok=True)
        log.info(f'  {name} → {out_path}')
        io.save_tif(str(out_path), vol)

    log.info('─' * 60)
    log.info('Done.')
    log.info(f'Samples  → {SAVING_FOLDER}')
    log.info(f'Figures  → {viz_dir}')
    log.info(f'Log      → {SAVING_FOLDER / "sample_isolation.log"}')


if __name__ == '__main__':
    main()
