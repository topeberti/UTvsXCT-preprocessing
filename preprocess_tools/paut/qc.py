"""QC figures for PAUT preprocessing (used by the single-file and batch notebooks)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _extent(x_mm, y_mm):
    dx = (x_mm[1] - x_mm[0]) / 2 if x_mm.size > 1 else 0.5
    dy = (y_mm[1] - y_mm[0]) / 2 if y_mm.size > 1 else 0.5
    return [x_mm[0] - dx, x_mm[-1] + dx, y_mm[-1] + dy, y_mm[0] - dy]


def kept_depth_range(record, spacing_mm):
    """(first, last) kept depth in mm relative to the entrance echo, whatever the crop mode."""
    cr = record['crop']
    lo = cr['source_start_mm']['depth']
    n = cr.get('depth_samples') or (cr['bounds']['depth'][1] - cr['bounds']['depth'][0])
    return lo, lo + (n - 1) * spacing_mm


def cscan(vol, depth_from_mm=0.5, origin_mm=0.0):
    """Max |a| over depth >= ``depth_from_mm`` (skips the entrance echo); ``origin_mm`` is the
    depth of the first sample relative to the echo (negative when pre-echo signal is kept)."""
    sel = vol.depth_mm + origin_mm >= depth_from_mm
    return np.abs(vol.data[..., sel if sel.any() else slice(None)]).max(-1)


def plot_qc(raw, out, record, qc, title='', index_row=None, depth_from_mm=0.5, fig=None):
    """Six panels: echo position before alignment, hits vs filled, auto detect piece,
    B-scan before / after, processed C-scan."""
    fig = fig or plt.figure(figsize=(17, 9))
    fig.clear()
    ax = fig.subplots(2, 3)
    ext = _extent(raw.scan_mm, raw.index_mm)
    es = qc.get('echo_start')
    b = record['crop']['bounds']
    rect = lambda a: a.add_patch(Rectangle((raw.scan_mm[b['scan'][0]], raw.index_mm[b['index'][0]]),
                                           raw.scan_mm[b['scan'][1] - 1] - raw.scan_mm[b['scan'][0]],
                                           raw.index_mm[b['index'][1] - 1] - raw.index_mm[b['index'][0]],
                                           fill=False, ec='r', lw=1.5))
    i = index_row if index_row is not None else (b['index'][0] + b['index'][1]) // 2
    if es is not None:
        im = ax[0, 0].imshow(es['echo_mm'], extent=ext, aspect='equal', cmap='viridis')
        fig.colorbar(im, ax=ax[0, 0], shrink=0.8, label='entrance echo (mm)')
        ax[0, 0].set_title('Echo position before alignment (pendulum offset)')
        r = record['echo_start']
        ax[0, 1].imshow(es['hit'], extent=ext, aspect='equal', cmap='gray')
        ax[0, 1].set_title(f"Hits (white) {r['hit_count']} / interpolated (k-NN IDW) {r['interpolated_count']}"
                           + (f" / unresolved {r['unresolved_count']}" if r['unresolved_count'] else ''))
    else:
        for a in ax[0, :2]:
            a.text(0.5, 0.5, record['echo_start'].get('skipped', 'echo start off'), ha='center',
                   va='center', transform=a.transAxes, wrap=True)
    fp = qc.get('footprint')
    if fp is not None:
        f = record['footprint']
        im = ax[0, 2].imshow(fp['peak'], extent=ext, aspect='equal', cmap='magma', vmin=0, vmax=100)
        ax[0, 2].contour(raw.scan_mm, raw.index_mm, fp['mask'].astype(float), [0.5], colors='c', linewidths=0.8)
        fig.colorbar(im, ax=ax[0, 2], shrink=0.8, label='max |a| in gate (%)')
        ax[0, 2].set_title(f"Auto detect piece: {f['threshold_pct']:g}% in {f['gate_mm']} mm")
    for a in ax[0]:
        rect(a)
        a.set_xlabel('scan (mm)')
        a.set_ylabel('index (mm)')
        a.axhline(raw.index_mm[i], color='w', lw=0.6, ls='--')
    # B-scans along scan at index row i
    ax[1, 0].imshow(raw.data[i].T, aspect='auto', cmap='seismic', vmin=-100, vmax=100,
                    extent=[raw.scan_mm[0], raw.scan_mm[-1], raw.depth_mm[-1], raw.depth_mm[0]])
    if es is not None:
        ax[1, 0].plot(raw.scan_mm, es['echo_mm'][i], 'y', lw=0.8, label='entrance echo')
        ax[1, 0].legend(loc='lower right', fontsize=8)
    ax[1, 0].set_ylim(min(raw.depth_mm[-1], (es['echo_mm'][i].max() + 8) if es is not None else 12),
                      raw.depth_mm[0])
    ax[1, 0].set_title(f'B-scan before (index {raw.index_mm[i]:.2f} mm)')
    al = qc['aligned']
    ax[1, 1].imshow(al.data[i].T, aspect='auto', cmap='seismic', vmin=-100, vmax=100,
                    extent=[al.scan_mm[0], al.scan_mm[-1], al.depth_mm[-1], al.depth_mm[0]])
    lo, hi = kept_depth_range(record, al.spacing_mm[2])
    ax[1, 1].axhline(lo, color='r', lw=0.8)
    ax[1, 1].axhline(hi, color='r', lw=0.8)
    ax[1, 1].set_ylim(min(al.depth_mm[-1], hi + 3), min(lo, 0) - 0.5)
    ax[1, 1].set_title(f'B-scan after alignment (kept {lo:.2f} to {hi:.2f} mm in red)')
    for a in ax[1, :2]:
        a.set_xlabel('scan (mm)')
        a.set_ylabel('depth (mm)')
    c = cscan(out, depth_from_mm, record['crop']['source_start_mm']['depth'])
    im = ax[1, 2].imshow(c, extent=_extent(out.scan_mm, out.index_mm), aspect='equal', cmap='viridis')
    fig.colorbar(im, ax=ax[1, 2], shrink=0.8, label=f'max |a| below {depth_from_mm} mm (%)')
    m = record['mirror']
    ax[1, 2].set_title(f"Processed C-scan {out.shape[0]}×{out.shape[1]}×{out.shape[2]}\n"
                       f"mirror index={m['index']}, scan={m['scan']}")
    ax[1, 2].set_xlabel('scan (mm)')
    ax[1, 2].set_ylabel('index (mm)')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


class AScanViewer:
    """Click on the processed C-scan (or a B-scan) to see the A-scan at that position.

    Needs an interactive backend: run ``%matplotlib widget`` (ipympl) in the cell first.
    With ``raw``, ``record`` and ``qc`` from :func:`paut.process`, the original A-scan at
    the same physical position is shown too, with the detected entrance echo and
    the cropped window.
    """

    def __init__(self, out, raw=None, record=None, qc=None, depth_from_mm=0.5, title='', fig=None):
        self.out, self.raw, self.record, self.qc = out, raw, record, qc
        origin = record['crop']['source_start_mm']['depth'] if record and 'crop' in record else 0.0
        self.depth = np.asarray(out.depth_mm) + origin   # mm relative to the entrance echo
        has_raw = raw is not None and record is not None
        self.fig = fig if fig is not None else plt.figure(figsize=(14, 8.5 if has_raw else 7))
        self.fig.clear()
        gs = self.fig.add_gridspec(3 if has_raw else 2, 2, width_ratios=[1.25, 1])
        self.ax_c = self.fig.add_subplot(gs[0, 0])
        self.ax_a = self.fig.add_subplot(gs[0, 1])
        self.ax_bs = self.fig.add_subplot(gs[1, 0])
        self.ax_bi = self.fig.add_subplot(gs[1, 1])
        self.ax_raw = self.fig.add_subplot(gs[2, :]) if has_raw else None
        c = cscan(out, depth_from_mm, origin)
        ext = _extent(out.scan_mm, out.index_mm)
        self.ax_c.imshow(c, extent=ext, aspect='equal', cmap='viridis')
        self.ax_c.set(title=f'C-scan (max |a| below {depth_from_mm} mm): click to select',
                      xlabel='scan (mm)', ylabel='index (mm)')
        self.mark, = self.ax_c.plot([], [], 'r+', ms=14, mew=2)
        self.line_a, = self.ax_a.plot(self.depth, np.zeros_like(self.depth), lw=1)
        self.ax_a.axvline(0, color='tab:red', lw=0.6, ls=':')
        self.ax_a.set(xlabel='depth relative to entrance echo (mm)', ylabel='amplitude (%)', ylim=(-105, 105))
        self.ax_a.grid(alpha=0.3)
        kw = dict(aspect='auto', cmap='seismic', vmin=-100, vmax=100)
        self.im_bs = self.ax_bs.imshow(out.data[0].T, extent=[out.scan_mm[0], out.scan_mm[-1], self.depth[-1], self.depth[0]], **kw)
        self.im_bi = self.ax_bi.imshow(out.data[:, 0].T, extent=[out.index_mm[0], out.index_mm[-1], self.depth[-1], self.depth[0]], **kw)
        self.v_bs = self.ax_bs.axvline(0, color='k', lw=0.8)
        self.v_bi = self.ax_bi.axvline(0, color='k', lw=0.8)
        self.ax_bs.set(xlabel='scan (mm)', ylabel='depth rel. echo (mm)')
        self.ax_bi.set(xlabel='index (mm)', ylabel='depth rel. echo (mm)')
        self.fig.suptitle(title)
        self.fig.tight_layout()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._click)
        self.select(out.shape[0] // 2, out.shape[1] // 2)

    def _source_index(self, i, j):
        """Processed (i, j) → (row, col) in the original export grid (undoing mirror and crop)."""
        b, m = self.record['crop']['bounds'], self.record['mirror']
        ni, nj = self.out.shape[:2]
        ii = ni - 1 - i if m['index'] else i
        jj = nj - 1 - j if m['scan'] else j
        return b['index'][0] + ii, b['scan'][0] + jj

    def select(self, i, j):
        o = self.out
        i, j = int(np.clip(i, 0, o.shape[0] - 1)), int(np.clip(j, 0, o.shape[1] - 1))
        self.mark.set_data([o.scan_mm[j]], [o.index_mm[i]])
        self.line_a.set_ydata(o.data[i, j])
        self.ax_a.set_title(f'A-scan at index {o.index_mm[i]:.2f} mm, scan {o.scan_mm[j]:.2f} mm  (row {i}, col {j})')
        self.im_bs.set_data(o.data[i].T)
        self.im_bi.set_data(o.data[:, j].T)
        self.v_bs.set_xdata([o.scan_mm[j]] * 2)
        self.v_bi.set_xdata([o.index_mm[i]] * 2)
        self.ax_bs.set_title(f'B-scan along scan at index {o.index_mm[i]:.2f} mm')
        self.ax_bi.set_title(f'B-scan along index at scan {o.scan_mm[j]:.2f} mm')
        if self.ax_raw is not None:
            r, c = self._source_index(i, j)
            ax, raw = self.ax_raw, self.raw
            ax.cla()
            ax.set(xlabel='original depth (mm)', ylabel='amplitude (%)', ylim=(-105, 105))
            ax.grid(alpha=0.3)
            ax.plot(raw.depth_mm, raw.data[r, c], lw=0.8, color='0.3', label='original A-scan')
            es = (self.qc or {}).get('echo_start')
            pe = self.record['params']['echo_start']
            if es is not None and pe.get('signal', 'abs') != 'abs':
                from .processing import detection_signal
                det = detection_signal(raw.data[r, c][None, None, :], pe['signal'], pe.get('hold_samples', 1),
                                       pe.get('average_samples', 1))[0, 0]
                ax.plot(raw.depth_mm, det, lw=1.2, color='tab:purple',
                        label='detection signal (Hilbert envelope)' if pe['signal'] == 'envelope' else
                        f"detection signal (peak hold {pe['hold_samples']}, average {pe['average_samples']})")
            if es is not None:
                e = es['echo_mm'][r, c]
                lo, hi = kept_depth_range(self.record, self.out.spacing_mm[2])
                ax.axvspan(e + lo, e + hi, color='tab:green', alpha=0.15, label=f'kept ({lo:.2f} to {hi:.2f} mm around the echo)')
                ax.axvline(e, color='tab:red', lw=1.2,
                           label=f"entrance echo {e:.2f} mm ({'hit' if es['hit'][r, c] else 'interpolated' if es['interpolated'][r, c] else 'unresolved, not shifted'})")
                g = self.record['echo_start']['gate_mm']
                ax.axvspan(*g, color='tab:orange', alpha=0.08, label=f'echo-start gate {g} mm')
                thr = self.record['echo_start']['threshold_pct']
                ax.axhline(thr, color='tab:orange', lw=0.6, ls='--')
                ax.axhline(-thr, color='tab:orange', lw=0.6, ls='--')
                ax.set_xlim(min(g[0], e - 1), e + hi + 4)
            ax.set_title(f'Original export A-scan (row {r}, col {c})')
            ax.legend(loc='upper right', fontsize=8)
        self.fig.canvas.draw_idle()

    def _click(self, ev):
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if ev.inaxes is None or ev.xdata is None or (tb is not None and getattr(tb, 'mode', '')):
            return
        o = self.out
        near = lambda axis, v: int(np.abs(axis - v).argmin())
        if ev.inaxes is self.ax_c:
            self.select(near(o.index_mm, ev.ydata), near(o.scan_mm, ev.xdata))
        elif ev.inaxes is self.ax_bs:
            i = near(o.index_mm, self.mark.get_ydata()[0])
            self.select(i, near(o.scan_mm, ev.xdata))
        elif ev.inaxes is self.ax_bi:
            j = near(o.scan_mm, self.mark.get_xdata()[0])
            self.select(near(o.index_mm, ev.xdata), j)


def ascan_viewer(out, raw=None, record=None, qc=None, depth_from_mm=0.5, title='', fig=None):
    """Interactive A-scan viewer (see :class:`AScanViewer`); keep the returned object alive."""
    return AScanViewer(out, raw, record, qc, depth_from_mm, title, fig)


def plot_mirror_options(out, depth_from_mm=0.5, title='', fig=None):
    """The processed C-scan in the four mirror combinations, to choose the orientation."""
    c = cscan(out, depth_from_mm)
    fig = fig if fig is not None else plt.figure(figsize=(17, 4))
    fig.clear()
    ax = fig.subplots(1, 4)
    for a, (mi, ms) in zip(ax, [(False, False), (True, False), (False, True), (True, True)]):
        img = c[::-1] if mi else c
        img = img[:, ::-1] if ms else img
        a.imshow(img, aspect='equal', cmap='viridis')
        a.set_title(f"mirror index={mi}, scan={ms}")
        a.set_xlabel('scan')
        a.set_ylabel('index')
    fig.suptitle(title or 'Choose the orientation (relative to the current mirror setting)')
    fig.tight_layout()
    return fig


def plot_compare(out, ref, hit=None, title='Ours vs reference'):
    """Max |difference| per A-scan between our output and a reference volume of the same shape."""
    d = np.abs(out.data - ref).max(-1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    im = ax[0].imshow(d, aspect='equal', cmap='magma')
    fig.colorbar(im, ax=ax[0], label='max |ours − ref| (%)')
    ax[0].set_title('Difference per A-scan')
    i = out.shape[0] // 2
    for a, v, t in ((ax[1], out.data[i].T, 'ours'), (ax[2], ref[i].T, 'reference')):
        a.imshow(v, aspect='auto', cmap='seismic', vmin=-100, vmax=100)
        a.set_title(f'B-scan {t} (row {i})')
    if hit is not None:
        ax[0].contour(hit.astype(float), [0.5], colors='c', linewidths=0.6)
        ax[0].set_title('Difference per A-scan (cyan: hit / filled boundary)')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


class SliceViewer:
    """Three linked orthogonal views of a PAUT volume (index, scan, depth):

    C-scan slice at depth ``k`` (index × scan), B-scan along scan at index ``i``
    (depth × scan) and B-scan along index at scan ``j`` (depth × index), with crosshairs.
    Drive it with :meth:`set` (sliders); clicks call ``on_pick(i, j, k)`` if given.

    ``depth_origin_mm`` is added to the depth axis (e.g. depth relative to the entrance echo).
    ``mode``: 'rf' (signed, seismic), 'abs' (|RF|) or 'envelope' (Hilbert envelope along depth).
    """

    MODES = ('rf', 'abs', 'envelope')

    def __init__(self, vol, fig=None, depth_origin_mm=0.0, title='', on_pick=None):
        self.vol, self.title, self.on_pick = vol, title, on_pick
        self.depth = np.asarray(vol.depth_mm) + depth_origin_mm
        self.fig = fig if fig is not None else plt.figure(figsize=(14, 9))
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
        self.ax_c = self.fig.add_subplot(gs[0, :])
        self.ax_bs = self.fig.add_subplot(gs[1, 0])
        self.ax_bi = self.fig.add_subplot(gs[1, 1])
        self._cache = {}
        self.mode, self.vmax = 'rf', 100.0
        ni, nj, nk = vol.shape
        self.i, self.j, self.k = ni // 2, nj // 2, nk // 2
        d0, d1 = self.depth[0], self.depth[-1]
        self.im_c = self.ax_c.imshow(np.zeros((ni, nj)), extent=_extent(vol.scan_mm, vol.index_mm),
                                     aspect='equal', interpolation='nearest')
        self.im_bs = self.ax_bs.imshow(np.zeros((nk, nj)), aspect='auto', interpolation='nearest',
                                       extent=[vol.scan_mm[0], vol.scan_mm[-1], d1, d0])
        self.im_bi = self.ax_bi.imshow(np.zeros((nk, ni)), aspect='auto', interpolation='nearest',
                                       extent=[vol.index_mm[0], vol.index_mm[-1], d1, d0])
        self.cbar = self.fig.colorbar(self.im_c, ax=self.ax_c, shrink=0.85)
        kw = dict(color='k', lw=0.7, alpha=0.8)
        self.c_h, self.c_v = self.ax_c.axhline(0, **kw), self.ax_c.axvline(0, **kw)
        self.bs_h, self.bs_v = self.ax_bs.axhline(0, **kw), self.ax_bs.axvline(0, **kw)
        self.bi_h, self.bi_v = self.ax_bi.axhline(0, **kw), self.ax_bi.axvline(0, **kw)
        self.ax_c.set(xlabel='scan (mm)', ylabel='index (mm)')
        self.ax_bs.set(xlabel='scan (mm)', ylabel='depth (mm)')
        self.ax_bi.set(xlabel='index (mm)', ylabel='depth (mm)')
        self.fig.suptitle(title)
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._click)
        self.set_display('rf', 100.0, redraw=False)
        self.set(self.i, self.j, self.k)

    def data(self):
        if self.mode not in self._cache:
            d = self.vol.data
            if self.mode == 'abs':
                d = np.abs(d)
            elif self.mode == 'envelope':
                from scipy.signal import hilbert
                d = np.abs(hilbert(d, axis=-1)).astype(np.float32)
            self._cache[self.mode] = d
        return self._cache[self.mode]

    def set_display(self, mode=None, vmax=None, redraw=True):
        self.mode = mode or self.mode
        self.vmax = float(vmax) if vmax is not None else self.vmax
        signed = self.mode == 'rf'
        cmap, lo = ('seismic', -self.vmax) if signed else ('magma', 0.0)
        for im in (self.im_c, self.im_bs, self.im_bi):
            im.set_cmap(cmap)
            im.set_clim(lo, self.vmax)
        self.cbar.set_label({'rf': 'RF (%)', 'abs': '|RF| (%)', 'envelope': 'envelope (%)'}[self.mode])
        if redraw:
            self.set(self.i, self.j, self.k)

    def set(self, i=None, j=None, k=None):
        ni, nj, nk = self.vol.shape
        self.i = int(np.clip(self.i if i is None else i, 0, ni - 1))
        self.j = int(np.clip(self.j if j is None else j, 0, nj - 1))
        self.k = int(np.clip(self.k if k is None else k, 0, nk - 1))
        d, v = self.data(), self.vol
        x_i, x_j, x_k = v.index_mm[self.i], v.scan_mm[self.j], self.depth[self.k]
        self.im_c.set_data(d[:, :, self.k])
        self.im_bs.set_data(d[self.i].T)
        self.im_bi.set_data(d[:, self.j].T)
        self.c_h.set_ydata([x_i] * 2)
        self.c_v.set_xdata([x_j] * 2)
        self.bs_h.set_ydata([x_k] * 2)
        self.bs_v.set_xdata([x_j] * 2)
        self.bi_h.set_ydata([x_k] * 2)
        self.bi_v.set_xdata([x_i] * 2)
        self.ax_c.set_title(f'C-scan slice at depth {x_k:.3f} mm (sample {self.k})')
        self.ax_bs.set_title(f'B-scan along scan at index {x_i:.2f} mm (row {self.i})')
        self.ax_bi.set_title(f'B-scan along index at scan {x_j:.2f} mm (col {self.j})')
        self.fig.canvas.draw_idle()

    def _click(self, ev):
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if ev.inaxes is None or ev.xdata is None or (tb is not None and getattr(tb, 'mode', '')):
            return
        v = self.vol
        near = lambda axis, x: int(np.abs(np.asarray(axis) - x).argmin())
        if ev.inaxes is self.ax_c:
            self.set(i=near(v.index_mm, ev.ydata), j=near(v.scan_mm, ev.xdata))
        elif ev.inaxes is self.ax_bs:
            self.set(j=near(v.scan_mm, ev.xdata), k=near(self.depth, ev.ydata))
        elif ev.inaxes is self.ax_bi:
            self.set(i=near(v.index_mm, ev.xdata), k=near(self.depth, ev.ydata))
        else:
            return
        if self.on_pick:
            self.on_pick(self.i, self.j, self.k)


def slice_viewer_widget(vol, depth_origin_mm=0.0, title=''):
    """Notebook version (``%matplotlib widget``): ipywidgets sliders for depth / index / scan,
    display mode and contrast around a :class:`SliceViewer`. Returns the widget box."""
    import ipywidgets as w
    from IPython.display import display
    ni, nj, nk = vol.shape
    sk = w.IntSlider(nk // 2, 0, nk - 1, description='depth', continuous_update=True, layout=w.Layout(width='32%'))
    si = w.IntSlider(ni // 2, 0, ni - 1, description='index', continuous_update=True, layout=w.Layout(width='32%'))
    sj = w.IntSlider(nj // 2, 0, nj - 1, description='scan', continuous_update=True, layout=w.Layout(width='32%'))
    mode = w.ToggleButtons(options=[('RF', 'rf'), ('|RF|', 'abs'), ('envelope', 'envelope')], value='rf')
    vmax = w.FloatSlider(100, min=1, max=100, step=1, description='contrast max %', layout=w.Layout(width='40%'))
    plt.ioff()
    fig = plt.figure(figsize=(12, 8))
    plt.ion()

    def picked(i, j, k):
        for s, val in ((si, i), (sj, j), (sk, k)):
            s.unobserve(update, 'value')
            s.value = val
            s.observe(update, 'value')

    viewer = SliceViewer(vol, fig, depth_origin_mm, title, on_pick=picked)

    def update(_=None):
        viewer.set(si.value, sj.value, sk.value)

    for s in (si, sj, sk):
        s.observe(update, 'value')
    mode.observe(lambda ch: viewer.set_display(mode=ch['new']), 'value')
    vmax.observe(lambda ch: viewer.set_display(vmax=ch['new']), 'value')
    canvas = fig.canvas
    if not isinstance(canvas, w.DOMWidget):  # not the ipympl backend: show a static figure instead
        canvas = w.Output()
        with canvas:
            display(fig)
    box = w.VBox([w.HBox([sk, si, sj]), w.HBox([mode, vmax]), canvas])
    box.viewer = viewer
    return box
