#!/usr/bin/env python
"""
PAUT preprocessing of every exported volume, in one go.

    python produccion/paut/paut_review.py

does everything for all samples:
  1. converts every export .npy to <id>.tif next to it (already converted ones are skipped),
  2. uploads those original TIFFs (+ metadata, conversion record) to 01_Raw on the NAS,
  3. imports the colleague's processed .h5 volumes into 02_Processed (with a .txt of what his
     tool applied); those samples are marked approved and skipped by the review,
  4. opens the review window and goes through every volume not reviewed yet. Acquisitions
     with "review": false in the config (echostar: echo start done by the equipment) are left out.

Single steps: ``convert``, ``upload-raw``, ``import-h5``, ``review``, ``status``.
Database (additions only): ``db-plan`` shows what would be added, ``db-insert`` adds it after confirmation.

Review window: each volume is processed with its parameters (config defaults, or the
ones you accepted before). Change parameters, press Reprocess (Ctrl+R) until it looks
right, then Accept & save (Ctrl+S): the processed TIFF, its JSON, the echo-start maps
and a .txt describing everything done are written to 02_Processed on the NAS (the raw
TIFF is uploaded to 01_Raw first if needed). Reject records the decision only.

Filters: --campaign Na_Panels --acquisition raw_3p5MHz_16elem --all (include reviewed).
Config: produccion/paut/paut_config.json (--config to use another).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from preprocess_tools import paut  # noqa: E402
from preprocess_tools.paut import qc as pqc  # noqa: E402
from preprocess_tools.paut import storage as st  # noqa: E402

DEFAULT_CONFIG = REPO / 'produccion' / 'paut' / 'paut_config.json'


def select(items, args):
    return [i for i in items if (not args.campaign or i.campaign in args.campaign)
            and (not args.acquisition or i.acquisition in args.acquisition)]


# ----------------------------------------------------------------------------- batch commands

def cmd_convert(cfg, items, args):
    run_id = st.new_run_id()
    counts = {}
    for n, it in enumerate(items, 1):
        try:
            r = st.convert_export(it, run_id, overwrite=args.overwrite)
        except Exception as e:
            r = f'error: {e}'
        counts[r.split(' ')[0]] = counts.get(r.split(' ')[0], 0) + 1
        if r != 'skipped':
            print(f"[{n}/{len(items)}] {it.key}: {r}  ({it.local_tif})", flush=True)
    print('conversion:', counts, '(skipped = already converted)')


def cmd_upload_raw(cfg, items, args):
    run_id = st.new_run_id()
    counts = {}
    for n, it in enumerate(items, 1):
        try:
            r = st.upload_raw(cfg, it, run_id)
        except Exception as e:
            r = f'error: {e}'
        counts[r.split(':')[0]] = counts.get(r.split(':')[0], 0) + 1
        if r != 'identical':
            print(f"[{n}/{len(items)}] {it.key}: {r}", flush=True)
    print('upload to', cfg['output_root'], ':', counts, '(identical = already on the NAS)')


def cmd_import_h5(cfg, items, args):
    run_id = st.new_run_id()
    counts = {}
    for n, it in enumerate(items, 1):
        try:
            r = st.import_colleague_h5(cfg, it, run_id)
        except Exception as e:
            r = f'error: {e}'
        counts[r.split(':')[0]] = counts.get(r.split(':')[0], 0) + 1
        if r not in ('no h5', 'already imported'):
            print(f"[{n}/{len(items)}] {it.key}: {r}", flush=True)
    print("colleague .h5 import:", counts)


def _db(cfg):
    import dbtools as db
    from dbtools import load
    db.load_credentials(str(Path(cfg.get('db_env', '~/Dev/db/.env')).expanduser()))
    return db, load


def cmd_db_plan(cfg, items, args):
    """Dry run: everything that would be added to the database (nothing is written)."""
    from preprocess_tools.paut import database as pdb
    db, _ = _db(cfg)
    plan = pdb.build_plan(cfg, items, db)
    print(pdb.summarize(plan))
    report = Path(cfg['export_root']).parent / 'db_plan.json'
    report.write_text(json.dumps(plan, indent=2, default=str))
    print(f"full plan (every insert with its arguments): {report}")


def cmd_db_insert(cfg, items, args):
    """Insert the plan into the database (additions only) after confirmation."""
    from preprocess_tools.paut import database as pdb
    db, load = _db(cfg)
    plan = pdb.build_plan(cfg, items, db)
    print(pdb.summarize(plan))
    n = len(plan['types']) + len(plan['raw']) + len(plan['processed'])
    if n == 0:
        print('nothing to insert')
        return 0
    if not args.yes and input(f"insert these {n} rows into the database? type 'yes': ").strip() != 'yes':
        print('cancelled; nothing written')
        return 1
    conn = db.connect()
    try:
        print('inserted:', pdb.apply_plan(plan, conn, db, load))
    finally:
        conn.close()


def reviewable(cfg, items):
    return [i for i in items if st.acquisition_entry(cfg, i.acq_key).get('review', True)]


def status_rows(cfg, items):
    review = st.load_review(cfg)
    rows = []
    for it in items:
        d = review.get(it.key, {})
        dec = d.get('status', 'pending') + (' (colleague h5)' if d.get('source') == 'colleague' else '')
        if not st.acquisition_entry(cfg, it.acq_key).get('review', True) and dec == 'pending':
            dec = 'not reviewed (config)'
        conv = st.conversion_status(it)
        rows.append({'key': it.key, 'converted': conv,
                     'raw_on_nas': st.raw_uploaded(cfg, it) if conv == 'current' else False,
                     'decision': dec,
                     'processed': ('present' if (Path(cfg['output_root']) / it.rel('processed', '.tif')).exists()
                                   else 'missing') if conv == 'current' else '-'})
    return rows


def cmd_status(cfg, items, args):
    import collections
    st.prune_stale_decisions(cfg, items)
    rows = status_rows(cfg, items)
    c = collections.Counter((r['key'].rsplit('/', 1)[0], r['converted'], r['raw_on_nas'], r['decision'], r['processed'])
                            for r in rows)
    print(f"{'campaign/acquisition':40s} {'converted':9s} {'on NAS':6s} {'decision':26s} {'processed':9s} files")
    for k, v in sorted(c.items()):
        print(f"{k[0]:40s} {k[1]:9s} {str(k[2]):6s} {k[3]:26s} {k[4]:9s} {v}")


# ----------------------------------------------------------------------------- review window

def cmd_review(cfg, items, args):
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeySequence, QShortcut
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    def canvas_widget():
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvasQTAgg(fig)
        canvas.setMinimumSize(400, 300)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(NavigationToolbar2QT(canvas, w))
        lay.addWidget(canvas)
        return w, fig, canvas

    def dspin(v, lo, hi, dec=3, step=0.1):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setValue(v)
        return s

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('PAUT preprocessing review')
            self.resize(1750, 1000)
            self.items = reviewable(cfg, items)
            self.current = None
            self.result = None
            self.params = None
            self.viewer = None
            # left: list of volumes
            self.filter = QtWidgets.QComboBox()
            self.filter.addItems(['pending', 'all', 'approved', 'rejected'])
            self.filter.setCurrentText('all' if args.all else 'pending')
            self.listw = QtWidgets.QListWidget()
            left = QtWidgets.QWidget()
            ll = QtWidgets.QVBoxLayout(left)
            ll.addWidget(QtWidgets.QLabel('Show'))
            ll.addWidget(self.filter)
            ll.addWidget(self.listw, 1)
            # center: figures
            self.tabs = QtWidgets.QTabWidget()
            w1, self.fig_qc, self.cv_qc = canvas_widget()
            w2, self.fig_as, self.cv_as = canvas_widget()
            w3, self.fig_mi, self.cv_mi = canvas_widget()
            self.tabs.addTab(w1, 'QC')
            self.tabs.addTab(w2, 'A-scans (click the C-scan)')
            self.tabs.addTab(w3, 'Mirror options')
            self.tabs.addTab(self.build_inspector(), 'Inspect volume (sliders)')
            # right: parameters
            self.w = {}
            form = QtWidgets.QWidget()
            fl = QtWidgets.QFormLayout(form)
            self.info = QtWidgets.QLabel()
            self.info.setWordWrap(True)
            self.info.setTextFormat(Qt.TextFormat.RichText)
            fl.addRow(self.info)
            fl.addRow(QtWidgets.QLabel('<b>Echo start</b>'))
            self.w['es_on'] = QtWidgets.QCheckBox('enabled')
            self.w['es_g0'] = dspin(-5, -100, 100)
            self.w['es_g1'] = dspin(5, -100, 100)
            self.w['es_thr'] = dspin(30, 0, 100, 1, 1)
            self.w['es_method'] = QtWidgets.QComboBox()
            self.w['es_method'].addItems(['first', 'maximum'])
            self.w['es_gain'] = dspin(1, 0.01, 100, 2)
            self.w['es_sig'] = QtWidgets.QComboBox()
            self.w['es_sig'].addItem('Hilbert envelope', 'envelope')
            self.w['es_sig'].addItem('|RF| + peak hold + moving average', 'abs_peakhold_ma')
            self.w['es_sig'].addItem('|RF| only', 'abs')
            self.w['es_hold'] = QtWidgets.QSpinBox()
            self.w['es_hold'].setRange(1, 101)
            self.w['es_avg'] = QtWidgets.QSpinBox()
            self.w['es_avg'].setRange(1, 101)
            self.w['es_interp'] = QtWidgets.QCheckBox('interpolate A-scans without a hit')
            self.w['es_nb'] = QtWidgets.QSpinBox()
            self.w['es_nb'].setRange(1, 200)
            fl.addRow(self.w['es_on'])
            fl.addRow('gate start (mm)', self.w['es_g0'])
            fl.addRow('gate end (mm)', self.w['es_g1'])
            fl.addRow('threshold (%)', self.w['es_thr'])
            fl.addRow('method', self.w['es_method'])
            fl.addRow('gain', self.w['es_gain'])
            fl.addRow('detect on', self.w['es_sig'])
            fl.addRow('peak hold (samples)', self.w['es_hold'])
            fl.addRow('moving average (samples)', self.w['es_avg'])
            sync = lambda: [self.w[k].setEnabled(self.w['es_sig'].currentData() == 'abs_peakhold_ma')
                            for k in ('es_hold', 'es_avg')]
            self.w['es_sig'].currentIndexChanged.connect(sync)
            sync()
            fl.addRow(self.w['es_interp'])
            fl.addRow('neighbours', self.w['es_nb'])
            fl.addRow(QtWidgets.QLabel('<b>Auto detect piece</b>'))
            self.w['fp_on'] = QtWidgets.QCheckBox('enabled')
            self.w['fp_g0'] = dspin(0, -100, 100)
            self.w['fp_g1'] = dspin(6, -100, 100)
            self.w['fp_thr'] = dspin(15, 0, 100, 1, 1)
            self.w['fp_mg'] = dspin(0.5, 0, 50, 2)
            fl.addRow(self.w['fp_on'])
            fl.addRow('gate start (mm, rel. echo)', self.w['fp_g0'])
            fl.addRow('gate end (mm, rel. echo)', self.w['fp_g1'])
            fl.addRow('threshold (%)', self.w['fp_thr'])
            fl.addRow('margin (mm)', self.w['fp_mg'])
            fl.addRow(QtWidgets.QLabel('<b>Depth crop (mm, relative to the entrance echo)</b>'))
            self.w['zmode'] = QtWidgets.QComboBox()
            self.w['zmode'].addItem('range in mm', 'mm')
            self.w['zmode'].addItem('fixed number of samples', 'samples')
            self.w['z0'] = dspin(0, -50, 100)
            self.w['z1'] = dspin(6, -50, 100)
            self.w['zn'] = QtWidgets.QSpinBox()
            self.w['zn'].setRange(2, 100000)
            fl.addRow('mode', self.w['zmode'])
            fl.addRow('from (mm)', self.w['z0'])
            fl.addRow('to (mm)', self.w['z1'])
            fl.addRow('samples', self.w['zn'])
            zsync = lambda: (self.w['z1'].setEnabled(self.w['zmode'].currentData() == 'mm'),
                             self.w['zn'].setEnabled(self.w['zmode'].currentData() == 'samples'))
            self.w['zmode'].currentIndexChanged.connect(zsync)
            zsync()
            fl.addRow(QtWidgets.QLabel('<b>Mirror</b>'))
            self.w['mi'] = QtWidgets.QCheckBox('flip index axis')
            self.w['ms'] = QtWidgets.QCheckBox('flip scan axis')
            fl.addRow(self.w['mi'])
            fl.addRow(self.w['ms'])
            self.note = QtWidgets.QLineEdit()
            self.note.setPlaceholderText('note (optional)')
            fl.addRow('note', self.note)
            self.b_re = QtWidgets.QPushButton('Reprocess  (Ctrl+R)')
            self.b_def = QtWidgets.QPushButton('Reset to defaults')
            self.b_ok = QtWidgets.QPushButton('Accept && save  (Ctrl+S)')
            self.b_ok.setStyleSheet('background-color: #2e7d32; color: white; font-weight: bold;')
            self.b_no = QtWidgets.QPushButton('Reject')
            self.b_prev = QtWidgets.QPushButton('◀ Previous')
            self.b_next = QtWidgets.QPushButton('Next ▶')
            for b in (self.b_re, self.b_def, self.b_ok, self.b_no):
                fl.addRow(b)
            nav = QtWidgets.QHBoxLayout()
            nav.addWidget(self.b_prev)
            nav.addWidget(self.b_next)
            fl.addRow(nav)
            self.result_lbl = QtWidgets.QLabel()
            self.result_lbl.setWordWrap(True)
            fl.addRow(self.result_lbl)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidget(form)
            scroll.setWidgetResizable(True)
            scroll.setMinimumWidth(340)
            split = QtWidgets.QSplitter()
            split.addWidget(left)
            split.addWidget(self.tabs)
            split.addWidget(scroll)
            split.setSizes([260, 1150, 340])
            self.setCentralWidget(split)
            # wiring
            self.filter.currentTextChanged.connect(self.refresh_list)
            self.listw.currentRowChanged.connect(self.on_select)
            self.b_re.clicked.connect(self.reprocess)
            self.b_def.clicked.connect(self.reset_defaults)
            self.b_ok.clicked.connect(self.accept)
            self.b_no.clicked.connect(self.reject)
            self.b_prev.clicked.connect(lambda: self.move(-1))
            self.b_next.clicked.connect(lambda: self.move(1))
            QShortcut(QKeySequence('Ctrl+R'), self, activated=self.reprocess)
            QShortcut(QKeySequence('Ctrl+S'), self, activated=self.accept)
            self.refresh_list()

        # ------------------------------------------------------------------ volume inspector
        def build_inspector(self):
            w4, self.fig_in, self.cv_in = canvas_widget()
            self.in_vol = QtWidgets.QComboBox()
            self.in_vol.addItems(['processed', 'original (before processing)'])
            self.in_mode_group = QtWidgets.QButtonGroup(self)
            mode_box = QtWidgets.QWidget()
            mode_lay = QtWidgets.QHBoxLayout(mode_box)
            mode_lay.setContentsMargins(0, 0, 0, 0)
            for n, (label, mode) in enumerate((('RF', 'rf'), ('|RF| (absolute)', 'abs'), ('envelope', 'envelope'))):
                b = QtWidgets.QPushButton(label)
                b.setCheckable(True)
                b.setChecked(n == 0)
                b.setProperty('mode', mode)
                self.in_mode_group.addButton(b, n)
                mode_lay.addWidget(b)
            self.in_mode_group.setExclusive(True)
            self.in_vmax = QtWidgets.QSpinBox()
            self.in_vmax.setRange(1, 100)
            self.in_vmax.setValue(100)
            self.in_vmax.setSuffix(' %')
            top = QtWidgets.QHBoxLayout()
            for lbl, wdg in (('volume', self.in_vol), ('show', mode_box), ('contrast max', self.in_vmax)):
                top.addWidget(QtWidgets.QLabel(lbl))
                top.addWidget(wdg)
            top.addStretch()
            self.sliders = {}
            rows = QtWidgets.QVBoxLayout()
            for name in ('depth', 'index', 'scan'):
                sl = QtWidgets.QSlider(Qt.Orientation.Horizontal)
                sp = QtWidgets.QSpinBox()
                mm = QtWidgets.QLabel()
                mm.setMinimumWidth(110)
                sl.valueChanged.connect(sp.setValue)
                sp.valueChanged.connect(sl.setValue)
                sl.valueChanged.connect(self.inspector_moved)
                h = QtWidgets.QHBoxLayout()
                lab = QtWidgets.QLabel(name)
                lab.setMinimumWidth(45)
                h.addWidget(lab)
                h.addWidget(sl, 1)
                h.addWidget(sp)
                h.addWidget(mm)
                rows.addLayout(h)
                self.sliders[name] = (sl, sp, mm)
            box = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(box)
            lay.addLayout(top)
            lay.addLayout(rows)
            lay.addWidget(w4, 1)
            self.in_vol.currentIndexChanged.connect(self.build_viewer)
            self.in_mode_group.buttonClicked.connect(
                lambda b: self.slicer and (self.slicer.set_display(mode=b.property('mode')), self.cv_in.draw_idle()))
            self.in_vmax.valueChanged.connect(lambda v: self.slicer and self.slicer.set_display(vmax=v))
            self.slicer = None
            return box

        def build_viewer(self):
            if self.result is None:
                return
            out, rec, _ = self.result
            if self.in_vol.currentIndex() == 0:
                vol, origin = out, rec['crop']['source_start_mm']['depth']
                title = f"{self.current.key}: processed (depth relative to the entrance echo)"
            else:
                vol, origin = self.raw, 0.0
                title = f"{self.current.key}: original export (equipment depth axis)"
            self.slicer = pqc.SliceViewer(vol, self.fig_in, origin, title, on_pick=self.inspector_picked)
            self.slicer.set_display(self.in_mode_group.checkedButton().property('mode'), self.in_vmax.value())
            for name, n in zip(('index', 'scan', 'depth'), vol.shape):
                sl, sp, _ = self.sliders[name]
                for wdg in (sl, sp):
                    wdg.blockSignals(True)
                    wdg.setRange(0, n - 1)
                    wdg.setValue(n // 2)
                    wdg.blockSignals(False)
            self.inspector_moved()

        def inspector_moved(self, *_):
            if self.slicer is None:
                return
            v = {k: self.sliders[k][0].value() for k in self.sliders}
            self.slicer.set(v['index'], v['scan'], v['depth'])
            self.update_mm()

        def inspector_picked(self, i, j, k):
            for name, val in (('index', i), ('scan', j), ('depth', k)):
                sl, sp, _ = self.sliders[name]
                for wdg in (sl, sp):
                    wdg.blockSignals(True)
                    wdg.setValue(val)
                    wdg.blockSignals(False)
            self.update_mm()

        def update_mm(self):
            s = self.slicer
            self.sliders['index'][2].setText(f"{s.vol.index_mm[s.i]:.2f} mm")
            self.sliders['scan'][2].setText(f"{s.vol.scan_mm[s.j]:.2f} mm")
            self.sliders['depth'][2].setText(f"{s.depth[s.k]:.3f} mm")
            self.cv_in.draw_idle()

        # ------------------------------------------------------------------ list
        def refresh_list(self, keep=None):
            review = st.load_review(cfg)
            want = self.filter.currentText()
            self.shown = [it for it in self.items
                          if want == 'all' or review.get(it.key, {}).get('status', 'pending') == want]
            self.listw.blockSignals(True)
            self.listw.clear()
            for it in self.shown:
                s = review.get(it.key, {}).get('status', 'pending')
                mark = {'approved': '✔', 'rejected': '✘'}.get(s, '·')
                self.listw.addItem(f"{mark} {it.key}")
            self.listw.blockSignals(False)
            self.statusBar().showMessage(f"{len(self.shown)} volumes ({want}) of {len(self.items)}")
            if self.shown:
                row = next((n for n, it in enumerate(self.shown) if keep is not None and it.key == keep), 0)
                self.listw.setCurrentRow(row)
                self.on_select(row)
            else:
                self.current = None
                self.info.setText('<b>Nothing to review with this filter.</b>')

        def move(self, step):
            r = self.listw.currentRow() + step
            if 0 <= r < self.listw.count():
                self.listw.setCurrentRow(r)

        # ------------------------------------------------------------------ parameters
        def set_form(self, p):
            es, fp = p['echo_start'], p['footprint']
            self.w['es_on'].setChecked(es['enabled'])
            self.w['es_g0'].setValue(es['gate_mm'][0])
            self.w['es_g1'].setValue(es['gate_mm'][1])
            self.w['es_thr'].setValue(es['threshold_pct'])
            self.w['es_method'].setCurrentText(es.get('method', 'first'))
            self.w['es_gain'].setValue(es.get('gain', 1.0))
            self.w['es_sig'].setCurrentIndex(max(0, self.w['es_sig'].findData(es.get('signal', 'abs'))))
            self.w['es_hold'].setValue(es.get('hold_samples', 1))
            self.w['es_avg'].setValue(es.get('average_samples', 1))
            self.w['es_interp'].setChecked(es.get('interpolate_missing', True))
            self.w['es_nb'].setValue(es.get('neighbors', 12))
            self.w['fp_on'].setChecked(fp['enabled'])
            self.w['fp_g0'].setValue(fp['gate_mm'][0])
            self.w['fp_g1'].setValue(fp['gate_mm'][1])
            self.w['fp_thr'].setValue(fp['threshold_pct'])
            self.w['fp_mg'].setValue(fp['margin_mm'])
            self.w['zmode'].setCurrentIndex(max(0, self.w['zmode'].findData(p.get('depth_mode', 'mm'))))
            self.w['z0'].setValue(p['depth_range_mm'][0])
            self.w['z1'].setValue(p['depth_range_mm'][1])
            self.w['zn'].setValue(p.get('depth_samples', 512))
            self.w['mi'].setChecked(p['mirror'].get('index', False))
            self.w['ms'].setChecked(p['mirror'].get('scan', False))

        def get_form(self):
            g = lambda k: self.w[k].value()
            return paut.merge_params({
                'echo_start': {'enabled': self.w['es_on'].isChecked(), 'gate_mm': [g('es_g0'), g('es_g1')],
                               'threshold_pct': g('es_thr'), 'method': self.w['es_method'].currentText(),
                               'gain': g('es_gain'), 'interpolate_missing': self.w['es_interp'].isChecked(),
                               'neighbors': g('es_nb'), 'signal': self.w['es_sig'].currentData(),
                               'hold_samples': g('es_hold'), 'average_samples': g('es_avg')},
                'footprint': {'enabled': self.w['fp_on'].isChecked(), 'gate_mm': [g('fp_g0'), g('fp_g1')],
                              'threshold_pct': g('fp_thr'), 'margin_mm': g('fp_mg')},
                'depth_range_mm': [g('z0'), g('z1')], 'depth_mode': self.w['zmode'].currentData(),
                'depth_samples': g('zn'),
                'mirror': {'index': self.w['mi'].isChecked(), 'scan': self.w['ms'].isChecked()}})

        # ------------------------------------------------------------------ processing and display
        def on_select(self, row):
            if row < 0 or row >= len(self.shown):
                return
            self.current = self.shown[row]
            dec = st.load_review(cfg).get(self.current.key, {})
            self.note.setText(dec.get('note', ''))
            self.set_form(st.item_params(cfg, self.current))
            self.reprocess()

        def reset_defaults(self):
            if self.current is not None:
                self.set_form(paut.merge_params(st.acquisition_entry(cfg, self.current.acq_key).get('params')))
                self.reprocess()

        def reprocess(self):
            it = self.current
            if it is None:
                return
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                if st.conversion_status(it) != 'current':
                    self.statusBar().showMessage(f"converting {it.key} …")
                    QtWidgets.QApplication.processEvents()
                    st.convert_export(it, overwrite=True)
                self.params = self.get_form()
                self.raw = st.load_raw_tif(it.local_tif, it.local_conversion, it.metadata)
                self.result = paut.process(self.raw, self.params)
                self.draw()
            except Exception as e:
                self.result = None
                QtWidgets.QMessageBox.critical(self, 'Processing failed', f"{e}\n\n{traceback.format_exc()}")
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()

        def draw(self):
            it, (out, rec, qc) = self.current, self.result
            review = st.load_review(cfg)
            dec = review.get(it.key, {})
            cmp, diffs = st.compare_with_saved(cfg, it, self.params)
            nas = {'missing': 'not saved yet',
                   'identical': "<span style='color:#2e7d32'><b>saved with exactly these transforms</b></span>",
                   'different': "<span style='color:#b26a00'><b>saved with different transforms</b></span>: "
                                + '; '.join(f"{w}: {a} → {b}" for w, a, b in diffs)}[cmp]
            es = rec.get('echo_start', {})
            es_txt = ('echo start off' if 'skipped' in es else
                      f"hits {es['hit_count']}, interpolated {es['interpolated_count']}, unresolved {es['unresolved_count']}")
            self.info.setText(
                f"<b>{it.key}</b><br>export {it.original_id} ({it.acq_key})<br>"
                f"decision: <b>{dec.get('status', 'pending')}</b><br>02_Processed on NAS: {nas}<br>"
                f"{es_txt}<br>crop {rec['crop']['bounds']['index']} × {rec['crop']['bounds']['scan']}, "
                f"output {out.shape[0]}×{out.shape[1]}×{out.shape[2]}")
            pqc.plot_qc(self.raw, out, rec, qc, title=it.key, fig=self.fig_qc)
            self.cv_qc.draw_idle()
            self.viewer = pqc.ascan_viewer(out, self.raw, rec, qc, title=it.key, fig=self.fig_as)
            self.cv_as.draw_idle()
            pqc.plot_mirror_options(out, title=f'{it.key}: orientation options (relative to the current mirror)',
                                    fig=self.fig_mi)
            self.cv_mi.draw_idle()
            self.build_viewer()
            self.result_lbl.setText('')

        # ------------------------------------------------------------------ decisions
        def accept(self):
            it = self.current
            if it is None or self.result is None:
                return
            if self.get_form() != self.params:
                if QtWidgets.QMessageBox.question(
                        self, 'Parameters changed',
                        'The parameters changed since the last Reprocess. Reprocess before saving?') \
                        == QtWidgets.QMessageBox.StandardButton.Yes:
                    self.reprocess()
                    return
            cmp, diffs = st.compare_with_saved(cfg, it, self.params)
            overwrite = False
            if cmp == 'identical':
                st.save_processed(cfg, it, self.result, self.params, self.note.text())  # records the decision only
                self.statusBar().showMessage(f"{it.key}: already saved with exactly these transforms; nothing written", 8000)
                self.after_decision()
                return
            if cmp == 'different':
                if QtWidgets.QMessageBox.question(
                        self, 'Replace?', f"{it.rel('processed', '.tif')} already exists on the NAS, made with different "
                        f"transforms:\n\n{st.describe_differences(diffs)}\n\nReplace it with the current result?") \
                        != QtWidgets.QMessageBox.StandardButton.Yes:
                    return
                overwrite = True
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                st.save_processed(cfg, it, self.result, self.params, self.note.text(), overwrite=overwrite)
                msg = f"saved {it.rel('processed', '.tif')} (+ .txt, .json) and 01_Raw"
            except Exception as e:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(self, 'Save failed', f"{e}\n\n{traceback.format_exc()}")
                return
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().showMessage(msg, 8000)
            self.after_decision()

        def reject(self):
            it = self.current
            if it is None:
                return
            st.reject(cfg, it, self.get_form(), self.note.text())
            self.statusBar().showMessage(f"{it.key}: rejected", 8000)
            self.after_decision()

        def after_decision(self):
            row = self.listw.currentRow()
            nxt = self.shown[row + 1].key if row + 1 < len(self.shown) else None
            if self.filter.currentText() == 'pending':
                self.refresh_list(keep=nxt)
            else:
                self.refresh_list(keep=nxt or self.current.key)

    st.prune_stale_decisions(cfg, items)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = Window()
    win.show()
    if args.screenshot:  # testing aid: render the first volume and save the window
        win.tabs.setCurrentIndex(args.screenshot_tab)
        QtCore.QTimer.singleShot(1500, lambda: (win.grab().save(args.screenshot), app.quit()))
    return app.exec()


def cmd_run(cfg, items, args):
    print('== 1/4 converting exports to TIFF')
    cmd_convert(cfg, items, args)
    print('== 2/4 uploading original TIFFs to 01_Raw')
    cmd_upload_raw(cfg, items, args)
    print('== 3/4 importing the colleague processed .h5 volumes')
    cmd_import_h5(cfg, items, args)
    print('== 4/4 review')
    return cmd_review(cfg, items, args)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', default='run',
                    choices=['run', 'convert', 'upload-raw', 'import-h5', 'status', 'review', 'db-plan', 'db-insert'])
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--campaign', nargs='*', help='only these campaigns (e.g. Na_Panels JI)')
    ap.add_argument('--acquisition', nargs='*', help='only these acquisitions (e.g. raw_3p5MHz_16elem)')
    ap.add_argument('--all', action='store_true', help='review: show every volume, not only pending')
    ap.add_argument('--overwrite', action='store_true', help='convert: redo TIFFs whose export changed')
    ap.add_argument('--yes', action='store_true', help='db-insert: do not ask for confirmation')
    ap.add_argument('--screenshot', help=argparse.SUPPRESS)
    ap.add_argument('--screenshot-tab', type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = st.load_config(args.config)
    items, problems = st.discover(cfg)
    for p in problems:
        print('WARNING', p)
    items = select(items, args)
    print(f"{len(items)} exported volumes selected; output root {cfg['output_root']}")
    return {'run': cmd_run, 'convert': cmd_convert, 'upload-raw': cmd_upload_raw, 'import-h5': cmd_import_h5,
            'status': cmd_status, 'review': cmd_review, 'db-plan': cmd_db_plan,
            'db-insert': cmd_db_insert}[args.command](cfg, items, args) or 0


if __name__ == '__main__':
    sys.exit(main())
