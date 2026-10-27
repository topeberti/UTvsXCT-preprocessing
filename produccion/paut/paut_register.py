#!/usr/bin/env python
"""
PAUT <-> XCT registration with testing/xct_ut_registration_gui.py, stored like the registrations already
in the database.

    python produccion/paut/paut_register.py

opens the registration GUI with a "Database samples" panel: every approved processed PAUT volume whose
sample has an aligned XCT in the database. "Load sample" fills the Load tab (UT: processed PAUT TIFF,
its spacings, Hilbert envelope; XCT: aligned volume, Y/X swapped view) and the Registration tab (gate,
UT front wall at the entrance echo). Then use the GUI as usual: inspect, change the orientation or the
parameters, Run registration, look at the Results. "Start from hole fit" optionally sets the initial
transform from the through-hole centres. "Accept → save record" converts the shown result to the
database matrix (whatever views you used) and, after confirmation, writes it to
registration_review.json and 02_Processed/<sample>/<sample>_registration_xct.txt/.json.
Nothing is written to the database by the GUI.

Database (additions only, like ``paut_review.py``):
    db-plan     what would be inserted (dry run)
    db-insert   insert the accepted registrations after confirmation
The processed PAUT volumes must be in the database first: ``paut_review.py db-insert``.

Other commands: ``status``. Filters: --campaign Na_Panels Pegaso JI, --all (list decided ones too).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from preprocess_tools.paut import registration_db as rdb  # noqa: E402
from preprocess_tools.paut import registration_sitk as rs  # noqa: E402
from preprocess_tools.paut import storage as st  # noqa: E402

DEFAULT_CONFIG = REPO / 'produccion' / 'paut' / 'paut_config.json'


def _db(cfg):
    import dbtools.dbtools as db
    from dbtools import load
    env = str(Path(cfg.get('db_env', '~/Dev/db/.env')).expanduser())
    db.load_credentials(env)
    return db, load, env


def get_pairs(cfg, items, args):
    db, _, env = _db(cfg)
    conn = db.connect(env)
    try:
        pairs = rdb.find_pairs(cfg, items, conn)
    finally:
        conn.close()
    for p in pairs:
        if p.xct_local is not None and not p.problem and not p.xct_local.exists():
            p.problem = f"XCT file not found: {p.xct_local}"
    return [p for p in pairs if not args.campaign or p.item.campaign in args.campaign]


def cmd_status(cfg, items, args):
    pairs = get_pairs(cfg, items, args)
    dec = rdb.load_decisions(cfg)
    c = collections.Counter((p.item.campaign, rdb.status(p, dec), 'yes' if p.ut_id else 'no') for p in pairs)
    print(f"{'campaign':12s} {'registration':14s} {'PAUT in DB':10s} volumes")
    for k, v in sorted(c.items()):
        print(f"{k[0]:12s} {k[1]:14s} {k[2]:10s} {v}")
    for p in pairs:
        if p.problem:
            print(f"   {p.key}: {p.problem}")


def cmd_db_plan(cfg, items, args):
    plan, skipped = rdb.build_plan(cfg, get_pairs(cfg, items, args))
    print(f"{len(plan)} registrations would be inserted")
    for r in plan:
        m = {x['key']: x['value'] for x in r['metadata']}
        print(f"   {r['key']}  rotation {m['rotation']}°, tz {m['tz']} mm, hole check {m.get('hole_check_rms', '-')} mm")
    for reason, n in collections.Counter(r for _, r in skipped).items():
        print(f"skipped {n}: {reason}")
    return plan


def cmd_db_insert(cfg, items, args):
    plan = cmd_db_plan(cfg, items, args)
    if not plan:
        print('nothing to insert')
        return 0
    if not args.yes and input(f"insert these {len(plan)} registrations into the database? type 'yes': ").strip() != 'yes':
        print('cancelled; nothing written')
        return 1
    db, load, env = _db(cfg)
    conn = db.connect(env)
    try:
        print('inserted:', rdb.apply_plan(plan, conn, load))
    finally:
        conn.close()


# ----------------------------------------------------------------------------- GUI

def cmd_review(cfg, items, args):
    print('looking up the XCT of every sample in the database …')
    pairs = get_pairs(cfg, items, args)
    G_ = rs.gui()
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtCore import Qt
    from preprocess_tools.paut import registration as holes_rg

    pool = ThreadPoolExecutor(max_workers=1)    # hole detection in the background
    holes = {}                                  # key -> Future[registration.Result]

    def record(p):
        return json.loads(p.ut_local.with_name(p.ut_local.stem + '_processing.json').read_text())

    def holes_future(p):
        k = (p.key, str(p.xct_local))
        if k not in holes:
            holes[k] = pool.submit(holes_rg.register, p.ut_local, p.xct_local, p.xct_voxel_mm, 'auto',
                                   ut_record=record(p))
        return holes[k]

    def same(a, b):
        return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()

    def resolve(p, ut_path, xct_path):
        """The pair for the files actually loaded: the UT must be the processed PAUT TIFF, the XCT one of the
        sample's XCT measurements in the database (its id, UNC path and voxel size are then used)."""
        if not same(ut_path, p.ut_local):
            raise ValueError(f"the UT loaded is {ut_path}, not the processed PAUT TIFF {p.ut_local}")
        for o in p.xct_options:
            if same(xct_path, o['local']):
                if o['voxel_mm'] is None:
                    raise ValueError(f"XCT measurement {o['id']} has no voxel_size in the database")
                return dataclasses.replace(p, xct_id=o['id'], xct_unc=o['unc'], xct_local=o['local'],
                                           xct_voxel_mm=o['voxel_mm'], existing_registration=o.get('registration'))
        raise ValueError(f"the XCT loaded ({xct_path}) is not an XCT measurement of sample {p.item.sample} "
                         f"in the database")

    def holes_result(p, wait=False, voxel_mm=None):
        """Hole fit of the pair; with ``voxel_mm`` (the XCT voxel set in the GUI) the XCT hole centres are
        rescaled from the database voxel to it and the fit redone."""
        f = holes_future(p)
        if not wait and not f.done():
            return None
        try:
            hres = f.result()
            if voxel_mm is None or np.isclose(voxel_mm, p.xct_voxel_mm, rtol=1e-6):
                return hres
            k = (p.key, str(p.xct_local), round(float(voxel_mm), 9))
            if k not in holes:
                c = voxel_mm / p.xct_voxel_mm
                xh = dataclasses.replace(hres.xct, voxel_mm=voxel_mm,
                                         holes=[{**h, 'row_mm': h['row_mm'] * c, 'col_mm': h['col_mm'] * c}
                                                for h in hres.xct.holes])
                holes[k] = holes_rg.register(p.ut_local, p.xct_local, voxel_mm, 'auto', ut_record=record(p),
                                             xct_cache=xh)
            return holes[k]
        except Exception as e:  # noqa: BLE001 - the hole check is optional
            print(f"{p.key}: hole detection failed: {e}")
            return None

    class Panel(QtWidgets.QWidget):
        def __init__(self, win):
            super().__init__()
            self.win = win
            self.hole_init = None       # init_params set from the hole fit
            self.snap = None            # GUI inputs when "Run registration" was pressed
            self.result = None          # (PipelineResult, snapshot)
            self.filter = QtWidgets.QComboBox()
            self.filter.addItems(['pending', 'all', 'accepted', 'rejected', 'in DB', 'not possible'])
            self.filter.setCurrentText('all' if args.all else 'pending')
            self.listw = QtWidgets.QListWidget()
            self.info = QtWidgets.QLabel()
            self.info.setWordWrap(True)
            self.info.setTextFormat(Qt.TextFormat.RichText)
            self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.xct_combo = QtWidgets.QComboBox()
            self.xct_combo.setToolTip('XCT measurements of this sample in the database: picking one puts its path '
                                      'in the XCT file field of the Load tab')
            self.btn_load = QtWidgets.QPushButton('Load volumes')
            self.btn_load.setToolTip('Load the UT and XCT files shown on the Load tab (the suggested paths, or the '
                                     'ones you typed / browsed)')
            self.btn_holes = QtWidgets.QPushButton('Start from hole fit')
            self.btn_holes.setToolTip('Set the initial transform of the Registration tab from the through-hole '
                                      'centres (uses the current Load-tab orientation)')
            self.note = QtWidgets.QLineEdit()
            self.note.setPlaceholderText('note (optional)')
            self.btn_ok = QtWidgets.QPushButton('Accept → save record')
            self.btn_ok.setStyleSheet('font-weight: bold')
            self.btn_no = QtWidgets.QPushButton('Reject')
            self.state = QtWidgets.QLabel()
            self.state.setWordWrap(True)
            self.state.setTextFormat(Qt.TextFormat.RichText)
            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(QtWidgets.QLabel('Show'))
            lay.addWidget(self.filter)
            lay.addWidget(self.listw, 1)
            lay.addWidget(self.info)
            lay.addWidget(QtWidgets.QLabel('XCT volume'))
            lay.addWidget(self.xct_combo)
            for w in (self.btn_load, self.btn_holes, self.note, self.btn_ok, self.btn_no, self.state):
                lay.addWidget(w)
            self.filter.currentTextChanged.connect(lambda _: self.refresh_list())
            self.listw.currentRowChanged.connect(self.show_info)
            self.listw.currentRowChanged.connect(lambda _: self.suggest())
            self.listw.itemDoubleClicked.connect(lambda _: self.load())
            self.xct_combo.activated.connect(self.pick_xct)
            self.btn_load.clicked.connect(self.load)
            self.current = None         # Pair whose paths / settings are on the Load tab
            win.load_tab.xct.path_edit.textChanged.connect(self.sync_xct)
            self.btn_holes.clicked.connect(self.start_from_holes)
            self.btn_ok.clicked.connect(self.accept)
            self.btn_no.clicked.connect(self.reject)
            win.reg_tab.run_btn.clicked.connect(self.snapshot)      # after the GUI's own run slot
            win.reg_tab.resultReady.connect(self.on_result)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.update_state)
            self.timer.start(1000)
            self.refresh_list()

        # ---------------------------------------------------------------- list
        def selected(self):
            r = self.listw.currentRow()
            return self.shown[r] if 0 <= r < len(self.shown) else None

        def refresh_list(self, keep=None):
            dec = rdb.load_decisions(cfg)
            f = self.filter.currentText()
            self.shown = [p for p in pairs if f == 'all' or rdb.status(p, dec) == f]
            self.listw.blockSignals(True)
            self.listw.clear()
            mark = {'accepted': '✔', 'rejected': '✘', 'in DB': '◆', 'not possible': '–', 'pending': '·'}
            for p in self.shown:
                self.listw.addItem(f"{mark[rdb.status(p, dec)]} {p.key}")
            self.listw.blockSignals(False)
            rows = [i for i, p in enumerate(self.shown) if keep and p.key == keep]
            if self.shown:
                self.listw.setCurrentRow(rows[0] if rows else 0)
            self.show_info()

        def show_info(self, *_):
            p = self.selected()
            if p is None:
                self.info.setText(f'nothing to show with this filter ({len(pairs)} volumes in total)')
                return
            d = rdb.load_decisions(cfg).get(p.key, {})
            self.info.setText(
                f"<b>{p.key}</b><br>status: <b>{rdb.status(p, rdb.load_decisions(cfg))}</b>"
                + (f" ({d.get('decided_at', '')})" if d else '')
                + "<br>PAUT in DB: " + (f"yes (measurement {p.ut_id})" if p.ut_id else "not yet")
                + (f"<br>XCT measurement {p.xct_id}, voxel {p.xct_voxel_mm} mm" if p.xct_id else '')
                + (f"<br>registration {p.existing_registration} already in the DB" if p.existing_registration else '')
                + (f"<br><span style='color:#c62828'>{p.problem}</span>" if p.problem else ''))
            self.btn_load.setEnabled(not p.problem)

        # ---------------------------------------------------------------- load
        def suggest(self):
            """Put the sample's paths, orientation and pipeline settings on the Load and Registration tabs
            (nothing is loaded until "Load volumes" or the Load buttons of the tab)."""
            p = self.selected()
            if p is None or p.problem or p is self.current:
                return
            rec = record(p)
            sp_i, sp_s, dz = rec['output']['spacing_mm_index_scan_depth']
            k0 = int(round(-rec['crop']['source_start_mm']['depth'] / dz))
            o = {**rs.DEFAULTS, **st.acquisition_entry(cfg, p.item.acq_key).get('registration', {})}
            dec = rdb.load_decisions(cfg).get(p.key, {})
            prev = (dec.get('options') or {}).get('gui_inputs')
            lt = self.win.load_tab
            for box in (lt.ut, lt.xct):
                box.raw = None                       # the previous sample's volumes are dropped
                box._emit()
            if prev:   # a decided sample: the files and views it was registered with
                for box, m in ((lt.ut, 'ut'), (lt.xct, 'xct')):
                    s_ = prev[m]['settings']
                    box.set_orientation(s_['axis_order'], s_['flips_zyx'], s_['spacing_zyx_mm'], s_['rot90'])
                lt.ut.envelope.setChecked(bool(prev['ut']['settings'].get('envelope')))
            else:
                lt.ut.set_orientation((0, 1, 2), (False,) * 3, (dz, sp_i, sp_s), 0)
                lt.ut.envelope.setChecked(True)
                lt.xct.set_orientation(o['xct_axes'], G_._parse_flips(o['xct_flip']), (p.xct_voxel_mm,) * 3, 0)
            lt.ut.path_edit.setText(str(p.ut_local))
            lt.ut._refresh_keys()
            self.xct_combo.clear()
            for x in p.xct_options:
                self.xct_combo.addItem(f"{x['id']}: {Path(x['unc'].replace(chr(92), '/')).name}"
                                       + (" (aligned)" if x['aligned'] else '')
                                       + ('' if x['local'] and Path(x['local']).exists() else ' — file not found'), x)
            xct_path = (prev or {}).get('xct', {}).get('settings', {}).get('path') or str(p.xct_local)
            i = [k for k, x in enumerate(p.xct_options) if same(x['local'], xct_path)]
            self.xct_combo.setCurrentIndex(i[0] if i else -1)
            lt.xct.path_edit.setText(xct_path)
            lt.xct._refresh_keys()
            for box, paths in ((lt.ut, [str(p.ut_local)]), (lt.xct, [str(x['local']) for x in p.xct_options if x['local']])):
                c = QtWidgets.QCompleter(paths, box)
                c.setCompletionMode(QtWidgets.QCompleter.CompletionMode.UnfilteredPopupCompletion)
                box.path_edit.setCompleter(c)
            self.win.reg_tab.form.set(G_.PipelineConfig.from_dict(
                dec.get('pipeline_config') or {**rs.PIPELINE_OVERRIDES, **o.get('pipeline', {}),
                                               'ut_front_wall_mm': k0 * dz}))
            self.current, self.hole_init, self.snap, self.result = p, None, None, None
            self.note.setText(dec.get('note', ''))
            self.win.setWindowTitle(f"XCT ↔ UT registration — {p.key}")
            self.win.tabs.setCurrentWidget(lt)
            self.update_state()

        def sync_xct(self, text):
            p = self.current
            if p is not None:
                i = [k for k, x in enumerate(p.xct_options) if same(x['local'], text.strip() or None)]
                self.xct_combo.setCurrentIndex(i[0] if i else -1)

        def pick_xct(self, i):
            x = self.xct_combo.itemData(i)
            if x and x['local']:
                box = self.win.load_tab.xct
                box.path_edit.setText(str(x['local']))
                box._refresh_keys()
                if x['voxel_mm']:
                    box.set_orientation(spacing=(x['voxel_mm'],) * 3)

        def load(self):
            self.suggest()
            if self.current is None:
                return
            lt = self.win.load_tab
            lt.ut.load()
            lt.xct.load()
            self.update_state()

        def loaded_pair(self):
            """The sample on the Load tab resolved to the files loaded there (raises if they do not match it)."""
            lt = self.win.load_tab
            if self.current is None or lt.ut.raw is None or lt.xct.raw is None:
                return None
            return resolve(self.current, lt.ut.path, lt.xct.path)

        def inputs_now(self):
            lt = self.win.load_tab
            if lt.ut.raw is None or lt.xct.raw is None:
                return None
            return {m: {'settings': box.settings(), 'raw_shape': list(box.raw.shape)}
                    for m, box in (('ut', lt.ut), ('xct', lt.xct))}

        def start_from_holes(self):
            try:
                p = self.loaded_pair()
            except ValueError as e:
                QtWidgets.QMessageBox.warning(self, 'Hole fit', str(e))
                return
            inp = self.inputs_now()
            if p is None or inp is None:
                QtWidgets.QMessageBox.information(self, 'Hole fit', 'Load the volumes first.')
                return
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                hres = holes_result(p, wait=True, voxel_mm=inp['xct']['settings']['spacing_zyx_mm'][1])
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            if hres is None or hres.fit is None:
                QtWidgets.QMessageBox.warning(self, 'Hole fit', 'No hole fit for this sample.')
                return
            Gu, gu, ut_view = rs.view_frame(inp['ut']['settings'], inp['ut']['raw_shape'])
            Gx, gx, _ = rs.view_frame(inp['xct']['settings'], inp['xct']['raw_shape'])
            shape = [inp['ut']['raw_shape'][a] for a in ut_view['axes']]
            sp = ut_view['spacing_zyx_mm']
            center = (0.5 * (shape[2] - 1) * sp[2], 0.5 * (shape[1] - 1) * sp[1])
            init = rs.hole_fit_to_init(hres.fit, (Gu, gu), (Gx, gx), center)
            if init is None:
                QtWidgets.QMessageBox.warning(self, 'Hole fit', 'The hole fit is mirrored with respect to these views: '
                                              'change the XCT flips / axis order on the Load tab, or do not use it.')
                return
            form = self.win.reg_tab.form
            c = form.get()
            c.init_params, c.init_center_mm = init
            form.set(c)
            self.hole_init = init[0]
            self.win.tabs.setCurrentWidget(self.win.reg_tab)
            QtWidgets.QMessageBox.information(
                self, 'Hole fit', f"Initial transform set from {len(hres.fit.pairs)} holes (fit RMS "
                f"{hres.fit.rms_mm:.2f} mm): rz {np.degrees(init[0][2]):+.2f}°, tx {init[0][3]:+.2f}, ty {init[0][4]:+.2f} mm."
                "\n\nRun registration to start from it (untick 'start from the given transform' to go back).")

        # -------------------------------------------------------------- result
        def snapshot(self):
            try:
                p, err = self.loaded_pair(), None
            except ValueError as e:
                p, err = None, str(e)
            self.snap = (p, self.inputs_now(), err, self.current)

        def on_result(self, res):
            self.result = (res, self.snap)
            self.update_state()

        def outcome(self, wait=False):
            res, (p, inp, _, _) = self.result
            return rs.from_gui(res, inp, p.xct_voxel_mm, record(p),
                               holes_result(p, wait, inp['xct']['settings']['spacing_zyx_mm'][1]),
                               init_from_holes=self.hole_init is not None and res.config.init_params == self.hole_init)

        def usable(self):
            if self.result is None or self.result[1] is None:
                return False
            p, inp, err, cur = self.result[1]
            return p is not None and inp is not None and cur is self.current

        def update_state(self):
            p = self.current
            lt = self.win.load_tab
            if p is None:
                self.state.setText('select a sample')
            elif self.result is not None and self.result[1] and self.result[1][2] and self.result[1][3] is p:
                self.state.setText(f"<span style='color:#c62828'>this result cannot be stored: {self.result[1][2]}</span>")
            elif not self.usable():
                self.state.setText(f"<b>{p.key}</b>: " + ("run the registration" if lt.ut.raw is not None and
                                                          lt.xct.raw is not None else
                                                          "paths suggested on the Load tab; press Load volumes"))
            else:
                try:
                    out = self.outcome()
                    f, hc = out.final.params(), out.hole_check
                    self.state.setText(
                        f"<b>{p.key}</b> result {out.result.final}: rz {f['rz_deg']:+.2f}°, tx {f['tx_mm']:+.2f}, "
                        f"ty {f['ty_mm']:+.2f}, tz {f['tz_mm']:+.2f} mm<br>stored frames: rotation "
                        f"{out.info['rotation_deg']:+.2f}°, mirror {'yes' if out.info['mirror'] else 'no'}<br>"
                        + (f"hole check <b>{hc['rms_mm']:.2f} mm</b> RMS ({len(hc['pairs'])} holes)" if hc else
                           'hole check: ' + ('detecting holes…' if not holes_future(p).done() else 'not available')))
                except Exception as e:  # noqa: BLE001
                    self.state.setText(f"<span style='color:#c62828'>cannot convert this result: {e}</span>")
            busy = self.win.reg_tab.worker is not None and not self.win.reg_tab.run_btn.isEnabled()
            self.btn_ok.setEnabled(self.usable() and not busy and not self.result[1][0].existing_registration)
            self.btn_holes.setEnabled(p is not None and not busy and lt.ut.raw is not None and lt.xct.raw is not None)
            self.btn_no.setEnabled(p is not None and not p.existing_registration)

        def accept(self):
            if not self.usable():
                return
            p = self.result[1][0]
            QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                out = self.outcome(wait=True)
            except Exception as e:  # noqa: BLE001
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(self, 'Cannot convert', f"{e}\n\n{traceback.format_exc()}")
                return
            QtWidgets.QApplication.restoreOverrideCursor()
            f, hc = out.final.params(), out.hole_check
            warn = list(out.result.warnings) + out.info['conversion_warnings']
            if hc and hc['rms_mm'] > 1.0:
                warn.append(f"the hole check is {hc['rms_mm']:.2f} mm")
            text = (f"{p.key}\n\nfinal {out.result.final}: rz {f['rz_deg']:+.3f}°, tx {f['tx_mm']:+.3f}, "
                    f"ty {f['ty_mm']:+.3f}, tz {f['tz_mm']:+.3f} mm (start: {out.info['init']})\n"
                    f"in the stored TIFF frames: rotation {out.info['rotation_deg']:+.3f}°, "
                    f"mirror {'yes' if out.info['mirror'] else 'no'}\n"
                    + (f"hole check: {hc['rms_mm']:.2f} mm RMS\n" if hc else "no hole check\n")
                    + "\nregistration_matrix (database):\n" + '\n'.join('   ' + '  '.join(f"{v:+.6f}" for v in r) for r in out.M)
                    + ('\n\nWARNINGS:\n' + '\n'.join('• ' + w for w in warn) if warn else '')
                    + "\n\nSave this registration for the database?")
            if QtWidgets.QMessageBox.question(self, 'Accept registration', text) != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            d = rdb.decision_record(p, out, out.info['options'], self.note.text().strip())
            rdb.save_decision(cfg, p.key, d)
            base = rdb.write_record(cfg, p, d)
            self.win.statusBar().showMessage(f"{p.key} accepted; record {base.name}.txt next to the processed TIFF", 15000)
            self.after_decision(p)

        def reject(self):
            p = self.current or self.selected()
            if p is None or p.problem or p.existing_registration:
                return
            rdb.save_decision(cfg, p.key, {'status': 'rejected', 'decided_at': st._now(), 'note': self.note.text().strip(),
                                           'xct_unc': p.xct_unc})
            self.win.statusBar().showMessage(f"{p.key} rejected", 15000)
            self.after_decision(p)

        def after_decision(self, p):
            keys = [q.key for q in self.shown]
            i = keys.index(p.key) if p.key in keys else -1
            nxt = keys[i + 1] if 0 <= i < len(keys) - 1 else None
            self.refresh_list(keep=nxt if self.filter.currentText() == 'pending' else (nxt or p.key))

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = G_.MainWindow()
    panel = Panel(win)
    dock = QtWidgets.QDockWidget('Database samples', win)
    dock.setWidget(panel)
    dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                     | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    win.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    win.resize(1850, 1000)
    win.show()
    if args.screenshot:  # testing aid
        QtCore.QTimer.singleShot(int(args.screenshot_delay * 1000), lambda: (win.grab().save(args.screenshot), app.quit()))
    try:
        return app.exec()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', nargs='?', default='review', choices=['review', 'status', 'db-plan', 'db-insert'])
    ap.add_argument('--config', default=str(DEFAULT_CONFIG))
    ap.add_argument('--campaign', nargs='*', help='only these campaigns (e.g. Na_Panels Pegaso JI)')
    ap.add_argument('--all', action='store_true', help='review: show every volume, not only pending')
    ap.add_argument('--yes', action='store_true', help='db-insert: do not ask for confirmation')
    ap.add_argument('--screenshot', help=argparse.SUPPRESS)
    ap.add_argument('--screenshot-delay', type=float, default=20, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    cfg = st.load_config(args.config)
    items, problems = st.discover(cfg)
    for p in problems:
        print('WARNING', p)
    r = {'review': cmd_review, 'status': cmd_status, 'db-plan': cmd_db_plan,
         'db-insert': cmd_db_insert}[args.command](cfg, items, args)
    return r if isinstance(r, int) else 0


if __name__ == '__main__':
    sys.exit(main())
