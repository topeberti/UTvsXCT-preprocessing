"""PAUT <-> XCT registrations: which pairs to register, review decisions, and the database inserts.

Stored like the registrations already in the database (``dbtools.load.load_registration``):
reference measurement = the processed PAUT TIFF (02_Processed), registered measurement = the
sample's aligned XCT (``aligned = True``), ``registration_matrix`` = the 3x3 in-plane matrix in the
existing convention (:data:`registration.CONVENTION`) computed from the result of
testing/xct_ut_registration_gui.py (:mod:`.registration_sitk`), metadata ``type`` (registration method) and ``axes`` ``['x', 'y']``,
plus tz, the Euler3D transform, the stages, the pipeline configuration and the hole check.
Additions only: a pair that already has a registration in the database is skipped.

Review decisions live in ``registration_review.json`` next to ``review.json`` on the NAS.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import numpy as np

from . import registration as rg
from . import storage as st
from .database import unc

DECISIONS = 'registration_review.json'


# ----------------------------------------------------------------------------- paths

def unc_mounts(cfg):
    """{UNC share prefix: local mount} from the config (``unc_mounts``), or derived from
    ``output_unc_root`` / ``output_root`` (their common trailing folders are cut off)."""
    if cfg.get('unc_mounts'):
        return dict(cfg['unc_mounts'])
    u = list(PureWindowsPath(cfg['output_unc_root']).parts)
    m = list(Path(cfg['output_root']).parts)
    while u and m and u[-1] == m[-1]:
        u.pop()
        m.pop()
    return {str(PureWindowsPath(*u)).rstrip('\\'): str(Path(*m))}


def to_local(cfg, unc_path):
    for share, mount in unc_mounts(cfg).items():
        if unc_path.lower().startswith(share.lower()):
            rest = unc_path[len(share):].lstrip('\\').replace('\\', '/')
            return Path(mount) / rest
    raise ValueError(f"no local mount for {unc_path} (add it to 'unc_mounts' in the config)")


# ----------------------------------------------------------------------------- decisions

def load_decisions(cfg):
    p = Path(cfg['output_root']) / DECISIONS
    return json.loads(p.read_text()) if p.exists() else {}


def save_decision(cfg, key, decision):
    d = load_decisions(cfg)
    d[key] = decision
    st._write_json(Path(cfg['output_root']) / DECISIONS, d)


# ----------------------------------------------------------------------------- pairs

@dataclass
class Pair:
    item: st.Item
    ut_unc: str
    ut_local: Path
    ut_id: int | None               # measurement id of the processed PAUT in the DB (None: not inserted yet)
    xct_id: int | None
    xct_unc: str | None
    xct_voxel_mm: float | None
    existing_registration: int | None
    problem: str | None = None      # why it cannot be registered
    xct_local: Path | None = None
    xct_options: list = field(default_factory=list)   # every XCT measurement of the sample: dicts id, unc, local,
    #                                                   voxel_mm, aligned (path suggestions of the GUI)

    @property
    def key(self):
        return self.item.key


def approved_items(cfg, items):
    review = st.load_review(cfg)
    root = Path(cfg['output_root'])
    return [it for it in items if review.get(it.key, {}).get('status') == 'approved'
            and (root / it.rel('processed', '.tif')).exists()]


def find_pairs(cfg, items, conn):
    """One Pair per approved processed PAUT volume, with its sample's aligned XCT (read-only queries)."""
    cur = conn.cursor()
    cur.execute("select id, file_path, parent_measurement_id from measurements")
    rows_m = cur.fetchall()
    by_path = {p.lower(): i for i, p, _ in rows_m}
    parent = {i: par for i, _, par in rows_m}
    has_child = {par for _, _, par in rows_m if par is not None}
    cur.execute("""select s.name, x.id, x.file_path, x.measurementtype_id, lower(coalesce(a.value, '')) = 'true'
                   from samples s join sample_measurements sm on sm.sample_id = s.id
                   join measurements x on x.id = sm.measurement_id
                   join measurementtypes t on t.id = x.measurementtype_id
                   left join measurement_metadata a on a.measurement_id = x.id and a.key = 'aligned'
                   where t.name ilike '%%XCT%%'""")
    xct, all_xct = {}, {}
    rows = cur.fetchall()
    cur.execute("select measurementtype_id, value from measurementtype_metadata where key = 'voxel_size'")
    voxel = {t: float(v) for t, v in cur.fetchall()}
    for name, i, path, tid, aligned in rows:
        xct.setdefault(name, []).append((i, path, tid, bool(aligned)))
        try:
            local = to_local(cfg, path)
        except ValueError:
            local = None
        all_xct.setdefault(name, []).append({'id': i, 'unc': path, 'local': local, 'voxel_mm': voxel.get(tid),
                                             'aligned': bool(aligned)})
    cur.execute("select reference_measurement_id, registered_measurement_id, id from registrations")
    existing = {(a, b): r for a, b, r in cur.fetchall()}
    cur.close()
    pairs = []
    for it in approved_items(cfg, items):
        rel = it.rel('processed', '.tif')
        ut_unc = unc(cfg, rel)
        p = Pair(it, ut_unc, Path(cfg['output_root']) / rel, by_path.get(ut_unc.lower()), None, None, None, None)
        # the registered XCT is the last measurement of the sample's XCT chain (parent -> child), as for every
        # registration already in the DB; it must be (or descend from) the aligned volume
        mine = xct.get(it.sample, [])
        aligned_ids = {c[0] for c in mine if c[3]}

        def from_aligned(i):
            while i is not None:
                if i in aligned_ids:
                    return True
                i = parent.get(i)
            return False
        cands = [c[:3] for c in mine if c[0] not in has_child and from_aligned(c[0])]
        p.xct_options = sorted(({**o, 'registration': existing.get((p.ut_id, o['id']))} for o in all_xct.get(it.sample, [])),
                               key=lambda o: (not o['aligned'], o['id']))
        if not cands:
            p.problem = 'no aligned XCT for this sample in the DB'
        elif len(cands) > 1:
            p.problem = (f"{len(cands)} last XCT measurements (chain ends) for this sample: "
                         + ', '.join(str(c[0]) for c in cands))
        else:
            p.xct_id, p.xct_unc, tid = cands[0]
            p.xct_voxel_mm = voxel.get(tid)
            p.xct_local = to_local(cfg, p.xct_unc)
            if p.xct_voxel_mm is None:
                p.problem = f"XCT measurement type {tid} has no voxel_size"
            resc = p.xct_local.with_name(p.xct_local.stem + '_rescaled' + p.xct_local.suffix)
            if not p.xct_local.stem.endswith('_rescaled') and resc.exists():
                p.problem = (f"{resc.name} exists on the share but is not in the DB (the chain ends at measurement "
                             f"{p.xct_id}): add it as a child of {p.xct_id} first")
            if p.ut_id is not None:   # already registered with any XCT of the sample
                p.existing_registration = next((o['registration'] for o in p.xct_options if o['registration']), None)
        pairs.append(p)
    return pairs


def status(pair, decisions):
    if pair.problem:
        return 'not possible'
    if pair.existing_registration:
        return 'in DB'
    d = decisions.get(pair.key, {})
    if d.get('status') == 'accepted' and d.get('xct_unc') in [o['unc'] for o in pair.xct_options]:
        return 'accepted'
    if d.get('status') == 'rejected':
        return 'rejected'
    return 'pending'


# ----------------------------------------------------------------------------- one registration

def decision_record(pair, out, options, note=''):
    """Everything about an accepted registration (registration_review.json, the .txt/.json record and
    the database). ``out``: :class:`registration_sitk.Outcome`."""
    from dataclasses import asdict
    from . import registration_sitk as rs
    res, f = out.result, out.final
    hu = out.holes
    return {
        'status': 'accepted', 'decided_at': st._now(), 'note': note,
        'ut_unc': pair.ut_unc, 'xct_unc': pair.xct_unc, 'xct_measurement_id': pair.xct_id,
        'xct_voxel_mm': pair.xct_voxel_mm, 'xct_voxel_used_mm': out.info.get('xct_voxel_used_mm', pair.xct_voxel_mm),
        'ut_spacing_zyx_mm': out.info['ut_spacing_zyx'],
        'registration_matrix': [[float(v) for v in row] for row in out.M],
        'matrix_mm': [[float(out.R[0, 0]), float(out.R[0, 1]), float(out.t[0])],
                      [float(out.R[1, 0]), float(out.R[1, 1]), float(out.t[1])], [0.0, 0.0, 1.0]],
        'rotation_deg': out.info['rotation_deg'], 'mirror': out.info['mirror'],
        'final_stage': res.final,
        'euler3d': {'parameters_rad_mm': [float(v) for v in f.transform.GetParameters()],
                    'center_xyz_mm': [float(v) for v in f.transform.GetCenter()],
                    'convention': 'SimpleITK Euler3D [rx, ry, rz, tx, ty, tz], UT physical mm (x = scan, y = index, '
                                  'z = depth of the UT view) -> XCT view physical mm (x, y, z of the XCT view)'},
        'ut_view': out.info.get('ut_view'), 'xct_view': out.info['xct_view'], 'init': out.info['init'],
        'stages': [{'name': s_.name, 'cost': float(s_.cost), 'footprint_dice': float(s_.footprint_dice),
                    'accepted': bool(s_.accepted), 'note': s_.note,
                    **{k: float(v) for k, v in s_.params().items()}} for s_ in res.stages],
        'pipeline_config': asdict(res.config), 'z_init': res.z_info,
        'warnings': list(res.warnings) + list(out.info.get('conversion_warnings', [])),
        'hole_check': out.hole_check,
        'holes_mm': None if hu is None else {'ut': [[h['row_mm'], h['col_mm']] for h in hu.ut.holes],
                                             'xct': [[h['row_mm'], h['col_mm']] for h in hu.xct.holes]},
        'options': options, 'ut_front_wall_sample': out.info['ut_front_wall_sample'],
        'registration_type': rs.REGISTRATION_TYPE, 'code_version': st.code_version(),
    }


def _m(key, value, typ):
    return {'key': key, 'value': value if isinstance(value, str) else json.dumps(value, default=float), 'type': typ}


def db_metadata(d):
    f = next(s for s in d['stages'] if s['name'] == d['final_stage'])
    hc = d.get('hole_check') or {}
    meta = [
        _m('convention', rg.CONVENTION, 'text'),
        _m('mirror', 'True' if d['mirror'] else 'False', 'bool'),
        _m('rotation', round(d['rotation_deg'], 6), 'deg'),
        _m('matrix_mm', d['matrix_mm'], 'list'),
        _m('tz', round(f['tz_mm'], 6), 'mm'),
        _m('final_stage', d['final_stage'], 'text'),
        _m('final_cost', f['cost'], 'Mattes MI cost'),
        _m('footprint_dice', f['footprint_dice'], 'ratio'),
        _m('euler3d', d['euler3d'], 'json'),
        _m('xct_view', d['xct_view'], 'json'),
        _m('ut_view', d.get('ut_view'), 'json'),
        _m('initialisation', d['init'], 'text'),
        _m('stages', d['stages'], 'json'),
        _m('pipeline_config', d['pipeline_config'], 'json'),
        _m('ut_spacing_zyx', d['ut_spacing_zyx_mm'], 'mm list'),
        _m('xct_voxel_size', d['xct_voxel_mm'], 'mm'),
        _m('xct_voxel_size_used', d.get('xct_voxel_used_mm', d['xct_voxel_mm']), 'mm'),
        _m('reviewed', f"accepted {d['decided_at']}" + (f": {d['note']}" if d.get('note') else ''), 'text'),
        _m('software', f"code {d['code_version']}", 'text'),
    ]
    if hc:
        meta.append(_m('hole_check_rms', round(hc['rms_mm'], 4), 'mm'))
    if d.get('warnings'):
        meta.append(_m('warnings', d['warnings'], 'list'))
    return meta


def build_plan(cfg, pairs):
    decisions = load_decisions(cfg)
    plan, skipped = [], []
    for p in pairs:
        s = status(p, decisions)
        if s != 'accepted':
            skipped.append((p.key, s if not p.problem else p.problem))
            continue
        if p.ut_id is None:
            skipped.append((p.key, 'processed PAUT not in the DB yet (run paut_review.py db-insert first)'))
            continue
        d = decisions[p.key]
        plan.append({'key': p.key, 'matrix': d['registration_matrix'], 'reference': p.ut_unc, 'registered': d['xct_unc'],
                     'type': d['registration_type'], 'axes': ['x', 'y'], 'metadata': db_metadata(d)})
    return plan, skipped


def apply_plan(plan, conn, load, log=print):
    n = 0
    for r in plan:
        rid = load.load_registration(conn, r['matrix'], r['reference'], r['registered'], r['type'], r['axes'],
                                     additional_metadata=[dict(m) for m in r['metadata']])
        if rid in (-1, None):
            raise RuntimeError(f"insert failed for {r['key']} (stopped; earlier inserts are kept)")
        n += 1
        log(f"registration {rid}: {r['key']}")
    return n


def describe(pair, d):
    """Readable record of an accepted registration (written next to the processed TIFF)."""
    f = next(s_ for s_ in d['stages'] if s_['name'] == d['final_stage'])
    hc = d.get('hole_check')
    L = [f"PAUT <-> XCT registration: {pair.key}", "=" * 72,
         f"Reference (UT)      : {d['ut_unc']}",
         f"Registered (XCT)    : {d['xct_unc']}  (DB measurement {d['xct_measurement_id']})",
         f"Accepted on         : {d['decided_at']}" + (f"   note: {d['note']}" if d.get('note') else ''),
         f"Code                : {d['code_version']}", "",
         "Method", "   " + d['registration_type'] + ".",
         f"   XCT view: axes {d['xct_view']['axes']} (Z, Y, X <- raw axes), flips {d['xct_view']['flips_zyx']}; "
         f"UT view: axes {(d.get('ut_view') or {}).get('axes')}, flips {(d.get('ut_view') or {}).get('flips_zyx')}"
         + (", Hilbert envelope along depth." if ((d.get('options') or {}).get('gui_inputs') or {}).get('ut', {})
            .get('settings', {}).get('envelope') else "."),
         f"   UT front wall: {d['z_init'].get('ut_front_wall_mm', float('nan')):.3f} mm"
         + (" (fixed at the entrance echo of the processed PAUT)" if d['pipeline_config'].get('ut_front_wall_mm')
            else " (detected)") + f", XCT top surface {d['z_init'].get('ct_top_surface_mm', float('nan')):.3f} mm.",
         f"   Start: " + {'holes': 'through-hole fit', 'given': 'given initial transform'}.get(
             d['init'], 'pipeline initializer (moments' + (' + 90 degree search)' if d['pipeline_config'].get('search_rot90')
                                                            else ')')) + ".",
         "", "Stages"]
    for s_ in d['stages']:
        L.append(f"   {s_['name']:8s} cost {s_['cost']:+.6f}  footprint Dice {s_['footprint_dice']:.3f}  "
                 f"rz {s_['rz_deg']:+.3f} deg  tx {s_['tx_mm']:+.3f}  ty {s_['ty_mm']:+.3f}  tz {s_['tz_mm']:+.3f} mm  "
                 f"{'accepted' if s_['accepted'] else 'rejected'}")
    L += ["", "Result",
          f"   final {d['final_stage']}: rz {f['rz_deg']:+.3f} deg, tx {f['tx_mm']:+.3f}, ty {f['ty_mm']:+.3f}, "
          f"tz {f['tz_mm']:+.3f} mm (Euler3D, UT -> XCT view)",
          f"   in the stored frames: rotation {d['rotation_deg']:+.3f} deg, mirror {'yes' if d['mirror'] else 'no'} "
          f"(the mirror comes from the XCT view)"]
    if hc:
        L.append(f"   hole check: the UT hole centres moved by the final transform are {hc['rms_mm']:.3f} mm RMS from the "
                 f"XCT hole centres (" + ', '.join(f"{v:.2f}" for v in hc['distance_mm']) + " mm)")
    for w in d.get('warnings') or []:
        L.append(f"   warning: {w}")
    L += ["", "Matrix in UT mm -> XCT mm ([row, col] = (index, scan) of the PAUT; (Y, X) of the XCT TIFF):",
          *[f"   {row}" for row in d['matrix_mm']],
          f"XCT voxel used {d.get('xct_voxel_used_mm', d['xct_voxel_mm'])} mm (database voxel_size "
          f"{d['xct_voxel_mm']} mm)" + ("; the matrix includes the scale between them" if
                                        abs(d.get('xct_voxel_used_mm', d['xct_voxel_mm']) - d['xct_voxel_mm']) > 1e-9 else ""),
          "registration_matrix (database convention):", *[f"   {row}" for row in d['registration_matrix']],
          "   " + rg.CONVENTION, ""]
    return '\n'.join(L) + '\n'


def write_record(cfg, pair, d):
    """``<sample>_registration_xct.txt`` and ``.json`` next to the processed TIFF (text records only)."""
    base = Path(cfg['output_root']) / pair.item.rel('processed', '_registration_xct')
    st._write_json(base.with_name(base.name + '.json'), d)       # not with_suffix: names like 1.10
    tmp = base.with_name(base.name + '.txt.part')
    tmp.write_text(describe(pair, d))
    tmp.replace(base.with_name(base.name + '.txt'))
    return base
