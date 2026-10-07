"""
Add the PAUT volumes stored on the NAS to the project database (``dbtools``), additions only.

* one measurement type per acquisition (``db_measurementtype`` in the config), created if missing,
  with the acquisition settings that are constant for it (from the FocusData export metadata);
* one measurement per ``01_Raw`` TIFF, linked to its sample;
* one measurement per approved ``02_Processed`` TIFF, child of its raw measurement, with the
  applied transforms in ``transformations`` and the processing record paths as metadata.

Nothing existing is modified: rows are only inserted, measurements whose path is already in the
DB are skipped, and samples that are not in the DB are reported and skipped.
:func:`build_plan` computes everything without writing; :func:`apply_plan` performs the inserts.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import storage as st

CONSTANT_FIELDS = ('freq_mhz', 'digitizing_hz', 'voltage_v', 'pulse_width_s', 'filter', 'v_material', 'v_water',
                   'inspection', 'compression', 'averaging', 'gain_db', 'angle_deg', 'index_res_mm', 'scan_res_mm',
                   'sample_s', 'depth_res_mm')


def unc(cfg, rel):
    return cfg['output_unc_root'].rstrip('\\') + '\\' + str(rel).replace('/', '\\')


def _acq_settings(md):
    ch = md['channels'][0]
    b = ch['beams'][0]
    g = b['gates'][0]
    dg = g['data_groups'][0]
    n = dg['sample_quantity']
    return {'freq_mhz': round(ch['nominal_frequency_mhz'], 4), 'digitizing_hz': ch['digitizing_frequency_hz'],
            'voltage_v': ch['pulser_voltage_v'], 'pulse_width_s': ch['pulse_width_s'],
            'filter': json.dumps(ch.get('filter')), 'v_material': ch['part_parameters']['material_velocity_m_s'],
            'v_water': ch['part_parameters']['interface_velocity_m_s'],
            'inspection': ch['part_parameters']['inspection_type'], 'compression': ch.get('compression'),
            'averaging': ch.get('averaging'), 'gain_db': b['gain_db'], 'angle_deg': b['angle_degrees'],
            'index_res_mm': round(dg['index_resolution_m'] * 1e3, 6), 'scan_res_mm': round(dg['scan_resolution_m'] * 1e3, 6),
            'sample_s': dg['sample_resolution_s'], 'depth_res_mm': round(g['width_mm'] / n, 6),
            'gate_start_mm': round(g['start_mm'], 6)}


def _m(key, value, typ):
    return {'key': key, 'value': str(value), 'type': typ}


def measurementtype_metadata(spec, s):
    meta = [_m('technique', 'Ultrasound Phased Array Pulse Echo', 'nominal'),
            _m('equipment', 'FocusData (.fpd), exported with ut-ideko/export_fpd_folder.py', 'nominal'),
            _m('transducer_elements', spec.get('elements', ''), 'cardinal'),
            _m('transducer_nominal_freq', s['freq_mhz'], 'MHz'),
            _m('x_resolution', s['scan_res_mm'], 'mm'), _m('y_resolution', s['index_res_mm'], 'mm'),
            _m('z_resolution', s['depth_res_mm'], 'mm'),
            _m('sampling_freq', s['digitizing_hz'] / 1e6, 'MHz'), _m('gain', s['gain_db'], 'dB'),
            _m('sound_velocity_material', s['v_material'], 'm/s'), _m('sound_velocity_water', s['v_water'], 'm/s'),
            _m('voltage', s['voltage_v'], 'V'), _m('pulse_width', s['pulse_width_s'] * 1e9, 'ns'),
            _m('filter', s['filter'], 'nominal'), _m('compression', s['compression'], 'cardinal'),
            _m('averaging', s['averaging'], 'cardinal'), _m('beam_angle', s['angle_deg'], 'deg'),
            _m('inspection_type', s['inspection'], 'nominal'),
            _m('axes', 'stored TIFFs: z = depth, y = index (array), x = scan', 'nominal')]
    meta += [_m(k, v, 'nominal') for k, v in spec.get('extra', {}).items()]
    return meta


def build_plan(cfg, items, db):
    """Everything that would be inserted, without writing. ``db`` is the ``dbtools`` module."""
    samples = set(db.get_data('samples').name_sample)
    existing_paths = set(db.get_data('measurements').file_path_measurement)
    types = db.get_data('measurementtypes')
    type_ids = dict(zip(types.name_measurementtype, types.id_measurementtype))
    review = st.load_review(cfg)
    root = Path(cfg['output_root'])
    plan = {'types': [], 'raw': [], 'processed': [], 'skipped': [], 'problems': []}

    by_acq = {}
    for it in items:
        by_acq.setdefault(it.acq_key, []).append(it)
    acq_type = {}
    for acq_key, its in by_acq.items():
        spec = st.acquisition_entry(cfg, acq_key).get('db_measurementtype')
        if not spec:
            plan['problems'].append(f"{acq_key}: no db_measurementtype in the config; its volumes are skipped")
            continue
        settings = [_acq_settings(json.loads(i.metadata.read_text())) for i in its]
        varying = [f for f in CONSTANT_FIELDS if len({json.dumps(s[f]) for s in settings}) > 1]
        if varying:
            plan['problems'].append(f"{acq_key}: settings differ between files ({varying}); not one measurement type")
            continue
        acq_type[acq_key] = spec['name']
        if spec['name'] not in type_ids:
            plan['types'].append({'name': spec['name'], 'metadata': measurementtype_metadata(spec, settings[0])})

    for it in items:
        if it.acq_key not in acq_type:
            continue
        raw_rel, proc_rel = it.rel('raw', '.tif'), it.rel('processed', '.tif')
        if it.sample not in samples:
            plan['skipped'].append((it.key, 'sample not in the DB'))
            continue
        if not (root / raw_rel).exists():
            plan['skipped'].append((it.key, '01_Raw TIFF not on the NAS'))
            continue
        conv = json.loads((root / it.rel('raw', '_conversion.json')).read_text())
        raw_path = unc(cfg, raw_rel)
        z, y, x = conv['shape_zyx']
        s = _acq_settings(json.loads(it.metadata.read_text()))
        if raw_path in existing_paths:
            plan['skipped'].append((it.key, 'raw already in the DB'))
        else:
            plan['raw'].append({
                'key': it.key, 'file_path': raw_path, 'type': acq_type[it.acq_key], 'shape': (z, y, x),
                'sample': it.sample,
                'metadata': [_m('original_id', it.original_id, 'nominal'),
                             _m('campaign', it.campaign, 'nominal'), _m('acquisition', it.acquisition, 'nominal'),
                             _m('source_fpd', conv.get('source_fpd'), 'path'),
                             _m('exported_at_utc', conv.get('exported_at_utc'), 'nominal'),
                             _m('source_npy_sha256', conv['source_npy_sha256'], 'text'),
                             _m('tif_sha256', conv['tif_sha256'], 'text'),
                             _m('gate_start', s['gate_start_mm'], 'mm'),
                             _m('spacing_zyx', [s['depth_res_mm'], s['index_res_mm'], s['scan_res_mm']], 'mm list'),
                             _m('amplitude_units', 'percent', 'nominal'),
                             _m('conversion_record', unc(cfg, it.rel('raw', '_conversion.json')), 'path'),
                             _m('equipment_metadata', unc(cfg, it.rel('raw', '_metadata.json')), 'path')]})
        dec = review.get(it.key, {})
        if not (root / proc_rel).exists():
            continue
        if dec.get('status') != 'approved':
            plan['skipped'].append((it.key, f"processed not approved ({dec.get('status', 'pending')})"))
            continue
        proc_path = unc(cfg, proc_rel)
        if proc_path in existing_paths:
            plan['skipped'].append((it.key, 'processed already in the DB'))
            continue
        rec = json.loads((root / it.rel('processed', '_processing.json')).read_text())
        pz, py, px = (rec['output']['shape_index_scan_depth'][i] for i in (2, 0, 1))
        if rec.get('source') == 'colleague':
            md = rec['colleague_metadata']
            es, ad = md['echo_start'], md['auto_detect']
            trans = (f"Colleague's fpd batch tool (processing_version {md.get('processing_version')}): echo start "
                     f"(first |a| >= {es['threshold_percent']} % in {es['gate_mm']} mm, gaps interpolated), auto detect piece "
                     f"({ad['threshold_percent']} % in {ad['gate_mm']} mm, margin {ad['margin_mm']} mm), crop "
                     f"{md['bounds']} with depth {md['final_z_mm']} mm relative to the entrance echo, mirror "
                     f"index={md['mirror']['horizontal']} scan={md['mirror']['vertical']}. Bit-identical to "
                     f"preprocess_tools.paut on the 01_Raw TIFF. Record: {unc(cfg, it.rel('processed', '_processing.txt'))}")
            by = "colleague's fpd batch tool"
        else:
            p, e, f = rec['params'], rec.get('echo_start', {}), rec.get('footprint') or {}
            es_txt = ('echo start not applied' if 'skipped' in e else
                      f"echo start ({p['echo_start']['method']} sample with "
                      + {'abs': '|RF|', 'envelope': 'Hilbert envelope'}.get(
                          p['echo_start'].get('signal', 'abs'),
                          f"|RF| peak-held over {p['echo_start'].get('hold_samples')} and averaged over "
                          f"{p['echo_start'].get('average_samples')} samples")
                      + f" x {p['echo_start']['gain']:g} >= "
                      f"{p['echo_start']['threshold_pct']:g} % in "
                      f"{p['echo_start']['gate_mm']} mm; {e['hit_count']} hits, {e['interpolated_count']} interpolated by "
                      f"k-NN IDW; RF shifted so the entrance echo is at 0 mm)")
            fp_txt = (f"auto detect piece ({f['threshold_pct']:g} % in {f['gate_mm']} mm, margin {f['margin_mm']:g} mm)"
                      if f else 'no auto detect piece')
            trans = (f"preprocess_tools.paut (processing_version {rec['processing_version']}, code {rec['code_version']}): "
                     f"{es_txt}; {fp_txt}; crop index {rec['crop']['bounds']['index']} scan {rec['crop']['bounds']['scan']}, "
                     + (f"depth {p['depth_samples']} samples from {p['depth_range_mm'][0]} mm" if p.get('depth_mode') == 'samples'
                        else f"depth {p['depth_range_mm']} mm") + f" relative to the entrance echo; mirror index={p['mirror']['index']} "
                     f"scan={p['mirror']['scan']}. Record: {unc(cfg, it.rel('processed', '_processing.txt'))}")
            by = 'preprocess_tools.paut (produccion/paut/paut_review.py)'
        meta = [_m('processed_by', by, 'nominal'),
                _m('processing_record', unc(cfg, it.rel('processed', '_processing.txt')), 'path'),
                _m('processing_record_json', unc(cfg, it.rel('processed', '_processing.json')), 'path'),
                _m('tif_sha256', rec['tif_sha256'], 'text'),
                _m('spacing_zyx', [round(rec['output']['spacing_mm_index_scan_depth'][i], 6) for i in (2, 0, 1)], 'mm list'),
                _m('amplitude_units', 'percent', 'nominal'),
                _m('depth_origin', 'entrance echo (axes rebased to 0 in the stored TIFF)', 'nominal')]
        if rec.get('params'):
            meta.append(_m('parameters', json.dumps(rec['params']), 'json'))
        if dec.get('note'):
            meta.append(_m('review_note', dec['note'], 'text'))
        plan['processed'].append({'key': it.key, 'file_path': proc_path, 'type': acq_type[it.acq_key],
                                  'shape': (pz, py, px), 'sample': it.sample, 'parent': raw_path,
                                  'transformations': trans, 'metadata': meta})
    return plan


def summarize(plan):
    lines = [f"measurement types to create: {len(plan['types'])}"]
    lines += [f"   + {t['name']}" for t in plan['types']]
    for kind in ('raw', 'processed'):
        c = Counter(r['key'].rsplit('/', 1)[0] for r in plan[kind])
        lines.append(f"{kind} measurements to insert: {len(plan[kind])}")
        lines += [f"   {k}: {v}" for k, v in sorted(c.items())]
    c = Counter((k.rsplit('/', 1)[0], why) for k, why in plan['skipped'])
    lines.append(f"skipped: {len(plan['skipped'])}")
    lines += [f"   {k} - {why}: {v}" for (k, why), v in sorted(c.items())]
    lines += [f"PROBLEM: {p}" for p in plan['problems']]
    return '\n'.join(lines)


def apply_plan(plan, conn, db, load, log=print):
    """Insert the plan. Stops at the first failed insert (dbtools returns -1)."""
    types = db.get_data('measurementtypes')
    type_ids = dict(zip(types.name_measurementtype, types.id_measurementtype))
    for t in plan['types']:
        new_id = load.load_measurementtype(conn, t['name'], t['metadata'])
        if new_id in (-1, None):
            raise RuntimeError(f"creating measurement type {t['name']!r} failed")
        type_ids[t['name']] = int(new_id)
        log(f"measurement type {new_id}: {t['name']}")
    done = {'raw': 0, 'processed': 0}
    for kind in ('raw', 'processed'):
        for r in plan[kind]:
            z, y, x = r['shape']
            kw = dict(parent_measurement_path=r['parent'], transformations=r['transformations']) if kind == 'processed' else {}
            new_id = load.load_ut_measurement(conn, r['file_path'], int(type_ids[r['type']]), int(z), int(y), int(x),
                                              'float32', 'tif', 'RF', ['z', 'y', 'x'], [r['sample']],
                                              additional_metadata=r['metadata'], **kw)
            if new_id in (-1, None):
                raise RuntimeError(f"insert failed for {r['file_path']} (stopped; earlier inserts are kept)")
            done[kind] += 1
            log(f"{kind} {new_id}: {r['key']}")
    return done
