"""
Repeatable storage of PAUT data.

1. :func:`convert_export`  exported ``.npy`` → ``<id>.tif`` + ``<id>_conversion.json`` next to it.
2. :func:`upload_raw`      that TIFF (+ metadata and conversion record) → NAS ``01_Raw``.
3. :func:`save_processed`  an accepted result → NAS ``02_Processed`` (TIFF + a human-readable
                           ``.txt`` and a ``.json`` of everything done). The echo-start maps
                           (``.npz``) stay local, next to the export: the NAS holds only TIFF
                           volumes and their text records.

NAS layout (``output_root``)::

    <campaign>/<acquisition>/01_Raw/<Sample>/<Sample>.tif                full RF scan, (Z=depth, Y=index, X=scan)
    <campaign>/<acquisition>/01_Raw/<Sample>/<Sample>_metadata.json      equipment metadata (verbatim export JSON)
    <campaign>/<acquisition>/01_Raw/<Sample>/<Sample>_conversion.json    export → TIFF provenance and axes
    <campaign>/<acquisition>/02_Processed/<Sample>/<Sample>.tif          echo start + auto detect piece + crop
    <campaign>/<acquisition>/02_Processed/<Sample>/<Sample>_processing.json   parameters, results, checksums
    <campaign>/<acquisition>/02_Processed/<Sample>/<Sample>_processing.txt    the same, readable
    manifest.csv   review.json   README.md                       ledgers at the root

Everything is driven by ``produccion/paut/paut_config.json``; re-running only touches
new or changed files, and an existing different file on the NAS is never overwritten
silently.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .processing import PAUTVolume, PROCESSING_VERSION, load_export, merge_params, process, to_zyx

MANIFEST_FIELDS = ['timestamp', 'run_id', 'key', 'campaign', 'acquisition', 'sample', 'original_id',
                   'stage', 'path', 'sha256', 'source', 'code_version']


# ----------------------------------------------------------------------------- helpers

def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def params_sha(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def code_version():
    """Git commit of this repository (+ '-dirty' with uncommitted changes in preprocess_tools)."""
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'preprocess_tools'], cwd=repo,
                               capture_output=True, text=True).stdout.strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def new_run_id():
    return time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]


def _now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float))


def write_tif(path, data_zyx, spacing_zyx):
    """float32 ImageJ hyperstack with the voxel size in mm."""
    dz, dy, dx = spacing_zyx
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.ascontiguousarray(data_zyx, dtype=np.float32), imagej=True,
                     resolution=(1.0 / dx, 1.0 / dy), metadata={'axes': 'ZYX', 'spacing': dz, 'unit': 'mm'})


def copy_verified(src, dst):
    """Copy ``src`` to ``dst`` (via ``.part``), verifying SHA-256.

    Returns 'copied' or 'identical' (already there); raises if a *different* file exists.
    """
    src, dst = Path(src), Path(dst)
    digest = sha256(src)
    if dst.exists():
        if sha256(dst) == digest:
            return 'identical'
        raise FileExistsError(f"{dst} exists with different content; not overwritten")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.part')
    shutil.copyfile(src, tmp)
    if sha256(tmp) != digest:
        tmp.unlink()
        raise IOError(f"checksum mismatch after copying {src} → {dst}")
    tmp.replace(dst)
    return 'copied'


def _write_new(dst, write_fn):
    """Write ``dst`` through ``write_fn(tmp_path)`` + rename; returns the file's SHA-256."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.part')
    write_fn(tmp)
    tmp.replace(dst)
    return sha256(dst)


# ----------------------------------------------------------------------------- config and discovery

def load_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg['_path'] = str(Path(path).resolve())
    for k in [k for k in cfg if k.endswith('_root') and not k.endswith('unc_root')]:
        cfg[k] = str(Path(cfg[k]).expanduser())
    return cfg


def acquisition_entry(cfg, acq_key):
    entry = cfg.get('acquisitions', {}).get(acq_key)
    if entry is None:  # a new acquisition folder: same name without the leading '_'
        entry = {'folder': acq_key.lstrip('_'), 'params': cfg.get('default_acquisition_params', {})}
    return entry


def sample_name(cfg, campaign, original_id):
    """DB sample name from the original file id, using the campaign's rule (identity if none)."""
    rule = cfg.get('campaigns', {}).get(campaign, {})
    m = re.match(rule['sample_regex'], original_id) if rule.get('sample_regex') else None
    return rule['sample_format'].format(*m.groups()) if m else original_id


@dataclass
class Item:
    acq_key: str          # export folder, e.g. _raw_3p5MHz_16elem
    acquisition: str      # NAS folder, e.g. raw_3p5MHz_16elem
    campaign: str
    original_id: str      # e.g. Na_01_01
    sample: str           # DB name, e.g. Na_01_1
    npy: Path
    metadata: Path

    @property
    def key(self):
        return f"{self.campaign}/{self.acquisition}/{self.sample}"

    @property
    def local_tif(self):
        """The converted raw TIFF, next to the export."""
        return self.npy.with_name(f"{self.original_id}.tif")

    @property
    def local_conversion(self):
        return self.npy.with_name(f"{self.original_id}_conversion.json")

    def rel(self, stage, suffix=''):
        folder = {'raw': '01_Raw', 'processed': '02_Processed'}[stage]
        return Path(self.campaign) / self.acquisition / folder / self.sample / f"{self.sample}{suffix}"


def discover(cfg):
    """All exported volumes under ``export_root`` (excluded folders such as ``_old`` skipped)."""
    root = Path(cfg['export_root'])
    exclude = set(cfg.get('exclude_dirs', []))
    items, problems = [], []
    for npy in sorted(root.rglob('*_ch*_b*_g*_dg*.npy')):
        parts = npy.relative_to(root).parts
        if exclude & set(parts[:-1]):
            continue
        acq_key = parts[0]
        original_id = npy.name.rsplit('_ch', 1)[0]
        if len(parts) > 2:
            campaign = parts[1]
        else:
            campaign = next((c for p, c in cfg.get('campaign_by_prefix', {}).items()
                             if original_id.startswith(p)), None)
            if campaign is None:
                problems.append(f"{npy}: no campaign folder and no campaign_by_prefix rule")
                continue
        meta = npy.with_name(f"{original_id}_metadata.json")
        if not meta.exists():
            problems.append(f"{npy}: metadata {meta.name} missing")
            continue
        acq = acquisition_entry(cfg, acq_key)
        items.append(Item(acq_key, acq['folder'], campaign, original_id,
                          sample_name(cfg, campaign, original_id), npy, meta))
    seen = {}
    for it in items:
        if it.key in seen:
            problems.append(f"duplicate output {it.key}: {seen[it.key].npy} and {it.npy}")
        seen[it.key] = it
    return items, problems


# ----------------------------------------------------------------------------- ledgers

def append_manifest(cfg, item, run_id, stage, rel_path, digest, source):
    mf = Path(cfg['output_root']) / 'manifest.csv'
    mf.parent.mkdir(parents=True, exist_ok=True)
    new = not mf.exists()
    with open(mf, 'a', newline='') as f:
        w = csv.DictWriter(f, MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({'timestamp': _now(), 'run_id': run_id, 'key': item.key, 'campaign': item.campaign,
                    'acquisition': item.acquisition, 'sample': item.sample, 'original_id': item.original_id,
                    'stage': stage, 'path': str(rel_path), 'sha256': digest, 'source': source,
                    'code_version': code_version()})


def load_review(cfg):
    p = Path(cfg['output_root']) / 'review.json'
    return json.loads(p.read_text()) if p.exists() else {}


def save_decision(cfg, item, status, params_override=None, note=''):
    """Store the reviewer's decision ('approved' or 'rejected') and parameter changes."""
    review = load_review(cfg)
    review[item.key] = {'status': status, 'params_override': params_override or {},
                        'note': note, 'decided_at': _now()}
    _write_json(Path(cfg['output_root']) / 'review.json', review)
    write_readme(cfg)
    return review[item.key]


def prune_stale_decisions(cfg, items, log=print):
    """Approved decisions whose 02_Processed TIFF no longer exists (e.g. deleted to redo it) are moved
    to ``review_history.json``, so those volumes are pending again and start from the defaults."""
    root = Path(cfg['output_root'])
    review = load_review(cfg)
    stale = {it.key: review[it.key] for it in items
             if review.get(it.key, {}).get('status') == 'approved' and not (root / it.rel('processed', '.tif')).exists()}
    if not stale:
        return 0
    hist_path = root / 'review_history.json'
    history = json.loads(hist_path.read_text()) if hist_path.exists() else []
    history += [{'key': k, 'removed_at': _now(), 'reason': '02_Processed TIFF deleted', 'decision': d}
                for k, d in stale.items()]
    _write_json(hist_path, history)
    for k in stale:
        review.pop(k)
    _write_json(root / 'review.json', review)
    log(f"{len(stale)} approved decisions had no processed file any more; moved to review_history.json "
        f"(those volumes are pending again)")
    return len(stale)


def item_params(cfg, item, review=None):
    """Processing parameters: defaults ← acquisition (config) ← reviewer override."""
    acq = acquisition_entry(cfg, item.acq_key)
    dec = (review if review is not None else load_review(cfg)).get(item.key, {})
    return merge_params(acq.get('params'), dec.get('params_override'))


def params_override(cfg, item, params):
    """The part of ``params`` that differs from the acquisition defaults (what the reviewer changed)."""
    base = merge_params(acquisition_entry(cfg, item.acq_key).get('params'))
    out = {}
    for k, v in params.items():
        if isinstance(v, dict):
            d = {kk: vv for kk, vv in v.items() if base.get(k, {}).get(kk) != vv}
            if d:
                out[k] = d
        elif base.get(k) != v:
            out[k] = v
    return out


README = """# PAUT (16-element phased array) UT data

Written by `preprocess_tools.paut` (UTvsXCT-preprocessing), config `produccion/paut/paut_config.json`,
reviewed with `produccion/paut/paut_review.py`.

    <campaign>/<acquisition>/01_Raw/<Sample>/<Sample>.tif               full RF scan converted from the FocusData export
    <campaign>/<acquisition>/01_Raw/<Sample>/<Sample>_metadata.json     equipment metadata (verbatim)
    <campaign>/<acquisition>/01_Raw/<Sample>/<Sample>_conversion.json   export → TIFF provenance, axes in mm, checksums
    <campaign>/<acquisition>/02_Processed/<Sample>/<Sample>.tif         echo start (entrance echo at 0 mm) + auto detect piece + depth crop
    <campaign>/<acquisition>/02_Processed/<Sample>/<Sample>_processing.txt   everything done to the file, readable
    <campaign>/<acquisition>/02_Processed/<Sample>/<Sample>_processing.json  the same, machine-readable (parameters, results, checksums)

- Volumes are float32 RF in percent, axes (Z = depth, Y = index, X = scan); TIFFs carry the voxel size (mm).
- `<Sample>` is the database sample name (e.g. Na_01_1, JI_10, 1.10) or the original id for samples not yet in the DB.
- 02_Processed is computed from the 01_Raw TIFF, so it can be regenerated from this folder alone.
- Only TIFF volumes are stored here (plus small text records); exports (.npy), the colleague's .h5 files and the
  echo-start maps (.npz) stay on the processing computer, referenced by path and checksum in the records.
- `manifest.csv` lists every file written (run, checksum, source); `review.json` holds the reviewer decisions.
"""


def write_readme(cfg):
    p = Path(cfg['output_root']) / 'README.md'
    if not p.exists() or p.read_text() != README:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(README)


# ----------------------------------------------------------------------------- 1. conversion (local)

def conversion_status(item, verify=False):
    """'missing', 'current' or 'stale' (the export changed) for the local TIFF.

    Fast by default: an export not modified since its conversion record was written is
    current without re-reading it; ``verify=True`` (or a newer export) compares checksums.
    """
    if not (item.local_tif.exists() and item.local_conversion.exists()):
        return 'missing'
    if not verify and item.npy.stat().st_mtime < item.local_conversion.stat().st_mtime:
        return 'current'
    conv = json.loads(item.local_conversion.read_text())
    return 'current' if conv.get('source_npy_sha256') == sha256(item.npy) else 'stale'


def convert_export(item, run_id=None, overwrite=False):
    """Export ``.npy`` → ``<id>.tif`` (Z=depth, Y=index, X=scan) + ``<id>_conversion.json`` next to it."""
    status = conversion_status(item)
    if status == 'current':
        return 'skipped'
    if status == 'stale' and not overwrite:
        return 'stale (export changed; pass overwrite=True)'
    vol = load_export(item.npy, item.metadata)
    sp = vol.spacing_mm
    write_tif(item.local_tif, to_zyx(vol.data), (sp[2], sp[0], sp[1]))
    conv = {
        'created': _now(), 'run_id': run_id, 'code_version': code_version(),
        'source_fpd': vol.metadata.get('source_file'),
        'exporter': 'ut-ideko/export_fpd_folder.py (FocusData COM)',
        'exported_at_utc': vol.metadata.get('exported_at_utc'),
        'source_npy': item.npy.name, 'source_npy_sha256': sha256(item.npy),
        'source_metadata': item.metadata.name, 'source_metadata_sha256': sha256(item.metadata),
        'original_id': item.original_id, 'sample': item.sample,
        'campaign': item.campaign, 'acquisition': item.acquisition,
        'definition_ch_beam_gate_datagroup': [1, 1, 1, 1],
        'export_axis_order': ['index', 'scan', 'depth'], 'stored_axis_order': ['depth', 'index', 'scan'],
        'stored_axes': ['z', 'y', 'x'], 'amplitude_units': 'percent', 'dtype': 'float32',
        'shape_zyx': [int(vol.shape[2]), int(vol.shape[0]), int(vol.shape[1])],
        'axes_mm': {k: {'start': float(a[0]), 'step': s, 'count': int(a.size),
                        'values': [float(x) for x in a]}   # exact export values (keep processing identical)
                    for k, a, s in (('index', vol.index_mm, sp[0]), ('scan', vol.scan_mm, sp[1]),
                                    ('depth', vol.depth_mm, sp[2]))},
        'tif_sha256': sha256(item.local_tif),
    }
    _write_json(item.local_conversion, conv)
    return 'written'


def load_raw_tif(path, conversion=None, metadata=None) -> PAUTVolume:
    """A raw TIFF back as (index, scan, depth) with its axes from the conversion record."""
    path = Path(path)
    conversion = Path(conversion) if conversion else path.with_name(f"{path.stem}_conversion.json")
    metadata = Path(metadata) if metadata else path.with_name(f"{path.stem}_metadata.json")
    conv = json.loads(conversion.read_text())
    data = np.transpose(tifffile.imread(path), (1, 2, 0))
    ax = conv['axes_mm']
    axes = [np.asarray(ax[k]['values'], np.float64) if 'values' in ax[k]
            else ax[k]['start'] + np.arange(ax[k]['count']) * ax[k]['step'] for k in ('index', 'scan', 'depth')]
    md = json.loads(metadata.read_text()) if metadata.exists() else {}
    return PAUTVolume(data, *axes, metadata=md)


# ----------------------------------------------------------------------------- 2. raw upload (NAS)

def raw_uploaded(cfg, item):
    """True when 01_Raw on the NAS holds this item's current local TIFF."""
    nas = Path(cfg['output_root']) / item.rel('raw', '_conversion.json')
    if not (nas.exists() and item.local_conversion.exists()):
        return False
    return json.loads(nas.read_text()).get('tif_sha256') == json.loads(item.local_conversion.read_text()).get('tif_sha256')


def upload_raw(cfg, item, run_id=None, verify=False):
    """Copy the local raw TIFF, metadata and conversion record to ``01_Raw`` on the NAS (verified).

    Returns 'uploaded' or 'identical'; raises if a different file is already there. Fast by
    default: when the NAS already holds the same conversion record (which carries the TIFF's
    checksum) and a TIFF of the same size, the big files are not re-read; ``verify=True``
    re-checks every byte.
    """
    if conversion_status(item) != 'current':
        raise RuntimeError(f"{item.key}: convert the export first ({conversion_status(item)})")
    root = Path(cfg['output_root'])
    nas_conv, nas_tif = root / item.rel('raw', '_conversion.json'), root / item.rel('raw', '.tif')
    if (not verify and nas_conv.exists() and nas_tif.exists()
            and nas_conv.read_bytes() == item.local_conversion.read_bytes()
            and nas_tif.stat().st_size == item.local_tif.stat().st_size):
        return 'identical'
    results = [copy_verified(src, root / item.rel('raw', suffix))
               for src, suffix in ((item.local_tif, '.tif'), (item.metadata, '_metadata.json'),
                                   (item.local_conversion, '_conversion.json'))]
    if results[0] == 'copied':
        append_manifest(cfg, item, run_id or new_run_id(), 'raw', item.rel('raw', '.tif'),
                        json.loads(item.local_conversion.read_text())['tif_sha256'], str(item.npy))
    write_readme(cfg)
    return 'uploaded' if 'copied' in results else 'identical'


# ----------------------------------------------------------------------------- 3. processing (NAS)

def run_processing(cfg, item, params=None, review=None):
    """Process the local raw TIFF with ``params`` (default: config + previous decision). No writes."""
    params = params if params is not None else item_params(cfg, item, review)
    vol = load_raw_tif(item.local_tif, item.local_conversion, item.metadata)
    return process(vol, params)


def processed_status(cfg, item, params):
    """'missing', 'current' or 'stale' (raw, parameters or processing version changed) on the NAS."""
    pj = Path(cfg['output_root']) / item.rel('processed', '_processing.json')
    if not pj.exists():
        return 'missing'
    rec = json.loads(pj.read_text())
    raw_sha = (json.loads(item.local_conversion.read_text()).get('tif_sha256')
               if item.local_conversion.exists() else None)
    same = (rec.get('input_tif_sha256') == raw_sha and rec.get('params_sha256') == params_sha(params)
            and rec.get('processing_version') == PROCESSING_VERSION)
    return 'current' if same else 'stale'


def describe(rec, item, conv):
    """Human-readable account of everything done to the file."""
    p, es, fp, cr, out = rec['params'], rec.get('echo_start', {}), rec.get('footprint'), rec['crop'], rec['output']
    L = [f"PAUT preprocessing record: {item.key}",
         "=" * 72,
         f"Sample (DB name)      : {item.sample}",
         f"Original id           : {item.original_id}   campaign: {item.campaign}   acquisition: {item.acquisition}",
         f"Equipment file (.fpd) : {conv.get('source_fpd')}",
         f"Processed on          : {rec['created']}   run {rec['run_id']}   code {rec['code_version']}",
         f"Reviewer decision     : {rec['review']['status']} at {rec['review']['decided_at']}"
         + (f"   note: {rec['review']['note']}" if rec['review'].get('note') else ''),
         "",
         "1. Conversion (01_Raw)",
         f"   FocusData export {conv.get('source_npy')} (exported {conv.get('exported_at_utc')}) by {conv.get('exporter')},",
         f"   channel/beam/gate/data group {conv.get('definition_ch_beam_gate_datagroup')}, RF in percent, float32.",
         f"   Reordered (index, scan, depth) -> (Z=depth, Y=index, X=scan); shape ZYX {conv.get('shape_zyx')}.",
         "   Axes: " + ", ".join(f"{k} {v['count']} x {v['step']:.4g} mm from {v['start']:.4g} mm"
                                for k, v in conv['axes_mm'].items()),
         f"   {item.rel('raw', '.tif').relative_to(Path(item.campaign) / item.acquisition)} sha256 {rec['input_tif_sha256']}",
         ""]
    if 'skipped' in es:
        L += ["2. Echo start: not applied (" + es['skipped'] + ")", ""]
    else:
        L += ["2. Echo start (colleague's algorithm, verbatim)",
              ({'abs_peakhold_ma': f"   Detection signal: |RF| -> peak hold over {es.get('hold_samples')} samples (causal "
                                   f"running maximum) -> moving average over {es.get('average_samples')} samples (centred).",
                'envelope': "   Detection signal: Hilbert envelope |hilbert(RF)| of each A-scan (mean removed first)."}
               .get(es.get('detection_signal'), "   Detection signal: |RF|.") + " The shift is applied to the original RF."),
              f"   Detection: {es['method']} sample with signal x gain ({es['gain']:g}) >= {es['threshold_pct']:g} % inside "
              f"{es['gate_mm'][0]:g} .. {es['gate_mm'][1]:g} mm (samples {es['gate_samples'][0]}-{es['gate_samples'][1] - 1}).",
              f"   Hits: {es['hit_count']} of {es['trace_count']} A-scans; interpolated: {es['interpolated_count']} "
              f"(robust k-NN inverse-distance weighting of the hits, {es['neighbors']} neighbours, in mm); "
              f"unresolved (not shifted): {es['unresolved_count']}.",
              f"   Entrance echo before alignment: {es['echo_mm']['min']:.3f} .. {es['echo_mm']['max']:.3f} mm "
              f"(median {es['echo_mm']['median']:.3f}).",
              "   Every A-scan of the original RF shifted so the entrance echo is at 0 mm"
              + (f", keeping {es['pre_echo_mm']:.3f} mm before it" if es.get('pre_echo_samples') else '')
              + (f" ({es['zero_padded_samples']} zero samples added before the start of the recording so A-scans whose "
                 f"echo is close to it stay aligned)" if es.get('zero_padded_samples') else '')
              + "; whole-sample shifts copied exactly, fractional ones linearly resampled; samples past the end set to 0."
              + (f" {es['clipped_pre_echo']} A-scans had less pre-echo signal than requested." if es.get('clipped_pre_echo') else ''),
              ""]
    if fp:
        L += ["3. Auto detect piece",
              f"   A-scans whose max |a| inside {fp['gate_mm'][0]:g} .. {fp['gate_mm'][1]:g} mm (below the entrance echo) "
              f"reaches {fp['threshold_pct']:g} %: {fp['selected_traces']}; bounding box widened by {fp['margin_mm']:g} mm.", ""]
    else:
        L += ["3. Auto detect piece: not applied", ""]
    b, s0 = cr['bounds'], cr['source_start_mm']
    sh, spc = out['shape_index_scan_depth'], out['spacing_mm_index_scan_depth']
    L += ["4. Crop",
          f"   index samples {b['index'][0]}-{b['index'][1] - 1} (from {s0['index']:.3f} mm), "
          f"scan samples {b['scan'][0]}-{b['scan'][1] - 1} (from {s0['scan']:.3f} mm),",
          (f"   depth: {cr['depth_samples']} samples from {p['depth_range_mm'][0]:g} mm relative to the entrance echo "
           + (f"({cr['zero_padded_end_samples']} zero samples appended where the recording ended) "
              if cr.get('zero_padded_end_samples') else '')
           if cr.get('depth_mode') == 'samples' else
           f"   depth {p['depth_range_mm'][0]:g} .. {p['depth_range_mm'][1]:g} mm relative to the entrance echo ")
          + f"(samples {b['depth'][0]}-{b['depth'][1] - 1}; first kept sample at {s0['depth']:.3f} mm).",
          "   All axes rebased to start at 0 in the stored volume.", "",
          "5. Mirror",
          f"   index axis flipped: {rec['mirror']['index']}, scan axis flipped: {rec['mirror']['scan']}", "",
          "Output (02_Processed)",
          f"   {item.sample}.tif, shape ZYX ({sh[2]}, {sh[0]}, {sh[1]}), spacing ZYX ({spc[2]:.4g}, {spc[0]:.4g}, "
          f"{spc[1]:.4g}) mm, float32 RF %.",
          f"   sha256 {rec['tif_sha256']}",
          "",
          "Parameters changed from the acquisition defaults: "
          + (json.dumps(rec['review'].get('params_override')) if rec['review'].get('params_override') else 'none'),
          "Full parameters:", json.dumps(p, indent=2)]
    return '\n'.join(L) + '\n'


def save_processed(cfg, item, result, params, note='', run_id=None, overwrite=False):
    """Accept a processing result: upload the raw (if needed) and write 02_Processed to the NAS.

    ``result`` is ``(volume, record, qc)`` from :func:`run_processing` computed with ``params``.
    An existing processed file is replaced only with ``overwrite=True``.
    """
    run_id = run_id or new_run_id()
    root = Path(cfg['output_root'])
    tif = root / item.rel('processed', '.tif')
    status, diffs = compare_with_saved(cfg, item, params)
    if status == 'identical':
        dec = load_review(cfg).get(item.key, {})
        if dec.get('status') != 'approved' or (note and dec.get('note') != note):
            save_decision(cfg, item, 'approved', dec.get('params_override') or params_override(cfg, item, params),
                          note or dec.get('note', ''))
        return 'identical (already saved with exactly these transforms; nothing written)'
    if tif.exists() and not overwrite:
        raise FileExistsError(f"{tif} exists with different transforms:\n{describe_differences(diffs)}\n"
                              f"accept with overwrite to replace it")
    upload_raw(cfg, item, run_id)
    vol, rec, qc = result
    decision = save_decision(cfg, item, 'approved', params_override(cfg, item, params), note)
    conv = json.loads(item.local_conversion.read_text())
    sp = vol.spacing_mm
    rec = dict(rec)
    rec.update({'created': _now(), 'run_id': run_id, 'code_version': code_version(),
                'sample': item.sample, 'original_id': item.original_id, 'campaign': item.campaign,
                'acquisition': item.acquisition, 'input_tif': str(item.rel('raw', '.tif')),
                'input_tif_sha256': conv['tif_sha256'], 'params_sha256': params_sha(params),
                'review': decision})
    rec['tif_sha256'] = _write_new(tif, lambda t: write_tif(t, to_zyx(vol.data), (sp[2], sp[0], sp[1])))
    es = qc.get('echo_start')
    if es is not None:
        def _npz(t):
            with open(t, 'wb') as f:
                np.savez_compressed(f, hit=es['hit'], interpolated=es['interpolated'], applied=es['applied'],
                                    shift_samples=es['shift_samples'], echo_mm=es['echo_mm'],
                                    cscan_zero=es['cscan_zero'])
        npz = item.npy.with_name(f"{item.original_id}_echo_start.npz")   # local only, not on the NAS
        rec['echo_start_maps'] = {'file': str(npz), 'stored': 'locally next to the export (not on the NAS)',
                                  'sha256': _write_new(npz, _npz),
                                  'grid': 'full export (index, scan), before crop and mirror'}
    _write_new(root / item.rel('processed', '_processing.json'),
               lambda t: t.write_text(json.dumps(rec, indent=2, default=float)))
    _write_new(root / item.rel('processed', '_processing.txt'), lambda t: t.write_text(describe(rec, item, conv)))
    append_manifest(cfg, item, run_id, 'processed', item.rel('processed', '.tif'), rec['tif_sha256'],
                    str(item.rel('raw', '.tif')))
    return 'written'


def reject(cfg, item, params, note=''):
    """Record a rejection (nothing is written to 02_Processed)."""
    return save_decision(cfg, item, 'rejected', params_override(cfg, item, params), note)


# ----------------------------------------------------------------------------- colleague's processed .h5

def colleague_h5(cfg, item):
    """The colleague's processed .h5 for ``item`` (``reference_root/<campaign>/<id>_fpd_processed.h5``),
    if it exists and was made from this item's acquisition."""
    p = Path(cfg.get('reference_root', '')) / item.campaign / f"{item.original_id}_fpd_processed.h5"
    if not p.exists():
        return None
    import h5py
    with h5py.File(p) as h:
        src = json.loads(h['metadata_json'][()]).get('source_file', '')
    return p if item.acq_key in src.replace('\\', '/').split('/') else None


def _colleague_log(h5_path):
    log = h5_path.with_name('batch-review-log.jsonl')
    if not log.exists():
        return None
    entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    mine = [e for e in entries if any(Path(o.replace('\\', '/')).name == h5_path.name for o in e.get('outputs', []))]
    return mine[-1] if mine else None


def describe_colleague(item, md, attrs, shape, spacing, conv, verification, log, h5_name, rec):
    es, ad, rb = md.get('echo_start', {}), md.get('auto_detect', {}), md.get('axis_rebase', {})
    b, m = md.get('bounds', {}), md.get('mirror', {})
    L = [f"PAUT preprocessing record: {item.key}",
         "=" * 72,
         f"Sample (DB name)      : {item.sample}",
         f"Original id           : {item.original_id}   campaign: {item.campaign}   acquisition: {item.acquisition}",
         f"Equipment file (.fpd) : {md.get('source_file')}",
         f"Processed by          : the colleague's fpd batch tool (schema {attrs.get('schema')} v{attrs.get('schema_version')}, "
         f"processing_version {md.get('processing_version')}, cache_version {md.get('cache_version')})",
         f"                        config sha256 {md.get('config_sha256')}, source signature {md.get('source_signature')}",
         f"Colleague review      : " + (f"{log.get('status')} at {log.get('timestamp')}" if log else 'no review-log entry'),
         f"Imported to the NAS   : {rec['created']}   run {rec['run_id']}   code {rec['code_version']}",
         "",
         "Applied (by the colleague's tool, from its metadata)",
         "1. Echo start",
         f"   first sample with |a| >= {es.get('threshold_percent')} % inside {es.get('gate_mm')} mm "
         f"(method {es.get('method')}, start mode {es.get('start_mode')}); hits {es.get('hit_count')} of "
         f"{es.get('trace_count')} A-scans, interpolated {es.get('interpolated_count')} "
         f"(interpolate_missing={es.get('interpolate_missing')}); every A-scan shifted so the entrance echo is at 0 mm.",
         "2. Auto detect piece",
         f"   |a| >= {ad.get('threshold_percent')} % inside {ad.get('gate_mm')} mm below the echo, margin {ad.get('margin_mm')} mm; "
         f"status {ad.get('status')}, selected {ad.get('diagnostics', {}).get('selected_pixel_count')} A-scans.",
         "3. Crop",
         f"   bounds (samples, half-open) index {b.get('index')}, scan {b.get('scan')}, depth {b.get('depth')}; "
         f"final depth {md.get('final_z_mm')} mm relative to the entrance echo.",
         "   Axes rebased to " + str(rb.get('reference')) + ": " + ", ".join(
             f"{k} {v.get('source_start_mm', 0):.3f}..{v.get('source_stop_mm', 0):.3f} mm -> "
             f"{v.get('export_start_mm', 0):.3f}..{v.get('export_stop_mm', 0):.3f} mm"
             for k, v in rb.get('axes', {}).items()),
         "4. Mirror",
         f"   {m.get('horizontal_axis', 'index')} axis flipped: {m.get('horizontal')}, "
         f"{m.get('vertical_axis', 'scan')} axis flipped: {m.get('vertical')}",
         "",
         "Not applied / skipped",
         "   No reprocessing here: the colleague's processed volume is stored as it is (no TCG, filter, envelope or",
         "   moving average; not reviewed again in paut_review.py for now).",
         "",
         "Verification",
         f"   {verification}",
         "",
         "Stored (02_Processed)",
         f"   {item.sample}.tif: the colleague's 'amplitude' reordered (index, scan, depth) -> (Z=depth, Y=index, X=scan),",
         f"   shape ZYX ({shape[2]}, {shape[0]}, {shape[1]}), spacing ZYX ({spacing[2]:.4g}, {spacing[0]:.4g}, {spacing[1]:.4g}) mm, "
         f"float32 RF {attrs.get('amplitude_unit', '')}; values unchanged.  sha256 {rec['tif_sha256']}",
         f"   Source: the colleague's file {h5_name} (kept locally, not on the NAS).  sha256 {rec['source_h5_sha256']}",
         f"   Input: {item.rel('raw', '.tif').relative_to(Path(item.campaign) / item.acquisition)} sha256 {conv['tif_sha256']}",
         "",
         "Colleague metadata (verbatim):", json.dumps(md, indent=2)]
    return '\n'.join(L) + '\n'


def import_colleague_h5(cfg, item, run_id=None, verify=True):
    """Store the colleague's processed .h5 for ``item`` in 02_Processed (TIFF + verbatim .h5 + JSON + .txt)
    and record the decision as approved (source 'colleague'), so the review skips it.

    Returns 'imported', 'already imported' or 'no h5'.
    """
    import h5py
    h5 = colleague_h5(cfg, item)
    if h5 is None:
        return 'no h5'
    root = Path(cfg['output_root'])
    pj = root / item.rel('processed', '_processing.json')
    h5_sha = sha256(h5)
    if pj.exists():
        old = json.loads(pj.read_text())
        if old.get('source_h5_sha256') == h5_sha:
            return 'already imported'
        raise FileExistsError(f"{pj} exists and was not made from {h5.name}; not overwritten")
    run_id = run_id or new_run_id()
    upload_raw(cfg, item, run_id)
    with h5py.File(h5) as h:
        amp = h['amplitude'][()].astype(np.float32)
        axes = [np.asarray(h[k][()], np.float64) for k in ('index_mm', 'scan_mm', 'depth_mm')]
        md = json.loads(h['metadata_json'][()])
        attrs = {k: (v.item() if hasattr(v, 'item') else v) for k, v in h.attrs.items()}
    spacing = [float(np.median(np.diff(a))) if a.size > 1 else 0.0 for a in axes]
    verification = 'not checked'
    if verify:
        mir = md.get('mirror', {})
        params = merge_params(acquisition_entry(cfg, item.acq_key).get('params'),
                              {'mirror': {'index': bool(mir.get('horizontal')), 'scan': bool(mir.get('vertical'))}})
        out, _, _ = run_processing(cfg, item, params)
        verification = ("bit-identical to preprocess_tools.paut (echo start with the colleague's own functions, "
                        "default parameters, same mirroring) run on 01_Raw" if out.data.shape == amp.shape and
                        np.array_equal(out.data, amp) else
                        f"DIFFERS from preprocess_tools.paut on 01_Raw (ours {out.data.shape}, his {amp.shape})")
    conv = json.loads(item.local_conversion.read_text())
    log = _colleague_log(h5)
    h5_name = str(h5)   # not copied: the NAS holds only TIFF volumes
    tif = root / item.rel('processed', '.tif')
    if tif.exists():
        raise FileExistsError(f"{tif} exists; not overwritten")
    rec = {'created': _now(), 'run_id': run_id, 'code_version': code_version(), 'source': 'colleague',
           'sample': item.sample, 'original_id': item.original_id, 'campaign': item.campaign,
           'acquisition': item.acquisition, 'input_tif': str(item.rel('raw', '.tif')),
           'input_tif_sha256': conv['tif_sha256'], 'source_h5': str(h5),
           'source_h5_stored': 'locally only (not on the NAS)', 'source_h5_sha256': h5_sha, 'h5_attrs': attrs, 'colleague_metadata': md,
           'colleague_review_log': log, 'verification': verification,
           'output': {'shape_index_scan_depth': list(amp.shape), 'spacing_mm_index_scan_depth': spacing,
                      'stored_axis_order': ['depth', 'index', 'scan'], 'stored_axes': ['z', 'y', 'x']}}
    rec['tif_sha256'] = _write_new(tif, lambda t: write_tif(t, to_zyx(amp), (spacing[2], spacing[0], spacing[1])))
    decision = save_decision(cfg, item, 'approved', {}, 'imported from the colleague processed .h5')
    decision['source'] = 'colleague'
    review = load_review(cfg)
    review[item.key] = decision
    _write_json(Path(cfg['output_root']) / 'review.json', review)
    rec['review'] = decision
    _write_new(pj, lambda t: t.write_text(json.dumps(rec, indent=2, default=float)))
    _write_new(root / item.rel('processed', '_processing.txt'),
               lambda t: t.write_text(describe_colleague(item, md, attrs, amp.shape, spacing, conv,
                                                         verification, log, h5_name, rec)))
    append_manifest(cfg, item, run_id, 'processed (colleague h5)', item.rel('processed', '.tif'),
                    rec['tif_sha256'], str(h5))
    return 'imported'


# ----------------------------------------------------------------------------- existing-file check

def _depth_axis(item):
    conv = json.loads(item.local_conversion.read_text())
    d = conv['axes_mm']['depth']
    return np.asarray(d['values'], np.float64) if 'values' in d else d['start'] + np.arange(d['count']) * d['step']


def effective_transforms(params, depth_mm):
    """The transforms ``params`` actually apply, normalised for comparison (gates as the sample
    ranges they select on this depth axis, numbers rounded)."""
    from .processing import _gate_indices
    r = lambda x: round(float(x), 6)
    es, fp = params['echo_start'], params['footprint']
    eff = {'echo_start': {'enabled': bool(es['enabled'])}}
    if es['enabled']:
        eff['echo_start'].update({
            'gate_samples': list(_gate_indices(np.asarray(depth_mm), es['gate_mm'])),
            'threshold_pct': r(es['threshold_pct']), 'method': es.get('method', 'first'),
            'gain': r(es.get('gain', 1.0)), 'interpolate_missing': bool(es.get('interpolate_missing', True)),
            'neighbors': int(es.get('neighbors', 12)), 'signal': es.get('signal', 'abs')})
        if eff['echo_start']['signal'] == 'abs_peakhold_ma':
            eff['echo_start'].update({'hold_samples': int(es.get('hold_samples', 1)),
                                      'average_samples': int(es.get('average_samples', 1))})
    eff['footprint'] = {'enabled': bool(fp['enabled'])}
    if fp['enabled']:
        eff['footprint'].update({'gate_mm': [r(g) for g in fp['gate_mm']], 'threshold_pct': r(fp['threshold_pct']),
                                 'margin_mm': r(fp['margin_mm'])})
    if params.get('depth_mode', 'mm') == 'samples':
        eff['depth_crop'] = {'from_mm': r(params['depth_range_mm'][0]), 'samples': int(params['depth_samples'])}
    else:
        eff['depth_crop'] = {'range_mm': [r(x) for x in params['depth_range_mm']]}
    eff['mirror'] = {'index': bool(params['mirror'].get('index', False)),
                     'scan': bool(params['mirror'].get('scan', False))}
    return eff


def colleague_equivalent_params(md):
    """Our parameters equivalent to what the colleague's tool recorded in its metadata."""
    es, ad, m = md.get('echo_start', {}), md.get('auto_detect', {}), md.get('mirror', {})
    return merge_params({
        'echo_start': {'enabled': True, 'gate_mm': es.get('gate_mm', [-5, 5]),
                       'threshold_pct': es.get('threshold_percent', 30), 'method': es.get('method', 'first'),
                       'gain': 1.0, 'interpolate_missing': bool(es.get('interpolate_missing', True)), 'neighbors': 12,
                       'signal': 'abs'},
        'footprint': {'enabled': True, 'gate_mm': ad.get('gate_mm', [0, 6]),
                      'threshold_pct': ad.get('threshold_percent', 15), 'margin_mm': ad.get('margin_mm', 0.5)},
        'depth_range_mm': md.get('final_z_mm', [0, 6]),
        'mirror': {'index': bool(m.get('horizontal')), 'scan': bool(m.get('vertical'))}})


def _diff(a, b, prefix=''):
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += _diff(va, vb, f"{prefix}{k}.")
        elif va != vb:
            out.append((f"{prefix}{k}", va, vb))
    return out


def compare_with_saved(cfg, item, params):
    """Does 02_Processed already hold this item, made with exactly these transforms?

    Returns ``(status, differences)``: status 'missing', 'identical' or 'different';
    differences is a list of ``(what, saved, now)``.
    """
    pj = Path(cfg['output_root']) / item.rel('processed', '_processing.json')
    tif = Path(cfg['output_root']) / item.rel('processed', '.tif')
    if not (pj.exists() and tif.exists()):
        return 'missing', []
    rec = json.loads(pj.read_text())
    saved_params = (colleague_equivalent_params(rec['colleague_metadata']) if rec.get('source') == 'colleague'
                    else rec.get('params'))
    depth = _depth_axis(item)
    diffs = _diff(effective_transforms(merge_params(saved_params), depth), effective_transforms(params, depth))
    raw_sha = json.loads(item.local_conversion.read_text()).get('tif_sha256')
    if rec.get('input_tif_sha256') != raw_sha:
        diffs.append(('input 01_Raw TIFF (sha256)', rec.get('input_tif_sha256'), raw_sha))
    if rec.get('source') != 'colleague' and rec.get('processing_version') != PROCESSING_VERSION:
        diffs.append(('processing version', rec.get('processing_version'), PROCESSING_VERSION))
    return ('identical' if not diffs else 'different'), diffs


def describe_differences(diffs):
    names = {'echo_start.gate_samples': 'echo-start gate (samples)', 'depth_range_mm': 'depth crop (mm)'}
    return '\n'.join(f"  {names.get(w, w)}: saved {s} → now {n}" for w, s, n in diffs)
