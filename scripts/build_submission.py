#!/usr/bin/env python3
"""Stage, validate and publish an artifact only after scientific + format gates.

No upload occurs. --experiment writes only to scratch/experiments. It cannot
make a failed hypothesis appear in the website's approved download manifest.
"""
from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gems10 import modeling, novelty, raster, release, spec
from gems10.raster import sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prob', required=True)
    ap.add_argument('--policy', default='topk02_binary')
    ap.add_argument('--name', default='gems10-submission')
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'docs/downloads')
    ap.add_argument('--note')
    ap.add_argument('--validation', type=Path, help='artifact-bound spatial validation JSON in reports/')
    ap.add_argument('--experiment', action='store_true', help='diagnostic only, never published')
    args = ap.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.name):
        raise ValueError('name must contain only letters, digits, hyphen, underscore')
    template, labels = args.data_dir / 'sample_submission.tif', args.data_dir / 'labels.tif'
    with rasterio.open(template) as ds:
        fp = np.isfinite(ds.read(1))
    prob = np.load(args.prob, allow_pickle=False)
    if prob.shape != (spec.HEIGHT, spec.WIDTH):
        raise ValueError(f'wrong probability grid: {prob.shape}')
    if not np.isfinite(prob[fp]).all() or np.any((prob[fp] < 0) | (prob[fp] > 1)):
        raise ValueError('raw predictions must be finite in [0,1] inside footprint; no silent repair')
    prob_sha = sha256_file(args.prob)
    validation = None
    if not args.experiment:
        if args.validation is None:
            raise ValueError('REFUSED: spatial holdout evidence required; --experiment is diagnostic only')
        if args.validation.resolve().parent != (ROOT / 'reports').resolve():
            raise ValueError('release evidence must be in reports/')
        validation = json.loads(args.validation.read_text())
        release.check_release(validation, prob_sha, args.policy)
        release.verify_training_binding(validation, ROOT)
    registry_path = ROOT / 'reports/submission_audit.json'
    if not registry_path.exists():
        raise ValueError('missing sibling audit; run audit_submissions.py')
    registry = json.loads(registry_path.read_text())['artifacts']
    if not registry or any(e.get('status') != 'measured' for e in registry):
        raise ValueError('incomplete sibling audit; refresh before building')
    output = ROOT / 'scratch/experiments' if args.experiment else args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    for folder in {output, ROOT / 'docs/downloads', ROOT / 'scratch/experiments'}:
        for previous in folder.glob('*.json'):
            registry.append(json.loads(previous.read_text()))
    emission = modeling.apply_policy(prob, fp, args.policy)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    filename = f'{args.name}-{stamp}-{prob_sha[:10]}.tif'
    # No incomplete or rejected TIF is exposed in downloads. Sidecar is moved last.
    with tempfile.TemporaryDirectory(dir=output, prefix='.staging-') as tmp:
        tif = Path(tmp) / filename
        raster.write_submission(emission, tif, template, footprint=fp)
        gate = raster.assert_submittable(tif, template)
        identity = novelty.raster_identity(tif, template, labels)
        novelty.refuse_duplicate(identity, registry)
        sha = sha256_file(tif)
        zpath = tif.with_suffix('.zip')
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.write(tif, arcname=tif.name)
        note = args.note or f'{args.name} | {args.policy} | sha {sha[:10]}'
        if args.experiment and not note.startswith('EXPERIMENT'):
            note = 'EXPERIMENT - DO NOT SUBMIT | ' + note
        meta = {**identity, 'name': args.name, 'policy': args.policy, 'generated_utc': stamp,
                'status': 'experimental_do_not_submit' if args.experiment else 'approved',
                'probability_sha256': prob_sha, 'file': tif.name, 'sha256': sha,
                'bytes': tif.stat().st_size, 'zip': zpath.name, 'zip_sha256': sha256_file(zpath),
                'validation_sha256': sha256_file(args.validation) if validation else None,
                'validation_file': args.validation.name if validation else None,
                'suggested_note': note, 'footprint_px': int(fp.sum()),
                'positive_px': int((emission[fp] > 0).sum()), 'mass': float(emission[fp].sum()),
                'gate': [{'name': c.name, 'ok': bool(c.ok), 'detail': str(c.detail),
                          'hard': bool(c.hard)} for c in gate.checks]}
        metadata = tif.with_suffix('.json')
        metadata.write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
        for p in (tif, zpath, metadata):
            if (output / p.name).exists():
                raise ValueError('refusing to overwrite an existing artifact')
        for p in (tif, zpath, metadata):
            p.replace(output / p.name)
    print(json.dumps({k: v for k, v in meta.items() if k != 'gate'}, indent=2))
    print('FORMAT PASSED; ' + ('EXPERIMENT ONLY — DO NOT SUBMIT' if args.experiment else 'APPROVED FOR DOWNLOAD'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
