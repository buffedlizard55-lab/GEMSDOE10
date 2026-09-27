#!/usr/bin/env python3
"""Exercise GeoTIFF generation with hash-verified OOF predictions, NEVER release.

This is a diagnostic of the format pipeline, not final-model inference and not
an approved competition artifact. A full-footprint top-k differs from per-fold
selection in the validation report; its score is deliberately not asserted.
"""
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from gems10 import cv, raster


def main():
    report = json.loads((ROOT/'reports/h12_blocked.json').read_text())
    if report.get('status') != 'completed' or len(report['folds']) != 4:
        raise ValueError('complete validation and persisted OOF hashes required')
    if report.get('decision', {}).get('eligible') is not False:
        raise ValueError('this diagnostic is only for the rejected H12 experiment')
    with rasterio.open(ROOT/'data/sample_submission.tif') as ds:
        fp = np.isfinite(ds.read(1))
    merged = np.full(fp.shape, np.nan, dtype='float32')
    count = np.zeros(fp.shape, dtype='uint8')
    folds = cv.make_folds(4, 4, 40)
    for fold in report['folds']:
        item = fold['prediction_files']['H12']
        path = (ROOT/item['file']).resolve()
        if not path.is_relative_to((ROOT/'scratch').resolve()):
            raise ValueError('OOF files must stay in scratch')
        if raster.sha256_file(path) != item['sha256']:
            raise ValueError(f'OOF hash mismatch: {path.name}; rerun, do not repair')
        p = np.load(path, allow_pickle=False)
        mask = folds.score_mask(fp.shape, fold['fold']) & fp
        if p.shape != fp.shape or not np.array_equal(np.isfinite(p), mask):
            raise ValueError('OOF mask differs from held-out geography')
        merged[mask] = p[mask]; count[mask] += 1
    if not np.all(count[fp] == 1) or not np.all(count[~fp] == 0):
        raise ValueError('overlapping or missing OOF support')
    destination = ROOT/'scratch/h12/oof_diagnostic.npy'
    stream = io.BytesIO(); np.save(stream, merged, allow_pickle=False)
    destination.write_bytes(stream.getvalue()); del stream
    subprocess.run([sys.executable, str(ROOT/'scripts/build_submission.py'), '--prob', str(destination),
                    '--policy', 'topk02_binary', '--name', 'gems10-h12-oof-rejected', '--experiment',
                    '--note', 'EXPERIMENT ONLY - DO NOT SUBMIT | H12 OOF | confirmation failed'], check=True)
    latest = sorted((ROOT/'scratch/experiments').glob('gems10-h12-oof-rejected-*.json'))[-1]
    metadata = json.loads(latest.read_text())
    raster.assert_submittable(latest.with_suffix('.tif'), ROOT/'data/sample_submission.tif')
    evidence = {'status': 'format_diagnostic_passed_not_approved',
                'artifact': metadata, 'validation_report_sha256': raster.sha256_file(ROOT/'reports/h12_blocked.json'),
                'notice': 'OOF-only diagnostic. No full-data final model. Do not submit. Not exposed as a site download.'}
    (ROOT/'reports/diagnostic_validation.json').write_text(json.dumps(evidence, indent=2)+'\n')


if __name__ == '__main__':
    main()
