"""Read-only source verification and resumable participant/segment feature cache."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from common import ROOT, RUN, CACHE, CONFIG, people, contexts, cache_path, sha, seed, write
from representations import fc_spectrum, phase_threshold

def prepare(person):
    with threadpool_limits(limits=1):
        if sha(person['path']) != person['sha256']:
            raise ValueError('Source hash mismatch: ' + person['subject_id'])
        x = loadmat(person['path'], variable_names=['ROISignals'])['ROISignals'][:, :116]
        assert x.shape == (person['T'], 116)
        records = []
        for context in contexts(person):
            path = cache_path(person, context)
            random_seed = seed(person['sha256'], context, 'AAL116', 'phase_B100_Q99')
            if path.exists():
                with np.load(path) as z:
                    assert str(z['source_sha']) == person['sha256'] and int(z['seed']) == random_seed
                    assert str(z['constructor_sha']) == sha(ROOT / 'scripts/representations.py')
                records.append({'file': path.name, 'sha256': sha(path)})
                continue
            segment = x[:190] if context == 'first190' else x[-190:] if context == 'last190' else x
            c, w, u = fc_spectrum(segment)
            threshold = phase_threshold(segment, random_seed)
            assert np.allclose(c, (u * w[None, :]) @ u.T, atol=1e-10)
            assert np.isclose(w.sum(), 116)
            np.savez_compressed(path, w=w, u=u, threshold=threshold, T=len(segment),
                                rank=int((w > threshold).sum()), seed=random_seed,
                                source_sha=person['sha256'], constructor_sha=sha(ROOT / 'scripts/representations.py'))
            records.append({'file': path.name, 'sha256': sha(path)})
        return records

def main():
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=6); args = p.parse_args()
    assert CONFIG.is_file(), 'Freeze protocol before features'
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = people(); start = time.time(); output = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(prepare, row) for row in rows]
        for i, f in enumerate(as_completed(futures), 1):
            output += f.result()
            if i % 50 == 0 or i == len(rows):
                print(f'spectra {i}/{len(rows)}; {time.time()-start:.1f}s', flush=True)
    write(RUN / 'spectral_cache_manifest.json', {'config_sha': sha(CONFIG), 'n_subjects': len(rows),
          'n_contexts': len(output), 'files': sorted(output, key=lambda r:r['file']), 'elapsed_seconds': time.time()-start})

if __name__ == '__main__': main()
