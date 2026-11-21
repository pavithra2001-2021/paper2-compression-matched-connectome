"""Bounded supplementary sensitivity; never writes into the frozen study.

Run with the original run.ps1 launcher and this script's relative path:
  scripts/run.ps1 ../supplementary/focused_strengthening_20261026/run_sensitivity.py prepare
  scripts/run.ps1 ../supplementary/focused_strengthening_20261026/run_sensitivity.py run
  scripts/run.ps1 ../supplementary/focused_strengthening_20261026/run_sensitivity.py analyze
The protocol is locked before training. Analyze replays saved checkpoints, not fits.
"""
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from common import RUN, CONFIG, read, write, sha, seed, people, cache_path
from representations import make_operators, fit_budget
from models import DEVICE, SpectralGNN, train_gnn, predict_gnn

ARMS = ('temporal', 'budget')
PROTOCOL = HERE / 'protocol.json'
AUDIT = ROOT / 'results/posthoc_graph_audit_20261013'


def original_folds():
    roles = read(RUN / 'participant_roles.json')
    folds = [f for f in roles['folds'] if f['scheme'] == 'loso' and f['fold'] != 'loso_S12']
    assert len(folds) == 13
    ix = [i for f in folds for i in f['test']]
    assert len(ix) == len(set(ix)) == 1150
    assert {roles['participants'][i]['site'] for i in ix}.isdisjoint({'S4', 'S12'})
    assert sum(roles['participants'][i]['site'] == 'S14' for i in ix) == 96
    return roles, folds


def recorded_integrity():
    manifests = [RUN / 'final/artifact_manifest.json', AUDIT / 'artifact_manifest.json',
                 ROOT / 'results/posthoc_spectral_v1_20261013/final/artifact_manifest.json']
    total = 0
    for m in manifests:
        for name, digest in read(m).items():
            assert sha(ROOT / name) == digest, name
            total += 1
    lock = read(ROOT / 'config/posthoc_graph_audit_20261013.json')['primary_unchanged']
    for name, digest in lock.items():
        assert sha(ROOT / name) == digest, name
    return {'manifest_entries_verified': total, 'primary_lock_entries_verified': len(lock)}


def check_protected(spec):
    for name, digest in spec['protected_files'].items():
        assert sha(ROOT / name) == digest, ('protected file changed', name)
    assert sha(__file__) == spec['runner_sha256'], 'Runner changed after protocol lock'


def prepare():
    assert not PROTOCOL.exists(), 'Existing protocol must not be overwritten'
    integrity = recorded_integrity()
    roles, folds = original_folds()
    tasks = []
    for f in folds:
        paired_seeds = []
        for arm in ARMS:
            meta = read(RUN / 'main' / (f['fold'] + '_' + arm + '.json'))
            paired_seeds.append(meta['training_audit']['seed'])
            for rep in (1, 2):
                tasks.append({'fold': f['fold'], 'arm': arm, 'replicate': rep,
                              'seed': seed('focused_strengthening_20261026', f['fold'], rep),
                              'selected_gnn': meta['selected_gnn'], 'budget': meta['budget'],
                              'original_seed': meta['training_audit']['seed']})
        assert len(set(paired_seeds)) == 1
    protected = {str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*')
                 if p.is_file() and not p.is_relative_to(HERE)}
    assert len(tasks) == 52
    write(PROTOCOL, {
        'status': 'LOCKED_BEFORE_ADDITIONAL_TRAINING', 'date': '2026-10-26',
        'design': 'Supplementary fixed-configuration initialization/dropout sensitivity; original nested selections reused.',
        'scope': '13 original primary held cohorts, two original representations, original plus two additional seeds.',
        'tasks': tasks, 'new_fits_maximum': 52, 'new_training_seconds_maximum': 1800,
        'new_epochs_total': sum(t['selected_gnn'][2] for t in tasks),
        'feasibility': 'Original complete nested model-task completion gaps: median 28.44 s, 90th percentile 39.58 s. Single outer fits are cheaper; conservative 52*40 s proxy is ~35 minutes, while nested tasks contain six inner training trajectories plus outer fitting and logistic fits. Expected outer-only time is minutes; strict training cap 30 minutes.',
        'seed_generation': "common.seed('focused_strengthening_20261026', fold, replicate), replicate 1 or 2; same integer for both arms within fold.",
        'selection_uncertainty': 'Not measured; original seed also informed original inner selection. No seed, epoch, cohort or configuration selected using new outcomes.',
        'splits': 'Original fixed participant roles, including S12 in training where originally included; no S12 test fold.',
        'metrics': {'performance': 'Equal-cohort macro AUC and Brier on first190; same participants paired by index/fold/replicate.',
                    'stability': 'Participant mean absolute native230-first190 probability change in 629 people; last190-first190 secondary.',
                    'uncertainty': 'Descriptive three-seed values/ranges/SD only. No seed p-values, bootstrap or algorithm-level CIs.',
                    'collapse': 'Report exact constant probabilities, probability SD/range and all-same hard decisions separately. Near-constant flag uses probability range <=1e-6; no exclusion based on it.'},
        'work_package_B': 'Reuse existing saved operator relative Frobenius changes, ranks, original predictions and fixed-model representation swaps; summarize identical/non-identical and cohort strata without subgroup predictive claims or p-values.',
        'B_distance': '||S_budget-S_temporal||_F / ||S_temporal||_F, from existing diagnostics; report denominator and reverse orientation separately where useful.',
        'B_associations': 'Descriptive Spearman magnitude association for all/non-identical and each cohort. No causal inference; separately trained-model gap differs from same-model swap sensitivity.',
        'stop_rule': '52 fits only, no replacement seeds or tuning; stop on integrity/leakage/numerical failure or training time cap. No new architecture/data/thresholds.',
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'device': str(DEVICE), 'gpu': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else None},
        'preflight_integrity': integrity, 'protected_files': protected,
        'runner_sha256': sha(__file__), 'primary_config_sha256': sha(CONFIG),
        'outputs': ['protocol.json', 'checkpoints/*.pt', 'participant_predictions.csv', 'overlap_diagnostics.csv', 'final_evidence.json']})
    print('LOCKED', len(tasks), 'new fits;', len(protected), 'protected files;', sum(t['selected_gnn'][2] for t in tasks), 'epochs', flush=True)


def cache_data():
    rows = people()
    roles, _ = original_folds()
    assert [r['subject_id'] for r in rows] == [r['id'] for r in roles['participants']]
    first = {k: [] for k in ('w', 'u', 'threshold', 'T')}
    for row in rows:
        with np.load(cache_path(row, 'first190')) as z:
            for k in first: first[k].append(z[k].copy())
    return rows, {k: np.asarray(v) for k, v in first.items()}


def segment_data(rows, ix, context):
    out = {k: [] for k in ('w', 'u', 'threshold', 'T')}
    for i in ix:
        with np.load(cache_path(rows[i], context)) as z:
            for k in out: out[k].append(z[k].copy())
    return {k: np.asarray(v) for k, v in out.items()}


def evaluate_contexts(model, rows, first, fold, arm, budget):
    ix = np.asarray(fold['test'])
    result = {k: [] for k in ('index', 'context', 'y', 'p')}
    for context in ('first190', 'native230', 'last190'):
        sub = ix if context == 'first190' else ix[[rows[i]['T'] == 230 for i in ix]]
        if not len(sub): continue
        data = {k: v[sub] for k, v in first.items()} if context == 'first190' else segment_data(rows, sub, context)
        s = make_operators(**data, arm=arm, budget=budget)[0]
        p = predict_gnn(model, s)[0]
        result['index'].append(sub); result['context'].append(np.full(len(sub), context))
        result['y'].append(np.array([rows[i]['diagnosis'] == 'MDD' for i in sub], int))
        result['p'].append(p)
    return {k: np.concatenate(v) for k, v in result.items()}


def run():
    spec = read(PROTOCOL); check_protected(spec)
    rows, first = cache_data(); roles, folds = original_folds()
    byfold = {f['fold']: f for f in folds}
    y = np.array([p['y'] for p in roles['participants']])
    ranks = (first['w'] > first['threshold'][:, None]).sum(1)
    (HERE / 'checkpoints').mkdir(exist_ok=True)
    elapsed = 0.0
    for t in spec['tasks']:
        path = HERE / 'checkpoints' / f"{t['fold']}_{t['arm']}_seed{t['replicate']}.pt"
        if path.exists():
            old = torch.load(path, weights_only=False, map_location='cpu')
            assert old['task'] == t and old['protocol_sha256'] == sha(PROTOCOL)
            elapsed += old['training_seconds']; continue
        assert elapsed < spec['new_training_seconds_maximum'], 'Training budget exhausted; do not select replacement cohorts'
        fold = byfold[t['fold']]; tr = np.asarray(fold['train']); te = np.asarray(fold['test'])
        assert not set(tr) & set(te)
        assert {rows[i]['site'] for i in tr}.isdisjoint({rows[i]['site'] for i in te})
        assert fit_budget(first['w'][tr], ranks[tr]) == t['budget']
        s = make_operators(**first, arm=t['arm'], budget=t['budget'])[0]
        if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        start = time.monotonic()
        model, _, audit = train_gnn(s[tr], y[tr], t['seed'], t['selected_gnn'][1], t['selected_gnn'][2])
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        seconds = time.monotonic() - start; elapsed += seconds
        preds = evaluate_contexts(model, rows, first, fold, t['arm'], t['budget'])
        assert np.isfinite(preds['p']).all()
        checkpoint = {'task': t, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                      'predictions': preds, 'training_audit': audit, 'training_seconds': seconds,
                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if DEVICE.type == 'cuda' else None,
                      'protocol_sha256': sha(PROTOCOL), 'train_indices': tr, 'test_indices': te}
        torch.save(checkpoint, path)
        print('FIT', t['fold'], t['arm'], t['replicate'], f'{seconds:.2f}s; cumulative {elapsed:.1f}s', flush=True)
        del model, s, checkpoint; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    check_protected(spec)
    print('ALL 52 ADDITIONAL FITS COMPLETE', flush=True)


def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return {'N': 0}
    return {'N': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)), 'SD': float(x.std()),
            'min': float(x.min()), 'q05': float(np.quantile(x, .05)), 'q95': float(np.quantile(x, .95)), 'max': float(x.max())}


def rho(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0: return None
    r = float(spearmanr(a, b).statistic)
    return r if np.isfinite(r) else None


def save_csv(path, records):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)


def collect_predictions(rows, first, folds):
    records, fits = [], []
    maximum_error = 0.0
    spec = read(PROTOCOL)
    for fold in folds:
      for arm in ARMS:
        name = fold['fold'] + '_' + arm
        meta = read(RUN / 'main' / (name + '.json'))
        for rep in (0, 1, 2):
            if rep == 0:
                ck = torch.load(RUN / 'main' / (name + '_gnn.pt'), map_location='cpu', weights_only=False)
                with np.load(RUN / 'main' / (name + '.npz')) as z:
                    preds = {k: z[k].copy() for k in ('index', 'context', 'y')}; preds['p'] = z['p_gnn'].copy()
                random_seed = meta['training_audit']['seed']
            else:
                ck = torch.load(HERE / 'checkpoints' / f'{name}_seed{rep}.pt', map_location='cpu', weights_only=False)
                assert ck['protocol_sha256'] == sha(PROTOCOL)
                assert ck['task'] in spec['tasks']
                np.testing.assert_array_equal(ck['train_indices'], fold['train'])
                preds = ck['predictions']; random_seed = ck['task']['seed']
                fits.append({'fold': fold['fold'], 'arm': arm, 'replicate': rep,
                             'seed': random_seed, 'training_seconds': ck['training_seconds'],
                             'peak_cuda_allocated_bytes': ck['peak_cuda_allocated_bytes'], 'training_audit': ck['training_audit']})
            model = SpectralGNN().to(DEVICE); model.load_state_dict(ck['state_dict'])
            replay = evaluate_contexts(model, rows, first, fold, arm, meta['budget'])
            for k in ('index', 'context', 'y'): np.testing.assert_array_equal(preds[k], replay[k])
            error = float(np.max(np.abs(preds['p'] - replay['p']))); maximum_error = max(maximum_error, error)
            assert error <= 2e-6
            for i, context, label, p in zip(preds['index'], preds['context'], preds['y'], preds['p']):
                assert label == (rows[i]['diagnosis'] == 'MDD')
                records.append({'index': int(i), 'id': rows[i]['subject_id'], 'site': rows[i]['site'],
                                'fold': fold['fold'], 'arm': arm, 'replicate': rep, 'seed': random_seed,
                                'context': str(context), 'y': int(label), 'p': float(p)})
            del model
    assert len(fits) == 52
    return records, fits, maximum_error


def summarize_A(records):
    cohort, summaries, contrasts = [], [], []
    lookup = {(r['replicate'], r['arm'], r['index'], r['context']): r for r in records}
    assert len(lookup) == len(records)
    for rep in (0, 1, 2):
      for arm in ARMS:
        first = [r for r in records if r['replicate'] == rep and r['arm'] == arm and r['context'] == 'first190']
        assert len(first) == 1150
        cs = []
        for site in sorted({r['site'] for r in first}):
            rr = [r for r in first if r['site'] == site]
            y = np.array([r['y'] for r in rr]); p = np.array([r['p'] for r in rr])
            d = {'replicate': rep, 'arm': arm, 'site': site, 'N': len(rr), 'auc': float(roc_auc_score(y, p)),
                 'brier': float(np.mean((y-p)**2)), 'probability_sd': float(p.std()), 'probability_range': float(np.ptp(p)),
                 'exact_constant': bool(np.ptp(p) == 0), 'near_constant_range_le_1e_6': bool(np.ptp(p) <= 1e-6),
                 'single_hard_class': bool(len(np.unique(p >= .5)) == 1)}
            for context in ('native230', 'last190'):
                pairs = [(r, lookup[(rep, arm, r['index'], context)]) for r in rr if (rep, arm, r['index'], context) in lookup]
                d[context+'_MAD'] = float(np.mean([abs(a['p']-b['p']) for a,b in pairs])) if pairs else None
            cs.append(d); cohort.append(d)
        s = {'replicate': rep, 'arm': arm, 'N': 1150, 'cohorts': 13,
             'macro_auc': float(np.mean([r['auc'] for r in cs])), 'macro_brier': float(np.mean([r['brier'] for r in cs])),
             'constant_cohorts': [r['site'] for r in cs if r['exact_constant']],
             'near_constant_cohorts': [r['site'] for r in cs if r['near_constant_range_le_1e_6']],
             'single_hard_class_cohorts': [r['site'] for r in cs if r['single_hard_class']]}
        for context in ('native230', 'last190'):
            pairs = [(r, lookup[(rep, arm, r['index'], context)]) for r in first if (rep, arm, r['index'], context) in lookup]
            assert len(pairs) == 629 and len({a['site'] for a,b in pairs}) == 6
            s[context+'_MAD'] = float(np.mean([abs(a['p']-b['p']) for a,b in pairs]))
        summaries.append(s)
      a,b = summaries[-2:]
      contrasts.append({'replicate': rep, **{k: a[k]-b[k] for k in ('macro_auc','macro_brier','native230_MAD','last190_MAD')}})
    variability = {}
    for arm in ARMS:
        ix = sorted({r['index'] for r in records if r['context']=='first190'})
        p = np.array([[lookup[(rep,arm,i,'first190')]['p'] for i in ix] for rep in (0,1,2)])
        variability[arm] = {'participant_probability_range_across_seeds': stats(np.ptp(p,axis=0)),
                            'participant_probability_SD_across_seeds': stats(p.std(axis=0)),
                            'pairwise_prediction_spearman': {f'{a}_{b}':rho(p[a],p[b]) for a,b in ((0,1),(0,2),(1,2))},
                            'summary_metrics_across_seeds': {k:stats([s[k] for s in summaries if s['arm']==arm]) for k in ('macro_auc','macro_brier','native230_MAD','last190_MAD')}}
    for c in cohort:
        peer = next(r for r in cohort if r['site']==c['site'] and r['replicate']==c['replicate'] and r['arm']!=c['arm'])
        c['brier_difference_vs_other_arm'] = c['brier']-peer['brier']
    original = contrasts[0]
    np.testing.assert_allclose(original['macro_brier'], -0.0005277419, atol=1e-9)
    np.testing.assert_allclose(original['native230_MAD'], 0.0000445981, atol=1e-9)
    return {'summaries':summaries,'temporal_minus_budget':contrasts,'cohort_metrics':cohort,'seed_variability':variability}


def summarize_B(folds, roles):
    records = []
    for fold in folds:
        with np.load(AUDIT/'checkpoints'/(fold['fold']+'_temporal.npz')) as a, np.load(AUDIT/'checkpoints'/(fold['fold']+'_budget.npz')) as b:
            np.testing.assert_array_equal(a['index'],b['index']); np.testing.assert_array_equal(a['index'],fold['test'])
            for j,i in enumerate(a['index']):
                rel = float(a['operator_relative_change_budget'][j]); identical = bool(a['rank'][j]==b['rank'][j])
                assert identical == (rel==0)
                records.append({'index':int(i),'id':roles['participants'][i]['id'],'site':str(a['site'][j]),
                    'identical_operator':identical, 'temporal_rank':int(a['rank'][j]),'budget_rank':int(b['rank'][j]),
                    'relative_operator_Frobenius_change':rel,
                    'between_trained_models_abs_probability_gap':float(abs(a['p_original'][j]-b['p_original'][j])),
                    'fixed_temporal_model_swap_abs_probability_change':float(abs(a['p_swap_budget'][j]-a['p_original'][j])),
                    'fixed_budget_model_swap_abs_probability_change':float(abs(b['p_swap_temporal'][j]-b['p_original'][j])),
                    'temporal_pooling_retained_difference_energy':float(a['pooling_retained_difference_energy_budget'][j])})
    assert len(records)==1150 and sum(r['identical_operator'] for r in records)==710
    def group(rr):
        d=np.array([r['relative_operator_Frobenius_change'] for r in rr])
        result={'N':len(rr),'identical_N':sum(r['identical_operator'] for r in rr)}
        for key in ('relative_operator_Frobenius_change','between_trained_models_abs_probability_gap',
                    'fixed_temporal_model_swap_abs_probability_change','fixed_budget_model_swap_abs_probability_change',
                    'temporal_pooling_retained_difference_energy'):
            x=np.array([r[key] for r in rr]); result[key]=stats(x)
            if 'probability' in key: result[key+'_spearman_with_operator_distance']=rho(d,x)
        return result
    out={'all':group(records),'identical':group([r for r in records if r['identical_operator']]),
         'nonidentical':group([r for r in records if not r['identical_operator']]),
         'cohorts':{s:{'all':group([r for r in records if r['site']==s]),
                       'nonidentical':group([r for r in records if r['site']==s and not r['identical_operator']])}
                    for s in sorted({r['site'] for r in records})},
         'interpretation':'Existing fixed-model swaps isolate an input change computationally for that fitted model. Between-model gaps also include different learned parameters/hyperparameters. Neither is causal evidence about diagnostic usefulness. No new primary subgroup or hypothesis test.'}
    return out, records


def analyze():
    spec=read(PROTOCOL); check_protected(spec)
    roles,folds=original_folds(); rows,first=cache_data()
    records,fits,replay_error=collect_predictions(rows,first,folds)
    a=summarize_A(records); b,diagnostics=summarize_B(folds,roles)
    save_csv(HERE/'participant_predictions.csv',records)
    save_csv(HERE/'overlap_diagnostics.csv',diagnostics)
    integrity=recorded_integrity(); check_protected(spec)
    outputs={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='final_evidence.json'}
    final={'status':'COMPLETE_SUPPLEMENTARY_FIXED_CONFIGURATION_SENSITIVITY_AND_POSTHOC_DIAGNOSTICS',
           'original_primary_study_unchanged':True,'original_primary_N':1150,'original_eligible_N':1188,
           'original_fits_reused':26,'additional_fits':len(fits),'total_conditions':78,'seeds_per_condition':3,
           'A':a,'B':b,'new_fit_metadata':fits,
           'verification':{'passed':True,'protected_files_unchanged':len(spec['protected_files']),**integrity,
                           'replayed_checkpoints':78,'maximum_probability_replay_error':replay_error,
                           'original_primary_contrast_reproduced':True,'participant_prediction_rows':len(records),
                           'new_training_seconds':sum(f['training_seconds'] for f in fits),
                           'additional_seed_retraining_performed_for_verification':False,
                           'reproducibility_scope':'All 78 checkpoints replayed; deterministic PyTorch settings and exact seeds/configurations recorded. No duplicate training fits claimed.',
                           'new_figures_created':0,'existing_figures_verified_by_frozen_hashes':True},
           'limitations':['Three seeds only; original hyperparameters reused, with original-seed selection coupling.',
                          'Initialization and dropout jointly vary; their separate contributions are not identified.',
                          'No split, selection, data-sampling or full-learning-procedure uncertainty estimated.',
                          'No new confidence intervals/p-values; no treating seeds as new participants.',
                          'No independent external MDD validation; derivative segments overlap and share full-run preprocessing.',
                          'Post-hoc diagnostics and earlier Chebyshev results remain separate from frozen primary evidence.'],
           'artifact_sha256':outputs,
           'reproduction':'Use original scripts/run.ps1 with ../supplementary/focused_strengthening_20261026/run_sensitivity.py analyze for saved-checkpoint/metric verification. prepare refuses an existing lock; run resumes only matching locked tasks. No original module is changed.'}
    final_path=HERE/'final_evidence.json'
    if final_path.exists():
        assert read(final_path)==json.loads(json.dumps(final,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))), 'Final evidence changed'
        print('FINAL EVIDENCE EXACTLY REPRODUCED',flush=True)
    else: write(final_path,final)
    print(json.dumps({'A':a['summaries'],'contrasts':a['temporal_minus_budget'],'verification':final['verification']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['prepare','run','analyze'])
    args=parser.parse_args()
    with threadpool_limits(limits=2): globals()[args.stage]()
