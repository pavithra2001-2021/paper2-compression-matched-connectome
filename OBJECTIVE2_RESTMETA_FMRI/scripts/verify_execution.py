"""Final source preservation, completeness and saved-checkpoint replay audit."""
import json
import platform
import sys
from pathlib import Path
import joblib
import numpy as np
import scipy
import sklearn
import torch
from common import ROOT,RUN,CACHE,CONFIG,EVIDENCE,read,write,sha,people,cache_path
from representations import ARMS,fit_budget,make_operators,vectorize
from models import SpectralGNN,predict_gnn,predict_lr,DEVICE
from experiment import load_cache,operators

def main():
    cfg=read(CONFIG); roles=read(RUN/'participant_roles.json'); rows=people()
    assert sha(RUN/'participant_roles.json')==cfg['participant_roles_sha']
    for name,h in cfg['research_packet_hashes'].items(): assert sha(ROOT/name)==h,name
    prior=read(EVIDENCE/'inspection_summary.json')
    sources=None
    for key,val in prior.items():
        if isinstance(val,dict) and val and all(isinstance(v,dict) and 'sha256' in v for v in val.values()):
            sources=val
    assert sources,'Original input hashes unavailable'
    preserved={p:sha(p)==v['sha256'] for p,v in sources.items()}; assert all(preserved.values())
    cache=read(RUN/'spectral_cache_manifest.json'); assert cache['n_contexts']==2522
    for row in rows: assert sha(row['path'])==row['sha256']
    for entry in cache['files']: assert sha(CACHE/entry['file'])==entry['sha256']
    data=load_cache(rows); first=data['first190']; ranks=(first['w']>first['threshold'][:,None]).sum(axis=1)
    code={name:sha(ROOT/'scripts'/name) for name in ['experiment.py','models.py','representations.py']}
    replay=[]; mainfits=0; failed=[]
    for fold in roles['folds']:
        tr=np.array(fold['train']); te=np.array(fold['test']); budget=fit_budget(first['w'][tr],ranks[tr])
        for arm in ARMS:
            prefix=RUN/'main'/(fold['fold']+'_'+arm); meta=read(prefix.with_suffix('.json'))
            assert meta['training_code_hashes']==code and meta['config_sha']==sha(CONFIG)
            assert meta['budget']==budget and meta['training_audit']['weight_update_l2']>0
            assert meta['training_audit']['epochs'] in cfg['models']['epoch_grid']
            for j,inner in enumerate(fold['inner']):
                it=np.array(inner['train']); iv=np.array(inner['validation'])
                assert not set(it)&set(iv) and not set(it)&set(te) and not set(iv)&set(te)
                assert meta['inner'][j]['budget']==fit_budget(first['w'][it],ranks[it])
            state=torch.load(str(prefix)+'_gnn.pt',map_location=DEVICE,weights_only=False)
            gnn=SpectralGNN().to(DEVICE); gnn.load_state_dict(state['state_dict'])
            linear=joblib.load(str(prefix)+'_lr.joblib')
            assert set(state['train_indices'])==set(tr)==set(linear['train_indices'])
            saved=np.load(prefix.with_suffix('.npz')); assert sha(prefix.with_suffix('.npz'))==meta['predictions_sha']
            max_g=max_l=max_h=0.
            for context in ['first190','native230','last190']:
                mask=saved['context']==context; ix=saved['index'][mask]
                if not len(ix): continue
                assert set(ix)<=set(te) and len(ix)==len(set(ix))
                for i in ix:
                    assert int(saved['y'][mask][np.flatnonzero(ix==i)[0]])==roles['participants'][i]['y']
                op,kk,_=operators({k:v[ix] for k,v in data[context].items()},arm,budget)
                pg,h=predict_gnn(gnn,op); pl=predict_lr(linear['model'],linear['scaler'],op)
                max_g=max(max_g,float(np.max(np.abs(pg-saved['p_gnn'][mask])))); max_l=max(max_l,float(np.max(np.abs(pl-saved['p_lr'][mask]))))
                max_h=max(max_h,float(np.max(np.abs(h-saved['embedding'][mask]))))
                np.testing.assert_array_equal(kk,saved['rank'][mask])
                assert np.isfinite(pg).all() and np.isfinite(pl).all()
            assert max_g<2e-6 and max_l<2e-6 and max_h<2e-6,(prefix,max_g,max_l,max_h)
            saved.close(); mainfits+=1
            replay.append({'fold':fold['fold'],'arm':arm,'GNN_probability_max_error':max_g,'LR_probability_max_error':max_l,'embedding_max_error':max_h})
        print('replayed',fold['fold'],flush=True)
    assert mainfits==95
    permutations=0
    for perm in range(1,11):
        folder=RUN/f'permutations/p{perm:02d}'; perm_labels=read(folder/'labels.json')['labels']
        original=np.array([r['y'] for r in roles['participants']]); sites=np.array([r['site'] for r in roles['participants']])
        for s in set(sites): assert np.sum(np.array(perm_labels)[sites==s])==np.sum(original[sites==s])
        for j in range(5):
            for arm in ARMS[:4]:
                m=read(folder/f'mixed_{j}_{arm}.json')
                assert m['training_code_hashes']==code and m['config_sha']==sha(CONFIG)
                assert sha(folder/f'mixed_{j}_{arm}.npz')==m['predictions_sha']; permutations+=1
    assert permutations==200
    oldroot=ROOT.parent/'MDD_GNN_Q1'; locked=read(oldroot/'results/objective2_20261007/locked_discovery/lock.json')
    # Preserve the entire locked manifest and its direct artifacts against the prior recorded audit.
    assert sha(oldroot/'results/objective2_20261007/locked_discovery/lock.json')=='67bf6aae532e4d9d017169e27b9644c85b5535a3c429d19bc92730b079c843ad'
    for name,h in locked['artifact_hashes'].items():
        assert sha(oldroot/'results/objective2_20261007/locked_discovery'/name)==h
    package_hashes={'OBJECTIVE2_COMPLETE_PACKAGE_20261007.zip':'3f4c1603215070b24a1f50f6127ca953747cb2256e1ebbe134f313f7ed6ca62e',
        'OBJECTIVE2_CURRENT_RESULTS_20261012.zip':'c4d65ad1d0ae4032e4cde1379c863cc0bbbeac52216deaa2b5b3340faa3bd02d',
        'TDBRAIN_AVAILABLE_FILES_20261012.zip':'e4827789839a8124a9b38ce8bdc968fb46f8907c253e6a95bf55c5b07fe44c85'}
    for name,h in package_hashes.items(): assert sha(oldroot/'deliverables'/name)==h
    required=['performance.json','stability.json','paired_contrasts.json','metric_uncertainty.json','calibration_intervals.json','latent_associations.json','permutation_sanity.json','site_decoding.json','nuisance_controls.json']
    for name in required: assert (RUN/'final'/name).is_file(),name
    output={'passed':True,'date':'2026-10-13','config_sha':sha(CONFIG),'main_outer_representation_tasks':mainfits,
        'primary_trained_models':95,'logistic_models':95,'permutation_outer_representation_tasks':permutations,
        'participant_contexts':2522,'participant_N':1188,'primary_N':1150,'primary_C230_N':629,
        'source_files_unchanged':preserved,'all_prediction_source_signals_preserved':True,'all_spectral_cache_hashes_match':True,
        'old_EEG_lock_models_and_packages_preserved':True,'checkpoint_replay':replay,
        'software':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,
                    'platform':platform.platform(),'device':str(DEVICE),'GPU':torch.cuda.get_device_name() if torch.cuda.is_available() else None},
        'code_sha256':{p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},
        'statistical_scope':'conditional fitted-prediction estimates; no full-pipeline resampling; no algorithm superiority claim',
        'numerical_solver_event':'Initial liblinear dual LR attempt failed its convergence guard before any outer score; rerun with LBFGS, unchanged C grid; all retained outputs have the final solver hash'}
    write(RUN/'final/reproducibility_checks.json',output)
    manifest={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'config').glob('*'))+list((RUN/'final').rglob('*')) if p.is_file()}
    write(RUN/'final/artifact_manifest.json',manifest)
    print('ALL FINAL EXECUTION CHECKS PASSED')

if __name__=='__main__': main()
