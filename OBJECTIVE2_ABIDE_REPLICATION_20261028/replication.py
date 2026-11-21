"""Single bounded ABIDE replication. Imports frozen REST algorithms read-only.

Stages: freeze, cache, train, analyze, verify. No source-data writes, searches,
architecture changes, external MDD scoring, or performance-driven extensions.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import numpy as np
import joblib
import torch
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REST = HERE.parent / 'OBJECTIVE2_RESTMETA_FMRI'
SOURCE = Path(os.environ.get('ABIDE_ROOT', str(HERE.parent / 'data' / 'ABIDE_pcp')))
sys.path.insert(0, str(REST / 'scripts'))
import common as original
import representations as rep
import models
import experiment as exp
from metrics import metrics, reliability, calibration, rho, paired_interval
from metric_uncertainty import bootstrap_site, weighted_auc

CFG = HERE / 'protocol.json'
ROLES = HERE / 'participant_roles.json'
CACHE = HERE / 'spectra'
FINAL = HERE / 'final'
read, write, sha = original.read, original.write, original.sha
ARMS = rep.ARMS

def seed(*parts):
    return original.seed('ABIDE_REPLICATION_20261028', *parts)

def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def center(site):
    return site.split('_')[0] if site.startswith(('LEUVEN_', 'UM_', 'UCLA_')) else site

def freeze():
    assert not CFG.exists(), 'A frozen protocol must never be overwritten'
    assert not (HERE/'main').exists() and not CACHE.exists()
    meta_path = SOURCE/'Phenotypic_V1_0b_preprocessed1.csv'
    with meta_path.open(encoding='utf-8-sig', newline='') as f:
        table = list(csv.DictReader(f))
    lookup = {r['FILE_ID']:r for r in table if r['FILE_ID'] != 'no_filename'}
    qc = {'qc_rater_1':['OK'], 'qc_anat_rater_2':['OK','maybe'], 'qc_func_rater_2':['OK','maybe'],
          'qc_anat_rater_3':['OK'], 'qc_func_rater_3':['OK']}
    records = []; headers = set(); seen = set(); signals = set()
    for p in sorted((SOURCE/'dparsf/filt_noglobal').glob('*.1D')):
        r = lookup[p.name.removesuffix('_rois_aal.1D')]
        assert r['SUB_ID'] not in seen; seen.add(r['SUB_ID'])
        with p.open() as f: header = f.readline().strip()
        headers.add(header)
        x = np.loadtxt(p); good = x.ndim == 2 and x.shape[1] == 116 and np.isfinite(x).all()
        if good:
            digest = hashlib.sha256(x.tobytes()).hexdigest()
            assert digest not in signals; signals.add(digest)
        constant = int((x.std(0, ddof=1) <= 1e-10).sum()) if good else None
        reasons=[]
        if not all(r[k] in v for k,v in qc.items()): reasons.append('provider_QC')
        if r['DX_GROUP'] not in ['1','2']: reasons.append('diagnosis')
        if not good or constant: reasons.append('incomplete_ROI')
        if len(x)<190: reasons.append('fewer_than_190_retained_volumes')
        if good and len(x)>=190 and np.any(x[:190].std(0,ddof=1)<=1e-10): reasons.append('constant_first190')
        row = {'id':r['SUB_ID'], 'subject_id':r['SUB_ID'], 'site_code':r['SITE_ID'], 'site':center(r['SITE_ID']),
               'y':int(r['DX_GROUP']=='1'), 'diagnosis':'ASD' if r['DX_GROUP']=='1' else 'control',
               'age':float(r['AGE_AT_SCAN']), 'sex_code':int(r['SEX']), 'mean_fd':float(r['func_mean_fd']),
               'source_T':len(x), 'path':str(p), 'sha256':sha(p), 'constant_regions':constant,
               'qc':{k:r[k] for k in qc}, 'exclusion_reasons':reasons}
        assert row['age']>0 and row['sex_code'] in [1,2] and np.isfinite(row['mean_fd']) and row['mean_fd']>=0
        row['finite230'] = not reasons and len(x)>=230 and np.all(x[:230].std(0,ddof=1)>1e-10) and np.all(x[40:230].std(0,ddof=1)>1e-10)
        row['finite230'] = bool(row['finite230'])
        records.append(row)
    assert len(records)==871 and len(headers)==1
    initial = [r for r in records if not r['exclusion_reasons']]
    assert len(initial)==399
    support = sorted(s for s in {r['site'] for r in initial} if sum(r['site']==s for r in initial)>=30 and
                     all(sum(r['site']==s and r['y']==d for r in initial)>=10 for d in [0,1]))
    assert support == ['LEUVEN','PITT','UM','USM','YALE']
    for r in records:
        if not r['exclusion_reasons'] and r['site'] not in support: r['exclusion_reasons'].append('unsupported_center')
    people = sorted([r for r in records if not r['exclusion_reasons']], key=lambda r:r['id'])
    assert len(people)==323 and sum(r['y'] for r in people)==156 and sum(r['finite230'] for r in people)==243
    for i,r in enumerate(people):
        r['index']=i
        # Adapter flag for frozen experiment replay; source length remains source_T.
        r['T']=230 if r['finite230'] else 190
    groups=np.array([r['site'] for r in people]); y=np.array([r['y'] for r in people])
    folds=[]
    for s in support:
        tr=np.flatnonzero(groups!=s); te=np.flatnonzero(groups==s); inner=[]
        for j,(a,b) in enumerate(GroupKFold(3).split(tr, y[tr], groups[tr])):
            it=tr[a]; iv=tr[b]
            assert not set(groups[it]) & set(groups[iv])
            assert len(set(y[it]))==2 and len(set(y[iv]))==2
            inner.append({'train':it.tolist(),'validation':iv.tolist()})
        fold={'fold':'loso_'+s,'scheme':'loso','held_center':s,'train':tr.tolist(),'test':te.tolist(),'inner':inner}
        fold['seeds']={'inner':[seed('training',0,fold['fold'],'inner',j) for j in range(3)],
                       'outer':seed('training',0,fold['fold'],'outer'),
                       'sensitivity':{str(k):seed('outer_sensitivity',k,fold['fold']) for k in [1,2]}}
        folds.append(fold)
    protected = {str(p.relative_to(REST)):sha(p) for p in REST.rglob('*') if p.is_file()}
    prior=read(REST/'supplementary/focused_strengthening_20261026/protocol.json')['protected_files']
    assert all(protected[k]==v for k,v in prior.items())
    roles={'participants':people,'folds':folds,'excluded': [r for r in records if r['exclusion_reasons']],
           'source_metadata':{'path':str(meta_path),'sha256':sha(meta_path)}, 'header':next(iter(headers)),
           'original_N':871,'complete_190_N':399,'primary_N':323,'finite230_N':243}
    write(ROLES,roles)
    mc=read(original.CONFIG)['models']
    config={'status':'FROZEN_BEFORE_ASD_FEATURES_AND_PREDICTIONS','date':'2026-10-28',
      'frozen_utc':datetime.now(timezone.utc).isoformat(), 'models':mc,'participant_roles_sha':sha(ROLES),
      'scope':'Secondary ASD cross-disorder methodological replication, NOT independent MDD clinical validation',
      'paper1_overlap':'Same ABIDE derivatives and temporal-null methodology as Paper 1, submitted to Array; new trained predictive endpoints only. Added after observing REST results; not part of original primary study.',
      'primary':'Equal-center GNN Brier temporal minus training-rank-budget, primary seed only; negative favors temporal. No equivalence margin or automatic superiority claim.',
      'arms':list(ARMS),'source_algorithm_hashes':{p:sha(REST/'scripts'/p) for p in ['common.py','representations.py','models.py','experiment.py','metrics.py','metric_uncertainty.py','analyze_results.py']},
      'implementation_hashes':{p:sha(HERE/p) for p in ['replication.py','analysis.py','run.ps1']},
      'protected_rest_files':protected,
      'preprocessing':{'pipeline':'PCP DPARSF filt_noglobal rois_aal','additional_filtering_GSR_smoothing_censoring':'none',
       'first190':'rows 0:190 of already preprocessed derivatives; no additional volume discard',
       'native230_adapter_name':'first230 = rows 0:230, NOT full original run', 'last190':'rows 40:230, fixed overlap150',
       'normalization':'Frozen Pearson sample standardization within person and segment; same largest full-FC eigenvalue scaling across arms',
       'phase_null':'B100 pooled .99; frozen DC/Nyquist convention; recomputed for every segment; seed from source hash and context',
       'length_scope':'retained derivative volumes, not equal physical duration; whole-run preprocessing precedes segmentation'},
      'eligibility':{'N':323,'ASD':156,'HC':167,'centers':support,'min_total':30,'min_each_class':10,
                     'no_ROI_imputation':True,'constant_sd_threshold':1e-10,'provider_QC':qc,'age_restriction':'none'},
      'validation':'5 held-center folds, GroupKFold(3) inner roles explicitly frozen; training-only scaler, rank budget and tuning. No mixed-site analysis or target adaptation.',
      'seed_protocol':{'function':'common.seed with fixed ABIDE_REPLICATION_20261028 namespace; exact fold seeds in participant_roles.json',
                       'primary':0,'additional':[1,2],'extra_arms':['temporal','budget'],'retune_extra_seeds':False,'extra_segment_replay':True},
      'confounding':{'baseline':'Age, recorded sex (male indicator), provider mean FD; training-only StandardScaler and logistic C grid with same inner macro Brier selection',
       'low_motion':'Evaluate existing primary predictions for provider func_mean_fd <=0.2; no refitting; report center/class counts and conditional scope',
       'associations':'Descriptive within-center probability correlations with age and mean FD, plus sample demographic summaries; no causal disease-specific claims'},
      'statistics':{'bootstrap_B':2000,'paired':'Same resamples for all model comparisons; within-center diagnosis-stratified and separate center-cluster resampling',
       'scope':'Conditional on fitted predictions; omits training-sample and selection variability; five centers limits primary inference and three centers limits stability inference',
       'primary_metric':'Equal-center macro Brier','secondary':'AUC, Brier, log loss, threshold0.5 balanced accuracy, calibration, stability and operator identity',
       'stability':'Pooled mean absolute probability change first230 versus first190, plus fixed last190 replay; equal-center summaries also reported',
       'calibration':'Pooled diagnostic intercept/slope using frozen metrics implementation; no recalibration; site slopes suppressed unless >=50 per class; 2000 pooled stratified fixed-score calibration resamples with failure counts',
       'collapse':'Report probability range<=1e-7 as constant; also range<=1e-4, standard deviation and all-one-class predictions; retain all results',
       'multiplicity':'Only primary GNN temporal-budget Brier contrast is primary; all other intervals descriptive, not a significance search'},
      'limits':{'outer_primary_GNN':25,'outer_primary_LR':25,'extra_GNN':20,'inner_GNN_trajectories':150,'inner_LR':375,'confound_LR':80,
                'stop':'One fixed experiment. Stop on unresolved identity/atlas/leakage/integrity errors; repair execution errors only with recorded provenance. No architecture, dataset, atlas, GSR or grid expansion after results.'}}
    write(CFG,config)
    (HERE/'protocol.sha256').write_text(sha(CFG)+'  protocol.json\n',encoding='ascii')
    print(json.dumps({'frozen':True,'N':323,'ASD':156,'HC':167,'centers':support,'finite230':243,'protocol_sha':sha(CFG),'protected_REST_files':len(protected)}),flush=True)

def check_lock(check_sources=False):
    cfg=read(CFG); assert sha(CFG)==(HERE/'protocol.sha256').read_text().split()[0]
    assert sha(ROLES)==cfg['participant_roles_sha']
    assert cfg['models']==read(original.CONFIG)['models']
    for p,h in cfg['source_algorithm_hashes'].items(): assert sha(REST/'scripts'/p)==h
    for p,h in cfg['implementation_hashes'].items(): assert sha(HERE/p)==h
    roles=read(ROLES)
    if check_sources:
        for r in roles['participants']: assert sha(r['path'])==r['sha256']
    return cfg,roles

def context_slice(x,context):
    return x[:190] if context=='first190' else x[:230] if context=='native230' else x[40:230]

def prepare_one(row):
    with threadpool_limits(1):
        assert sha(row['path'])==row['sha256']; x=np.loadtxt(row['path']); out=[]
        for context in ['first190']+(['native230','last190'] if row['finite230'] else []):
            f=CACHE/(row['id']+'_'+context+'.npz'); rs=seed(row['sha256'],context,'AAL116','phase_B100_Q99')
            if not f.exists():
                segment=context_slice(x,context); c,w,u=rep.fc_spectrum(segment); th=rep.phase_threshold(segment,rs)
                assert np.isclose(w.sum(),116) and np.allclose(c,(u*w[None,:])@u.T,atol=1e-10)
                np.savez_compressed(f,w=w,u=u,threshold=th,T=len(segment),seed=rs,source_sha=row['sha256'],constructor_sha=sha(REST/'scripts/representations.py'))
            else:
                with np.load(f) as z:
                    assert int(z['seed'])==rs and str(z['source_sha'])==row['sha256'] and str(z['constructor_sha'])==sha(REST/'scripts/representations.py')
            out.append({'file':f.name,'sha256':sha(f)})
        return out

def cache():
    cfg,roles=check_lock(True); CACHE.mkdir(exist_ok=True); start=time.time(); files=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(prepare_one,r) for r in roles['participants']]
        for i,f in enumerate(as_completed(jobs),1):
            files+=f.result()
            if i%50==0 or i==323:print('spectra',i,'/323',round(time.time()-start,1),'seconds',flush=True)
    assert len(files)==809
    write(HERE/'spectral_cache_manifest.json',{'protocol_sha':sha(CFG),'n_contexts':809,'files':files,'elapsed_seconds':time.time()-start})

def load_data(roles):
    result={}
    for context in ['first190','native230','last190']:
        fields={k:[] for k in ['w','u','threshold','T']}
        for r in roles['participants']:
            actual=context if r['finite230'] else 'first190'
            with np.load(CACHE/(r['id']+'_'+actual+'.npz')) as z:
                for k in fields:fields[k].append(z[k].copy())
        result[context]={k:np.array(v) for k,v in fields.items()}
    return result

def train_confound(fold,rows,y,groups):
    dest=HERE/'confounding'/ (fold['fold']+'.joblib')
    if dest.exists(): return
    x=np.array([[r['age'],int(r['sex_code']==1),r['mean_fd']] for r in rows]); tr=np.array(fold['train']); te=np.array(fold['test'])
    grid=read(CFG)['models']['C_grid']; scores=[]
    for c in grid:
        vals=[]; iy=[]
        for f in fold['inner']:
            a=np.array(f['train']); b=np.array(f['validation']);sc=StandardScaler().fit(x[a])
            lr=LogisticRegression(C=c,solver='lbfgs',max_iter=3000,tol=1e-6).fit(sc.transform(x[a]),y[a])
            assert lr.n_iter_.max()<3000; vals.extend(lr.predict_proba(sc.transform(x[b]))[:,1]);iy.extend(b)
        iy=np.array(iy);scores.append(exp.macro_brier(y[iy],np.array(vals),groups[iy]))
    best=0
    for j in range(1,len(grid)):
        if scores[j]<scores[best]-1e-10:best=j
    sc=StandardScaler().fit(x[tr]);lr=LogisticRegression(C=grid[best],solver='lbfgs',max_iter=3000,tol=1e-6).fit(sc.transform(x[tr]),y[tr])
    assert lr.n_iter_.max()<3000
    dest.parent.mkdir(exist_ok=True)
    joblib.dump({'model':lr,'scaler':sc,'features':['age','male_indicator','mean_fd'],'train':tr,'test':te,
                 'probabilities':lr.predict_proba(sc.transform(x[te]))[:,1],'selected_C':grid[best],'inner_scores':scores,'protocol_sha':sha(CFG)},dest,compress=3)

def extra_seed(fold,arm,k,data,y,rows):
    name=fold['fold']+'_'+arm; folder=HERE/'seed_sensitivity'/('seed'+str(k)); folder.mkdir(parents=True,exist_ok=True)
    dest=folder/(name+'.npz')
    if dest.exists():
        audit=read(folder/(name+'.json'));assert audit['protocol_sha']==sha(CFG) and audit['predictions_sha']==sha(dest);return
    original_fit=read(HERE/'main'/(name+'.json')); b=original_fit['budget'];hp=original_fit['selected_gnn']
    tr=np.array(fold['train']); te=np.array(fold['test']);rs=fold['seeds']['sensitivity'][str(k)]
    ops,_,_=exp.operators(data['first190'],arm,b)
    model,_,audit=models.train_gnn(ops[tr],y[tr],rs,hp[1],hp[2])
    chunks=[]
    for context in ['first190','native230','last190']:
        ix=te if context=='first190' else te[[rows[i]['finite230'] for i in te]]
        if not len(ix):continue
        s,r,_=exp.operators({key:v[ix] for key,v in data[context].items()},arm,b)
        p,h=models.predict_gnn(model,s);chunks.append({'index':ix,'context':np.array([context]*len(ix)),'y':y[ix],'p_gnn':p,'embedding':h,'rank':r})
    np.savez_compressed(dest,**{key:np.concatenate([c[key] for c in chunks]) for key in chunks[0]})
    torch.save({'state_dict':{key:v.cpu() for key,v in model.state_dict().items()},'budget':b,'train_indices':tr,
                'hyperparameters':hp,'training_audit':audit,'config_sha':sha(CFG)},folder/(name+'_gnn.pt'))
    write(folder/(name+'.json'),{'protocol_sha':sha(CFG),'predictions_sha':sha(dest),'hyperparameters':hp,'budget':b,'training_audit':audit,'retuned':False})

def train():
    cfg,roles=check_lock(True)
    for r in read(HERE/'spectral_cache_manifest.json')['files']: assert sha(CACHE/r['file'])==r['sha256']
    data=load_data(roles);rows=roles['participants'];y=np.array([r['y'] for r in rows]);groups=np.array([r['site'] for r in rows])
    exp.RUN=HERE; exp.CONFIG=CFG; exp.people_rows=rows; exp.seed=seed
    models.CONFIG=CFG
    start=time.time();torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    with threadpool_limits(2):
        for fold in roles['folds']:
            for arm in ARMS:
                exp.fit_one(fold,arm,data,y,groups)
                print('PRIMARY_DONE',fold['held_center'],arm,round(time.time()-start,1),'seconds',flush=True)
            train_confound(fold,rows,y,groups)
        for k in [1,2]:
            for fold in roles['folds']:
                for arm in ['temporal','budget']:
                    extra_seed(fold,arm,k,data,y,rows)
                    print('SENSITIVITY_DONE',k,fold['held_center'],arm,flush=True)
    write(HERE/'runtime.json',{'elapsed_training_seconds':time.time()-start,'device':str(models.DEVICE),
      'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
      'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
      'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
      'host_peak_RAM':'not recorded','python':sys.version,'numpy':np.__version__})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','cache','train','analyze','verify']);a=parser.parse_args()
    if a.stage in ['freeze','cache','train']:globals()[a.stage]()
    else:
        # Import through this module's canonical name so verification imports are stable.
        import analysis
        getattr(analysis,a.stage)()
