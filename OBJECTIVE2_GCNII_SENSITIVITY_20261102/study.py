"""One approved GCNII-derived architecture sensitivity; originals are read-only."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
sys.dont_write_bytecode = True
import argparse
import csv
import gc
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REST = HERE.parent/'OBJECTIVE2_RESTMETA_FMRI'
ABIDE = HERE.parent/'OBJECTIVE2_ABIDE_REPLICATION_20261028'
sys.path.insert(0, str(REST/'scripts'))
import common as original
import models as original_models
import representations as rep
from metrics import metrics, calibration, reliability, rho, paired_interval
from metric_uncertainty import bootstrap_site, weighted_auc
read, write, sha, seed = original.read, original.write, original.sha, original.seed
DEVICE = original_models.DEVICE
CFG = HERE/'protocol.json'
ROLES = HERE/'participant_roles.json'
FINAL = HERE/'final'
ARMS = rep.ARMS
CONTEXTS = ['first190', 'native230', 'last190']
CODE = ['study.py', 'analysis.py', 'run.ps1']

class GCNIIComparator(nn.Module):
    """GCNII-derived signed graph-shift comparator, not canonical GCNII."""
    def __init__(self, n=116, hidden=16, dropout=.2):
        super().__init__()
        self.input_weight = nn.Parameter(torch.empty(n, hidden))
        self.input_bias = nn.Parameter(torch.zeros(hidden))
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(hidden, hidden)) for _ in range(2)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(2)])
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.input_weight)
        for w in self.weights: nn.init.xavier_uniform_(w)

    def forward(self, s, embedding=False):
        h0 = torch.relu(self.input_weight + self.input_bias).unsqueeze(0).expand(s.shape[0], -1, -1)
        h = h0
        eye = torch.eye(h.shape[-1], dtype=h.dtype, device=h.device)
        for layer, (w, b) in enumerate(zip(self.weights, self.biases), 1):
            beta = math.log(1 + .5/layer)
            h = torch.relu((.9*(s @ self.dropout(h)) + .1*h0) @ ((1-beta)*eye + beta*w) + b)
        pooled = h.mean(1)
        z = self.head(self.dropout(pooled)).squeeze(-1)
        return (z, pooled) if embedding else z

def predict(model, s):
    return original_models.predict_gnn(model, s)

def fit(s, y, random_seed, decay, epochs, val=None, snapshots=()):
    # Only replace the constructor in process memory; original file is never edited.
    previous = original_models.SpectralGNN
    original_models.SpectralGNN = GCNIIComparator
    try:
        return original_models.train_gnn(s, y, random_seed, decay, epochs, val, snapshots)
    finally:
        original_models.SpectralGNN = previous

def csvwrite(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError('Empty output '+str(path))
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

def fixtures():
    torch.manual_seed(371)
    m = GCNIIComparator(n=5, hidden=3).double().eval()
    s = torch.tensor([[[1.,-.25,.1,0,0],[-.25,1,0,.1,0],[.1,0,1,0,-.2],[0,.1,0,1,.1],[0,0,-.2,.1,1]]], dtype=torch.double)
    s = s/torch.linalg.eigvalsh(s).max()
    # Independent node-by-node equation, not a second call to forward.
    h0 = torch.relu(torch.eye(5, dtype=torch.double) @ m.input_weight + m.input_bias)
    h = h0.clone()
    for ell in (1,2):
        b = math.log(1+.5/ell)
        h = torch.stack([torch.relu((.9*sum(s[0,i,j]*h[j] for j in range(5))+.1*h0[i]) @
                       ((1-b)*torch.eye(3,dtype=torch.double)+b*m.weights[ell-1])+m.biases[ell-1]) for i in range(5)])
    expected = m.head(h.mean(0)).squeeze()
    torch.testing.assert_close(m(s)[0], expected, atol=1e-12, rtol=1e-12)
    assert sum(p.numel() for p in GCNIIComparator().parameters()) == 2433
    assert not torch.allclose(m(s), m(s.abs()), atol=1e-10, rtol=0)
    assert not torch.allclose(m(s), m(torch.zeros_like(s)), atol=1e-10, rtol=0)
    x = s.clone().requires_grad_(True); g = torch.autograd.grad(m(x).sum(), x)[0]
    assert torch.isfinite(g).all() and g.norm()>0
    d = torch.randn_like(x); eps=1e-6
    fd=(m(x+eps*d)-m(x-eps*d))/(2*eps)
    torch.testing.assert_close(fd, (g*d).sum((1,2)), atol=1e-7, rtol=1e-5)
    # A joint relabeling of anatomical coordinates preserves the output.
    perm=torch.tensor([2,0,4,1,3]); other=GCNIIComparator(5,3).double().eval();other.load_state_dict(m.state_dict())
    with torch.no_grad(): other.input_weight.copy_(m.input_weight[perm])
    torch.testing.assert_close(m(s),other(s[:,perm][:,:,perm]),atol=1e-12,rtol=1e-12)
    m.train();torch.manual_seed(63);a=m(s);torch.manual_seed(63);b=m(s)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert not torch.allclose(a,m.eval()(s),atol=1e-12,rtol=0)
    # Tie-aware AUC and paired contrast algebra fixtures.
    from sklearn.metrics import roc_auc_score
    yy=np.array([0,1,0,1]); pp=np.array([.5,.5,.1,.9])
    assert abs(weighted_auc(yy,pp,np.ones((1,4)))[0]-roc_auc_score(yy,pp))<1e-12
    return {'passed':True,'equations_nodewise':True,'parameter_count':2433,'signed_weights_used':True,
      'graph_used':True,'gradient_finite_difference':True,'anatomical_joint_permutation':True,
      'dropout_determinism':True,'tie_aware_AUC':True,'no_training_performed':True}

def validate_roles(roles):
    for dataset,d in roles.items():
        rows=d['participants']; groups=np.array([r['site'] for r in rows]); seen=[]
        for f in d['folds']:
            tr=set(f['train']);te=set(f['test']);assert not tr & te
            assert tr | te == set(range(len(rows)))
            assert not set(groups[list(tr)]) & set(groups[list(te)])
            inner_seen=[]
            for v in f['inner']:
                it=set(v['train']);iv=set(v['validation'])
                assert not it & iv and it | iv == tr
                assert not set(groups[list(it)]) & set(groups[list(iv)])
                assert len({rows[i]['y'] for i in it})==2 and len({rows[i]['y'] for i in iv})==2
                inner_seen += v['validation']
            assert len(inner_seen)==len(set(inner_seen))==len(tr)
            seen += f['test']
        expected=[i for i,r in enumerate(rows) if dataset!='REST' or r['site']!='S12']
        assert sorted(seen)==expected and len(seen)==len(set(seen))
        assert len(seen)==(1150 if dataset=='REST' else 323)

def freeze():
    assert not CFG.exists() and not (HERE/'fits').exists()
    checks=fixtures(); roles={}
    import analysis
    mc=read(original.CONFIG)['models']
    assert (mc['hidden'],mc['dropout'],mc['learning_rate'],mc['gradient_clip'])==(16,.2,.01,5.)
    assert mc['weight_decay_grid']==[.01,.0001] and mc['epoch_grid']==[80,160]
    restrows=original.people(); old=read(original.RUN/'participant_roles.json')
    assert [r['subject_id'] for r in restrows]==[r['id'] for r in old['participants']]
    for r in restrows: r.update(y=int(r['diagnosis']=='MDD'), finite230=r['T']==230, id=r['subject_id'])
    assert len(restrows)==1188 and not any(r['site']=='S4' or r['id'].startswith('S4-') for r in restrows)
    assert any(r['site']=='S14' for r in restrows)
    roles['REST']={'participants':restrows,'folds':[f for f in old['folds'] if f['scheme']=='loso' and f['fold']!='loso_S12']}
    a=read(ABIDE/'participant_roles.json')
    roles['ABIDE']={'participants':a['participants'],'folds':a['folds']}
    for ds,d in roles.items():
        for f in d['folds']:
            if ds=='REST':
                f['seeds']={'inner':[seed('training',0,f['fold'],'inner',j) for j in range(3)],
                  'outer':seed('training',0,f['fold'],'outer'),
                  'sensitivity':{str(k):seed('focused_strengthening_20261026',f['fold'],k) for k in [1,2]}}
    validate_roles(roles)
    # Verify availability and shape of cached inputs and confounding metadata without diagnostic scoring.
    external_hashes={}
    for ds,d in roles.items():
        cached=load_data(ds,d['participants'])
        for ctx,v in cached.items():
            assert v['w'].shape==(len(d['participants']),116) and v['u'].shape==(len(d['participants']),116,116)
            assert all(np.isfinite(x).all() for x in v.values())
        _,hh=analysis.metadata(ds,d['participants']);external_hashes.update(hh)
        del cached
    # Check existing preservation records before constructing a new snapshot.
    for p,h in read(ABIDE/'protocol.json')['protected_rest_files'].items(): assert sha(REST/p)==h,p
    for p,h in read(ABIDE/'final/artifact_manifest.json')['files'].items(): assert sha(ABIDE/p)==h,p
    protected={}
    for root in [REST,ABIDE]:
        for p in root.rglob('*'):
            if p.is_file(): protected[str(p.relative_to(HERE.parent))]=sha(p)
    write(HERE/'protected_originals.json',protected);write(ROLES,roles)
    write(HERE/'external_metadata_sources.json',external_hashes)
    cfg={'status':'LOCKED_BEFORE_NEW_TRAINING_OR_SCORING','frozen_utc':datetime.now(timezone.utc).isoformat(),
      'scope':'Subsequently motivated architecture sensitivity. REST primary; ABIDE secondary ASD replication, not MDD external validation. No original results replaced.',
      'model':{'name':'GCNII-derived signed graph-shift comparator','n':116,'hidden':16,'layers':2,'alpha':.1,'beta':'log(1+0.5/l), l=1,2',
        'input':'I116','initial':'H0=ReLU(I W_in+b_in)','layer':'Hl=ReLU((0.9*S*Dropout(Hprev)+0.1*H0)*((1-beta)*I16+beta*Wl)+bl)',
        'head':'Linear(Dropout(mean_nodes(H2))); BCEWithLogitsLoss; sigmoid for prediction',
        'dropout':.2,'parameters':2433,'initialization':'Xavier uniform input and propagation weights; zero explicit biases; PyTorch Linear default head',
        'reference':'Chen et al. ICML 2020, https://proceedings.mlr.press/v119/chen20v.html',
        'adaptations':'Signed spectral-normalized S replaces normalized adjacency; anatomical identity features; graph-level mean head; explicit biases. No canonical GCNII theorem claimed.'},
      'models':read(original.CONFIG)['models'], 'arms':list(ARMS),'participant_roles_sha':sha(ROLES),
      'protected_originals_sha':sha(HERE/'protected_originals.json'),
      'external_metadata_sources_sha':sha(HERE/'external_metadata_sources.json'),
      'source_hashes':{str(p.relative_to(HERE.parent)):sha(p) for p in [original.CONFIG]+[REST/'scripts'/q for q in ['common.py','models.py','representations.py','experiment.py','metrics.py','metric_uncertainty.py','analyze_results.py']]},
      'implementation_hashes':{p:sha(HERE/p) for p in CODE},
      'selection':'Three fixed grouped inner partitions; macro Brier; grid wd [0.01,0.0001], epochs [80,160]; tolerance1e-10 ties prefer larger wd then fewer epochs. lr0.01 Adam, clip5, all parameters decay. No target calibration.',
      'seed_protocol':'Exact integers in roles; primary seed reused from original; extra seeds1/2 temporal/budget only, primary-selected hyperparameters, no retuning. Same integer is not identical initialization across architectures.',
      'primary':'REST equal-cohort Gamma=(new temporal Brier-new budget Brier)-(original temporal Brier-original budget Brier), primary seed0',
      'secondary':'Both architecture contrasts; ABIDE Gamma; all-arm AUC/Brier/log loss/calibration; collapse; three seeds; finite-input replay; existing metadata/prior/LR controls',
      'bootstrap':{'B':2000,'scheme':'Paired within cohort x diagnosis; separate cohort-cluster with fresh within resample; all models share draws',
        'limitations':'Conditional on fixed predictions, excludes training/selection uncertainty. No equivalence or algorithm-level superiority. ABIDE5 centers; finite3.',
        'calibration':'2000 pooled fixed-cohort diagnosis-stratified draws, failure counts; per-cohort slope suppressed unless >=50 per class; no recalibration'},
      'collapse':{'constant_range':1e-7,'near_constant_range':1e-4,'also_report':'exact zero range, <=1e-6 REST historical sensitivity, SD, logits and embeddings, same hard class; never exclude'},
      'finite_inputs':'REST native230/last190; ABIDE native230 means first230. first190 and last190 overlap150; preprocessing already full-run; all arms primary, extra seeds temporal/budget; no refit.',
      'confounding':'Reuse original metadata predictions, cohort handling and metadata sources. ABIDE low FD<=0.2 fixed-score sensitivity; within-cohort age/FD/sex association; REST raw QC categories and whole-derivative FD, no invented temporal alignment.',
      'limits':{'outer_primary':90,'outer_extra':72,'inner_trajectories':540,'total_training_trajectories':702,
        'stop':'One architecture only. Stop on unresolved provenance/leakage/nonfinite error; repair only demonstrated implementation errors with disclosure. No grid/depth/width/readout/seed/cohort expansion; report all results; stop implementation after verification.'}}
    write(CFG,cfg);(HERE/'protocol.sha256').write_text(sha(CFG)+'  protocol.json\n',encoding='ascii')
    write(HERE/'mathematical_checks.json',checks)
    print('FROZEN',sha(CFG),'protected',len(protected),'702 training trajectories maximum',flush=True)

def check_lock():
    c=read(CFG);assert sha(CFG)==(HERE/'protocol.sha256').read_text().split()[0]
    assert sha(ROLES)==c['participant_roles_sha'] and sha(HERE/'protected_originals.json')==c['protected_originals_sha']
    assert sha(HERE/'external_metadata_sources.json')==c['external_metadata_sources_sha']
    for p,h in c['implementation_hashes'].items():assert sha(HERE/p)==h,p
    for p,h in c['source_hashes'].items():assert sha(HERE.parent/p)==h,p
    roles=read(ROLES);validate_roles(roles)
    return c,roles

def load_data(ds,rows):
    result={}
    for ctx in CONTEXTS:
        d={k:[] for k in ['w','u','threshold','T']}
        for r in rows:
            actual=ctx if r['finite230'] else 'first190'
            path=original.cache_path(r,actual) if ds=='REST' else ABIDE/'spectra'/(r['id']+'_'+actual+'.npz')
            with np.load(path) as z:
                for k in d:d[k].append(z[k].copy())
        result[ctx]={k:np.asarray(v) for k,v in d.items()}
    return result

def operators(d,arm,budget):
    return rep.make_operators(**d,arm=arm,budget=budget)

def macro(values,groups):
    return float(np.mean([values[groups==s].mean() for s in sorted(set(groups))]))

def destination(ds,fold,arm,k):
    return HERE/'fits'/ds/('seed'+str(k))/(fold+'_'+arm)

def fit_one(ds,f,arm,k,data,rows):
    dest=destination(ds,f['fold'],arm,k);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.with_suffix('.json').exists():
        m=read(dest.with_suffix('.json'));assert m['protocol_sha']==sha(CFG)
        assert sha(dest.with_suffix('.npz'))==m['predictions_sha'] and sha(dest.with_suffix('.pt'))==m['checkpoint_sha'];return
    y=np.array([r['y'] for r in rows]); groups=np.array([r['site'] for r in rows]);tr=np.array(f['train']);te=np.array(f['test'])
    first=data['first190'];ranks=(first['w']>first['threshold'][:,None]).sum(1)
    b=rep.fit_budget(first['w'][tr],ranks[tr]);inner_records=[];candidates=[];inner_predictions=[]
    if k==0:
        vals={(wd,ep):[] for wd in [.01,.0001] for ep in [80,160]}
        for j,v in enumerate(f['inner']):
            it=np.array(v['train']);iv=np.array(v['validation']);ib=rep.fit_budget(first['w'][it],ranks[it])
            ops,_,_=operators(first,arm,ib)
            audits=[]
            for wd in [.01,.0001]:
                model,ps,audit=fit(ops[it],y[it],f['seeds']['inner'][j],wd,160,ops[iv],(80,160))
                audits.append(audit)
                for ep,p in ps.items():vals[wd,ep].append((iv,p))
                del model
            inner_records.append({'inner':j,'budget':ib,'seed':f['seeds']['inner'][j],'training_audits':audits})
        for (wd,ep),parts in vals.items():
            ix=np.concatenate([v[0] for v in parts]);p=np.concatenate([v[1] for v in parts]);assert set(ix)==set(tr) and len(ix)==len(tr)
            candidates.append({'wd':wd,'epochs':ep,'macro_brier':macro((p-y[ix])**2,groups[ix])})
            inner_predictions.append({'wd':wd,'epochs':ep,'index':ix,'p':p})
        hp=candidates[0]
        for h in candidates[1:]:
            if h['macro_brier']<hp['macro_brier']-1e-10:hp=h
    else:
        main=read(destination(ds,f['fold'],arm,0).with_suffix('.json'));hp=main['selected'];assert b==main['budget']
    rs=f['seeds']['outer'] if k==0 else f['seeds']['sensitivity'][str(k)]
    ops,_,_=operators(first,arm,b);model,_,audit=fit(ops[tr],y[tr],rs,hp['wd'],hp['epochs'])
    chunks=[]
    for ctx in CONTEXTS:
        ix=te if ctx=='first190' else te[[rows[i]['finite230'] for i in te]]
        if not len(ix):continue
        s,rank,_=operators({key:v[ix] for key,v in data[ctx].items()},arm,b)
        p,h=predict(model,s)
        with torch.no_grad():z=model(torch.tensor(s,dtype=torch.float32,device=DEVICE)).cpu().numpy()
        chunks.append({'index':ix,'context':np.full(len(ix),ctx),'y':y[ix],'p_gnn':p,'embedding':h,'logit':z,'rank':rank})
    np.savez_compressed(dest.with_suffix('.npz'),**{key:np.concatenate([v[key] for v in chunks]) for key in chunks[0]})
    torch.save({'state_dict':{q:v.detach().cpu() for q,v in model.state_dict().items()},'train_indices':tr,'test_indices':te,
      'budget':b,'hyperparameters':hp,'training_audit':audit,'inner_validation_predictions':inner_predictions,'protocol_sha':sha(CFG)},dest.with_suffix('.pt'))
    write(dest.with_suffix('.json'),{'dataset':ds,'fold':f['fold'],'arm':arm,'seed_id':k,'seed':rs,'budget':b,
      'selected':hp,'candidate_scores':candidates,'inner':inner_records,'retuned':k==0,'training_audit':audit,
      'protocol_sha':sha(CFG),'predictions_sha':sha(dest.with_suffix('.npz')),'checkpoint_sha':sha(dest.with_suffix('.pt'))})
    del model;gc.collect()

def train():
    cfg,roles=check_lock();assert read(HERE/'mathematical_checks.json')['passed']
    protected=read(HERE/'protected_originals.json')
    for p,h in protected.items():assert sha(HERE.parent/p)==h,p
    start=time.time()
    if DEVICE.type=='cuda':torch.cuda.reset_peak_memory_stats()
    with threadpool_limits(2):
        for ds,d in roles.items():
            data=load_data(ds,d['participants'])
            for k in [0,1,2]:
                for f in d['folds']:
                    for arm in ARMS if k==0 else ['temporal','budget']:
                        fit_one(ds,f,arm,k,data,d['participants'])
                        print('FIT_DONE',ds,k,f['fold'],arm,round(time.time()-start,1),'seconds',flush=True)
            del data
    write(HERE/'runtime.json',{'elapsed_training_seconds':time.time()-start,'device':str(DEVICE),
      'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if DEVICE.type=='cuda' else None,
      'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated() if DEVICE.type=='cuda' else None,
      'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved() if DEVICE.type=='cuda' else None,
      'host_peak_RAM':'not recorded','python':sys.version,'numpy':np.__version__})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fixtures','freeze','train','analyze','verify']);a=p.parse_args()
    if a.stage in ['fixtures','freeze','train']:
        value=globals()[a.stage]()
        if value:print(value)
    else:
        import analysis
        getattr(analysis,a.stage)()
