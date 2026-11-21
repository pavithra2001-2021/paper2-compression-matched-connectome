"""ONE post-hoc alternative: masked, factorized Chebyshev spectral filters.

Only the per-layer frequency response is changed. Identity ROI features, width,
pooling, dropout, head, folds and selection policy remain those of the primary.
No functions in this module write into the primary result namespace.
"""
import argparse
import copy
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT,RUN,CONFIG,read,write,sha,seed,people
from representations import ARMS,fit_budget,make_operators
from models import SpectralGNN,DEVICE
from experiment import load_cache,macro_brier

ALT=ROOT/'results/posthoc_spectral_v1_20261013'
ACONFIG=ROOT/'config/posthoc_spectral_v1_20261013.json'

def basis(data,arm,budget):
    s,rank,raw=make_operators(**data,arm=arm,budget=budget)
    w=data['w']/data['w'][:,0,None]; u=data['u']
    m=np.arange(w.shape[1])[None,:]<rank[:,None]
    x=2*w-1
    values=np.stack([m,m*x,m*(2*x*x-1)],axis=1)
    b=np.einsum('nik,nqk,njk->nqij',u,values,u,optimize=True).astype(np.float32)
    return b,rank,s,raw

class MaskedChebyshevGNN(SpectralGNN):
    def __init__(self,n=116,hidden=16,dropout=.2):
        super().__init__(n,hidden,dropout)
        # g(x)=.5+.5*(2x-1)=x: original model is an exact initial special case.
        self.theta=torch.nn.Parameter(torch.tensor([[.5,.5,0.],[.5,.5,0.]]))

    def forward(self,b,embedding=False):
        f0=torch.einsum('q,bqij->bij',self.theta[0],b)
        f1=torch.einsum('q,bqij->bij',self.theta[1],b)
        h1=torch.relu(f0@self.w0+self.b0)
        h2=torch.relu((f1@self.dropout(h1))@self.w1+self.b1)
        h=h2.mean(1); z=self.head(self.dropout(h)).squeeze(-1)
        return (z,h) if embedding else z

def predict(model,b):
    model.eval(); ps=[]; hs=[]
    with torch.no_grad():
        for start in range(0,len(b),128):
            z,h=model(torch.as_tensor(b[start:start+128],device=DEVICE),embedding=True)
            ps.append(torch.sigmoid(z).cpu().numpy()); hs.append(h.cpu().numpy())
    return np.concatenate(ps).astype(float),np.concatenate(hs)

def train(b,y,rs,decay,epochs,val=None,snapshots=()):
    mc=read(CONFIG)['models']; torch.manual_seed(rs); torch.cuda.manual_seed_all(rs)
    model=MaskedChebyshevGNN().to(DEVICE)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    opt=torch.optim.Adam(model.parameters(),lr=mc['learning_rate'],weight_decay=decay)
    x=torch.as_tensor(b,device=DEVICE); yy=torch.as_tensor(y,dtype=torch.float32,device=DEVICE)
    preds={}; losses=[]
    for epoch in range(1,epochs+1):
        model.train(); opt.zero_grad(set_to_none=True)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),yy)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; no automatic rescue or retuning')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),mc['gradient_clip']); opt.step()
        if epoch in snapshots: preds[epoch]=predict(model,val)[0]
        if epoch in [1,epochs]: losses.append(float(loss.detach().cpu()))
    delta=sum(torch.sum((v.detach().cpu()-initial[k])**2).item() for k,v in model.state_dict().items())**.5
    return model,preds,{'seed':rs,'epochs':epochs,'weight_decay':decay,'training_loss_first_last':losses,
             'weight_update_l2':delta,'theta':model.theta.detach().cpu().tolist()}

def fixtures():
    rng=np.random.default_rng(421); a=rng.normal(size=(9,8,8)); c=a@a.transpose(0,2,1)
    w,u=np.linalg.eigh(c); w=w[:,::-1].copy(); u=u[:,:,::-1].copy()
    data={'w':w,'u':u,'threshold':w[:,3]*1.01,'T':np.full(9,190)}
    budget={'cutoff':float(np.median(w[:,3])),'fixed_k':3}
    torch.manual_seed(91); old=SpectralGNN(8,3).double().eval()
    torch.manual_seed(91); alt=MaskedChebyshevGNN(8,3).double().eval()
    for arm in ARMS:
        b,k,s,_=basis(data,arm,budget); bb=torch.tensor(b,dtype=torch.float64)
        np.testing.assert_allclose(old(torch.tensor(s,dtype=torch.float64)).detach(),alt(bb).detach(),atol=1e-7)
        # General spectral response == masked Chebyshev matrix implementation.
        th=np.array([.2,-.1,.3]); x=2*w/w[:,0,None]-1; m=np.arange(8)<k[:,None]
        spec=(u*(m*(th[0]+th[1]*x+th[2]*(2*x*x-1)))[:,None,:])@u.transpose(0,2,1)
        mat=np.einsum('q,nqij->nij',th,b)
        np.testing.assert_allclose(mat,spec,atol=1e-7)
        if np.any(k<8):
            for j in range(9): np.testing.assert_allclose(mat[j]@u[j,:,k[j]:],0,atol=1e-7)
        # Joint graph/node-coordinate relabeling preserves graph-level inference.
        perm=rng.permutation(8); state=copy.deepcopy(alt.state_dict()); permuted=copy.deepcopy(alt)
        state['w0']=state['w0'][perm]; permuted.load_state_dict(state)
        np.testing.assert_allclose(alt(bb).detach(),permuted(bb[:,:,perm][:,:,:,perm]).detach(),atol=1e-7)
    # Explicitly test empty retained subspace and gradient flow to all filter coefficients.
    empty={**data,'threshold':w[:,0]+1}; eb,_,_,_=basis(empty,'temporal',budget); assert np.max(np.abs(eb))==0
    b,_,_,_=basis(data,'full',budget); xx=torch.tensor(b,dtype=torch.float64)
    alt.zero_grad(); alt(xx).sum().backward(); assert torch.isfinite(alt.theta.grad).all() and (alt.theta.grad.abs()>1e-12).all()
    return {'passed':True,'original_nested_at_initialization':True,'spectral_matrix_equivalence':True,
      'discarded_modes_stay_zero':True,'empty_subspace_safe':True,'joint_permutation_invariance':True,'theta_gradients':True}

def freeze():
    checks=fixtures(); parent=read(CONFIG)
    spec={'status':'FROZEN_BEFORE_ALTERNATIVE_FITS_POST_HOC_EXPLORATORY','date':'2026-10-13',
       'primary_config_sha':sha(CONFIG),'participant_roles_sha':sha(RUN/'participant_roles.json'),
       'diagnostic_sha':sha(ROOT/'results/posthoc_graph_audit_20261013/diagnostics.json'),
       'training_code_sha':sha(__file__),'constructor_sha':sha(ROOT/'scripts/representations.py'),
       'architecture':'Two-layer masked Chebyshev spectral-filter GNN, shared scalar response per layer, order two',
       'operator':'F_l=U diag[m_k sum(q=0..2) theta_lq T_q(2 lambda_k/lambda_max(full C)-1)] U^T',
       'features':'I_116; no participant-specific node covariates',
       'forward':'H1=ReLU(F0 W0+b0); H2=ReLU(F1 Dropout(H1) W1+b1); h=mean_nodes(H2); logit=head(Dropout(h))',
       'theta_initial':[[.5,.5,0],[.5,.5,0]],'parameter_count':2167,
       'changes_from_primary':'Six learnable frequency-response coefficients only; same node features, pooling, width, head, loss and optimizer selection grid',
       'filter_regularization':'Same chosen Adam weight decay applies to all parameters including theta; no extra constraint or retuning',
       'representations':list(ARMS),'selection':parent['models'],
       'validation':parent['validation'],'segments':['first190','native230','last190'],
       'seed_rule':'Exactly original training-role seed with permutation=0; theta initialization consumes no randomness',
       'analysis':{'metrics':['AUC','AP','balanced_accuracy','sensitivity','specificity','Brier','log_loss','calibration','reliability'],
          'comparisons':['alternative temporal minus budget','alternative minus original GNN in each arm',
             'mixed versus held-cohort','native230 and last190 versus first190'],
          'intervals':'2000 paired fixed-prediction within-site diagnosis and site-cluster bootstrap; calibration within-site; omit retraining uncertainty',
          'calibration_scope':'Pooled primary and sites meeting original 50/class rule; suppress unsupported site slopes',
          'mechanism':'Saved theta, spectral response at each participant retained eigenvalue, effective embedding rank; descriptive',
          'multiplicity':'Exploratory, no confirmatory p-values or selection by held-out results'},
       'stop_rule':'One architecture, one frozen run per original role, no rescue attempts or extra models',
       'no_new_permutations':'Original sanity results remain primary provenance; new method has mathematical/training fixtures, not a new permutation significance claim',
       'original_result_authority':'Primary negative findings unchanged regardless of this post-hoc analysis',
       'fixtures':checks}
    if ACONFIG.exists(): assert read(ACONFIG)==spec,'Do not change frozen alternative specification'
    else: write(ACONFIG,spec)
    print('SINGLE EXPLORATORY ALTERNATIVE FROZEN; FIXTURES PASSED',flush=True)

def fit_one(fold,arm,data,y,sites,rows):
    name=fold['fold']+'_'+arm; folder=ALT/'main'; folder.mkdir(parents=True,exist_ok=True)
    meta_path=folder/(name+'.json')
    if meta_path.exists():
        old=read(meta_path); assert old['config_sha']==sha(ACONFIG) and old['training_code_sha']==sha(__file__)
        assert old['predictions_sha']==sha(folder/(name+'.npz')); return
    mc=read(CONFIG)['models']; tr=np.array(fold['train']); te=np.array(fold['test'])
    first=data['first190']; ranks=(first['w']>first['threshold'][:,None]).sum(1)
    candidates={(wd,ep):[] for wd in mc['weight_decay_grid'] for ep in mc['epoch_grid']}; inner_records=[]
    for j,inner in enumerate(fold['inner']):
        it=np.array(inner['train']); iv=np.array(inner['validation'])
        assert not set(it)&set(iv) and not set(sites[it])&set(sites[iv]) and not set(tr)&set(te)
        budget=fit_budget(first['w'][it],ranks[it]); b,_,_,_=basis(first,arm,budget)
        rs=seed('training',0,fold['fold'],'inner',j)
        for wd in mc['weight_decay_grid']:
            model,ps,_=train(b[it],y[it],rs,wd,max(mc['epoch_grid']),b[iv],mc['epoch_grid'])
            for ep,p in ps.items(): candidates[(wd,ep)].append((iv,p))
            del model
        inner_records.append({'train_indices':it.tolist(),'validation_indices':iv.tolist(),'budget':budget,'seed':rs})
        del b
    scores={}
    for key,chunks in candidates.items():
        idx=np.concatenate([v[0] for v in chunks]); p=np.concatenate([v[1] for v in chunks]); assert set(idx)==set(tr) and len(idx)==len(tr)
        scores[key]=macro_brier(y[idx],p,sites[idx])
    best=next(iter(scores))
    for key in scores:
        if scores[key]<scores[best]-1e-10: best=key
    budget=fit_budget(first['w'][tr],ranks[tr]); b,_,_,_=basis({k:v[tr] for k,v in first.items()},arm,budget)
    rs=seed('training',0,fold['fold'],'outer'); model,_,audit=train(b,y[tr],rs,best[0],best[1]); del b
    records={k:[] for k in ['index','context','y','probability','embedding','rank','operator_distance','embedding_distance']}
    ref_b,_,ref_s,_=basis({k:v[te] for k,v in first.items()},arm,budget); _,ref_h=predict(model,ref_b); del ref_b
    for context in ['first190','native230','last190']:
        local=np.arange(len(te)) if context=='first190' else np.array([j for j,i in enumerate(te) if rows[i]['T']==230],dtype=int)
        ix=te[local]
        if not len(ix): continue
        b,k,s,_=basis({k:v[ix] for k,v in data[context].items()},arm,budget); p,h=predict(model,b)
        values={'index':ix,'context':np.array([context]*len(ix)),'y':y[ix],'probability':p,'embedding':h,'rank':k,
          'operator_distance':np.linalg.norm(s-ref_s[local],axis=(1,2)),'embedding_distance':np.linalg.norm(h-ref_h[local],axis=1)}
        for k,v in values.items(): records[k].append(v)
        del b
    predpath=folder/(name+'.npz'); np.savez_compressed(predpath,**{k:np.concatenate(v) for k,v in records.items()})
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'train_indices':tr,
        'config_sha':sha(ACONFIG),'budget':budget,'hyperparameters':best,'training_audit':audit},folder/(name+'.pt'))
    write(meta_path,{'fold':fold['fold'],'scheme':fold['scheme'],'arm':arm,'config_sha':sha(ACONFIG),'training_code_sha':sha(__file__),
        'predictions_sha':sha(predpath),'budget':budget,'inner':inner_records,'selected':best,
        'candidate_scores':[{'weight_decay':k[0],'epochs':k[1],'macro_brier':v} for k,v in scores.items()],'training_audit':audit})
    del model; torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--freeze',action='store_true'); parser.add_argument('--fixtures-only',action='store_true'); args=parser.parse_args()
    if args.fixtures_only: print(fixtures()); return
    if args.freeze: freeze(); return
    spec=read(ACONFIG); assert spec['training_code_sha']==sha(__file__) and spec['primary_config_sha']==sha(CONFIG)
    assert spec['participant_roles_sha']==sha(RUN/'participant_roles.json') and spec['constructor_sha']==sha(ROOT/'scripts/representations.py')
    rows=people(); data=load_cache(rows); roles=read(RUN/'participant_roles.json')
    y=np.array([r['y'] for r in roles['participants']]); sites=np.array([r['site'] for r in rows]); start=time.time()
    with threadpool_limits(limits=2):
        for fold in roles['folds']:
            for arm in ARMS:
                fit_one(fold,arm,data,y,sites,rows)
                print('completed',fold['fold'],arm,'elapsed_seconds',round(time.time()-start),flush=True)
    print('ALL 95 EXPLORATORY ALTERNATIVE TASKS COMPLETE',flush=True)

if __name__=='__main__': main()
