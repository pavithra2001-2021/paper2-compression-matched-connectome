"""Meaningful mathematical, held-out isolation and training/replay fixtures."""
import json
import numpy as np
import torch
from common import ROOT,RUN,CONFIG,read,write,sha,seed
from representations import correlation,fc_spectrum,phase_threshold,reconstruct,fit_budget,make_operators,vectorize
from models import SpectralGNN,train_gnn,predict_gnn,train_lr,predict_lr

def main():
    rng=np.random.default_rng(11)
    x=rng.normal(size=(190,12)); x[:,1]+=.5*x[:,0]
    c,w,u=fc_spectrum(x)
    np.testing.assert_allclose(c,np.corrcoef(x,rowvar=False),atol=1e-12)
    np.testing.assert_allclose(reconstruct(w,u,np.ones(12)),c,atol=1e-12)
    mask=w>1.2; k=int(mask.sum())
    np.testing.assert_allclose(reconstruct(w,u,mask),reconstruct(w,u,np.arange(12)<k),atol=1e-12)
    flips=rng.choice([-1,1],12)
    np.testing.assert_allclose(reconstruct(w,u*flips,mask),reconstruct(w,u,mask),atol=1e-12)
    a=phase_threshold(x,123,B=10); b=phase_threshold(x,123,B=10)
    assert a==b and a>0
    # Exact agreement with the historical implementation for the SAME seed.
    amp=np.abs(np.fft.rfft(x,axis=0)); rr=np.random.default_rng(123); vals=[]
    for _ in range(10):
        ph=rr.uniform(0,2*np.pi,size=amp.shape); ph[0]=0; ph[-1]=0
        z=np.fft.irfft(amp*np.exp(1j*ph),n=190,axis=0)
        vals.extend(np.linalg.eigvalsh(np.corrcoef(z,rowvar=False)))
    np.testing.assert_allclose(a,np.percentile(vals,99),atol=1e-12)
    ww=np.stack([w,w*.9,w*1.1]); uu=np.stack([u]*3)
    ranks=np.array([3,4,2]); d=fit_budget(ww,ranks)
    assert np.sum(ww>d['cutoff'])==sum(ranks)
    tied=fit_budget(np.array([[4.,2.,2.,1.],[4.,2.,2.,1.]]),[2,1])
    assert tied['attained']==2 # equidistant feasible counts 2 and 6? nearest 2
    heldout=ww.copy()*1000
    assert fit_budget(ww,ranks)==d and not np.array_equal(heldout,ww)
    zero,zero_k,_=make_operators(ww,uu,np.ones(3)*100,np.ones(3)*190,'temporal')
    assert np.count_nonzero(zero)==0 and np.count_nonzero(zero_k)==0
    s,_,_=make_operators(ww,uu,np.ones(3)*1.2,np.ones(3)*190,'temporal')
    assert vectorize(s).shape==(3,78)
    assert np.max(np.linalg.eigvalsh(s))<=1+1e-6
    try: correlation(np.ones((190,12)))
    except ValueError: pass
    else: raise AssertionError('Constant inputs accepted')
    roles=read(RUN/'participant_roles.json'); n=len(roles['participants'])
    for scheme in ('mixed','loso'):
        tests=[]
        for f in roles['folds']:
            if f['scheme']!=scheme: continue
            assert not set(f['train'])&set(f['test']); tests+=f['test']
            for inner in f['inner']:
                assert set(inner['train'])|set(inner['validation'])==set(f['train'])
                assert not set(inner['train'])&set(inner['validation'])
                train_sites={roles['participants'][i]['site'] for i in inner['train']}
                val_sites={roles['participants'][i]['site'] for i in inner['validation']}
                assert not train_sites&val_sites
        assert sorted(tests)==list(range(n))
    # Synthetic two-topology graph fixture: actual backprop must update weights.
    y=np.tile([0,1],24); graphs=np.stack([np.eye(12) if v==0 else np.ones((12,12))/12 for v in y]).astype('float32')
    model,_,audit=train_gnn(graphs,y,66,.0001,160)
    p,h=predict_gnn(model,graphs)
    assert audit['weight_update_l2']>0 and h.shape==(48,16)
    assert np.mean((p>.5)==y)>.9, ('trained synthetic topology fixture',p.min(),p.max())
    # Replay exact checkpoint and identical input is invariant.
    clone=SpectralGNN(12).to(next(model.parameters()).device); clone.load_state_dict(model.state_dict())
    np.testing.assert_array_equal(p,predict_gnn(clone,graphs)[0])
    linear,scaler=train_lr(graphs,y,1,66)
    assert np.mean((predict_lr(linear,scaler,graphs)>.5)==y)==1
    write(RUN/'fixture_checks.json',{'passed':True,'config_sha':sha(CONFIG),'source_sha':{p.name:sha(p) for p in (ROOT/'scripts').glob('*.py')},
        'checks':['Pearson parity','historical phase-null parity','same-K identity','eigenvector-sign invariance','zero-rank handling','budget count and tie rule','fixed participant and inner-site isolation','operator bound','trained GNN synthetic topology learning','checkpoint replay','linear comparator topology learning'],
        'synthetic_training_audit':audit})
    print('All fixtures passed; synthetic accuracy',np.mean((p>.5)==y),'device',next(model.parameters()).device)

if __name__=='__main__': main()
