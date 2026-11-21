"""Post-hoc, inference-only mechanism audit. Never fits or edits primary models.

Perturbations are sensitivity probes, not new performance experiments. Primary
and secondary scopes are kept separate; participant coordinates are never pooled
across independently fitted hidden spaces.
"""
import argparse
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import ROOT, RUN, CONFIG, read, write, sha, seed, people, cache_path
from representations import ARMS, make_operators
from models import SpectralGNN, DEVICE

OUT = ROOT / 'results/posthoc_graph_audit_20261013'
LOCK = ROOT / 'config/posthoc_graph_audit_20261013.json'

def ratio(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 1e-20)

def squared(x):
    return np.sum(np.asarray(x, float)**2, axis=tuple(range(1, x.ndim)))

def stats(x):
    x = np.asarray(x, float).ravel(); x = x[np.isfinite(x)]
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'q05': float(np.quantile(x, .05)), 'q95': float(np.quantile(x, .95))} if len(x) else {'n': 0}

def effective_rank(x):
    x = np.asarray(x, float); x = x-x.mean(axis=0)
    eig = np.maximum(np.linalg.eigvalsh(x.T@x), 0)
    if eig.sum() < 1e-20: return {'rank': 0., 'variance': 0., 'leading_fraction': None}
    p = eig[eig > 0]/eig.sum()
    return {'rank': float(np.exp(-np.sum(p*np.log(p)))),
            'variance': float(eig.sum()/len(x)), 'leading_fraction': float(p.max())}

def forward(model, s, first_weights=None):
    w0 = model.w0 if first_weights is None else first_weights
    a1 = s @ w0; h1 = torch.relu(a1+model.b0)
    a2 = (s @ h1) @ model.w1; h2 = torch.relu(a2+model.b1)
    h = h2.mean(1); z = model.head(h).squeeze(-1)
    return z, h1, h2, h, a1, a2

def evaluate(model, s, first_weights=None):
    with torch.no_grad():
        result = forward(model, s, first_weights)
        return [a.cpu().numpy() for a in result]

def fixtures():
    torch.manual_seed(112); model = SpectralGNN(n=8, hidden=3).double().eval()
    rng = np.random.default_rng(711); a=rng.normal(size=(5,8,8)); s=a@a.transpose(0,2,1)
    w,u=np.linalg.eigh(s); s=s/w[:,-1,None,None]; w=w/w[:,-1,None]
    st=torch.tensor(s); orig=forward(model,st)
    np.testing.assert_allclose(orig[0].detach(),model(st).detach(),atol=1e-12)
    p=rng.permutation(8); joint=forward(model,st[:,p][:,:,p],model.w0[p])
    np.testing.assert_allclose(orig[0].detach(),joint[0].detach(),atol=1e-12)
    np.testing.assert_allclose(s@model.w0.detach().numpy(),
        (u*w[:,None,:])@(u.transpose(0,2,1)@model.w0.detach().numpy()),atol=1e-12)
    zero=forward(model,torch.zeros_like(st))[0].detach().numpy()
    assert np.ptp(zero)==0
    # Mean pooling retains at most the node-wise difference energy.
    delta=rng.normal(size=(5,8,3)); fraction=8*squared(delta.mean(1))/squared(delta)
    assert np.all((fraction>=0)&(fraction<=1))
    # Autograd directional sensitivity of the fixed network, no optimizer.
    st.requires_grad_(True); z=forward(model,st)[0]
    grad=torch.autograd.grad(z.sum(),st)[0]; d=torch.tensor(rng.normal(size=s.shape))
    eps=1e-6
    fd=(forward(model,st+eps*d)[0]-forward(model,st-eps*d)[0])/(2*eps)
    np.testing.assert_allclose(fd.detach(),(grad*d).sum((1,2)).detach(),rtol=1e-5,atol=1e-7)
    return {'passed':True,'spectral_equals_matrix':True,'joint_permutation_invariant':True,
            'graph_zero_constant':True,'pooling_energy_bound':True,'gradient_finite_difference':True}

def protected_paths():
    paths=list(RUN.rglob('*'))+list((ROOT/'data/spectral_v2').glob('*.npz'))
    paths += [CONFIG,ROOT/'README_REPRODUCIBILITY.md', ROOT/'OBJECTIVE2_RESTMETA_Q1_STUDY_PLAN.md']
    paths += [ROOT/'scripts'/name for name in ['models.py','representations.py','experiment.py','common.py']]
    return sorted(p for p in paths if p.is_file())

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--fixtures-only',action='store_true'); args=parser.parse_args()
    checks=fixtures()
    if args.fixtures_only: print(checks); return
    OUT.mkdir(parents=True,exist_ok=True)
    spec={'status':'FROZEN_POST_HOC_DIAGNOSTICS_BEFORE_EXECUTION','date':'2026-10-13',
          'primary_config_sha':sha(CONFIG),'code_sha':sha(__file__),
          'scope':'All 95 frozen models, first190 held-out participants. No new fitting, tuning, outcomes on perturbed inputs, or changes to primary results.',
          'diagnostics':['saved-output replay','graph zero/identity/diagonal only','X=0 and X=11T/n',
             'one anatomy-disrupting ROI permutation and joint-coordinate invariance control',
             'same-person frozen representation swaps','layer/pooling energy and effective embedding rank',
             'local logit sensitivity to signed operator','spectral-matrix equivalence and power attenuation',
             'training-selection logs and existing site/diagnosis analyses'],
          'interpretation':'Descriptive mechanism probes; extreme ablations OOD; cannot identify a causal explanation for diagnostic performance. No alternative training authorized by this lock.',
          'primary_unchanged':{str(p.relative_to(ROOT)):sha(p) for p in protected_paths()}}
    if LOCK.exists():
        old=read(LOCK); assert old==spec,'Frozen diagnostic specification or protected input changed'
    else: write(LOCK,spec)
    rows=people(); roles=read(RUN/'participant_roles.json')
    fields={k:[] for k in ['w','u','threshold','T']}
    for row in rows:
        with np.load(cache_path(row,'first190')) as cache:
            for k in fields: fields[k].append(cache[k].copy())
    first={k:np.asarray(v) for k,v in fields.items()}
    fold_records=[]; parts=[]; replay_error=0.; equivariance_error=0.; spectral_error=0.
    with threadpool_limits(limits=2):
      for fold in roles['folds']:
        ix=np.array(fold['test']); sites=np.array([rows[i]['site'] for i in ix]); primary=sites!='S12'
        d={k:v[ix] for k,v in first.items()}; n=len(ix)
        refmeta=read(RUN/'main'/(fold['fold']+'_full.json')); budget=refmeta['budget']
        allops={}; allranks={}
        for arm in ARMS: allops[arm],allranks[arm],_=make_operators(**d,arm=arm,budget=budget)
        permutation=np.random.default_rng(seed('posthoc_anatomy_permutation',fold['fold'])).permutation(116)
        for arm in ARMS:
            name=fold['fold']+'_'+arm; meta=read(RUN/'main'/(name+'.json'))
            state=torch.load(RUN/'main'/(name+'_gnn.pt'),map_location=DEVICE,weights_only=False)
            assert state['config_sha']==sha(CONFIG) and meta['budget']==budget
            model=SpectralGNN().to(DEVICE); model.load_state_dict(state['state_dict']); model.eval()
            s=torch.tensor(allops[arm],device=DEVICE); z,h1,h2,h,a1,a2=evaluate(model,s)
            p=1/(1+np.exp(-z))
            with np.load(RUN/'main'/(name+'.npz')) as saved:
                mask=saved['context']=='first190'; np.testing.assert_array_equal(ix,saved['index'][mask])
                err=max(np.max(np.abs(p-saved['p_gnn'][mask])),np.max(np.abs(h-saved['embedding'][mask])))
                replay_error=max(replay_error,float(err)); assert err<2e-6
            arrays={'index':ix,'primary':primary,'site':sites,'p_original':p,'logit':z,'rank':allranks[arm]}
            for label,op in [('zero_graph',torch.zeros_like(s)),('identity_graph',torch.eye(116,device=DEVICE).expand_as(s)),
                 ('diagonal_only',torch.diag_embed(torch.diagonal(s,dim1=1,dim2=2))),
                 ('permuted_anatomy',s[:,permutation][:,:,permutation])]:
                zz=evaluate(model,op)[0]; arrays['p_'+label]=1/(1+np.exp(-zz))
                if label in ['zero_graph','identity_graph']: assert np.ptp(zz)<2e-6
            for label,w0 in [('zero_features',torch.zeros_like(model.w0)),
                  ('constant_features',model.w0.mean(0,keepdim=True).expand_as(model.w0))]:
                zz=evaluate(model,s,w0)[0]; arrays['p_'+label]=1/(1+np.exp(-zz))
            joint=evaluate(model,s[:,permutation][:,:,permutation],model.w0[permutation])[0]
            ee=float(np.max(np.abs(joint-z))); equivariance_error=max(equivariance_error,ee); assert ee<3e-6
            # Test derivative of logit, avoiding sigmoid saturation as an explanation.
            sd=s.detach().requires_grad_(True); zz=forward(model,sd)[0]
            grad=torch.autograd.grad(zz.sum(),sd)[0]; grad=(grad+grad.transpose(1,2))/2
            arrays['symmetric_logit_gradient_norm']=torch.linalg.vector_norm(grad,dim=(1,2)).detach().cpu().numpy()
            arrays['scaled_logit_gradient_norm']=arrays['symmetric_logit_gradient_norm']*np.sqrt(squared(allops[arm]))
            arrays['h1_node_variation_fraction']=ratio(squared(h1-h1.mean(1,keepdims=True)),squared(h1))
            arrays['h2_node_variation_fraction']=ratio(squared(h2-h2.mean(1,keepdims=True)),squared(h2))
            arrays['h1_zero_fraction']=(h1==0).mean((1,2)); arrays['h2_zero_fraction']=(h2==0).mean((1,2))
            arrays['h1_signal_norm']=np.sqrt(squared(a1)); arrays['h2_signal_norm']=np.sqrt(squared(a2))
            arrays['h1_bias_norm']=np.full(n,float(np.sqrt(116)*torch.linalg.vector_norm(model.b0).detach().cpu()))
            arrays['h2_bias_norm']=np.full(n,float(np.sqrt(116)*torch.linalg.vector_norm(model.b1).detach().cpu()))
            w0=model.w0.detach().cpu().numpy(); u=d['u']; w=d['w']/d['w'][:,0,None]
            spectral_input=u.transpose(0,2,1)@w0; power=np.sum(spectral_input**2,axis=2)*w**2
            mask=np.arange(116)[None,:]<allranks[arm][:,None]
            spec_a=(u*(w*mask)[:,None,:])@spectral_input
            se=float(np.max(np.abs(spec_a-a1))); spectral_error=max(spectral_error,se); assert se<3e-6
            arrays['h1_full_spectral_energy_retained']=ratio(np.sum(power*mask,axis=1),power.sum(1))
            arrays['full_first_shift_top1_energy_fraction']=ratio(w[:,0]**2,np.sum(w**2,axis=1))
            arrays['full_two_shift_top1_energy_fraction']=ratio(w[:,0]**4,np.sum(w**4,axis=1))
            arrays['lambda2_over_lambda1']=w[:,1]
            for target in ARMS:
                if target==arm: continue
                ss=torch.tensor(allops[target],device=DEVICE); tz,th1,th2,th,_,_=evaluate(model,ss)
                delta=allops[target]-allops[arm]
                arrays['p_swap_'+target]=1/(1+np.exp(-tz))
                arrays['operator_relative_change_'+target]=ratio(np.sqrt(squared(delta)),np.sqrt(squared(allops[arm])))
                arrays['h1_relative_change_'+target]=ratio(np.sqrt(squared(th1-h1)),np.sqrt(squared(h1)))
                arrays['h2_relative_change_'+target]=ratio(np.sqrt(squared(th2-h2)),np.sqrt(squared(h2)))
                arrays['pooling_retained_difference_energy_'+target]=ratio(116*squared(th-h),squared(th2-h2))
                arrays['embedding_relative_change_'+target]=ratio(np.sqrt(squared(th-h)),np.sqrt(squared(h)))
                arrays['local_logit_direction_'+target]=(grad*torch.tensor(delta,device=DEVICE)).sum((1,2)).detach().cpu().numpy()
            (OUT/'checkpoints').mkdir(exist_ok=True)
            np.savez_compressed(OUT/'checkpoints'/(name+'.npz'),**arrays)
            for scope,take in [('all',np.ones(n,dtype=bool)),('primary',primary)]:
                if not take.any(): continue
                fold_records.append({'fold':fold['fold'],'scheme':fold['scheme'],'arm':arm,'scope':scope,'N':int(take.sum()),
                   'embedding':effective_rank(h[take]),'h1_pooled_embedding':effective_rank(h1.mean(1)[take]),
                   'probability_sd':float(p[take].std()),'logit_sd':float(z[take].std()),
                   'selected_gnn':meta['selected_gnn'],'training_audit':meta['training_audit'],
                   'weights':{k:float(torch.linalg.vector_norm(v).detach().cpu()) for k,v in model.state_dict().items()}})
            parts.append((fold['scheme'],arm,arrays))
        print('diagnosed',fold['fold'],flush=True)
    summaries=[]
    for scheme in ['mixed','loso']:
      for arm in ARMS:
        chunks=[p[2] for p in parts if p[0]==scheme and p[1]==arm]
        take=np.concatenate([p['primary'] for p in chunks]); joined={k:np.concatenate([p[k] for p in chunks])[take] for k in chunks[0]}
        assert len(joined['index'])==1150 and len(set(joined['index']))==1150
        vals={k:stats(v) for k,v in joined.items() if k not in ['index','site','primary'] and not k.startswith('p_')}
        for k,v in joined.items():
            if k.startswith('p_') and k!='p_original':
                vals[k+'_absolute_change']=stats(np.abs(v-joined['p_original']))
                vals[k+'_flip_fraction']=float(np.mean((v>=.5)!=(joined['p_original']>=.5)))
        summaries.append({'scheme':scheme,'arm':arm,'scope':'primary','N':1150,'diagnostics':vals})
    existing=read(RUN/'final/latent_associations.json')['results']
    site_summary=[]
    for arm in ARMS:
        items=[r for r in existing if r['arm']==arm]
        site_summary.append({'arm':arm,'site_given_diagnosis_adjusted_R2':stats([r['adjusted_partial_R2_site_given_diagnosis'] for r in items]),
           'diagnosis_given_site_adjusted_R2':stats([r['adjusted_partial_R2_diagnosis_given_site'] for r in items]),
           'note':'Five correlated outer folds, descriptive only; reused primary analysis, no new probes fitted'})
    preserved=all(sha(ROOT/p)==h for p,h in spec['primary_unchanged'].items()); assert preserved
    write(OUT/'diagnostics.json',{'status':'COMPLETE_POST_HOC_INFERENCE_ONLY','summaries':summaries,'fold_records':fold_records,
        'existing_latent_summary':site_summary,'interpretation_limits':spec['interpretation']})
    write(OUT/'verification.json',{'passed':True,'fixtures':checks,'models':95,'new_model_fits':0,
        'primary_files_verified_unchanged':len(spec['primary_unchanged']),'all_primary_files_unchanged':preserved,
        'max_saved_output_replay_error':replay_error,'max_joint_permutation_logit_error':equivariance_error,
        'max_spectral_matrix_first_layer_error':spectral_error,'code_sha':sha(__file__),'lock_sha':sha(LOCK)})
    write(OUT/'artifact_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in [LOCK,ROOT/'scripts/audit_graph_use.py']+list(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
    print('POST-HOC GRAPH AUDIT COMPLETE; ALL PRIMARY FILES UNCHANGED',flush=True)

if __name__=='__main__': main()
