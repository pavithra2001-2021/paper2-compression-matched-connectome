"""Efficient paired stratified bootstrap for fixed held-out probability metrics.

Weights are bootstrap multiplicities, preserving score ties exactly. All models
share multiplicities. Cluster draws preserve paired methods and draw a fresh
within-site replicate for every sampled site occurrence.
"""
import numpy as np
from common import RUN,write,seed
from analyze_results import collect,OUT
from representations import ARMS

def weighted_auc(y,p,weights):
    order=np.argsort(p,kind='stable'); yy=y[order]; pp=p[order]; ww=weights[:,order]
    starts=np.r_[0,np.flatnonzero(np.diff(pp)!=0)+1]
    pos=np.add.reduceat(ww*yy,starts,axis=1); neg=np.add.reduceat(ww*(1-yy),starts,axis=1)
    return np.sum(pos*(np.cumsum(neg,axis=1)-.5*neg),axis=1)/(pos.sum(axis=1)*neg.sum(axis=1))

def bootstrap_site(y,probabilities,rng,B=2000):
    weights=np.zeros((B,len(y)),dtype=float)
    for label in [0,1]:
        ix=np.flatnonzero(y==label); weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=B)
    result=[]
    for p in probabilities:
        auc=weighted_auc(y,p,weights)
        brier=weights@((p-y)**2)/len(y)
        pp=np.clip(p,1e-6,1-1e-6); ll=weights@(-y*np.log(pp)-(1-y)*np.log1p(-pp))/len(y)
        result.append(np.stack([auc,brier,ll],axis=1))
    return np.stack(result,axis=1)

def main():
    sets,_=collect(save=False); B=2000; output=[]; paired=[]
    methods=[(model,arm) for model in ['gnn','lr'] for arm in ARMS]+[('prior','full')]
    for scheme in ['mixed','loso']:
        base=sets[(scheme,'full','first190')]; take=base['primary']; y=base['y'][take]; sites=base['site'][take]
        ps=np.stack([sets[(scheme,arm,'first190')]['p_'+model][take] for model,arm in methods])
        rng=np.random.default_rng(seed('metric_bootstrap',scheme)); site_draws=[]; site_points=[]; site_names=sorted(set(sites),key=lambda s:int(s[1:]))
        for s in site_names:
            ix=sites==s; draws=bootstrap_site(y[ix],ps[:,ix],rng,B); site_draws.append(draws)
            # Direct weighted metrics also test the tie-safe AUC primitive.
            ones=np.ones((1,ix.sum())); points=[]
            for p in ps[:,ix]:
                pp=np.clip(p,1e-6,1-1e-6)
                points.append([weighted_auc(y[ix],p,ones)[0],np.mean((p-y[ix])**2),np.mean(-y[ix]*np.log(pp)-(1-y[ix])*np.log1p(-pp))])
            site_points.append(points)
            for j,(model,arm) in enumerate(methods):
                output.append({'scheme':scheme,'site':s,'model':model,'arm':arm,'N':int(ix.sum()),
                  'metrics':{name:{'estimate':float(points[j][k]),'low':float(np.quantile(draws[:,j,k],.025)),'high':float(np.quantile(draws[:,j,k],.975))} for k,name in enumerate(['auc','brier','log_loss'])}})
        sd=np.stack(site_draws,axis=1); points=np.mean(np.asarray(site_points),axis=0)
        within=sd.mean(axis=1)
        sampled_sites=rng.integers(len(site_names),size=(B,len(site_names)))
        sampled_replicates=rng.integers(B,size=sampled_sites.shape)
        cluster=sd[sampled_replicates,sampled_sites].mean(axis=1)
        for j,(model,arm) in enumerate(methods):
            output.append({'scheme':scheme,'site':'primary_macro','model':model,'arm':arm,'N':len(y),
              'metrics':{name:{'estimate':float(points[j,k]),'within_site_low':float(np.quantile(within[:,j,k],.025)),
               'within_site_high':float(np.quantile(within[:,j,k],.975)),'cluster_low':float(np.quantile(cluster[:,j,k],.025)),
               'cluster_high':float(np.quantile(cluster[:,j,k],.975))} for k,name in enumerate(['auc','brier','log_loss'])}})
        contrasts=[]
        for model in ['gnn','lr']:
            for ref in ['full','mp','budget','fixed']:
                contrasts.append((f'{model}_temporal_minus_{ref}',methods.index((model,'temporal')),methods.index((model,ref))))
        for arm in ARMS: contrasts.append((f'{arm}_gnn_minus_lr',methods.index(('gnn',arm)),methods.index(('lr',arm))))
        for name,j,k in contrasts:
            paired.append({'scheme':scheme,'contrast':name,'metrics':{m:{'estimate':float(points[j,t]-points[k,t]),
               'within_site_low':float(np.quantile(within[:,j,t]-within[:,k,t],.025)),
               'within_site_high':float(np.quantile(within[:,j,t]-within[:,k,t],.975)),
               'cluster_low':float(np.quantile(cluster[:,j,t]-cluster[:,k,t],.025)),
               'cluster_high':float(np.quantile(cluster[:,j,t]-cluster[:,k,t],.975))} for t,m in enumerate(['auc','brier','log_loss'])}})
    write(OUT/'metric_uncertainty.json',{'scope':'2000 paired fixed-prediction resamples, within site x diagnosis and separate site-cluster/within resampling; omits training variability','results':output,'paired_comparisons':paired})
    print('Conditional metric intervals complete')

if __name__=='__main__': main()
