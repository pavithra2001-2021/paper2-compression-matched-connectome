"""Held-out descriptive calibration and conditional paired uncertainty."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,confusion_matrix

def calibration(y,p,with_slope=True):
    y=np.asarray(y); z=logit(np.clip(p,1e-6,1-1e-6))
    if len(np.unique(y))<2: return {'status':'single_class','offset':None,'slope':None,'joint_intercept':None}
    if np.std(z)<1e-8: return {'status':'constant_predictions','offset':float(logit(y.mean())-z.mean()),'slope':None,'joint_intercept':None}
    def solve(X,offset):
        def fun(b):
            v=X@b+offset
            return np.logaddexp(0,v).sum()-y@v, X.T@(expit(v)-y)
        fit=minimize(fun,np.zeros(X.shape[1]),jac=True,method='BFGS',options={'gtol':1e-6,'maxiter':300})
        grad=np.max(np.abs(fun(fit.x)[1]))
        prob=expit(X@fit.x+offset); info=(X.T*(prob*(1-prob)))@X
        valid=(fit.success or grad<1e-4) and np.isfinite(fit.x).all() and np.max(np.abs(fit.x))<100 and np.linalg.cond(info)<1e10
        return fit.x,valid
    a,ok=solve(np.ones((len(y),1)),z)
    if not with_slope: return {'status':'ok' if ok else 'unstable','offset':float(a[0]) if ok else None,'slope':None,'joint_intercept':None}
    beta,ok2=solve(np.c_[np.ones(len(y)),z],np.zeros(len(y)))
    return {'status':'ok' if ok and ok2 else 'unstable_or_separation','offset':float(a[0]) if ok else None,
            'slope':float(beta[1]) if ok2 else None,'joint_intercept':float(beta[0]) if ok2 else None}

def metrics(y,p,include_calibration=True):
    y=np.asarray(y); p=np.asarray(p); hard=p>=.5
    tn,fp,fn,tp=confusion_matrix(y,hard,labels=[0,1]).ravel()
    pp=np.clip(p,1e-6,1-1e-6)
    out={'N':len(y),'MDD':int(y.sum()),'HC':int((1-y).sum()),'auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
         'average_precision':float(average_precision_score(y,p)) if y.sum()>0 else None,
         'balanced_accuracy':float(balanced_accuracy_score(y,hard)),
         'sensitivity':float(tp/(tp+fn)) if tp+fn else None,'specificity':float(tn/(tn+fp)) if tn+fp else None,
         'brier':float(np.mean((p-y)**2)),'log_loss':float(-np.mean(y*np.log(pp)+(1-y)*np.log1p(-pp))),
         'probability_mean':float(p.mean()),'probability_sd':float(p.std()),'prevalence':float(y.mean())}
    if include_calibration: out['calibration']=calibration(y,p)
    return out

def rho(a,b):
    if np.std(a)<1e-12 or np.std(b)<1e-12: return None
    r=float(spearmanr(a,b).statistic)
    return r if np.isfinite(r) else None

def reliability(y,p):
    bins=np.minimum((np.clip(p,0,1)*5).astype(int),4)
    return [{'low':j/5,'high':(j+1)/5,'N':int((bins==j).sum()),
             'mean_prediction':float(np.mean(p[bins==j])) if np.any(bins==j) else None,
             'observed_fraction':float(np.mean(y[bins==j])) if np.any(bins==j) else None} for j in range(5)]

def sample_indices(y,sites,rng,clusters=False):
    keys=sorted(set(sites)); chosen=rng.choice(keys,len(keys),replace=True) if clusters else keys
    groups=[]
    for s in chosen:
        parts=[]
        for label in (0,1):
            ix=np.flatnonzero((sites==s)&(y==label))
            if len(ix): parts.append(rng.choice(ix,len(ix),replace=True))
        groups.append(np.concatenate(parts))
    return groups

def paired_interval(values,y,sites,random_seed,B=2000,macro=False):
    """values is per-person paired loss/stability difference; never independent arms."""
    rng=np.random.default_rng(random_seed)
    out={}
    for kind,clusters in [('within_site_diagnosis',False),('site_cluster_and_within',True)]:
        draws=[]
        for _ in range(B):
            groups=sample_indices(y,sites,rng,clusters)
            draws.append(float(np.mean([values[ix].mean() for ix in groups])) if macro else float(np.mean(values[np.concatenate(groups)])))
        out[kind]={'low':float(np.quantile(draws,.025)),'high':float(np.quantile(draws,.975)),'replicates':B}
    return out
