"""Fixed post-hoc analysis of the single spectral alternative; no selection."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import roc_auc_score
from common import ROOT,RUN,read,write,sha,seed
from representations import ARMS
from metrics import metrics,reliability,rho,paired_interval
from metric_uncertainty import bootstrap_site,weighted_auc
from analyze_results import calibration_job
from secondary_spectral import ALT,ACONFIG

OUT=ALT/'final'
MODELS=['original_gnn','spectral_gnn','logistic']

def collect(save=True):
    roles=read(RUN/'participant_roles.json'); sets={}; participant_rows=defaultdict(list)
    for scheme in ['mixed','loso']:
      for arm in ARMS:
        parts=defaultdict(list)
        for fold in roles['folds']:
            if fold['scheme']!=scheme: continue
            name=fold['fold']+'_'+arm; meta=read(ALT/'main'/(name+'.json'))
            assert meta['config_sha']==sha(ACONFIG) and meta['predictions_sha']==sha(ALT/'main'/(name+'.npz'))
            with np.load(ALT/'main'/(name+'.npz')) as new,np.load(RUN/'main'/(name+'.npz')) as old:
                for k in ['index','context','y','rank']: np.testing.assert_array_equal(new[k],old[k])
                for k in ['index','context','y']: parts[k].append(new[k])
                parts['spectral_gnn'].append(new['probability']); parts['original_gnn'].append(old['p_gnn']); parts['logistic'].append(old['p_lr'])
        joined={k:np.concatenate(v) for k,v in parts.items()}
        for context in ['first190','native230','last190']:
            mask=joined['context']==context; part={k:v[mask] for k,v in joined.items()}; order=np.argsort(part['index']); part={k:v[order] for k,v in part.items()}
            assert len(set(part['index']))==len(part['index'])==(1188 if context=='first190' else 667)
            part['site']=np.array([roles['participants'][i]['site'] for i in part['index']]); part['primary']=part['site']!='S12'
            sets[(scheme,arm,context)]=part
            for model in MODELS:
                for k,v in {'index':part['index'],'id':np.array([roles['participants'][i]['id'] for i in part['index']]),
                    'site':part['site'],'y':part['y'],'probability':part[model],'primary':part['primary'],
                    'scheme':np.repeat(scheme,len(order)),'arm':np.repeat(arm,len(order)),
                    'context':np.repeat(context,len(order)),'model':np.repeat(model,len(order))}.items(): participant_rows[k].append(v)
    OUT.mkdir(parents=True,exist_ok=True)
    if save: np.savez_compressed(OUT/'participant_predictions.npz',**{k:np.concatenate(v) for k,v in participant_rows.items()})
    return sets

def performance(sets):
    tables=[]
    for (scheme,arm,context),a in sets.items():
      for model in MODELS:
        for scope,take in [('primary',a['primary']),('S12_descriptive',~a['primary'])]:
            if not take.any(): continue
            y=a['y'][take]; p=a[model][take]; site=a['site'][take]; per_site={}
            for s in sorted(set(site)):
                ix=site==s; m=metrics(y[ix],p[ix]); supported=min(int(y[ix].sum()),int((1-y[ix]).sum()))>=50
                if not supported:
                    m['calibration']['slope']=None; m['calibration']['joint_intercept']=None; m['calibration']['status']='small_site_slope_suppressed'
                per_site[s]=m
            tables.append({'scheme':scheme,'arm':arm,'model':model,'context':context,'scope':scope,
               'pooled':metrics(y,p),'reliability':reliability(y,p),'sites':per_site,
               'site_macro':{k:float(np.mean([m[k] for m in per_site.values()])) for k in
                  ['auc','average_precision','balanced_accuracy','sensitivity','specificity','brier','log_loss']}})
    authoritative=read(RUN/'final/performance.json')
    original_lookup={(r['scheme'],r['arm'],r['model'],r['context'],r['scope']):r for r in authoritative}
    for r in tables:
        if r['model']=='spectral_gnn': continue
        key=(r['scheme'],r['arm'],'gnn' if r['model']=='original_gnn' else 'lr',r['context'],r['scope'])
        prior=original_lookup[key]
        for metric,value in r['site_macro'].items(): assert abs(value-prior['site_macro'][metric])<1e-12
    write(OUT/'performance.json',{'status':'EXPLORATORY_POST_HOC','original_metrics_exactly_reproduced':True,
        'unchanged_training_prior_reference':[r for r in authoritative if r['model']=='prior' and r['scope']=='primary'],
        'results':tables})
    lookup={(r['scheme'],r['arm'],r['model']):r for r in tables if r['context']=='first190' and r['scope']=='primary'}
    drops=[]
    for arm in ARMS:
      for model in MODELS:
        a=lookup[('mixed',arm,model)]['site_macro']; b=lookup[('loso',arm,model)]['site_macro']
        drops.append({'arm':arm,'model':model,'macro_auc_mixed_minus_loso':a['auc']-b['auc'],'macro_brier_loso_minus_mixed':b['brier']-a['brier']})
    write(OUT/'generalization_drop.json',{'scope':'Paired participants, differing training roles and sizes; not a causal acquisition effect','results':drops})
    print('performance/calibration point estimates complete',flush=True)

def uncertainty(sets):
    result=[]; contrasts=[]; methods=[(m,a) for m in MODELS for a in ARMS]
    for scheme in ['mixed','loso']:
        base=sets[(scheme,'full','first190')]; take=base['primary']; y=base['y'][take]; sites=base['site'][take]
        probs=np.stack([sets[(scheme,a,'first190')][m][take] for m,a in methods])
        rng=np.random.default_rng(seed('secondary_metric_intervals',scheme)); names=sorted(set(sites)); points=[]; draws=[]
        for s in names:
            mask=sites==s; draws.append(bootstrap_site(y[mask],probs[:,mask],rng))
            sitepoints=[]
            for p in probs[:,mask]:
                pp=np.clip(p,1e-6,1-1e-6)
                sitepoints.append([weighted_auc(y[mask],p,np.ones((1,mask.sum())))[0],
                    np.mean((p-y[mask])**2),np.mean(-y[mask]*np.log(pp)-(1-y[mask])*np.log1p(-pp))])
            points.append(sitepoints)
        p=np.mean(points,axis=0); d=np.stack(draws,axis=1); within=d.mean(1)
        ss=rng.integers(len(names),size=(2000,len(names))); rr=rng.integers(2000,size=ss.shape); cluster=d[rr,ss].mean(1)
        def pack(point,w,c):
            return {metric:{'estimate':float(point[j]),'within_site_low':float(np.quantile(w[:,j],.025)),
               'within_site_high':float(np.quantile(w[:,j],.975)),'cluster_low':float(np.quantile(c[:,j],.025)),
               'cluster_high':float(np.quantile(c[:,j],.975))} for j,metric in enumerate(['auc','brier','log_loss'])}
        for j,(model,arm) in enumerate(methods): result.append({'scheme':scheme,'model':model,'arm':arm,'metrics':pack(p[j],within[:,j],cluster[:,j])})
        pairs=[('spectral_minus_original_'+a,('spectral_gnn',a),('original_gnn',a)) for a in ARMS]
        pairs += [('spectral_temporal_minus_'+a,('spectral_gnn','temporal'),('spectral_gnn',a)) for a in ['budget','full','mp','fixed']]
        for name,a,b in pairs:
            j=methods.index(a); k=methods.index(b)
            contrasts.append({'scheme':scheme,'contrast':name,'metrics':pack(p[j]-p[k],within[:,j]-within[:,k],cluster[:,j]-cluster[:,k])})
    write(OUT/'paired_metric_uncertainty.json',{'scope':'2000 paired conditional fitted-prediction bootstraps; no training uncertainty, no confirmatory or algorithm-superiority claim','results':result,'contrasts':contrasts})
    print('paired metric intervals complete',flush=True)

def stability(sets):
    summary=[]; participant=defaultdict(list); contrasts=[]
    for scheme in ['mixed','loso']:
      for arm in ARMS:
        a=sets[(scheme,arm,'first190')]; assert np.array_equal(a['index'],np.arange(1188))
        for context in ['native230','last190']:
            b=sets[(scheme,arm,context)]; ix=b['index']; take=b['primary']; y=b['y'][take]; sites=b['site'][take]
            for model in MODELS:
                p0=a[model][ix][take]; p1=b[model][take]; dp=np.abs(p1-p0); flips=(p0>=.5)!=(p1>=.5)
                summary.append({'scheme':scheme,'arm':arm,'context':context,'model':model,'N':len(y),
                    'mean_absolute_probability_change':float(dp.mean()),'decision_flip_fraction':float(flips.mean()),
                    'rank_correlation':rho(p0,p1),'auc_change':float(roc_auc_score(y,p1)-roc_auc_score(y,p0)),
                    'brier_change':float(np.mean((p1-y)**2-(p0-y)**2))})
                vals={'index':ix[take],'site':sites,'y':y,'p_first190':p0,'p_context':p1,'absolute_change':dp,'decision_flip':flips,
                    'scheme':np.repeat(scheme,len(y)),'arm':np.repeat(arm,len(y)),'context':np.repeat(context,len(y)),'model':np.repeat(model,len(y))}
                for k,v in vals.items():participant[k].append(v)
    # Paired finite/position sensitivity contrasts under the same frozen fits.
    for context in ['native230','last190']:
        b=sets[('loso','full',context)]; take=b['primary']; ix=b['index'][take]; y=b['y'][take]; sites=b['site'][take]
        def changes(model,arm):
            p0=sets[('loso',arm,'first190')][model][ix]; p1=sets[('loso',arm,context)][model][take]
            return np.abs(p1-p0),((p0>=.5)!=(p1>=.5)).astype(float)
        pairs=[('spectral_temporal_minus_budget',('spectral_gnn','temporal'),('spectral_gnn','budget'))]
        pairs += [('spectral_minus_original_'+a,('spectral_gnn',a),('original_gnn',a)) for a in ARMS]
        for name,a,b in pairs:
            aa=changes(*a); bb=changes(*b)
            for j,label in enumerate(['absolute_probability_change','decision_flip_fraction']):
                d=aa[j]-bb[j]
                contrasts.append({'context':context,'contrast':name,'endpoint':label,'N':len(y),'estimate':float(d.mean()),
                     'conditional_intervals':paired_interval(d,y,sites,seed('secondary_stability',context,name,label))})
            print('stability intervals',context,name,flush=True)
    write(OUT/'stability.json',{'scope':'First/last190 overlap 150 volumes; whole-run preprocessing before segmentation. Conditional replay, not acquired-duration or test-retest evidence','results':summary,'contrasts':contrasts})
    np.savez_compressed(OUT/'participant_stability.npz',**{k:np.concatenate(v) for k,v in participant.items()})

def calibration_intervals(sets):
    tasks=[]
    for arm in ARMS:
        a=sets[('loso',arm,'first190')]
        for scope in ['primary','S8','S15','S21','S25']:
            take=a['primary'] if scope=='primary' else a['site']==scope
            tasks.append((arm,'spectral_gnn',scope,a['y'][take],a['spectral_gnn'][take],a['site'][take]))
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(calibration_job,t) for t in tasks]):
            r=future.result(); results.append(r); print('calibration interval',r['arm'],r['scope'],flush=True)
    write(OUT/'calibration_intervals.json',{'scope':'Conditional within-site diagnosis-stratified intervals; successful/failed fits shown; original calibration stays in primary results','results':results})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['base','uncertainty','calibration'],default='base'); args=parser.parse_args()
    sets=collect(save=args.stage=='base')
    if args.stage=='base': performance(sets); stability(sets)
    elif args.stage=='uncertainty': uncertainty(sets)
    else: calibration_intervals(sets)

if __name__=='__main__': main()
