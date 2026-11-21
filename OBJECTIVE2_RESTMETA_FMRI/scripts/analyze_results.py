"""Final traceable results; descriptive/conditional scope, no outcome-based tuning."""
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from common import ROOT,RUN,CONFIG,read,write,sha,seed,people
from representations import ARMS
from metrics import metrics,reliability,rho,calibration,paired_interval,sample_indices

OUT=RUN/'final'

def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)

def collect(save=True):
    roles=read(RUN/'participant_roles.json'); rows=roles['participants']
    sets={}; long=[]; tables=[]
    for scheme in ['mixed','loso']:
        for arm in ARMS:
            chunks=defaultdict(list)
            for fold in roles['folds']:
                if fold['scheme']!=scheme: continue
                path=RUN/'main'/(fold['fold']+'_'+arm+'.npz')
                status=read(path.with_suffix('.json'))
                assert sha(path)==status['predictions_sha']
                with np.load(path) as z:
                    for k in z.files: chunks[k].append(z[k])
            joined={k:np.concatenate(v) for k,v in chunks.items()}
            for context in ['first190','native230','last190']:
                mask=joined['context']==context
                part={k:v[mask] for k,v in joined.items()}
                order=np.argsort(part['index']); part={k:v[order] for k,v in part.items()}
                assert len(set(part['index']))==len(part['index'])
                assert len(part['index'])==(1188 if context=='first190' else 667)
                part['site']=np.array([rows[i]['site'] for i in part['index']]); part['primary']=part['site']!='S12'
                sets[(scheme,arm,context)]=part
                for model in ['gnn','lr']+(['prior'] if arm=='full' else []):
                    p=part['p_'+model]; y=part['y']; sites=part['site']; pri=part['primary']
                    for scope,choose in [('primary',pri),('all',np.ones(len(y),dtype=bool)),('S12_descriptive',sites=='S12')]:
                        if not choose.any(): continue
                        m=metrics(y[choose],p[choose]); sr={}
                        for s in sorted(set(sites[choose]),key=lambda s:int(s[1:])):
                            ix=choose&(sites==s); sr[s]=metrics(y[ix],p[ix])
                            sr[s]['calibration_slope_supported']=int(y[ix].sum())>=50 and int((1-y[ix]).sum())>=50
                            # Small-site slope is suppressed rather than presented as precise.
                            if not sr[s]['calibration_slope_supported']:
                                sr[s]['calibration']['slope']=None; sr[s]['calibration']['joint_intercept']=None
                                sr[s]['calibration']['status']='small_site_slope_suppressed'
                        key=f'{scheme}/{arm if model!="prior" else "prior"}/{model}/{context}/{scope}'
                        result={'scheme':scheme,'arm':arm if model!='prior' else 'prior','model':model,'context':context,'scope':scope,
                                'pooled':m,'site_macro':{k:float(np.mean([v[k] for v in sr.values()])) for k in ['auc','average_precision','balanced_accuracy','sensitivity','specificity','brier','log_loss']},
                                'reliability':reliability(y[choose],p[choose]),'sites':sr}
                        tables.append(result)
                    for j,i in enumerate(part['index']):
                        long.append({'id':rows[i]['id'],'index':int(i),'diagnosis':int(y[j]),'site':sites[j],'primary':bool(pri[j]),
                          'scheme':scheme,'arm':arm if model!='prior' else 'prior','model':model,'context':context,'probability':float(p[j]),
                          'rank':int(part['rank'][j]) if model!='prior' else None,'operator_norm':float(part['operator_norm'][j]) if model!='prior' else None,
                          'operator_distance_from_first190':float(part['operator_distance'][j]) if model!='prior' else None,
                          'raw_distance_from_first190':float(part['raw_distance'][j]) if model!='prior' else None})
    if save:
        write(OUT/'performance.json',tables); save_csv(OUT/'participant_predictions.csv',long)
    flat=[]
    for row in tables:
        flat.append({k:row[k] for k in ['scheme','arm','model','context','scope']} | row['site_macro'] |
                    {'N':row['pooled']['N'],'pooled_auc':row['pooled']['auc'],'pooled_brier':row['pooled']['brier'],
                     'pooled_calibration_offset':row['pooled']['calibration']['offset'],'pooled_calibration_slope':row['pooled']['calibration']['slope']})
    if save: save_csv(OUT/'performance_summary.csv',flat)
    return sets,tables

def stability(sets):
    output=[]; person_rows=[]
    for scheme in ['mixed','loso']:
        for arm in ARMS:
            a=sets[(scheme,arm,'first190')]
            for context in ['native230','last190']:
                b=sets[(scheme,arm,context)]; ix=b['index']; assert np.array_equal(a['index'],np.arange(1188))
                for model in ['gnn','lr']+(['prior'] if arm=='full' else []):
                    p0=a['p_'+model][ix]; p1=b['p_'+model]; y=b['y']; sites=b['site']
                    dp=np.abs(p1-p0); dk=b['rank']-a['rank'][ix]
                    for scope,take in [('primary',b['primary']),('all',np.ones(len(ix),bool)),('S12_descriptive',sites=='S12')]:
                        rr={'scheme':scheme,'arm':arm if model!='prior' else 'prior','model':model,'context':context,'scope':scope,'N':int(take.sum()),
                            'mean_abs_probability_change':float(dp[take].mean()),'decision_flip_fraction':float(np.mean((p1[take]>=.5)!=(p0[take]>=.5))),
                            'prediction_rank_correlation':rho(p0[take],p1[take]),
                            'pooled_auc_change':float(roc_auc_score(y[take],p1[take])-roc_auc_score(y[take],p0[take])),
                            'brier_change':float(np.mean((p1[take]-y[take])**2-(p0[take]-y[take])**2)),
                            'mean_operator_distance':float(b['operator_distance'][take].mean()),'mean_raw_distance':float(b['raw_distance'][take].mean()),
                            'mean_embedding_distance':float(b['embedding_distance'][take].mean()) if model=='gnn' else None,
                            'rank_change_mean':float(dk[take].mean()),'rank_change_abs_probability_rho':rho(np.abs(dk[take]),dp[take]),
                            'operator_distance_abs_probability_rho':rho(b['operator_distance'][take],dp[take]),
                            'sites':{s:{'N':int(np.sum(take&(sites==s))),'mean_abs_probability_change':float(dp[take&(sites==s)].mean()),
                                       'operator_probability_rho':rho(b['operator_distance'][take&(sites==s)],dp[take&(sites==s)])} for s in sorted(set(sites[take]))}}
                        rr['site_macro_auc_change']=float(np.mean([roc_auc_score(y[take&(sites==s)],p1[take&(sites==s)])-roc_auc_score(y[take&(sites==s)],p0[take&(sites==s)]) for s in sorted(set(sites[take]))]))
                        rr['site_macro_brier_change']=float(np.mean([np.mean((p1[take&(sites==s)]-y[take&(sites==s)])**2-(p0[take&(sites==s)]-y[take&(sites==s)])**2) for s in sorted(set(sites[take]))]))
                        if model!='prior':
                            for field in ['operator_relative_distance','raw_relative_distance']:
                                vals=b[field][take]; valid=np.isfinite(vals)
                                rr['mean_'+field]=float(vals[valid].mean()) if valid.any() else None
                                rr[field+'_undefined_N']=int((~valid).sum())
                            rr['mean_raw_norm_context']=float(b['raw_norm'][take].mean())
                            rr['mean_operator_norm_context']=float(b['operator_norm'][take].mean())
                            rr['mean_raw_norm_first190']=float(a['raw_norm'][ix][take].mean())
                            rr['mean_operator_norm_first190']=float(a['operator_norm'][ix][take].mean())
                        if model=='gnn': rr['embedding_distance_abs_probability_rho']=rho(b['embedding_distance'][take],dp[take])
                        output.append(rr)
                        if model=='prior':
                            for key in ['mean_operator_distance','mean_raw_distance','rank_change_mean','rank_change_abs_probability_rho','operator_distance_abs_probability_rho']:
                                rr[key]=None
                    for j,i in enumerate(ix):
                        person_rows.append({'index':int(i),'site':sites[j],'primary':bool(b['primary'][j]),'scheme':scheme,'arm':arm if model!='prior' else 'prior','model':model,'context':context,
                           'p_first190':float(p0[j]),'p_context':float(p1[j]),'abs_probability_change':float(dp[j]),'rank_change':int(dk[j]),
                           'decision_flip':bool((p0[j]>=.5)!=(p1[j]>=.5)),'operator_distance':float(b['operator_distance'][j]),'raw_distance':float(b['raw_distance'][j]),
                           'operator_relative_distance':float(b['operator_relative_distance'][j]),'embedding_distance':float(b['embedding_distance'][j]) if model=='gnn' else None})
                        if model=='prior':
                            for key in ['rank_change','operator_distance','raw_distance','operator_relative_distance']:
                                person_rows[-1][key]=None
    write(OUT/'stability.json',output); save_csv(OUT/'participant_stability.csv',person_rows)
    save_csv(OUT/'stability_summary.csv',[{k:v for k,v in r.items() if k!='sites'} for r in output])
    return output

def contrasts(sets,tables):
    records=[]
    for model in ['gnn','lr']:
        for arm in ['budget','full','mp','fixed']:
            for endpoint in ['site_macro_brier','finite_probability_change']:
                base=sets[('loso','temporal','first190')]; ref=sets[('loso',arm,'first190')]
                if endpoint=='site_macro_brier':
                    take=base['primary']; y=base['y'][take]; sites=base['site'][take]
                    d=((base['p_'+model]-base['y'])**2-(ref['p_'+model]-ref['y'])**2)[take]
                    macro=True
                else:
                    a=sets[('loso','temporal','native230')]; b=sets[('loso',arm,'native230')]
                    take=a['primary']; ix=a['index']; y=a['y'][take]; sites=a['site'][take]
                    d=(np.abs(a['p_'+model]-base['p_'+model][ix])-np.abs(b['p_'+model]-ref['p_'+model][ix]))[take]; macro=False
                estimate=float(np.mean([d[sites==s].mean() for s in set(sites)])) if macro else float(d.mean())
                ci=paired_interval(d,y,sites,seed('contrast',model,arm,endpoint),macro=macro)
                records.append({'model':model,'contrast':'temporal_minus_'+arm,'endpoint':endpoint,'primary':model=='gnn' and arm=='budget',
                       'estimate':estimate,'lower_is_better_for_temporal':True,'N':len(y),'sites':len(set(sites)),'conditional_intervals':ci})
                print('interval',model,arm,endpoint,flush=True)
    # Paired GNN-versus-logistic proper-score comparisons, conditional only.
    for arm in ARMS:
        a=sets[('loso',arm,'first190')]; take=a['primary']; y=a['y'][take]; sites=a['site'][take]
        d=((a['p_gnn']-a['y'])**2-(a['p_lr']-a['y'])**2)[take]
        records.append({'contrast':'gnn_minus_lr','arm':arm,'endpoint':'site_macro_brier','primary':False,
          'estimate':float(np.mean([d[sites==s].mean() for s in set(sites)])),
          'conditional_intervals':paired_interval(d,y,sites,seed('model_contrast',arm),macro=True)})
    write(OUT/'paired_contrasts.json',{'scope':'Conditional on fitted predictions; no training/selection uncertainty; no algorithm superiority or equivalence claim','results':records})
    lookup={(r['scheme'],r['arm'],r['model']):r for r in tables if r['context']=='first190' and r['scope']=='primary'}
    transfer=[]
    for arm in ARMS:
        for model in ['gnn','lr']:
            mixed=lookup[('mixed',arm,model)]; loso=lookup[('loso',arm,model)]
            transfer.append({'arm':arm,'model':model,'N':1150,'macro_auc_mixed_minus_loso':mixed['site_macro']['auc']-loso['site_macro']['auc'],
                 'macro_brier_loso_minus_mixed':loso['site_macro']['brier']-mixed['site_macro']['brier'],
                 'interpretation':'Validation-procedure contrast; training size and site composition also differ, not a pure causal site-shift effect'})
    write(OUT/'generalization_drop.json',transfer)

def embedding_analysis():
    roles=read(RUN/'participant_roles.json'); sites=np.array([r['site'] for r in roles['participants']]); ys=np.array([r['y'] for r in roles['participants']]); out=[]
    for fold in roles['folds']:
        if fold['scheme']!='mixed': continue
        for arm in ARMS:
            with np.load(RUN/'main'/(fold['fold']+'_'+arm+'.npz')) as f:
                take=f['context']=='first190'; ix=f['index'][take]; h=f['embedding'][take].astype(float)
            site=sites[ix]; y=ys[ix]; rng=np.random.default_rng(seed('latent_probe',fold['fold'])); train=[]; test=[]
            for s in sorted(set(site)):
                for label in [0,1]:
                    cell=np.flatnonzero((site==s)&(y==label)); rng.shuffle(cell); cut=max(1,len(cell)//2)
                    train.extend(cell[:cut]); test.extend(cell[cut:])
            train=np.array(train); test=np.array(test); scaler=StandardScaler().fit(h[train]); xtr=scaler.transform(h[train]); xte=scaler.transform(h[test])
            scores={}
            for target,label in [(site,'site'),(y,'diagnosis')]:
                model=LogisticRegression(C=1,max_iter=3000).fit(xtr,target[train]); pred=model.predict(xte)
                scores[label+'_probe_balanced_accuracy']=float(balanced_accuracy_score(target[test],pred))
                scores[label+'_probe_classes']=len(set(target[train]))
                chance=1/len(set(target[train]))
                scores[label+'_probe_chance_corrected_balanced_accuracy']=(scores[label+'_probe_balanced_accuracy']-chance)/(1-chance)
                if label=='diagnosis': scores['diagnosis_probe_auc']=float(roc_auc_score(target[test],model.predict_proba(xte)[:,1]))
            onehot=np.stack([site==s for s in sorted(set(site))[1:]],axis=1).astype(float)
            base=np.ones((len(y),1)); siteX=np.c_[base,onehot]; dxX=np.c_[base,y]; full=np.c_[siteX,y]
            def rss(X):
                residual=h-X@np.linalg.lstsq(X,h,rcond=None)[0]
                return float(np.sum(residual**2)),len(y)-np.linalg.matrix_rank(X)
            rss_full,df_full=rss(full)
            for X,label in [(siteX,'diagnosis_given_site'),(dxX,'site_given_diagnosis')]:
                r,df=rss(X)
                scores['partial_R2_'+label]=float(1-rss_full/r) if r>1e-16 else None
                scores['adjusted_partial_R2_'+label]=float(1-(rss_full/df_full)/(r/df)) if r>1e-16 else None
            out.append({'fold':fold['fold'],'arm':arm,'N':len(y),'probe_train_indices':ix[train].tolist(),'probe_test_indices':ix[test].tolist(),**scores})
    write(OUT/'latent_associations.json',{'scope':'Secondary frozen-model within-fold test-embedding analysis; no coordinate pooling across outer models; no causal or target-adjustment claims','results':out})

def calibration_job(task):
    from threadpoolctl import threadpool_limits
    arm,model,scope,y,p,sites=task
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(seed('calibration_bootstrap',arm,model,scope)); values=[]; failed=0
        for _ in range(2000):
            ix=np.concatenate(sample_indices(y,sites,rng))
            c=calibration(y[ix],p[ix])
            if c['status']=='ok': values.append([c['offset'],c['joint_intercept'],c['slope']])
            else: failed+=1
        ci={}
        if values:
            limits=np.quantile(np.asarray(values),[.025,.975],axis=0)
            ci={k:{'low':float(limits[0,j]),'high':float(limits[1,j])} for j,k in enumerate(['offset','joint_intercept','slope'])}
        return {'arm':arm,'model':model,'scope':scope,'N':len(y),'replicates':2000,'failed':failed,'successful':len(values),'intervals':ci}

def calibration_intervals(sets):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    out=[]; tasks=[]
    for arm in ARMS:
        a=sets[('loso',arm,'first190')]
        for model in ['gnn','lr']:
            for scope in ['primary','S8','S15','S21','S25']:
                take=a['primary'] if scope=='primary' else a['site']==scope
                y=a['y'][take]; p=a['p_'+model][take]; sites=a['site'][take]
                tasks.append((arm,model,scope,y,p,sites))
    with ProcessPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(calibration_job,t) for t in tasks]):
            r=future.result(); out.append(r); print('calibration interval',r['arm'],r['model'],r['scope'],flush=True)
    out.sort(key=lambda r:(r['arm'],r['model'],r['scope']))
    write(OUT/'calibration_intervals.json',{'scope':'Within-site diagnosis-stratified conditional fixed-prediction percentile bootstrap; successful/failed fits reported; no prediction recalibration','results':out})

def permutation_summary():
    rows=read(RUN/'participant_roles.json')['participants']; sites=np.array([r['site'] for r in rows]); primary=sites!='S12'; output=[]
    for perm in range(1,11):
        folder=RUN/f'permutations/p{perm:02d}'; labels=read(folder/'labels.json')['labels']
        for arm in ARMS[:4]:
            agg=defaultdict(list)
            for j in range(5):
                with np.load(folder/f'mixed_{j}_{arm}.npz') as f:
                    for k in ['index','y','p_gnn','p_lr']: agg[k].extend(f[k])
            order=np.argsort(agg['index']); y=np.array(agg['y'])[order]; assert np.array_equal(y,labels)
            for model in ['gnn','lr']:
                p=np.array(agg['p_'+model])[order]
                values=[metrics(y[sites==s],p[sites==s],False) for s in sorted(set(sites[primary]))]
                output.append({'permutation':perm,'arm':arm,'model':model,'primary_macro_auc':float(np.mean([v['auc'] for v in values])),
                               'primary_macro_brier':float(np.mean([v['brier'] for v in values]))})
    write(OUT/'permutation_sanity.json',{'scope':'Ten nested mixed-site within-site label shuffles; sanity check only, not a hypothesis-test p-value','results':output})

def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['base','calibration','permutations'],default='base'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=='permutations': permutation_summary(); return
    sets,tables=collect(save=args.stage=='base')
    if args.stage=='calibration': calibration_intervals(sets); return
    stability(sets); contrasts(sets,tables); embedding_analysis()
    print('Base final analyses complete')

if __name__=='__main__': main()
