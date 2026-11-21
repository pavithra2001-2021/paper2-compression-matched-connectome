"""Frozen ABIDE summaries, conditional inference, figures and complete replay."""
import warnings
import numpy as np
import joblib
import torch
from scipy.stats import spearmanr
import replication as r

def measure(y,p):
    result=r.metrics(y,p);result['ASD']=result.pop('MDD');return result

def collect():
    cfg,roles=r.check_lock(); rows=roles['participants'];sets={};long=[]
    for k in [0,1,2]:
        for arm in r.ARMS if k==0 else ['temporal','budget']:
            chunks={}
            for fold in roles['folds']:
                folder=r.HERE/'main' if k==0 else r.HERE/'seed_sensitivity'/('seed'+str(k))
                path=folder/(fold['fold']+'_'+arm+'.npz')
                with np.load(path) as z:
                    for key in z.files:chunks.setdefault(key,[]).append(z[key].copy())
            joined={key:np.concatenate(v) for key,v in chunks.items()}
            for ctx in ['first190','native230','last190']:
                a={key:v[joined['context']==ctx] for key,v in joined.items()};order=np.argsort(a['index']);a={key:v[order] for key,v in a.items()}
                expected=np.array([i for i,x in enumerate(rows) if ctx=='first190' or x['finite230']])
                np.testing.assert_array_equal(a['index'],expected)
                a['site']=np.array([rows[i]['site'] for i in a['index']]);sets[k,arm,ctx]=a
                learners=['gnn','lr']+(['prior'] if arm=='full' else []) if k==0 else ['gnn']
                for learner in learners:
                    for j,i in enumerate(a['index']):
                        p=float(a['p_'+learner][j]);assert np.isfinite(p) and 0<=p<=1
                        long.append({'id':rows[i]['id'],'index':int(i),'center':rows[i]['site'],'site_code':rows[i]['site_code'],
                         'ASD':int(a['y'][j]),'seed_id':k,'arm':arm if learner!='prior' else 'prior','model':learner,
                         'context':'first230' if ctx=='native230' else ctx,'probability':p,
                         'rank':int(a['rank'][j]) if learner!='prior' else None})
    conf=np.empty(323)
    for fold in roles['folds']:
        m=joblib.load(r.HERE/'confounding'/(fold['fold']+'.joblib'));conf[m['test']]=m['probabilities']
    for i,x in enumerate(rows):long.append({'id':x['id'],'index':i,'center':x['site'],'site_code':x['site_code'],'ASD':x['y'],
     'seed_id':0,'arm':'metadata_only','model':'lr','context':'first190','probability':float(conf[i]),'rank':None})
    assert len(long)==12458
    return cfg,roles,sets,long,conf

def uncertainty(y,sites,ps,methods):
    rng=np.random.default_rng(r.seed('conditional_metrics'));B=2000;names=sorted(set(sites));draws=[];points=[];bycenter=[]
    for s in names:
        ix=sites==s;d=r.bootstrap_site(y[ix],ps[:,ix],rng,B);draws.append(d)
        vals=[]
        for j,(model,arm) in enumerate(methods):
            m=measure(y[ix],ps[j,ix]);vals.append([m[k] for k in ['auc','brier','log_loss']])
            bycenter.append({'center':s,'model':model,'arm':arm,'N':int(ix.sum()),
              'metrics':{key:{'estimate':vals[-1][t],'low':float(np.quantile(d[:,j,t],.025)),
                              'high':float(np.quantile(d[:,j,t],.975))} for t,key in enumerate(['auc','brier','log_loss'])}})
        points.append(vals)
    d=np.stack(draws,axis=1);point=np.mean(points,axis=0);within=d.mean(1)
    si=rng.integers(len(names),size=(B,len(names)));bi=rng.integers(B,size=si.shape);cluster=d[bi,si].mean(1)
    def summarize(point,a,b):
        return {'estimate':float(point),'within_low':float(np.quantile(a,.025)),'within_high':float(np.quantile(a,.975)),
                'cluster_low':float(np.quantile(b,.025)),'cluster_high':float(np.quantile(b,.975))}
    macro=[{'model':m,'arm':a,'metrics':{key:summarize(point[j,t],within[:,j,t],cluster[:,j,t]) for t,key in enumerate(['auc','brier','log_loss'])}} for j,(m,a) in enumerate(methods)]
    paired=[]
    for m in ['gnn','lr']:
        for a in ['budget','full','mp','fixed']:
            i=methods.index((m,'temporal'));j=methods.index((m,a))
            paired.append({'model':m,'contrast':'temporal_minus_'+a,'primary_Brier':m=='gnn' and a=='budget',
              'metrics':{key:summarize(point[i,t]-point[j,t],within[:,i,t]-within[:,j,t],cluster[:,i,t]-cluster[:,j,t]) for t,key in enumerate(['auc','brier','log_loss'])}})
    for a in r.ARMS:
        i=methods.index(('gnn',a));j=methods.index(('lr',a))
        paired.append({'arm':a,'contrast':'gnn_minus_lr','primary_Brier':False,
          'metrics':{key:summarize(point[i,t]-point[j,t],within[:,i,t]-within[:,j,t],cluster[:,i,t]-cluster[:,j,t]) for t,key in enumerate(['auc','brier','log_loss'])}})
    return {'scope':'Conditional fixed-prediction intervals; five centers; not training/selection uncertainty or equivalence evidence',
            'bootstrap_replicates':B,'macro':macro,'per_center':bycenter,'paired':paired}

def stability(sets,rows):
    summary=[];individual=[];contrasts=[]
    for k in [0,1,2]:
        for arm in r.ARMS if k==0 else ['temporal','budget']:
            base=sets[k,arm,'first190']
            for ctx in ['native230','last190']:
                a=sets[k,arm,ctx];ix=a['index'];y=a['y'];sites=a['site']
                for model in ['gnn','lr'] if k==0 else ['gnn']:
                    p0=base['p_'+model][ix];p1=a['p_'+model];delta=np.abs(p1-p0)
                    sr={s:{'N':int((sites==s).sum()),'mean_absolute_change':float(delta[sites==s].mean()),
                      'auc_change':float(r.roc_auc_score(y[sites==s],p1[sites==s])-r.roc_auc_score(y[sites==s],p0[sites==s]))} for s in sorted(set(sites))}
                    summary.append({'seed_id':k,'arm':arm,'model':model,'context':'first230' if ctx=='native230' else ctx,'N':len(ix),
                      'centers':len(sr),'mean_absolute_change':float(delta.mean()),'center_macro_absolute_change':float(np.mean([x['mean_absolute_change'] for x in sr.values()])),
                      'decision_flip_fraction':float(np.mean((p0>=.5)!=(p1>=.5))),'probability_rank_correlation':r.rho(p0,p1),
                      'pooled_auc_change':float(r.roc_auc_score(y,p1)-r.roc_auc_score(y,p0)),
                      'mean_brier_change':float(np.mean((p1-y)**2-(p0-y)**2)),'per_center':sr})
                    for j,i in enumerate(ix):individual.append({'index':int(i),'id':rows[i]['id'],'center':sites[j],'seed_id':k,'arm':arm,'model':model,
                     'context':'first230' if ctx=='native230' else ctx,'p_first190':float(p0[j]),'p_changed':float(p1[j]),'absolute_change':float(delta[j]),
                     'decision_flip':bool((p0[j]>=.5)!=(p1[j]>=.5)),'rank_change':int(a['rank'][j]-base['rank'][i])})
        for model in ['gnn','lr'] if k==0 else ['gnn']:
            for ctx in ['native230','last190']:
                a=sets[k,'temporal',ctx];b=sets[k,'budget',ctx];ix=a['index'];sites=a['site'];y=a['y']
                d=np.abs(a['p_'+model]-sets[k,'temporal','first190']['p_'+model][ix])-np.abs(b['p_'+model]-sets[k,'budget','first190']['p_'+model][ix])
                contrasts.append({'seed_id':k,'model':model,'contrast':'temporal_minus_budget','context':'first230' if ctx=='native230' else ctx,
                  'N':len(ix),'centers':3,'estimate':float(d.mean()),'conditional_intervals':r.paired_interval(d,y,sites,r.seed('stability',k,model,ctx),macro=False)})
    return summary,individual,contrasts

def calibration_intervals(y,sites,ps,methods):
    rng=np.random.default_rng(r.seed('calibration'));out=[]
    # Fixed-center, class-stratified draws preserve sampling composition. Not population-risk inference.
    cells=[np.flatnonzero((sites==s)&(y==d)) for s in sorted(set(sites)) for d in [0,1]]
    samples=[np.concatenate([rng.choice(c,len(c),replace=True) for c in cells]) for _ in range(2000)]
    for (model,arm),p in zip(methods,ps):
        values={'offset':[],'slope':[]};status={}
        for ix in samples:
            m=r.calibration(y[ix],p[ix]);status[m['status']]=status.get(m['status'],0)+1
            for key in values:
                if m[key] is not None:values[key].append(m[key])
        out.append({'model':model,'arm':arm,'estimate':r.calibration(y,p),'replicates':2000,'status_counts':status,
          'intervals':{key:{'successful':len(a),'failed':2000-len(a),'low':float(np.quantile(a,.025)) if a else None,
                            'high':float(np.quantile(a,.975)) if a else None} for key,a in values.items()}})
        print('CALIBRATION_DONE',model,arm,flush=True)
    return {'scope':'Pooled, fixed-center class-stratified, conditional on fitted scores and successful calibration estimation; not external recalibration', 'results':out}

def analyze():
    cfg,roles,sets,long,conf=collect();r.FINAL.mkdir(exist_ok=True);rows=roles['participants']
    y=np.array([x['y'] for x in rows]);sites=np.array([x['site'] for x in rows]);age=np.array([x['age'] for x in rows]);fd=np.array([x['mean_fd'] for x in rows])
    methods=[(m,a) for m in ['gnn','lr'] for a in r.ARMS]+[('prior','prior'),('lr','metadata_only')]
    ps=np.stack([sets[0,a,'first190']['p_'+m] for m in ['gnn','lr'] for a in r.ARMS]+[sets[0,'full','first190']['p_prior'],conf])
    tables=[];flat=[];collapse=[];association=[]
    for j,(model,arm) in enumerate(methods):
        p=ps[j];ss={}
        for s in sorted(set(sites)):
            take=sites==s;m=measure(y[take],p[take])
            if min(int(y[take].sum()),int((1-y[take]).sum()))<50:
                m['calibration']['slope']=None;m['calibration']['joint_intercept']=None;m['calibration']['status']='small_center_slope_suppressed'
            ss[s]=m
            collapse.append({'seed_id':0,'center':s,'model':model,'arm':arm,'N':int(take.sum()),'probability_min':float(p[take].min()),
              'probability_max':float(p[take].max()),'range':float(np.ptp(p[take])),'sd':float(p[take].std()),'constant_range_le_1e_7':bool(np.ptp(p[take])<=1e-7),
              'near_constant_range_le_1e_4':bool(np.ptp(p[take])<=1e-4),'all_same_threshold_class':bool(len(np.unique(p[take]>=.5))==1)})
            association.append({'model':model,'arm':arm,'center':s,'probability_age_spearman':r.rho(p[take],age[take]),'probability_fd_spearman':r.rho(p[take],fd[take])})
        macro={key:float(np.mean([a[key] for a in ss.values()])) for key in ['auc','brier','log_loss','balanced_accuracy','sensitivity','specificity']}
        tables.append({'model':model,'arm':arm,'N':323,'pooled':measure(y,p),'center_macro':macro,'centers':ss,'reliability_bins':r.reliability(y,p)})
        flat.append({'model':model,'arm':arm,'N':323,**macro,'pooled_auc':float(r.roc_auc_score(y,p)),'pooled_brier':float(np.mean((p-y)**2))})
    u=uncertainty(y,sites,ps,methods)
    st,individual,stc=stability(sets,rows)
    sensitivity=[]
    for k in [0,1,2]:
        p=sets[k,'temporal','first190']['p_gnn'];q=sets[k,'budget','first190']['p_gnn'];d=(p-y)**2-(q-y)**2
        for s in ['macro']+sorted(set(sites)):
            take=np.ones(323,bool) if s=='macro' else sites==s
            avg=lambda a:float(np.mean([a[sites==z].mean() for z in set(sites)])) if s=='macro' else float(a[take].mean())
            sensitivity.append({'seed_id':k,'center':s,'temporal_brier':avg((p-y)**2),'budget_brier':avg((q-y)**2),'temporal_minus_budget_brier':avg(d)})
        if k:
            for arm in ['temporal','budget']:
                pp=sets[k,arm,'first190']['p_gnn']
                for s in sorted(set(sites)):
                    a=pp[sites==s];collapse.append({'seed_id':k,'center':s,'model':'gnn','arm':arm,'N':len(a),'probability_min':float(a.min()),
                     'probability_max':float(a.max()),'range':float(np.ptp(a)),'sd':float(a.std()),'constant_range_le_1e_7':bool(np.ptp(a)<=1e-7),
                     'near_constant_range_le_1e_4':bool(np.ptp(a)<=1e-4),'all_same_threshold_class':bool(len(np.unique(a>=.5))==1)})
    low=fd<=.2;low_results=[]
    for (m,a),p in zip(methods,ps):
        sr={s:measure(y[low&(sites==s)],p[low&(sites==s)]) for s in sorted(set(sites))}
        low_results.append({'model':m,'arm':a,'N':int(low.sum()),'center_macro_auc':float(np.mean([x['auc'] for x in sr.values()])),
          'center_macro_brier':float(np.mean([x['brier'] for x in sr.values()]))})
    diagnostics=[]
    for fold in roles['folds']:
        for ctx in ['first190','native230','last190']:
            a=sets[0,'temporal',ctx];b=sets[0,'budget',ctx];take=a['site']==fold['held_center']
            if not take.any():continue
            for j in np.flatnonzero(take):
                diagnostics.append({'index':int(a['index'][j]),'center':fold['held_center'],'context':'first230' if ctx=='native230' else ctx,
                  'temporal_rank':int(a['rank'][j]),'budget_rank':int(b['rank'][j]),'identical_operator':bool(a['rank'][j]==b['rank'][j])})
    demographics={s:{str(d):{'N':len(a:=[x for x in rows if x['site']==s and x['y']==d]),
      'mean_age':float(np.mean([x['age'] for x in a])),'male':sum(x['sex_code']==1 for x in a),
      'mean_fd':float(np.mean([x['mean_fd'] for x in a])),'low_motion_N':sum(x['mean_fd']<=.2 for x in a)} for d in [0,1]} for s in sorted(set(sites))}
    r.csvwrite(r.FINAL/'participant_predictions.csv',long);r.write(r.FINAL/'performance.json',tables);r.csvwrite(r.FINAL/'performance_summary.csv',flat)
    r.write(r.FINAL/'metric_uncertainty.json',u);r.write(r.FINAL/'paired_contrasts.json',{'primary_and_secondary_prediction':u['paired'],'secondary_stability':stc})
    r.write(r.FINAL/'stability.json',st);r.csvwrite(r.FINAL/'stability_summary.csv',[{k:v for k,v in a.items() if k!='per_center'} for a in st]);r.csvwrite(r.FINAL/'participant_stability.csv',individual)
    r.csvwrite(r.FINAL/'seed_sensitivity.csv',sensitivity);r.csvwrite(r.FINAL/'collapse_diagnostics.csv',collapse);r.csvwrite(r.FINAL/'operator_identity.csv',diagnostics)
    r.write(r.FINAL/'confounding.json',{'demographics':demographics,'low_motion_results':low_results,'low_motion_N':int(low.sum()),
       'low_motion_interpretation':'Conditional evaluation of existing fits, not retraining or elimination of motion confounding','within_center_associations':association})
    r.write(r.FINAL/'calibration_intervals.json',calibration_intervals(y,sites,ps,methods))
    figures(tables,u,st,sensitivity)
    print('ANALYSIS_COMPLETE',flush=True)

def figures(tables,u,st,seeds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    folder=r.FINAL/'figures';folder.mkdir(exist_ok=True);labels=['Full FC','MP','Temporal null','Rank budget','Fixed rank']
    lookup={(x['model'],x['arm']):x for x in tables};ci={(x['model'],x['arm']):x['metrics'] for x in u['macro']}
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,key in zip(axes,['auc','brier']):
        for m,offset,color in [('gnn',-.14,'#2676ac'),('lr',.14,'#c66b29')]:
            a=[ci[m,z][key] for z in r.ARMS];p=np.array([x['estimate'] for x in a]);lo=np.array([x['cluster_low'] for x in a]);hi=np.array([x['cluster_high'] for x in a])
            ax.errorbar(np.arange(5)+offset,p,yerr=np.maximum(np.stack([p-lo,hi-p]),0),fmt='o',capsize=3,label=m.upper(),color=color)
        ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_ylabel('Equal-center '+('AUC' if key=='auc' else 'Brier loss'));ax.grid(axis='y',alpha=.2)
        ax.axhline(.5 if key=='auc' else .25,color='gray',ls=':',lw=1);ax.legend()
    fig.suptitle('ABIDE ASD replication: 323 participants, five held centers\nConditional 95% center-cluster intervals; not MDD validation',fontsize=12)
    save(fig,'01_performance')
    sites=sorted(next(iter(lookup.values()))['centers']);fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,m in zip(axes,['gnn','lr']):
        a=np.array([[lookup[m,z]['centers'][s]['auc'] for z in r.ARMS] for s in sites]);im=ax.imshow(a,vmin=.3,vmax=.8,cmap='viridis')
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',color='white' if a[i,j]<.6 else 'black')
        ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_yticks(range(5),sites);ax.set_title(m.upper()+' held-center AUC')
    fig.colorbar(im,ax=axes,label='AUC',shrink=.8);save(fig,'02_center_generalization')
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained');ax.plot([0,1],[0,1],':',color='gray')
    for m,a in [('gnn','temporal'),('gnn','budget'),('lr','full'),('prior','prior')]:
        bins=[b for b in lookup[m,a]['reliability_bins'] if b['N']];ax.plot([b['mean_prediction'] for b in bins],[b['observed_fraction'] for b in bins],'o-',label=m.upper()+' '+a)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean held-out probability',ylabel='Observed ASD fraction',title='Pooled calibration: case-control sample\nFive fixed probability bins; no recalibration');ax.legend();save(fig,'03_calibration')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,ctx in zip(axes,['first230','last190']):
        for m,off in [('gnn',-.18),('lr',.18)]:
            a=[next(x for x in st if x['seed_id']==0 and x['model']==m and x['arm']==z and x['context']==ctx)['mean_absolute_change'] for z in r.ARMS]
            ax.bar(np.arange(5)+off,a,.34,label=m.upper())
        ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set_ylabel('Mean absolute probability change');ax.set_title(ctx+' versus first190');ax.legend()
    fig.suptitle('Finite-input replay: 243 participants, three centers\nOverlapping derivative windows; not independent scans',fontsize=12);save(fig,'04_finite_input_stability')
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    order=['macro']+sites
    for k,off in [(0,-.15),(1,0),(2,.15)]:
        vals=[next(x for x in seeds if x['seed_id']==k and x['center']==s)['temporal_minus_budget_brier'] for s in order]
        ax.plot(np.arange(len(order))+off,vals,'o',label='Primary seed' if k==0 else 'Sensitivity seed '+str(k))
    ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(len(order)),order);ax.set_ylabel('Brier: temporal null minus rank budget');ax.set_title('Primary contrast and bounded seed sensitivity\nNegative favors temporal null; additional seeds are not retuned');ax.legend();save(fig,'05_temporal_budget_seed_sensitivity')

def verify():
    cfg,roles=r.check_lock(True);rows=roles['participants'];data=r.load_data(roles);groups=np.array([x['site'] for x in rows]);y=np.array([x['y'] for x in rows]);replay=[]
    protected={p:r.sha(r.REST/p)==h for p,h in cfg['protected_rest_files'].items()};assert all(protected.values())
    original_files=set(cfg['protected_rest_files']);current_files={str(p.relative_to(r.REST)) for p in r.REST.rglob('*') if p.is_file()};assert original_files==current_files,'REST file inventory changed'
    for f in r.read(r.HERE/'spectral_cache_manifest.json')['files']:assert r.sha(r.CACHE/f['file'])==f['sha256']
    first=data['first190'];ranks=(first['w']>first['threshold'][:,None]).sum(1);all_test=[]
    for fold in roles['folds']:
        tr=np.array(fold['train']);te=np.array(fold['test']);all_test.extend(te);assert not set(groups[tr])&set(groups[te])
        assert set(tr)|set(te)==set(range(323));inner_test=[]
        for f in fold['inner']:
            a=np.array(f['train']);b=np.array(f['validation']);assert not set(a)&set(b) and set(a)|set(b)==set(tr);assert not set(groups[a])&set(groups[b]);inner_test.extend(b)
        assert sorted(inner_test)==sorted(tr)
        budget=r.rep.fit_budget(first['w'][tr],ranks[tr])
        for k in [0,1,2]:
            for arm in r.ARMS if k==0 else ['temporal','budget']:
                folder=r.HERE/'main' if k==0 else r.HERE/'seed_sensitivity'/('seed'+str(k));prefix=folder/(fold['fold']+'_'+arm)
                meta=r.read(prefix.with_suffix('.json'));assert r.sha(prefix.with_suffix('.npz'))==meta['predictions_sha'];assert meta['budget']==budget
                assert meta['training_audit']['weight_update_l2']>0
                expected_seed=fold['seeds']['outer'] if k==0 else fold['seeds']['sensitivity'][str(k)];assert meta['training_audit']['seed']==expected_seed
                if k==0:
                    assert meta['config_sha']==r.sha(r.CFG)
                    for j,inner in enumerate(fold['inner']):
                        it=np.array(inner['train']);assert meta['inner'][j]['budget']==r.rep.fit_budget(first['w'][it],ranks[it])
                        assert meta['inner'][j]['seed']==fold['seeds']['inner'][j]
                state=torch.load(str(prefix)+'_gnn.pt',map_location=r.models.DEVICE,weights_only=False)
                assert set(state['train_indices'])==set(tr);model=r.models.SpectralGNN().to(r.models.DEVICE);model.load_state_dict(state['state_dict'])
                if k:assert state['hyperparameters']==r.read(r.HERE/'main'/(fold['fold']+'_'+arm+'.json'))['selected_gnn']
                linear=joblib.load(str(prefix)+'_lr.joblib') if k==0 else None
                if linear:assert set(linear['train_indices'])==set(tr)
                maxg=maxl=maxh=0.
                with np.load(prefix.with_suffix('.npz')) as z:
                    for ctx in ['first190','native230','last190']:
                        take=z['context']==ctx;ix=z['index'][take]
                        expected=te if ctx=='first190' else te[[rows[i]['finite230'] for i in te]]
                        np.testing.assert_array_equal(ix,expected)
                        if not len(ix):continue
                        np.testing.assert_array_equal(z['y'][take],y[ix]);op,rank,_=r.exp.operators({key:v[ix] for key,v in data[ctx].items()},arm,budget)
                        p,h=r.models.predict_gnn(model,op);np.testing.assert_array_equal(rank,z['rank'][take])
                        maxg=max(maxg,float(np.max(np.abs(p-z['p_gnn'][take]))));maxh=max(maxh,float(np.max(np.abs(h-z['embedding'][take]))))
                        if linear:
                            pp=r.models.predict_lr(linear['model'],linear['scaler'],op);maxl=max(maxl,float(np.max(np.abs(pp-z['p_lr'][take]))))
                assert maxg<2e-6 and maxl<2e-6 and maxh<2e-6
                replay.append({'fold':fold['fold'],'arm':arm,'seed_id':k,'GNN_error':maxg,'LR_error':maxl if linear else None,'embedding_error':maxh})
        print('VERIFIED_CENTER',fold['held_center'],flush=True)
    assert sorted(all_test)==list(range(323)) and len(replay)==45
    cfg,roles,sets,long,conf=collect()
    # Independent direct recomputation of the primary estimand from held-out probabilities.
    a=sets[0,'temporal','first190']['p_gnn'];b=sets[0,'budget','first190']['p_gnn'];d=(a-y)**2-(b-y)**2
    estimate=float(np.mean([d[groups==s].mean() for s in sorted(set(groups))]));paired=r.read(r.FINAL/'paired_contrasts.json')
    saved=next(x for x in paired['primary_and_secondary_prediction'] if x['model']=='gnn' and x['contrast']=='temporal_minus_budget')['metrics']['brier']['estimate']
    assert abs(estimate-saved)<1e-12
    # Weighted AUC implementation must agree on tied-score data, including constants.
    yy=np.array([0,1,0,1]);pp=np.array([.2,.2,.8,.8]);assert r.weighted_auc(yy,pp,np.ones((1,4)))[0]==r.roc_auc_score(yy,pp)==.5
    for fold in roles['folds']:
        q=joblib.load(r.HERE/'confounding'/(fold['fold']+'.joblib'));tr=np.array(fold['train']);te=np.array(fold['test'])
        x=np.array([[a['age'],int(a['sex_code']==1),a['mean_fd']] for a in rows]);np.testing.assert_allclose(q['scaler'].mean_,x[tr].mean(0))
        np.testing.assert_allclose(q['model'].predict_proba(q['scaler'].transform(x[te]))[:,1],q['probabilities'],atol=1e-12)
    required=['participant_predictions.csv','performance.json','performance_summary.csv','paired_contrasts.json','metric_uncertainty.json','calibration_intervals.json',
              'stability.json','stability_summary.csv','participant_stability.csv','seed_sensitivity.csv','collapse_diagnostics.csv','operator_identity.csv','confounding.json']
    for p in required:assert (r.FINAL/p).is_file()
    assert len(list((r.FINAL/'figures').glob('*.png')))==5 and len(list((r.FINAL/'figures').glob('*.svg')))==5
    output={'passed':True,'status':'COMPLETED_SINGLE_ABIDE_EXPERIMENT_STOP_IMPLEMENTATION','protocol_sha':r.sha(r.CFG),'N':323,'ASD':156,'controls':167,'centers':5,
      'finite_N':243,'finite_centers':3,'primary_outer_GNN':25,'primary_outer_LR':25,'additional_GNN':20,'metadata_outer_LR':5,
      'participant_contexts':809,'prediction_rows':len(long),'all_REST_files_unchanged':True,'protected_REST_file_count':len(protected),
      'all_ABIDE_included_sources_unchanged':True,'all_cache_hashes_match':True,'grouped_roles_verified':True,'all_checkpoints_replayed':replay,
      'primary_contrast_independent_recalculation':estimate,'no_MDD_external_validation':True,'no_architecture_search':True,
      'scope':'ASD methodological replication; shared ABIDE data and temporal-null methodology with Paper 1 submitted to Array; frozen REST primary unchanged'}
    r.write(r.FINAL/'verification.json',output)
    readme='''# Frozen ABIDE-I ASD replication — 28 October 2026

This is the single approved secondary cross-disorder methodological replication.
It is NOT independent MDD clinical validation. REST-meta-MDD remains the unchanged primary study.
ABIDE derivatives and temporal-null methodology were also used in Paper 1, submitted to Array.
New endpoints here are trained held-center probabilities, calibration and finite-input replay.

## Reproduce

Run `./run.ps1 cache`, `./run.ps1 train`, `./run.ps1 analyze`, `./run.ps1 verify` in PowerShell.
The existing protocol must never be overwritten; `freeze` refuses an existing lock.
The launcher uses the installed bundled Python and read-only scientific runtimes in the sibling MDD_GNN_Q1 folder.
Source paths and hashes are in participant_roles.json. Runtime versions/device are in runtime.json.
Frozen algorithms are imported read-only from the sibling OBJECTIVE2_RESTMETA_FMRI/scripts directory.
Source code, protocol and participant roles must match their pre-scoring hashes.

## Design

323 participants: 156 ASD, 167 controls, five supported centers. Leuven and Michigan sub-releases are grouped.
Five held-center outer folds, three grouped inner folds, five original representations, unchanged GNN/logistic grids.
The primary estimand is equal-center GNN Brier temporal-null minus rank budget, primary seed only.
Two additional GNN seeds replay temporal/budget with primary-selected hyperparameters; no retuning.
243 participants from three centers support first230/first190/last190 replay without refitting.
The internal `native230` key ALWAYS means rows 0:230 of the original derivative, not the full original run.
Last190 means rows 40:230; first/last190 share150 volumes. Full-run preprocessing precedes all windows.
No extra filtering, GSR, harmonization, censoring, ROI imputation or architecture experiments.

## Evidence

- final/participant_predictions.csv: all primary, sensitivity, prior and metadata probabilities, identified by seed and context.
- final/performance_summary.csv and performance.json: primary discrimination, proper scores, calibration and center metrics.
- final/metric_uncertainty.json and paired_contrasts.json: paired conditional fixed-prediction uncertainty.
- final/calibration_intervals.json: pooled calibration intervals, including successful/failed estimation counts.
- final/stability_summary.csv and participant_stability.csv: finite-input replay, not repeat-scan validation.
- final/seed_sensitivity.csv, collapse_diagnostics.csv and operator_identity.csv: retained negative/degenerate outcomes.
- final/confounding.json: demographics, metadata-only baseline context, low-motion fixed-score sensitivity and descriptive associations.
- final/figures/: five PNG/SVG figure pairs, all secondary ABIDE evidence relative to the frozen REST study.
- final/verification.json: original preservation, source hashes, fold isolation, all checkpoint replay and completeness.
- final/artifact_manifest.json: hashes of all final code/configuration/cache/checkpoint/evidence artifacts.

Bootstrap intervals condition on fitted scores and omit training-sample and selection uncertainty.
Only five held centers and three finite-input centers are available; center-cluster inference is limited.
Per-center calibration slopes are suppressed below50 participants in either diagnosis.
Case-control calibration is not population clinical risk. No diagnostic benefit or equivalence is guaranteed.
Do not select a seed, center, representation or subset based on favorable results.

Implementation is complete. STOP: no more dataset searches, architecture experiments or strengthening cycles.
The next phase is manuscript preparation using both frozen evidence sets.
'''
    (r.HERE/'README_REPRODUCIBILITY.md').write_text(readme,encoding='utf-8')
    manifest={str(p.relative_to(r.HERE)):r.sha(p) for p in r.HERE.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    r.write(r.FINAL/'artifact_manifest.json',{'protocol_sha':r.sha(r.CFG),'files':manifest})
    print('VERIFICATION_PASSED',len(protected),'REST files unchanged; primary contrast',estimate,flush=True)
