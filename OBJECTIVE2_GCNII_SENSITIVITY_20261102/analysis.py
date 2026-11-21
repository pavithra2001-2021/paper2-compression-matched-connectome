"""Frozen paired architecture analysis and checkpoint replay; no fitting here."""
import csv
import warnings
import numpy as np
import torch
import joblib
from sklearn.metrics import roc_auc_score, log_loss
import study as r

KEYS = ['auc','brier','log_loss','balanced_accuracy','sensitivity','specificity']
LABELS = ['Full FC','MP','Temporal null','Rank budget','Fixed rank']

def csvread(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def measure(y,p,cal=True):
    a=r.metrics(y,p,include_calibration=cal);a['cases']=a.pop('MDD');a['controls']=a.pop('HC');return a

def collect(ds,d):
    rows=d['participants'];sets={};long=[]
    def add(model,k,arm,context,index,y,p,extra=None):
        order=np.argsort(index);index=np.asarray(index)[order].astype(int);y=np.asarray(y)[order].astype(int);p=np.asarray(p)[order].astype(float)
        expected=np.array([i for i,x in enumerate(rows) if (ds!='REST' or x['site']!='S12') and (context=='first190' or x['finite230'])])
        np.testing.assert_array_equal(index,expected);np.testing.assert_array_equal(y,[rows[i]['y'] for i in index])
        assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        a={'index':index,'y':y,'p':p,'site':np.array([rows[i]['site'] for i in index])}
        if extra:
            for key,v in extra.items():a[key]=np.asarray(v)[order]
        sets[model,k,arm,context]=a
        for j,i in enumerate(index):long.append({'dataset':ds,'index':int(i),'id':rows[i]['id'],'site':rows[i]['site'],
            'y':int(y[j]),'model':model,'seed_id':k,'arm':arm,'context':context,'probability':float(p[j])})
    for model in ['gcnii','original']:
        for k in [0,1,2]:
            for arm in r.ARMS if k==0 else ['temporal','budget']:
                if model=='original' and ds=='REST' and k:
                    source=csvread(r.REST/'supplementary/focused_strengthening_20261026/participant_predictions.csv')
                    for ctx in r.CONTEXTS:
                        v=[x for x in source if int(x['replicate'])==k and x['arm']==arm and x['context']==ctx and x['site']!='S12']
                        add(model,k,arm,ctx,[int(x['index']) for x in v],[int(x['y']) for x in v],[float(x['p']) for x in v])
                    continue
                chunks={}
                for f in d['folds']:
                    if model=='gcnii':path=r.destination(ds,f['fold'],arm,k).with_suffix('.npz')
                    elif ds=='REST':path=r.original.RUN/'main'/(f['fold']+'_'+arm+'.npz')
                    else:path=(r.ABIDE/'main' if k==0 else r.ABIDE/'seed_sensitivity'/('seed'+str(k)))/(f['fold']+'_'+arm+'.npz')
                    with np.load(path) as z:
                        for key in z.files:chunks.setdefault(key,[]).append(z[key].copy())
                a={key:np.concatenate(v) for key,v in chunks.items()}
                for ctx in r.CONTEXTS:
                    take=a['context']==ctx;extra={key:a[key][take] for key in ['embedding','logit','rank'] if key in a}
                    add(model,k,arm,ctx,a['index'][take],a['y'][take],a['p_gnn'][take],extra)
                    if model=='original' and k==0:
                        add('lr',0,arm,ctx,a['index'][take],a['y'][take],a['p_lr'][take])
                        if arm=='full':add('prior',0,'prior',ctx,a['index'][take],a['y'][take],a['p_prior'][take])
    base=sets['gcnii',0,'full','first190'];ix=base['index']
    if ds=='REST':
        pred=r.read(r.original.RUN/'final/nuisance_controls.json')['predictions']
        m={x['index']:x['metadata_probability'] for x in pred if x['scheme']=='loso'};p=[m[i] for i in ix]
    else:
        p=np.empty(len(rows))
        for f in d['folds']:
            m=joblib.load(r.ABIDE/'confounding'/(f['fold']+'.joblib'));p[m['test']]=m['probabilities']
        p=p[ix]
    add('metadata',0,'metadata','first190',ix,base['y'],p)
    return sets,long

def interval(point,within,cluster):
    return {'estimate':float(point),'within_low':float(np.quantile(within,.025)),'within_high':float(np.quantile(within,.975)),
            'cluster_low':float(np.quantile(cluster,.025)),'cluster_high':float(np.quantile(cluster,.975))}

def uncertainty(ds,y,sites,ps,methods):
    rng=np.random.default_rng(r.seed('gcnii_conditional_metrics',ds));B=2000;names=sorted(set(sites));draws=[];points=[];center=[]
    for s in names:
        take=sites==s;dd=r.bootstrap_site(y[take],ps[:,take],rng,B);draws.append(dd)
        v=np.array([[measure(y[take],p[take],False)[key] for key in KEYS[:3]] for p in ps]);points.append(v)
        for j,(model,arm) in enumerate(methods):center.append({'site':s,'model':model,'arm':arm,'N':int(take.sum()),
            'metrics':{key:{'estimate':float(v[j,t]),'low':float(np.quantile(dd[:,j,t],.025)),'high':float(np.quantile(dd[:,j,t],.975))} for t,key in enumerate(KEYS[:3])}})
    dd=np.stack(draws,axis=1);point=np.mean(points,axis=0);within=dd.mean(1)
    si=rng.integers(len(names),size=(B,len(names)));bi=rng.integers(B,size=si.shape);cluster=dd[bi,si].mean(1)
    macro=[{'model':m,'arm':a,'metrics':{key:interval(point[j,t],within[:,j,t],cluster[:,j,t]) for t,key in enumerate(KEYS[:3])}} for j,(m,a) in enumerate(methods)]
    contrasts=[]
    def contrast(name,terms,primary=False):
        coeff=np.zeros(len(methods))
        for method,c in terms:coeff[methods.index(method)]+=c
        contrasts.append({'contrast':name,'primary_REST_Brier':primary and ds=='REST',
            'metrics':{key:interval(coeff@point[:,t],within[:,:,t]@coeff,cluster[:,:,t]@coeff) for t,key in enumerate(KEYS[:3])}})
    for m in ['gcnii','original','lr']:
        for arm in ['budget','full','mp','fixed']:contrast(m+'_temporal_minus_'+arm,[((m,'temporal'),1),((m,arm),-1)])
    contrast('Gamma',[ (('gcnii','temporal'),1),(('gcnii','budget'),-1),(('original','temporal'),-1),(('original','budget'),1)],True)
    for arm in r.ARMS:
        for ref in ['original','lr']:contrast('gcnii_minus_'+ref+'_'+arm,[(('gcnii',arm),1),((ref,arm),-1)])
    return {'scope':'2000 paired conditional fixed-prediction draws; within cohort x diagnosis and cohort-cluster with fresh within draws. No training/selection uncertainty, equivalence, or residual-mechanism causality.',
        'macro':macro,'per_cohort':center,'paired':contrasts}

def calibration_intervals(ds,y,sites,ps):
    rng=np.random.default_rng(r.seed('gcnii_calibration',ds));cells=[np.flatnonzero((sites==s)&(y==d)) for s in sorted(set(sites)) for d in [0,1]]
    samples=[np.concatenate([rng.choice(c,len(c),replace=True) for c in cells]) for _ in range(2000)];out=[]
    for arm,p in zip(r.ARMS,ps):
        values={'offset':[],'slope':[]};statuses={}
        for ix in samples:
            a=r.calibration(y[ix],p[ix]);statuses[a['status']]=statuses.get(a['status'],0)+1
            for key in values:
                if a[key] is not None:values[key].append(a[key])
        out.append({'arm':arm,'estimate':r.calibration(y,p),'status_counts':statuses,'intervals':{key:{'successful':len(v),'failed':2000-len(v),
            'low':float(np.quantile(v,.025)) if v else None,'high':float(np.quantile(v,.975)) if v else None} for key,v in values.items()}})
        print('CALIBRATION_DONE',ds,arm,flush=True)
    return {'scope':'Pooled descriptive calibration; fixed cohort/class counts; conditional on fitted scores and successful estimation; no recalibration. Pooled slope can reflect between-site variation.', 'results':out}

def stability(ds,sets):
    out=[];individual=[];contrasts=[];vectors={}
    for (m,k,arm,ctx),a in sets.items():
        if ctx=='first190' or m in ['metadata','prior']:continue
        base=sets[m,k,arm,'first190'];positions={int(i):j for j,i in enumerate(base['index'])};p0=np.array([base['p'][positions[int(i)]] for i in a['index']]);p=a['p'];y=a['y'];sites=a['site']
        change=np.abs(p-p0);vectors[m,k,arm,ctx]=change
        per={s:{'N':int((sites==s).sum()),'mean_absolute_change':float(change[sites==s].mean()),
            'auc_change':float(roc_auc_score(y[sites==s],p[sites==s])-roc_auc_score(y[sites==s],p0[sites==s]))} for s in sorted(set(sites))}
        out.append({'model':m,'seed_id':k,'arm':arm,'context':ctx,'N':len(y),'cohorts':len(per),'mean_absolute_change':float(change.mean()),
            'macro_absolute_change':r.macro(change,sites),'decision_flip_fraction':float(np.mean((p>=.5)!=(p0>=.5))),
            'probability_rank_correlation':r.rho(p,p0),'mean_brier_change':float(np.mean((p-y)**2-(p0-y)**2)),
            'macro_brier_change':r.macro((p-y)**2-(p0-y)**2,sites),'per_cohort':per})
        for j,i in enumerate(a['index']):individual.append({'index':int(i),'site':sites[j],'model':m,'seed_id':k,'arm':arm,'context':ctx,
            'p_first190':float(p0[j]),'p_changed':float(p[j]),'absolute_change':float(change[j]),'decision_flip':bool((p[j]>=.5)!=(p0[j]>=.5))})
    for ctx in ['native230','last190']:
        a=sets['gcnii',0,'temporal',ctx]
        for k in [0,1,2]:
            vv={}
            for m in ['gcnii','original']+(['lr'] if k==0 else []):vv[m]=vectors[m,k,'temporal',ctx]-vectors[m,k,'budget',ctx]
            vv['architecture_difference']=vv['gcnii']-vv['original']
            for m,v in vv.items():contrasts.append({'model':m,'seed_id':k,'context':ctx,'contrast':'temporal_minus_budget_absolute_change',
                'estimate_pooled':float(v.mean()),'estimate_macro':r.macro(v,a['site']),
                'conditional_pooled_intervals':r.paired_interval(v,a['y'],a['site'],r.seed('gcnii_stability',ds,k,m,ctx),macro=False)})
    return out,individual,contrasts

def metadata(ds,rows):
    if ds=='ABIDE':return [{'age':x['age'],'fd':x['mean_fd'],'sex':x['sex_code'],'qc':'all_pass'} for x in rows],{}
    import openpyxl
    from nuisance_controls import BOOK
    expected=r.read(r.original.RUN/'nuisance_control_spec.json')['source_sha'];assert r.sha(BOOK)==expected
    wb=openpyxl.load_workbook(BOOK,read_only=True,data_only=True);lookup={}
    for sheet in wb:
        it=sheet.iter_rows(values_only=True);headers=next(it)
        for v in it:
            if v and v[0] is not None:lookup[str(v[0]).strip()]=dict(zip(headers,v))
    wb.close();assert r.sha(BOOK)==expected
    output=[];hashes={str(BOOK):expected}
    def number(v):
        try:
            z=float(v);return z if np.isfinite(z) and z>=0 else None
        except (ValueError,TypeError):return None
    for x in rows:
        m=lookup[x['id']];path=r.original.Path(x['FD_Jenkinson_path']);h=r.sha(path);fd=np.loadtxt(path);assert np.isfinite(fd).all() and r.sha(path)==h
        hashes[str(path)]=h;age=number(m.get('Age'))
        output.append({'age':age if age and age>0 else None,'fd':float(fd.mean()),'sex':number(m.get('Sex')),'qc':x['QC_score']})
    return output,hashes

def confounding(ds,rows,base,ps,methods):
    meta,hashes=metadata(ds,rows);ix=base['index'];sites=base['site'];y=base['y'];associations=[];qc=[]
    def vector(key):return np.array([meta[i][key] if meta[i][key] is not None else np.nan for i in ix])
    age=vector('age');fd=vector('fd');sex=vector('sex');qc_values=np.array([str(meta[i]['qc']) for i in ix])
    for (model,arm),p in zip(methods,ps):
        for s in sorted(set(sites)):
            take=sites==s;entry={'model':model,'arm':arm,'site':s}
            for name,v in [('age',age),('fd',fd)]:
                ok=take & np.isfinite(v);entry[name+'_N']=int(ok.sum());entry['probability_'+name+'_spearman']=r.rho(p[ok],v[ok]) if ok.any() else None
            one=p[take&(sex==1)];two=p[take&(sex==2)];entry['sex_code1_minus_code2_probability']=float(one.mean()-two.mean()) if len(one) and len(two) else None
            associations.append(entry)
            if ds=='REST':
                for category in sorted(set(qc_values[take])):
                    z=take&(qc_values==category);qc.append({'model':model,'arm':arm,'site':s,'QC_raw':category,**measure(y[z],p[z],False)})
    low=[]
    if ds=='ABIDE':
        select=fd<=.2
        for (m,a),p in zip(methods,ps):
            vals=[measure(y[select&(sites==s)],p[select&(sites==s)],False) for s in sorted(set(sites))]
            low.append({'model':m,'arm':a,'N':int(select.sum()),'macro_auc':float(np.mean([v['auc'] for v in vals])),
                'macro_brier':float(np.mean([v['brier'] for v in vals]))})
    return {'scope':'Descriptive fixed-score sensitivities; not causal adjustment or evidence of confounding removal. REST FD is whole derivative, no segment alignment inferred.',
        'within_cohort_associations':associations,'REST_QC_strata':qc,'ABIDE_low_FD_le_0_2':low,'metadata_source_hashes':hashes}

def analyze():
    cfg,roles=r.check_lock();r.FINAL.mkdir(exist_ok=True)
    with r.threadpool_limits(2):
        for ds,d in roles.items():
            folder=r.FINAL/ds;sets,long=collect(ds,d);base=sets['gcnii',0,'full','first190'];y=base['y'];sites=base['site']
            methods=[(m,a) for m in ['gcnii','original','lr'] for a in r.ARMS]+[('prior','prior'),('metadata','metadata')]
            ps=np.stack([sets[m,0,a,'first190']['p'] for m,a in methods]);tables=[];flat=[];cohort=[];collapse=[];seedrows=[]
            for (model,arm),p in zip(methods,ps):
                ss={}
                for s in sorted(set(sites)):
                    take=sites==s;v=measure(y[take],p[take]);cal=v['calibration']
                    if min(int(y[take].sum()),int((1-y[take]).sum()))<50:cal['slope']=None;cal['joint_intercept']=None;cal['status']='small_cohort_slope_suppressed'
                    ss[s]=v;cohort.append({'model':model,'arm':arm,'site':s,**{k:v[k] for k in ['N','cases','controls']+KEYS},**{'calibration_'+k:v for k,v in cal.items()}})
                macro={key:float(np.mean([v[key] for v in ss.values()])) for key in KEYS};pooled=measure(y,p)
                tables.append({'model':model,'arm':arm,'N':len(y),'macro':macro,'pooled':pooled,'cohorts':ss,'reliability_bins':r.reliability(y,p)})
                flat.append({'model':model,'arm':arm,'N':len(y),**macro,'pooled_auc':pooled['auc'],'pooled_brier':pooled['brier']})
            for (m,k,arm,ctx),a in sets.items():
                if ctx!='first190' or m not in ['gcnii','original']:continue
                for s in sorted(set(a['site'])):
                    take=a['site']==s;p=a['p'][take];span=float(np.ptp(p));z=a.get('logit');h=a.get('embedding')
                    collapse.append({'model':m,'seed_id':k,'arm':arm,'site':s,'N':len(p),'minimum':float(p.min()),'maximum':float(p.max()),
                        'range':span,'sd':float(p.std()),'q05':float(np.quantile(p,.05)),'median':float(np.median(p)),'q95':float(np.quantile(p,.95)),
                        'exact_constant':span==0,'constant_le_1e_7':span<=1e-7,'near_constant_le_1e_4':span<=1e-4,'REST_historical_le_1e_6':span<=1e-6,
                        'all_same_threshold_class':bool(len(set(p>=.5))==1),'logit_range':float(np.ptp(z[take])) if z is not None else None,
                        'embedding_mean_component_sd':float(np.std(h[take],axis=0).mean()) if h is not None else None})
            for k in [0,1,2]:
                for s in ['macro']+sorted(set(sites)):
                    avg=lambda v:r.macro(v,sites) if s=='macro' else float(v[sites==s].mean())
                    record={'seed_id':k,'site':s}
                    for m in ['gcnii','original']:
                        for a in ['temporal','budget']:record[m+'_'+a+'_brier']=avg((sets[m,k,a,'first190']['p']-y)**2)
                        record[m+'_contrast']=record[m+'_temporal_brier']-record[m+'_budget_brier']
                    record['Gamma']=record['gcnii_contrast']-record['original_contrast'];seedrows.append(record)
            u=uncertainty(ds,y,sites,ps,methods);st,individual,stc=stability(ds,sets)
            r.csvwrite(folder/'participant_predictions.csv',long);r.write(folder/'performance.json',tables);r.csvwrite(folder/'performance_summary.csv',flat)
            r.csvwrite(folder/'cohort_performance.csv',cohort);r.write(folder/'metric_uncertainty.json',u);r.write(folder/'paired_contrasts.json',{'performance':u['paired'],'finite_input':stc})
            r.write(folder/'stability.json',st);r.csvwrite(folder/'stability_summary.csv',[{k:v for k,v in a.items() if k!='per_cohort'} for a in st]);r.csvwrite(folder/'participant_stability.csv',individual)
            r.csvwrite(folder/'collapse_diagnostics.csv',collapse);r.csvwrite(folder/'seed_sensitivity.csv',seedrows)
            r.write(folder/'confounding.json',confounding(ds,d['participants'],base,ps,methods))
            r.write(folder/'calibration_intervals.json',calibration_intervals(ds,y,sites,ps[:5]))
            figures(ds,tables,u,st,seedrows,collapse,sets)
            print('ANALYSIS_COMPLETE',ds,flush=True)

def figures(ds,tables,u,st,seeds,collapse,sets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    folder=r.FINAL/ds/'figures';folder.mkdir(parents=True,exist_ok=True);lookup={(x['model'],x['arm']):x for x in tables}
    ci={(x['model'],x['arm']):x['metrics'] for x in u['macro']};colors={'gcnii':'#186a8c','original':'#777777','lr':'#bb5b25'}
    names={'gcnii':'GCNII-derived','original':'Original GNN','lr':'Logistic'}
    def save(fig,name):
        fig.savefig(folder/(name+'.png'),bbox_inches='tight');fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,key in zip(axes,['auc','brier']):
        for m,offset in [('gcnii',-.2),('original',0),('lr',.2)]:
            vals=[ci[m,a][key] for a in r.ARMS];point=np.array([v['estimate'] for v in vals]);lo=np.array([v['cluster_low'] for v in vals]);hi=np.array([v['cluster_high'] for v in vals])
            ax.errorbar(np.arange(5)+offset,point,yerr=np.maximum(np.stack([point-lo,hi-point]),0),fmt='o',capsize=3,color=colors[m],label=names[m])
        ax.set_xticks(range(5),LABELS,rotation=25,ha='right');ax.set_ylabel('Equal-cohort '+key);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
        ax.axhline(ci['prior','prior'][key]['estimate'],ls=':',color='black',label='Training prior')
    fig.suptitle(ds+': subsequently motivated architecture sensitivity\nConditional 95% cohort-cluster intervals; original results unchanged',fontsize=12);save(fig,'01_performance')
    fig,axes=plt.subplots(1,2,figsize=(12,5.7),layout='constrained');site_names=sorted(lookup['gcnii','full']['cohorts'])
    for m,offset in [('gcnii',-.12),('original',.12)]:
        values=[next(v for v in seeds if v['seed_id']==0 and v['site']==s)[m+'_contrast'] for s in site_names]
        axes[0].plot(values,np.arange(len(site_names))+offset,'o',label=names[m],color=colors[m])
    axes[0].set_yticks(range(len(site_names)),site_names);axes[0].axvline(0,color='black',lw=1);axes[0].legend();axes[0].set_xlabel('Temporal-null minus rank-budget Brier')
    entries=[next(v for v in u['paired'] if v['contrast']==name)['metrics']['brier'] for name in ['gcnii_temporal_minus_budget','original_temporal_minus_budget','Gamma']]
    for j,v in enumerate(entries):axes[1].errorbar(v['estimate'],j,xerr=np.maximum([[v['estimate']-v['cluster_low']],[v['cluster_high']-v['estimate']]],0),fmt='o',capsize=4)
    axes[1].set_yticks(range(3),['GCNII-derived contrast','Original GNN contrast','Gamma (difference)']);axes[1].axvline(0,color='black',lw=1);axes[1].set_xlabel('Brier difference (conditional 95% cluster interval)')
    fig.suptitle(ds+': paired architecture contrast; primary seed');save(fig,'02_paired_architecture_contrast')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for a,label in zip(r.ARMS,LABELS):
        t=lookup['gcnii',a];bins=[x for x in t['reliability_bins'] if x['N']];axes[0].plot([x['mean_prediction'] for x in bins],[x['observed_fraction'] for x in bins],'o-',label=label)
        axes[1].hist(sets['gcnii',0,a,'first190']['p'],bins=np.linspace(0,1,26),histtype='step',label=label)
    axes[0].plot([0,1],[0,1],'k:');axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed case proportion');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Held-out probability',ylabel='Participants');axes[1].legend(fontsize=8)
    fig.suptitle(ds+': GCNII-derived descriptive calibration and score distributions\nPooled across cohorts; calibration is not deployment-risk validation',fontsize=12);save(fig,'03_calibration_distributions')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for ax,ctx in zip(axes,['native230','last190']):
        for m in ['gcnii','original','lr']:
            v=[next(x for x in st if x['model']==m and x['seed_id']==0 and x['arm']==a and x['context']==ctx)['mean_absolute_change'] for a in r.ARMS]
            ax.plot(range(5),v,'o-',label=names[m],color=colors[m])
        ax.set_xticks(range(5),LABELS,rotation=25,ha='right');ax.set_ylabel('Mean absolute probability change');ax.set_title('First 230' if ds=='ABIDE' and ctx=='native230' else ctx);ax.legend(fontsize=8)
    fig.suptitle(ds+': finite-input replay; low change is not useful stability under collapse',fontsize=11);save(fig,'04_finite_input')
    fig,axes=plt.subplots(1,3,figsize=(13,5.8),layout='constrained')
    for ax,m in zip(axes,['gcnii','original','lr']):
        a=np.array([[lookup[m,arm]['cohorts'][s]['auc'] for arm in r.ARMS] for s in site_names]);im=ax.imshow(a,vmin=.25,vmax=.85,cmap='viridis',aspect='auto')
        for i in range(len(site_names)):
            for j in range(5):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]<.55 else 'black')
        ax.set_xticks(range(5),['FC','MP','TN','RB','Fixed']);ax.set_yticks(range(len(site_names)),site_names);ax.set_title(names[m])
    fig.colorbar(im,ax=axes,label='Held-cohort AUC',shrink=.7);fig.suptitle(ds+': all held cohorts retained, including collapsed predictions');save(fig,'05_cohort_auc')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    v=[next(x for x in seeds if x['site']=='macro' and x['seed_id']==k) for k in [0,1,2]]
    for key,label in [('gcnii_contrast','GCNII-derived contrast'),('original_contrast','Original contrast'),('Gamma','Gamma')]:axes[0].plot(range(3),[x[key] for x in v],'o-',label=label)
    axes[0].set_xticks(range(3));axes[0].set(xlabel='Predefined seed ID',ylabel='Equal-cohort Brier difference');axes[0].axhline(0,color='black',lw=.8);axes[0].legend(fontsize=8)
    for m,off in [('gcnii',-.15),('original',.15)]:
        counts=[sum(x['constant_le_1e_7'] for x in collapse if x['model']==m and x['seed_id']==0 and x['arm']==a) for a in r.ARMS]
        axes[1].bar(np.arange(5)+off,counts,.3,label=names[m],color=colors[m])
    axes[1].set_xticks(range(5),LABELS,rotation=25,ha='right');axes[1].set_ylabel('Cohorts with probability range <= 1e-7');axes[1].legend(fontsize=8)
    fig.suptitle(ds+': limited seed sensitivity and primary-seed collapse');save(fig,'06_seeds_collapse')

def verify():
    cfg,roles=r.check_lock();assert r.fixtures()['passed'];checks=[];maxerr=0.;count=0
    with r.threadpool_limits(2):
        for ds,d in roles.items():
            data=r.load_data(ds,d['participants']);first=data['first190'];ranks=(first['w']>first['threshold'][:,None]).sum(1)
            for k in [0,1,2]:
                for f in d['folds']:
                    tr=np.array(f['train']);te=np.array(f['test']);budget=r.rep.fit_budget(first['w'][tr],ranks[tr])
                    for arm in r.ARMS if k==0 else ['temporal','budget']:
                        path=r.destination(ds,f['fold'],arm,k);meta=r.read(path.with_suffix('.json'));assert meta['budget']==budget and meta['protocol_sha']==r.sha(r.CFG)
                        assert meta['predictions_sha']==r.sha(path.with_suffix('.npz')) and meta['checkpoint_sha']==r.sha(path.with_suffix('.pt'))
                        rs=f['seeds']['outer'] if k==0 else f['seeds']['sensitivity'][str(k)];assert meta['seed']==meta['training_audit']['seed']==rs
                        assert meta['training_audit']['weight_update_l2']>0 and np.isfinite(meta['training_audit']['training_loss_first_last']).all()
                        if k==0:
                            best=meta['candidate_scores'][0]
                            for v in meta['candidate_scores'][1:]:
                                if v['macro_brier']<best['macro_brier']-1e-10:best=v
                            assert best==meta['selected'] and len(meta['inner'])==3
                            for j,inner in enumerate(f['inner']):
                                it=np.array(inner['train']);assert meta['inner'][j]['budget']==r.rep.fit_budget(first['w'][it],ranks[it])
                                assert meta['inner'][j]['seed']==f['seeds']['inner'][j]
                        else:assert meta['selected']==r.read(r.destination(ds,f['fold'],arm,0).with_suffix('.json'))['selected'] and not meta['retuned']
                        oldpath=(r.original.RUN/'main' if ds=='REST' else r.ABIDE/'main')/(f['fold']+'_'+arm+'.json')
                        assert r.read(oldpath)['budget']==budget
                        cp=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False);np.testing.assert_array_equal(cp['train_indices'],tr);np.testing.assert_array_equal(cp['test_indices'],te)
                        assert cp['budget']==budget and cp['hyperparameters']==meta['selected'] and cp['protocol_sha']==r.sha(r.CFG)
                        if k==0:
                            assert len(cp['inner_validation_predictions'])==4
                            for ip,candidate in zip(cp['inner_validation_predictions'],meta['candidate_scores']):
                                ix=ip['index'];assert sorted(ix.tolist())==sorted(tr.tolist())
                                yy=np.array([d['participants'][i]['y'] for i in ix]);gg=np.array([d['participants'][i]['site'] for i in ix])
                                assert abs(r.macro((ip['p']-yy)**2,gg)-candidate['macro_brier'])<1e-12
                        model=r.GCNIIComparator().to(r.DEVICE);model.load_state_dict(cp['state_dict']);model.eval()
                        with np.load(path.with_suffix('.npz')) as z:
                            for ctx in r.CONTEXTS:
                                take=z['context']==ctx;ix=z['index'][take];expected=te if ctx=='first190' else te[[d['participants'][i]['finite230'] for i in te]];np.testing.assert_array_equal(ix,expected)
                                if not len(ix):continue
                                s,rank,_=r.operators({key:v[ix] for key,v in data[ctx].items()},arm,budget);p,h=r.predict(model,s)
                                np.testing.assert_allclose(p,z['p_gnn'][take],atol=2e-6,rtol=0);np.testing.assert_allclose(h,z['embedding'][take],atol=2e-6,rtol=0);np.testing.assert_array_equal(rank,z['rank'][take])
                                np.testing.assert_array_equal(z['y'][take],[d['participants'][i]['y'] for i in ix]);maxerr=max(maxerr,float(np.max(np.abs(p-z['p_gnn'][take]))))
                                zz=z['logit'][take];np.testing.assert_allclose(1/(1+np.exp(-zz)),p,atol=2e-6,rtol=0)
                        count+=1
            del data
            sets,long=collect(ds,d);saved=csvread(r.FINAL/ds/'participant_predictions.csv');assert len(saved)==len(long)
            for a,b in zip(saved,long):
                for key in b:
                    assert (abs(float(a[key])-b[key])<1e-15 if key=='probability' else a[key]==str(b[key])),(ds,key)
            flat=csvread(r.FINAL/ds/'performance_summary.csv');original=csvread((r.original.RUN if ds=='REST' else r.ABIDE)/'final/performance_summary.csv')
            for row in flat:
                m=row['model'];arm=row['arm'];a=sets[m,0,arm,'first190'];sites=a['site'];y=a['y'];p=a['p']
                for key,fun in [('auc',lambda yy,pp:roc_auc_score(yy,pp)),('brier',lambda yy,pp:np.mean((yy-pp)**2)),('log_loss',lambda yy,pp:log_loss(yy,np.clip(pp,1e-6,1-1e-6)))]:
                    value=np.mean([fun(y[sites==s],p[sites==s]) for s in sorted(set(sites))]);assert abs(value-float(row[key]))<1e-12
                if m in ['original','lr']:
                    om='gnn' if m=='original' else 'lr';c=[z for z in original if z['model']==om and z['arm']==arm and (ds!='REST' or z['scheme']=='loso' and z['context']=='first190' and z['scope']=='primary')];assert len(c)==1
                    for key in KEYS[:3]:assert abs(float(row[key])-float(c[0][key]))<1e-12,(ds,m,arm,key)
            a=sets['gcnii',0,'temporal','first190'];y=a['y'];sites=a['site']
            v=sum(coeff*(sets[m,0,arm,'first190']['p']-y)**2 for m,arm,coeff in [('gcnii','temporal',1),('gcnii','budget',-1),('original','temporal',-1),('original','budget',1)])
            gamma=r.macro(v,sites);paired=r.read(r.FINAL/ds/'paired_contrasts.json')['performance'];assert abs(gamma-next(x for x in paired if x['contrast']=='Gamma')['metrics']['brier']['estimate'])<1e-12
            # Reproduce every conditional metric interval from locked deterministic draws.
            methods=[(m,a) for m in ['gcnii','original','lr'] for a in r.ARMS]+[('prior','prior'),('metadata','metadata')]
            ps=np.stack([sets[m,0,a,'first190']['p'] for m,a in methods]);assert uncertainty(ds,y,sites,ps,methods)==r.read(r.FINAL/ds/'metric_uncertainty.json')
            for p,h in r.read(r.FINAL/ds/'confounding.json')['metadata_source_hashes'].items():assert r.sha(p)==h
            checks.append({'dataset':ds,'participants_scored':len(y),'cohorts':len(set(sites)),'Gamma_independent_recalculation':gamma,'baseline_metrics_match_frozen':True,'conditional_intervals_reproduced':True})
            print('VERIFIED',ds,flush=True)
    assert count==162
    protected=r.read(r.HERE/'protected_originals.json');current={str(p.relative_to(r.HERE.parent)):r.sha(p) for root in [r.REST,r.ABIDE] for p in root.rglob('*') if p.is_file()}
    for p,h in r.read(r.HERE/'external_metadata_sources.json').items():assert r.sha(p)==h
    assert current==protected,'Original inventory or bytes changed'
    assert len(list((r.HERE/'fits').rglob('*.pt')))==162 and len(list((r.HERE/'fits').rglob('*.npz')))==162
    r.write(r.FINAL/'verification.json',{'status':'COMPLETE_VERIFIED_FROZEN_STOP_IMPLEMENTATION','protocol_sha':r.sha(r.CFG),'checks':checks,
        'all_162_checkpoints_replayed':True,'maximum_probability_replay_error':maxerr,'all_saved_prediction_rows_verified':True,
        'original_files_count':len(protected),'original_file_inventory_and_SHA256_unchanged':True,'mathematical_fixtures_passed':True,
        'training_budget_hyperparameters_seed_and_partitions_verified':True,'indispensable_implementation_correction_identified':False,
        'scope':'No new experiment authorized after this completed sensitivity analysis; manuscript preparation next.'})
    manifest={str(p.relative_to(r.HERE)):r.sha(p) for p in r.HERE.rglob('*') if p.is_file() and p!=r.HERE/'artifact_manifest.json'}
    r.write(r.HERE/'artifact_manifest.json',manifest)
    assert all(r.sha(r.HERE/p)==h for p,h in r.read(r.HERE/'artifact_manifest.json').items())
    print('FINAL_FREEZE',len(manifest),'artifacts;',len(protected),'original files unchanged;',count,'checkpoints replayed;',maxerr,'max probability error',flush=True)
