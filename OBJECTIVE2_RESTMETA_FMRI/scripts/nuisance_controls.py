"""Pre-specified ascertainment controls, without case-only clinical fields."""
import os
from pathlib import Path
import warnings
import numpy as np
import openpyxl
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from common import ROOT,RUN,CONFIG,read,write,sha,seed,people
from metrics import metrics

BOOK=Path(os.environ.get('REST_PHENOTYPES', str(ROOT / 'data' / 'phenotypes.xlsx')))

def main():
    rows=people(); roles=read(RUN/'participant_roles.json'); before=sha(BOOK)
    fields=['age','education_years','mean_FD_Jenkinson_whole_derivative','sex_raw_code_1','sex_raw_code_2','QC_raw_3','QC_raw_4','QC_raw_5']
    spec={'fields':fields,'no_clinical_case_only_fields':True,'imputation':'training median, no missingness indicators','selection':'same inner site folds; C grid as primary LR; macro Brier',
          'site_control':'training site prevalence when represented; otherwise overall training prevalence',
          'source_sha':before,'no_sex_code_meanings_inferred':True,'FD_scope':'mean file value; no segment temporal alignment assumed'}
    specpath=RUN/'nuisance_control_spec.json'
    if specpath.exists(): assert read(specpath)==spec
    else: write(specpath,spec)
    wb=openpyxl.load_workbook(BOOK,read_only=True,data_only=True); meta={}
    for sheet in wb:
        it=sheet.iter_rows(values_only=True); headings=next(it)
        for values in it:
            if values and values[0] is not None: meta[str(values[0]).strip()]=dict(zip(headings,values))
    wb.close(); assert sha(BOOK)==before
    features=[]
    for r in rows:
        m=meta[r['subject_id']]
        def value(name):
            try:
                v=float(m.get(name)); return v if np.isfinite(v) and v>=0 else np.nan
            except (ValueError,TypeError): return np.nan
        age=value('Age'); age=age if age>0 else np.nan
        edu=value('Education (years)'); sex=value('Sex')
        fd=np.loadtxt(r['FD_Jenkinson_path']); assert np.isfinite(fd).all()
        features.append([age,edu,float(fd.mean()),float(sex==1),float(sex==2),*[float(r['QC_score']==str(k)) for k in [3,4,5]]])
    x=np.array(features); y=np.array([r['diagnosis']=='MDD' for r in rows],int); sites=np.array([r['site'] for r in rows])
    results=[]; settings=[]
    def model(C): return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),LogisticRegression(C=C,max_iter=3000,tol=1e-6))
    with threadpool_limits(limits=2):
        for fold in roles['folds']:
            scores=[]
            for C in read(CONFIG)['models']['C_grid']:
                preds=[]; indices=[]
                for inner in fold['inner']:
                    tr=np.array(inner['train']); va=np.array(inner['validation']); fit=model(C).fit(x[tr],y[tr]); preds.extend(fit.predict_proba(x[va])[:,1]); indices.extend(va)
                ix=np.array(indices); p=np.array(preds)
                scores.append((float(np.mean([np.mean((p[sites[ix]==s]-y[ix][sites[ix]==s])**2) for s in set(sites[ix])])),C))
            _,best=min(scores); tr=np.array(fold['train']); te=np.array(fold['test']); fit=model(best).fit(x[tr],y[tr]); p=fit.predict_proba(x[te])[:,1]
            sp={s:float(y[tr][sites[tr]==s].mean()) for s in set(sites[tr])}
            ps=np.array([sp.get(s,float(y[tr].mean())) for s in sites[te]])
            for i,a,b in zip(te,p,ps): results.append({'index':int(i),'id':rows[i]['subject_id'],'site':sites[i],'y':int(y[i]),'scheme':fold['scheme'],'fold':fold['fold'],'metadata_probability':float(a),'site_probability':float(b)})
            settings.append({'fold':fold['fold'],'selected_C':best,'inner_scores':scores})
    summaries=[]
    for scheme in ['mixed','loso']:
        records=[r for r in results if r['scheme']==scheme and r['site']!='S12']; yy=np.array([r['y'] for r in records]); ss=np.array([r['site'] for r in records])
        for name in ['metadata','site']:
            pp=np.array([r[name+'_probability'] for r in records]); per={s:metrics(yy[ss==s],pp[ss==s]) for s in sorted(set(ss))}
            summaries.append({'scheme':scheme,'control':name,'pooled':metrics(yy,pp),'macro_auc':float(np.mean([m['auc'] for m in per.values()])),
                  'macro_brier':float(np.mean([m['brier'] for m in per.values()])),'sites':per})
    write(RUN/'final/nuisance_controls.json',{'spec':spec,'predictions':results,'selected_settings':settings,'summaries':summaries,'source_preserved':sha(BOOK)==before})
    print('Metadata and site controls complete; source workbook unchanged')

if __name__=='__main__': main()
