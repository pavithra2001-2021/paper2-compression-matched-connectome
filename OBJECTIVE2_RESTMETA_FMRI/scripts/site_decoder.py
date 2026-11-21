"""Known-site representation decoder, separate from the diagnosis models."""
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,log_loss
from threadpoolctl import threadpool_limits
from common import RUN,CONFIG,read,write,sha,people
from representations import ARMS,fit_budget,vectorize
from experiment import load_cache,operators

def main():
    spec={'model':'multinomial L2 logistic','C':1.,'tuning':'none; fixed secondary probe','splits':'same five outer mixed participant folds',
          'scaling':'training-only standardization','budget':'each outer training set only','labels':'cohort codes; all 14 are supported as site classes',
          'interpretation':'representation association, not evidence diagnosis model causally relies on acquisition'}
    path=RUN/'site_decoder_spec.json'
    if path.exists(): assert read(path)==spec
    else: write(path,spec)
    rows=people(); roles=read(RUN/'participant_roles.json'); data=load_cache(rows)['first190']
    sites=np.array([r['site'] for r in rows]); y=np.array([r['diagnosis']=='MDD' for r in rows],int)
    ranks=(data['w']>data['threshold'][:,None]).sum(axis=1); records=[]
    with threadpool_limits(limits=2):
        for fold in roles['folds']:
            if fold['scheme']!='mixed': continue
            tr=np.array(fold['train']); te=np.array(fold['test']); b=fit_budget(data['w'][tr],ranks[tr])
            for arm in ARMS:
                s,_,_=operators(data,arm,b); scaler=StandardScaler().fit(vectorize(s[tr]))
                m=LogisticRegression(C=1,max_iter=3000,tol=1e-5).fit(scaler.transform(vectorize(s[tr])),sites[tr])
                assert np.max(m.n_iter_)<3000
                prob=m.predict_proba(scaler.transform(vectorize(s[te]))); pred=m.classes_[prob.argmax(axis=1)]
                for i,p,pr in zip(te,prob,pred): records.append({'index':int(i),'site':sites[i],'diagnosis':int(y[i]),'arm':arm,'fold':fold['fold'],'prediction':pr,
                      'probabilities':p.tolist(),'classes':m.classes_.tolist()})
                folder=RUN/'secondary_checkpoints'; folder.mkdir(exist_ok=True)
                joblib.dump({'model':m,'scaler':scaler,'budget':b,'train_indices':tr,'config_sha':sha(CONFIG)},folder/(fold['fold']+'_'+arm+'_site.joblib'),compress=3)
            print('site decoder',fold['fold'],'complete',flush=True)
    summaries=[]
    for arm in ARMS:
        rr=[r for r in records if r['arm']==arm]; truth=np.array([r['site'] for r in rr]); pred=np.array([r['prediction'] for r in rr]); yy=np.array([r['diagnosis'] for r in rr])
        standard=float(np.mean([np.mean([np.mean(pred[(truth==s)&(yy==v)]==s) for v in [0,1]]) for s in set(truth)]))
        summaries.append({'arm':arm,'N':len(rr),'classes':14,'chance_balanced_accuracy':1/14,
          'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
          'diagnosis_standardized_balanced_accuracy':standard,
          'multiclass_log_loss':float(log_loss(truth,np.array([r['probabilities'] for r in rr]),labels=rr[0]['classes']))})
    write(RUN/'final/site_decoding.json',{'spec':spec,'summaries':summaries,'predictions':records})

if __name__=='__main__': main()
