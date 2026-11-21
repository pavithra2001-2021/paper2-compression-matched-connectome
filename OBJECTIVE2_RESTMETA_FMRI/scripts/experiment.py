"""Nested subject/site validation; frozen-model segment replay; resumable fits.

Run main first, then --permutation 1..10. Each result includes person indices,
labels, context, ranks, embeddings and probabilities; no fold aggregate replaces
participant-level evidence. The cache contains only participant-local quantities.
"""
import argparse
import gc
import time
import joblib
import numpy as np
import torch
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits
from common import ROOT,RUN,CACHE,CONFIG,read,write,sha,seed,people,cache_path
from representations import ARMS,fit_budget,make_operators
from models import train_gnn,predict_gnn,train_lr,predict_lr

def load_cache(rows):
    out={}
    for context in ['first190','native230','last190']:
        fields={k:[] for k in ['w','u','threshold','T']}
        for row in rows:
            actual=context if row['T']==230 else 'first190'
            with np.load(cache_path(row,actual)) as z:
                for key in fields: fields[key].append(z[key].copy())
        out[context]={k:np.asarray(v) for k,v in fields.items()}
    return out

def operators(data,arm,budget):
    return make_operators(data['w'],data['u'],data['threshold'],data['T'],arm,budget)

def macro_brier(y,p,sites):
    return float(np.mean([np.mean((p[sites==s]-y[sites==s])**2) for s in sorted(set(sites))]))

def fit_one(fold,arm,data,y,sites,permutation=0):
    cfg=read(CONFIG); mc=cfg['models']
    name=fold['fold']+'_'+arm
    folder=RUN/('main' if permutation==0 else f'permutations/p{permutation:02d}')
    folder.mkdir(parents=True,exist_ok=True)
    done=folder/(name+'.json')
    code_hashes={p:sha(ROOT/'scripts'/p) for p in ['experiment.py','models.py','representations.py']}
    if done.exists():
        old=read(done)
        assert old['config_sha']==sha(CONFIG) and old['training_code_hashes']==code_hashes
        assert sha(folder/(name+'.npz'))==old['predictions_sha']
        return
    tr=np.asarray(fold['train']); te=np.asarray(fold['test'])
    assert not set(tr)&set(te)
    first=data['first190']; ranks=(first['w']>first['threshold'][:,None]).sum(axis=1)
    budgets=[]; candidates={}
    for wd in mc['weight_decay_grid']:
        for ep in mc['epoch_grid']: candidates[('gnn',wd,ep)]=[]
    for C in mc['C_grid']: candidates[('lr',C,0)]=[]
    inner_roles=[]
    for j,inner in enumerate(fold['inner']):
        it=np.asarray(inner['train']); iv=np.asarray(inner['validation'])
        assert not set(sites[it])&set(sites[iv])
        b=fit_budget(first['w'][it],ranks[it]); budgets.append(b)
        op,_,_=operators(first,arm,b)
        train_op=op[it]; val_op=op[iv]
        rs=seed('training',permutation,fold['fold'],'inner',j)
        for wd in mc['weight_decay_grid']:
            model,preds,audit=train_gnn(train_op,y[it],rs,wd,max(mc['epoch_grid']),val_op,mc['epoch_grid'])
            for ep,p in preds.items(): candidates[('gnn',wd,ep)].append((iv,p))
            del model
        for C in mc['C_grid']:
            model,scaler=train_lr(train_op,y[it],C,rs)
            candidates[('lr',C,0)].append((iv,predict_lr(model,scaler,val_op)))
        inner_roles.append({'inner':j,'train_indices':it.tolist(),'validation_indices':iv.tolist(),'budget':b,'seed':rs})
        del op,train_op,val_op
        print(f"inner complete p{permutation} {fold['fold']} {arm} {j+1}/3",flush=True)
    scores={}
    for key,entries in candidates.items():
        ix=np.concatenate([a for a,p in entries]); p=np.concatenate([p for a,p in entries])
        assert set(ix)==set(tr) and len(ix)==len(tr)
        scores[key]=macro_brier(y[ix],p,sites[ix])
    # Frozen tie preference: configs iterated strongest regularization first.
    def choose(kind):
        keys=[k for k in scores if k[0]==kind]; best=keys[0]
        for k in keys[1:]:
            if scores[k]<scores[best]-1e-10: best=k
        return best
    gkey=choose('gnn'); lkey=choose('lr')
    b=fit_budget(first['w'][tr],ranks[tr]); op,_,_=operators(first,arm,b)
    rs=seed('training',permutation,fold['fold'],'outer')
    gnn,_,audit=train_gnn(op[tr],y[tr],rs,gkey[1],gkey[2])
    linear,scaler=train_lr(op[tr],y[tr],lkey[1],rs)
    records={k:[] for k in ['index','context','y','p_gnn','p_lr','p_prior','embedding','rank','threshold','raw_norm','operator_norm','raw_distance','operator_distance','raw_relative_distance','operator_relative_distance','embedding_distance']}
    reference_s,reference_rank,reference_r=operators(first,arm,b)
    ref_p,ref_h=predict_gnn(gnn,reference_s[te])
    ref_embedding={int(i):h for i,h in zip(te,ref_h)}
    for context in (['first190'] if permutation else ['first190','native230','last190']):
        ix=te if context=='first190' else te[np.array([people_rows[i]['T']==230 for i in te])]
        if not len(ix): continue
        current=data[context]
        ss,kk,rr=operators({k:v[ix] for k,v in current.items()},arm,b)
        pg,hh=predict_gnn(gnn,ss); pl=predict_lr(linear,scaler,ss)
        dr=np.linalg.norm(rr-reference_r[ix],axis=(1,2)); ds=np.linalg.norm(ss-reference_s[ix],axis=(1,2))
        rn=np.linalg.norm(rr,axis=(1,2)); sn=np.linalg.norm(ss,axis=(1,2))
        rref=np.linalg.norm(reference_r[ix],axis=(1,2)); sref=np.linalg.norm(reference_s[ix],axis=(1,2))
        if arm=='temporal': th=current['threshold'][ix]
        elif arm=='mp': th=(1+np.sqrt(116/current['T'][ix]))**2
        elif arm=='budget': th=np.full(len(ix),b['cutoff'])
        else: th=np.full(len(ix),np.nan)
        values={'index':ix,'context':np.array([context]*len(ix)),'y':y[ix],'p_gnn':pg,'p_lr':pl,
                'p_prior':np.full(len(ix),y[tr].mean()),'embedding':hh,'rank':kk,'threshold':th,
                'raw_norm':rn,'operator_norm':sn,'raw_distance':dr,'operator_distance':ds,
                'raw_relative_distance':np.divide(dr,rref,out=np.full(len(ix),np.nan),where=rref>1e-12),
                'operator_relative_distance':np.divide(ds,sref,out=np.full(len(ix),np.nan),where=sref>1e-12),
                'embedding_distance':np.linalg.norm(hh-np.stack([ref_embedding[int(i)] for i in ix]),axis=1)}
        for k,v in values.items(): records[k].append(v)
    out=folder/(name+'.npz'); np.savez_compressed(out,**{k:np.concatenate(v) for k,v in records.items()})
    if permutation==0:
        torch.save({'state_dict':{k:v.cpu() for k,v in gnn.state_dict().items()},'config_sha':sha(CONFIG),
                    'train_indices':tr,'budget':b,'hyperparameters':gkey,'training_audit':audit},folder/(name+'_gnn.pt'))
        joblib.dump({'model':linear,'scaler':scaler,'train_indices':tr,'budget':b,'config_sha':sha(CONFIG)},folder/(name+'_lr.joblib'),compress=3)
    write(done,{'fold':fold['fold'],'scheme':fold['scheme'],'arm':arm,'permutation':permutation,
          'config_sha':sha(CONFIG),'training_code_hashes':code_hashes,'predictions_sha':sha(out),
          'n_train':len(tr),'n_test':len(te),'budget':b,'inner':inner_roles,
          'candidate_scores':[{'model':k[0],'regularization':k[1],'epochs':k[2],'macro_brier':v} for k,v in scores.items()],
          'selected_gnn':gkey,'selected_lr':lkey,'training_audit':audit})
    del gnn,linear,scaler; gc.collect(); torch.cuda.empty_cache()

people_rows=[]

def main():
    global people_rows
    parser=argparse.ArgumentParser(); parser.add_argument('--permutation',type=int,default=0); parser.add_argument('--fold'); parser.add_argument('--arm'); args=parser.parse_args()
    assert 0<=args.permutation<=10
    assert read(RUN/'fixture_checks.json')['passed']
    assert sha(RUN/'participant_roles.json')==read(CONFIG)['participant_roles_sha']
    assert read(RUN/'spectral_cache_manifest.json')['n_contexts']==2522
    people_rows=people(); roles=read(RUN/'participant_roles.json'); data=load_cache(people_rows)
    y=np.array([int(r['diagnosis']=='MDD') for r in people_rows]); sites=np.array([r['site'] for r in people_rows])
    if args.permutation:
        rng=np.random.default_rng(seed('label_permutation',args.permutation))
        for s in sorted(set(sites)): y[sites==s]=rng.permutation(y[sites==s])
        write(RUN/f'permutations/p{args.permutation:02d}/labels.json',{'seed':seed('label_permutation',args.permutation),'labels':y.tolist(),'subject_ids':[r['subject_id'] for r in people_rows]})
    start=time.time()
    with threadpool_limits(limits=2):
        for fold in roles['folds']:
            if args.fold and fold['fold']!=args.fold: continue
            if args.permutation and fold['scheme']!='mixed': continue
            for arm in (ARMS[:4] if args.permutation else ARMS):
                if args.arm and arm!=args.arm: continue
                t=time.time(); fit_one(fold,arm,data,y,sites,args.permutation)
                print(f"done p{args.permutation} {fold['fold']} {arm} {time.time()-t:.1f}s total {time.time()-start:.1f}s",flush=True)
    print('Requested experiment tasks complete',flush=True)

if __name__=='__main__': main()
