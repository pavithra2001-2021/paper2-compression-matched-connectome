"""Independent checks for score ties, conditional pairing and train-only fitting."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from common import RUN,write
from representations import fit_budget
from metric_uncertainty import weighted_auc
from metrics import paired_interval,calibration

def main():
    y=np.array([0,1,0,1,0,1]); p=np.array([.1,.2,.2,.7,.7,.7])
    weights=np.array([[1,1,1,1,1,1],[2,1,0,2,1,0]],float)
    for w,a in zip(weights,weighted_auc(y,p,weights)):
        assert abs(a-roc_auc_score(y,p,sample_weight=w))<1e-12
    assert np.all(weighted_auc(y,np.ones(6)*.5,weights)==.5)
    sites=np.array(['A','A','B','B','C','C']); v=np.full(6,.03)
    cis=paired_interval(v,y,sites,7,B=50,macro=True)
    for ci in cis.values(): assert abs(ci['low']-.03)<1e-12 and abs(ci['high']-.03)<1e-12
    c=calibration(y,np.ones(6)*.5)
    assert c['status']=='constant_predictions' and c['slope'] is None and c['offset']==0
    whole=np.array([[4.,2.,2.,1.],[4.,2.,2.,1.],[40.,20.,10.,1.]])
    train=[0,1]; ranks=np.array([2,2,3]); before=fit_budget(whole[train],ranks[train])
    assert before['target']==4 and before['attained']==2
    whole[2]=[10000.,100.,50.,10.]
    assert before==fit_budget(whole[train],ranks[train])
    x=np.arange(30,dtype=float).reshape(10,3); scaler=StandardScaler().fit(x[:7]); original_mean=scaler.mean_.copy()
    x[7:]*=10000; scaler.transform(x[7:]); np.testing.assert_array_equal(original_mean,scaler.mean_)
    write(RUN/'statistical_fixture_checks.json',{'passed':True,'checks':['weighted tied-score AUC matches sklearn','constant-score AUC .5','paired constant-difference bootstrap','undefined constant-prediction slope handled','heldout data perturbation cannot change training budget','equidistant pooled mode-count ties choose fewer modes','heldout transform cannot change training scaler']})
    print('Statistical and train-only fixtures passed')

if __name__=='__main__': main()
