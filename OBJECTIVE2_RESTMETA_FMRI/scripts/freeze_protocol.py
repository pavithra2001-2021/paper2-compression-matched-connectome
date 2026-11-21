"""Apply the explicitly requested trained-GNN amendment before prediction."""
import copy
import re
import zipfile
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from common import ROOT, RUN, CONFIG, EVIDENCE, read, write, sha, seed, people

def main():
    if CONFIG.exists():
        raise RuntimeError('Frozen protocol already exists; refusing overwrite')
    old_path = ROOT / 'config/OBJECTIVE2_RESTMETA_PROPOSED_FROZEN_DESIGN_20261012.json'
    old = read(old_path)
    plan = ROOT / 'OBJECTIVE2_RESTMETA_Q1_STUDY_PLAN.md'
    assert sha(plan) == old['research_packet_hashes'][plan.name]
    with zipfile.ZipFile(ROOT / 'config/design_v1_provenance.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        z.write(plan, plan.name); z.write(old_path, old_path.name)
    objective = 'To evaluate the predictive validity, calibration, finite-scan robustness and cross-site generalizability of trained graph neural networks using autocorrelation-aware graph-spectral representations of resting-state fMRI.'
    cfg = copy.deepcopy(old)
    cfg.update(version='2.0', date='2026-10-13', status='FROZEN_BEFORE_FEATURES_AND_PREDICTIONS', formal_objective=objective,
               parent_config_sha=sha(old_path), amendment_authority='User attachment baccc533-80ba-4c8d-ab3e-1089bd52960f',
               architecture_novelty_claim=False)
    cfg['input']['features'] = 'GNN identity nodes plus signed operator S; LR upper triangle INCLUDING diagonal (6786 entries) of same S'
    cfg['input']['operator'] = 'S=R/lambda_max(full Pearson C), same scalar across arms within participant/context; no degree normalization or absolute edges; retain reconstructed diagonal'
    cfg['models'] = {'primary':'two-layer first-order spectral graph convolution', 'node_features':'I_116',
        'forward':'H1=ReLU(S W0+b0); H2=ReLU(S H1 W1+b1); h=mean_nodes(H2); logit=h w+b',
        'hidden':16, 'pooling':'mean', 'head':'linear', 'dropout':.2,
        'loss':'unweighted binary cross entropy with logits', 'optimizer':'Adam', 'learning_rate':.01,
        'weight_decay_grid':[.01,.0001], 'epoch_grid':[80,160], 'batch':'full training fold',
        'gradient_clip':5., 'seed_count':1, 'seed_rule':'same initialization/dropout stream for arms in a given fold/inner role; independent outer contexts; frozen hash seeds',
        'inner_selection':'equal validation-site macro Brier across all inner predictions',
        'tie_rule':'within 1e-10 prefer larger weight decay, then fewer epochs',
        'comparator':'L2 logistic', 'C_grid':[.001,.01,.1,1.,10.], 'scaling':'training-only StandardScaler for LR',
        'class_weight':None,'threshold':.5,'target_recalibration':False}
    cfg['primary_contrasts'] = ['GNN temporal_minus_budget_site_macro_Brier','GNN temporal_minus_budget_mean_abs_p_native230_minus_p_first190']
    cfg['secondary_models'] = ['logistic on same operators; no MLP']
    cfg['permutation_scope'] = '10 within-site shuffles; full mixed-site nested fitting for both models and all four core arms; duration replay omitted for shuffled labels; sanity only'
    cfg['embedding_analysis'] = 'Within each mixed outer fold, split outer test people by site x diagnosis into probe train/test; train scaled L2 probes there for site and diagnosis. Never concatenate embedding coordinates across separately fitted GNNs. Partial multivariate R2 diagnosis|site and site|diagnosis within folds; paired context distances within each frozen model.'
    cfg['execution_gates'] = {'QC':'Use audited investigator rule 3/4/5; official scale dictionary remains unavailable; no universal QC or clinical-quality claim',
        'categorical_fields':'Sex numeric codes retained as categories; no case-only clinical features; no guessed meanings',
        'physical_sites_TR':'Unverified; cohort-code and derivative-volume claims only',
        'source_identity':'Source hashes verified before feature construction',
        'fixtures':'Must pass before predictive training', 'predictive_hyperparameters':'Frozen before first score'}
    cfg['hypothesis'] = 'Temporal-null rank selection reduces GNN sensitivity to finite derivative length and held-cohort shift while retaining useful discrimination relative to full FC and ordinary compression. Both failure and benefit are reportable; no equivalence margin assumed.'
    cfg['amendments'] = ['Trained GNN primary; LR control per explicit user instruction',
        'Common participant/context spectral-norm scaling, reconstructed diagonals retained, LR includes diagonals to match information',
        'QC documentation uncertainty recorded as scope limitation for existing audited sample; no undocumented score semantics inferred',
        'Ten nested label permutations confined to mixed-site leakage sanity, all four core arms; no claim of LOSO permutation significance',
        'Single pre-specified optimization seed; conditional intervals omit optimizer/training uncertainty']
    rows = people(); assert len(rows) == 1188 and sum(r['diagnosis']=='MDD' for r in rows)==646
    ids = [r['subject_id'] for r in rows]; groups=np.array([r['site'] for r in rows]); y=np.array([int(r['diagnosis']=='MDD') for r in rows])
    assert len(ids)==len(set(ids))
    strat = np.array([f'{g}_{v}' for g,v in zip(groups,y)])
    mixed = StratifiedKFold(5, shuffle=True, random_state=seed('mixed_splits'))
    folds=[]
    for i,(tr,te) in enumerate(mixed.split(np.zeros(len(rows)),strat)):
        folds.append({'scheme':'mixed','fold':f'mixed_{i}','train':tr.tolist(),'test':te.tolist()})
    for s in sorted(set(groups), key=lambda s:int(s[1:])):
        folds.append({'scheme':'loso','fold':f'loso_{s}','train':np.flatnonzero(groups!=s).tolist(),'test':np.flatnonzero(groups==s).tolist()})
    for fold in folds:
        tr=np.array(fold['train']); te=np.array(fold['test'])
        assert not set(tr)&set(te)
        fold['inner']=[]
        for a,b in GroupKFold(3).split(tr,y[tr],groups[tr]):
            assert not set(groups[tr[a]])&set(groups[tr[b]])
            fold['inner'].append({'train':tr[a].tolist(),'validation':tr[b].tolist()})
    split_path = RUN / 'participant_roles.json'
    write(split_path, {'participants':[{'id':r['subject_id'],'site':r['site'],'y':int(r['diagnosis']=='MDD'),'native_T':r['T'],'source_sha':r['sha256'],'primary':r['site']!='S12'} for r in rows], 'folds':folds})
    cfg['participant_roles_sha']=sha(split_path)
    text=plan.read_text(encoding='utf-8')
    sections={
      'Objective 2':objective,
      'Main Scientific Contribution':'A controlled test of how temporal-null graph rank selection changes a genuinely trained GNN under finite-input uncertainty and held-cohort shift. The same architecture is used throughout; regularized logistic regression isolates representation effects from nonlinear graph learning. Neither graph convolution nor spectral compression is claimed as novel.',
      'Prediction Models':'''Primary: one two-layer first-order spectral graph convolution network. Fixed AAL-116 identity node features. For each reconstructed signed correlation operator R, use S=R/lambda_max(C), where C is the full participant/segment Pearson matrix. The scalar is shared across representation arms within that context. Preserve signs, reconstructed diagonal, rank and eigenvectors. No absolute edges, Laplacian substitution, added self loops, per-node degree normalization or learnable graph construction.

H1=ReLU(S W0+b0), H2=ReLU(S Dropout(H1) W1+b1), h=mean_nodes(H2), logit=linear(Dropout(h)); width 16, dropout .2. This is a trained graph-convolution network on a correlation graph shift, not a claim of standard nonnegative normalized-adjacency GCN or new architecture. Identity ROI features make anatomical correspondence explicit. BCE-with-logits, full-training-fold Adam, learning rate .01, gradient norm clip 5. Inner selection evaluates weight decay {.01,.0001} and epochs {80,160} by equal-site macro Brier. Tie within 1e-10: larger decay then fewer epochs. One deterministic seed per training role, shared across representations; optimizer uncertainty is not included in conditional confidence intervals. Test labels never determine epoch count or regularization.

Control: L2 logistic regression using all 6,786 unique entries, including reconstructed diagonals, of the same normalized operator S; StandardScaler fit on training participants only. C={.001,.01,.1,1,10}, inner site-macro Brier, tie to smaller C. No class weighting or target recalibration. Threshold .5 for both. Including diagonals amends the old 6,670-feature specification so both model families receive the same operator information. No MLP or architecture sweep.''',
      'Functional-Connectivity Construction':'''For a T-by-116 segment, sample-standardize within person and construct C=Z'Z/(T-1), symmetric unit diagonal. Descending eigendecomposition in fixed AAL coordinates. Reject nonfinite/constant required ROIs; no Fisher transform or absolute conversion. Reconstruction R=U diag(lambda * retained_mask) U'. Do not reset its diagonal. For downstream computation S=R/lambda_max(C), using the same participant/segment scalar for all arms; zero-rank R maps to zero S. Save both R and S norms/distances. GNN uses S with identity node features; logistic uses its full upper triangle including diagonal, 6,786 entries. No cohort fitting occurs in this scalar normalization.''',
      'Validation Strategy':'''Use the same fixed GNN architecture/training grid and logistic control across all representations and both schemes. All 1,188 participant roles and 19 outer splits (five mixed, fourteen held-code) are saved before training. Inner GroupKFold(3) holds entire training cohort codes apart. Every cohort-fitted cutoff, fixed rank, scaler, regularization choice and epoch choice is inner/outer training-only. Participant-local FC and phase thresholds are label-free and may be cached for held-out people. No population graph, target cohort normalization, harmonization or target calibration. Both families refit selected settings on the complete outer training set before test prediction.''',
      'Negative Controls':'''Training-prevalence intercept-only prediction is mandatory. Ten deterministic within-site diagnosis shuffles rerun the entire nested mixed-site label-dependent procedure for both model families and four core representations, reusing only label-free spectral caches and fixed subject roles. This is a leakage sanity check, not a significance test and not a held-site permutation analysis. All segments from a participant retain one permuted label; no duration replay is required in this sanity control. Include independently fitted training-budget and fixed-k controls. Verify same-participant same-K identity numerically without treating it as another scored representation. Metadata-only and site-identity diagnosis controls are secondary and never use case-only clinical fields.''',
      'Representation-to-Prediction Analysis':'''For every outer test participant/context, retain rank, threshold, raw and normalized representation norms, probability and 16-dimensional GNN graph embedding. Compute paired raw/normalized Frobenius distances, embedding distances, rank changes, probability changes, flips and rank correlations. Use within-model coordinates only: embeddings from separately trained outer models cannot be stacked for a single latent-coordinate decoder. For mixed-site test embeddings, use a frozen within-outer-test site-by-diagnosis split for secondary site and diagnosis probes (fixed C=1 L2, probe-training scaler). These probes never select or alter the GNN. Also report within-fold multivariate partial R2 for diagnosis conditional on site and site conditional on diagnosis; do not equate unadjusted dimension-dependent R2 or decoder accuracies across targets. Site association is descriptive, not causal evidence of scanner confounding.''',
      'Final Objective 2 Wording':f'**Official PhD objective:** {objective}\n\n**Viva:** Objective 1: What graph-spectral information can be trusted? Objective 2: Can a trained GNN make reliable predictions from that information when recording length and acquisition cohort change?',
      'Required Remaining Work':'Execution is now authorized by the user on 13 October 2026. Freeze the amended configuration and participant roles, pass mathematical/leakage/reproducibility fixtures, construct participant-local spectral caches, execute nested GNN/logistic comparisons and frozen context replay, then produce traceable results and final figures. QC numeric meanings and exact scanner/TR mapping remain unverified: retain the already audited selection and explicit cohort-code/derivative-volume interpretation. No guessed clinical or acquisition claims. Source/code hashes and failures must remain visible.',
      'Stop Rule':'Stop after the one-GNN/one-logistic representation study, both validation schemes, finite-input and position replay, calibration, controls and final reproducibility checks. Do not expand models or tune on held-out results. Source/identity/atlas/finite-value/leakage failures block scoring. Preserve Objective 1 and EEG archives. Negative results are retained. Conditional prediction intervals support descriptive fitted-prediction comparisons only; algorithm superiority requires additional prospective full-pipeline resampling.'}
    for name, body in sections.items():
        pat=r'(## '+re.escape(name)+r'\n).*?(?=\n## |\Z)'
        text, n=re.subn(pat, lambda m:m.group(1)+'\n'+body+'\n',text,flags=re.S)
        assert n==1,name
    text=text.replace('12 October 2026. Proposed frozen design, version 1.0. This formal fMRI direction replaces the prior EEG designation. Audit and design only: no FC/surrogate prediction pipeline, trained model, harmonization, hyperparameter evaluation or new dataset download has been executed. Objective 1 and the archived EEG branch remain unchanged. Counts below distinguish the authoritative measurement cohort from the newly specified predictive cohort. This is a local dated design lock, not a public preregistration.',
       '13 October 2026. Frozen execution protocol, version 2.0. User-authorized amendment makes one trained GNN primary and logistic regression the control. Frozen before feature construction or predictive results. Version 1 is preserved in config/design_v1_provenance.zip; other audit documents remain historical provenance. This is a local dated lock, not a public preregistration. Objective 1 and EEG archives remain unchanged.')
    text=text.replace('training scaler and logistic weights','training scaler and frozen GNN/logistic weights')
    text=text.replace('select C using three site-grouped inner folds','select GNN decay/epochs and logistic C using three site-grouped inner folds')
    text=text.replace('logistic family, C grid','GNN/logistic families and frozen grids')
    text=text.replace('source score-scale documentation must be recorded before execution','unavailable source score-scale documentation is recorded as an unresolved limitation; the investigator selection is unchanged')
    assert len(re.findall(r'^## ',text,re.M))==44
    plan.write_text(text,encoding='utf-8')
    cfg['research_packet_hashes'][plan.name]=sha(plan)
    cfg['constructor_sha']=sha(ROOT/'scripts/representations.py')
    write(CONFIG,cfg)
    print('Frozen',sha(CONFIG),'participants',len(rows),'folds',len(folds))

if __name__=='__main__': main()
