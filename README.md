# Paper 2 research code

A compression-matched evaluation of individualized spectral selection for multisite functional connectome prediction.

## Scope

Code for the frozen REST-meta-MDD primary experiment, separately trained ABIDE-I ASD replication, and explicitly post-hoc sensitivity analyses. This is a code-only distribution. Source scripts preserve the frozen algorithms. Two release copies replace workstation-specific data paths with environment-variable configuration; all other scripts are byte-identical to their source. Public configuration copies redact local paths and administrative provenance. The code provenance file records source hashes.

## Environment

Python 3.12. The recorded ABIDE execution used PyTorch 2.7.1+cu128, CUDA 12.8 and an NVIDIA GeForce RTX 4060 Laptop GPU. Install the declared Python dependencies in a separate environment. Hardware and package changes can affect numerical reproducibility.

## Data and restricted prerequisites

Obtain REST-meta-MDD derivatives and phenotypes from https://rfmri.org/REST-meta-MDD under its data-use agreement. Obtain ABIDE-I DPARSF filt_noglobal AAL derivatives and provider phenotype/QC records through https://preprocessed-connectomes-project.org/abide/. No datasets, participant identifiers, individual predictions, checkpoints, private credentials, or participant manifests are distributed. Exact subject-level replay requires the authorized local eligibility and partition manifests and original data hashes. The public package alone cannot reproduce subject-level results without those prerequisites.

## Execution

Preserve the three sibling project directories. Configure local data paths in a working copy. Set ABIDE_ROOT to the locally obtained ABIDE_pcp folder and REST_PHENOTYPES to the local REST phenotype workbook. REST common.py locates the local eligibility manifest under evidence/local_only/inspection_v2/prediction_eligibility_manifest.json. REST scripts freeze_protocol.py and prepare_spectra.py establish roles and spectral caches before experiment.py trains and scores. analyze_results.py and metric_uncertainty.py summarize stored predictions. Follow replication.py stages for the separately trained ABIDE analysis. study.py implements the GCNII-derived comparator, with analysis.py providing its summaries. The supplementary run_sensitivity.py implements the post-hoc analysis. Read each executable's argparse entry point before use. Never run these commands on the original frozen evidence tree.

## Interpretation

The primary comparison did not demonstrate a temporal-null advantage over matched compression. Conditional bootstrap intervals omit retraining and model-selection uncertainty. ABIDE is an ASD methodological replication. It supplies no independent MDD clinical validation. The later architecture analyses remain exploratory.

## Version

v1.0.0. Author-approved public code release. No reuse license has been added without rights-holder confirmation. Restricted-data permissions remain with the providers.
