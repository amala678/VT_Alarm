"""Generate portable figures and a Markdown research report from frozen runs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def report(cache,out,frozen,frame,x,probabilities,rows):
    dest=Path(out)/'evaluation'; synthetic=frozen['provenance']['synthetic']
    banner='SYNTHETIC SOFTWARE DEMO — NOT RESEARCH RESULTS' if synthetic else 'VTaC v1.1 held-out evaluation'
    if not synthetic and frozen['provenance'].get('available_headers',0)<frozen['provenance'].get('official_events',0):
        banner='PARTIAL REAL-DATA SMOKE CHECK — NOT THE COMPLETED STUDY'
    first=frozen['seeds'][0]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for mode in ('ecg','fusion','shuffled'):
        p=probabilities[f'{mode}_{first}']; y=frame.label.to_numpy()
        fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p)
        axes[0].plot(fpr,tpr,label=mode); axes[1].plot(recall,precision,label=mode)
    axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='ROC')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Precision–recall')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(banner,fontsize=10); fig.tight_layout(); fig.savefig(dest/'curves.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for ax,mode in zip(axes,('ecg','fusion','shuffled')):
        for seed in frozen['seeds']:
            h=pd.read_csv(Path(out)/f'{mode}_{seed}_learning.csv')
            ax.plot(h.epoch,h.validation_auprc,label=str(seed))
        ax.set(title=mode,xlabel='Epoch',ylabel='Validation AUPRC'); ax.legend()
    fig.tight_layout(); fig.savefig(dest/'learning_curves.png',dpi=150); plt.close(fig)
    # Deterministic examples, selected only for reporting after all choices are frozen.
    ep=probabilities[f'ecg_{first}']; fp=probabilities[f'fusion_{first}']
    et=frozen['runs'][f'ecg_{first}']['threshold']; ft=frozen['runs'][f'fusion_{first}']['threshold']
    y=frame.label.to_numpy(); categories={'false_negative':(y==1)&(fp<ft),'false_positive':(y==0)&(fp>=ft),
                                       'disagreement':(ep>=et)!=(fp>=ft)}
    examples=[]
    for category,mask in categories.items():
        for i in np.flatnonzero(mask)[:2]:
            fig,axes=plt.subplots(2,1,figsize=(10,4),sharex=True)
            t=np.arange(x.shape[-1])/frozen['provenance']['fs']-frozen['provenance']['seconds']
            for c,label in enumerate(('ECG','PPG')): axes[c].plot(t,x[i,c],lw=.6); axes[c].set_ylabel(label)
            axes[-1].set_xlabel('Seconds relative to alarm')
            fig.suptitle(f'{category}: {frame.event.iloc[i]} | true={y[i]} | ECG={ep[i]:.3f}, fusion={fp[i]:.3f}')
            fig.tight_layout(); filename=f'{category}_{i}.png'; fig.savefig(dest/filename,dpi=120); plt.close(fig)
            examples.append(f'![{category}]({filename})')
    table=pd.DataFrame(rows)
    columns=['model','threshold','n','true','sensitivity','specificity','auroc','auprc','fn','fp']
    def markdown(df):
        header='| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
        return header+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
    cohort=pd.read_csv(Path(cache)/'paired.csv').groupby('split').agg(events=('event','size'),true=('label','sum'),records=('record','nunique')).reset_index()
    intervals=[]
    for seed in frozen['seeds']:
        b=json.loads((dest/f'bootstrap_{seed}.json').read_text())['overall']['fusion_minus_ecg']
        for metric in ('specificity','sensitivity'):
            r=b[metric]; intervals.append({'seed':seed,'difference':metric,**r})
    text=f'''# Reducing false VT alarms\n\n{banner}\n\n## Cohort\n\n{markdown(cohort)}\n\n## Frozen operating points\n\nTrue VT is positive. Thresholds were selected on validation data to maximize specificity subject to sensitivity >= {frozen['target']:.0%}; ties select the lower threshold. Test sensitivity is measured, not guaranteed.\n\n{markdown(table[columns])}\n\n## Paired uncertainty\n\nFusion minus ECG; record-clustered percentile 95% intervals, frozen thresholds, identical draws. These intervals condition on each fitted model, and do not capture retraining uncertainty.\n\n{markdown(pd.DataFrame(intervals))}\n\n## Interpretation\n\nCompare sensitivity and missed true-event counts together with false-alarm rejection. A specificity gain with lower sensitivity may be a tradeoff. Inspect bootstrap intervals and quality subgroup counts before claiming a benefit. The shuffled control breaks target-event pairing and target-event PPG information together; it does not isolate timing alone.\n\nQuality cutoffs are training-set medians of waveform proxy scores. They are not clinical quality annotations. Small groups have unstable estimates; no post-test pooling or rule changes are applied. Per-modality and joint ECG/PPG group metrics and intervals are in bootstrap_SEED.json.\n\n## Figures\n\n![ROC and PR](curves.png)\n\n![Learning curves](learning_curves.png)\n\n## Error examples\n\n'''+('\n\n'.join(examples) if examples else 'No errors/disagreements selected.')+'''\n\n## Limits\n\nThis is an offline research pipeline, not a clinical alarm filter. A 95% validation sensitivity target is not a safety standard. The peak-derived features and waveform quality scores are approximate research proxies. No ABP or external-validation result is included. Synthetic outputs, when marked above, only verify software execution.\n\n## References\n\n- VTaC v1.1: https://physionet.org/content/vtac/1.1/\n- Lehman et al., VTaC, NeurIPS 2023: https://github.com/ML-Health/VTaC\n'''
    (dest/'REPORT.md').write_text(text)
