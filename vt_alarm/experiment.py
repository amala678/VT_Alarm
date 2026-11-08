import hashlib
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from .evaluation import select_threshold, metrics, shuffled_indices, bootstrap


def write_json(path, value):
    def clean(x):
        if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)): return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x): return None
        if isinstance(x,np.generic): return x.item()
        return x
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False))


def fingerprint(cache):
    return {name:hashlib.sha256((Path(cache)/name).read_bytes()).hexdigest()
            for name in ('paired.csv','paired.npz','provenance.json','ppg_missing.csv','ppg_missing.npz') if (Path(cache)/name).exists()}


def load(cache):
    cache=Path(cache); df=pd.read_csv(cache/'paired.csv',dtype={'record':str,'event':str})
    data=np.load(cache/'paired.npz'); x=data['x']; f=data['features']
    if len(df)!=len(x) or len(f)!=len(df) or not np.isfinite(x).all() or not np.isfinite(f).all():
        raise ValueError('Invalid cache')
    if df.event.duplicated().any() or (df.groupby('record').split.nunique()>1).any():
        raise ValueError('Duplicate events or record leakage')
    for s in ('train','validation','test'):
        if set(df.loc[df.split==s,'label'])!={0,1}: raise ValueError(f'{s} needs both classes')
    return df,x,f


def inputs(x,frame,mode,shuffle_seed):
    if mode=='ecg': return x[:,:1].copy(),None
    x=x.copy()
    if mode=='shuffled':
        donors=shuffled_indices(frame.record.to_numpy(),shuffle_seed)
        x[:,1]=x[donors,1]
        return x,donors
    return x,None


def train(cache,out,seeds=(11,22,33),epochs=40,target=.95,quick=False,device='cpu'):
    import torch
    from .models import fit,predict
    torch.set_num_threads(2)
    out=Path(out)
    if (out/'frozen.json').exists(): raise ValueError('Frozen run already exists; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    df,x,f=load(cache); masks={s:df.split.eq(s).to_numpy() for s in ('train','validation','test')}
    tr,va=masks['train'],masks['validation']; y=df.label.to_numpy()
    # Feasibility is a record-count check only; no test predictions or outcome-driven selection.
    for split,mask in masks.items():
        shuffled_indices(df.loc[mask,'record'].to_numpy(),703)
    for s in ('train','validation'):
        if set(y[masks[s]])!={0,1}: raise ValueError(f'{s} needs both labels')
    provenance=json.loads((Path(cache)/'provenance.json').read_text())
    frozen={'target':target,'seeds':list(seeds),'cache_fingerprint':fingerprint(cache),
            'provenance':provenance,'python':platform.python_version(),'torch':torch.__version__,
            'quality_cutoffs':{k:float(df.loc[tr,k].median()) for k in ('ecg_quality','ppg_quality')},
            'runs':{},'search':{},'selection':'validation AUPRC; earliest epoch wins ties'}
    # Logistic regression tuning uses the same paired training/validation cohort.
    trials=[]; best=None
    for c in (.1,1.,10.):
        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=2000,random_state=seeds[0]))
        model.fit(f[tr],y[tr]); p=model.predict_proba(f[va])[:,1]
        score=average_precision_score(y[va],p); trials.append({'C':c,'validation_auprc':score})
        if best is None or score>best[0]: best=(score,model,c,p)
    _,baseline,c,p=best; joblib.dump(baseline,out/'baseline.joblib')
    frozen['baseline']={'C':c,'threshold':select_threshold(y[va],p,target),'trials':trials}
    pd.DataFrame({'event':df.loc[va,'event'],'label':y[va],'probability':p}).to_csv(out/'baseline_validation.csv',index=False)
    grid=[(1e-3,1e-4)] if quick else [(1e-3,1e-4),(3e-4,1e-4),(1e-3,1e-3)]
    for mode in ('ecg','fusion','shuffled'):
        tx,td=inputs(x[tr],df.loc[tr],mode,701)
        vx,vd=inputs(x[va],df.loc[va],mode,702)
        if td is not None:
            for s,mask,donors in [('train',tr,td),('validation',va,vd)]:
                frame=df.loc[mask].reset_index(drop=True)
                pd.DataFrame({'event':frame.event,'record':frame.record,'donor_event':frame.event.iloc[donors].to_numpy(),
                              'donor_record':frame.record.iloc[donors].to_numpy()}).to_csv(out/f'shuffle_{s}.csv',index=False)
        trials=[]; winner=None
        print(f'Training {mode}',flush=True)
        for lr,wd in grid:
            start=time.perf_counter()
            model,history,score=fit(tx,y[tr],vx,y[va],seeds[0],lr,wd,epochs,device=device)
            trial={'lr':lr,'weight_decay':wd,'validation_auprc':score,'seconds':time.perf_counter()-start}
            trials.append(trial)
            if winner is None or score>winner[0]: winner=(score,model,history,lr,wd)
        score,first,first_history,lr,wd=winner
        frozen['search'][mode]=trials
        for index,seed in enumerate(seeds):
            start=time.perf_counter()
            if index==0: model,history=first,first_history
            else: model,history,_=fit(tx,y[tr],vx,y[va],seed,lr,wd,epochs,device=device)
            vp=predict(model,vx,device); threshold=select_threshold(y[va],vp,target)
            name=f'{mode}_{seed}'; torch.save(model.state_dict(),out/f'{name}.pt')
            pd.DataFrame(history).to_csv(out/f'{name}_learning.csv',index=False)
            pd.DataFrame({'event':df.loc[va,'event'],'label':y[va],'probability':vp}).to_csv(out/f'{name}_validation.csv',index=False)
            frozen['runs'][name]={'mode':mode,'seed':seed,'lr':lr,'weight_decay':wd,
                                 'threshold':threshold,'parameters':sum(p.numel() for p in model.parameters()),
                                 'repeat_seconds':time.perf_counter()-start,'best_epoch':max(history,key=lambda h:h['validation_auprc'])['epoch'],
                                 'validation':metrics(y[va],vp,threshold)}
    # Freeze artifact hashes; evaluation refuses changed models or baseline.
    frozen['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.suffix in ('.pt','.joblib')}
    write_json(out/'frozen.json',frozen)
    print('Training and validation complete. Frozen settings saved; test predictions have not been calculated.',flush=True)


def evaluate(cache,out,repetitions=1000,device='cpu'):
    import torch
    from .models import CompactCNN,predict
    from .report import report
    torch.set_num_threads(2)
    out=Path(out); dest=out/'evaluation'
    if dest.exists(): raise ValueError('Evaluation already exists. Preserve it; do not tune against test results.')
    frozen=json.loads((out/'frozen.json').read_text())
    if fingerprint(cache)!=frozen['cache_fingerprint']: raise ValueError('Cache changed after freezing')
    for name,digest in frozen['artifact_sha256'].items():
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=digest: raise ValueError('Model changed after freezing')
    df,x,f=load(cache); mask=df.split.eq('test').to_numpy(); frame=df.loc[mask].reset_index(drop=True)
    y=frame.label.to_numpy(); dest.mkdir()
    thresholds={'baseline':frozen['baseline']['threshold']}
    probabilities={'baseline':joblib.load(out/'baseline.joblib').predict_proba(f[mask])[:,1]}
    full={}; rows=[]
    for name,settings in frozen['runs'].items():
        mode=settings['mode']; tx,donors=inputs(x[mask],frame,mode,703)
        if donors is not None:
            pd.DataFrame({'event':frame.event,'record':frame.record,'donor_event':frame.event.iloc[donors].to_numpy(),
                          'donor_record':frame.record.iloc[donors].to_numpy()}).to_csv(dest/'shuffle_test.csv',index=False)
        model=CompactCNN(tx.shape[1]).to(device)
        model.load_state_dict(torch.load(out/f'{name}.pt',map_location=device,weights_only=True))
        full[name]=predict(model,tx,device)
        rows.append({'model':name,'threshold':settings['threshold'],**metrics(y,full[name],settings['threshold'])})
    rows.append({'model':'baseline','threshold':thresholds['baseline'],**metrics(y,probabilities['baseline'],thresholds['baseline'])})
    pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
    repeated=pd.DataFrame([r for r in rows if r['model']!='baseline'])
    repeated['mode']=repeated.model.str.rsplit('_',n=1).str[0]
    repeated.groupby('mode')[['sensitivity','specificity','auroc','auprc']].agg(['mean','std','min','max']).to_csv(dest/'seed_summary.csv')
    for seed in frozen['seeds']:
        probs={mode:full[f'{mode}_{seed}'] for mode in ('ecg','fusion','shuffled')}
        ts={mode:frozen['runs'][f'{mode}_{seed}']['threshold'] for mode in probs}
        result={'overall':bootstrap(frame,probs,ts,repetitions,seed=2026)}
        # Cutoffs were established using training only; observed quality, not missingness.
        for channel in ('ecg','ppg'):
            cut=frozen['quality_cutoffs'][channel+'_quality']
            for group,m in [('low',frame[channel+'_quality'].to_numpy()<cut),('high',frame[channel+'_quality'].to_numpy()>=cut)]:
                result[f'{channel}_{group}']={'n':int(m.sum()),'true':int(y[m].sum()),
                    'metrics':{k:metrics(y[m],v[m],ts[k]) for k,v in probs.items()},
                    'bootstrap':bootstrap(frame.loc[m].reset_index(drop=True),{k:v[m] for k,v in probs.items()},ts,repetitions,seed=2026) if m.any() else None}
        for egroup in ('low','high'):
            for pgroup in ('low','high'):
                em=frame.ecg_quality.to_numpy()<frozen['quality_cutoffs']['ecg_quality']
                pm=frame.ppg_quality.to_numpy()<frozen['quality_cutoffs']['ppg_quality']
                m=(em if egroup=='low' else ~em)&(pm if pgroup=='low' else ~pm)
                result[f'ecg_{egroup}_ppg_{pgroup}']={'n':int(m.sum()),'true':int(y[m].sum()),
                    'metrics':{k:metrics(y[m],v[m],ts[k]) for k,v in probs.items()},
                    'bootstrap':bootstrap(frame.loc[m].reset_index(drop=True),{k:v[m] for k,v in probs.items()},ts,repetitions,seed=2026) if m.any() else None}
        write_json(dest/f'bootstrap_{seed}.json',result)
    # Baseline uncertainty is also record-clustered.
    write_json(dest/'bootstrap_baseline.json',bootstrap(frame,probabilities,thresholds,repetitions))
    pred=frame.copy()
    for name,p in {**probabilities,**full}.items(): pred[name]=p
    pred.to_csv(dest/'predictions.csv',index=False)
    # PPG-missing events are a separate test population; models and thresholds stay unchanged.
    missing_path=Path(cache)/'ppg_missing.csv'
    if missing_path.exists():
        md=pd.read_csv(missing_path,dtype={'record':str,'event':str}); mm=md.split.eq('test').to_numpy()
        if mm.any():
            mz=np.load(Path(cache)/'ppg_missing.npz'); mf=md.loc[mm].reset_index(drop=True)
            missing_results={}; missing_pred=mf.copy()
            for seed in frozen['seeds']:
                name=f'ecg_{seed}'; model=CompactCNN(1).to(device)
                model.load_state_dict(torch.load(out/f'{name}.pt',map_location=device,weights_only=True))
                p=predict(model,mz['x'][mm,None,:],device); t=frozen['runs'][name]['threshold']
                missing_pred[name]=p
                missing_results[name]={'metrics':metrics(mf.label.to_numpy(),p,t),
                    'bootstrap':bootstrap(mf,{name:p},{name:t},repetitions)}
            missing_pred.to_csv(dest/'ppg_missing_predictions.csv',index=False)
            write_json(dest/'ppg_missing_results.json',missing_results)
    report(cache,out,frozen,frame,x[mask],full,rows)
    print(f'Evaluation saved to {dest}',flush=True)
