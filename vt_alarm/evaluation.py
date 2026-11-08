import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def metrics(y, p, threshold):
    y=np.asarray(y); p=np.asarray(p)
    tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
    both=len(np.unique(y))==2
    return dict(n=len(y),true=int(y.sum()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                sensitivity=float(tp/(tp+fn)) if tp+fn else float('nan'),
                specificity=float(tn/(tn+fp)) if tn+fp else float('nan'),
                precision=float(tp/(tp+fp)) if tp+fp else 0.,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
                auroc=float(roc_auc_score(y,p)) if both else float('nan'),
                auprc=float(average_precision_score(y,p)) if both else float('nan'))


def select_threshold(y, p, target=.95):
    y=np.asarray(y); p=np.asarray(p)
    if not 0<target<=1 or set(np.unique(y))!={0,1} or not np.isfinite(p).all():
        raise ValueError('Threshold selection requires both classes, finite probabilities, and target in (0,1]')
    # Specificity is monotone in threshold. Search by rank rather than rounding probabilities.
    candidates=np.unique(np.r_[0.,p])
    feasible=[(metrics(y,p,t)['specificity'],float(t)) for t in candidates
              if np.mean(p[y==1]>=t)>=target]
    best=max(s for s,t in feasible)
    return min(t for s,t in feasible if s==best)


def shuffled_indices(groups, seed):
    """One-to-one permutation with no donor from the target patient record."""
    groups=np.asarray(groups); n=len(groups)
    if n<2:
        raise ValueError('Need at least two records to shuffle PPG')
    _,counts=np.unique(groups,return_counts=True)
    if counts.max()>n/2:
        raise ValueError('Cross-record one-to-one shuffle impossible: a record holds >half the events')
    rng=np.random.default_rng(seed)
    # Sorted contiguous record blocks rotated by maximum block size guarantee cross-record donors.
    unique=rng.permutation(np.unique(groups))
    order=np.concatenate([rng.permutation(np.flatnonzero(groups==g)) for g in unique])
    donors=np.empty(n,dtype=int); donors[order]=np.roll(order,int(counts.max()))
    assert np.all(groups[donors]!=groups) and len(np.unique(donors))==n
    return donors


def bootstrap(df, probabilities, thresholds, repetitions=1000, seed=2026):
    """Same cluster draws for all models and differences; thresholds remain frozen."""
    groups=df.record.to_numpy(); unique=np.unique(groups); y=df.label.to_numpy()
    blocks={g:np.flatnonzero(groups==g) for g in unique}; rng=np.random.default_rng(seed)
    keys=['sensitivity','specificity','auroc','auprc','precision','f1']
    values={name:{k:[] for k in keys} for name in probabilities}
    differences={k:[] for k in keys}; excluded={name:{k:0 for k in keys} for name in probabilities}
    for _ in range(repetitions):
        idx=np.concatenate([blocks[g] for g in rng.choice(unique,len(unique),replace=True)])
        results={name:metrics(y[idx],p[idx],thresholds[name]) for name,p in probabilities.items()}
        for name,r in results.items():
            for k in keys:
                if np.isfinite(r[k]): values[name][k].append(r[k])
                else: excluded[name][k]+=1
        if 'ecg' in results and 'fusion' in results:
            for k in keys:
                delta=results['fusion'][k]-results['ecg'][k]
                if np.isfinite(delta): differences[k].append(delta)
    def interval(vals):
        return {'low':float(np.percentile(vals,2.5)),'high':float(np.percentile(vals,97.5)),
                'valid_replicates':len(vals),'excluded':repetitions-len(vals)} if vals else {
                    'low':None,'high':None,'valid_replicates':0,'excluded':repetitions}
    return {'repetitions':repetitions,'seed':seed,
            'models':{n:{k:interval(v) for k,v in r.items()} for n,r in values.items()},
            'fusion_minus_ecg':{k:interval(v) for k,v in differences.items()}}
