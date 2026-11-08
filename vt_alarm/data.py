import hashlib
import json
from pathlib import Path
import urllib.request
import shutil
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .signals import prepare, ECG_PRIORITY

BASE = 'https://physionet.org/files/vtac/1.1/'

def download_file(url, dest):
    import time
    import urllib.error

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip files that are already downloaded
    if dest.exists() and dest.stat().st_size > 0:
        return

    temp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with temp.open("wb") as stream:
                    shutil.copyfileobj(response, stream)

            temp.replace(dest)
            return

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError
        ) as error:

            temp.unlink(missing_ok=True)

            if attempt == 5:
                print(f"FAILED after 6 attempts: {url}", flush=True)
                raise

            delay = min(2 ** attempt * 2, 30)

            print(
                f"Download failed: {error}. "
                f"Retry {attempt + 1}/5 in {delay}s",
                flush=True
            )

            time.sleep(delay)

def metadata(root, split_column=None):
    root = Path(root)
    labels = pd.read_csv(root/'event_labels.csv', dtype=str)
    splits = pd.read_csv(root/'benchmark_data_split.csv', dtype=str)
    for frame in (labels, splits):
        frame.columns = frame.columns.str.strip().str.lower()
        for col in frame:
            frame[col] = frame[col].str.strip()
    keys = ['record', 'event'] if 'event' in splits else ['record']
    if split_column is None:
        candidates = [c for c in splits if c not in ('record','event','decision') and
                      set(splits[c].str.lower()) <= {'train','training','val','valid','validation','test','testing'}]
        if len(candidates) != 1:
            raise ValueError(f'Cannot identify official split column in {list(splits)}. Use --split-column NAME; numeric encodings require an explicitly mapped CSV.')
        split_column = candidates[0]
    if labels.duplicated(['record','event']).any() or splits.duplicated(keys).any():
        raise ValueError('Duplicate event/split keys')
    merged = labels.merge(splits[keys+[split_column]].rename(columns={split_column:'split'}), on=keys, how='left', validate='many_to_one')
    merged['split'] = merged['split'].str.lower().replace({'training':'train','val':'validation','valid':'validation','testing':'test'})
    if merged['split'].isna().any() or not set(merged['split']) <= {'train','validation','test'}:
        raise ValueError('Unmapped labels or invalid split values')
    merged['label'] = merged.decision.str.lower().map({'true':1,'false':0})
    if merged.label.isna().any():
        raise ValueError('Labels must be True or False')
    if (merged.groupby('record').split.nunique()>1).any():
        raise ValueError('Record leakage across official splits')
    return merged

def download(root, limit=None, split_column=None):
    root = Path(root)
    for name in ('event_labels.csv','benchmark_data_split.csv','LICENSE.txt'):
        download_file(BASE+name, root/name)
    df = metadata(root, split_column)
    if limit:
        df = df.groupby('split', group_keys=False).head(limit)
    def fetch_event(row):
        relative = f'annotated_waveforms/{row.record}/{row.event}'
        download_file(BASE+relative+'.hea', root/(relative+'.hea'))
        header = (root/(relative+'.hea')).read_text().splitlines()
        nsig = int(header[0].split()[1])
        for line in header[1:1+nsig]:
            name = line.split()[0]
            if name != '~':
                if Path(name).name != name:
                    raise ValueError('Unexpected waveform filename in header')
                download_file(BASE+f'annotated_waveforms/{row.record}/'+name, root/'annotated_waveforms'/row.record/name)
    with ThreadPoolExecutor(max_workers=1) as pool:
        list(pool.map(fetch_event, df.itertuples()))
    print(f'Downloaded {len(df)} events; metadata retains the complete official allocation.')

def build_cache(root, output, seconds=30, split_column=None):
    import wfdb
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'paired.npz').exists():
        raise ValueError('Cache already exists; use a fresh directory to preserve previous runs')
    df = metadata(root, split_column)
    audit, rows, waveforms, feats, hashes = [], [], [], [], {}
    missing_rows, missing_x, missing_f = [], [], []
    channels = {}
    for row in df.itertuples():
        info = {'record':row.record,'event':row.event,'split':row.split,'label':int(row.label)}
        path = root/'annotated_waveforms'/row.record/row.event
        try:
            hdr = wfdb.rdheader(str(path))
            names = [n.strip().upper() for n in hdr.sig_name]
            for name in names:
                channels[name] = channels.get(name, 0)+1
            ecg = next((names.index(n) for n in ECG_PRIORITY if n in names), None)
            ppg = next((i for i,n in enumerate(names) if n in ('PLETH','PPG')), None)
            if ecg is None:
                raise ValueError('no ECG lead in fixed priority list')
            if hdr.fs != 250:
                raise ValueError(f'expected 250Hz, received {hdr.fs}')
            end = int(300*hdr.fs); start = end-int(seconds*hdr.fs)
            if start < 0 or hdr.sig_len < end:
                raise ValueError('insufficient pre-alarm duration')
            # Read only samples [300-window, 300); post-alarm samples never enter processing.
            ecg_raw = wfdb.rdrecord(str(path), sampfrom=start, sampto=end, channels=[ecg]).p_signal[:,0]
            ex, ef, eq = prepare(ecg_raw, hdr.fs, 'ecg')
            info.update(ecg_lead=names[ecg], ecg_quality=eq)
            if ppg is None:
                missing_rows.append(info.copy()); missing_x.append(ex); missing_f.append(ef)
                raise ValueError('PPG missing; saved for separate ECG-only analysis')
            raw = wfdb.rdrecord(str(path), sampfrom=start, sampto=end, channels=[ecg,ppg]).p_signal
            px, pf, pq = prepare(raw[:,1], hdr.fs, 'ppg')
            digest = hashlib.sha256(np.nan_to_num(raw).tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f'duplicate waveform of {hashes[digest]}')
            hashes[digest] = row.event
            info.update(ppg_quality=pq, sha256=digest)
            rows.append(info.copy()); waveforms.append(np.stack([ex,px])); feats.append(ef)
            info['status'] = 'included'
        except (ValueError, OSError, FileNotFoundError) as exc:
            info['status'] = 'excluded'; info['reason'] = str(exc)
        audit.append(info)
    pd.DataFrame(audit).to_csv(output/'audit.csv', index=False)
    (output/'channel_counts.json').write_text(json.dumps(channels, indent=2))
    if not rows:
        raise ValueError('No paired events. Inspect audit.csv and download the waveforms.')
    paired = pd.DataFrame(rows)
    for split in ('train','validation','test'):
        if set(paired.loc[paired.split==split, 'label']) != {0,1}:
            raise ValueError(f'{split} paired cohort needs both classes; see audit.csv')
    np.savez_compressed(output/'paired.npz', x=np.stack(waveforms), features=np.stack(feats))
    paired.to_csv(output/'paired.csv', index=False)
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(output/'ppg_missing.csv', index=False)
        np.savez_compressed(output/'ppg_missing.npz', x=np.stack(missing_x), features=np.stack(missing_f))
    provenance = {'synthetic':False,'version':'1.1','seconds':seconds,'fs':250,
                  'ecg_priority':ECG_PRIORITY,'n_eligible':len(paired),
                  'official_events':len(df),'available_headers':sum((root/'annotated_waveforms'/r.record/(r.event+'.hea')).exists() for r in df.itertuples()),
                  'metadata_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'event_labels.csv',root/'benchmark_data_split.csv')}}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(paired.groupby('split').agg(events=('event','size'), true=('label','sum')))

def synthetic(output, seed=42):
    """Toy data for software verification. Has no clinical interpretation."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed); rows=[]; xs=[]; fs=[]
    hz=250; t=np.arange(2500)/hz
    for split,n in [('train',80),('validation',24),('test',24)]:
        for g in range(n//2):
            for event in range(2):
                y=int(rng.random()<.35)
                rate=rng.uniform(2.0,3.5) if y else rng.uniform(.9,1.8)
                phase=2*np.pi*rate*t
                ecg=np.sin(phase)**15 + .18*rng.normal(size=len(t))
                if rng.random()<.35:
                    ecg += .9*rng.normal(size=len(t))
                ppg=np.sin(phase-.6)+.2*np.sin(2*phase)+.12*rng.normal(size=len(t))
                if rng.random()<.25:
                    ppg=rng.normal(size=len(t))
                ex,ef,eq=prepare(ecg,hz,'ecg'); px,_,pq=prepare(ppg,hz,'ppg')
                rows.append(dict(record=f'{split}_{g:03}',event=f'{split}_{g:03}_{event}',split=split,label=y,ecg_quality=eq,ppg_quality=pq))
                xs.append(np.stack([ex,px])); fs.append(ef)
    pd.DataFrame(rows).to_csv(output/'paired.csv',index=False)
    np.savez_compressed(output/'paired.npz',x=np.stack(xs),features=np.stack(fs))
    (output/'provenance.json').write_text(json.dumps({'synthetic':True,'seconds':10,'fs':hz,'seed':seed},indent=2))
