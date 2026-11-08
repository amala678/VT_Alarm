"""Processing only the cropped pre-alarm interval; no post-alarm context."""
import numpy as np
from scipy import signal

ECG_PRIORITY = ('II', 'I', 'V', 'V1', 'III', 'AVF', 'AVL', 'AVR', 'V2', 'V3', 'V4', 'V5', 'V6')

def longest_run(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(max(ends-starts, default=0))

def prepare(raw, fs, modality):
    x = np.asarray(raw, dtype=float).copy()
    missing = ~np.isfinite(x)
    if missing.mean() > .05 or longest_run(missing) > round(.2*fs):
        raise ValueError('missing fraction >5% or gap >200ms')
    if missing.all() or len(x) < fs:
        raise ValueError('no usable interval')
    x[missing] = np.interp(np.flatnonzero(missing), np.flatnonzero(~missing), x[~missing])
    scale = np.percentile(x, 95)-np.percentile(x, 5)
    if scale < 1e-10:
        raise ValueError('constant signal')
    flat = np.abs(np.diff(x)) < max(1e-10, scale*1e-5)
    clipping = np.mean((x == x.min()) | (x == x.max()))
    quality = [missing.mean(), longest_run(flat)/fs, clipping]
    lo, hi = (.5, 40.) if modality == 'ecg' else (.3, 8.)
    sos = signal.butter(3, [lo, hi], fs=fs, btype='bandpass', output='sos')
    # Zero-phase filtering is allowed here: the full input ends strictly before alarm onset.
    filtered = signal.sosfiltfilt(sos, x)
    median = np.median(filtered)
    mad = np.median(np.abs(filtered-median))*1.4826
    normalized = np.clip((filtered-median)/max(mad, 1e-6), -12, 12)
    peaks, _ = signal.find_peaks(filtered, distance=max(1, int(.25*fs)), prominence=np.std(filtered)*.5)
    beats = [normalized[p-int(.15*fs):p+int(.25*fs)] for p in peaks
             if p >= int(.15*fs) and p+int(.25*fs) <= len(x)]
    if len(beats) >= 3:
        beats = np.stack(beats); template = np.median(beats, axis=0)
        corrs = [np.corrcoef(b, template)[0,1] if b.std()>1e-8 and template.std()>1e-8 else 0 for b in beats]
        consistency = float(np.nan_to_num(np.median(corrs)))
    else:
        consistency = 0.
    quality.append(consistency)
    # Proxy score, not a clinical quality label; physiology may affect this score.
    qscore = consistency - 2*quality[0] - min(quality[1]/2, 1) - quality[2]
    rr = np.diff(peaks)/fs
    freq, power = signal.welch(filtered, fs, nperseg=min(len(x), int(fs*4)))
    total = max(power.sum(), 1e-12)
    features = [60/np.mean(rr) if len(rr) else 0, np.std(rr) if len(rr) else 0,
                np.sqrt(np.mean(np.diff(rr)**2)) if len(rr)>1 else 0,
                np.ptp(x), np.std(x), np.std(np.diff(x)),
                power[(freq>=.5)&(freq<5)].sum()/total,
                power[(freq>=5)&(freq<15)].sum()/total, *quality]
    return normalized.astype('float32'), np.asarray(features, dtype='float32'), float(qscore)
