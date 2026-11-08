import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import pandas as pd
from vt_alarm.data import build_cache

class TemporalTests(unittest.TestCase):
    def test_loader_reads_only_pre_alarm_and_same_paired_events(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rows=[]; splits=[]
            for split in ('train','validation','test'):
                for y in (0,1):
                    rows.append({'record':split+str(y),'event':split+str(y)+'_1','decision':str(bool(y))})
                    splits.append({'record':split+str(y),'event':split+str(y)+'_1','split':split})
            pd.DataFrame(rows).to_csv(root/'event_labels.csv',index=False)
            pd.DataFrame(splits).to_csv(root/'benchmark_data_split.csv',index=False)
            calls=[]
            def reader(path,sampfrom,sampto,channels):
                calls.append((sampfrom,sampto))
                n=sampto-sampfrom; t=np.arange(n)/250
                raw=np.stack([np.sin(2*np.pi*2*t),np.sin(2*np.pi*2*t-.5)],axis=1)
                return SimpleNamespace(p_signal=raw[:,channels]+sum(map(ord,path))*.001)
            header=SimpleNamespace(sig_name=['II','PLETH'],fs=250,sig_len=90000)
            with patch('wfdb.rdheader',return_value=header),patch('wfdb.rdrecord',side_effect=reader):
                build_cache(root,root/'cache',30)
            self.assertTrue(all(a==67500 and b==75000 for a,b in calls))
            x=np.load(root/'cache/paired.npz')['x']
            self.assertEqual(x.shape,(6,2,7500))
            cohort=pd.read_csv(root/'cache/paired.csv')
            self.assertEqual(set(cohort.event),set(r['event'] for r in rows))

if __name__=='__main__': unittest.main()
