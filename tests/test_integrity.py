import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from vt_alarm.evaluation import select_threshold,metrics,shuffled_indices,bootstrap
from vt_alarm.signals import prepare
from vt_alarm.data import metadata

class IntegrityTests(unittest.TestCase):
    def test_threshold_ties_and_target(self):
        y=np.array([1,1,0,0]); p=np.array([.2,.9,.1,.8])
        t=select_threshold(y,p,.95)
        self.assertEqual(t,.2)
        self.assertEqual(metrics(y,p,t)['sensitivity'],1)
    def test_no_rejection_is_reported(self):
        y=np.array([1,1,0,0]); p=np.array([.1,.2,.8,.9])
        t=select_threshold(y,p,.95)
        self.assertEqual(t,0.)
        self.assertEqual(metrics(y,p,t)['specificity'],0)
    def test_shuffle(self):
        g=np.array(['a','a','a','b','b','c','c','d'])
        for seed in range(20):
            idx=shuffled_indices(g,seed)
            self.assertTrue(np.all(g[idx]!=g))
            self.assertEqual(len(np.unique(idx)),len(g))
    def test_shuffle_impossible(self):
        with self.assertRaises(ValueError): shuffled_indices(['a','a','a','b'],1)
    def test_paired_cluster_bootstrap(self):
        df=pd.DataFrame({'record':['a','a','b','b','c','c'],'label':[1,0,1,0,1,0]})
        p=np.array([.9,.2,.8,.3,.7,.1])
        r=bootstrap(df,{'ecg':p,'fusion':p},{'ecg':.5,'fusion':.5},20)
        self.assertEqual(r['fusion_minus_ecg']['specificity']['low'],0.)
        self.assertEqual(r['fusion_minus_ecg']['sensitivity']['high'],0.)
    def test_preprocess_gaps(self):
        x=np.sin(np.arange(2500)/20); x[10:12]=np.nan
        self.assertTrue(np.isfinite(prepare(x,250,'ecg')[0]).all())
        x[10:100]=np.nan
        with self.assertRaises(ValueError): prepare(x,250,'ecg')
    def test_reject_split_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame({'record':['a','a'],'event':['a1','a2'],'decision':['True','False']}).to_csv(Path(d)/'event_labels.csv',index=False)
            pd.DataFrame({'record':['a','a'],'event':['a1','a2'],'split':['train','test']}).to_csv(Path(d)/'benchmark_data_split.csv',index=False)
            with self.assertRaisesRegex(ValueError,'leakage'): metadata(d)

if __name__=='__main__': unittest.main()
