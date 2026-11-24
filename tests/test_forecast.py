import numpy as np
import pandas as pd
from forecast import features,load_series,evaluate
from demo import generate

def test_target_and_lags():
    s=pd.Series(np.arange(100,dtype=float),index=pd.date_range('2025-01-01',periods=100,freq='h',tz='UTC'))
    f=features(s); assert f.iloc[30].target==31; assert f.iloc[30].lag_24==6; assert f.iloc[30].mean_24==18.5
    changed=s.copy(); changed.iloc[31:]=999
    assert features(changed).iloc[30].drop('target').equals(f.iloc[30].drop('target'))

def test_pipeline(tmp_path):
    p=tmp_path/'synthetic.csv'; generate(p); r=evaluate(load_series(p),tmp_path/'out'); assert r['test_rows']>0; assert len(r['validation_folds'])==5; assert (tmp_path/'out'/'predictions.png').exists()
