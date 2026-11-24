"""Synthetic data for testing software only; not measurements from Almaty."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate(path='data/synthetic.csv'):
    rng=np.random.default_rng(42); n=24*90; t=pd.date_range('2025-01-01',periods=n,freq='h',tz='UTC'); x=np.arange(n); noise=np.zeros(n)
    for i in range(1,n): noise[i]=.85*noise[i-1]+rng.normal(0,3)
    y=np.maximum(0,30+12*np.sin(x*2*np.pi/24)+8*np.cos(x*2*np.pi/(24*14))+noise)
    Path(path).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame({'timestamp':t,'pm25':y}).to_csv(path,index=False)
if __name__=='__main__': generate()
