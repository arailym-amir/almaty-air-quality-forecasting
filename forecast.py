"""One-hour-ahead PM2.5 forecasting from information available at issue time."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FEATURES=['pm25_now','lag_1','lag_24','mean_24','hour_sin','hour_cos','year_sin','year_cos']

def load_series(path, time='timestamp', target='pm25', station=None):
    raw=pd.read_csv(path)
    if station:
        col,value=station.split('=',1)
        raw=raw[raw[col].astype(str)==value]
    t=pd.to_datetime(raw[time],errors='coerce',utc=True)
    y=pd.to_numeric(raw[target],errors='coerce').where(lambda x:x>=0)
    s=pd.Series(y.to_numpy(),index=t,name='pm25').dropna()
    s=s[~s.index.isna()].sort_index().resample('h').median()
    if len(s)<100: raise ValueError('Need at least 100 hourly bins; preferably a year of data.')
    return s

def features(s):
    f=pd.DataFrame(index=s.index)
    f['pm25_now']=s
    f['lag_1']=s.shift(1); f['lag_24']=s.shift(24)
    f['mean_24']=s.rolling(24,min_periods=18).mean()
    # UTC timestamp is the issue time; cycles represent local Almaty time.
    local=s.index.tz_convert('Asia/Almaty')
    f['hour_sin']=np.sin(2*np.pi*local.hour/24); f['hour_cos']=np.cos(2*np.pi*local.hour/24)
    f['year_sin']=np.sin(2*np.pi*local.dayofyear/365.25); f['year_cos']=np.cos(2*np.pi*local.dayofyear/365.25)
    f['target']=s.shift(-1)
    return f

def model(): return RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=42,n_jobs=-1)

def metrics(y,p): return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p))}

def evaluate(s,out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    f=features(s).dropna(); cut=int(len(f)*.8)
    if cut<70 or len(f)-cut<10: raise ValueError('Insufficient complete hourly windows.')
    train=f.iloc[:cut-1]; test=f.iloc[cut:]; folds=[]
    for a,b in TimeSeriesSplit(n_splits=5,gap=1).split(train):
        tr,va=train.iloc[a],train.iloc[b]; m=model().fit(tr[FEATURES],tr.target)
        folds.append({'random_forest':metrics(va.target,m.predict(va[FEATURES])),'persistence':metrics(va.target,va.pm25_now)})
    m=model().fit(train[FEATURES],train.target)
    p=m.predict(test[FEATURES]); results={'hourly_bins':len(s),'observed_bins':int(s.notna().sum()),'complete_windows':len(f),'train_rows':len(train),'test_rows':len(test),'test_start':str(test.index[0]),'test_end':str(test.index[-1]),'random_forest':metrics(test.target,p),'persistence':metrics(test.target,test.pm25_now),'validation_folds':folds}
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    pred=pd.DataFrame({'actual':test.target,'random_forest':p,'persistence':test.pm25_now}); pred.index=pred.index+pd.Timedelta(hours=1); pred.index.name='target_hour_utc'; pred.to_csv(out/'predictions.csv')
    pd.Series(m.feature_importances_,index=FEATURES).sort_values().to_csv(out/'importance.csv',header=['importance'])
    fig,ax=plt.subplots(figsize=(11,4)); pred.tail(168).plot(ax=ax,color=['#172b4d','#008b83','#aaa']); ax.set(ylabel='PM2.5 (µg/m³)',title='Holdout predictions | last 168 available observations'); fig.tight_layout(); fig.savefig(out/'predictions.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); pd.Series(m.feature_importances_,index=FEATURES).sort_values().plot.barh(ax=ax,color='#008b83'); ax.set(title='Random forest feature importance (not causal)'); fig.tight_layout(); fig.savefig(out/'importance.png',dpi=160); plt.close(fig)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('--time',default='timestamp'); p.add_argument('--target',default='pm25'); p.add_argument('--station'); p.add_argument('--out',default='reports/run'); a=p.parse_args(); print(json.dumps(evaluate(load_series(a.csv,a.time,a.target,a.station),a.out),indent=2))
