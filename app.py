import streamlit as st
import pandas as pd
import json
from pathlib import Path
import tempfile
from forecast import load_series,evaluate
st.set_page_config(page_title='Almaty | PM2.5 forecasting',layout='wide')
st.title('Almaty | Air quality forecasting')
st.caption('Research prototype · one-hour-ahead prediction · not a live public health advisory')
u=st.file_uploader('Upload an hourly or sub-hourly CSV',type='csv')
if u:
    cols=pd.read_csv(u,nrows=0).columns.tolist(); u.seek(0)
    t=st.selectbox('UTC timestamp column',cols); y=st.selectbox('PM2.5 concentration column (µg/m³)',cols,index=min(1,len(cols)-1)); station=st.text_input('Optional station filter (column=value)')
    if st.button('Train and evaluate'):
        try:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'input.csv'; p.write_bytes(u.getvalue()); r=evaluate(load_series(p,t,y,station or None),d); st.session_state['results']=r; st.session_state['pred']=pd.read_csv(Path(d)/'predictions.csv'); st.session_state['importance']=pd.read_csv(Path(d)/'importance.csv',index_col=0)
        except Exception as e: st.error(str(e))
    if 'results' in st.session_state:
        r=st.session_state['results']; st.subheader('Chronological holdout'); a,b,c=st.columns(3); a.metric('Random forest MAE',f"{r['random_forest']['MAE']:.2f}"); b.metric('Persistence MAE',f"{r['persistence']['MAE']:.2f}"); c.metric('Test windows',r['test_rows']); st.line_chart(st.session_state['pred'].set_index('target_hour_utc').tail(168)); st.bar_chart(st.session_state['importance']); st.json(r)
else:
    st.info('Upload real observations, or generate the clearly labeled synthetic demo using python demo.py. No live data is supplied.')
