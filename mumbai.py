# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from scipy.signal import savgol_filter

@st.cache(allow_output_mutation=True)
def load_data():
    url = 'https://api.covid19india.org/csv/latest/districts.csv'
    data = pd.read_csv(url)
    cdata = data[data['District']=='Mumbai']
    cdata['Daily Cases']=cdata['Confirmed'].diff()
    cdata = cdata[1:]
    cdata.reset_index(inplace=True)
    n = np.arange(0,len(cdata['Daily Cases']),1)
    
    for i in n:
        if cdata['Daily Cases'].iloc[i]<=0:
            cdata['Daily Cases'][i]=(cdata['Daily Cases'].iloc[i-1]+cdata['Daily Cases'].iloc[i+1])/2
    
    raw = cdata['Daily Cases'].values.tolist()
    smooth = savgol_filter(raw,15,1)
    cdata['Smoothened']=smooth
    return cdata

cdata = load_data()

@st.cache()
def create_mumfigs():
    fig = px.line(cdata,x='Date',y=['Daily Cases','Smoothened'])
    # fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_layout(title_text = 'Covid-19 - Mumbai - Daily Cases',
                       width = 700, height = 480,
                       showlegend=False,
                        annotations=[
                            dict(
                                x=0.96,
                                y=0.03,
                                xref="paper",
                                yref="paper",
                                text="Kaustuv",
                                ax=0,
                                ay=0
                                )
                            ]
                       )
    max_y = cdata['Daily Cases'].max()+100
    min_x = cdata['Date'][0]
    max_x = '2021-07-31'
    fig.update_yaxes(range=(0,max_y))
    fig.update_xaxes(range=(min_x,max_x))
    fig.update_layout(shapes=[
        dict(
          type= 'line',
          yref= 'paper', y0= 0, y1= 1,
          xref= 'x', x0= '2020-08-22', x1= '2020-08-22'
        ),
        dict(
          type= 'line',
          yref= 'paper', y0= 0, y1= 1,
          xref= 'x', x0= '2020-11-14', x1= '2020-11-14'
        )
    ])
    
    fig.add_annotation(x='2020-08-22', y=2300,
                text="Ganesh Chaturthi",
                showarrow=True,
                arrowhead=1,
                ax=-60,ay=-20)
    fig.add_annotation(x='2020-11-14', y=2000,
                text="Diwali",
                showarrow=True,
                arrowhead=1,
                ax=40,ay=-20)

    return fig