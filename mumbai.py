# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

#@st.cache
def load_data():
    url = 'https://api.covid19india.org/csv/latest/districts.csv'
    data = pd.read_csv(url)
    return data

data = load_data()


cdata = data[data['District']=='Mumbai']
cdata['Daily Cases']=cdata['Confirmed'].diff()
cdata = cdata[1:]
if cdata['Daily Cases'].iloc[-1]==0:
    cdata = cdata[:-1]
cdata.reset_index(inplace=True)
n = np.arange(0,len(cdata['Daily Cases'])-1,1)

for i in n:
    if cdata['Daily Cases'].iloc[i]<=0 & i<len(cdata['Daily Cases'])-1:
        cdata['Daily Cases'][i]=(cdata['Daily Cases'].iloc[i-1]+cdata['Daily Cases'].iloc[i+1])/2

raw = cdata['Daily Cases'].values.tolist()
smooth = savgol_filter(raw,15,1)
cdata['Smoothened']=smooth
    



def create_mumfigs():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdata['Date'], y=cdata['Daily Cases'], mode='lines', name='Daily Cases',line={'dash': 'dot', 'color': 'cadetblue'}))
    fig.add_trace(go.Scatter(x=cdata['Date'], y=cdata['Smoothened'], mode='lines', name='Smoothened', line={'dash': 'solid', 'color': 'red'}))
    
    fig.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "bottom"},
                      title_text = 'Covid-19 - Mumbai - Daily Cases',
                       width = 740, height = 480,
                       margin=dict(r=20, b=10, l=10, t=30),
                       template = 'plotly_white',
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
    fig.update_yaxes(title_text="No of Cases", range=(0,max_y))
    fig.update_xaxes(title_text="Date", range=(min_x,max_x))
    
#    fig.update_layout(shapes=[
#        dict(
#          type = 'line',
#          yref = 'paper', y0= 0, y1= 1,
#          xref = 'x', x0= '2020-08-22', x1= '2020-08-22',
#          line = dict(color='black', dash='dashdot')
#        ),
#        dict(
#          type= 'line',
#          yref= 'paper', y0= 0, y1= 1,
#          xref= 'x', x0= '2020-11-14', x1= '2020-11-14'
#        )
#    ])
    fig.add_shape(type="line",
        x0= '2020-08-22', x1= '2020-08-22',
        y0= 0, y1= 1,
        xref='x',yref='paper',
        line=dict(
            color="black",
            width=1,
            dash="dashdot",
        )
    )
    fig.add_shape(type="line",
        x0= '2020-11-14', x1= '2020-11-14',
        y0= 0, y1= 1,
        xref='x',yref='paper',
        line=dict(
            color="black",
            width=1,
            dash="dashdot",
        )
    )
        
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
