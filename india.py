#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:04:44 2020
@author: kaustuv
"""

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

# Read data from eCDC website
@st.cache(allow_output_mutation=True)
def load_data():
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    data = pd.read_csv(url)
    data['dateRep'] = pd.to_datetime(data['dateRep'], format = '%d/%m/%Y')
    data = data.reindex(index=data.index[::-1])
    data = data.dropna()
    cdata = data[data['geoId']=='IN']
    cdata['cumcases']=cdata['cases'].cumsum()
    cdata['cumdeaths']=cdata['deaths'].cumsum()
    cdata['dpm'] = round(cdata['cumdeaths']*1e6/cdata['popData2019'],2)
    cdata['dpc'] = round(cdata['cumcases']*1e6/cdata['popData2019'],2)
    return cdata




def create_indfigs():
    data = load_data()
    ifig0 = make_subplots(rows=2, cols=2,
                         subplot_titles=("Daily Cases", "Daily Deaths",
                                         "Total Cases","Total Deaths"))
    
    ifig0.add_trace(go.Scatter(x=data['dateRep'], y=data['cases'],name='Daily Cases',line_color='blue'),row=1, col=1)
    ifig0.add_trace(go.Scatter(x=data['dateRep'], y=data['cumcases'], name='Total Cases',line_color='blue'),row=2, col=1)
    ifig0.add_trace(go.Scatter(x=data['dateRep'], y=data['deaths'],name='Daily Deaths',line_color='blue'),row=1, col=2)
    ifig0.add_trace(go.Scatter(x=data['dateRep'], y=data['cumdeaths'], name='Total Deaths',line_color='blue'),row=2, col=2)
    ifig0.update_layout( xaxis_title='Date',
                        yaxis_title='Deaths',
                        width = 700, height=480,
                        showlegend=False,
                        template = 'seaborn'
                        )
    
    ifig1 = make_subplots(rows=2, cols=1)
    
    ifig1.add_trace(go.Scatter(x=data['dateRep'], y=data['dpc'],name='Cases per million',line_color='blue'),row=1, col=1)
    ifig1.add_trace(go.Scatter(x=data['dateRep'], y=data['dpm'], name='Deaths per million',line_color='blue'),row=2, col=1)
    ifig1.update_layout(width = 800, height=480,
                        showlegend=False,
                        )
    ifig1.update_yaxes(type="log")
    ifig1.update_layout(title_text = 'Covid-19 - India - Morbidity & Mortality',
                        template = 'seaborn')

    
    annotations = []
    annotations.extend([
        dict(
            x=data['dateRep'].iloc[-1], y=np.log10(data['dpc'].iloc[-1]), # annotation point
            xref='x1', 
            yref='y1',
            text=data['dpc'].iloc[-1],
            showarrow=False,
            font = dict(size=12),
            xanchor='left', yanchor='middle'
        ),
        dict(
            x=data['dateRep'].iloc[-1], y=np.log10(data['dpm'].iloc[-1]), # annotation point
            xref='x2', 
            yref='y2',
            text=data['dpm'].iloc[-1],
            showarrow=False,
            font = dict(size=12),
            xanchor='left', yanchor='middle'
        ),
         dict(
            x=0.5,y=1, # annotation point
            xref='paper', 
            yref='paper',
            text='Cases per Million Population',
            showarrow=False,
            font = dict(size=14),
            xanchor='center', yanchor='bottom'
        ),
                 dict(
            x=0.5,y=0.42, # annotation point
            xref='paper', 
            yref='paper',
            text='Deaths per Million Population',
            showarrow=False,
            font = dict(size=14),
            xanchor='center', yanchor='bottom'
        ),             
                ])
        
    ifig1.layout.annotations=annotations

    

    #Calculate Growth Rate
    t0 = 80
    dates = pd.to_datetime(data['dateRep']).dt.date.unique().tolist()
    dates = dates[t0:]
    end = len(dates)
    start = end - 14
    dp = pd.date_range(dates[start],periods=114).tolist()
    
    g1 = np.array(data['cumcases'][t0:].values.tolist())
    g2 = np.array(data['cumcases'][t0-1:-1].values.tolist())
    gr = 100*(g1-g2)/g1
    grC = savgol_filter(gr,7,1)
    
    
    
    t = np.array(range(start,end))
    y = grC[start:end]
    a,b = np.polyfit(t,y,1)
    
    t = np.array(range(start,end+100))
    ypC = a*t+b
    
    
    g1 = np.array(data['cumdeaths'][t0:].values.tolist())
    g2 = np.array(data['cumdeaths'][t0-1:-1].values.tolist())
    gr = 100*(g1-g2)/g1
    grD = savgol_filter(gr,7,1)
    
    end = len(dates)
    start = end - 14
    
    t = np.array(range(start,end))
    y = grD[start:end]
    a,b = np.polyfit(t,y,1)
    
    t = np.array(range(start,end+100))
    ypD = a*t+b
    
    ifig2 = go.Figure()
    
    ifig2.add_trace(go.Scatter(x=dates,y=grC, mode="lines", name="Cases",line={'dash': 'solid', 'color': 'blue'}))
    ifig2.add_trace(go.Scatter(x=dp,y=ypC, mode="lines", name="Cases Predicted",line={'dash': 'dot', 'color': 'blue'}))
    ifig2.add_trace(go.Scatter(x=dates,y=grD, mode="lines", name="Deaths",line={'dash': 'solid', 'color': 'red'}))
    ifig2.add_trace(go.Scatter(x=dp,y=ypD, mode="lines", name="Deaths Predicted",line={'dash': 'dot', 'color': 'red'}))
    
    ifig2.update_yaxes(range=[0, 10])
    ifig2.update_layout(title_text = 'Covid-19 - India - Growth Rates',
                    xaxis_title='Date',
                    yaxis_title='Growth rate (%)',
                    width = 800, height=480,
                    showlegend = False,
                    template = 'seaborn'
                    )
    txt1 = "Growth Rate (Cases): {a: .2f}%"
    txt2 = "Growth Rate (Deaths): {a: .2f}%"
    ifig2.add_annotation(x=0.9, y=0.9,
                          text = txt1.format(a=grC[-1]),
                          xref='paper',yref='paper',
                          showarrow=False)
    ifig2.add_annotation(x=0.9, y=0.8,
                          text = txt2.format(a=grD[-1]),
                          xref='paper',yref='paper',
                          showarrow=False)    
    return ifig0, ifig1, ifig2
