#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:53:34 2020

@author: kaustuv
"""
# Import Libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import savgol_filter

# Read data from JHU website
# @st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    data = pd.read_csv(url)
    data.head()
     # Filter data for India
    india_cdata = data.loc[data['Country/Region']=='India']
    india_cdata.drop(['Province/State','Country/Region','Lat','Long'],
      axis='columns', inplace=True)
    total=india_cdata.values.tolist()[0]
    
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    data = pd.read_csv(url)
    # Filter data for India
    india_cdata = data.loc[data['Country/Region']=='India']
    india_cdata.drop(['Province/State','Country/Region','Lat','Long'],
      axis='columns', inplace=True)
    deaths=india_cdata.values.tolist()[0]
    
#    popData2019 = 1366417756
    popData2019 = 1380004385
    cdata = pd.DataFrame([total,deaths]).transpose()
    cdata["date_id"] = cdata.index
    cdata.columns=["cumcases", "cumdeaths","date_id"]
    #cdata.rename(columns={0: "cumcases",1: "cumdeaths"},inplace=True)
    startdate = pd.Timestamp('2020-01-22')
    cdata['time_added'] = pd.to_timedelta(cdata['date_id'],'d')
    cdata['Date'] = startdate+cdata['time_added']
    cdata.drop(['time_added'],axis='columns', inplace=True)
    
    dailycases = cdata.cumcases.diff()
    dailydeaths = cdata.cumdeaths.diff()
    cdata["dailycases"] = dailycases
    cdata["dailydeaths"] = dailydeaths
    
    cdata['dpm'] = round(cdata['cumdeaths']*1e6/popData2019,2)
    cdata['dpc'] = round(cdata['cumcases']*1e6/popData2019,2)

    return cdata

def create_indfigs():
    data = load_data()
    ifig0 = make_subplots(rows=2, cols=2,
                         subplot_titles=("Daily Cases", "Daily Deaths",
                                         "Total Cases","Total Deaths"))
    
    ifig0.add_trace(go.Scatter(x=data['Date'], y=data['dailycases'],name='Daily Cases',line_color='blue'),row=1, col=1)
    ifig0.add_trace(go.Scatter(x=data['Date'], y=data['cumcases'], name='Total Cases',line_color='blue'),row=2, col=1)
    ifig0.add_trace(go.Scatter(x=data['Date'], y=data['dailydeaths'],name='Daily Deaths',line_color='blue'),row=1, col=2)
    ifig0.add_trace(go.Scatter(x=data['Date'], y=data['cumdeaths'], name='Total Deaths',line_color='blue'),row=2, col=2)

    ifig0.update_layout(title={"text": "Covid-19 - India - Cases & Deaths",
                               "x": 0.5,"y": 0.95,"xanchor": "center","yanchor": "bottom"},
                        width = 740, height=480,
                        margin=dict(r=20, b=10, l=10, t=60),
                        showlegend=False,
                        template='seaborn',
                        xaxis_title='Date',
                        yaxis_title='Deaths'
                        )
    
    ifig1 = make_subplots(rows=2, cols=1)
    
    ifig1.add_trace(go.Scatter(x=data['Date'], y=data['dpc'],name='Cases per million',line_color='blue'),row=1, col=1)
    ifig1.add_trace(go.Scatter(x=data['Date'], y=data['dpm'], name='Deaths per million',line_color='blue'),row=2, col=1)
    ifig1.update_layout(title={"text": "Covid-19 - India - Morbidity & Mortality",
                               "x": 0.5,"y": 0.95,"xanchor": "center","yanchor": "bottom"},
                        width = 740, height=480,
                        margin=dict(r=20, b=10, l=10, t=60),
                        showlegend=False,
                        template='seaborn'
                        )
    ifig1.update_yaxes(type="log")

    
    annotations = []
    annotations.extend([
        dict(
            x=data['Date'].iloc[-1], y=np.log10(data['dpc'].iloc[-1]), # annotation point
            xref='x1', 
            yref='y1',
            text=data['dpc'].iloc[-1],
            showarrow=False,
            font = dict(size=12),
            xanchor='left', yanchor='middle'
        ),
        dict(
            x=data['Date'].iloc[-1], y=np.log10(data['dpm'].iloc[-1]), # annotation point
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
    dates = pd.to_datetime(data['Date']).dt.date.unique().tolist()
    dates = dates[t0:]
    end = len(dates)
    start = end - 14
    dp = pd.date_range(dates[start],periods=114, freq="D").tolist()
    
    g1 = np.array(data['cumcases'][t0:].values.tolist())
    g2 = np.array(data['cumcases'][t0-1:-1].values.tolist())
    gr = 100*(g1-g2)/g1
#    gr = gr/7
    grC = savgol_filter(gr,15,1)
    
    
    
    t = np.array(range(start,end))
    y = grC[start:end]
    a,b = np.polyfit(t,y,1)
    
    t = np.array(range(start,end+100))
    ypC = a*t+b
    
    
    g1 = np.array(data['cumdeaths'][t0:].values.tolist())
    g2 = np.array(data['cumdeaths'][t0-1:-1].values.tolist())
    gr = 100*(g1-g2)/g1
#    gr = gr/7
    grD = savgol_filter(gr,15,1)
    
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
                    width = 740, height=480,
                    margin=dict(r=20, b=10, l=10, t=30),
                    showlegend = False,
                    template = 'seaborn'
                    )
    txt1 = "Current Growth Rate (Cases): {a: .2f}%"
    txt2 = "Current Growth Rate (Deaths): {a: .2f}%"
    ifig2.add_annotation(x=0.9, y=0.9,
                          text = txt1.format(a=grC[-1]),
                          xref='paper',yref='paper',
                          showarrow=False)
    ifig2.add_annotation(x=0.9, y=0.86,
                          text = txt2.format(a=grD[-1]),
                          xref='paper',yref='paper',
                          showarrow=False)    
    return ifig0, ifig1, ifig2
