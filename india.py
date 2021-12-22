#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 11:53:34 2020

@author: kaustuv
"""
# Import Libraries
# import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import streamlit as st

# Read data from JHU website
# @st.cache
def load_data():
    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    # data = pd.read_csv(url)
    # # data.head()
    #  # Filter data for India
    # india_cdata = data.loc[data['Country/Region']=='India']
    # india_cdata.drop(['Province/State','Country/Region','Lat','Long'],
    #   axis='columns', inplace=True)
    # total=india_cdata.values.tolist()[0]
    # if total[-1]-total[-2]<=0:
    #     total = total[:-1]
    
    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    # data = pd.read_csv(url)
    # # Filter data for India
    # india_cdata = data.loc[data['Country/Region']=='India']
    # india_cdata.drop(['Province/State','Country/Region','Lat','Long'],
    #   axis='columns', inplace=True)
    # deaths=india_cdata.values.tolist()[0]
    # if deaths[-1]-deaths[-2]<=0:
    #     deaths = deaths[:-1]

    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    # data = pd.read_csv(url)
    # # Filter data for India
    # india_cdata = data.loc[data['Country/Region']=='India']
    # india_cdata.drop(['Province/State','Country/Region','Lat','Long'],
    #   axis='columns', inplace=True)
    # recovered=india_cdata.values.tolist()[0]
    # if recovered[-1]-recovered[-2]<=0:
    #     recovered = recovered[:-1]
    
    # url = 'https://api.covid19india.org/csv/latest/case_time_series.csv'
    # data = pd.read_csv(url)
    # total = data['Total Confirmed'].tolist()
    # deaths = data['Total Deceased'].tolist()
    # recovered = data['Total Recovered'].tolist()    
    
    url = 'https://data.covid19india.org/csv/latest/states.csv'
    data = pd.read_csv(url)
    data = data[data["State"]=="India"]
    data["Date"] = pd.to_datetime(data['Date'])
    
    r = pd.date_range(start=data.Date.min(), end=data.Date.max())
    data = data.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()
    
    
    for row in range(len(data)):
        if data['State'][row] == 0.0:
            data['Confirmed'].iloc[row] = data['Confirmed'].iloc[row-1]
            data['Recovered'].iloc[row] = data['Recovered'].iloc[row-1]
            data['Deceased'].iloc[row] = data['Deceased'].iloc[row-1]
            data['State'].iloc[row] = 'India'
    
    total = data['Confirmed'].tolist()
    deaths = data['Deceased'].tolist()
    recovered = data['Recovered'].tolist()
    
#     url = 'https://raw.githubusercontent.com/datameet/covid19/master/data/all_totals.json'
#     data = pd.read_json(url)
#     df = pd.json_normalize(data.rows)
    
#     date = []
#     active_cases = []
#     cured = []
#     deaths = []
#     total = []
    
    
#     for i in range(len(df)):
#         if df.key[i][1] == 'active_cases':
#             date.append(df.key[i][0])
#             active_cases.append(df['value'].iloc[i])
        
#         if df.key[i][1] == 'cured':
#             cured.append(df['value'].iloc[i])
    
#         if df.key[i][1] == 'death':
#             deaths.append(df['value'].iloc[i]) 
            
#         if df.key[i][1] == 'total_confirmed_cases':
#             total.append(df['value'].iloc[i])
        
#     data = pd.DataFrame({'date': date, 'active_cases': active_cases, 'cured': cured, 'deaths': deaths, 'total': total})
#     data['date'] = pd.to_datetime(data['date'])
    
#     # Cleanup
    
#     for i in range(len(data)-2):
#         if data['cured'].iloc[i+1]<data['cured'].iloc[i]:
#             data['cured'].iloc[i+1] = (data['cured'].iloc[i]+data['cured'].iloc[i+2])/2
#         if data['deaths'].iloc[i+1]<data['deaths'].iloc[i]:
#             data['deaths'].iloc[i+1] = (data['deaths'].iloc[i]+data['deaths'].iloc[i+2])/2
#         if data['total'].iloc[i+1]<data['total'].iloc[i]:
#             data['total'].iloc[i+1] = (data['total'].iloc[i]+data['total'].iloc[i+2])/2

#     # Missing Dates
#     data = data.groupby(pd.Grouper(key='date',freq='D')).max().rename_axis('date').reset_index()
    
#     for i in range(len(data)):
#         if np.isnan(data['active_cases'].iloc[i]):
#             data['active_cases'].iloc[i] = data['active_cases'][i-1]
#             data['cured'].iloc[i] = data['cured'][i-1]
#             data['deaths'].iloc[i] = data['deaths'][i-1]
#             data['total'].iloc[i] = data['total'][i-1]        
       
#     total = data['total'].tolist()
#     deaths = data['deaths'].tolist()
#     recovered = data['cured'].tolist()
    


#    popData2019 = 1366417756
    popData2019 = 1380004385
    cdata = pd.DataFrame([total,deaths,recovered]).transpose()
    cdata["date_id"] = cdata.index
    cdata.columns=["cumcases", "cumdeaths","cumrecovered","date_id"]
    #cdata.rename(columns={0: "cumcases",1: "cumdeaths"},inplace=True)
    startdate = pd.Timestamp('2020-01-30')
    cdata['time_added'] = pd.to_timedelta(cdata['date_id'],'d')
    cdata['Date'] = startdate+cdata['time_added']
    cdata.drop(['time_added'],axis='columns', inplace=True)
    
    dailycases = cdata.cumcases.diff()
    dailydeaths = cdata.cumdeaths.diff()
    dailyrecovered = cdata.cumrecovered.diff()
    cdata["dailycases"] = dailycases
    cdata["dailydeaths"] = dailydeaths
    cdata["dailyrecovered"] = dailyrecovered
    
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
                               "x": 0.5,"y": 0.9,"xanchor": "center","yanchor": "bottom"},
                        width = 740, height=480,
                        margin=dict(r=20, b=10, l=10, t=100),
                        showlegend=False,
                        template='seaborn',
                        xaxis_title='Date',
                        yaxis_title='Deaths'
                        )
    
    ifig1 = make_subplots(rows=2, cols=1)
    
    ifig1.add_trace(go.Scatter(x=data['Date'], y=data['dpc'],name='Cases per million',line_color='blue'),row=1, col=1)
    ifig1.add_trace(go.Scatter(x=data['Date'], y=data['dpm'], name='Deaths per million',line_color='blue'),row=2, col=1)
    ifig1.update_layout(title={"text": "Covid-19 - India - Morbidity & Mortality",
                               "x": 0.5,"y": 0.9,"xanchor": "center","yanchor": "bottom"},
                        width = 740, height=480,
                        margin=dict(r=20, b=10, l=10, t=100),
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
    
    def func(x, a, b, c): ###
        return a * np.exp(-b * x) + c
    
    t = np.array(range(0,14))
    y = grC[start:end]
    # a,b = np.polyfit(t,y,1)
    z = np.polyfit(t,y,2)
    # popt, pcov = curve_fit(func, t, y, maxfev = 2000)
    
    
    t = np.array(range(0,14+100))
    # ypC = a*t+b
    ypC = z[0]*t*t+z[1]*t+z[2]
    if any(ypC<=0):
        ix = np.where(ypC <= 0)[0][0]
        t=t[:ix]
        ypC=ypC[:ix]
    # ypC = func(t, *popt)
    
    
    
    g1 = np.array(data['cumdeaths'][t0:].values.tolist())
    g2 = np.array(data['cumdeaths'][t0-1:-1].values.tolist())
    gr = 100*(g1-g2)/g1
#    gr = gr/7
    grD = savgol_filter(gr,15,1)
    
    end = len(dates)
    start = end - 14
    
    t = np.array(range(0,14))
    y = grD[start:end]
    z = np.polyfit(t,y,2)
    # popt, pcov = curve_fit(func, t, y, maxfev = 2000)
    
    t = np.array(range(0,14+100))
    # ypD = a*t+b
    ypD = z[0]*t*t+z[1]*t+z[2]
    
    if any(ypD<0):
        ix = np.where(ypD <= 0)[0][0]
        print(ix)
        t=t[:ix]
        ypD=ypD[:ix]

    # ypD = func(t, *popt)
    
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
                    margin=dict(r=20, b=4, l=10, t=80),
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
    
    # Calculate CFR
    cfr = data['dailydeaths'][148:]*100/(data['dailydeaths'][148:]+data['dailyrecovered'][148:])
    cfrT = savgol_filter(cfr,7,1)
    
    ifig3 = go.Figure()
    ifig3.add_trace(go.Scatter(x=data['Date'][148:],y=cfr, mode="lines", name="CFR",line={'dash': 'dot', 'color': 'teal'}))
    ifig3.add_trace(go.Scatter(x=data['Date'][148:],y=cfrT, mode="lines", name="CFR Trend",line={'dash': 'solid', 'color': 'red'}))
    
    ifig3.update_yaxes(range=[0, 3])
    ifig3.update_layout(title_text = 'Covid-19 - India - Case Fatality Ratio',
                    xaxis_title='Date',
                    yaxis_title='Case Fatality Ratio (%)',
                    width = 740, height=480,
                    margin=dict(r=20, b=10, l=10, t=100),
                    showlegend = False,
                    template = 'seaborn'
                    )
    
    return ifig0, ifig1, ifig2, ifig3
