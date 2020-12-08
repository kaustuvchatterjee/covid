#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:44:11 2020

@author: kaustuv
"""
# Import Libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.integrate as spi
from scipy.signal import savgol_filter
from datetime import datetime, timedelta

#---------------------------------------------------------------------
# India - Prediction
#--------------------------------------------------------------------
def sir(y,t,N,beta, gamma):
    S,I,R = y
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - gamma*I
    dRdt = gamma*I
    return dSdt, dIdt, dRdt

# Read data from JHU website
st.cache
def load_inddata():
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    data = pd.read_csv(url)
    data.head()
     # Filter data for India
    india_df = data.loc[data['Country/Region']=='India']
    india_df.drop(['Province/State','Country/Region','Lat','Long'],
      axis='columns', inplace=True)
    total=india_df.values.tolist()[0]
    
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    data = pd.read_csv(url)
    # Filter data for India
    india_df = data.loc[data['Country/Region']=='India']
    india_df.drop(['Province/State','Country/Region','Lat','Long'],
      axis='columns', inplace=True)
    deaths=india_df.values.tolist()[0]
    
    url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    data = pd.read_csv(url)
    # Filter data for India
    india_df = data.loc[data['Country/Region']=='India']
    india_df.drop(['Province/State','Country/Region','Lat','Long'],
      axis='columns', inplace=True)
    recovered=india_df.values.tolist()[0]
    
    return total,deaths,recovered


total,deaths,recovered = load_inddata()


def india_pred():
    total,deaths,recovered = load_inddata()
    RR=[x+y for x, y in zip(deaths, recovered)]
    II=[x-y for x, y in zip(total, RR)]
    
    window = 7
    start = 43
    end = start+window
    
    idx = []
    glist = []
    blist = []
    mlist = []
    rtlist = []
    
    while end<=len(total):
        y=np.log(II[start:end])
        t=np.array(range(start,end))
        m,b = np.polyfit(t,y,1)
    
        g=[]
        for i in range(start, end-1):
            oo=((RR[i+1]-RR[i])/II[i])
            g.append(oo)
        
        gamma=np.mean(g)
        beta = m+gamma
        R0 = beta/gamma
        
        idx.append(start)
        mlist.append(m)
        glist.append(gamma)
        blist.append(beta)
        rtlist.append(R0)
        
        start = end
        end = start+window
        
    #idx = np.array(idx[::-1])
    #mlist = mlist[::-1]
    #glist = glist[::-1]
    #blist = blist[::-1]
    #rtlist = rtlist[::-1]
        
    df = pd.DataFrame([idx,mlist,glist,blist,rtlist]).transpose()
    df.columns=["date_id", "m", "gamma", "beta", "Rt"]
    df['Infectious Pd']=1/df['gamma']
    df['date_id']=df['date_id'].astype(int)
    startdate = pd.Timestamp('2020-01-22')
    df['time_added'] = pd.to_timedelta(df['date_id'],'d')
    df['Date'] = startdate+df['time_added']
    df.drop(['time_added'],axis='columns', inplace=True)
    # df =df.drop([0,1,2,3])
    df.drop(df[df['Rt'] > 2.65].index, inplace = True) 
    df.reset_index(drop=True)
    df.head()
    
    # Plots
    pfig0 = make_subplots(rows=1, cols=2,subplot_titles=('Beta vs Gamma', "R(t)"))
    pfig0.add_trace(go.Scatter(x=df['Date'], y=df['gamma'].round(3),line_color='blue'),row=1, col=1)
    pfig0.add_trace(go.Scatter(x=df['Date'], y=df['beta'].round(3),line_color='red'),row=1, col=1)
    pfig0.add_trace(go.Scatter(x=df['Date'], y=df['Rt'].round(3),line_color='blue'),row=1, col=2)
    
    # Update xaxis properties
    pfig0.update_xaxes(title_text="Date", row=1, col=1)
    pfig0.update_xaxes(title_text="Date", row=1, col=2)
    
    
    pfig0.update_layout( xaxis_title='Date',
                        #yaxis_title='Deaths',
                        width=740, height=320,
                        margin=dict(r=10, b=10, l=10, t=30),
                        template = 'seaborn',
                        showlegend=False,
                        )
    
    
    
    pfig0.add_shape(type="line",
        x0= df['Date'].iloc[0], x1= df['Date'].iloc[-1],
        y0= 1, y1= 1,
        xref='x2',yref='y2',
        line=dict(
            color="black",
            width=1,
            dash="dashdot",
        )
    )
    
    
    ## Model
    
    # Global Variables
    N = 1387e6 #Total Population
    
    timestops = df['date_id'].values.tolist()
    tdate = []
    Imod = []
    Smod = []
    Rmod = []
    
    #Initial Conditions
    I0 = II[timestops[0]]
    R0 = RR[timestops[0]]
    S0 = N-I0-R0
    days = window
    t = np.linspace(0,days,days)
    
    for tx in timestops:
      
        # Initial coditions vector
        y0 = S0,I0,R0
        beta = df[df['date_id']==tx]['beta'].iloc[0]
        gamma = df[df['date_id']==tx]['gamma'].iloc[0]
        ret = spi.odeint(sir,y0,t,args=(N,beta,gamma))
        S,I,R = ret.T
    
        tt = list(map(lambda x : x + tx, t)) 
        I = I.tolist()
        tdate.append(tt[1:])
        Imod.append(I[1:])
        Smod.append(S[1:])
        Rmod.append(R[1:])
        tlist = [val for sublist in tdate for val in sublist]
        Ilist = [val for sublist in Imod for val in sublist]
        Slist = [val for sublist in Smod for val in sublist]
        Rlist = [val for sublist in Rmod for val in sublist]
        
        #reset initial conditions
        I0 = I[-1]
        R0 = R[-1]
        S0 = S[-1]
    
    #Initial Conditions
    I0 = Ilist[-1]
    R0 = Rlist[-1]
    S0 = N-I0-R0
    days = 180
    t = np.linspace(tlist[-1]+1,tlist[-1]+days,days)
    # Initial coditions vector
    y0 = S0,I0,R0
    gamma = df['gamma'].iloc[-3:-1].mean()
    beta = df['beta'].iloc[-3:-1].mean()
    
    ret = spi.odeint(sir,y0,t,args=(N,beta,gamma))
    S,I,R = ret.T
    
    startdate = datetime.strptime('2020-01-22','%Y-%m-%d')
    n = len(total)
    trange = np.arange(0,n-1).tolist()
    trange
    
    tdate = []
    td = []
    ta = []
    
    
    for i in tlist:
        tdate.append(startdate+timedelta(i))
    
    for i in t:
        td.append(startdate+timedelta(i+1))
        
    for i in trange:
        ta.append(startdate+timedelta(i))
    
    
    Total_cases_mod = list(map(lambda x : N-x, Slist)) 
    Daily_cases_mod = np.diff(Total_cases_mod)
    daily_cases = np.diff(total)
    Daily_cases_mod = savgol_filter(Daily_cases_mod, 21, 1) 
    
    C = list(map(lambda x : N-x, S))
    dc = np.diff(C)
    
    pfig1 = go.Figure()
    pfig1.add_trace(go.Scatter(x=ta,y=daily_cases, mode="markers", name="Actual"))
    pfig1.add_trace(go.Scatter(x=tdate,y=Daily_cases_mod, mode="lines", name="Model"))
    pfig1.add_trace(go.Scatter(x=td[1:],y=dc, mode="lines", name="Predicted"))
    
    
    pfig1.update_layout( xaxis_title='Date',
                        yaxis_title='No of Cases',
                        margin=dict(r=10, b=0, l=10, t=30),
                        title={"text": "Covid-19 India Prediction",
                               "x": 0.5,"y": 0.97,"xanchor": "center","yanchor": "bottom",
                            "font": {'size': 14}
                        },
                        legend=dict(x=.84,y=.98),
                        width=740, height=420,
                        template = 'seaborn',
                        annotations=[
                            dict(
                                x=0.97,
                                y=0.03,
                                xref="paper",
                                yref="paper",
                                text="Kaustuv",
                                ax=0,
                                ay=0
                                )
                            ]
                      )
  

    return pfig0, pfig1
