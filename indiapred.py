#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:44:11 2020

@author: kaustuv
"""
# Import Libraries
# import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.integrate as spi
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
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
# @st.cache
def load_inddata():
    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    # data = pd.read_csv(url)
    # # data.head()
    #  # Filter data for India
    # india_df = data.loc[data['Country/Region']=='India']
    # india_df.drop(['Province/State','Country/Region','Lat','Long'],
    #   axis='columns', inplace=True)
    # total=india_df.values.tolist()[0]
    
    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    # data = pd.read_csv(url)
    # # Filter data for India
    # india_df = data.loc[data['Country/Region']=='India']
    # india_df.drop(['Province/State','Country/Region','Lat','Long'],
    #   axis='columns', inplace=True)
    # deaths=india_df.values.tolist()[0]
    
    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    # data = pd.read_csv(url)
    # # Filter data for India
    # india_df = data.loc[data['Country/Region']=='India']
    # india_df.drop(['Province/State','Country/Region','Lat','Long'],
    #   axis='columns', inplace=True)
    # recovered=india_df.values.tolist()[0]
    
    #################
    url = 'https://api.covid19tracker.in/data/csv/latest/case_time_series.csv'
    data = pd.read_csv(url)
    
    drop_cols = ['Date_YMD','Daily Confirmed','Daily Recovered','Daily Deceased']
    data = data.drop(drop_cols,axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    
     # Missing Dates
    data = data.groupby(pd.Grouper(key='Date',freq='D')).max().rename_axis('Date').reset_index()
    for i in range(len(data)):
        if np.isnan(data['Total Confirmed'].iloc[i]):
            data['Total Confirmed'].iloc[i] = data['Total Confirmed'].iloc[i-1]
            data['Total Recovered'].iloc[i] = data['Total Recovered'].iloc[i-1]
            data['Total Deceased'].iloc[i] = data['Total Deceased'].iloc[i-1]    

    
    total = data['Total Confirmed'].tolist()
    deaths = data['Total Deceased'].tolist()
    recovered = data['Total Recovered'].tolist()
    startdate = data['Date'].iloc[0]
    #################
    
    # url = 'https://data.covid19india.org/csv/latest/states.csv'
    # data = pd.read_csv(url)
    # data = data[data["State"]=="India"]
    # data["Date"] = pd.to_datetime(data['Date'])
    
    # r = pd.date_range(start=data.Date.min(), end=data.Date.max())
    # data = data.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()
    
    
    # for row in range(len(data)):
    #     if data['State'][row] == 0.0:
    #         data['Confirmed'].iloc[row] = data['Confirmed'].iloc[row-1]
    #         data['Recovered'].iloc[row] = data['Recovered'].iloc[row-1]
    #         data['Deceased'].iloc[row] = data['Deceased'].iloc[row-1]
    #         data['State'].iloc[row] = 'India'
   
    
    # url = 'https://raw.githubusercontent.com/datameet/covid19/master/data/all_totals.json'
    # data = pd.read_json(url)
    # df = pd.json_normalize(data.rows)
    
    # date = []
    # active_cases = []
    # cured = []
    # deaths = []
    # total = []
    
    
    # for i in range(len(df)):
    #     if df.key[i][1] == 'active_cases':
    #         date.append(df.key[i][0])
    #         active_cases.append(df['value'].iloc[i])
        
    #     if df.key[i][1] == 'cured':
    #         cured.append(df['value'].iloc[i])
    
    #     if df.key[i][1] == 'death':
    #         deaths.append(df['value'].iloc[i]) 
            
    #     if df.key[i][1] == 'total_confirmed_cases':
    #         total.append(df['value'].iloc[i])
        
    # data = pd.DataFrame({'date': date, 'active_cases': active_cases, 'cured': cured, 'deaths': deaths, 'total': total})
    # data['date'] = pd.to_datetime(data['date'])
    
    # # Cleanup
    
    # for i in range(len(data)-2):
    #     if data['cured'].iloc[i+1]<data['cured'].iloc[i]:
    #         data['cured'].iloc[i+1] = (data['cured'].iloc[i]+data['cured'].iloc[i+2])/2
    #     if data['deaths'].iloc[i+1]<data['deaths'].iloc[i]:
    #         data['deaths'].iloc[i+1] = (data['deaths'].iloc[i]+data['deaths'].iloc[i+2])/2
    #     if data['total'].iloc[i+1]<data['total'].iloc[i]:
    #         data['total'].iloc[i+1] = (data['total'].iloc[i]+data['total'].iloc[i+2])/2

    # # Missing Dates
    # data = data.groupby(pd.Grouper(key='date',freq='D')).max().rename_axis('date').reset_index()
    
    # for i in range(len(data)):
    #     if np.isnan(data['active_cases'].iloc[i]):
    #         data['active_cases'].iloc[i] = data['active_cases'][i-1]
    #         data['cured'].iloc[i] = data['cured'][i-1]
    #         data['deaths'].iloc[i] = data['deaths'][i-1]
    #         data['total'].iloc[i] = data['total'][i-1]        
       
    # total = data['total'].tolist()
    # deaths = data['deaths'].tolist()
    # recovered = data['cured'].tolist()
    
    return total,deaths,recovered


# total,deaths,recovered, startdate = load_inddata()


def india_pred():
    total,deaths,recovered, startdate = load_inddata()
    
    # Check for zero at last record
    if (total[-1]-total[-2] <= 0) | (deaths[-1]-deaths[-2] <= 0) | (recovered[-1]-recovered[-2] <= 0):
        total = total[:-1]
        deaths = deaths[:-1]
        recovered = recovered[:-1] 
    
    RR=[x+y for x, y in zip(deaths, recovered)]
    II=[x-y for x, y in zip(total, RR)]
    
    l = len(total)
    # s = 43
    s = 35
    
    # while s<=49:
    # while s<=41:
    #     if (l-s)%7 == 0:
    #         start = s
    #         break
    #     s+=1
    
    window = 7
    start = 0
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
            print(II[i])
            if II[i]==0.0:
                oo = 0.0
            else:
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
        
        start = start+1
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
    # startdate = pd.Timestamp('2020-01-22')
    startdate = pd.Timestamp('2020-03-03')
    df['time_added'] = pd.to_timedelta(df['date_id'],'d')
    df['Date'] = startdate+df['time_added']
    df.drop(['time_added'],axis='columns', inplace=True)
    df = df[df['Date']>='2020-04-01']
    df.reset_index(drop=True, inplace=True)
    # df =df.drop([0,1,2,3])
    idx = df[df['Rt']<2.65].index[0]
    idx = np.arange(0,idx).tolist()
    df =df.drop(index=idx)
    # df.drop(df[df['Rt'] > 2.65].index, inplace = True) 
    df.reset_index(drop=True, inplace=True)
    # df.head()
    
   
    ## Model
    
    # Global Variables
    N = 1387e6 #Total Population
    
    timestops = df['date_id'].values.tolist()
    tdate = []
    Imod = []
    Smod = []
    Rmod = []
    
    #Initial Conditions

    days = window
    t = np.linspace(0,days-1,days)
    
    for tx in timestops:
        I0 = II[tx]
        R0 = RR[tx]
        S0 = N-I0-R0
      
        # Initial coditions vector
        y0 = S0,I0,R0
        beta = df[df['date_id']==tx]['beta'].iloc[0]
        gamma = df[df['date_id']==tx]['gamma'].iloc[0]
        ret = spi.odeint(sir,y0,t,args=(N,beta,gamma))
        S,I,R = ret.T
    
        tt = list(map(lambda x : x + tx, t)) 
        tdate.append(tt)
        Imod.append(I)
        Smod.append(S)
        Rmod.append(R)
        tlist = [val for sublist in tdate for val in sublist]
        Ilist = [val for sublist in Imod for val in sublist]
        Slist = [val for sublist in Smod for val in sublist]
        Rlist = [val for sublist in Rmod for val in sublist]
        
    
    ## Prediction
    #Initial Conditions
    I0 = II[-2]
    R0 = RR[-2]
    # S0 = N-I0-R0
    S0 = N-total[-1]
    days = 180
    t = np.linspace(tlist[-1],tlist[-1]+days-1,days)
    # Initial coditions vector
    y0 = S0,I0,R0
    # gamma = df['gamma'].iloc[-2:].mean()
    # beta = df['beta'].iloc[-2:].mean()
    
    # Predict gamma, beta 1 week ahead
   # gamma = df['gamma'].iloc[-1]
    #beta = df['beta'].iloc[-1]
    
    # n=3.4
    # x=[0,1,2,3]
    # g = df['gamma'].iloc[-4:]
    # z = np.polyfit(x,g,2)
    # p = np.poly1d(z)
    # gamma = p(n)

    # b = df['beta'].iloc[-4:]
    # z = np.polyfit(x,b,2)
    # p = np.poly1d(z)
    # beta = p(n)    
    
    
    ret = spi.odeint(sir,y0,t,args=(N,beta,gamma))
    S,I,R = ret.T
    
    # startdate = datetime.strptime('2020-01-23','%Y-%m-%d')
    startdate = datetime.strptime('2020-03-04','%Y-%m-%d')
    n = len(total)
    trange = np.arange(0,n-1).tolist()
    #trange
    
    tdate = []
    td = []
    ta = []
    
    
    for i in tlist:
        tdate.append(startdate+timedelta(i)) ###
    
    for i in t:
        td.append(startdate+timedelta(i))
        
    for i in trange:
        ta.append(startdate+timedelta(i))
    
    
    Total_cases_mod = list(map(lambda x : N-x, Slist))
    Daily_cases_mod = np.ediff1d(Total_cases_mod)
    daily_cases = np.ediff1d(total)
    Daily_cases_mod = savgol_filter(Daily_cases_mod, 21, 1) 
    
    C = list(map(lambda x : N-x, S))
    dc = np.diff(C)
    
    
    ## Predicted R(t) Worst case
    # Get Max R(t)
    rt = np.array(df['Rt'])
    max_rt_idx = find_peaks(rt, distance=3,height=1.5)
    max_rt_idx = max_rt_idx[0][-1]
    max_rt_date = df['Date'].iloc[max_rt_idx]
    max_rt = rt[max_rt_idx]
    A = [x+y for x, y in zip(II[1:], RR[1:])]
    A = np.array(A)
    SS = N-A
    idx = ta.index(max_rt_date)
    
    # Initial Conditions
    SS0 = SS[idx]
    II0 = II[idx+1]
    RR0 = RR[idx+1]
    
    # get beta & gamma at max R(t)
    β = df['beta'].iloc[max_rt_idx]
    γ = df['gamma'].iloc[max_rt_idx]
    
    # Initial conditions vector
    y0 = SS0, II0, RR0
    # A grid of time points (in days)
    d = 180
    t = np.linspace(0, d, d)
    # Integrate the SIR equations over the time grid, t.
    ret = spi.odeint(sir, y0, t, args=(N, β, γ))
    SP, IP, RP = ret.T
    
    stime = max_rt_date
    tr = []
    for i in t:
        tr.append(stime+timedelta(i))
    
    Rt = (SP/SP[0])*(max_rt)

    # Plots
    t1 = '<i>β</i> v/s <i>γ</i>'
    t2 = 'R<sub>t</sub>'
    pfig0 = make_subplots(rows=1, cols=2,subplot_titles=(t1, t2))
    pfig0.add_trace(go.Scatter(x=df['Date'], y=df['gamma'].round(3),line_color='blue',name='Gamma'),row=1, col=1)
    pfig0.add_trace(go.Scatter(x=df['Date'], y=df['beta'].round(3),line_color='red',name='Beta'),row=1, col=1)
    pfig0.add_trace(go.Scatter(x=df['Date'], y=df['Rt'].round(3),line_color='blue',name='R(t)'),row=1, col=2)
    pfig0.add_trace(go.Scatter(x=[df['Date'].iloc[max_rt_idx]], y=[df['Rt'].iloc[max_rt_idx]]),row=1, col=2)
    pfig0.add_trace(go.Scatter(x=tr, y=Rt,line_color='red',name='Pred R(t)',mode="lines",line=dict(dash= "dashdot",width=1)),row=1, col=2)
    
    # Update xaxis properties
    pfig0.update_xaxes(title_text="Date", row=1, col=1)
    pfig0.update_xaxes(title_text="Date", row=1, col=2)
    
    
    pfig0.update_layout( xaxis_title='Date',
                        #yaxis_title='Deaths',
                        width=740, height=320,
                        margin=dict(r=10, b=10, l=10, t=60),
                        template = 'seaborn',
                        showlegend=False,
                        )
    
    
    
    pfig0.add_shape(type="line",
        x0= df['Date'].iloc[0], x1= tr[-1],
        y0= 1, y1= 1,
        xref='x2',yref='y2',
        line=dict(
            color="black",
            width=1,
            dash="dashdot",
        )
    )
    
    
    pfig1 = go.Figure()
    pfig1.add_trace(go.Scatter(x=ta,y=daily_cases, mode="markers", name="Actual"))
    pfig1.add_trace(go.Scatter(x=tdate,y=Daily_cases_mod, mode="lines", name="Model"))
    pfig1.add_trace(go.Scatter(x=td,y=dc, mode="lines", name="Predicted"))
    
    
    pfig1.update_layout( xaxis_title='Date',
                        yaxis_title='No of Cases',
                        margin=dict(r=10, b=0, l=10, t=60),
                        title={"text": "Covid-19 India Prediction - Cases",
                               "x": 0.5,"y": 0.9,"xanchor": "center","yanchor": "bottom",
                            "font": {'size': 14}
                        },
                        legend=dict(x=.1,y=.98),
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
  
    
    #Predicted Deaths
    daily_deaths = np.diff(deaths)
    cfr = daily_deaths/daily_cases
    cfr = cfr[-14:].mean()
    predicted_deaths = dc*cfr
    smooth_deaths = savgol_filter(daily_deaths, 21, 1)

    pfig2 = go.Figure()
    pfig2.add_trace(go.Scatter(x=ta,y=daily_deaths, mode="markers", name="Actual"))
    pfig2.add_trace(go.Scatter(x=ta,y=smooth_deaths, mode="lines", name="Smoothed"))
    pfig2.add_trace(go.Scatter(x=td,y=predicted_deaths, mode="lines", name="Predicted"))    


    pfig2.update_layout( xaxis_title='Date',
                        yaxis_title='No of Deaths',
                        margin=dict(r=10, b=0, l=10, t=60),
                        title={"text": "Covid-19 India Prediction - Deaths",
                               "x": 0.5,"y": 0.9,"xanchor": "center","yanchor": "bottom",
                            "font": {'size': 14}
                        },
                        legend=dict(x=.1,y=.98),
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

    
    return pfig0, pfig1, pfig2
