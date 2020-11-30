#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:04:44 2020

@author: kaustuv
"""

# Import Libraries
import streamlit as stl
import pandas as pd
#import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.integrate as spi
#from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from datetime import datetime, timedelta

# Read data from eCDC website
@stl.cache
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
data = load_data()


fig1 = make_subplots(rows=2, cols=2,
                     subplot_titles=("Daily Cases", "Daily Deaths",
                                     "Total Cases","Total Deaths"))

fig1.add_trace(go.Scatter(x=data['dateRep'], y=data['cases'],name='Daily Cases',line_color='blue'),row=1, col=1)
fig1.add_trace(go.Scatter(x=data['dateRep'], y=data['cumcases'], name='Total Cases',line_color='blue'),row=2, col=1)
fig1.add_trace(go.Scatter(x=data['dateRep'], y=data['deaths'],name='Daily Deaths',line_color='blue'),row=1, col=2)
fig1.add_trace(go.Scatter(x=data['dateRep'], y=data['cumdeaths'], name='Total Deaths',line_color='blue'),row=2, col=2)
fig1.update_layout( xaxis_title='Date',
                    yaxis_title='Deaths',
                    width=1200, height=600,
                    showlegend=False,
                    )

fig2 = make_subplots(rows=2, cols=1,
                     subplot_titles=("Cases per million", "Deaths per million"))
fig2.add_trace(go.Scatter(x=data['dateRep'], y=data['dpc'],name='Cases per million',line_color='blue'),row=1, col=1)
fig2.add_trace(go.Scatter(x=data['dateRep'], y=data['dpm'], name='Deaths per million',line_color='blue'),row=2, col=1)
fig2.update_layout( xaxis_title='Date',
                    #yaxis_title='Deaths',
                    width=1000, height=600,
                    showlegend=False,
                    )
fig2.update_yaxes(type="log")
#Streamlit render
'''
## India - Cases & Deaths

'''
stl.plotly_chart(fig1)

'''
## India - Mortality & Morbidity

'''
stl.plotly_chart(fig2)

#---------------------------------------------------------------------
# India - Prediction
#--------------------------------------------------------------------


@stl.cache
# Read data from JHU website
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

RR=[x+y for x, y in zip(deaths, recovered)]
II=[x-y for x, y in zip(total, RR)]

window = 7
en = len(total)
st = en-window

idx = []
glist = []
blist = []
mlist = []
rtlist = []

while st>40:
    y=np.log(II[st:en])
    t=np.array(range(st,en))
    m,b = np.polyfit(t,y,1)

    g=[]
    for i in range(st, en-1):
        oo=((RR[i+1]-RR[i])/II[i])
        g.append(oo)
    
    gamma=np.mean(g)
    beta = m+gamma
    R0 = beta/gamma
    
    idx.append(st)
    mlist.append(m)
    glist.append(gamma)
    blist.append(beta)
    rtlist.append(R0)
    
    en = en-window
    st = en-window
    
idx = np.array(idx[::-1])
mlist = mlist[::-1]
glist = glist[::-1]
blist = blist[::-1]
rtlist = rtlist[::-1]

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
fig3 = make_subplots(rows=1, cols=3,
                     subplot_titles=('Beta vs Gamma', "R(t)",
                                     "m = Beta-Gamma"))
fig3.add_trace(go.Scatter(x=df['Date'], y=df['gamma'],line_color='blue'),row=1, col=1)
fig3.add_trace(go.Scatter(x=df['Date'], y=df['beta'],line_color='red'),row=1, col=1)
fig3.add_trace(go.Scatter(x=df['Date'], y=df['Rt'],line_color='blue'),row=1, col=2)
fig3.add_trace(go.Scatter(x=df['Date'], y=df['m'],line_color='blue'),row=1, col=3)

# Update xaxis properties
fig3.update_xaxes(title_text="Date", row=1, col=1)
fig3.update_xaxes(title_text="Date", row=1, col=2)
fig3.update_xaxes(title_text="Date", row=1, col=3)


fig3.update_layout( xaxis_title='Date',
                    #yaxis_title='Deaths',
                    width=1200, height=400,
                    showlegend=False,
                    )
fig3.update_layout(shapes=[
    # adds line at y=1
    dict(
      type= 'line',
      xref= 'x2', x0= df['Date'].iloc[0], x1= df['Date'].iloc[-1],
      yref= 'y2', y0= 1, y1= 1,
    ),
    dict(
      type= 'line',
      xref= 'x3', x0= df['Date'].iloc[0], x1= df['Date'].iloc[-1],
      yref= 'y3', y0= 0, y1= 0,
    ),    ],
    )

#Streamlit render
r'''
## India - Prediction
### Model Parameters
Piecewise estimation of $$\beta$$ and $$\gamma$$ by fitting data to SIR model.
'''
stl.plotly_chart(fig3)


def sir(y,t,N,beta, gamma):
    S,I,R = y
    dSdt = -beta*S*I/N
    dIdt = beta*S*I/N - gamma*I
    dRdt = gamma*I
    return dSdt, dIdt, dRdt

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
#     t=t.tolist()
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

tdate = []
td = []
ta = []


for i in tlist:
    tdate.append(startdate+timedelta(i))

for i in t:
    td.append(startdate+timedelta(i))
    
for i in trange:
    ta.append(startdate+timedelta(i))


Total_cases_mod = list(map(lambda x : N-x, Slist)) 
Daily_cases_mod = np.diff(Total_cases_mod)
daily_cases = np.diff(total)
Daily_cases_mod = savgol_filter(Daily_cases_mod, 21, 1) 

C = list(map(lambda x : N-x, S))
dc = np.diff(C)


fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=ta,y=daily_cases, mode="markers", name="Actual"))
fig4.add_trace(go.Scatter(x=tdate,y=Daily_cases_mod, mode="lines", name="Model"))
fig4.add_trace(go.Scatter(x=td[1:],y=dc, mode="lines", name="Predicted"))


fig4.update_layout( xaxis_title='Date',
                    yaxis_title='No of Cases',
                    margin=dict(r=20, b=10, l=10, t=10),
                    title={
                        "text": "Covid-19 India Prediction",
                        "x": 0.5,
                        "y": 0.97,
                        "xanchor": "center",
                        "yanchor": "top",
                        "font": {'size': 14}
                    },
                    width=1200, height=600,
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


#Streamlit render
'''
### Prediction
'''
r'''
Model based on SIR model with estimated $$\beta$$ & $$\gamma$$.
Prediction based on average of estimated parameters for last 3 weeks. 
'''
stl.plotly_chart(fig4)