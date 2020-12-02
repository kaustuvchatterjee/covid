#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:04:44 2020
@author: kaustuv
"""

# Import Libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

@st.cache(allow_output_mutation=True)
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
                        )
    
    ifig1 = make_subplots(rows=2, cols=1,
                         subplot_titles=("Cases per million", "Deaths per million"))
    ifig1.add_trace(go.Scatter(x=data['dateRep'], y=data['dpc'],name='Cases per million',line_color='blue'),row=1, col=1)
    ifig1.add_trace(go.Scatter(x=data['dateRep'], y=data['dpm'], name='Deaths per million',line_color='blue'),row=2, col=1)
    ifig1.update_layout( xaxis_title='Date',
                        #yaxis_title='Deaths',
                        width = 700, height=480,
                        showlegend=False,
                        )
    ifig1.update_yaxes(type="log")
    return ifig0, ifig1
