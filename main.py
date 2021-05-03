#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 10:54:49 2020

@author: kaustuv
"""
# Import Libraries

import streamlit as st

# Import Files
from world import create_worldfigs
from states import create_statefigs
from districts import create_distfigs
from india import create_indfigs
from indiapred import india_pred
from mumbai import create_mumfigs

#Streamlit Code
#st.title('Covid - 19 in Charts')
#---------------------------------
st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 320px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)
#-----------------------------------

domain = ['World','India','India - States','India - Districts','India - Prediction','Mumbai']
domain_opt = st.sidebar.selectbox('Select Domain:',domain)


if domain_opt == domain[0]:

    wfig0, wfig1, wfig2, wfig3 = create_worldfigs()
    
    choices = ['Population v/s Deaths per Million',
               'Map - Deaths per Million',
               'Map - Growth Rate',
               'Death per Million Population']
    
    '''
    ## World
    '''
    
    option = st.sidebar.selectbox('Select Chart:',choices)
    if option == choices[0]:
        st.plotly_chart(wfig0)
        st.text('Hover on data points to see additional data')
        st.text('Data Source: ECDC - https://opendata.ecdc.europa.eu/covid19/nationalcasedeath/csv')
    if option == choices[1]:
        st.plotly_chart(wfig1)
        st.text('Hover on countries to see additional data')
        st.text('Data Source: ECDC - https://opendata.ecdc.europa.eu/covid19/nationalcasedeath/csv')
    if option == choices[2]:
        st.plotly_chart(wfig2)
        st.text('Hover on countries to see additional data')
        st.text('Data Source: ECDC - https://opendata.ecdc.europa.eu/covid19/nationalcasedeath/csv')
    if option == choices[3]:
        st.plotly_chart(wfig3)
        st.text('Data Source: ECDC - https://opendata.ecdc.europa.eu/covid19/nationalcasedeath/csv')
        
        
if domain_opt == domain[1]:
    '''
    # India
    '''
    ifig0, ifig1, ifig2, ifig3 = create_indfigs()
    
    choices = ['Cases & Deaths',
               'Morbidity & Mortality Rates',
               'Growth Rates',
               'Case Fatality Ratio']
    
    option = st.sidebar.selectbox('Select Chart:',choices)
    if option == choices[0]:
        st.plotly_chart(ifig0)
        st.text('Data Source: JHU CSSE COVID-19 Data - https://github.com/CSSEGISandData/COVID-19')
    if option == choices[1]:
        st.plotly_chart(ifig1)
        st.text('Data Source: JHU CSSE COVID-19 Data - https://github.com/CSSEGISandData/COVID-19')
    if option == choices[2]:
        st.plotly_chart(ifig2)
        st.text('Data Source: JHU CSSE COVID-19 Data - https://github.com/CSSEGISandData/COVID-19')
    if option == choices[3]:
        st.plotly_chart(ifig3)
        st.text('Data Source: JHU CSSE COVID-19 Data - https://github.com/CSSEGISandData/COVID-19')
        st.text('CFR calculated using following formula -')
        st.text('CFR = Deaths*100/(Deaths+Recovered)')
        st.text('Ref: Estimating Mortality from Covid-19, A Scientific Brief. WHO. 04 August 2020.')    
        
if domain_opt==domain[2]:
    '''
    ## India - States
    '''
    sfig0, sfig1, sfig2, sfig3, sfig4 = create_statefigs()
    
    choices = ['Population v/s Deaths per Million',
               'Estimated Prevalence',
               'Map - Deaths per Million',
               'Map - Growth Rate',
               'Map - Estimated Prevalence',]
    
    option = st.sidebar.selectbox('Select Chart:',choices)
    if option == choices[0]:
        st.plotly_chart(sfig0)
        st.text('Hover on data points to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    if option == choices[1]:
        st.plotly_chart(sfig1)
        st.text('Hover on data points to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    if option == choices[2]:
        st.plotly_chart(sfig2)
        st.text('Hover on states to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    if option == choices[3]:
        st.plotly_chart(sfig3)
        st.text('Hover on states to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/state_wise_daily.csv')
    if option == choices[4]:
        st.plotly_chart(sfig4)
        st.text('Hover on states points to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/state_wise_daily.csv')
        
        
if domain_opt == domain[3]:
    '''
    ## India - Districts
    '''
    dfig0, dfig1, dfig2, dfig3, dfig4 = create_distfigs()
    
    choices = ['District-wise Total Deaths',
               'Deaths per Million',
               'Map - Deaths per Million',
               'Map - Growth Rate',
               'Map - Estimated Prevalence',]
    
    option = st.sidebar.selectbox('Select Chart:',choices)
    if option == choices[0]:
        st.plotly_chart(dfig3)
        st.text('Hover on data points to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/districts.csv')
    if option == choices[1]:
        st.plotly_chart(dfig4)
        st.text('Hover on data points to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/districts.csv')
    if option == choices[2]:
        st.plotly_chart(dfig0)
        st.text('Hover on districts to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/districts.csv')
    if option == choices[3]:
        st.plotly_chart(dfig1)
        st.text('Hover on districts to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/districts.csv')
    if option == choices[4]:
        st.plotly_chart(dfig2)
        st.text('Hover on districts to see additional data')
        st.text('Data Source: https://api.covid19india.org/csv/latest/districts.csv')
        
if domain_opt == domain[4]:
    r'''
    ## India - Prediction
    '''
    try:
        pfig0, pfig1, pfig2 = india_pred()
        r'''
        Model based on SIR model with $$\beta$$ & $$\gamma$$ estimated piecewise over 7-day sliding window.
        '''
        st.plotly_chart(pfig0)
        
        r'''
        Prediction based on estimated parameters of previous 7 days.
        '''
        
        st.plotly_chart(pfig1)
        st.text('Data Source: JHU CSSE COVID-19 Data - https://github.com/CSSEGISandData/COVID-19')
        st.plotly_chart(pfig2)
        st.text('Data Source: JHU CSSE COVID-19 Data - https://github.com/CSSEGISandData/COVID-19')       
    
    except:
        st.text('Unable to load data! Please try after some time.')
            


if domain_opt == domain[5]:
    '''
    ## Mumbai
    '''
    mfig0 = create_mumfigs()
    st.plotly_chart(mfig0)
    st.text('Data Source: https://api.covid19india.org/csv/latest/districts.csv')