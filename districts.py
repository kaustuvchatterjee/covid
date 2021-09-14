#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 13:18:59 2020

@author: kaustuv
"""

# Import libraries

import pandas as pd
import numpy as np
import plotly.express as px
import geopandas as gpd
import streamlit as st

# Read data from covid19india website
# @st.cache(allow_output_mutation=True)
def load_data():
    url = 'https://api.covid19india.org/csv/latest/districts.csv'
    data = pd.read_csv(url)
    cdata = data
    cdata = cdata.drop(columns=["Other","Tested"])
    idx = cdata[(cdata["District"]=="Unknown") | (cdata["District"]=="Others") | 
                (cdata["District"]=="State Pool") |
               (cdata["District"]=="Foreign Evacuees") |
               (cdata["District"]=="Other State") |
               (cdata["District"]=="Italians") |
               (cdata["District"]=="Airport Quarantine") |
               (cdata["District"]=="Railway Quarantine") |
               (cdata["District"]=="Evacuees")].index
    cdata.drop(idx, inplace = True)
    
    popdata = pd.read_csv('distpop.csv')
    sdf = pd.DataFrame(columns = ["state","district","total_deaths","growth_rate","dpm",'prev',"population","censuscode"])
    udf = cdata.groupby(["State","District"]).size().reset_index()
    
    for i in range(len(udf)):
        df = cdata[(cdata["State"]==udf.iloc[i]["State"]) & (cdata["District"]==udf.iloc[i]["District"])]
        state = df["State"].iloc[0]
        district = df["District"].iloc[0]
        total_deaths = df["Deceased"].max()
            
        p =  popdata[(popdata["State"]==state) & (popdata["District"]==district)]
        if not p.empty:
            pop = p["Population"].iloc[0]
            censuscode = p['Censuscode'].iloc[0].astype(int)
            
        dpm = round(total_deaths*1e6/pop,2)
        
        d = df["Deceased"]
        delta = d.diff()
        gr = delta[1::]*100/d.iloc[0:len(d)]
        gr = gr[-14::]
        gr_mean = round(gr.mean(),2)
        prev = round(dpm*0.0666,2)
        
    
        
        sdf = sdf.append({'state': state, 'district': district, 'total_deaths': total_deaths,
                         'growth_rate': gr_mean, "dpm": dpm, 'prev': prev, "population": pop, "censuscode": censuscode},
                        ignore_index=True)
        
    idx = sdf[sdf["growth_rate"]<0].index
    sdf.drop(idx, inplace = True)
    idx = sdf[sdf["total_deaths"]==0].index
    sdf.drop(idx, inplace = True)
    sdf = sdf.replace([np.inf, -np.inf],np.nan)
    sdf = sdf.dropna()
    sdf['censuscode'] = sdf['censuscode'].astype(int, errors='ignore')
    return sdf


@st.cache
def load_geodata(sdf):
    filename = 'maps/d1.shp'
    geo_df = gpd.read_file(filename).to_crs("EPSG:4326")

    dpm=[]
    gr=[]
    prev=[]
    
    cc = geo_df['censuscode'].values
    
    for s in cc:
        if len(sdf[sdf['censuscode']==s]['dpm'])>0:
            dpm.append(round(sdf[sdf['censuscode']==s]['dpm'].values[0],2))
        else:
            dpm.append(0)
        
        if len(sdf[sdf['censuscode']==s]['growth_rate'])>0:
            gr.append(round(sdf[sdf['censuscode']==s]['growth_rate'].values[0],2))
        else:
            gr.append(0)
            
        if len(sdf[sdf['censuscode']==s]['prev'])>0:
            prev.append(round(sdf[sdf['censuscode']==s]['prev'].values[0],2))
        else:
            prev.append(0)
    
    geo_df['dpm']=dpm
    geo_df['gr']=gr
    geo_df['prev']=prev
    
    return geo_df


def create_distfigs():
    sdf = load_data()
    geo_df = load_geodata(sdf)
    
    max_col = geo_df['dpm'].max()
    
    dfig0 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                        color="dpm",
                        locations=geo_df.index,
                        hover_name= "DISTRICT",
                        color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                        range_color=[0, max_col],
                        hover_data=['dpm','gr'],
                        labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)'}
                       )
    dfig0.update_geos(fitbounds="locations", visible=False)
    dfig0.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    dfig0.update_layout(title={
                            "text": "Covid-19 - India Districts - Deaths per Million Population",
                            "x": 0.5,
                            "y": 0.95,
                            "xanchor": "center",
                            "yanchor": "top"
                        },
                        width = 740, height = 480,                      
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
    
    dfig1 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                        color="gr",
                        locations=geo_df.index,
                        hover_name= "DISTRICT",
                        color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                        range_color=[0, 3],
                        hover_data=['dpm','gr'],
                        labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)'}
                       )
    dfig1.update_geos(fitbounds="locations", visible=False)
    dfig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    dfig1.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                       title_text = 'Covid-19 - India Districts - Growth Rate',
                       width = 740, height = 480,
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
    
    dfig2 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                        color="prev",
                        locations=geo_df.index,
                        hover_name= "DISTRICT",
                        color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green
                        range_color=[0, 100],
                        hover_data=['dpm','gr','prev'],
                        labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)','prev': 'Est Prevalence(%)'}
                       )
    dfig2.update_geos(fitbounds="locations", visible=False)
    dfig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    dfig2.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                       title_text = 'Covid-19 - India Districts - Estimated Prevalence',
                       width = 740, height = 480,
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
    
    dfig3 = px.scatter(sdf, x="state", y="total_deaths",
                     size='total_deaths',
                     color='growth_rate',
                     range_color = [0,3],
                     labels={"state": "State", "total_deaths": "Deaths", 
                             "growth_rate": "Growth Rate(%)",
                             "dpm": "Deaths/million",
                             "prev": "Est Prevalence(%)"},
                     hover_name="district",
                     hover_data=["dpm","prev"],
                      color_continuous_scale='RdYlGn_r'
                     )
    
    dfig3.update_layout(title={'text':'Covid-19 District-wise Total Deaths',
                               "x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                       width = 740, height = 480,
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
    dfig3.update(layout_coloraxis_showscale=True)
    dfig3.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    
    dfig4 = px.scatter(sdf, x="population", y="dpm",
                     size='dpm',
                     color='growth_rate',
                     range_color = [0,3],
                     labels={"state": "State", "total_deaths": "Deaths", 
                             "growth_rate": "Growth Rate(%)",
                             "dpm": "Deaths/million",
                             "prev": "Est Prevalence(%)",
                             "population":"Population"},
                     hover_name="district",
                     hover_data=["dpm","prev"],
                     color_continuous_scale='RdYlGn_r',
                     log_x=True
                     )
    
    dfig4.update_layout(title={"text":"Covid-19 District-wise Mortality Rate","x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                        width = 740, height = 480,
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
    dfig4.update(layout_coloraxis_showscale=True)
    dfig4.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    return dfig0, dfig1, dfig2, dfig3, dfig4
