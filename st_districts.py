#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:52:46 2020

@author: kaustuv
"""

# Import Libraries
import streamlit as stl
import pandas as pd
import plotly.express as px
import geopandas as gpd
import numpy as np

# Read data from covid19india website
@stl.cache
def load_data():
    url = 'https://api.covid19india.org/csv/latest/districts.csv'
    data = pd.read_csv(url)
    return data


data = load_data()
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
    gr_mean = gr.mean()
    prev = dpm*0.0666
    

    
    sdf = sdf.append({'state': state, 'district': district, 'total_deaths': total_deaths,
                     'growth_rate': gr_mean, "dpm": dpm, 'prev': prev, "population": pop, "censuscode": censuscode},
                    ignore_index=True)
    
idx = sdf[sdf["growth_rate"]<0].index
sdf.drop(idx, inplace = True)
idx = sdf[sdf["total_deaths"]==0].index
sdf.drop(idx, inplace = True)
sdf = sdf.replace([np.inf, -np.inf],np.nan)
sdf.dropna()
sdf['censuscode'] = sdf['censuscode'].astype(int, errors='ignore')


@stl.cache(allow_output_mutation=True)
def load_geodata():
    path = '/run/media/kaustuv/Data/Python/Notebooks/covid/maps/'
    # path = '/run/media/kaustuv/Data/Python/Notebooks/'
    filename = path+'d1.shp'
    geo_df = gpd.read_file(filename).to_crs("EPSG:4326")
    return geo_df

geo_df = load_geodata()

censuscodes = geo_df['censuscode']
dpm=[]
gr=[]
prev=[]

st = geo_df['censuscode'].values

for s in st:
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


#Plot Districts DPM
max_col = geo_df['dpm'].max()

fig1 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                    color="dpm",
                    locations=geo_df.index,
                    hover_name= "DISTRICT",
                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                    range_color=[0, max_col],
                    hover_data=['dpm','gr'],
                    labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)'}
                   )
fig1.update_geos(fitbounds="locations", visible=False)
fig1.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.update_layout(title_text = 'Covid-19 - India Districts - Deaths per Million Population')
fig1.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - India Districts - Deaths per Million Population',
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

#Plot
max_col = geo_df['gr'].max()

fig2 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                    color="gr",
                    locations=geo_df.index,
                    hover_name= "DISTRICT",
                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                    range_color=[0, 3],
                    hover_data=['dpm','gr'],
                    labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)'}
                   )
fig2.update_geos(fitbounds="locations", visible=False)
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.update_layout(title_text = 'Covid-19 - India Districts - Growth Rate')
fig2.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - India Districts - Growth Rate',
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


#Plot
max_col = geo_df['prev'].max()

fig3 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                    color="prev",
                    locations=geo_df.index,
                    hover_name= "DISTRICT",
                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green
                    range_color=[0, 100],
                    hover_data=['dpm','gr','prev'],
                    labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)','prev': 'Est Prevalence(%)'}
                   )
fig3.update_geos(fitbounds="locations", visible=False)
fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.update_layout(title_text = 'Covid-19 - India Districts - Growth Rate')
fig3.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - India Districts - Estimated Prevalence',
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



#Plot
fig4 = px.scatter(sdf, x="population", y="dpm",
                 size='dpm',
                 color='growth_rate',
                 range_color = [0,3],
                 labels={"state": "State", "total_deaths": "Deaths", 
                         "growth_rate": "Growth Rate(%)",
                         "dpm": "Deaths/million",
                         "prev": "Est Prevalence(%)"},
                 hover_name="district",
                 hover_data=["dpm","prev"],
                 title='Covid-19 District-wise Mortality Rate',
                 color_continuous_scale='RdYlGn_r',
                 log_x=True,
                  opacity=.7
                 )

fig4.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
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
fig4.update(layout_coloraxis_showscale=True)
fig4.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

#Streamlit render
'''
## Districts - Mortality vs Population
'''
stl.plotly_chart(fig4)
'''
## Districts - Mortality
'''
stl.plotly_chart(fig1)

'''
## Districts - Growth Rate
'''
stl.plotly_chart(fig2)

'''
## Districts - Estimated Prevalence
'''
stl.plotly_chart(fig3)
