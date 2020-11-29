#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:54:25 2020

@author: kaustuv
"""

# Import Libraries
import streamlit as stl
import pandas as pd
import plotly.express as px
import geopandas as gpd

# Read data from covid19india website
@stl.cache
def load_data():
    url = 'https://api.covid19india.org/csv/latest/state_wise_daily.csv'
    data = pd.read_csv(url)
    return data


data = load_data()

# Calculate statewise stats
state_codes = ['AN','AP','AR','AS','BR','CH','CT','DN','DD','DL','GA','GJ','HR','HP','JK','JH',
              'KA','KL','LA','LD','MP','MH', 'MN','ML','MZ','NL','OR','PY','PB','RJ','SK','TN',
              'TG','TR','UP','UT','WB']

state_names = ['Andaman & Nicobar Is', 'Andhra Pradesh', 'Arunachal Pradesh',
              'Assam', 'Bihar', 'Chandigarh', 'Chattisgarh', 'Dadra & Nagar Haveli',
              'Daman & Diu', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh',
              'Jammu & Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep',
              'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland',
              'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamilnadu', 'Telengana',
              'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal']

state_pop = [380581, 49577103, 1383727,
            31205576, 104099452, 1055450, 25545198, 533688,
            52076, 16787941, 1458545, 60439692, 25351462, 6864602,
            12267032, 32988134, 61095297, 33406061, 274000, 64473,
            72626809, 112374333, 2570390, 2966889, 1097206, 1978502,
            41974219, 1247953, 27743338, 68548437, 610577, 72147030, 35003674,
            3673917, 199812341, 10086292, 91276115]

sdf = pd.DataFrame(columns=['state', 'ST_CD','population', 'total_deaths', 'dpm', 'gr'])

sdata = data[data["Status"]=="Deceased"]

for state_code in state_codes:
    idx = state_codes.index(state_code)
    state = state_names[idx]
    population = state_pop[idx];
    total_deaths = sdata[state_codes[idx]].cumsum().max()
    dpm = round(total_deaths * 1e6 / population,2)
    
    d = sdata[state_codes[idx]].cumsum()
    ldpm = d * 1e6 / population
    delta = ldpm.iloc[-15::].diff()
    gr = delta[1::]*100/ldpm[-14::]
    gr = round(gr.mean(),2)
    prev = round(dpm*0.0666,2)
#     prev = dpm/8.472
    
    sdf = sdf.append({'population': population, 'total_deaths': total_deaths,
                      'dpm': dpm, 'state': state, 'ST_CD': state_code, 'gr': gr, 'prev': prev }, ignore_index=True)
    
sdf = sdf.dropna()

#Plot Death per million scatter plot
fig1 = px.scatter(sdf, x="population", y="dpm",
                     log_x=True,
                     color = "gr",
                     range_color = [0,3],
                     size="dpm",
                     size_max = 20,
                     labels={"gr": "Growth Rate(%)", "population": "Population",
                             "dpm": "Deaths/million", "prev": "Prevalence(%)"},
                     hover_name=sdf["state"],
                     color_continuous_scale='RdYlGn_r')

fig1.update_layout( xaxis_title='Population',
                    yaxis_title='Deaths per million',
                    margin=dict(r=20, b=10, l=10, t=10),
                    title={
                        "text": "Covid-19 - Mortality",
                        "x": 0.5,
                        "y": 0.95,
                        "xanchor": "center",
                        "yanchor": "top"
                    },
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
                   
fig1.update(layout_coloraxis_showscale=True)

fig1.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))

#Prepare GIS data
filename = '/run/media/kaustuv/Data/Python/Notebooks/covid/maps/s2.shp'
geo_df = gpd.read_file(filename).to_crs("EPSG:4326")

dpm = []
gr=[]
prev=[]
st = geo_df['ST_CD'].values

for s in st:
  if len(sdf[sdf['ST_CD']==s]['dpm'])>0:
    dpm.append(round(sdf[sdf['ST_CD']==s]['dpm'].values[0],2))
  else:
    dpm.append(0)

  if len(sdf[sdf['ST_CD']==s]['gr'])>0:
    gr.append(round(sdf[sdf['ST_CD']==s]['gr'].values[0],2))
  else:
    gr.append(0)

  if len(sdf[sdf['ST_CD']==s]['prev'])>0:
    prev.append(round(sdf[sdf['ST_CD']==s]['prev'].values[0],2))
  else:
    prev.append(0)
  


geo_df['dpm']=dpm
geo_df['gr']=gr
geo_df['prev']=prev

#Plot Indian States Mortality
max_col = geo_df['dpm'].max()

fig2 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                    color="dpm",
                    locations=geo_df.index,
                    hover_name= "ST_NM",
                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                    range_color=[0, max_col],
                    hover_data=['dpm','gr','prev'],
                    labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)',
                             'prev': 'Est Prevalence(%)', 'ST_NM': 'State'}
                   )
fig2.update_geos(fitbounds="locations", visible=False)
fig2.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig2.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - India - Deaths per Million Population',
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

#Plot Growth Rate
fig3 = px.choropleth(geo_df, geojson=geo_df.geometry, 
        color="gr",
        locations=geo_df.index,
        hover_name= "ST_NM",
        color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
        range_color=[0, 3],
        hover_data=['dpm','gr','prev'],
        labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)',
                 'prev': 'Est Prevalence(%)', 'ST_NM': 'State'}
           )
fig3.update_geos(fitbounds="locations", visible=False)
fig3.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig3.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - India - Growth Rate',
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

#Plot Estimated Prevalence
fig4 = px.choropleth(geo_df, geojson=geo_df.geometry, 
                    color="prev",
                    locations=geo_df.index,
                    hover_name= "ST_NM",
                    color_continuous_scale= 'RdYlGn',  #  color scale red, yellow green
                    range_color=[0, 100],
                    hover_data=['dpm','gr','prev'],
                    labels={'dpm': 'Deaths/million','gr':'Growth Rate(%)',
                             'prev': 'Est Prevalence(%)', 'ST_NM': 'State'}
                   )
fig4.update_geos(fitbounds="locations", visible=False)
fig4.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig4.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - India - Estimated Prevalence',
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



#Streamlit render
'''
## States - Mortality vs Population
'''
stl.plotly_chart(fig1,use_container_width=True)
'''
## States - Mortality
'''
stl.plotly_chart(fig2)

'''
## States - Growth Rate
'''
stl.plotly_chart(fig3)