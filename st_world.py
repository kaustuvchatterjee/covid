#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 13:27:59 2020

@author: kaustuv
"""
# Import Libraries
import streamlit as st
import pandas as pd
import plotly.express as px

# Read data from eCDC website
@st.cache
def load_data():
    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    data = pd.read_csv(url)
    data['dateRep'] = pd.to_datetime(data['dateRep'], format = '%d/%m/%Y')
    data = data.reindex(index=data.index[::-1])
    data = data.dropna()
    return data
data = load_data()

# Calculate Stats for countires
cntry_id = data["geoId"].unique()
stats_df = pd.DataFrame(columns=['iso','country', 'population', 'total_deaths', 'dpm', 'gr'])
for geoId in cntry_id:
    cdata = data[data["geoId"]==geoId]
    pop = cdata["popData2019"].max()
    total_deaths = cdata["deaths"].cumsum().max()
    dpm = round(total_deaths * 1e6 / pop,0)
    country = cdata["countriesAndTerritories"]
    country=country.iloc[0]
    country = country.replace("_"," ")
    iso = cdata["countryterritoryCode"]
    iso = iso.iloc[0]
    d = cdata["deaths"].cumsum()
    ldpm = d * 1e6 / pop
    delta = ldpm.iloc[-15::].diff()
    gr = delta[1::]*100/ldpm[-14::]
    gr = round(gr.mean(),2)

    stats_df = stats_df.append({'population': pop, 'total_deaths': total_deaths, 
                                'dpm': dpm, 'country': country, 'gr': gr, 'iso': iso },
                               ignore_index=True)

stats_df = stats_df.dropna()

#World deaths per million scatter plot
fig1 = px.scatter(stats_df, x="population", y="dpm",
                     log_x=True,
                     color = "gr",
                     range_color = [0,3],
                     size="dpm",
                     labels={"gr": "Growth Rate(%)", "population": "Population", "dpm": "Deaths/million"},
                     hover_name=stats_df["country"],
                     color_continuous_scale='RDYlGn_r',
                     width=1200, height=600)

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

#World deaths per million map
max_dpm = stats_df['dpm'].max()
fig2 = px.choropleth(data_frame = stats_df,
                    locations= "iso",
                    color="dpm",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                    range_color=[0, 1000],
                    hover_data=["dpm", "gr"],
                    labels={'dpm':'Death/million', 'gr':'Growth Rate(%)'},
                    width=1200, height=600)

fig2.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - Deaths per Million Population',
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

#World growth rate map
max_col = stats_df['gr'].max()
fig3 = px.choropleth(data_frame = stats_df,
                    locations= "iso",
                    color="gr",  # value in column 'Confirmed' determines color
                    hover_name= "country",
                    color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                    range_color=[0, 3],
                    hover_data=['dpm','gr'],
                    labels={'dpm':'Deaths/million', 'gr':'Growth Rate(%)'},
                    width=1200, height=600
                    )

# fig.update_layout(title_text = 'Covid-19 - Current Growth Rate')
fig3.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                   title_text = 'Covid-19 - Current Growth Rate',
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
## Mortality vs Population
'''
st.plotly_chart(fig1)

'''
## Mortality
'''
st.plotly_chart(fig2)

'''
Growth Rate
'''
st.plotly_chart(fig3)
