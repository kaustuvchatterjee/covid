# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


#"""
#WORLD CHARTS
#"""

# Read data from eCDC website

@st.cache
def load_worlddata():
#    url = 'https://opendata.ecdc.europa.eu/covid19/casedistribution/csv'
    url = 'https://opendata.ecdc.europa.eu/covid19/nationalcasedeath/csv'
    data = pd.read_csv(url)
    data['dateRep'] = data['year_week'].str.replace('-','')+'0'
    data['dateRep'] = pd.to_datetime(data['dateRep'], format = '%Y%W%w')
#    data = data.reindex(index=data.index[::-1])
    data = data.dropna()
    return data



def create_worldfigs():
    
    data = load_worlddata()
    
    # Collect stats
    cntry_id = data["country_code"].unique()
    stats_df = pd.DataFrame(columns=['iso','country', 'population', 'total_deaths', 'dpm', 'gr'])
    for country_code in cntry_id:
        cdata = data[data["country_code"]==country_code]
        pop = cdata["population"].max()
        total_deaths = cdata[cdata["indicator"]=="deaths"]["weekly_count"].cumsum().max()
        dpm = round(total_deaths * 1e6 / pop,0)
        country = cdata["country"]
        country=country.iloc[0]
        country = country.replace("_"," ")
        iso = cdata["country_code"]
        iso = iso.iloc[0]
        d = cdata[cdata["indicator"]=="deaths"]["weekly_count"].cumsum()
        ldpm = d * 1e6 / pop
        delta = ldpm.iloc[-3::].diff()
        gr = delta[1::]*100/ldpm[-3:-1]
        gr = round(gr.mean()/7,2)
    
        stats_df = stats_df.append({'population': pop, 'total_deaths': total_deaths, 
                                    'dpm': dpm, 'country': country, 'gr': gr, 'iso': iso },
                                   ignore_index=True)
    
    stats_df = stats_df.dropna()
    
    # Figure 1: Population v/s Deaths per Million
    wfig0 = px.scatter(stats_df, x="population", y="dpm",
                         log_x=True,
                         color = "gr",
                         range_color = [0,3],
                         size="dpm",
                         labels={"gr": "Growth Rate(%)", "population": "Population", "dpm": "Deaths/million"},
                         hover_name=stats_df["country"],
                         color_continuous_scale='RDYlGn_r')
    
    wfig0.update_layout( xaxis_title='Population',
                        yaxis_title='Deaths per million',
                        margin=dict(r=20, b=10, l=10, t=30),
                        title={
                            "text": "Covid-19 - Mortality",
                            "x": 0.5,
                            "y": 0.95,
                            "xanchor": "center",
                            "yanchor": "bottom"
                        },
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
                       
    wfig0.update(layout_coloraxis_showscale=True)
    
    wfig0.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    
    #Figure 2: World Map - DPM
    wfig1 = px.choropleth(data_frame = stats_df,
                        locations= "iso",
                        color="dpm",  # value in column 'Confirmed' determines color
                        hover_name= "country",
                        color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                        range_color=[0, 1000],
                        hover_data=["dpm", "gr"],
                        labels={'dpm':'Death/million', 'gr':'Growth Rate(%)'})
    
    
    wfig1.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "top"},
                       title_text = 'Covid-19 - Deaths per Million Population',
                       width = 740, height = 480,
                       margin=dict(r=20, b=10, l=10, t=30),
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
    
    #Figure 3: World Map - Growth Rate
    wfig2 = px.choropleth(data_frame = stats_df,
                        locations= "iso",
                        color="gr",  # value in column 'Confirmed' determines color
                        hover_name= "country",
                        color_continuous_scale= 'RdYlGn_r',  #  color scale red, yellow green
                        range_color=[0, 3],
                        hover_data=['dpm','gr'],
                        labels={'dpm':'Deaths/million', 'gr':'Growth Rate(%)'}
                        )
    
    wfig2.update_layout(title={"x": 0.5, "y": 0.95, "xanchor": "center", "yanchor": "bottom"},
                       title_text = 'Covid-19 - Current Growth Rate',
                       width = 740, height = 480,
#                       margin=dict(r=20, b=10, l=10, t=30),
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


    # DPM Time-series
    wfig3 = go.Figure()
    cntry_id = data["country_code"].unique().tolist()
    annotations = []
#    stats_df = pd.DataFrame(columns=['iso','country', 'population', 'total_deaths', 'dpm', 'gr'])
    for geoId in cntry_id:
        cdata = data[data["country_code"]==geoId]
        pop = cdata["population"].max()
        date = cdata["dateRep"]
        total_deaths = cdata[cdata["indicator"]=="deaths"]["weekly_count"].cumsum().max()
        dpm = round(total_deaths * 1e6 / pop,0)
        if total_deaths > 50000:
            country = cdata["country"]
            country=country.iloc[0]
            country = country.replace("_"," ")
            iso = cdata["country_code"]
            iso = iso.iloc[0]
            d = cdata[cdata["indicator"]=="deaths"]["weekly_count"].cumsum()
            ldpm = round(d * 1e6 / pop, 2)
            delta = ldpm.iloc[-3::].diff()
            gr = delta[1::]*100/ldpm[-3:-1]
            gr = round(gr.mean()/7,2)
            
            wfig3.add_trace(go.Scatter(x=date, y=ldpm, mode='lines',name=iso))
            annot = dict(
                x=date.iloc[-1], y=ldpm.iloc[-1], # annotation point
                xref='x', 
                yref='y',
                text="("+str(gr)+"%)",
                showarrow=False,
                font = dict(size=12),
                xanchor='left', yanchor='middle')
            annotations.append(annot)
    
    wfig3.update_layout(title_text = 'Covid-19 - World - Deaths per Million <br> (Countries with more than 50,000 deaths)',
                    xaxis_title='Date',
                    yaxis_title='Deaths per million',
                    width = 740, height=480,
                    margin=dict(r=20, b=10, l=10, t=30),
                    showlegend = True,
                    template = 'seaborn'
                    )
    wfig3.layout.annotations=annotations    
    
    return wfig0, wfig1, wfig2, wfig3
