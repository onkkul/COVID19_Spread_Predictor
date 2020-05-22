#!/usr/bin/env python
# coding: utf-8

# In[18]:


# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:38:54 2020

@author: Onkar
"""

import pycountry
import datetime
import pandas as pd
from datetime import date
from functools import reduce

import plotly.express as px
from fbprophet import Prophet


# In[19]:



def getDaysTillToday(today):
    date_time_obj = datetime.datetime.strptime(today, '%m/%d/%Y')
    today = date.today()
    delta = today - date_time_obj.date()
    return delta.days


def readData(data, case_type):
    # data.head()
    dataset = data.groupby(['ObservationDate']).sum()
    observationDates = dataset.index.values.tolist()
    cases = dataset[case_type].tolist()

    dataset = pd.DataFrame()
    dataset.insert(0, "ds", observationDates, True)
    dataset.insert(1, "y", cases, True)

    return dataset


# In[20]:



def trainModel(dataset, prophet):
    prophet.fit(dataset)
    future = prophet.make_future_dataframe(periods=days)
    forecast = prophet.predict(future)
    return forecast
    


# In[21]:



def getCountryMap():

    countries = list(pycountry.countries)
    Country_Codes = []
    
    for each_country in range(len(countries)):
        n = [countries[each_country].name, countries[each_country].alpha_3]
        Country_Codes.append(n)
    
    Country_Codes = pd.DataFrame(Country_Codes)
    Country_Codes.drop_duplicates()
    Country_Codes.columns = ["Country", "ISO_Code"]
    Country_Codes.set_index('Country', inplace=True)
    
    country_map = Country_Codes.to_dict('index')
    return country_map


def addCountryCode(data):
    Country_Region = data['Country/Region']
    ISO_Code = []
    
    for each_country in Country_Region:
        try:
            ISO_Code.append(country_map[each_country]['ISO_Code'])
        except:
            ISO_Code.append(each_country)

    exceptions = {'Macau': 'MAC', 'South Korea': 'KOR', 'Ivory Coast': 'CIV', 
                  'Others': 'Others', 'North Ireland': 'GBR', 'Republic of Ireland': 'IRL',
                  'St. Martin': 'MAF', 'occupied Palestinian territory': 'PSE',
                  "('St. Martin',)": 'MAF', 'Channel Islands': 'GBR', 'Gambia, The': 'GMB',
                  'Congo (Kinshasa)': 'COD', 'Congo (Brazzaville)': 'COD', 'Bahamas, The': 'BHS',
                  'Cape Verde': 'CPV', 'East Timor': 'TLS', 'Laos': 'LAO',
                  'Diamond Princess': 'Others', 'West Bank and Gaza': 'TKM', 'MS Zaandam': 'Others',
                  'Taiwan':'TWN', 'Vietnam':'VNM', 'Russia':'RUS', 'Others':'Others',
                  'Iran':'IRN', 'Azerbaijan':'AZE', 'Czech Republic':'CZE',
                  'Saint Barthelemy':'BLM', 'Palestine':"PSE", 'Vatican City':'VAT', 
                  'Moldova':'MDA', 'Brunei':'BRN', 'Holy See':'VAT', 'Bolivia':'BOL', 
                  'Reunion':'REU', 'Venezuela':'VEN', 'Curacao':'CUW', 'Kosovo':'RKS',
                  'Republic of the Congo':'COG', 'Tanzania':'TZA', 'The Bahamas':'BHS', 
                  'The Gambia':'GMB', 'Syria':'SYR', ' Azerbaijan':'AZE'
                 }

    for x in range(len(ISO_Code)):
        if ISO_Code[x] in exceptions.keys():
            ISO_Code[x] = exceptions[ISO_Code[x]]
    
    data["ISO_Code"] = ISO_Code
    return data


def getTopCountries(cap):
    
    USA = data.loc[data['ISO_Code'] == 'USA']
    for x in range(1, (len(Top_Country_Codes)-(34-cap))):
        USA = USA.append(data.loc[data['ISO_Code'] == Top_Country_Codes[x]])
    
    return USA


# In[22]:



def plotMap(dataset, isPred):
    if isPred:
        text = "Corona Virus Prediction"
    else:
        text = 'Corona Virus Spread in The World'

    df_plot = dataset.groupby('ISO_Code').max().reset_index()
    fig = px.choropleth(df_plot, locations="ISO_Code",
                        color="Confirmed",
                        hover_data=["Confirmed", "Deaths", "Recovered"],
                        color_continuous_scale="Viridis")
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, title_text = text)
    return fig


def plotTopCountries(cap):
    topCountries = getTopCountries(cap)
    put_text = "Prediction for Top {} Countries".format(cap)
    fig = px.line(topCountries, x="ObservationDate", y="Confirmed", color='ISO_Code', hover_data=['Recovered', 'Deaths'])
    annotations = []
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                                  xanchor='left', yanchor='bottom',
                                  text=put_text,
                                  font=dict(family='Arial', size=30,
                                color='rgb(37,37,37)'), showarrow=False))

    fig.update_layout(annotations=annotations)
    fig.show()


# In[23]:



def predictTopCountries(cap):
    predictionsForAllCountries = []

    for country in range(len(Top_Country_Codes)-(34-cap)):

        country_data = data[data['ISO_Code']==Top_Country_Codes[country]]

        country_confirmed = readData(country_data, 'Confirmed')
        country_recovered = readData(country_data, 'Recovered')
        country_deaths = readData(country_data, 'Deaths')

        model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        country_confirmed_forecast = trainModel(country_confirmed, model)

        country_confirmed_forecast = country_confirmed_forecast[["ds","yhat"]]
        country_confirmed_forecast = country_confirmed_forecast.rename(columns = {'yhat':'Confirmed'})

        model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        country_recovered_forecast = trainModel(country_recovered, model)

        country_recovered_forecast = country_recovered_forecast[["ds","yhat"]]
        country_recovered_forecast = country_recovered_forecast.rename(columns = {'yhat':'Recovered'})

        model = Prophet(yearly_seasonality=True, daily_seasonality=True)
        country_deaths_forecast = trainModel(country_deaths, model)

        country_deaths_forecast = country_deaths_forecast[["ds","yhat"]]
        country_deaths_forecast = country_deaths_forecast.rename(columns = {'yhat':'Deaths'})

        predictedDataset = []
        predictedDataset.append(country_confirmed_forecast)
        predictedDataset.append(country_deaths_forecast)
        predictedDataset.append(country_recovered_forecast)

        predictedDataset = reduce(lambda left,right: pd.merge(left,right,on='ds'), predictedDataset)
        predictedDataset["ISO_Code"] = Top_Country_Codes[country]
        predictedDataset["Country/Region"] = Top_Country_Names[country]

        predictionsForAllCountries.append(predictedDataset)

    return predictionsForAllCountries


# In[24]:



Top_Country_Codes = ['USA','ESP','RUS','GBR','ITA','BRA',
                   'FRA','DEU','TUR','IRN','CHN', 'IND', 
                   'PER', 'CAN', 'BEL', 'SAU', 'NLD', 'MEX', 
                   'CHL', 'PAK', 'ECU', 'CHE', 'SWE', 'PRT', 
                   'QAT', 'BLR', 'SGP', 'IRL', 'ARE', 'BGD',
                   'POL', 'UKR', 'JPN']

Top_Country_Names = ['United States','Spain','Russia', 'UK', 'Italy',
                 'Brazil','France','Germany','Turkey', 'Iran', 'China',
                 'India','Peru','Canada','Belgium','Saudi Arabia',
                 'Netherlands','Mexico','Chile','Pakistan','Ecuador',
                 'Switzerland','Sweden','Portugal','Qatar','Belarus',
                 'Singapore','Ireland','United Arab Emirates',
                 'Bangladesh','Poland','Ukraine','Japan']


# In[25]:



data = pd.read_csv("./novel-corona-virus-2019-dataset/covid_19_data.csv")

today = data['ObservationDate'].unique().max()
days = getDaysTillToday(today)


# In[26]:



country_map = getCountryMap()
data.replace({'Country/Region': 'Mainland China'}, 'China', inplace=True)
data.replace({'Country/Region': 'US'}, 'USA', inplace=True)
data.replace({'Country/Region': 'Burma'}, 'Myanmar', inplace=True)

data = addCountryCode(data)

data.head()


# In[27]:



fig = plotMap(data, False)
fig.show()


# In[11]:



prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)

confirmed_cases = readData(data, 'Confirmed')
confirmed_forecast = trainModel(confirmed_cases, prophet)
fig_confirmed = prophet.plot(confirmed_forecast, xlabel="Date", ylabel="Number of Confirmed cases worldwide")


# In[28]:



prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)

death_cases = readData(data, 'Deaths')
death_forecast = trainModel(death_cases, prophet)
fig_death = prophet.plot(death_forecast, xlabel="Date", ylabel="Number of deaths worldwide")


# In[29]:



prophet = Prophet(yearly_seasonality=True, daily_seasonality=True)

recovered_cases = readData(data, 'Recovered')
recovered_forecast = trainModel(recovered_cases, prophet)
fig_recovered = prophet.plot(recovered_forecast, xlabel="Date", ylabel="Number of recoveries worldwide")


# In[14]:



predictionsForTopCountries = predictTopCountries(5)

topCountries = pd.concat(predictionsForTopCountries)

fig = plotMap(topCountries, True)
fig.show()


# In[16]:


plotTopCountries(5)


# In[ ]:




