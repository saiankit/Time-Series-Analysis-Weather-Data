import streamlit as st
import time
import numpy as np
import os
import tensorflow as tf
import csv
import model as m
import pandas as pd
import time



st.title('📈 Temperature forecast using Time Series Analysis')
first, second = st.beta_columns(2)

dfMinimum = first.file_uploader('Import the Minimum Temperature csv file here.', type='csv')
dfMaximum = second.file_uploader('Import the Maximum Temperature csv file here.', type='csv')

time_stepMinimum=[]
tempsMinimum=[]
time_stepMaximum=[]
tempsMaximum=[]

if dfMinimum is not None and dfMaximum is not None:
    dataMinimum = pd.read_csv(dfMinimum)
    dataMaximum = pd.read_csv(dfMaximum)
    stepMinimum = 0
    stepMaximum = 0
    for row in dataMinimum.itertuples():
        tempsMinimum.append(float(row[2]))
        time_stepMinimum.append(stepMinimum)
        stepMinimum+=1
    for row in dataMaximum.itertuples():
        tempsMaximum.append(float(row[2]))
        time_stepMaximum.append(stepMaximum)
        stepMaximum+=1
    first.write("Minimum Temperature Data")
    first.write(dataMinimum)
    second.write("Maximum Temperature Data")
    second.write(dataMaximum)
    days = st.slider('Select the number of days for the forecast', 0,365)
    seriesMinimum=np.array(tempsMinimum)
    timeMinimum=np.array(time_stepMinimum)
    st.line_chart(seriesMinimum)
    seriesMaximum=np.array(tempsMaximum)
    timeMaximum=np.array(time_stepMaximum)
    st.line_chart(seriesMaximum)

    accuracyMinimum, maeMinimum, minimumTemperature = m.forecast_model(seriesMinimum,timeMinimum,days)
    accuracyMaximum, maeMaximum, maximumTemperature = m.forecast_model(seriesMaximum,timeMaximum,days)
    my_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.write("The MAEMinimum is :")
    st.write(maeMinimum)

    accuracyMinimum=100-maeMinimum
    st.write("The accuracyMinimum is :")
    st.write(accuracyMinimum)
    first.write("Minimum Temperature is:",minimumTemperature)



    st.write("The MAEMaximum is :")
    st.write(maeMaximum)

    accuracyMaximum=100-maeMaximum
    st.write("The accuracyMaximum is :")
    st.write(accuracyMaximum)
    second.write("Minimum Temperature is:",minimumTemperature)


hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.button("Re-run")