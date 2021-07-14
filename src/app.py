import streamlit as st
import numpy as np
import model as m
import pandas as pd

st.set_page_config(
     page_title="Time Series Temperature Forecast",
     page_icon="📈",
     layout="wide",
   initial_sidebar_state="expanded",
 )

st.title('📈 Temperature forecast using Time Series Analysis')
first, second = st.beta_columns(2)

minimumTemperaturedf = first.file_uploader('Import the Minimum Temperature csv file here.', type='csv')
maximumTemperaturedf = second.file_uploader('Import the Maximum Temperature csv file here.', type='csv')

# Asserting the temperature files are imported properly

if minimumTemperaturedf is None:
    first.write('Please upload the minimum temperature csv file.')
if maximumTemperaturedf is None:
    second.write('Please upload the maximum temperature csv file.')

# Modify the csv data to time series data
time_stepMinimum=[]
tempsMinimum=[]
time_stepMaximum=[]
tempsMaximum=[]

if minimumTemperaturedf is not None and maximumTemperaturedf is not None:

    minimumTemperatureData = pd.read_csv(minimumTemperaturedf)
    maximumTemperatureData = pd.read_csv(maximumTemperaturedf)

    steps = 0
    for row in minimumTemperatureData.itertuples():
        tempsMinimum.append(float(row[2]))
        time_stepMinimum.append(steps)
        steps+=1

    steps = 0
    for row in maximumTemperatureData.itertuples():
        tempsMaximum.append(float(row[2]))
        time_stepMaximum.append(steps)
        steps+=1

    #  Visualising the Data
    first.write("Minimum Temperature Data")
    first.write(minimumTemperatureData)

    second.write("Maximum Temperature Data")
    second.write(maximumTemperatureData)

    seriesMinimum=np.array(tempsMinimum)
    timeMinimum=np.array(time_stepMinimum)
    first.line_chart(seriesMinimum)

    seriesMaximum=np.array(tempsMaximum)
    timeMaximum=np.array(time_stepMaximum)
    second.line_chart(seriesMaximum)

    c1, c2 = st.beta_columns((4, 1))

    days = c1.slider('Select the number of days for the forecast', 0,365)
    isForecast = c2.button("Forecast :smile:")

    if isForecast:
        accuracyMinimum, maeMinimum, minimumTemperature = m.forecast_model(seriesMinimum,timeMinimum,days)
        accuracyMaximum, maeMaximum, maximumTemperature = m.forecast_model(seriesMaximum,timeMaximum,days)

        first.write("Minimum Temperature at Day "+str(days) + " : " +str(minimumTemperature[days-1]))
        second.write("Maximum Temperature at Day "+str(days)+ " : " +str(maximumTemperature[days-1]))


st.markdown("""
<style>
.big-font {
    font-size:15px !important;
    text-align: center;
    color: purple;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Developed by Sai Ankit & Sarveshwar Mahapatro</p>', unsafe_allow_html=True)
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
