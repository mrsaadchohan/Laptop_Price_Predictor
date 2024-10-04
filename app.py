import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the pipeline and dataframe
pipe = pickle.load(open('pipess.pkl', 'rb'))
df = pickle.load(open('dfss.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screensize in inches', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', 
                                                '3200x1800', '2880x1800', '2560x1600', 
                                                '2560x1440', '2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu_brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['GPU_Company'].unique())

# OS
os = st.selectbox('OS', df['OS'].unique())

# Predict button
if st.button('Predict Price'):
    # Convert touchscreen and ips to binary
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    
    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create a query array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Convert to DataFrame with matching columns
    query_df = pd.DataFrame([query], columns=['Company', 'TypeName', 'RAM', 'Weight', 'Touch_Screen', 
                                              'IPS', 'ppi', 'Cpu_brand', 'HDD', 'SSD', 
                                              'GPU_Company', 'OS'])

    # Predict log price using the pipeline
    predicted_log_price = pipe.predict(query_df)

    # Apply exponential transformation to get the predicted price
    predicted_price = np.exp(predicted_log_price)
    predicted_price*=100
    
    # Multiply by a correction factor if necessary (adjust for scaling in the model)
    st.title(f"The predicted price of this configuration is {int(predicted_price)}")
