import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.arima.model import ARIMA

st.title("Stock Market Trend Analysis")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    st.write(df.head())

    df['year'] = pd.to_datetime(df['year'], format='%Y')
    df = df.sort_values('year')

    fig, ax = plt.subplots()
    ax.plot(df['year'], df['value'])
    st.pyplot(fig)
