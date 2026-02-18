import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Universal Time Series Analyzer", layout="wide")

st.title("📊 Universal Stock / Time Series Analysis App")

uploaded_file = st.file_uploader("Upload Any CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.subheader("📂 Dataset Preview")
    st.write(df.head())

    # ------------------ AUTO DETECT DATE COLUMN ------------------
    date_column = None
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            date_column = col
            break
        except:
            continue

    if date_column is None:
        st.error("No date column detected. Please upload a time-series dataset.")
    else:
        st.success(f"Date column detected: {date_column}")

        df = df.sort_values(date_column)

        # ------------------ SELECT NUMERIC COLUMN ------------------
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_columns) == 0:
            st.error("No numeric column found for analysis.")
        else:
            selected_column = st.selectbox(
                "Select Numeric Column for Analysis",
                numeric_columns
            )

            # ================= TREND =================
            st.subheader("📈 Trend Analysis")

            fig1, ax1 = plt.subplots()
            ax1.plot(df[date_column], df[selected_column])
            ax1.set_xlabel("Date")
            ax1.set_ylabel(selected_column)
            st.pyplot(fig1)

            # ================= ROLLING MEAN =================
            st.subheader("📊 Rolling Mean (Smoothing Trend)")
            df['RollingMean'] = df[selected_column].rolling(30).mean()

            fig2, ax2 = plt.subplots()
            ax2.plot(df[date_column], df[selected_column], label="Original")
            ax2.plot(df[date_column], df['RollingMean'], label="Rolling Mean")
            ax2.legend()
            st.pyplot(fig2)

            # ================= REGRESSION =================
            st.subheader("📉 Linear Regression Prediction")

            df['date_ordinal'] = df[date_column].map(pd.Timestamp.toordinal)

            model = LinearRegression()
            model.fit(df[['date_ordinal']], df[selected_column])

            df['Predicted'] = model.predict(df[['date_ordinal']])

            fig3, ax3 = plt.subplots()
            ax3.plot(df[date_column], df[selected_column], label="Actual")
            ax3.plot(df[date_column], df['Predicted'], label="Predicted")
            ax3.legend()
            st.pyplot(fig3)

            # ================= ARIMA =================
            st.subheader("🔮 ARIMA Forecast")

            try:
                model_arima = ARIMA(df[selected_column], order=(1,1,1))
                model_fit = model_arima.fit()
                forecast = model_fit.forecast(steps=30)

                fig4, ax4 = plt.subplots()
                ax4.plot(df[selected_column], label="Historical")
                ax4.plot(
                    range(len(df), len(df)+30),
                    forecast,
                    label="Forecast"
                )
                ax4.legend()
                st.pyplot(fig4)

            except:
                st.warning("ARIMA model could not be applied to this dataset.")

            st.success("Analysis Completed Successfully!")
