import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="PredictX",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ PredictX")

@st.cache_data
def load_data():
    return pd.read_csv("fleet_data.csv")

fleet = load_data()

st.write("Application Loaded Successfully")

st.write("Fleet Shape:")
st.write(fleet.shape)

st.dataframe(fleet.head())
