import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="PredictX",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ PredictX")

@st.cache_data
def load_data():
    return pd.read_csv("fleet_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("predictx_best_model.pkl")

fleet = load_data()

model = load_model()

st.success("Model Loaded Successfully")

st.write("Fleet Shape:")
st.write(fleet.shape)

st.dataframe(fleet.head())
