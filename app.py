import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ---------------------------------

# PAGE CONFIG

# ---------------------------------

st.set_page_config(
page_title="PredictX",
page_icon="⚙️",
layout="wide"
)

# ---------------------------------

# LOAD DATA

# ---------------------------------

@st.cache_data
def load_data():
return pd.read_csv("fleet_data.csv")

fleet = load_data()

# ---------------------------------

# LOAD MODEL

# ---------------------------------

@st.cache_resource
def load_model():
return joblib.load("predictx_best_model.pkl")

model = load_model()

# ---------------------------------

# SIDEBAR

# ---------------------------------

page = st.sidebar.selectbox(
"Navigate",
[
"Dashboard",
"Fleet Intelligence",
"Maintenance Center"
]
)

# ---------------------------------

# DASHBOARD

# ---------------------------------

if page == "Dashboard":

```
st.title("⚙️ PredictX")
st.subheader(
    "AI-Powered Predictive Maintenance Platform"
)

total_assets = len(fleet)

critical_assets = (
    fleet["status"] == "Critical"
).sum()

avg_risk = (
    fleet["failure_probability"].mean()
    * 100
)

avg_health = (
    fleet["health_score"].mean()
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Assets",
    total_assets
)

c2.metric(
    "Critical Assets",
    critical_assets
)

c3.metric(
    "Average Risk %",
    f"{avg_risk:.2f}"
)

c4.metric(
    "Average Health",
    f"{avg_health:.1f}"
)

st.markdown("---")

fig = px.histogram(
    fleet,
    x="failure_probability",
    color="status",
    title="Fleet Risk Distribution"
)

st.plotly_chart(
    fig,
    use_container_width=True
)
```

# ---------------------------------

# FLEET INTELLIGENCE

# ---------------------------------

elif page == "Fleet Intelligence":

```
st.title("📊 Fleet Intelligence")

fig = px.scatter(
    fleet,
    x="tool_wear_min",
    y="torque_nm",
    color="status",
    size="failure_probability",
    hover_data=["Machine_ID"]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader(
    "Top 20 High Risk Assets"
)

top20 = (
    fleet
    .sort_values(
        "failure_probability",
        ascending=False
    )
    .head(20)
)

st.dataframe(top20)
```

# ---------------------------------

# MAINTENANCE CENTER

# ---------------------------------

elif page == "Maintenance Center":

```
st.title("🔧 Maintenance Center")

maintenance_queue = (
    fleet
    .sort_values(
        "rul_days"
    )
)

st.dataframe(
    maintenance_queue[
        [
            "Machine_ID",
            "failure_probability",
            "rul_days",
            "maintenance_priority",
            "recommended_maintenance_date"
        ]
    ]
)
```

