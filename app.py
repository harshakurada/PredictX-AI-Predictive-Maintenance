import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PredictX",
    page_icon="⚙️",
    layout="wide"
)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():
    return pd.read_csv("fleet_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("predictx_best_model.pkl")

fleet = load_data()
model = load_model()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ PredictX")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Failure Prediction",
        "Digital Twin",
        "What-If Analysis",
        "Fleet Intelligence",
        "Maintenance Center",
        "Model Benchmarking"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================

if page == "Executive Dashboard":

    st.title("⚙️ PredictX")
    st.subheader("AI-Powered Predictive Maintenance & Asset Intelligence Platform")

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
        f"{total_assets}"
    )

    c2.metric(
        "Critical Assets",
        f"{critical_assets}"
    )

    c3.metric(
        "Average Risk %",
        f"{avg_risk:.2f}"
    )

    c4.metric(
        "Health Score",
        f"{avg_health:.1f}"
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.histogram(
            fleet,
            x="failure_probability",
            color="status",
            nbins=30,
            title="Fleet Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        status_counts = (
            fleet["status"]
            .value_counts()
            .reset_index()
        )

        status_counts.columns = [
            "Status",
            "Count"
        ]

        fig = px.pie(
            status_counts,
            names="Status",
            values="Count",
            hole=0.4,
            title="Fleet Status Breakdown"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# FAILURE PREDICTION
# ==================================================

elif page == "Failure Prediction":

    st.title("🔮 Failure Prediction")

    col1, col2 = st.columns(2)

    with col1:

        air_temp = st.slider(
            "Air Temperature (K)",
            295,
            305,
            300
        )

        process_temp = st.slider(
            "Process Temperature (K)",
            305,
            315,
            310
        )

        rpm = st.slider(
            "Rotational Speed (RPM)",
            1100,
            1800,
            1500
        )

    with col2:

        torque = st.slider(
            "Torque (Nm)",
            10,
            80,
            40
        )

        tool_wear = st.slider(
            "Tool Wear (min)",
            0,
            250,
            100
        )

        machine_type = st.selectbox(
            "Machine Type",
            [0, 1, 2]
        )

    if st.button("Predict Failure Risk"):

        temp_difference = (
            process_temp - air_temp
        )

        power_index = (
            torque * rpm
        )

        wear_torque_interaction = (
            tool_wear * torque
        )

        wear_speed_ratio = (
            tool_wear / rpm
        )

        health_score = (
            100
            -
            (tool_wear / 250) * 50
            -
            (torque / 80) * 50
        )

        thermal_stress = (
            temp_difference * torque
        )

        mechanical_stress = (
            tool_wear * torque
        )

        input_df = pd.DataFrame({
            "air_temperature_k":[air_temp],
            "process_temperature_k":[process_temp],
            "rotational_speed_rpm":[rpm],
            "torque_nm":[torque],
            "tool_wear_min":[tool_wear],
            "temp_difference":[temp_difference],
            "power_index":[power_index],
            "wear_torque_interaction":[wear_torque_interaction],
            "wear_speed_ratio":[wear_speed_ratio],
            "health_score":[health_score],
            "thermal_stress":[thermal_stress],
            "mechanical_stress":[mechanical_stress],
            "type_encoded":[machine_type]
        })

        prob = model.predict_proba(input_df)[0][1]

        st.metric(
            "Failure Probability",
            f"{prob:.2%}"
        )

        if prob < 0.20:
            st.success("Healthy")

        elif prob < 0.50:
            st.warning("Warning")

        elif prob < 0.80:
            st.warning("High Risk")

        else:
            st.error("Critical")

# ==================================================
# FLEET INTELLIGENCE
# ==================================================

elif page == "Fleet Intelligence":

    st.title("📊 Fleet Intelligence")

    fig = px.scatter(
        fleet,
        x="tool_wear_min",
        y="torque_nm",
        color="status",
        size="failure_probability",
        hover_data=["Machine_ID"],
        title="Fleet Risk Map"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Top 20 High-Risk Assets"
    )

    top20 = (
        fleet
        .sort_values(
            "failure_probability",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top20,
        use_container_width=True
    )

# ==================================================
# MAINTENANCE CENTER
# ==================================================

elif page == "Maintenance Center":

    st.title("🔧 Maintenance Center")

    priority = st.selectbox(
        "Priority Filter",
        [
            "All",
            "Immediate",
            "Urgent",
            "Scheduled",
            "Normal"
        ]
    )

    maintenance = fleet.copy()

    if priority != "All":

        maintenance = (
            maintenance[
                maintenance[
                    "maintenance_priority"
                ] == priority
            ]
        )

    st.dataframe(
        maintenance[
            [
                "Machine_ID",
                "failure_probability",
                "rul_days",
                "maintenance_priority",
                "recommended_maintenance_date"
            ]
        ],
        use_container_width=True
    )

# ==================================================
# BENCHMARKING
# ==================================================

elif page == "Model Benchmarking":

    st.title("🏆 Model Benchmarking")

    benchmark = pd.DataFrame({
        "Model":[
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "LightGBM"
        ],
        "ROC_AUC":[
            0.9259,
            0.9803,
            0.9847,
            0.9791
        ]
    })

    fig = px.bar(
        benchmark,
        x="Model",
        y="ROC_AUC",
        color="Model",
        text="ROC_AUC",
        title="Model Performance Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        benchmark,
        use_container_width=True
    )
