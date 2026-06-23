import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap


import shap
import matplotlib.pyplot as plt

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="PredictX",
    page_icon="⚙️",
    layout="wide"
)
st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #374151;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
}

div[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: bold;
}

h1,h2,h3 {
    color: #f9fafb;
}

</style>
""", unsafe_allow_html=True)
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
def create_features(
    air_temp,
    process_temp,
    rpm,
    torque,
    tool_wear,
    machine_type
):

    temp_difference = process_temp - air_temp

    power_index = torque * rpm

    wear_torque_interaction = tool_wear * torque

    wear_speed_ratio = tool_wear / rpm

    health_score = (
        100
        - (tool_wear / 250) * 50
        - (torque / 80) * 50
    )

    thermal_stress = temp_difference * torque

    mechanical_stress = torque * tool_wear

    return pd.DataFrame({
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

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.markdown("""
# ⚙️ PredictX

### AI Maintenance Platform

Built with XGBoost • SHAP • Streamlit
""")

page = st.sidebar.radio(
    "Navigation",
    [
        
        "Executive Dashboard",
        "Failure Prediction",
        "Digital Twin",
        "What-If Analysis",
        "SHAP Explainability",
        "Fleet Intelligence",
        "Maintenance Center",
        "Executive Analytics",
        "Download Center",
        "Model Benchmarking"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================

if page == "Executive Dashboard":

    st.markdown("""
    <div style='padding:10px 0px 20px 0px;'>

    <h1 style='font-size:48px;margin-bottom:0px;'>
    👋 Welcome Back
    </h1>

    <p style='font-size:22px;color:#94a3b8;'>
    Here's what's happening with your fleet today.
    </p>

    </div>
    """, unsafe_allow_html=True)

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

    

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#4f46e5,#7c3aed);
        padding:20px;
        border-radius:20px;
        color:white;
        text-align:center;">
        <h4>📦 Total Assets</h4>
        <h1>{total_assets}</h1>
        </div>
    """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#dc2626,#ef4444);
        padding:20px;
        border-radius:20px;
        color:white;
        text-align:center;">
        <h4>🚨 Critical Assets</h4>
        <h1>{critical_assets}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#f59e0b,#f97316);
        padding:20px;
        border-radius:20px;
        color:white;
        text-align:center;">
        <h4>📈 Average Risk %</h4>
        <h1>{avg_risk:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#059669,#10b981);
        padding:20px;
        border-radius:20px;
        color:white;
        text-align:center;">
        <h4>💚 Health Score</h4>
        <h1>{avg_health:.1f}</h1>
        </div>
        """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div style="
    background:linear-gradient(135deg,#059669,#10b981);
    padding:20px;
    border-radius:20px;
    color:white;
    text-align:center;">
    <h4>💚 Health Score</h4>
    <h1>{avg_health:.1f}</h1>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 📊 Fleet Overview")

col1, col2 = st.columns(2)
        

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
# DIGITAL TWIN
# ==================================================

elif page == "Digital Twin":

    st.title("🛰️ Digital Twin Simulator")

    col1, col2 = st.columns(2)

    with col1:

        air_temp = st.slider(
            "Air Temperature",
            295,305,300,
            key="dt_air"
        )

        process_temp = st.slider(
            "Process Temperature",
            305,315,310,
            key="dt_process"
        )

        rpm = st.slider(
            "RPM",
            1100,1800,1500,
            key="dt_rpm"
        )

    with col2:

        torque = st.slider(
            "Torque",
            10,80,40,
            key="dt_torque"
        )

        tool_wear = st.slider(
            "Tool Wear",
            0,250,100,
            key="dt_wear"
        )

        machine_type = st.selectbox(
            "Machine Type",
            [0,1,2],
            key="dt_type"
        )

    sample = create_features(
        air_temp,
        process_temp,
        rpm,
        torque,
        tool_wear,
        machine_type
    )

    risk = model.predict_proba(sample)[0][1]

    health = max(
        0,
        100 - (risk * 100)
    )

    c1,c2 = st.columns(2)

    with c1:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk*100,
                title={"text":"Failure Risk %"},
                gauge={
                    "axis":{
                        "range":[0,100]
                    }
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with c2:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health,
                title={"text":"Health Score"},
                gauge={
                    "axis":{
                        "range":[0,100]
                    }
                }
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader(
        "Maintenance Recommendations"
    )

    if tool_wear > 180:
        st.warning(
            "Replace Tool Immediately"
        )

    if torque > 60:
        st.warning(
            "Inspect Motor Assembly"
        )

    if risk > 0.80:
        st.error(
            "Immediate Maintenance Required"
        )

# ==================================================
# WHAT IF ANALYSIS
# ==================================================

elif page == "What-If Analysis":

    st.title("🔬 What-If Analysis")

    left,right = st.columns(2)

    with left:

        st.subheader(
            "Current Machine"
        )

        current_torque = st.slider(
            "Current Torque",
            10,80,40
        )

        current_wear = st.slider(
            "Current Wear",
            0,250,100
        )

    with right:

        st.subheader(
            "Future Scenario"
        )

        future_torque = st.slider(
            "Future Torque",
            10,80,60
        )

        future_wear = st.slider(
            "Future Wear",
            0,250,180
        )

    current_risk = min(
        1,
        current_torque/100 +
        current_wear/400
    )

    future_risk = min(
        1,
        future_torque/100 +
        future_wear/400
    )

    comparison = pd.DataFrame({
        "Scenario":[
            "Current",
            "Future"
        ],
        "Risk":[
            current_risk*100,
            future_risk*100
        ]
    })

    fig = px.bar(
        comparison,
        x="Scenario",
        y="Risk",
        color="Scenario",
        title="Risk Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.metric(
        "Risk Change",
        f"{(future_risk-current_risk)*100:.2f}%"
    )

# ==================================================
# SHAP EXPLAINABILITY
# ==================================================

elif page == "SHAP Explainability":

    st.title("🧠 Explainable AI Dashboard")

    st.markdown(
        """
        Understand why the XGBoost model predicts
        machine failures.
        """
    )

    feature_cols = [
        'air_temperature_k',
        'process_temperature_k',
        'rotational_speed_rpm',
        'torque_nm',
        'tool_wear_min',
        'temp_difference',
        'power_index',
        'wear_torque_interaction',
        'wear_speed_ratio',
        'health_score',
        'thermal_stress',
        'mechanical_stress',
        'type_encoded'
    ]

    sample_data = fleet.head(200).copy()

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        sample_data[feature_cols]
    )

    st.subheader(
        "Global Feature Importance"
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    shap.summary_plot(
        shap_values,
        sample_data[feature_cols],
        plot_type="bar",
        show=False
    )

    st.pyplot(fig)

    st.markdown("---")

    st.subheader(
        "Machine Level Explanation"
    )

    selected_machine = st.selectbox(
        "Select Machine",
        sample_data["Machine_ID"]
    )

    machine_row = sample_data[
        sample_data["Machine_ID"]
        ==
        selected_machine
    ]

    machine_features = (
        machine_row[feature_cols]
    )

    local_shap = explainer.shap_values(
        machine_features
    )

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Impact": np.abs(
            local_shap[0]
        )
    })

    importance = (
        importance
        .sort_values(
            "Impact",
            ascending=False
        )
        .head(10)
    )

    fig = px.bar(
        importance,
        x="Impact",
        y="Feature",
        orientation="h",
        title="Top Drivers of Failure Risk"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    prob = model.predict_proba(
        machine_features
    )[0][1]

    st.metric(
        "Failure Probability",
        f"{prob:.2%}"
    )

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

elif page == "Executive Analytics":

    st.title("📈 Executive Analytics")

    avg_risk = (
        fleet["failure_probability"]
        .mean() * 100
    )

    critical_assets = (
        fleet["status"] == "Critical"
    ).sum()

    avg_rul = (
        fleet["rul_days"]
        .mean()
    )

    estimated_downtime = (
        critical_assets * 2.5
    )

    estimated_cost = (
        critical_assets * 25000
    )

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Avg Fleet Risk",
        f"{avg_risk:.2f}%"
    )

    c2.metric(
        "Critical Assets",
        critical_assets
    )

    c3.metric(
        "Predicted Downtime",
        f"{estimated_downtime:.1f} hrs"
    )

    c4.metric(
        "Maintenance Cost",
        f"₹{estimated_cost:,.0f}"
    )
    st.markdown("---")

    trend = (
        fleet
        .sort_values(
            "failure_probability"
        )
        .reset_index()
    )

    fig = px.line(
        trend,
        y="failure_probability",
        title="Fleet Risk Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("🚨 Top 10 Highest Risk Assets")

    top10 = (
        fleet
        .sort_values(
            "failure_probability",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        top10[
            [
                "Machine_ID",
                "failure_probability",
                "rul_days",
                "maintenance_priority"
            ]
        ],
        use_container_width=True
    )

elif page == "Download Center":

    st.title("📥 Download Center")

    csv = fleet.to_csv(
        index=False
    )

    st.download_button(
    label="Download Fleet Report",
    data=csv,
    file_name="fleet_report.csv",
    mime="text/csv"
    )

    high_risk = fleet[
        fleet["failure_probability"] > 0.7
    ]

    csv2 = high_risk.to_csv(
        index=False
    )

    st.download_button(
        label="Download High Risk Assets",
        data=csv2,
        file_name="high_risk_assets.csv",
        mime="text/csv"
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

st.markdown("---")

st.markdown("""
<center>

### ⚡ PredictX Enterprise Edition

AI-Powered Predictive Maintenance & Asset Intelligence Platform

Built using XGBoost • SHAP • Streamlit • Plotly

</center>
""", unsafe_allow_html=True)
