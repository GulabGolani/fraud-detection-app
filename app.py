import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import datetime

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Fraud Detection", layout="wide")

# -----------------------------
# CUSTOM CSS (🔥 PREMIUM UI)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.card {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.4);
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
.label {
    color: #9aa4b2;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=[
        "Amount", "Location", "Device", "Hour", "Risk Score", "Status"
    ])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("💳 Fraud AI System")
page = st.sidebar.radio("Navigate", ["Live Detection", "Dashboard", "Admin"])

# -----------------------------
# MODEL
# -----------------------------
def train_model():
    np.random.seed(42)
    data = np.random.rand(300, 2)
    model = IsolationForest(contamination=0.1)
    model.fit(data)
    return model

model = train_model()

# -----------------------------
# RISK ENGINE
# -----------------------------
def calculate_risk(amount, location, device, hour):
    risk = 0
    reasons = []

    if amount > 50000:
        risk += 30
        reasons.append("High amount")

    if location == "International":
        risk += 25
        reasons.append("Unusual location")

    if device == "Unknown":
        risk += 20
        reasons.append("Unknown device")

    if hour < 6 or hour > 23:
        risk += 15
        reasons.append("Odd time")

    ml_input = np.array([[amount / 100000, hour / 24]])
    ml_score = -model.decision_function(ml_input)[0] * 100

    total = int(risk + ml_score)

    if total > 70:
        status = "Fraud"
    elif total > 40:
        status = "Suspicious"
    else:
        status = "Safe"

    return total, status, reasons

# -----------------------------
# PAGE 1: LIVE DETECTION
# -----------------------------
if page == "Live Detection":
    st.title("🔍 Real-Time Fraud Detection")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        amount = st.number_input("Amount (₹)", min_value=0)

    with c2:
        location = st.selectbox("Location", ["Local", "National", "International"])

    with c3:
        device = st.selectbox("Device", ["Mobile", "Laptop", "Unknown"])

    with c4:
        hour = st.slider("Hour", 0, 23, datetime.datetime.now().hour)

    if st.button("Analyze Transaction"):
        risk, status, reasons = calculate_risk(amount, location, device, hour)

        colA, colB = st.columns([1,1])

        # RESULT CARD
        with colA:
            st.markdown(f"""
            <div class="card">
                <div class="label">Risk Score</div>
                <div class="metric">{risk}</div>
                <div class="label">Status: {status}</div>
            </div>
            """, unsafe_allow_html=True)

        # GAUGE
        with colB:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"},
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

        st.write("### 🧠 Why flagged?")
        for r in reasons:
            st.write(f"- {r}")

        new_row = pd.DataFrame([[amount, location, device, hour, risk, status]],
                               columns=st.session_state.data.columns)
        st.session_state.data = pd.concat([st.session_state.data, new_row], ignore_index=True)

# -----------------------------
# PAGE 2: DASHBOARD
# -----------------------------
elif page == "Dashboard":
    st.title("📊 Analytics Dashboard")

    df = st.session_state.data

    if df.empty:
        st.warning("Run transactions first.")
    else:
        col1, col2, col3 = st.columns(3)

        # KPI CARDS
        with col1:
            st.markdown(f"""<div class="card">
            <div class="label">Total Transactions</div>
            <div class="metric">{len(df)}</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            fraud_rate = round((df["Status"] == "Fraud").mean() * 100, 2)
            st.markdown(f"""<div class="card">
            <div class="label">Fraud %</div>
            <div class="metric">{fraud_rate}%</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            avg_risk = int(df["Risk Score"].mean())
            st.markdown(f"""<div class="card">
            <div class="label">Avg Risk</div>
            <div class="metric">{avg_risk}</div>
            </div>""", unsafe_allow_html=True)

        # CHARTS
        c1, c2 = st.columns(2)

        with c1:
            fig1 = px.histogram(df, x="Risk Score", color="Status")
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            fig2 = px.pie(df, names="Status")
            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(df, y="Risk Score", title="Risk Trend")
        st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# PAGE 3: ADMIN
# -----------------------------
elif page == "Admin":
    st.title("🛠️ Admin Panel")

    df = st.session_state.data

    if df.empty:
        st.warning("No data yet.")
    else:
        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "transactions.csv"
        )