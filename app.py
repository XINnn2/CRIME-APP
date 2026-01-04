import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import lightgbm

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Crime Analytics & Prediction System",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Model & File Path Configuration (Root Folder)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crime_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
DEFAULT_PATH = os.path.join(BASE_DIR, "defaults.pkl")

# --------------------------------------------------
# Utility: Load Model / Encoders / Defaults
# --------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model file not found at:\n{MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_encoders():
    if not os.path.exists(ENCODER_PATH):
        st.error("⚠️ Label encoders file not found.")
        st.stop()
    return joblib.load(ENCODER_PATH)

@st.cache_resource
def load_defaults():
    if not os.path.exists(DEFAULT_PATH):
        st.error("⚠️ defaults.pkl not found.")
        st.stop()
    return joblib.load(DEFAULT_PATH)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
selected_page = st.sidebar.selectbox(
    "Select a Page",
    ["🏠 Home", "📊 Dashboard", "🔮 Prediction", "🔔 Alerts & Feedback"]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if selected_page == "🏠 Home":
    st.title("🚨 Crime Analytics & Prediction System")

    st.image("crime.jpg", use_container_width=True)

    st.markdown("""
    \nWelcome to the **Crime Analytics Platform**.
    """)

    st.markdown("""
    This platform helps you to explore historical crime data from 2016 to 2022 and visualize crime trends across different states and categories.

    It also provides **crime prediction** for future years using machine learning models, helping to take relevant prevention plans and respond effectively.
    """)

    st.info("👈 Use the sidebar to navigate between **📊 Dashboard**, **🔮 Prediction**, and **🔔 Alerts & Feedback** sections.")

    col1, col2, col3 = st.columns(3)
    col1.metric("States Covered", "14")
    col2.metric("ML Model", "LightGBM")
    col3.metric("Latest Data Year", "2025")

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------
elif selected_page == "📊 Dashboard":
    st.header("📊 Crime Distribution Dashboard")

    st.markdown("""
    Interactive dashboard showing crime distribution by **state** and **category** from year 2016 to 2022. 
    \nYou can click on any **state**, **category** or **year** to view the details of crime distribution.
    """)

    st.markdown("""
    <iframe width="100%" height="500"
    src="https://app.powerbi.com/view?r=eyJrIjoiZjM2MTBkNTEtMGNiNy00NGZmLTllNWUtYWI2OTM3MDkyODNiIiwidCI6IjdmMDQ4ZmMxLTJlYTMtNDhlNC1hYzkyLTkxZDFlYjA5ODA3YyIsImMiOjEwfQ%3D%3D"
    frameborder="0" allowFullScreen="true"></iframe>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif selected_page == "🔮 Prediction":
    st.header("🔮 Crime Prediction")

    # ===============================
    # Load trained model, encoders, defaults
    # ===============================
    model = load_model()
    encoders = load_encoders()
    DEFAULTS = load_defaults()

    # ===============================
    # UI Inputs
    # ===============================
    st.markdown(
        "Select a **state**, **crime category**, and **target year**. "
        "The system will display the **top 5 predicted crime types**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        state = st.selectbox("State", encoders["state"].classes_)
    with col2:
        category = st.selectbox("Crime Category", encoders["category"].classes_)
    with col3:
        year = st.slider("Target Year", 2026, 2029, 2027)

    # ===============================
    # Prediction
    # ===============================
    if st.button("Predict Crime Trends", use_container_width=True):
        results = []
        crime_types = encoders["type"].classes_

        for crime_type in crime_types:
            # ---- Base input ----
            input_df = pd.DataFrame({
                "state": [state],
                "category": [category],
                "type": [crime_type],
                "year": [year]
            })
            # ---- Add defaults ----
            for col, val in DEFAULTS.items():
                input_df[col] = val
            # ---- Encode ----
            input_df["state"] = encoders["state"].transform(input_df["state"])
            input_df["category"] = encoders["category"].transform(input_df["category"])
            input_df["type"] = encoders["type"].transform(input_df["type"])
            # ---- Align columns ----
            input_df = input_df[model.feature_name_]
            # ---- Predict ----
            pred = model.predict(input_df)[0]
            results.append((crime_type, pred))

        # ===============================
        # Display Top 5
        # ===============================
        top5 = sorted(results, key=lambda x: x[1], reverse=True)[:5]

        st.subheader("🚨 Top 5 Predicted Crime Types")

        high_risk_detected = False
        moderate_risk_detected = False

        for i, (crime, score) in enumerate(top5, start=1):
            if score > 150:
                risk = "🔴 High Risk"
                high_risk_detected = True
            elif score > 80:
                risk = "🟠 Moderate Risk"
                moderate_risk_detected = True
            else:
                risk = "🟢 Low Risk"
            st.markdown(f"**{i}. {crime.replace('_', ' ').title()}** — {risk}")

        # --------------------------------------------------
        # Priority-based Alert Message
        # --------------------------------------------------
        if high_risk_detected:
            st.warning(
                "🚨 Breaking Alert: High-risk crime detected. Immediate attention recommended."
            )
        elif moderate_risk_detected:
            st.warning(
                "🟠 Alert: Moderate-risk crime detected. Recommended actions include increased situational awareness, targeted patrols, and enhanced monitoring."
            )
        else:
            st.success("🟢 Situation Stable: No moderate or high-risk crime detected.")

        st.markdown("---")
        st.subheader("📌 Recommended Actions")
        st.markdown("""
        - 🚓 Conduct random spot checks in areas with past incidents
        - 🏘️ Strengthen community-based prevention programmes
        - 🧑‍🤝‍🧑 Run awareness campaigns on personal safety and conflict de-escalation 
        - 📊 Continuously monitor crime trends via the dashboard  
        """)

# --------------------------------------------------
# ALERTS & FEEDBACK PAGE
# --------------------------------------------------
elif selected_page == "🔔 Alerts & Feedback":
    st.header("🔔 Alerts, Notes & Feedback")

    st.markdown("""
    **Operational reminders:**
    - 🚓 Increase patrols during predicted high-risk periods
    - 🕒 Focus on property crime prevention
    - 🏘️ Community engagement improves prevention outcomes
    """)

    with st.form("feedback_form"):
        feedback_type = st.radio(
            "Feedback Type",
            ["Bug Report", "Prediction Feedback", "General Suggestion"]
        )
        feedback_text = st.text_area("Your feedback")
        submitted = st.form_submit_button("Submit")

        if submitted:
            st.success("✅ Thank you for your feedback!")
