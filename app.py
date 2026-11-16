import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Load trained LightGBM classification model
# Model paths
CLASSIFIER_PATH = "models/lightgbm_classifier.pkl"
REGRESSOR_PATH = "models/lightgbm_regressor.pkl"

@st.cache_resource
def load_classifier():
    return joblib.load(CLASSIFIER_PATH)

@st.cache_resource
def load_regressor():
    return joblib.load(REGRESSOR_PATH)

# Load both trained models
lightgbm_classifier = load_classifier()
lightgbm_regressor = load_regressor()
st.success("LightGBM classification model loaded successfully!")

# Page configuration
st.set_page_config(
    page_title="EV Charging Demand Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Dashboard title
st.title("⚡ EV Charging Demand Prediction")
st.markdown(
    "### Electric Vehicle Charging Usage Pattern Analysis"
)

# Load dataset
DATA_PATH = "dataset"

df = pd.read_csv(
    os.path.join(DATA_PATH, "ChargingRecords_Feature_Engineered.csv")
)

# Sidebar
st.sidebar.title("Dashboard Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Overview",
        "Charging Analysis",
        "Model Comparison",
        "Demand Prediction"
    ]
)

# Overview page
if page == "Overview":

    st.header("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sessions", f"{len(df):,}")
    col2.metric("Average Demand", f"{df['Demand'].mean():.2f}")
    col3.metric("Average Duration", f"{df['Duration'].mean():.2f} min")
    col4.metric("Total Locations", df["Location"].nunique())

    st.subheader("Charging Demand Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(df["Demand"], bins=30, kde=True, ax=ax)

    ax.set_xlabel("Demand")
    ax.set_ylabel("Number of Sessions")

    st.pyplot(fig)

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10))

# Charging Analysis page

elif page == "Charging Analysis":

    st.header("Interactive Charging Analysis")

    # Sidebar filters
    st.sidebar.subheader("Analysis Filters")

    locations = ["All"] + sorted(df["Location"].dropna().unique().tolist())
    selected_location = st.sidebar.selectbox("Location", locations)

    charger_types = ["All"] + sorted(df["ChargerType"].dropna().unique().tolist())
    selected_charger = st.sidebar.selectbox("Charger Type", charger_types)

    months = ["All"] + sorted(df["Month"].dropna().unique().tolist())
    selected_month = st.sidebar.selectbox("Month", months)

    time_periods = ["All"] + sorted(df["TimeOfDay"].dropna().unique().tolist())
    selected_time = st.sidebar.selectbox("Time of Day", time_periods)

    # Filter dataset
    filtered_df = df.copy()

    if selected_location != "All":
        filtered_df = filtered_df[
            filtered_df["Location"] == selected_location
        ]

    if selected_charger != "All":
        filtered_df = filtered_df[
            filtered_df["ChargerType"] == selected_charger
        ]

    if selected_month != "All":
        filtered_df = filtered_df[
            filtered_df["Month"] == selected_month
        ]

    if selected_time != "All":
        filtered_df = filtered_df[
            filtered_df["TimeOfDay"] == selected_time
        ]

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sessions", f"{len(filtered_df):,}")

    col2.metric(
        "Average Demand",
        f"{filtered_df['Demand'].mean():.2f}"
        if not filtered_df.empty else "N/A"
    )

    col3.metric(
        "Average Duration",
        f"{filtered_df['Duration'].mean():.2f} min"
        if not filtered_df.empty else "N/A"
    )

    if filtered_df.empty:
        st.warning("No records match the selected filters.")

    else:
        st.subheader("Average Demand by Location")
        st.bar_chart(
            filtered_df.groupby("Location")["Demand"].mean()
        )

        st.subheader("Average Demand by Hour")
        st.line_chart(
            filtered_df.groupby("Hour")["Demand"].mean()
        )

        st.subheader("Average Demand by Month")
        st.line_chart(
            filtered_df.groupby("Month")["Demand"].mean()
        )

        st.subheader("Filtered Dataset")
        st.dataframe(filtered_df.head(100))

# Model Comparison page
elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")

    regression_path = os.path.join(
        DATA_PATH, "regression_model_results.csv"
    )

    classification_path = os.path.join(
        DATA_PATH, "classification_model_results.csv"
    )

    if os.path.exists(regression_path):
        st.subheader("Regression Results")
        st.dataframe(pd.read_csv(regression_path))

    if os.path.exists(classification_path):
        st.subheader("Classification Results")
        st.dataframe(pd.read_csv(classification_path))

    feature_path = os.path.join(
        DATA_PATH, "feature_importance_results.csv"
    )

    if os.path.exists(feature_path):
        st.subheader("Feature Importance")
        importance = pd.read_csv(feature_path).head(15)
        st.bar_chart(importance.set_index("Feature")["Importance"])

# Demand Prediction page
elif page == "Demand Prediction":

    st.header("EV Charging Demand Prediction")

    st.write("Enter charging session details to predict demand.")

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox(
            "Select Location",
            sorted(df["Location"].dropna().unique())
        )

        charger_type = st.selectbox(
            "Charger Type",
            sorted(df["ChargerType"].dropna().unique())
        )

        charger_company = st.selectbox(
            "Charger Company",
            sorted(df["ChargerCompany"].dropna().unique())
        )

        hour = st.slider("Charging Hour", 0, 23, 19)

    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday"
            ][x]
        )

        month = st.selectbox(
            "Month",
            list(range(1, 13))
        )

        is_weekend = 1 if day_of_week >= 5 else 0

        if 5 <= hour < 12:
            time_of_day = "Morning"
        elif 12 <= hour < 17:
            time_of_day = "Afternoon"
        elif 17 <= hour < 21:
            time_of_day = "Evening"
        else:
            time_of_day = "Night"

    if st.button("Predict Demand", type="primary"):

        input_data = pd.DataFrame([{
            "Location": location,
            "ChargerType": charger_type,
            "ChargerCompany": charger_company,
            "Hour": hour,
            "DayOfWeek": day_of_week,
            "Month": month,
            "IsWeekend": is_weekend,
            "TimeOfDay": time_of_day
        }])

        prediction = lightgbm_classifier.predict(input_data)[0]

        probabilities = lightgbm_classifier.predict_proba(input_data)[0]

        classes = lightgbm_classifier.named_steps["model"].classes_
        high_index = list(classes).index("High")
        high_probability = probabilities[high_index] * 100

  # Regression prediction
        demand_value = lightgbm_regressor.predict(input_data)[0]

        st.subheader("Prediction Result")

        st.metric("Predicted Demand Category", prediction)

        st.metric(
          "Predicted Numerical Demand",
           f"{demand_value:.2f}"
         )

        st.metric(
          "High Demand Probability",
          f"{high_probability:.2f}%"
        )
