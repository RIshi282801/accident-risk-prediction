import streamlit as st
import joblib
import pandas as pd
st.set_page_config(
    page_title="Accident Risk Prediction",
    page_icon="🚗",
    layout="centered"
)
# 🎨 Custom Dashboard Styling
st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0f766e 100%);
}

/* Main content text */
.stApp p,
.stApp label,
.stApp span,
.stApp div {
    color: #f8fafc;
}

/* Main title */
h1 {
    color: #ffffff !important;
    font-weight: 800;
}

/* Headings */
h2, h3 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.12);
    padding: 18px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1rem;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
    color: white !important;
}

/* Input boxes */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    border-radius: 10px;
    border: 1px solid #475569;
}

div[data-baseweb="select"] * {
    color: white !important;
}

/* Number inputs */
input {
    background-color: #1e293b !important;
    color: white !important;
}

/* Text area / inputs */
.stTextInput input,
.stNumberInput input {
    color: white !important;
}

/* Dataframe/table */
.stDataFrame {
    background-color: #ffffff;
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.25);
}

/* Info / success / warning boxes */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Accident Risk Prediction",
    page_icon="🚗",
    layout="centered"
)

# Load trained model
model = joblib.load("models/accident_risk_model.pkl")
# Load accident dataset for hotspot analysis
accident_data = pd.read_csv("data/cleaned_accident_data.csv")

# Title
st.title("🚗 AI-Based Road Accident Risk Prediction")

st.subheader("🤖 AI-Powered Road Safety System")

st.write(
    "This system uses Machine Learning to predict "
    "road accident risk based on road, traffic, "
    "weather and environmental conditions."
)

st.info(
    "💡 Enter the conditions below and click "
    "'Predict Accident Risk'."
)

st.divider()

# Input section
st.subheader("🛣️ Enter Road Conditions")
st.markdown("### 📍 Location & Time")

city = st.selectbox(
    "City",
    ["Chennai", "Mumbai", "Delhi", "Pune",
     "Bangalore", "Hyderabad", "Kolkata", "Chandigarh"]
)

state = st.selectbox(
    "State",
    ["Tamil Nadu", "Maharashtra", "Delhi",
     "Karnataka", "Telangana", "West Bengal",
     "Punjab", "Other"]
)
st.markdown("### 🌦️ Road & Environmental Conditions")

weather = st.selectbox(
    "Weather",
    ["clear", "rain", "fog"]
)

road_type = st.selectbox(
    "Road Type",
    ["urban", "highway", "rural"]
)

traffic_density = st.selectbox(
    "Traffic Density",
    ["low", "medium", "high"]
)

visibility = st.selectbox(
    "Visibility",
    ["low", "medium", "high"]
)

hour = st.slider(
    "Hour of Day",
    0,
    23,
    18
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday",
     "Thursday", "Friday", "Saturday", "Sunday"]
)

is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

lanes = st.number_input(
    "Number of Lanes",
    min_value=1,
    max_value=8,
    value=2
)

traffic_signal = st.selectbox(
    "Traffic Signal",
    ["Yes", "No"]
)

traffic_signal_value = 1 if traffic_signal == "Yes" else 0

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=28.0
)

is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
st.markdown("### 🚨 Accident Factors")



latitude = st.number_input(
    "Latitude",
    value=13.08
)

longitude = st.number_input(
    "Longitude",
    value=80.27
)

year = st.number_input(
    "Year",
    min_value=2022,
    max_value=2030,
    value=2025
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=8
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=15
)

st.divider()

# Prediction
st.markdown("### 🤖 AI Prediction")
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "🔮 Predict Accident Risk",
        use_container_width=True
    )

with col2:
    reset_button = st.button(
        "🔄 Reset",
        use_container_width=True
    )

if reset_button:
    st.rerun()

if predict_button:

    input_data = pd.DataFrame([{
        "city": city,
        "state": state,
        "latitude": latitude,
        "longitude": longitude,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "road_type": road_type,
        "lanes": lanes,
        "traffic_signal": traffic_signal_value,
        "weather": weather,
        "visibility": visibility,
        "temperature": temperature,
        "traffic_density": traffic_density,
        "is_peak_hour": is_peak_hour,
        "year": year,
        "month": month,
        "day": day
    }])

    # Predict risk
    prediction = model.predict(input_data)[0]

    # Get prediction probability
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    prediction_probability = (
        probabilities[list(classes).index(prediction)] * 100
    )
       # Prediction Result

    if prediction == "High":
        st.error(
            f"🔴 PREDICTED RISK LEVEL: HIGH\n\n"
            f"Risk Probability: {prediction_probability:.2f}%\n\n"
            f"⚠️ High-risk conditions detected. Drive carefully and reduce speed."
        )

    elif prediction == "Medium":
        st.warning(
            f"🟡 PREDICTED RISK LEVEL: MEDIUM\n\n"
            f"Risk Probability: {prediction_probability:.2f}%\n\n"
            f"⚠️ Moderate-risk conditions detected. Maintain safe speed and attention."
        )

    else:
        st.success(
            f"🟢 PREDICTED RISK LEVEL: LOW\n\n"
            f"Risk Probability: {prediction_probability:.2f}%\n\n"
            f"✅ Low-risk conditions detected. Continue following safe driving practices."
        )

    st.progress(
        int(prediction_probability),
        text=f"Confidence: {prediction_probability:.2f}%"
    )
        
# Accident statistics
st.subheader("📊 Accident Statistics")

selected_city = st.selectbox(
    "🏙️ Select City",
    ["All"] + sorted(accident_data["city"].unique().tolist())
)

if selected_city == "All":
    analysis_data = accident_data
else:
    analysis_data = accident_data[
        accident_data["city"] == selected_city
    ]

total_accidents = len(analysis_data)

high_risk = len(analysis_data[analysis_data["risk_score"] > 0.66])

medium_risk = len(
    analysis_data[
        (analysis_data["risk_score"] > 0.33) &
        (analysis_data["risk_score"] <= 0.66)
    ]
)

low_risk = len(
    analysis_data[
        analysis_data["risk_score"] <= 0.33
    ]
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accidents", total_accidents)
col2.metric("🔴 High Risk", high_risk)
col3.metric("🟠 Medium Risk", medium_risk)
col4.metric("🟢 Low Risk", low_risk)
st.subheader("🗺️ Accident Hotspot Analysis")

st.write(
    "Use the filters below to explore accident hotspots "
    "by city and risk level."
)

# Create risk level from risk score
def get_risk_level(score):
    if score <= 0.33:
        return "Low"
    elif score <= 0.66:
        return "Medium"
    else:
        return "High"
accident_data["risk_level"] = accident_data["risk_score"].apply(
    get_risk_level
)


accident_data["risk_level"] = accident_data["risk_score"].apply(
    get_risk_level
)

# Risk level filter
selected_risk = st.selectbox(
    "Select Risk Level",
    ["All", "Low", "Medium", "High"]
)

# Filter data
if selected_risk == "All":
    filtered_data = accident_data
else:
    filtered_data = accident_data[
        accident_data["risk_level"] == selected_risk
    ]

if selected_city != "All":
    filtered_data = filtered_data[
        filtered_data["city"] == selected_city
    ]

# Prepare map data
map_data = filtered_data[
    ["latitude", "longitude"]
].dropna()

# Display number of accidents
st.metric(
    "📍 Accidents Displayed on Map",
    len(map_data)
)

# Display map
st.map(
    map_data,
    latitude="latitude",
    longitude="longitude",
    size=20
)
st.divider()

st.subheader("📈 Accident Analysis")

# Accidents by city
city_counts = analysis_data["city"].value_counts().reset_index()
city_counts.columns = ["City", "Accidents"]

st.write("### 🚦 Accidents by City")
st.bar_chart(city_counts.set_index("City"))
# Accidents by weather
weather_counts = analysis_data["weather"].value_counts().reset_index()
weather_counts.columns = ["Weather", "Accidents"]

st.write("### 🌦️ Accidents by Weather Condition")
st.bar_chart(weather_counts.set_index("Weather"))
# Accidents by severity
severity_counts = analysis_data["accident_severity"].value_counts().reset_index()
severity_counts.columns = ["Severity", "Accidents"]

st.write("### ⚠️ Accidents by Severity")
st.bar_chart(severity_counts.set_index("Severity"))
st.write("### 🎯 Accident Risk Distribution")

risk_counts = (
    analysis_data["risk_level"]
    .value_counts()
    .reindex(["Low", "Medium", "High"])
    .dropna()
)

st.bar_chart(risk_counts)
st.write("### 🕐 Accidents by Hour")

hour_counts = analysis_data["hour"].value_counts().sort_index()

st.write("### 🕐 Accidents by Hour")
st.line_chart(hour_counts)

cause_counts = (
    analysis_data["cause"]
    .value_counts()
)

st.bar_chart(cause_counts)
st.write("### 📅 Accidents by Day of Week")

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_counts = (
    analysis_data["day_of_week"]
    .value_counts()
    .reindex(day_order)
)

st.bar_chart(day_counts)
st.divider()
st.subheader("🤖 Machine Learning Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("🏆 Best Model", "Random Forest")
col2.metric("🎯 Accuracy", "94.37%")
col3.metric("🧪 Test Samples", "4,000")

st.write("### 📊 Model Comparison")

comparison_data = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Decision Tree",
        "Logistic Regression"
    ],
    "Accuracy": [
        94.375,
        90.05,
        88.00
    ]
})

st.bar_chart(
    comparison_data.set_index("Model")
)

st.info(
    "Random Forest achieved the highest accuracy "
    "of 94.375% among the tested models."
)

st.info(
    "Random Forest was selected as the current best-performing "
    "model for accident risk classification."
)
st.divider()
st.subheader("📊 Confusion Matrix")
st.write("This shows how well the model predicts Low, Medium and High risk.")
st.write("**Rows = Actual Risk | Columns = Predicted Risk**")
confusion_data = pd.DataFrame(
    [
        [324, 0, 90],
        [0, 1387, 0],
        [13, 122, 2064]
    ],
    columns=["High", "Low", "Medium"],
    index=["High", "Low", "Medium"]
)

st.table(confusion_data)

st.divider()
st.subheader("🔍 Important Accident Risk Factors")

# Get feature names after preprocessing
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

# Get feature importance
importance = classifier.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

# Select top 10 features
top_features = (
    feature_importance
    .sort_values("Importance", ascending=False)
    .head(10)
)

st.bar_chart(
    top_features.set_index("Feature")
)
st.divider()

st.subheader("📥 Download Dataset")

csv_data = accident_data.to_csv(index=False)

st.download_button(
    label="⬇️ Download Accident Dataset",
    data=csv_data,
    file_name="accident_data.csv",
    mime="text/csv"
)