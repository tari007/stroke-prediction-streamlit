import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Stroke Prediction", page_icon="🩺", layout="centered")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "Decision Tree": "decision_tree_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
}

@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_FILES[model_name])

st.title(":material/stethoscope: Stroke Prediction Using Machine Learning")
st.write("Welcome to Group 6 Capstone Project.")
st.caption("🩺This is a decision-support demo developed as an academic project to demonstrate the application of machine learning in stroke risk prediction. It does not provide a medical diagnosis.")

with st.sidebar:
    st.header("Model")
    selected_model = st.selectbox("Choose trained model", list(MODEL_FILES.keys()))
    st.caption("All models use the same preprocessing pipeline fitted on the training data.")

model = load_model(selected_model)

st.subheader("Patient details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤Gender", ["Female", "Male", "Other"])
    age = st.number_input(":material/calendar_today: Age (years)", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    hypertension = st.selectbox(":material/monitor_heart: Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox(":material/cardiology: Heart disease", ["No", "Yes"])
    ever_married = st.selectbox(":material/favorite: Ever married", ["No", "Yes"])

with col2:
    work_type = st.selectbox(
        ":material/work: Work type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )
    residence_type = st.selectbox(":material/home: Residence type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input(
        ":material/bloodtype: Average glucose level", min_value=0.0, max_value=500.0, value=100.0, step=0.01
    )
    bmi = st.number_input(":material/monitor_weight: BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    smoking_status = st.selectbox(
        ":material/smoking_rooms: Smoking status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"],
    )

input_data = pd.DataFrame([{
    "gender": gender,
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}])

st.subheader("Input sent to the model")
st.dataframe(input_data, use_container_width=True)

if st.button("Predict stroke risk", type="primary"):
    try:
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0, 1])

        st.subheader("Prediction")
        col_a, col_b = st.columns(2)
        col_a.metric("Stroke probability", f"{probability:.1%}")
        col_b.metric("Predicted class", "Stroke (1)" if prediction == 1 else "No stroke (0)")

        if prediction == 1:
            st.error("The model predicts a stroke case at the default 0.5 threshold.")
        else:
            st.success("The model does not predict a stroke case at the default 0.5 threshold.")

        st.info(
            "A probability below 50% can still represent elevated clinical risk. "
            "Use this output only as decision support alongside professional assessment."
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
