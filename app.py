import streamlit as st
import joblib
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM DASHBOARD STYLING
# =========================================================

st.markdown(
    """
    <style>

    /* -----------------------------------------------------
       GLOBAL
    ----------------------------------------------------- */

    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap'
    );

    html,
    body,
    [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 10% 0%,
                rgba(59, 130, 246, 0.10),
                transparent 28%
            ),
            #07111f;
        color: #e8eef7;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.85);
    }

    [data-testid="stSidebar"] {
        background: #091525;
        border-right: 1px solid rgba(148, 163, 184, 0.14);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }


    /* -----------------------------------------------------
       SECTION HEADINGS
    ----------------------------------------------------- */

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 6px;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 20px;
        line-height: 1.6;
    }


    /* -----------------------------------------------------
       DASHBOARD METRIC CARDS
    ----------------------------------------------------- */

    .metric-card {
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(15, 31, 53, 0.78);
        border-radius: 16px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.07em;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 8px;
    }

    .metric-note {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 4px;
    }


    /* -----------------------------------------------------
       PREDICTION RESULT CARDS
    ----------------------------------------------------- */

    .result-card {
        border: 1px solid rgba(148, 163, 184, 0.16);
        background: rgba(15, 31, 53, 0.82);
        border-radius: 18px;
        padding: 22px;
        min-height: 140px;
    }

    .result-label {
        color: #94a3b8;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .result-value {
        font-size: 1.65rem;
        font-weight: 700;
        margin-top: 10px;
    }


    /* -----------------------------------------------------
       DISCLAIMER
    ----------------------------------------------------- */

    .disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.22);
        border-radius: 12px;
        padding: 14px 16px;
        color: #f5d28b;
        font-size: 0.82rem;
        line-height: 1.65;
        margin: 10px 0 20px;
    }


    /* -----------------------------------------------------
       INFORMATION CARD
    ----------------------------------------------------- */

    .info-card {
        border: 1px solid rgba(148, 163, 184, 0.15);
        background: rgba(15, 31, 53, 0.70);
        border-radius: 16px;
        padding: 20px;
        line-height: 1.7;
        min-height: 150px;
    }


    /* -----------------------------------------------------
       STREAMLIT METRICS
    ----------------------------------------------------- */

    div[data-testid="stMetric"] {
        background: rgba(15, 31, 53, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 15px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }


    /* -----------------------------------------------------
       BUTTON
    ----------------------------------------------------- */

    .stButton > button {
        border-radius: 10px;
        min-height: 46px;
        font-weight: 600;
    }


    /* -----------------------------------------------------
       INPUT CONTAINERS
    ----------------------------------------------------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 15px;
    }


    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "Decision Tree": "decision_tree_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
}


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_FILES[model_name])


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🩺 Stroke Prediction")

    st.caption(
        "Machine-learning decision-support demonstration."
    )

    st.divider()

    st.markdown("### Prediction model")

    selected_model = st.selectbox(
        "Choose trained model",
        list(MODEL_FILES.keys()),
    )

    st.caption(
        "All models use the same preprocessing pipeline "
        "fitted on the training data."
    )

    st.divider()

    st.markdown("### Available models")

    st.markdown(
        """
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - XGBoost
        """
    )

    st.divider()

    st.caption(
        "Academic machine-learning project. "
        "This system does not provide a medical diagnosis."
    )


# =========================================================
# LOAD SELECTED MODEL
# =========================================================

try:

    model = load_model(selected_model)

except Exception as e:

    st.error(
        "The selected model could not be loaded."
    )

    st.exception(e)

    st.stop()


# =========================================================
# MEDICAL DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important medical notice:</strong>
        This application is an academic machine-learning
        decision-support demonstration. It does not provide
        a medical diagnosis and should not be used to make
        clinical decisions or replace assessment by a qualified
        healthcare professional.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DASHBOARD OVERVIEW
# =========================================================

st.markdown("### Dashboard overview")

overview_1, overview_2, overview_3, overview_4 = st.columns(4)


with overview_1:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Application
            </div>
            <div class="metric-value">
                Stroke Prediction
            </div>
            <div class="metric-note">
                Machine learning system
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with overview_2:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Models
            </div>
            <div class="metric-value">
                4
            </div>
            <div class="metric-note">
                Classification algorithms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with overview_3:

    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">
                Input variables
            </div>
            <div class="metric-value">
                10
            </div>
            <div class="metric-note">
                Patient-level features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with overview_4:

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Active model
            </div>
            <div class="metric-value">
                {selected_model}
            </div>
            <div class="metric-note">
                Currently selected
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    "<div class='section-title'>Patient stroke-risk prediction</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-subtitle">
        Enter the patient's demographic, medical and lifestyle
        information below. The selected machine-learning model
        will use these inputs to generate a predicted class and
        estimated probability.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PATIENT INPUT SECTION
# =========================================================

st.markdown("### 👤 Patient details")

input_col1, input_col2 = st.columns(2)


# ---------------------------------------------------------
# LEFT COLUMN
# ---------------------------------------------------------

with input_col1:

    with st.container(border=True):

        st.markdown("#### 👤 Demographic information")

        gender = st.selectbox(
            "👤 Gender",
            ["Female", "Male", "Other"],
        )

        age = st.number_input(
            ":material/calendar_today: Age (years)",
            min_value=0.0,
            max_value=120.0,
            value=45.0,
            step=1.0,
        )

        ever_married = st.selectbox(
            ":material/favorite: Ever married",
            ["No", "Yes"],
        )

        residence_type = st.selectbox(
            ":material/home: Residence type",
            ["Urban", "Rural"],
        )


# ---------------------------------------------------------
# RIGHT COLUMN
# ---------------------------------------------------------

with input_col2:

    with st.container(border=True):

        st.markdown("#### 🏥 Medical and lifestyle information")

        hypertension = st.selectbox(
            ":material/monitor_heart: Hypertension",
            ["No", "Yes"],
        )

        heart_disease = st.selectbox(
            ":material/cardiology: Heart disease",
            ["No", "Yes"],
        )

        smoking_status = st.selectbox(
            ":material/smoking_rooms: Smoking status",
            [
                "formerly smoked",
                "never smoked",
                "smokes",
                "Unknown",
            ],
        )


# =========================================================
# ADDITIONAL PATIENT INFORMATION
# =========================================================

st.markdown("### 📋 Clinical and social information")

clinical_col1, clinical_col2, clinical_col3 = st.columns(3)


with clinical_col1:

    work_type = st.selectbox(
        ":material/work: Work type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked",
        ],
    )


with clinical_col2:

    avg_glucose_level = st.number_input(
        ":material/bloodtype: Average glucose level",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=0.01,
    )


with clinical_col3:

    bmi = st.number_input(
        ":material/monitor_weight: BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
    )


# =========================================================
# PREPARE INPUT DATA
# =========================================================

input_data = pd.DataFrame(
    [
        {
            "gender": gender,
            "age": age,
            "hypertension": (
                1 if hypertension == "Yes" else 0
            ),
            "heart_disease": (
                1 if heart_disease == "Yes" else 0
            ),
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status,
        }
    ]
)


# =========================================================
# INPUT SUMMARY
# =========================================================

with st.expander(
    "🔍 View input sent to the model",
    expanded=False,
):

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("### 🧠 Run prediction")

predict_clicked = st.button(
    "🩺 Predict Stroke Risk",
    type="primary",
    use_container_width=True,
)


# =========================================================
# PREDICTION
# =========================================================

if predict_clicked:

    try:

        prediction = int(
            model.predict(input_data)[0]
        )

        probability = float(
            model.predict_proba(input_data)[0, 1]
        )


        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability
        st.session_state["predicted_model"] = selected_model


        # -------------------------------------------------
        # RESULT HEADER
        # -------------------------------------------------

        st.divider()

        st.markdown("### 📊 Prediction result")

        result_col1, result_col2, result_col3 = st.columns(3)


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Stroke probability
                    </div>

                    <div class="result-value">
                        {probability:.1%}
                    </div>

                    <div class="metric-note">
                        Estimated probability of class 1
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # -------------------------------------------------
        # PREDICTED CLASS
        # -------------------------------------------------

        with result_col2:

            predicted_class = (
                "Stroke (1)"
                if prediction == 1
                else "No stroke (0)"
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted class
                    </div>

                    <div class="result-value">
                        {predicted_class}
                    </div>

                    <div class="metric-note">
                        Default classification threshold: 50%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # -------------------------------------------------
        # MODEL
        # -------------------------------------------------

        with result_col3:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Selected model
                    </div>

                    <div class="result-value"
                         style="font-size:1.25rem;">
                        {selected_model}
                    </div>

                    <div class="metric-note">
                        Trained classification pipeline
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        st.markdown("### Probability estimate")

        st.progress(
            min(
                max(probability, 0.0),
                1.0,
            ),
            text=(
                f"Estimated stroke probability: "
                f"{probability:.1%}"
            ),
        )


        # -------------------------------------------------
        # RESULT MESSAGE
        # -------------------------------------------------

        if prediction == 1:

            st.error(
                "The selected model predicts a stroke case "
                "at the default 0.5 classification threshold."
            )

        else:

            st.success(
                "The selected model does not predict a stroke "
                "case at the default 0.5 classification threshold."
            )


        # -------------------------------------------------
        # CLINICAL CONTEXT
        # -------------------------------------------------

        st.info(
            "A probability below 50% does not mean that a "
            "person has no clinical risk. This output is a "
            "machine-learning prediction and should be "
            "interpreted only alongside appropriate professional "
            "medical assessment."
        )


        # -------------------------------------------------
        # INPUT SUMMARY
        # -------------------------------------------------

        st.markdown("### Patient input summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)


        with summary_col1:

            st.metric(
                "Age",
                f"{age:.0f} years",
            )

            st.metric(
                "Hypertension",
                hypertension,
            )

            st.metric(
                "Heart disease",
                heart_disease,
            )


        with summary_col2:

            st.metric(
                "Average glucose",
                f"{avg_glucose_level:.2f}",
            )

            st.metric(
                "BMI",
                f"{bmi:.1f}",
            )

            st.metric(
                "Smoking status",
                smoking_status,
            )


        with summary_col3:

            st.metric(
                "Gender",
                gender,
            )

            st.metric(
                "Work type",
                work_type,
            )

            st.metric(
                "Residence",
                residence_type,
            )


    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# =========================================================
# METHODOLOGY SECTION
# =========================================================

st.divider()

st.markdown("### 📘 About the system")

about_col1, about_col2, about_col3 = st.columns(3)


with about_col1:

    st.markdown(
        """
        <div class="info-card">

        <strong>Machine learning</strong>

        <br><br>

        The system supports four trained classification
        algorithms:

        <br><br>

        • Logistic Regression<br>
        • Decision Tree<br>
        • Random Forest<br>
        • XGBoost

        </div>
        """,
        unsafe_allow_html=True,
    )


with about_col2:

    st.markdown(
        """
        <div class="info-card">

        <strong>Patient variables</strong>

        <br><br>

        The prediction pipeline uses demographic,
        medical, lifestyle and clinical variables.

        <br><br>

        These include age, hypertension, heart disease,
        glucose level, BMI, smoking status, work type
        and residence type.

        </div>
        """,
        unsafe_allow_html=True,
    )


with about_col3:

    st.markdown(
        """
        <div class="info-card">

        <strong>Prediction output</strong>

        <br><br>

        The system produces:

        <br><br>

        • Stroke probability<br>
        • Predicted class<br>
        • Selected model<br>
        • Input summary

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("### 🔬 Model information")

model_information = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
        ],
        "Type": [
            "Linear classification",
            "Tree-based classification",
            "Ensemble tree classification",
            "Gradient boosting classification",
        ],
        "Available": [
            "✓",
            "✓",
            "✓",
            "✓",
        ],
    }
)

st.dataframe(
    model_information,
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# FINAL DISCLAIMER
# =========================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Academic project disclaimer:</strong>
        This system demonstrates the application of machine
        learning to stroke prediction. The predictions are
        generated from trained machine-learning models and
        should not be interpreted as a medical diagnosis,
        clinical assessment, or substitute for professional
        healthcare advice.
    </div>
    """,
    unsafe_allow_html=True,
)
