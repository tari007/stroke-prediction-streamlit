# Stroke Prediction Streamlit App

This app deploys the trained stroke-classification pipelines from the Google Colab notebook.

## Files required

Place these files in this folder before deployment:

- logistic_regression_pipeline.joblib
- decision_tree_pipeline.joblib
- random_forest_pipeline.joblib
- xgboost_pipeline.joblib

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

1. Push this folder to a public or private GitHub repository.
2. Sign in to https://share.streamlit.io with GitHub.
3. Choose **Create app** > **Yup, I have an app**.
4. Select the repository, branch, and `app.py`.
5. Click **Deploy**.

## Model inputs

The app sends the following raw features, in this order:

`gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

The saved scikit-learn pipelines perform median imputation and scaling for numeric features, and most-frequent imputation and one-hot encoding for categorical features.
