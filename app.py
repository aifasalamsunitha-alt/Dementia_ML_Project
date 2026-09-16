from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dementia_patients_health_data.csv"
MODEL_PATH = BASE_DIR / "models" / "dementia_model.pkl"

st.set_page_config(page_title="Dementia Classification", page_icon="🧠")
st.title("🧠 Dementia Classification")
st.caption("Educational machine-learning demonstration — not a medical diagnosis")

# Use caching so resources load only once per session
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# 1. Check for missing files
if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error("Dataset or trained model is missing. Run your training script to generate `dementia_model.pkl`.")
    st.stop()

# 2. Attempt model & dataset loading with error handling
try:
    model = load_model(MODEL_PATH)
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(
        "Failed to load the model file. This usually happens when the Python or `scikit-learn` "
        "version used to save `dementia_model.pkl` differs from the deployment environment."
    )
    st.info("💡 **Fix:** Re-run `python dementia_model.py` in your current environment to re-generate `dementia_model.pkl`.")
    st.exception(e)
    st.stop()

features = [c for c in df.columns if c.strip().lower() != "dementia"]

st.write("Enter patient information and click **Predict**.")

with st.form("prediction_form"):
    values = {}
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            default = float(s.median()) if s.notna().any() else 0.0
            if pd.api.types.is_integer_dtype(s.dropna()):
                values[col] = st.number_input(col, value=int(default))
            else:
                values[col] = st.number_input(col, value=default, format="%.2f")
        else:
            options = sorted(s.dropna().astype(str).unique().tolist()) or [""]
            values[col] = st.selectbox(col, options)
            
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame([values], columns=features)
        prediction = int(model.predict(input_df)[0])

        if prediction == 1:
            st.error("Prediction: Dementia")
        else:
            st.success("Prediction: No Dementia")

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_df)[0][1])
            st.write(f"Model probability for Dementia: **{probability:.2%}**")
            
    except Exception as e:
        st.error("An error occurred during inference. Ensure feature formatting matches model expectations.")
        st.exception(e)
            
